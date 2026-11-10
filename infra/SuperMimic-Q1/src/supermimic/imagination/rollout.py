"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.policy.nets import Policy
from supermimic.robot import NJ, XI_BASE_DIM, XI_EXTRA_NAMES
from supermimic.tele import slots as S
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_relative_obs, delay_policy_obs, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * NJ + [0.5] * NJ + [0.2] * 3 + [0.05] * 3 + [0.0] * NJ)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, ·) critic goal input
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)
    align_loss: torch.Tensor | None = None  # latent alignment of E_tele to E_ref


def _yaw_aligned(q_ref: torch.Tensor, q_robot: torch.Tensor) -> torch.Tensor:
    """q_ref rotated about z so that its heading equals the robot's."""
    from supermimic.kinematics.rotations import heading_quat, quat_inv, quat_mul

    return quat_mul(quat_mul(heading_quat(q_robot), quat_inv(heading_quat(q_ref))), q_ref)


class _ObsHistory:
    """Noisy, delayed term-major observation history of a rollout (shared by both rollout kinds)."""

    def __init__(self, start: StartBatch, q0: torch.Tensor, obs_noise: bool):
        self.noise_half = OBS_NOISE_HALF.to(start.state.device)
        self.obs_noise = obs_noise
        self.lag = start.xi[:, XI_BASE_DIM + XI_EXTRA_NAMES.index("obs_delay")]
        self.hist = [self._noisy(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])]

    def _noisy(self, o):
        return o + (torch.rand_like(o) * 2 - 1) * self.noise_half if self.obs_noise else o

    def flat(self) -> torch.Tensor:
        return delay_policy_obs(flatten_history_term_major(torch.stack(self.hist, 0)), self.lag)

    def push(self, obs_clean: torch.Tensor) -> None:
        self.hist = self.hist[1:] + [self._noisy(obs_clean)]


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
    anchor_ori_only: bool = False,
    tele: dict | None = None,
    anchor_pos_noise: float = 0.25,
) -> ImagRollout:
    """Motion-tracking rollout through E_ref. With ``tele`` = dict(align=bool) the policy acts through
    E_tele on the reference-derived slots (``lib.tele_slots``) with the start row's slot mask
    (distillation, same reward and termination); ``goals`` then holds the normalised tele goal for
    that batch's critic and ``align_loss`` the MSE to the (stop-gradient) E_ref latent."""
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagStep(wm, q0)
    anchor_noise_half = ANCHOR_NOISE_HALF.to(device).clone()
    anchor_noise_half[:3] = anchor_pos_noise
    ai = wm.heads.anchor_index

    def make_goal(frame_t, ref_t, anchor_pose):
        g = lib.goal_window(frame_t, future_frames)
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :])
        half = anchor_noise_half
        if anchor_ori_only:
            rel, half = rel[..., 3:], half[3:]
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * half
        return torch.cat([g, rel], dim=-1)

    hist = _ObsHistory(start, q0, obs_noise)
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references for every step in one batched call.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    ref = ref_at(0)
    if anchor_obs:
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    if tele is not None:
        mask = start.cmd[:, S.SLOT_DIM :]
        tele_goal = lambda f: torch.cat([lib.tele_slots[f], mask], dim=-1)  # noqa: E731
    goal_m = make_goal(frame, ref, anchor_pose)
    goals = [goal_m if tele is None else tele_goal(frame)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)
    align_loss = torch.zeros((), device=device)

    for t in range(horizon):
        obs_flat = hist.flat()
        if tele is None:
            a = policy(obs_flat, goals[-1])
        else:
            z = policy.latent(obs_flat, goals[-1], "tele")
            if tele.get("align"):
                with torch.no_grad():
                    z_m = policy.latent(obs_flat, goal_m, "ref")
                align_loss = align_loss + ((z - z_m) ** 2).mean() / horizon
            a = policy.decoder(z, obs_flat)
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, ref)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist.push(obs_clean)
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goal_m = make_goal(frame, ref, anchor_pose)
        goals.append(goal_m if tele is None else tele_goal(frame))
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    goals_t = torch.stack(goals)
    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=goals_t if tele is None else S.normalized(goals_t),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
        align_loss=align_loss if tele is not None else None,
    )


def imagine_free(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    step_fn,
    stand_height: float,
    future_frames: int = 10,
    obs_noise: bool = True,
    anchor_obs: bool = False,
    anchor_ori_only: bool = False,
    align: bool = True,
    anchor_pos_noise: float = 0.25,
) -> ImagRollout:
    """Free-command rollout (``step_fn`` = ``FreeImagStep``). Goals: the start row's held slots, or for
    reference-command rows the clip's slots at each step, with the start row's mask. ``align_loss``:
    MSE between the E_tele latent and the (stop-gradient) E_ref latent of the same step on
    reference-command rows, with the reference heading aligned to the robot's (slots carry no heading)."""
    from supermimic.tele.interface import FREE_REWARD_TERM_NAMES

    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    anchor_noise_half = ANCHOR_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index

    hist = _ObsHistory(start, q0, obs_noise)
    s = start.state
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    last = lib.last_frame_of(start.frame)
    frames_all = torch.minimum(start.frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])
    from_ref = start.cmd_ref
    mask = start.cmd[:, S.SLOT_DIM :]
    slots_all = torch.where(from_ref[None, :, None], lib.tele_slots[frames_all], start.cmd[None, :, : S.SLOT_DIM])  # (H+1, n, SLOT_DIM)
    goals_all = torch.cat([slots_all, mask[None].expand(horizon + 1, -1, -1)], dim=-1)
    hs = S.SLICE["height"]
    virt_pos = torch.zeros(horizon + 1, n, 3, device=device)
    virt_pos[..., 2] = stand_height + slots_all[..., hs.start] * mask[None, :, S.SLOT_NAMES.index("height")]
    virt_quat = torch.zeros(n, 4, device=device)
    virt_quat[:, 0] = 1.0
    n_ref = int(from_ref.sum()) if align else 0
    if n_ref > 0:
        ref_all = reference_at(lib, frames_all.reshape(-1))
        rq = ref_all.body_quat_w[:, ai, :].reshape(horizon + 1, n, 4)
    anchor_quat = wm.heads.robot_bodies(s)["quat"][..., ai, :]

    states, actions, rewards = [s], [], []
    dhats, survivals = [torch.zeros(n, device=device)], [c]
    term_sum = torch.zeros(len(FREE_REWARD_TERM_NAMES), device=device)
    align_loss = torch.zeros((), device=device)
    for t in range(horizon):
        obs_flat = hist.flat()
        z = policy.latent(obs_flat, goals_all[t], "tele")
        if n_ref > 0:
            with torch.no_grad():
                g = lib.goal_window(frames_all[t], future_frames)
                if anchor_obs:
                    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

                    rel = matrix_from_quat(quat_mul(quat_inv(anchor_quat), _yaw_aligned(rq[t], anchor_quat)))[..., :2].reshape(n, 6)
                    if not anchor_ori_only:
                        rel = torch.cat([torch.zeros(n, 3, device=device), rel], dim=-1)
                    half = anchor_noise_half[3:] if anchor_ori_only else anchor_noise_half
                    if obs_noise:
                        rel = rel + (torch.rand_like(rel) * 2 - 1) * half
                    g = torch.cat([g, rel], dim=-1)
                z_m = policy.latent(obs_flat, g, "ref")
            err = ((z - z_m) ** 2).mean(-1)
            align_loss = align_loss + (err * from_ref).sum() / n_ref / horizon
        a = policy.decoder(z, obs_flat)
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, goals_all[t], virt_pos[t + 1], virt_quat)
        anchor_quat = anchor_pose[..., 3:7]
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        hist.push(obs_clean)
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=S.normalized(goals_all),
        e_root=torch.zeros(horizon + 1, n, 4, device=device),
        frames=frames_all,
        reward_terms={k: float(v / horizon) for k, v in zip(FREE_REWARD_TERM_NAMES, term_sum.tolist())},
        align_loss=align_loss,
    )
