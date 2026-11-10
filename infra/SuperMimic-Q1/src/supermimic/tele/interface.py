"""Teleoperation encoder training (docs/training.md): two kinds of envs / imagined batches share E_tele.

distill  envs replay ``tele_ok`` clips (starts balanced over command categories); E_tele reads the
         slots derived from the reference (tele/labels.py) and the rollout is scored with the
         motion-tracking reward and termination; latent alignment to E_ref on the same reference.
free     envs hold commands, either read from the clip they replay (``ref_prob``, with latent
         alignment) or sampled; the rollout is scored with the slot-tracking reward below and a
         fall-only termination against a virtual reference whose pelvis height follows the
         commanded height (squatting is not a fall).

Both kinds draw a slot mask per command (slot dropout inside the teleoperation slots); a masked slot
is rewarded towards its default with ``masked_weight`` (lost tracking degrades to standing / natural
hand positions). The decoder is never updated by these batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.kinematics.rotations import heading_quat, quat_apply, quat_apply_inverse
from supermimic.robot import SPEC
from supermimic.tele import slots as S

HIP_YAW = [SPEC.joint_names.index(f"{s}_hip_yaw_joint") for s in ("left", "right")]


@dataclass
class TeleCfg:
    enabled: bool = False
    distill_env_fraction: float = 0.25
    free_env_fraction: float = 0.25
    distill_batch_fraction: float = 0.25
    free_batch_fraction: float = 0.25
    encoder_hidden: tuple[int, ...] = (512, 512)
    embed: int = 64
    align_weight: float = 0.1
    critic_warmup_updates: int = 300  # tele critics fit alone first (fresh critics on a resumed policy)
    # Reserved slots switched on in the distill batches (new-slot procedure: their projections already exist;
    # labels come from the reference). Free-command batches keep them masked.
    enable_slots: tuple[str, ...] = ()
    resample_steps: int = 250  # commands and masks are re-drawn every 5 s and at every reset
    # Slot dropout (both kinds): every non-reserved slot dropped independently, or base only.
    drop_prob: float = 0.15
    base_only_prob: float = 0.1
    # Distill starts: share of resets balanced over command categories (rest: return-based on tele_ok bins).
    balance_ratio: float = 0.5
    # Free commands.
    ref_prob: float = 0.5
    stand_prob: float = 0.1
    vx_range: tuple[float, float] = (-0.6, 1.0)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    wz_range: tuple[float, float] = (-1.5, 1.5)
    height_range: tuple[float, float] = (-0.15, 0.02)
    height_zero_prob: float = 0.4
    height_rate: float = 0.25  # m/s, the commanded height moves to its target at this rate (as the PICO adapter)
    pitch_range: tuple[float, float] = (-0.15, 0.6)
    roll_range: tuple[float, float] = (-0.25, 0.25)
    torso_zero_prob: float = 0.4
    hand_default_prob: float = 0.3
    hand_same_frame_prob: float = 0.5  # both hands from one library frame, else from two
    stand_height: float = S.H_STAND
    # Free-command reward (every term x dt); effort terms reuse the tracking weights.
    w_lin: float = 2.0
    var_lin: float = 0.1
    w_yaw: float = 2.0
    var_yaw: float = 0.5
    w_height: float = 1.0
    var_height: float = 0.05**2
    w_torso: float = 1.0
    var_torso: float = 0.2**2
    w_hand: float = 0.75
    var_hand: float = 0.1**2
    masked_weight: float = 0.3
    w_action_rate: float = -0.3
    w_joint_limit: float = -2.0
    w_rp_rate: float = -0.05
    # Leg posture: free commands have no reference to shape the gait (without these the walk drifts to
    # toe-out with the feet on one line). Hip yaw beyond a deadband around the default; lateral foot
    # distance in the pelvis heading frame below a minimum (standing 0.175 m, library walks 0.15-0.21 m).
    w_hip_yaw: float = -4.0  # -1.0 stalled at ~14 deg toe-out (the squared excess fades)
    hip_yaw_deadband: float = 0.1
    w_feet_width: float = -5.0
    feet_width_min: float = 0.14

    @property
    def max_step_reward(self) -> float:
        return 0.02 * (self.w_lin + self.w_yaw + self.w_height + self.w_torso + 2 * self.w_hand)


def sample_masks(cfg: TeleCfg, n: int, device, distill: torch.Tensor | None = None) -> torch.Tensor:
    """(n, NUM_SLOTS) float mask bits; ``distill`` (n,) bool rows may also use ``cfg.enable_slots``."""
    reserved = S.reserved_mask(device).expand(n, -1)
    if cfg.enable_slots and distill is not None:
        on = torch.tensor([s in cfg.enable_slots for s in S.SLOT_NAMES], device=device)
        reserved = reserved & ~(on[None, :] & distill[:, None])
    m = (torch.rand(n, S.NUM_SLOTS, device=device) >= cfg.drop_prob).float()
    base_only = torch.zeros(S.NUM_SLOTS, device=device)
    base_only[S.SLOT_NAMES.index("base")] = 1.0
    m = torch.where((torch.rand(n, 1, device=device) < cfg.base_only_prob), base_only.expand(n, -1), m)
    return m * (~reserved).float()


def sample_free_slots(cfg: TeleCfg, n: int, device, lib) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled free commands (n, SLOT_DIM) and the height target (n,) (the slot's height starts at 0
    and ramps to it). Hands come from tele_ok library frames (reachable, natural arm poses)."""
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=device)  # noqa: E731
    out = S.default_vector(device).expand(n, -1).clone()
    base = torch.stack([r(*cfg.vx_range), r(*cfg.vy_range), r(*cfg.wz_range)], dim=-1)
    base = torch.where((torch.rand(n, 1, device=device) < cfg.stand_prob), torch.zeros_like(base), base)
    out[:, S.SLICE["base"]] = base
    h = torch.where(torch.rand(n, device=device) < cfg.height_zero_prob, torch.zeros(n, device=device), r(*cfg.height_range))
    lean = torch.rand(n, device=device) >= cfg.torso_zero_prob
    out[:, S.SLICE["torso"]] = S.torso_gravity(r(*cfg.pitch_range) * lean, r(*cfg.roll_range) * lean)
    frames = lib.tele_ok_frames
    fl = frames[torch.randint(0, frames.numel(), (n,), device=device)]
    fr = torch.where(torch.rand(n, device=device) < cfg.hand_same_frame_prob, fl, frames[torch.randint(0, frames.numel(), (n,), device=device)])
    keep = (torch.rand(n, 1, device=device) >= cfg.hand_default_prob).float()
    for name, f in (("hand_L", fl), ("hand_R", fr)):
        sl = S.SLICE[name]
        out[:, sl] = keep * lib.tele_slots[f, sl] + (1 - keep) * out[:, sl]
    return out, h


def robot_slots(root_pos, root_quat, root_lin_vel_w, root_ang_vel_b, torso_pos, torso_quat, hands, feet) -> torch.Tensor:
    """(…, SLOT_DIM) instantaneous slots of the robot (base = current base velocity)."""
    p = S.pose_slots(root_pos, torso_pos, torso_quat, hands, feet)
    p["base"] = base_velocity(root_quat, root_lin_vel_w, root_ang_vel_b)
    return S.assemble(p)


def base_velocity(root_quat: torch.Tensor, root_lin_vel_w: torch.Tensor, root_ang_vel_b: torch.Tensor) -> torch.Tensor:
    """(…, 3): planar root velocity in the heading frame and world yaw rate (the base slot)."""
    v_h = quat_apply_inverse(heading_quat(root_quat), root_lin_vel_w)
    wz = quat_apply(root_quat, root_ang_vel_b)[..., 2:3]
    return torch.cat([v_h[..., :2], wz], dim=-1)


def free_reward_terms(
    cfg: TeleCfg,
    *,
    robot: torch.Tensor,  # (N, SLOT_DIM) robot_slots
    goal: torch.Tensor,  # (N, GOAL_DIM) commanded slots + mask
    torso_quat: torch.Tensor,
    torso_ang_vel_w: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,
    root_quat: torch.Tensor,
    feet_pos: torch.Tensor,  # (N, 2, 3) left, right (world)
    default_joint_pos: torch.Tensor,
    effort: dict[str, torch.Tensor] | None = None,
    effort_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dt: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Per-term dt-scaled free-command rewards (N,) plus 'total'."""
    slots, mask = S.split_goal(goal)
    dflt = S.default_vector(goal.device)
    tgt = torch.where(S.expand_mask(mask) > 0.5, slots, dflt)
    wmask = lambda name: cfg.masked_weight + (1.0 - cfg.masked_weight) * mask[..., S.SLOT_NAMES.index(name)]  # noqa: E731
    d = robot - tgt
    sq = lambda name: (d[..., S.SLICE[name]] ** 2).sum(-1)  # noqa: E731
    e_lin = (d[..., S.SLICE["base"]][..., :2] ** 2).sum(-1)
    e_yaw = d[..., S.SLICE["base"]][..., 2] ** 2
    hq = heading_quat(torso_quat)
    w_h = quat_apply_inverse(hq, torso_ang_vel_w)
    rp_rate = w_h[..., 0] ** 2 + w_h[..., 1] ** 2
    lo, hi = soft_limits[..., 0], soft_limits[..., 1]
    joint_limit = ((lo - joint_pos).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)).sum(-1)
    hy = (joint_pos[..., HIP_YAW] - default_joint_pos[HIP_YAW]).abs()
    hip_yaw = ((hy - cfg.hip_yaw_deadband).clamp(min=0.0) ** 2).sum(-1)
    width = quat_apply_inverse(heading_quat(root_quat), feet_pos[..., 0, :] - feet_pos[..., 1, :])[..., 1]
    wb = wmask("base")
    terms = {
        "vel_lin": dt * cfg.w_lin * wb * torch.exp(-e_lin / cfg.var_lin),
        "vel_yaw": dt * cfg.w_yaw * wb * torch.exp(-e_yaw / cfg.var_yaw),
        "height": dt * cfg.w_height * wmask("height") * torch.exp(-sq("height") / cfg.var_height),
        "torso": dt * cfg.w_torso * wmask("torso") * torch.exp(-sq("torso") / cfg.var_torso),
        "hand_L": dt * cfg.w_hand * wmask("hand_L") * torch.exp(-sq("hand_L") / cfg.var_hand),
        "hand_R": dt * cfg.w_hand * wmask("hand_R") * torch.exp(-sq("hand_R") / cfg.var_hand),
        "action_rate_l2": dt * cfg.w_action_rate * ((action - prev_action) ** 2).sum(-1),
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
        "torso_rp_rate": dt * cfg.w_rp_rate * rp_rate,
        "hip_yaw": dt * cfg.w_hip_yaw * hip_yaw,
        "feet_width": dt * cfg.w_feet_width * (cfg.feet_width_min - width).clamp(min=0.0),
    }
    zero = torch.zeros_like(rp_rate)
    for k, w in zip(("torque", "power", "over_rated"), effort_weights):
        terms[k] = dt * w * effort[k] if effort is not None else zero
    total = zero
    for v in terms.values():
        total = total + v
    terms["total"] = total
    return terms


FREE_REWARD_TERM_NAMES = (
    "vel_lin", "vel_yaw", "height", "torso", "hand_L", "hand_R", "action_rate_l2", "joint_limit", "torso_rp_rate", "hip_yaw", "feet_width",
    "torque", "power", "over_rated",
)


def virtual_height(cfg: TeleCfg, goal: torch.Tensor) -> torch.Tensor:
    """(…,) pelvis height of the free envs' fall-test reference: standing height + commanded Δh."""
    slots, mask = S.split_goal(goal)
    return cfg.stand_height + slots[..., S.SLICE["height"]][..., 0] * mask[..., S.SLOT_NAMES.index("height")]


def tracked_indices(body_names: tuple[str, ...]) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """(torso, hands, feet) indices into the tracked-body list."""
    return (
        body_names.index("torso_link"),
        tuple(body_names.index(b) for b in S.HAND_BODIES),
        tuple(body_names.index(b) for b in S.FOOT_BODIES),
    )


def slots_from_bodies(root_pos, root_quat, root_lin_vel_w, root_ang_vel_b, body_pos, body_quat, idx) -> torch.Tensor:
    """robot_slots from tracked-body world poses (…, B, ·) and ``tracked_indices``."""
    ti, hi, fi = idx
    return robot_slots(root_pos, root_quat, root_lin_vel_w, root_ang_vel_b, body_pos[..., ti, :], body_quat[..., ti, :], body_pos[..., list(hi), :], body_pos[..., list(fi), :])
