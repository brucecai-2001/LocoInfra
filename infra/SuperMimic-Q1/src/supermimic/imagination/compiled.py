"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, analytic FK reward, termination logits and the
next noise-free observation into one call so ``torch.compile`` can fuse the hundreds of
small kernels of the per-body FK loop. Reference quantities are passed as plain tensors
(already gathered from the motion library) so the compiled graph has static shapes.
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
    "torque",
    "power",
    "over_rated",
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        s_next = self.wm.step(s, a, a_prev, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies, xi)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (terms, n)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)


class FreeImagStep:
    """One imagined step of a free-command teleoperation rollout: dynamics, slot-tracking reward
    (supermimic/tele/interface.py), fall-only termination against the upright virtual reference at the
    commanded pelvis height, next clean observation."""

    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, tele_cfg, soft_limits: torch.Tensor, effort_weights: tuple[float, float, float], body_idx, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self.tele_cfg = tele_cfg
        self.soft_limits = soft_limits
        self.effort_weights = effort_weights
        self.body_idx = body_idx  # tele.interface.tracked_indices
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, goal, virt_root_pos, virt_root_quat):
        from supermimic.envs.mdp.rewards import effort_penalties
        from supermimic.robot import NQ, STATE_DIM
        from supermimic.tele.interface import FREE_REWARD_TERM_NAMES, free_reward_terms, slots_from_bodies

        s_next = self.wm.step(s, a, a_prev, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        ai = self.wm.heads.anchor_index  # torso_link
        q, qd = s_next[..., 7:NQ], s_next[..., NQ + 6 : STATE_DIM]
        act = self.wm.heads.actuator
        effort = effort_penalties(act.torque(q, qd, a, xi), qd, act.effort, act.rated) if act is not None else None
        robot = slots_from_bodies(
            s_next[..., 0:3], s_next[..., 3:7], s_next[..., NQ : NQ + 3], s_next[..., NQ + 3 : NQ + 6], bodies["pos"], bodies["quat"], self.body_idx
        )
        terms = free_reward_terms(
            self.tele_cfg, robot=robot, goal=goal,
            torso_quat=bodies["quat"][..., ai, :], torso_ang_vel_w=bodies["ang_vel"][..., ai, :],
            action=a, prev_action=a_prev, joint_pos=q, soft_limits=self.soft_limits,
            root_quat=s_next[..., 3:7], feet_pos=bodies["pos"][..., list(self.body_idx[2]), :], default_joint_pos=self.q0,
            effort=effort, effort_weights=self.effort_weights,
        )
        term_vec = torch.stack([terms[k] for k in FREE_REWARD_TERM_NAMES], dim=0)
        logits = self.wm.termination_logits(s_next, ReferenceFrame(None, None, None, None, virt_root_pos, virt_root_quat))
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)
        return s_next, terms["total"], term_vec, logits, obs_clean, anchor_pose

    def __call__(self, s, a, a_prev, xi, goal, virt_root_pos, virt_root_quat):
        return self._fn(s, a, a_prev, xi, goal, virt_root_pos, virt_root_quat)
