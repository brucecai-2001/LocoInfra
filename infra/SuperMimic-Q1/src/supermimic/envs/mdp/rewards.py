"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    # Effort terms on the joint torques tau (N·m) and velocities qd:
    #   torque     sum (tau / tau_peak)^2
    #   power      sum |tau qd|                       (W)
    #   over_rated sum relu(|tau| / tau_rated - 1)^2  (per-step proxy for time above the rated torque)
    w_torque: float = -0.02
    w_power: float = -1.0e-4
    w_over_rated: float = -0.05
    # Weight of the hand bodies in the body-orientation error (Q1 has no wrist: hands follow position only).
    hand_ori_weight: float = 0.2
    # Contact terms (envs/mdp/contact.py), off by default; used by the get-up / lie-down skills.
    #   extra_contact  a body touches the floor (clearance < contact_margin) while it is clearly up in the
    #                  reference (gate 0 at contact_ref_free, 1 at contact_ref_free + contact_ref_free_band)
    #   impact         lowest point of a body within impact_band of the floor moves down faster than impact_speed
    w_extra_contact: float = 0.0
    w_impact: float = 0.0
    contact_margin: float = 0.01
    contact_ref_free: float = 0.03
    contact_ref_free_band: float = 0.02
    impact_band: float = 0.03
    impact_speed: float = 0.6
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def effort_penalties(torque: torch.Tensor, joint_vel: torch.Tensor, torque_peak: torch.Tensor, torque_rated: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unweighted effort penalties (N,), shared by the simulator rewards and the imagination."""
    return {
        "torque": ((torque / torque_peak) ** 2).sum(-1),
        "power": (torque * joint_vel).abs().sum(-1),
        "over_rated": ((torque.abs() / torque_rated - 1.0).clamp(min=0.0) ** 2).sum(-1),
    }


def body_ori_weights(body_names: tuple[str, ...], hand_bodies: tuple[str, ...], hand_weight: float) -> torch.Tensor:
    w = torch.tensor([hand_weight if n in hand_bodies else 1.0 for n in body_names])
    return w * (len(w) / w.sum())


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
    body_ori_weights: torch.Tensor | None = None,  # (B,) normalised to mean 1
    torque: torch.Tensor | None = None,  # (N, NJ) joint torques; None -> effort terms are zero
    joint_vel: torch.Tensor | None = None,  # (N, NJ)
    torque_peak: torch.Tensor | None = None,  # (NJ,)
    torque_rated: torch.Tensor | None = None,  # (NJ,)
    contact_table: tuple[torch.Tensor, torch.Tensor] | None = None,  # (verts (B,V,3), mask (B,)) from contact.py
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w)
    e_body_ori = (e_body_ori * body_ori_weights).mean(-1) if body_ori_weights is not None else e_body_ori.mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    if torque is not None:
        e = effort_penalties(torque, joint_vel, torque_peak, torque_rated)
        terms["torque"] = dt * cfg.w_torque * e["torque"]
        terms["power"] = dt * cfg.w_power * e["power"]
        terms["over_rated"] = dt * cfg.w_over_rated * e["over_rated"]
    else:
        zero = torch.zeros_like(action_rate)
        terms.update(torque=zero, power=zero, over_rated=zero)
    if contact_table is not None and (cfg.w_extra_contact or cfg.w_impact):
        from supermimic.envs.mdp.contact import contact_penalties

        c = contact_penalties(robot_body_pos_w, robot_body_quat_w, robot_body_lin_vel_w, robot_body_ang_vel_w,
                              ref_body_pos_w, ref_body_quat_w, *contact_table, cfg)
        terms["extra_contact"] = dt * cfg.w_extra_contact * c["extra_contact"]
        terms["impact"] = dt * cfg.w_impact * c["impact"]
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms
