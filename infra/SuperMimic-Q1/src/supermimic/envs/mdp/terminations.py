"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out.

Free-command teleoperation envs (``cmd.free_mask``) test against a virtual reference instead of the
clip: upright, pelvis at the standing height plus the commanded height offset (``cmd.virt_height``),
and never time out at the clip end. Distill envs time out when the reference leaves the tele_ok frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    ref_h = cmd.root_pos_w[:, 2]
    if cmd.free_mask.any():
        ref_h = torch.where(cmd.free_mask, env.scene.env_origins[:, 2] + cmd.virt_height, ref_h)
    return (ref_h - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = torch.where(cmd.free_mask, g[:, 2], quat_apply_inverse(cmd.root_quat_w, g)[:, 2])
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    end = cmd.clip_end
    if cmd.tele_cfg is not None:
        end = end | (cmd.distill_mask & ~cmd.library.tele_ok[cmd.frame])
    return end & ~cmd.free_mask
