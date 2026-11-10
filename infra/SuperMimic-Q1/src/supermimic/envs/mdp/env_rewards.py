"""Simulator-side reward terms that have no mjlab counterpart (mirrors of ``rewards.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from supermimic.envs.dr import actuator_model
from supermimic.envs.mdp.rewards import effort_penalties

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_relative_body_orientation_error_exp_weighted(env: ManagerBasedRlEnv, command_name: str, std: float, weights: tuple[float, ...]) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    err = quat_error_magnitude(cmd.body_quat_relative_w, cmd.robot_body_quat_w) ** 2
    w = torch.tensor(weights, device=env.device)
    return torch.exp(-(err * w).mean(-1) / std**2)


def _effort(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
    robot = env.scene["robot"]
    act = actuator_model(env)
    return effort_penalties(robot.data.qfrc_actuator, robot.data.joint_vel, act.effort, act.rated)[key]


def joint_torque(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _effort(env, "torque")


def joint_power(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _effort(env, "power")


def torque_over_rated(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _effort(env, "over_rated")


_CONTACT_TABLES: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _contact(env: ManagerBasedRlEnv, command_name: str, cfg, key: str) -> torch.Tensor:
    from supermimic.envs.mdp.contact import contact_penalties, contact_vertex_table
    from supermimic.robot import load_robot_model

    cmd = env.command_manager.get_term(command_name)
    if env.device not in _CONTACT_TABLES:
        verts, mask = contact_vertex_table(load_robot_model(), tuple(cmd.cfg.body_names))
        _CONTACT_TABLES[env.device] = (verts.to(env.device), mask.to(env.device))
    return contact_penalties(cmd.robot_body_pos_w, cmd.robot_body_quat_w, cmd.robot_body_lin_vel_w, cmd.robot_body_ang_vel_w,
                             cmd.body_pos_w, cmd.body_quat_w, *_CONTACT_TABLES[env.device], cfg)[key]


def extra_contact(env: ManagerBasedRlEnv, command_name: str, cfg) -> torch.Tensor:
    return _contact(env, command_name, cfg, "extra_contact")


def contact_impact(env: ManagerBasedRlEnv, command_name: str, cfg) -> torch.Tensor:
    return _contact(env, command_name, cfg, "impact")
