"""Per-environment physics randomization that the world model sees through the sim context xi.

Every term draws one scalar per environment (shared by all joints / bodies) so that the
context stays small and the world model can condition on the exact values:

    mass_scale      all robot body masses and inertias            (startup: needs set_const)
    armature_scale  joint armature                                 (startup: needs set_const_0)
    kp_scale        position-actuator stiffness                    (reset)
    kd_scale        position-actuator damping                      (reset)
    effort_scale    actuator force range (torque limit)            (reset)
    joint_friction  joint friction loss, N·m, absolute             (reset)
    action_delay    command latency, fraction of a control step    (reset; executed in actions.py)
    obs_delay       sensor latency, fraction of a control step     (reset; ``delay_policy_obs``)

The values live in ``env._sm_dr`` (num_envs, len(XI_EXTRA_NAMES)), nominal until an event writes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from supermimic.robot import XI_EXTRA_NAMES, XI_EXTRA_NOMINAL

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

IDX = {n: i for i, n in enumerate(XI_EXTRA_NAMES)}


def dr_context(env: ManagerBasedRlEnv) -> torch.Tensor:
    ctx = getattr(env, "_sm_dr", None)
    if ctx is None:
        ctx = torch.tensor(XI_EXTRA_NOMINAL, device=env.device).repeat(env.num_envs, 1)
        env._sm_dr = ctx
    return ctx


def _ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device, dtype=torch.long)


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def _robot(env: ManagerBasedRlEnv):
    return env.scene["robot"]


def _ctrl_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.cat([a.global_ctrl_ids for a in _robot(env).actuators]).to(env.device, dtype=torch.long)


def _joint_dof_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _robot(env).indexing.joint_v_adr.to(env.device, dtype=torch.long)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def mass_scale(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, scale_range: tuple[float, float]) -> None:
    ids = _ids(env, env_ids)
    s = _uniform(*scale_range, len(ids), env.device)
    bodies = _robot(env).indexing.body_ids.to(env.device, dtype=torch.long)
    m0 = env.sim.get_default_field("body_mass")[bodies]
    i0 = env.sim.get_default_field("body_inertia")[bodies]
    env.sim.model.body_mass[ids[:, None], bodies] = m0[None] * s[:, None]
    env.sim.model.body_inertia[ids[:, None], bodies] = i0[None] * s[:, None, None]
    dr_context(env)[ids, IDX["mass_scale"]] = s


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
def armature_scale(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, scale_range: tuple[float, float]) -> None:
    ids = _ids(env, env_ids)
    s = _uniform(*scale_range, len(ids), env.device)
    dofs = _joint_dof_ids(env)
    a0 = env.sim.get_default_field("dof_armature")[dofs]
    env.sim.model.dof_armature[ids[:, None], dofs] = a0[None] * s[:, None]
    dr_context(env)[ids, IDX["armature_scale"]] = s


@requires_model_fields("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
def actuator_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    effort_range: tuple[float, float],
) -> None:
    """Scale the built-in position actuators: force = kp (target - q) - kd qd, clipped to the force range."""
    ids = _ids(env, env_ids)
    n = len(ids)
    kp, kd, ef = (_uniform(*r, n, env.device) for r in (kp_range, kd_range, effort_range))
    c = _ctrl_ids(env)
    g0 = env.sim.get_default_field("actuator_gainprm")[c]
    b0 = env.sim.get_default_field("actuator_biasprm")[c]
    f0 = env.sim.get_default_field("actuator_forcerange")[c]
    m = env.sim.model
    m.actuator_gainprm[ids[:, None], c, 0] = g0[None, :, 0] * kp[:, None]
    m.actuator_biasprm[ids[:, None], c, 1] = b0[None, :, 1] * kp[:, None]
    m.actuator_biasprm[ids[:, None], c, 2] = b0[None, :, 2] * kd[:, None]
    m.actuator_forcerange[ids[:, None], c, 0] = f0[None, :, 0] * ef[:, None]
    m.actuator_forcerange[ids[:, None], c, 1] = f0[None, :, 1] * ef[:, None]
    ctx = dr_context(env)
    ctx[ids, IDX["kp_scale"]] = kp
    ctx[ids, IDX["kd_scale"]] = kd
    ctx[ids, IDX["effort_scale"]] = ef


@requires_model_fields("dof_frictionloss")
def joint_friction(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, friction_range: tuple[float, float]) -> None:
    ids = _ids(env, env_ids)
    f = _uniform(*friction_range, len(ids), env.device)
    env.sim.model.dof_frictionloss[ids[:, None], _joint_dof_ids(env)] = f[:, None]
    dr_context(env)[ids, IDX["joint_friction"]] = f


def latency(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    action_delay_s: tuple[float, float],
    obs_delay_s: tuple[float, float],
) -> None:
    """Action latency is quantised to physics substeps; observation latency is continuous."""
    ids = _ids(env, env_ids)
    dec = env.cfg.decimation
    phys = env.physics_dt
    lo, hi = round(action_delay_s[0] / phys), round(action_delay_s[1] / phys)
    lag = torch.randint(lo, hi + 1, (len(ids),), device=env.device).clamp_max(dec)
    ctx = dr_context(env)
    ctx[ids, IDX["action_delay"]] = lag.float() / dec
    ctx[ids, IDX["obs_delay"]] = (_uniform(*obs_delay_s, len(ids), env.device) / env.step_dt).clamp(0.0, 1.0)


def action_lag_substeps(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.round(dr_context(env)[:, IDX["action_delay"]] * env.cfg.decimation).long()


def actuator_model(env: ManagerBasedRlEnv):
    """Nominal position-actuator parameters in joint order (cached on the env)."""
    import re

    from supermimic.robot import SPEC
    from supermimic.world_model.heads import ActuatorModel

    cached = getattr(env, "_sm_act", None)
    if cached is not None:
        return cached
    robot = _robot(env)
    m = env.sim.mj_model
    dof_adr = robot.indexing.joint_v_adr.cpu().tolist()
    nj = len(dof_adr)
    kp, kd, effort = (torch.zeros(nj) for _ in range(3))
    for c in _ctrl_ids(env).cpu().tolist():
        j = dof_adr.index(int(m.jnt_dofadr[m.actuator_trnid[c, 0]]))
        kp[j] = float(m.actuator_gainprm[c, 0])
        kd[j] = -float(m.actuator_biasprm[c, 2])
        effort[j] = float(m.actuator_forcerange[c, 1])
    rated = effort.clone()
    if SPEC.rated_torque is not None:
        for pat, v in SPEC.rated_torque().items():
            for j, name in enumerate(SPEC.joint_names):
                if re.fullmatch(pat, name):
                    rated[j] = v
    scale = env.action_manager.get_term("joint_pos")._scale
    scale = scale[0].detach().cpu() if isinstance(scale, torch.Tensor) else torch.full((nj,), float(scale))
    d = env.device
    act = ActuatorModel(kp.to(d), kd.to(d), effort.to(d), rated.to(d), scale.to(d), robot.data.default_joint_pos[0].clone())
    env._sm_act = act
    return act


def policy_obs(env: ManagerBasedRlEnv, obs: dict) -> torch.Tensor:
    """The policy-group observation with this env's sensor latency applied."""
    from supermimic.world_model.heads import delay_policy_obs

    return delay_policy_obs(obs["policy"], dr_context(env)[:, IDX["obs_delay"]])
