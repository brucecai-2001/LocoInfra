"""Checks of the physics randomization (envs/dr.py) on the selected robot.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/q1_dr_check.py --npz-dir data/npz/q1_walk

* the context xi carries the sampled values and the model fields really changed;
* command latency: with a full-step lag the applied set-point is still the previous one;
* the analytic actuator model (used by the imagination's effort terms) matches the simulator torque;
* sensor latency: lam = 0 is the identity, lam = 1 shifts every sensor frame by one step.
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.dr import IDX, actuator_model, dr_context
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robot import NJ, OBS_DIM, XI_BASE_DIM, XI_EXTRA_NAMES
from supermimic.world_model.heads import OBS_TERMS, delay_policy_obs


def main(npz_dir: Path, num_envs: int = 64, device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir)]
    env = make_env(
        SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, action_noise_std=0.0, obs_noise=False, sampling_mode="uniform", seed=0),
        device=device,
    )
    obs, _ = env.reset()
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(num_envs, NJ, device=device))
    rec = env.recorder_manager.get_term("replay")
    xi = rec.ctx.read()
    ex = xi[:, XI_BASE_DIM:]
    for i, n in enumerate(XI_EXTRA_NAMES):
        print(f"  {n:15s} min {ex[:, i].min():.3f} max {ex[:, i].max():.3f}")

    # Model fields follow the context.
    m, robot = env.sim.model, env.scene["robot"]
    act = actuator_model(env)
    c = torch.cat([a.global_ctrl_ids for a in robot.actuators]).long()
    kp_ratio = m.actuator_gainprm[:, c, 0] / env.sim.get_default_field("actuator_gainprm")[c, 0]
    assert torch.allclose(kp_ratio, ex[:, IDX["kp_scale"], None].expand_as(kp_ratio), atol=1e-5)
    bodies = robot.indexing.body_ids.long()
    mass_ratio = m.body_mass[:, bodies].sum(-1) / env.sim.get_default_field("body_mass")[bodies].sum()
    assert torch.allclose(mass_ratio, ex[:, IDX["mass_scale"]], atol=1e-5)
    dofs = robot.indexing.joint_v_adr.long()
    assert torch.allclose(m.dof_frictionloss[:, dofs], ex[:, IDX["joint_friction"], None].expand(-1, NJ), atol=1e-6)
    print("model fields match xi")

    # Command latency: full-step lag keeps the old set-point for the whole step.
    term = env.action_manager.get_term("joint_pos")
    ctx = dr_context(env)
    ctx[:, IDX["action_delay"]] = 0.0
    ctx[: num_envs // 2, IDX["action_delay"]] = 1.0
    a0 = torch.zeros(num_envs, NJ, device=device)
    env.step(a0)
    target0 = term._processed_actions.clone()
    a1 = torch.full((num_envs, NJ), 0.5, device=device)
    env.step(a1)
    applied = robot.data.joint_pos_target + robot.data.encoder_bias
    held, fresh = applied[: num_envs // 2], applied[num_envs // 2 :]
    reset = env.reset_buf
    ok_held = torch.allclose(held[~reset[: num_envs // 2]], target0[: num_envs // 2][~reset[: num_envs // 2]], atol=1e-5)
    ok_fresh = torch.allclose(fresh[~reset[num_envs // 2 :]], term._processed_actions[num_envs // 2 :][~reset[num_envs // 2 :]], atol=1e-5)
    print(f"command latency: held={ok_held} fresh={ok_fresh}")
    assert ok_held and ok_fresh

    # Analytic torque vs simulator (the simulator force is from the last substep's pre-integration state).
    ctx[:, IDX["action_delay"]] = 0.0
    a2 = 0.3 * torch.randn(num_envs, NJ, device=device)
    env.step(a2)
    s_q, s_qd = robot.data.joint_pos, robot.data.joint_vel
    tau_model = act.torque(s_q, s_qd, a2, rec.ctx.read())
    tau_sim = robot.data.qfrc_actuator
    keep = ~env.reset_buf
    rel = (tau_model - tau_sim)[keep].abs().mean() / tau_sim[keep].abs().mean()
    print(f"torque model vs sim: mean |diff| / mean |tau| = {rel:.3f}  (|tau| mean {tau_sim[keep].abs().mean():.2f} N·m, max {tau_sim[keep].abs().max():.1f})")
    assert rel < 0.25

    # Sensor latency.
    o = obs["policy"]
    assert torch.equal(delay_policy_obs(o, torch.zeros(num_envs, device=device)), o)
    d1 = delay_policy_obs(o, torch.ones(num_envs, device=device))
    H = o.shape[-1] // OBS_DIM
    start = 0
    for name, d in OBS_TERMS:
        x, y = o[:, start : start + H * d].reshape(num_envs, H, d), d1[:, start : start + H * d].reshape(num_envs, H, d)
        start += H * d
        if name == "actions":
            assert torch.equal(x, y)
        else:
            assert torch.equal(y[:, 1:], x[:, :-1]) and torch.equal(y[:, 0], x[:, 0])
    print("sensor latency ok")
    env.close()
    print("all checks passed")


if __name__ == "__main__":
    tyro.cli(main)
