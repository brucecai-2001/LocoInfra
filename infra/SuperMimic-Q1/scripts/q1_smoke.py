"""Q1 environment smoke test: dimensions, contacts, and an open-loop reference-PD rollout.

  SUPERMIMIC_ROBOT=q1 uv run python scripts/q1_smoke.py --npz-dir data/npz/q1_walk

The "reference PD" policy sets each PD target to the reference joint angle of the current
frame (no feedback). It cannot balance for long, but how long it survives and how far it is
from the reference tells whether gains, action scale, contacts and the reference are sane.
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robot import GOAL_FEAT_DIM, NJ, OBS_DIM, SPEC


def main(npz_dir: Path, num_envs: int = 64, steps: int = 250, device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir)]
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, num_envs=num_envs, action_noise_std=0.0, obs_noise=False, reset_noise=False,
            domain_randomization=True, sampling_mode="uniform", record_transitions=True, seed=0,
        ),
        device=device,
    )
    robot = env.scene["robot"]
    print(f"[{SPEC.name}] NJ={NJ} joints={robot.joint_names[:3]}... bodies={len(robot.body_names)}")
    print("default_joint_pos", [round(x, 2) for x in robot.data.default_joint_pos[0].tolist()])
    obs, _ = env.reset()
    assert obs["policy"].shape == (num_envs, 10 * OBS_DIM), obs["policy"].shape
    assert obs["goal"].shape == (num_envs, 10 * GOAL_FEAT_DIM), obs["goal"].shape
    assert env.action_manager.total_action_dim == NJ
    term = env.action_manager.get_term("joint_pos")
    scale = term._scale if isinstance(term._scale, torch.Tensor) else torch.full((NJ,), float(term._scale), device=device)
    print("action scale", scale.reshape(-1)[:NJ].tolist()[:6], "...")

    cmd = env.command_manager.get_term("motion")
    q0 = robot.data.default_joint_pos
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survived = torch.zeros(num_envs, device=device)
    err_j, err_root_z, ncon = [], [], []
    for t in range(steps):
        act = (cmd.joint_pos - q0) / scale
        obs, rew, terminated, tout, _ = env.step(act)
        alive &= ~terminated
        survived += alive.float()
        err_j.append((cmd.joint_pos - robot.data.joint_pos).abs().mean().item())
        err_root_z.append((cmd.root_pos_w[:, 2] - robot.data.body_link_pos_w[:, cmd.robot_root_body_index, 2]).abs().mean().item())
        ncon.append(float(env.sim.data.nacon[0]) if hasattr(env.sim.data, "nacon") else float("nan"))
    print(f"reference-PD: survived mean {survived.mean().item() / 50:.2f}s  min {survived.min().item() / 50:.2f}s  "
          f"over {steps / 50:.0f}s; |q - q_ref| {sum(err_j) / len(err_j):.3f} rad; |z - z_ref| {sum(err_root_z) / len(err_root_z) * 100:.1f} cm; "
          f"contacts/world ~{sum(ncon) / len(ncon) / num_envs:.1f}")
    rec = env.recorder_manager.get_term("replay")
    print("replay buffer size", rec.buffer.size, "xi", rec.ctx.read()[0, :3].tolist())
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
