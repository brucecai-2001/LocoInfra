"""Simulator stress test: random actions, report non-finite / extreme states and which bodies explode.

  SUPERMIMIC_ROBOT=q1 uv run python scripts/q1_stress.py --npz-dir data/npz/q1_walk
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import NQ, StateIO
from supermimic.robot import NJ


def main(npz_dir: Path, num_envs: int = 4096, steps: int = 1500, action_std: float = 2.0, device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir)]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, record_transitions=False, seed=1), device=device)
    io = StateIO(env)
    env.reset()
    robot = env.scene["robot"]
    worst_qd, worst_v, bad_total = 0.0, 0.0, 0
    for t in range(steps):
        act = action_std * torch.randn(num_envs, NJ, device=device)
        env.step(act)
        s = io.read()
        finite = torch.isfinite(s).all(-1)
        qd = s[:, NQ + 6 :].abs().max(-1).values
        v = s[:, NQ : NQ + 6].abs().max(-1).values
        bad = (~finite) | (qd > 200) | (v > 50)
        if bad.any():
            ids = bad.nonzero().squeeze(-1)
            bad_total += ids.numel()
            if bad_total <= 5 * ids.numel():
                e = int(ids[0])
                j = int(s[e, NQ + 6 :].abs().argmax())
                print(f"step {t}: {ids.numel()} bad envs; env {e} finite={bool(finite[e])} max|qd|={qd[e]:.1f} ({robot.joint_names[j]}) max|v|={v[e]:.1f}")
        worst_qd = max(worst_qd, float(qd[finite].max()) if finite.any() else 0.0)
        worst_v = max(worst_v, float(v[finite].max()) if finite.any() else 0.0)
    print(f"{steps} steps x {num_envs} envs, action std {action_std}: bad env-steps {bad_total}, worst |qd| {worst_qd:.1f} rad/s, worst root |v|/|w| {worst_v:.1f}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
