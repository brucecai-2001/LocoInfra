"""E_tele robustness (base slot only, the WASD / joystick use): push recovery and actuator-parameter offsets.

All randomization is pinned to nominal (foot friction 1.0, no encoder bias / COM offset / latency /
joint friction, nominal masses); no observation / action noise. Every env starts on a standing frame.

* push      each command of ``PUSH_CMDS`` is held; after ``push_at_s`` the pelvis gets an extra
            horizontal velocity of each speed in a random direction (``per_case`` envs per
            command x speed). Recovered = no fall within ``recover_s``. Also the velocity error
            over the last second.
* actuator  the base step-test grid of ``eval_tele.py`` under fixed per-env scales of kp, kd, torque
            limit and armature (one condition per block of envs); falls and tracking error
            against the nominal block.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/eval_tele_robust.py --ckpt logs/<run>/ckpt_latest.pt \
      --npz-dir data/npz/loco_v1 --stand-clip loco_v1__accad_Female1General_A1_Stand__0 --out logs/<run>/eval/robust.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.managers.event_manager import RecomputeLevel

from eval_tele import goal  # noqa: E402  (same directory)
from supermimic.envs import dr as sm_dr
from supermimic.envs import make_env
from supermimic.envs.dr import policy_obs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import deterministic_env_args, load_policy_from_checkpoint
from supermimic.tele import slots as S
from supermimic.tele.interface import TeleCfg, base_velocity

PUSH_CMDS = {"stand": (0.0, 0.0, 0.0), "fwd_0.3": (0.3, 0.0, 0.0), "fwd_0.6": (0.6, 0.0, 0.0), "side_0.3": (0.0, 0.3, 0.0), "turn_1.0": (0.0, 0.0, 1.0)}
ACT_CONDS = {
    "nominal": (1.0, 1.0, 1.0, 1.0),
    "kp_0.8": (0.8, 1.0, 1.0, 1.0), "kp_1.2": (1.2, 1.0, 1.0, 1.0),
    "kd_0.8": (1.0, 0.8, 1.0, 1.0), "kd_1.2": (1.0, 1.2, 1.0, 1.0),
    "effort_0.8": (1.0, 1.0, 0.8, 1.0),
    "armature_0.8": (1.0, 1.0, 1.0, 0.8), "armature_1.2": (1.0, 1.0, 1.0, 1.2),
    "soft (kp0.8 kd0.8 effort0.8 arm1.2)": (0.8, 0.8, 0.8, 1.2),
    "stiff (kp1.2 kd0.8 arm0.8)": (1.2, 0.8, 1.0, 0.8),
}


def make(cfg: dict, files: list[str], n: int, device: str):
    tc = TeleCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["tele"].items()})
    tc.enabled, tc.free_env_fraction, tc.distill_env_fraction, tc.resample_steps = True, 1.0, 0.0, 10**9
    e = cfg["env"]
    args = deterministic_env_args(
        files, n, future_frames=e["future_frames"], anchor_obs=e["anchor_obs"], anchor_ori_only=e["anchor_ori_only"], tele=tc,
        domain_randomization=True, physics_dr=True,  # registers the per-env model fields; every range pinned below
        foot_friction_range=(1.0, 1.0), encoder_bias_range=(0.0, 0.0), com_ranges={0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)},
        mass_scale_range=(1.0, 1.0), kp_scale_range=(1.0, 1.0), kd_scale_range=(1.0, 1.0), effort_scale_range=(1.0, 1.0),
        armature_scale_range=(1.0, 1.0), joint_friction_range=(0.0, 0.0), action_delay_s=(0.0, 0.0), obs_delay_s=(0.0, 0.0),
    )
    return make_env(args, device=device)


@torch.no_grad()
def rollout(env, policy, cmds: torch.Tensor, seconds: float, stand_frame: int, push_at_s: float | None = None, push_speed: torch.Tensor | None = None,
            scales: torch.Tensor | None = None, measure_s: float = 2.0):
    """-> fell (n,), mean base velocity over the last ``measure_s`` (n, 3)."""
    term_cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    n = cmds.shape[0]
    term_cmd.forced_frame[:] = stand_frame
    term_cmd.tele_forced = torch.tensor(np.stack([goal(("base",), base=tuple(c)) for c in cmds.cpu().numpy()]), device=cmds.device, dtype=torch.float32)
    obs, _ = env.reset()
    if scales is not None:
        all_ids = torch.arange(n, device=cmds.device)
        # One call per distinct condition (the event functions draw one scalar per call range).
        for s in torch.unique(scales, dim=0):
            ids = all_ids[(scales == s).all(-1)]
            kp, kd, ef, arm = (float(x) for x in s)
            sm_dr.actuator_scale(env, ids, (kp, kp), (kd, kd), (ef, ef))
            sm_dr.armature_scale(env, ids, (arm, arm))
        env.sim.recompute_constants(RecomputeLevel.set_const_0)
    steps = int(seconds / env.step_dt)
    tail = int(measure_s / env.step_dt)
    push_step = int(push_at_s / env.step_dt) if push_at_s is not None else -1
    fell = torch.zeros(n, dtype=torch.bool, device=cmds.device)
    acc = torch.zeros(n, 3, device=cmds.device)
    for t in range(steps):
        if t == push_step:
            ang = torch.rand(n, device=cmds.device) * 2 * np.pi
            v = robot.data.root_link_vel_w.clone()
            v[:, 0] += push_speed * torch.cos(ang)
            v[:, 1] += push_speed * torch.sin(ang)
            robot.write_root_link_velocity_to_sim(v)
        act = policy(policy_obs(env, obs), term_cmd.tele_goal, "tele")
        obs, _, term, _, _ = env.step(act)
        fell |= term
        if t >= steps - tail:
            d = robot.data
            acc += base_velocity(d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_b)
    return fell.cpu().numpy(), (acc / tail).cpu().numpy()


def main(ckpt: Path, npz_dir: Path, out: Path, stand_clip: str | None = None, push_speeds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
         per_case: int = 32, push_at_s: float = 3.0, recover_s: float = 4.0, step_seconds: float = 6.0, device: str = "cuda:0", seed: int = 0) -> None:
    torch.manual_seed(seed)
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    files = [str(f) for f in discover_npz(npz_dir)]
    report: dict = {"ckpt": str(ckpt)}

    # ---------------------------------------------------------------- push recovery
    cases = [(c, sp) for c in PUSH_CMDS for sp in push_speeds]
    n = len(cases) * per_case
    env = make(cfg, files, n, device)
    lib = env.command_manager.get_term("motion").library
    names = [c.name for c in lib.clips]
    ci = names.index(stand_clip) if stand_clip else 0
    stand_frame = int(lib.clip_start[ci])
    cmds = torch.tensor([PUSH_CMDS[c] for c, _ in cases], device=device).repeat_interleave(per_case, 0)
    speeds = torch.tensor([sp for _, sp in cases], device=device).repeat_interleave(per_case)
    fell, mean_v = rollout(env, policy, cmds, push_at_s + recover_s, stand_frame, push_at_s, speeds, measure_s=1.0)
    env.close()
    err = np.abs(mean_v - cmds.cpu().numpy())
    push = {}
    print(f"push recovery ({per_case} random directions per case, recovered = no fall within {recover_s:.0f} s)")
    print(f"{'command':10s} " + " ".join(f"{sp:>8.1f}" for sp in push_speeds) + "   m/s")
    for c in PUSH_CMDS:
        row = {}
        for sp in push_speeds:
            k = cases.index((c, sp))
            sl = slice(k * per_case, (k + 1) * per_case)
            ok = ~fell[sl]
            row[f"{sp:.1f}"] = {"falls": int(fell[sl].sum()), "pushes": per_case, "vel_abs_err_last_1s": [round(float(x), 3) for x in err[sl][ok].mean(0)] if ok.any() else None}
        push[c] = row
        print(f"{c:10s} " + " ".join(f"{row[f'{sp:.1f}']['falls']:>5d}/{per_case}" for sp in push_speeds))
    report["push"] = {"push_at_s": push_at_s, "recover_s": recover_s, "by_command": push}

    # ---------------------------------------------------------------- actuator offsets
    from eval_tele import cases

    grid = torch.tensor(np.stack([g[S.SLICE["base"]] for grp, _, g in cases() if grp == "base"]), device=device, dtype=torch.float32)
    G = grid.shape[0]
    conds = list(ACT_CONDS)
    env = make(cfg, files, G * len(conds), device)
    cmds = grid.repeat(len(conds), 1)
    scales = torch.tensor([ACT_CONDS[c] for c in conds], device=device).repeat_interleave(G, 0)
    fell, mean_v = rollout(env, policy, cmds, step_seconds, stand_frame, scales=scales)
    env.close()
    c_np = cmds.cpu().numpy()
    e = mean_v - c_np
    rel = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(c_np, axis=1), 1e-6)
    act = {}
    print("\nstep test under actuator offsets (29 commands each)")
    for i, c in enumerate(conds):
        sl = slice(i * G, (i + 1) * G)
        ok = ~fell[sl]
        ok_nz = ok.copy()
        ok_nz[0] = False  # the standing command has no relative error
        act[c] = {
            "falls": int(fell[sl].sum()),
            "median_rel_err": float(np.median(rel[sl][ok_nz])) if ok_nz.any() else None,
            "mean_abs_err": [round(float(x), 3) for x in np.abs(e[sl][ok]).mean(0)] if ok.any() else None,
        }
    base = act["nominal"]
    for c in conds:
        a = act[c]
        a["abs_err_vs_nominal"] = round(float(np.mean(a["mean_abs_err"]) / np.mean(base["mean_abs_err"])), 3) if a["mean_abs_err"] else None
        print(f"{c:38s} falls {a['falls']:2d}/{G}  median rel {a['median_rel_err']:.3f}  abs err {a['mean_abs_err']}  x{a['abs_err_vs_nominal']:.2f} vs nominal")
    report["actuator"] = act
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("written:", out)


if __name__ == "__main__":
    tyro.cli(main)
