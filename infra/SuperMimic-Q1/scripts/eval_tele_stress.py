"""E_tele stress test: time-varying teleoperation commands (fast switching, reversals, joystick noise,
moving hands, squat bobbing, slot dropouts), the way an operator drives PICO / a joystick.

Every env starts on a standing frame and follows one command timeline for ``seconds``; the height
command ramps at the training rate (as the PICO adapter). A fall ends the env (counted once).
Scenarios (``per_case`` envs each, different random timelines):

    steady        one random command (base / height / torso / hands as the free sampler) for the whole run
    switch_5s     a new random command every 5 s (the training distribution)
    switch_fast   a new random command every 0.3-1.5 s
    base_fast     base slot only, a new random velocity every 0.3-1.5 s (WASD / joystick)
    reversal      base only, one axis flipping between its extremes every 0.5-1.2 s
    stop_go       base only, random velocity and zero alternating every 0.4-1.2 s
    jitter        base only, a random velocity plus per-step noise (0.15 m/s, 0.1 m/s, 0.3 rad/s)
    hand_static   random base every 2 s, hands held at one library pose (control for the two below)
    hand_swing    walking (random base every 2 s) with both hands on 0.5-1.5 Hz sinusoids (12 cm) around a library pose
    hand_jump     random base every 2 s, hand targets jumping to a new library pose every 0.3-1.0 s
    squat_bob     random base every 2 s, height target toggling 0 / -0.15 m every 0.6-1.5 s
    mask_flicker  a random command every 3 s, every slot's mask bit toggling every 0.2-1.0 s (tracking loss)

``--randomize`` turns on the training randomization (friction, PD, latency, masses) and noise.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/eval_tele_stress.py --ckpt logs/<run>/ckpt_latest.pt \
      --npz-dir data/npz/mix_v1 --stand-clip <clip> --out logs/<run>/eval/tele_stress.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import make_env
from supermimic.envs.dr import policy_obs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import deterministic_env_args, load_policy_from_checkpoint
from supermimic.tele import slots as S
from supermimic.tele.interface import TeleCfg, base_velocity, sample_free_slots

SCENARIOS = ("steady", "switch_5s", "switch_fast", "base_2s", "base_fast", "reversal", "stop_go", "jitter",
             "hand_static", "hand_static_torso", "hand_swing", "hand_jump", "squat_bob", "mask_flicker")
LIVE = [i for i, s in enumerate(S.SLOTS) if not s.reserved]
BASE_I = S.SLOT_NAMES.index("base")


def change_points(rng: np.random.Generator, steps: int, lo: float, hi: float, dt: float) -> np.ndarray:
    """(steps,) segment index: a new segment every U(lo, hi) s."""
    seg = np.zeros(steps, np.int64)
    t, k = 0, 0
    while t < steps:
        d = max(1, int(rng.uniform(lo, hi) / dt))
        seg[t : t + d] = k
        t, k = t + d, k + 1
    return seg


def timelines(tc: TeleCfg, lib, n_per: int, steps: int, dt: float, seed: int, device: str):
    """-> slots (T, N, SLOT_DIM), mask (T, N, NUM_SLOTS), height target (T, N), scenario id (N,)."""
    rng = np.random.default_rng(seed)
    N = n_per * len(SCENARIOS)
    pool_n = 1 << 15
    pool, pool_h = (x.cpu().numpy() for x in sample_free_slots(tc, pool_n, device, lib))
    hands = lib.tele_slots[lib.tele_ok_frames[torch.randint(0, lib.tele_ok_frames.numel(), (pool_n,), device=device)]].cpu().numpy()
    dflt = S.default_vector().numpy()
    all_live = np.zeros(S.NUM_SLOTS, np.float32)
    all_live[LIVE] = 1.0
    base_only = np.zeros(S.NUM_SLOTS, np.float32)
    base_only[BASE_I] = 1.0
    sl, hs = S.SLICE["base"], S.SLICE["height"]
    hl, hr = S.SLICE["hand_L"], S.SLICE["hand_R"]
    lo = np.array([tc.vx_range[0], tc.vy_range[0], tc.wz_range[0]])
    hi = np.array([tc.vx_range[1], tc.vy_range[1], tc.wz_range[1]])
    slots = np.repeat(dflt[None, None], steps, 0).repeat(N, 1).astype(np.float32)
    mask = np.zeros((steps, N, S.NUM_SLOTS), np.float32)
    htgt = np.zeros((steps, N), np.float32)
    scen = np.repeat(np.arange(len(SCENARIOS)), n_per)
    t_s = np.arange(steps) * dt

    def draw(seg):  # pool rows per segment
        return rng.integers(0, pool_n, seg.max() + 1)[seg]

    for e in range(N):
        name = SCENARIOS[scen[e]]
        s, m, h = slots[:, e], mask[:, e], htgt[:, e]
        m[:] = all_live
        if name in ("steady", "switch_5s", "switch_fast"):
            seg = np.zeros(steps, np.int64) if name == "steady" else (change_points(rng, steps, 5.0, 5.0, dt) if name == "switch_5s" else change_points(rng, steps, 0.3, 1.5, dt))
            r = draw(seg)
            s[:] = pool[r]
            h[:] = pool_h[r]
        elif name in ("base_2s", "base_fast", "reversal", "stop_go", "jitter"):
            m[:] = base_only
            if name in ("base_2s", "base_fast"):
                seg = change_points(rng, steps, 2.0, 2.0, dt) if name == "base_2s" else change_points(rng, steps, 0.3, 1.5, dt)
                s[:, sl] = rng.uniform(lo, hi, (seg.max() + 1, 3))[seg]
            elif name == "reversal":
                ax = rng.integers(0, 3)
                seg = change_points(rng, steps, 0.5, 1.2, dt)
                v = np.where(seg % 2 == 0, hi[ax], lo[ax])
                s[:, sl] = 0.0
                s[:, sl.start + ax] = v
            elif name == "stop_go":
                seg = change_points(rng, steps, 0.4, 1.2, dt)
                v = rng.uniform(lo, hi, (seg.max() + 1, 3))[seg]
                s[:, sl] = np.where((seg % 2 == 0)[:, None], v, 0.0)
            else:
                v = rng.uniform(lo, hi, 3)
                s[:, sl] = np.clip(v + rng.normal(0, [0.15, 0.1, 0.3], (steps, 3)), lo, hi)
        elif name in ("hand_static", "hand_static_torso", "hand_swing", "hand_jump", "squat_bob"):
            seg = change_points(rng, steps, 2.0, 2.0, dt)
            r = draw(seg)
            s[:, sl] = pool[r][:, sl]
            m[:] = 0.0
            m[:, [BASE_I, S.SLOT_NAMES.index("height"), S.SLOT_NAMES.index("hand_L"), S.SLOT_NAMES.index("hand_R")]] = 1.0
            if name == "hand_static_torso":
                m[:, S.SLOT_NAMES.index("torso")] = 1.0
            if name in ("hand_static", "hand_static_torso"):
                c = hands[rng.integers(0, pool_n)]
                s[:, hl], s[:, hr] = c[hl], c[hr]
            elif name == "hand_swing":
                c = hands[rng.integers(0, pool_n)]
                f = rng.uniform(0.5, 1.5)
                ph = rng.uniform(0, 2 * np.pi, 6)
                w = 0.12 * np.sin(2 * np.pi * f * t_s[:, None] + ph[None])
                s[:, hl] = c[hl] + w[:, :3]
                s[:, hr] = c[hr] + w[:, 3:]
            elif name == "hand_jump":
                hseg = change_points(rng, steps, 0.3, 1.0, dt)
                hr_ = rng.integers(0, pool_n, hseg.max() + 1)[hseg]
                s[:, hl] = hands[hr_][:, hl]
                s[:, hr] = hands[hr_][:, hr]
            else:
                hseg = change_points(rng, steps, 0.6, 1.5, dt)
                h[:] = np.where(hseg % 2 == 0, 0.0, -0.15)
                s[:, sl] *= 0.5
        elif name == "mask_flicker":
            seg = change_points(rng, steps, 3.0, 3.0, dt)
            r = draw(seg)
            s[:] = pool[r]
            h[:] = pool_h[r]
            for i in LIVE:
                ms = change_points(rng, steps, 0.2, 1.0, dt)
                on = rng.random(ms.max() + 1) < 0.6
                m[:, i] = on[ms]
    tt = lambda x: torch.tensor(x, device=device)  # noqa: E731
    return tt(slots), tt(mask), tt(htgt), scen


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path, out: Path, stand_clip: str | None = None, per_case: int = 64, seconds: float = 20.0,
         randomize: bool = False, push_speed: float = 0.0, push_every_s: float = 4.0, device: str = "cuda:0", seed: int = 0) -> None:
    """``push_speed`` > 0: every ``push_every_s`` (random phase per env, first after 2 s) the pelvis gets that
    extra horizontal velocity in a random direction."""
    torch.manual_seed(seed)
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    tc = TeleCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["tele"].items()})
    tc.enabled, tc.free_env_fraction, tc.distill_env_fraction, tc.resample_steps = True, 1.0, 0.0, 10**9
    files = [str(f) for f in discover_npz(npz_dir)]
    N = per_case * len(SCENARIOS)
    e = cfg["env"]
    extra = dict(domain_randomization=True, physics_dr=True, obs_noise=True, action_noise_std=e.get("action_noise_std", 0.1)) if randomize else {}
    env = make_env(deterministic_env_args(files, N, future_frames=e["future_frames"], anchor_obs=e["anchor_obs"], anchor_ori_only=e["anchor_ori_only"], tele=tc, **extra), device=device)
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    names = [c.name for c in lib.clips]
    ci = names.index(stand_clip) if stand_clip else int(np.argmin([np.abs(lib.ref_cmd[c.start : c.end].cpu().numpy()).mean() for c in lib.clips]))
    steps = int(seconds / env.step_dt)
    slots, mask, htgt, scen = timelines(tc, lib, per_case, steps, env.step_dt, seed, device)
    hs = S.SLICE["height"]
    cmd.forced_frame[:] = lib.clip_start[ci]
    g0 = torch.cat([S.default_vector(device).expand(N, -1), mask[0]], -1)
    cmd.tele_forced = g0.clone()
    obs, _ = env.reset()
    d = cmd.robot.data
    fell = torch.zeros(N, dtype=torch.bool, device=device)
    t_fall = torch.full((N,), float(seconds), device=device)
    err = torch.zeros(N, 3, device=device)
    cnt = torch.zeros(N, device=device)
    lag = int(0.5 / env.step_dt)
    settle = int(1.0 / env.step_dt)
    b = slots[..., S.SLICE["base"]]
    changed = torch.cat([torch.ones(1, N, dtype=torch.bool, device=device), (b[1:] - b[:-1]).abs().sum(-1) > 1e-6], 0)
    idx = torch.arange(steps, device=device)[:, None].expand(-1, N)
    last_change = torch.cummax(torch.where(changed, idx, torch.zeros_like(idx)), 0).values
    settled = (idx - last_change) >= settle
    serr = torch.zeros(N, 3, device=device)
    scnt = torch.zeros(N, device=device)
    walk_cmd = torch.zeros(N, device=device)
    walk_meas = torch.zeros(N, device=device)
    robot = env.scene["robot"]
    push_n = int(push_every_s / env.step_dt)
    push_phase = int(2.0 / env.step_dt) + torch.randint(0, push_n, (N,), device=device)
    for t in range(steps):
        if push_speed > 0:
            due = (t >= push_phase) & ((t - push_phase) % push_n == 0) & ~fell
            if due.any():
                ang = torch.rand(N, device=device) * 2 * np.pi
                v = robot.data.root_link_vel_w.clone()
                v[:, 0] += push_speed * torch.cos(ang) * due
                v[:, 1] += push_speed * torch.sin(ang) * due
                robot.write_root_link_velocity_to_sim(v)
        h = cmd.tele_slots[:, hs].clone()
        cmd.tele_slots[:] = slots[t]
        cmd.tele_slots[:, hs] = h
        cmd.tele_slot_mask[:] = mask[t]
        cmd.height_target[:] = htgt[t]
        act = policy(policy_obs(env, obs), cmd.tele_goal, "tele")
        obs, _, term, _, _ = env.step(act)
        new = term & ~fell
        t_fall[new] = (t + 1) * env.step_dt
        fell |= term
        v = base_velocity(d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_b)
        if t >= lag:
            # base tracking against the command 0.5 s earlier, only while base is commanded and the env is alive
            c = slots[t - lag, :, S.SLICE["base"]]
            ok = (~fell) & (mask[t - lag, :, BASE_I] > 0)
            err += (v - c).abs() * ok[:, None]
            cnt += ok.float()
        # settled: the base command has not changed for 1 s
        ok = (~fell) & (mask[t, :, BASE_I] > 0) & settled[t]
        c = b[t]
        serr += (v - c).abs() * ok[:, None]
        scnt += ok.float()
        walk_cmd += c[:, :2].norm(dim=-1) * ok
        walk_meas += (v[:, :2] * c[:, :2]).sum(-1) / c[:, :2].norm(dim=-1).clamp_min(1e-6) * ok
    env.close()
    fell_np, tf, err_np = fell.cpu().numpy(), t_fall.cpu().numpy(), (err / cnt.clamp_min(1)[:, None]).cpu().numpy()
    serr_np, scnt_np = serr.cpu().numpy(), scnt.cpu().numpy()
    wc, wm = walk_cmd.cpu().numpy(), walk_meas.cpu().numpy()
    rows = {}
    print(f"{'scenario':14s} {'falls':>8s} {'t_fall':>7s}  {'|err| 0.5 s lag':>22s}  {'|err| settled':>22s}  speed ratio")
    for i, nme in enumerate(SCENARIOS):
        sel = scen == i
        f = fell_np[sel]
        n_s = scnt_np[sel].sum()
        rows[nme] = {"envs": int(sel.sum()), "falls": int(f.sum()), "fall_rate": float(f.mean()),
                     "mean_t_fall_s": float(tf[sel][f].mean()) if f.any() else None,
                     "base_abs_err_lag": [round(float(x), 3) for x in err_np[sel].mean(0)],
                     "base_abs_err_settled": [round(float(x), 3) for x in serr_np[sel].sum(0) / n_s] if n_s > 0 else None,
                     "planar_speed_ratio_settled": round(float(wm[sel].sum() / max(wc[sel].sum(), 1e-6)), 3) if n_s > 0 else None}
        r = rows[nme]
        print(f"{nme:14s} {r['falls']:>3d}/{r['envs']:<4d} {r['mean_t_fall_s'] or 0:>6.1f}s  {str(r['base_abs_err_lag']):>22s}  "
              f"{str(r['base_abs_err_settled']):>22s}  {r['planar_speed_ratio_settled']}")
    tot = {"envs": N, "falls": int(fell_np.sum()), "fall_rate": float(fell_np.mean())}
    print(f"{'total':14s} {tot['falls']:>3d}/{N}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": str(ckpt), "start_clip": names[ci], "seconds": seconds, "randomize": randomize, "seed": seed,
                               "push_speed": push_speed, "push_every_s": push_every_s,
                               "scenarios": rows, "total": tot,
                               "per_env": {"scenario": [SCENARIOS[i] for i in scen], "fell": fell_np.tolist(),
                                           "speed_ratio_settled": (wm / np.maximum(wc, 1e-6)).round(3).tolist(),
                                           "hand_z_cmd": slots[:, :, S.SLICE["hand_L"].start + 2].mean(0).cpu().numpy().round(3).tolist()}}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    tyro.cli(main)
