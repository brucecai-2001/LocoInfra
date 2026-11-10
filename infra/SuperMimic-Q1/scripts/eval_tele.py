"""E_tele command tests (docs/training.md), deterministic physics, no noise / randomization, foot friction 1.0.

From a standing reference frame every env holds one teleoperation goal for ``seconds`` (the height
command ramps at the training rate, as the PICO adapter does). Over the last 2 s the robot's own slots
are averaged and compared with the command. Groups:

    base     29 velocity commands, base slot only (the WASD app)                 relative error as eval_velocity
    height   Δh steps standing and while walking, base + height                 |Δh error| (m)
    torso    pitch / roll leans, base + height + torso                          angle between gravity directions (rad)
    hands    hand targets standing / walking / squatting, base + height + hands  mean hand position error (m)
    mask     every slot at its default, then each slot masked in turn, none     falls only

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/eval_tele.py --ckpt logs/<run>/ckpt_latest.pt \
      --npz-dir data/npz/loco_v1 --out logs/<run>/eval/tele_steps.json
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
from supermimic.tele.interface import TeleCfg, slots_from_bodies


def goal(mask: tuple[str, ...], **vals) -> np.ndarray:
    g = np.concatenate([S.default_vector().numpy(), np.zeros(S.NUM_SLOTS, dtype=np.float32)])
    for k, v in vals.items():
        g[S.SLICE[k]] = v
    for n in mask:
        g[S.SLOT_DIM + S.SLOT_NAMES.index(n)] = 1.0
    return g


def lean(pitch: float, roll: float = 0.0) -> np.ndarray:
    return S.torso_gravity(torch.tensor(pitch), torch.tensor(roll)).numpy()


def cases() -> list[tuple[str, str, np.ndarray]]:
    out = []
    base = [(0.0, 0.0, 0.0)]
    base += [(vx, 0.0, 0.0) for vx in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0)]
    base += [(0.0, vy, 0.0) for vy in (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)]
    base += [(0.0, 0.0, wz) for wz in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)]
    base += [(vx, vy, 0.0) for vx in (-0.3, 0.4) for vy in (-0.3, 0.3)]
    base += [(vx, 0.0, wz) for vx in (0.3, 0.6) for wz in (-1.0, 1.0)]
    out += [("base", f"base {b}", goal(("base",), base=b)) for b in base]
    bh = ("base", "height")
    out += [("height", f"dh {h:+.2f}", goal(bh, height=h)) for h in (-0.12, -0.08, -0.04, 0.02)]
    out += [("height", f"dh {h:+.2f} vx 0.3", goal(bh, height=h, base=(0.3, 0.0, 0.0))) for h in (-0.08, -0.04)]
    out += [("height", "dh -0.06 wz 0.8", goal(bh, height=-0.06, base=(0.0, 0.0, 0.8)))]
    bht = ("base", "height", "torso")
    for p, r in ((0.2, 0.0), (0.4, 0.0), (0.0, 0.15), (0.0, -0.15), (0.3, 0.1)):
        out.append(("torso", f"pitch {p} roll {r}", goal(bht, torso=lean(p, r))))
    out.append(("torso", "pitch 0.3 dh -0.06", goal(bht, torso=lean(0.3), height=-0.06)))
    out.append(("torso", "pitch 0.2 vx 0.3", goal(bht, torso=lean(0.2), base=(0.3, 0.0, 0.0))))
    bhh = ("base", "height", "hand_L", "hand_R")
    reach = {"forward": ((0.20, 0.12, -0.15), (0.20, -0.12, -0.15)), "up": ((0.08, 0.15, 0.05), (0.08, -0.15, 0.05)),
             "side": ((0.0, 0.30, -0.20), (0.0, -0.30, -0.20)), "left_only": ((0.22, 0.05, -0.10), S.SLOTS[4].default)}
    for k, (hl, hr) in reach.items():
        out.append(("hands", f"{k}", goal(bhh, hand_L=hl, hand_R=hr)))
    out.append(("hands", "forward vx 0.3", goal(bhh, hand_L=reach["forward"][0], hand_R=reach["forward"][1], base=(0.3, 0.0, 0.0))))
    out.append(("hands", "forward wz 0.8", goal(bhh, hand_L=reach["forward"][0], hand_R=reach["forward"][1], base=(0.0, 0.0, 0.8))))
    out.append(("hands", "forward dh -0.08", goal(bhh, hand_L=((0.22, 0.12, -0.25)), hand_R=((0.22, -0.12, -0.25)), height=-0.08)))
    live = [s.name for s in S.SLOTS if not s.reserved]
    out.append(("mask", "all default", goal(tuple(live))))
    for n in live:
        out.append(("mask", f"without {n}", goal(tuple(x for x in live if x != n))))
    out.append(("mask", "none", goal(())))
    return out


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path, out: Path, stand_clip: str | None = None, seconds: float = 6.0, device: str = "cuda:0") -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    assert "tele" in policy.encoders, "checkpoint has no E_tele"
    tc = TeleCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["tele"].items()})
    tc.enabled, tc.free_env_fraction, tc.distill_env_fraction, tc.resample_steps = True, 1.0, 0.0, 10**9
    files = [str(f) for f in discover_npz(npz_dir)]
    cs = cases()
    n = len(cs)
    env = make_env(
        deterministic_env_args(files, n, future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"]["anchor_obs"], anchor_ori_only=cfg["env"]["anchor_ori_only"], tele=tc),
        device=device,
    )
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    names = [c.name for c in lib.clips]
    ci = names.index(stand_clip) if stand_clip else int(np.argmin([np.abs(lib.ref_cmd[c.start : c.end].cpu().numpy()).mean() for c in lib.clips]))
    cmd.forced_frame[:] = lib.clip_start[ci]
    cmd.tele_forced = torch.tensor(np.stack([c[2] for c in cs]), device=device, dtype=torch.float32)
    obs, _ = env.reset()
    d = cmd.robot.data
    steps = int(seconds / env.step_dt)
    tail = int(2.0 / env.step_dt)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    acc = torch.zeros(n, S.SLOT_DIM, device=device)
    for t in range(steps):
        act = policy(policy_obs(env, obs), cmd.tele_goal, "tele")
        obs, _, term, _, _ = env.step(act)
        fell |= term
        if t >= steps - tail:
            acc += slots_from_bodies(d.root_link_pos_w - env.scene.env_origins, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_b,
                                     cmd.robot_body_pos_w, cmd.robot_body_quat_w, cmd.tele_idx)
    env.close()
    meas = (acc / tail).cpu().numpy()
    tgt = np.stack([c[2][: S.SLOT_DIM] for c in cs])
    rows = []
    for i, (group, name, g) in enumerate(cs):
        m, c = meas[i], tgt[i]
        r = {"group": group, "name": name, "fell": bool(fell[i])}
        b = S.SLICE["base"]
        r["base_cmd"], r["base_meas"] = [round(float(x), 3) for x in c[b]], [round(float(x), 3) for x in m[b]]
        r["base_rel_err"] = float(np.linalg.norm(m[b] - c[b]) / max(np.linalg.norm(c[b]), 1e-6)) if np.linalg.norm(c[b]) > 0 else None
        r["base_abs_err"] = [round(float(x), 3) for x in np.abs(m[b] - c[b])]
        r["height_err"] = round(float(abs(m[S.SLICE["height"]][0] - c[S.SLICE["height"]][0])), 4)
        gt, gm = c[S.SLICE["torso"]], m[S.SLICE["torso"]] / max(np.linalg.norm(m[S.SLICE["torso"]]), 1e-6)
        r["torso_err_rad"] = round(float(np.arccos(np.clip(np.dot(gt, gm), -1, 1))), 4)
        r["hand_err"] = round(float(0.5 * (np.linalg.norm(m[S.SLICE["hand_L"]] - c[S.SLICE["hand_L"]]) + np.linalg.norm(m[S.SLICE["hand_R"]] - c[S.SLICE["hand_R"]]))), 4)
        rows.append(r)
        key = {"base": f"rel {r['base_rel_err'] if r['base_rel_err'] is None else round(r['base_rel_err'], 3)} meas {r['base_meas']}", "height": f"dh err {r['height_err']:.3f} m",
               "torso": f"torso err {r['torso_err_rad']:.3f} rad", "hands": f"hand err {100 * r['hand_err']:.1f} cm", "mask": ""}[group]
        print(f"{group:6s} {name:24s} {key} {'FELL' if r['fell'] else ''}", flush=True)
    summary = {"start_clip": names[ci], "seconds": seconds}
    for grp in ("base", "height", "torso", "hands", "mask"):
        sel = [r for r in rows if r["group"] == grp]
        ok = [r for r in sel if not r["fell"]]
        s = {"cases": len(sel), "falls": len(sel) - len(ok)}
        if grp == "base":
            rel = [r["base_rel_err"] for r in ok if r["base_rel_err"] is not None]
            s["median_rel_err"] = float(np.median(rel)) if rel else None
            s["mean_abs_err"] = [round(float(np.mean([r["base_abs_err"][k] for r in ok])), 3) for k in range(3)] if ok else None
        elif ok:
            k = {"height": "height_err", "torso": "torso_err_rad", "hands": "hand_err", "mask": "hand_err"}[grp]
            s[f"mean_{k}"] = float(np.mean([r[k] for r in ok]))
            s["mean_height_err"] = float(np.mean([r["height_err"] for r in ok]))
            s["mean_torso_err_rad"] = float(np.mean([r["torso_err_rad"] for r in ok]))
        summary[grp] = s
    print(json.dumps(summary, indent=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
