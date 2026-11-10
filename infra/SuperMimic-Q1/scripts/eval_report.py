"""Acceptance evaluation of a checkpoint -> report.json + report.html (+ replay videos).

  MUJOCO_GL=egl SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/eval_report.py \
      --ckpt logs/<run>/ckpt_latest.pt --npz-dir data/npz/<set> --out-dir logs/<run>/eval \
      --feasibility ../jobs/<job>/check/optimize_report.json

``--feasibility`` takes the job's ``check/report.json`` or ``check/optimize_report.json`` (the
``after`` section is used) for the time-above-rated-torque estimate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.eval.report import render_video, write_html
from supermimic.eval.suite import SuiteCfg, run_suite


def feasibility_over_rated(path: Path) -> dict:
    r = json.loads(path.read_text())
    if "after" in r:
        r = r["after"]
    return r["torque"]["over_rated_s"]


def main(
    ckpt: Path,
    npz_dir: Path,
    out_dir: Path | None = None,
    clips: list[str] | None = None,
    feasibility: Path | None = None,
    dr_episodes: int = 1000,
    push_per_speed: int = 64,
    video: bool = True,
    device: str = "cuda:0",
) -> None:
    import numpy as np
    import torch

    out_dir = out_dir or ckpt.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    feas = feasibility_over_rated(feasibility) if feasibility is not None else None
    t0 = time.time()
    res = run_suite(policy, cfg, files, device, SuiteCfg(dr_episodes=dr_episodes, push_per_speed=push_per_speed), feas)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = {"checkpoint": ckpt.name, "iteration": ck.get("iteration"), "train_hours": round(float(ck.get("train_hours", float("nan"))), 2), "eval_minutes": round((time.time() - t0) / 60, 1)}

    videos = {}
    for c in res["deterministic"]["clips"]:
        qpos, ref = c.pop("_qpos"), c.pop("_ref_bodies")
        np.savez_compressed(out_dir / f"trace_{c['name']}.npz", qpos=qpos, ref_bodies=ref, joint_err=np.asarray(c["traces"]["joint_err_frames"]), torque=np.asarray(c["traces"]["torque"]))
        if video:
            name = f"play_{c['name']}.mp4"
            render_video(qpos, ref, out_dir / name)
            videos[c["name"]] = name
    write_html(res, out_dir / "report.html", f"评测报告：{', '.join(c['name'] for c in res['deterministic']['clips'])}", meta, videos)
    for c in res["deterministic"]["clips"]:
        for k in ("joint_err_frames", "torque"):
            c["traces"].pop(k)
    (out_dir / "report.json").write_text(json.dumps({"meta": meta, **res}, indent=1, ensure_ascii=False))
    print("\n".join(f"  [{'PASS' if c['passed'] else ('FAIL' if c['passed'] is False else ' -- ')}] {c['name']}: {c['value']} (验收线 {c['threshold']})" for c in res["checks"]))
    print(f"overall: {'PASS' if res['passed'] else 'FAIL'}  ->  {out_dir / 'report.html'}")


if __name__ == "__main__":
    tyro.cli(main)
