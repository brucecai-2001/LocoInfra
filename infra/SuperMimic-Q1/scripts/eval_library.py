"""Whole-clip deterministic playback of a motion library (e.g. the held-out set) through E_ref and
through E_tele under one or more slot masks (paper Appendix F protocol; docs/training.md).
E_tele reads the slots derived from the reference (tele/labels.py); hand error is relative to the head.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/eval_library.py --ckpt logs/<run>/ckpt_latest.pt \
      --npz-dir data/npz/mix_v1_holdout --out logs/<run>/eval/holdout.json \
      --masks all base,height,torso hand_L,hand_R,base       # slot sets of E_tele ("all" = every non-reserved slot)

The summary also reports E_tele separately on clips that are entirely tele_ok and on the rest.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import BenchmarkRunner, load_policy_from_checkpoint, summarize
from supermimic.tele import slots as S


def mask_of(spec: str) -> torch.Tensor:
    if spec == "all":
        return (~S.reserved_mask()).float()
    names = spec.split(",")
    bad = [n for n in names if n not in S.SLOT_NAMES]
    assert not bad, f"unknown slots {bad}; registry: {S.SLOT_NAMES}"
    return torch.tensor([float(n in names) for n in S.SLOT_NAMES])


def main(ckpt: Path, npz_dir: Path, out: Path, masks: tuple[str, ...] = ("all",), device: str = "cuda:0", skip_ref: bool = False) -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    e = cfg["env"]
    files = [str(f) for f in discover_npz(npz_dir)]
    runner = BenchmarkRunner(files, device, future_frames=e["future_frames"], anchor_obs=e["anchor_obs"], anchor_ori_only=e["anchor_ori_only"])
    report = {"ckpt": str(ckpt), "clips": len(files)}
    runs = ([] if skip_ref else [("ref", None)]) + ([(f"tele[{m}]", mask_of(m)) for m in masks] if "tele" in policy.encoders else [])
    lib = runner.env.command_manager.get_term("motion").library
    for name, m in runs:
        res = runner.run(policy, interface="ref" if m is None else "tele", slot_mask=m)
        s = summarize(res)
        s.pop("terms", None)
        s["fall_rate"] = s["failed"] / max(s["clips"], 1)
        if m is not None:
            ok_clip = [bool(lib.tele_ok[c.start : c.end].all()) for c in lib.clips]
            for tag, sel in (("tele_ok_clips", True), ("other_clips", False)):
                sub = [r for r, o in zip(res, ok_clip) if o == sel]
                if sub:
                    ss = summarize(sub)
                    s[tag] = {"clips": ss["clips"], "fall_rate": ss["failed"] / ss["clips"], "mean_norm_return": ss["mean_norm_return"], "hand_err_m": ss.get("hand_err_m")}
        report[name] = {"summary": s, "clips": [asdict(r) | {"terms": None} for r in res]}
        hand = f", hand err {100 * s['hand_err_m']:.1f} cm" if s.get("hand_err_m") is not None else ""
        extra = f" | tele_ok clips {s['tele_ok_clips']['mean_norm_return']:.3f} / {100 * s['tele_ok_clips']['fall_rate']:.1f}%" if "tele_ok_clips" in s else ""
        print(f"{name}: return {s['mean_norm_return']:.3f}, falls {s['failed']}/{s['clips']} ({100 * s['fall_rate']:.1f}%){hand}{extra}", flush=True)
    runner.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
