"""Teleoperation label audit of a motion library (docs/training.md): tele_ok share and rejection
reasons, hours per command category (tele_ok bins), slot percentiles, and the slots of the default
stance (the registry defaults).

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/tele_coverage.py --npz-dir data/npz/mix_v1 --out logs/tele_coverage_mix_v1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
from supermimic.robot import NJ, SPEC, load_robot_model
from supermimic.tele import labels as L
from supermimic.tele import slots as S


def default_stance_slots(device: str) -> dict[str, list[float]]:
    """Slots of the robot standing in its default stance (upright pelvis at H_STAND)."""
    import re

    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.q1 import STAND_KEYFRAME

    model = load_robot_model()
    fk = G1Kinematics(model, device=device)
    q = torch.zeros(NJ, device=device)
    for pat, v in STAND_KEYFRAME.joint_pos.items():
        for j, n in enumerate(SPEC.joint_names):
            if re.fullmatch(pat, n):
                q[j] = v
    qpos = torch.cat([torch.tensor([0.0, 0.0, S.H_STAND, 1.0, 0.0, 0.0, 0.0], device=device), q])[None]
    out = fk(qpos, torch.zeros(1, 6 + NJ, device=device))
    idx = lambda n: fk.body_index(n)  # noqa: E731
    p = S.pose_slots(
        qpos[:, :3], out["pos"][:, idx("torso_link")], out["quat"][:, idx("torso_link")],
        out["pos"][:, [idx(b) for b in S.HAND_BODIES]], out["pos"][:, [idx(b) for b in S.FOOT_BODIES]],
    )
    return {k: [round(float(x), 4) for x in v[0]] for k, v in p.items()}


@torch.no_grad()
def main(npz_dir: Path, out: Path, device: str = "cuda:0") -> None:
    files = discover_npz(npz_dir)
    lib = MotionLibrary(files, TRACKED_BODIES, torch.zeros(NJ), device=device)
    print(lib.summary())
    lab = L.compute_labels(lib)
    F = lib.num_frames
    ok = lab["tele_ok"]
    report: dict = {"npz_dir": str(npz_dir), "clips": lib.num_clips, "hours": F / 50 / 3600}
    report["tele_ok_frames"] = float(ok.float().mean())
    report["tele_ok_bins"] = float(lab["tele_ok_bin"].float().mean())
    report["reject_reason_frames"] = {k: float(v.float().mean()) for k, v in lab["reasons"].items()}
    cat, okb = lab["category"], lab["tele_ok_bin"]
    bin_h = (lib.bin_end - lib.bin_start).float() / 50 / 3600
    rows = []
    for c in range(1 << len(L.CATEGORY_BITS)):
        sel = cat == c
        if sel.any():
            rows.append({"category": L.category_name(c), "hours": round(float(bin_h[sel].sum()), 3), "tele_ok_hours": round(float(bin_h[sel & okb].sum()), 3)})
    report["categories"] = sorted(rows, key=lambda r: -r["tele_ok_hours"])
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=device)
    x = lab["slots"][ok]
    sub = x[torch.randint(0, x.shape[0], (min(x.shape[0], 2_000_000),), device=device)]
    pct = {}
    for s in S.SLOTS:
        sl = S.SLICE[s.name]
        pct[s.name] = [[round(float(v), 4) for v in torch.quantile(sub[:, j], qs)] for j in range(sl.start, sl.stop)]
    report["slot_percentiles_1_5_50_95_99"] = pct
    report["default_stance_slots"] = default_stance_slots(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "slot_percentiles_1_5_50_95_99"}, indent=1, ensure_ascii=False))
    for k, v in pct.items():
        print(k, v)


if __name__ == "__main__":
    tyro.cli(main)
