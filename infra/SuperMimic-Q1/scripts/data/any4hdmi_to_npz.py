"""any4hdmi datasets (manifest.json + motions/**/*.npz with qpos) -> training motion npz.

  SUPERMIMIC_ROBOT=q1 uv run python scripts/data/any4hdmi_to_npz.py \
      --datasets ../dataset_jobs/lafan_walk1_subject1/retarget ... --output-dir data/npz/q1_walk

Each any4hdmi motion becomes ``<output-dir>/<dataset_name>.npz`` (or ``<dataset_name>__<stem>.npz``
when a dataset holds several motions). qpos is resampled to 50 Hz and run through CPU MuJoCo FK
exactly like the CSV path (``motion_convert.convert_qpos_to_npz``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro

from supermimic.data.motion_convert import JOINT_NAMES, convert_qpos_to_npz
from supermimic.robot import load_robot_model


def main(datasets: list[Path], output_dir: Path, output_fps: float = 50.0) -> None:
    model = load_robot_model()
    output_dir.mkdir(parents=True, exist_ok=True)
    for root in datasets:
        manifest = json.loads((root / "manifest.json").read_text())
        names = manifest["qpos_names"]
        if tuple(names[7:]) != JOINT_NAMES:
            raise ValueError(f"{root}: qpos_names differ from the robot joint order")
        fps = 1.0 / float(manifest["timestep"])
        motions = sorted((root / manifest.get("motions_subdir", "motions")).rglob("*.npz"))
        for path in motions:
            qpos = np.load(path)["qpos"]
            stem = manifest["dataset_name"] if len(motions) == 1 else f"{manifest['dataset_name']}__{path.stem}"
            info = convert_qpos_to_npz(qpos, output_dir / f"{stem}.npz", fps, output_fps, model, source=str(path))
            print(f"{stem}: {info['output_frames']} frames, {info['duration_s']:.1f} s")


if __name__ == "__main__":
    tyro.cli(main)
