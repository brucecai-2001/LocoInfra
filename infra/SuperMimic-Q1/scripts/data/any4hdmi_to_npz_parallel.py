"""any4hdmi datasets -> training motion npz, in parallel (large libraries; same output as any4hdmi_to_npz.py).

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/data/any4hdmi_to_npz_parallel.py \
      --datasets ../datasets/amass_v1 ../datasets/base_v2 --output-dir data/npz/mix_v1 --workers 20 --holdout 0.03

``--holdout`` moves that fraction of the clips (random, fixed seed) into ``<output-dir>_holdout`` for the
held-out evaluation.
"""

from __future__ import annotations

import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tyro


def _convert(args: tuple[str, str, float, float]) -> tuple[str, int]:
    from supermimic.data.motion_convert import convert_qpos_to_npz
    from supermimic.robot import load_robot_model

    src, out, fps, out_fps = args
    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = load_robot_model()
    qpos = np.load(src)["qpos"]
    info = convert_qpos_to_npz(qpos, Path(out), fps, out_fps, _MODEL, source=src)
    return out, int(info["output_frames"])


def main(datasets: list[Path], output_dir: Path, output_fps: float = 50.0, workers: int = 20, holdout: float = 0.0, seed: int = 0) -> None:
    from supermimic.data.motion_convert import JOINT_NAMES

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for root in datasets:
        manifest = json.loads((root / "manifest.json").read_text())
        if tuple(manifest["qpos_names"][7:]) != JOINT_NAMES:
            raise ValueError(f"{root}: qpos_names differ from the robot joint order")
        fps = 1.0 / float(manifest["timestep"])
        motions = sorted((root / manifest.get("motions_subdir", "motions")).rglob("*.npz"))
        for p in motions:
            stem = manifest["dataset_name"] if len(motions) == 1 else f"{manifest['dataset_name']}__{p.stem}"
            out = output_dir / f"{stem}.npz"
            if not out.exists():
                tasks.append((str(p), str(out), fps, output_fps))
    print(f"{len(tasks)} motions to convert", flush=True)
    frames = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (_, n) in enumerate(ex.map(_convert, tasks, chunksize=16), 1):
            frames += n
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {frames / 50 / 3600:.2f} h", flush=True)
    if holdout > 0:
        files = sorted(output_dir.glob("*.npz"))
        rng = random.Random(seed)
        held = rng.sample(files, int(round(holdout * len(files))))
        hdir = output_dir.parent / f"{output_dir.name}_holdout"
        hdir.mkdir(exist_ok=True)
        for f in held:
            shutil.move(str(f), hdir / f.name)
        print(f"held out {len(held)} clips -> {hdir}")
    print("train clips:", len(list(output_dir.glob("*.npz"))))


if __name__ == "__main__":
    tyro.cli(main)
