"""用法（在 infra/ 下）：
    bin/q1py -m q1infra.retarget --bvh data/lafan1/walk1_subject1.bvh --job walk1_s1 --end-sec 30 --render
    bin/q1py -m q1infra.retarget --smplx <AMASS SMPL-X .npz> --job <任务名> --render
             [--smplx-model-dir data/body_models/smplx]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from q1infra.retarget.gmr_q1 import INFRA_ROOT, SMPLX_MODEL_DIR, retarget_bvh, retarget_smplx
from q1infra.retarget.render import render_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="人体动作（LAFAN1 BVH / SMPL-X）→ Q1（GMR），输出 any4hdmi")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bvh", type=Path, help="LAFAN1 BVH")
    src.add_argument("--smplx", type=Path, help="SMPL-X 动作（AMASS npz：root_orient / pose_body / trans / betas）")
    parser.add_argument("--smplx-model-dir", type=Path, default=SMPLX_MODEL_DIR, help="SMPLX_{NEUTRAL,MALE,FEMALE}.npz|pkl 所在目录")
    parser.add_argument("--smplx-fps", type=float, default=None, help="SMPL-X 文件没有 mocap_frame_rate 时指定源帧率")
    parser.add_argument("--job", required=True, help="任务名，输出到 infra/jobs/<job>/")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--render", action="store_true", help="离屏渲染 mp4 到 jobs/<job>/retarget_preview.mp4")
    parser.add_argument("--flat-feet", action="store_true", help="着地脚朝向放平（深蹲、仆步等 Q1 踝关节行程不够的动作）")
    parser.add_argument("--min-pelvis-height", type=float, default=None, help="骨盆比较低那只脚的脚底至少高多少 m（站立约 0.42）")
    args = parser.parse_args()

    job_dir = INFRA_ROOT / "jobs" / args.job
    kw = {"flat_feet": args.flat_feet, "min_pelvis_height": args.min_pelvis_height}
    if args.bvh is not None:
        report = retarget_bvh(args.bvh, job_dir, args.start_sec, args.end_sec, **kw)
    else:
        report = retarget_smplx(args.smplx, job_dir, args.start_sec, args.end_sec, args.smplx_model_dir, args.smplx_fps, **kw)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.render:
        duration = report["duration_s"]
        stills = render_dataset(
            job_dir / "retarget",
            job_dir / "retarget_preview.mp4",
            keypoints_npz=job_dir / "retarget_debug.npz",
            stills=[duration * f for f in (0.1, 0.4, 0.7)],
        )
        print("preview:", job_dir / "retarget_preview.mp4", *stills, sep="\n  ")


if __name__ == "__main__":
    main()
