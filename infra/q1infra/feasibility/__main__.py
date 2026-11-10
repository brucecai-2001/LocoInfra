"""用法（在 infra/ 下）：
    bin/q1py -m q1infra.feasibility jobs/walk1/retarget                  # 平地，写到 jobs/walk1/check/
    bin/q1py -m q1infra.feasibility dataset_jobs/lafan_walk*/retarget    # 批量
    bin/q1py -m q1infra.feasibility traj.npz                             # 含 qpos / fps 的 npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from q1infra.feasibility.check import check_path, summary_line
from q1infra.feasibility.scene import FLAT


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 轨迹可行性检查（只报告，不改轨迹）")
    parser.add_argument("inputs", type=Path, nargs="+", help="any4hdmi 数据集目录，或含 qpos / fps 的 npz")
    args = parser.parse_args()
    for path in args.inputs:
        for name, report in check_path(path, FLAT).items():
            print(summary_line(name, report), flush=True)


if __name__ == "__main__":
    main()
