"""单目视频 → SMPL-X（GVHMR）→ Q1 重定向（同 --smplx）。用法（在 infra/ 下）：

    bin/q1py -m q1infra.video <视频> --job <任务名> [--static-cam] [--f-mm 24] [--start-sec 1 --end-sec 9] [--render]

GVHMR 需要使用者自行安装（见 docs/apps.md）。默认在本机跑：GVHMR 目录取 --gvhmr-root、$GVHMR_ROOT、
infra/.deps/GVHMR，其 Python 取 --gvhmr-python、$GVHMR_PYTHON、<GVHMR>/.venv/bin/python。
GVHMR 装在另一台 GPU 机器上时，给 --host（ssh 主机名）与 --remote-gvhmr-root、--remote-gvhmr-python，
视频和推理脚本经 rsync 传过去，结果拉回本机。

写入 jobs/<任务名>/video/（GVHMR 结果、<名字>_smplx.npz、video_report.json），之后与 SMPL-X 输入完全相同：
input/、retarget/、retarget_report.json。质量门槛（全身可见帧 ≥ 90%、2–60 s）不过时停在重定向前，--force 跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from q1infra.retarget.gmr_q1 import INFRA_ROOT, SMPLX_MODEL_DIR, _sha256, retarget_smplx
from q1infra.video import convert

INFER = Path(__file__).with_name("gvhmr_infer.py")
FETCH = ("hmr4d_results.pt", "bbx.pt", "vitpose.pt", "meta.json", "0_input_video.mp4")


def infer_args(static_cam: bool, f_mm: int | None, device: str) -> list[str]:
    return (["--static-cam"] if static_cam else []) + (["--f-mm", str(f_mm)] if f_mm else []) + ["--device", device]


def run_local(video: Path, out: Path, root: str | None, python: str | None, args: list[str]) -> None:
    root = Path(root or os.environ.get("GVHMR_ROOT") or INFRA_ROOT / ".deps" / "GVHMR").resolve()
    python = python or os.environ.get("GVHMR_PYTHON") or str(root / ".venv" / "bin" / "python")
    if not (root / "tools" / "demo" / "demo.py").exists() or not Path(python).exists():
        sys.exit(f"[video] 没找到 GVHMR（{root}）或其 Python（{python}）：按 docs/apps.md 安装，或用 --gvhmr-root / --gvhmr-python 指定")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # GVHMR 用它自己的环境
    subprocess.run([python, str(INFER), str(video), str(out), *args], cwd=root, env=env, check=True)


def run_remote(video: Path, job: str, out: Path, host: str, root: str, python: str, args: list[str]) -> None:
    rdir = f"{root}/outputs/locoinfra/{job}"
    rvid = f"{root}/outputs/locoinfra/_src/{job}{video.suffix}"
    sh = lambda cmd: subprocess.run(["ssh", host, cmd], check=True)  # noqa: E731
    sh(f"mkdir -p {root}/outputs/locoinfra/_src")
    subprocess.run(["rsync", "-az", str(video), f"{host}:{rvid}"], check=True)
    subprocess.run(["rsync", "-az", str(INFER), f"{host}:{root}/locoinfra_gvhmr_infer.py"], check=True)
    sh(f"cd {root} && {python} locoinfra_gvhmr_infer.py {rvid} {rdir} {' '.join(args)}")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", *[f"--include={f}" for f in FETCH], "--exclude=*", f"{host}:{rdir}/", str(out) + "/"], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="单目视频 → GVHMR SMPL-X → Q1（GMR），输出 any4hdmi")
    p.add_argument("video", type=Path)
    p.add_argument("--job", required=True, help="任务名，输出到 infra/jobs/<job>/")
    p.add_argument("--static-cam", action="store_true", help="固定机位（跳过视觉里程计）")
    p.add_argument("--f-mm", type=int, default=None, help="等效全画幅焦距 mm（iPhone 主摄 24）；不给用 GVHMR 默认估计")
    p.add_argument("--start-sec", type=float, default=None)
    p.add_argument("--end-sec", type=float, default=None)
    p.add_argument("--render", action="store_true", help="离屏渲染重定向结果到 jobs/<job>/retarget_preview.mp4")
    p.add_argument("--force", action="store_true", help="质量门槛不过也继续重定向")
    p.add_argument("--gvhmr-root", default=None, help="本机 GVHMR 仓库根目录（含 tools/demo 与 inputs/checkpoints）")
    p.add_argument("--gvhmr-python", default=None, help="本机装好 GVHMR 依赖的 Python")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"), help="GVHMR 推理设备")
    p.add_argument("--host", default=None, help="在这台 ssh 主机上跑 GVHMR（不给则在本机跑）")
    p.add_argument("--remote-gvhmr-root", default=None, help="远程 GVHMR 仓库根目录")
    p.add_argument("--remote-gvhmr-python", default=None, help="远程装好 GVHMR 依赖的 Python")
    a = p.parse_args()

    video = a.video.resolve()
    job_dir = INFRA_ROOT / "jobs" / a.job
    vdir = job_dir / "video"
    if (vdir / "hmr4d_results.pt").exists():
        print(f"[video] 复用已有 GVHMR 结果 {vdir}")
    elif a.host:
        if not (a.remote_gvhmr_root and a.remote_gvhmr_python):
            sys.exit("[video] 远程运行需要 --remote-gvhmr-root 与 --remote-gvhmr-python")
        print(f"[video] GVHMR（{a.host}）…")
        run_remote(video, a.job, vdir, a.host, a.remote_gvhmr_root, a.remote_gvhmr_python, infer_args(a.static_cam, a.f_mm, a.device))
    else:
        print("[video] GVHMR（本机）…")
        run_local(video, vdir, a.gvhmr_root, a.gvhmr_python, infer_args(a.static_cam, a.f_mm, a.device))

    import torch

    vis = convert.visibility(torch.load(vdir / "vitpose.pt", map_location="cpu").numpy())
    g = convert.gate(vis, a.start_sec, a.end_sec)
    npz = vdir / f"{video.stem}_smplx.npz"
    conv = convert.gvhmr_to_amass(vdir / "hmr4d_results.pt", npz, SMPLX_MODEL_DIR)
    meta = json.loads((vdir / "meta.json").read_text())
    report = {
        "video": video.name, "video_sha256": _sha256(video), "gvhmr": meta, "static_cam": a.static_cam, "f_mm": a.f_mm,
        "gate": g, "smplx": conv,
    }
    (vdir / "video_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[video] 全身可见 {g['visible_frac']:.1%}（最长连续不可见 {g['longest_invisible_s']:.1f} s），时长 {g['duration_s']:.1f} s："
          + ("通过" if g["passed"] else "未通过门槛"))
    if not g["passed"] and not a.force:
        sys.exit("[video] 质量门槛未通过，停在重定向前（--force 继续）")

    source_extra = {
        "input_type": "video_gvhmr",
        "video": {"file": video.name, "sha256": report["video_sha256"], "static_cam": a.static_cam, "f_mm": a.f_mm, "visible_frac": g["visible_frac"]},
    }
    rep = retarget_smplx(npz, job_dir, a.start_sec, a.end_sec, SMPLX_MODEL_DIR, source_extra=source_extra)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.render:
        from q1infra.retarget.render import render_dataset

        stills = render_dataset(job_dir / "retarget", job_dir / "retarget_preview.mp4", keypoints_npz=job_dir / "retarget_debug.npz",
                                stills=[rep["duration_s"] * f for f in (0.1, 0.4, 0.7)])
        print("preview:", job_dir / "retarget_preview.mp4", *stills, sep="\n  ")


if __name__ == "__main__":
    main()
