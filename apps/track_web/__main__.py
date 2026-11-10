"""动作跟踪网页：上传 10 s 以内的动作（视频 / BVH / SMPL-X npz），基础模型 E_ref 直接跟踪并回放。

整条流程都在本机跑：视频提取用本机 GVHMR（CUDA，或 macOS 的 MPS），重定向、优化、跟踪用 CPU。

    infra/bin/q1py -m apps.track_web --package <权重包目录> [--host 0.0.0.0]

GVHMR 目录按 --gvhmr-root、$GVHMR_ROOT、infra/.deps/GVHMR 依次查找；其 Python 取 --gvhmr-python、
$GVHMR_PYTHON、<GVHMR>/.venv/bin/python。找不到时只接受动作文件（BVH / SMPL-X npz / Q1 qpos / motion.npz）。
GVHMR 需要使用者自行安装，见 docs/apps.md。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from apps.track_web.server import App, detect_device, serve  # noqa: E402

GVHMR_ROOTS = (PROJECT / "infra" / ".deps" / "GVHMR",)
GVHMR_PYTHONS = (".venv/bin/python",)


def find_gvhmr(root: str | None, python: str | None) -> tuple[Path, str] | None:
    roots = [Path(root)] if root else [Path(p) for p in [os.environ.get("GVHMR_ROOT")] if p] + list(GVHMR_ROOTS)
    r = next((p.resolve() for p in roots if (p / "tools" / "demo" / "demo.py").exists()), None)
    if r is None:
        return None
    pys = [python] if python else [p for p in [os.environ.get("GVHMR_PYTHON")] if p] + [str(r / p) for p in GVHMR_PYTHONS]
    py = next((p for p in pys if Path(p).exists()), None)
    return (r, py) if py else None


def main() -> None:
    p = argparse.ArgumentParser(description="动作跟踪网页")
    p.add_argument("--package", type=Path, required=True, help="基础模型权重包目录（export_base_package.py 导出）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址；局域网 / 远程访问用 0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--jobs-dir", type=Path, default=PROJECT / "infra" / "jobs")
    p.add_argument("--gvhmr-root", default=None, help="GVHMR 仓库根目录（含 tools/demo 与 inputs/checkpoints）")
    p.add_argument("--gvhmr-python", default=None, help="装好 GVHMR 依赖的 Python")
    p.add_argument("--gvhmr-device", default="auto", choices=("auto", "cuda", "mps", "cpu"), help="auto：CUDA → MPS → CPU")
    a = p.parse_args()

    if not (a.package / "deploy.yaml").exists():
        sys.exit(f"{a.package} 不是权重包（缺 deploy.yaml）")
    device = detect_device()
    g = find_gvhmr(a.gvhmr_root, a.gvhmr_python)
    if g is None:
        print("[track_web] 没找到 GVHMR（或其 Python），视频上传不可用，只接受动作文件", flush=True)
        extract = {"backend": "off"}
    else:
        gpu = a.gvhmr_device
        if gpu == "auto":
            gpu = "cuda" if device["cuda"] else ("mps" if device["mps"] else "cpu")
        extract = {"backend": "local", "gvhmr_root": str(g[0]), "gvhmr_python": g[1], "device": gpu}
    serve(App(a.package, a.jobs_dir, extract, device), a.host, a.port)


if __name__ == "__main__":
    main()
