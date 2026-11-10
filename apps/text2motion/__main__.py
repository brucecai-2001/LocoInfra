"""文本生成动作网页：输入一句动作描述，GPU 上的 HY-Motion 1.0 生成人体动作，重定向到 Q1、可行性检查与优化后，
基础模型 E_ref 直接跟踪并回放（三栏：生成的人体 | Q1 参考 | 物理仿真）。

HY-Motion 只能在 GPU 上跑（约 26 GB 显存），作为常驻服务部署在 GPU 机器上；本网页和其余步骤（重定向、优化、跟踪）
只用 CPU，可以和服务在同一台机器，也可以在另一台机器上通过 --t2m-url 找服务。

GPU 机器（HY-Motion 需要使用者自行部署，见 docs/apps.md 与 infra/q1infra/text2motion/setup_gpu.sh）：
    cd $HY_ROOT && $HY_ROOT/.venv/bin/python <LocoInfra>/infra/q1infra/text2motion/hymotion_server.py
网页（服务在别的机器时先转发端口：ssh -N -L 8770:127.0.0.1:8770 <GPU 机器> &）：
    infra/bin/q1py -m apps.text2motion --package <权重包目录>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from apps.text2motion.server import TextApp  # noqa: E402
from apps.track_web.server import detect_device, serve  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="文本生成动作网页")
    p.add_argument("--package", type=Path, required=True, help="基础模型权重包目录（export_base_package.py 导出）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址；局域网 / 远程访问用 0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--jobs-dir", type=Path, default=PROJECT / "infra" / "jobs")
    p.add_argument("--t2m-url", default=os.environ.get("HYMOTION_URL", "http://127.0.0.1:8770"), help="HY-Motion 生成服务地址")
    a = p.parse_args()

    if not (a.package / "deploy.yaml").exists():
        sys.exit(f"{a.package} 不是权重包（缺 deploy.yaml）")
    serve(TextApp(a.package, a.jobs_dir, a.t2m_url, detect_device()), a.host, a.port)


if __name__ == "__main__":
    main()
