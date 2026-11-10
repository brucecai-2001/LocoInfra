"""遥操作开机自启（SDK 快速入门第 4 节）：把 start_teleop 装到 /robot/software/kabutack_teleop/bin/，
并在运控板 process_manager 配置的 apps 段与第二启动阶段登记 kabutack_teleop。

在运控板上跑（重启后生效；配置首次改动前备份为 *.kabutack_bak，我们加的每行都带 "# kabutack_teleop" 标记）：
    python3 ~/LocoInfra/apps/teleop/autostart/install.py                                     # 安装 / 更新
    python3 ~/LocoInfra/apps/teleop/autostart/install.py --speed-scale 0.5 --pad-lost liedown  # 其余参数原样传给 apps.teleop
    python3 ~/LocoInfra/apps/teleop/autostart/install.py --uninstall

process_manager 只在开机时读配置：装完要重启一次，之后 yamo em list-apps 里才有它，
才能 yamo em stop-app / start-app kabutack_teleop 启停（stop 发 SIGTERM，遥操作切阻尼后退出）。
不重启想先试：KABUTACK_HOME=~/LocoInfra bash /robot/software/kabutack_teleop/bin/start_teleop（日志在 /tmp/kabutack_teleop/）。
日志：/robot/persist/log/kabutack_teleop/teleop_<时间>.log。
标准模式（厂商运控 mc 在跑）下照样会被拉起，但 apps.teleop 检测到 mc_app_main 就播报“高层运控正在运行”后退出。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

NAME = "kabutack_teleop"
MARK = f"  # {NAME}"
DIR = Path("/robot/software") / NAME
CFG = Path("/robot/software/process_manager/bin/cfg/q1_v2d0_soc0_config.yaml")
LOG_PATH = f"/robot/persist/log/{NAME}"
PROJECT = Path(__file__).resolve().parents[3]


def strip(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not ln.rstrip("\n").endswith(MARK)]


def add(lines: list[str], teleop_args: str) -> list[str]:
    lines = strip(lines)
    i_apps = next(i for i, ln in enumerate(lines) if ln.rstrip() == "  apps:")
    block = [f'    "{NAME}":', f'      path: "{DIR}/bin/start_teleop"', "      sudo: false", f'      stderr: "/tmp/{NAME}.err"',
             "      env:", f"        LOG_PATH: {LOG_PATH}", f"        KABUTACK_HOME: {json.dumps(str(PROJECT))}", f"        TELEOP_ARGS: {json.dumps(teleop_args)}"]
    lines[i_apps + 1:i_apps + 1] = [b + MARK + "\n" for b in block]
    i_stages = next(i for i, ln in enumerate(lines) if ln.strip() == "startup_stages:")
    i_abox = next(i for i in range(i_stages, len(lines)) if lines[i].strip().startswith('"abox"'))  # SDK：加在第二阶段末尾
    indent = lines[i_abox][:len(lines[i_abox]) - len(lines[i_abox].lstrip())]
    lines.insert(i_abox + 1, f'{indent}"{NAME}",{MARK}\n')
    return lines


def check(text: str, installed: bool) -> None:
    pm = yaml.safe_load(text)["process_manager"]
    in_apps = NAME in pm["apps"]
    stages = [i for i, s in enumerate(pm["startup_stages"]) if NAME in s["apps"]]
    if installed and not (in_apps and stages == [1]):
        raise RuntimeError(f"改完的配置不对：apps 里有 {in_apps}，所在启动阶段 {stages}")
    if not installed and (in_apps or stages):
        raise RuntimeError("卸载后配置里还有 kabutack_teleop")


def main() -> None:
    p = argparse.ArgumentParser(description="遥操作开机自启：安装 / 卸载")
    p.add_argument("--uninstall", action="store_true")
    a, teleop_args = p.parse_known_args()

    old = CFG.read_text()
    lines = old.splitlines(keepends=True)
    new = "".join(strip(lines) if a.uninstall else add(lines, " ".join(teleop_args) or "--speed-scale 0.5"))
    check(new, installed=not a.uninstall)
    bak = CFG.with_name(CFG.name + ".kabutack_bak")
    if not bak.exists():
        shutil.copy2(CFG, bak)
    tmp = CFG.with_name(CFG.name + ".tmp")
    tmp.write_text(new)
    os.replace(tmp, CFG)

    if a.uninstall:
        shutil.rmtree(DIR, ignore_errors=True)
        print(f"已从 {CFG} 移除 {NAME}，删掉 {DIR}；重启后不再自启（备份 {bak}）")
        return
    (DIR / "bin").mkdir(parents=True, exist_ok=True)
    dst = DIR / "bin" / "start_teleop"
    shutil.copy2(Path(__file__).with_name("start_teleop"), dst)
    dst.chmod(0o755)
    print(f"已装 {dst}，并在 {CFG} 登记 {NAME}（第二启动阶段，参数 {' '.join(teleop_args) or '--speed-scale 0.5'}）；备份 {bak}")
    print(f"重启后生效（process_manager 只在开机时读配置），之后可用 yamo em stop-app / start-app {NAME}")


if __name__ == "__main__":
    sys.exit(main())
