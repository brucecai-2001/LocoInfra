#!/usr/bin/env bash
# 搭建数据处理 / 部署环境：按固定 commit 克隆外部依赖到 infra/.deps/，建 infra/.venv（Python 3.11）。
#   bash infra/setup.sh
# 训练环境另建：cd infra/SuperMimic-Q1 && uv sync（依赖这里克隆的 infra/.deps/mjlab）。
set -euo pipefail
INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$INFRA/.deps"

clone() {  # 名字 地址 commit
  local dir="$INFRA/.deps/$1"
  [ -d "$dir/.git" ] || git clone "$2" "$dir"
  git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$3"
  echo "$1 @ $(git -C "$dir" rev-parse --short HEAD)"
}
clone GMR https://github.com/YanjieZe/GMR.git 25f56db79ac411258b377b6d4ccba0c8ff3372ac
clone any4hdmi https://github.com/EGalahad/any4hdmi.git bf83602f36ff6fc9796367cfc2b707a2526e1a54
clone mjlab https://github.com/mujocolab/mjlab.git 8bb2bc9e187797689ec82c78c35019afd2fd0376

command -v uv >/dev/null || { echo "需要 uv：https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d "$INFRA/.venv" ] || uv venv --python 3.11 "$INFRA/.venv"
uv pip install --python "$INFRA/.venv/bin/python" -e "$INFRA/.deps/GMR" -e "$INFRA/.deps/any4hdmi" -r "$INFRA/requirements.txt"
echo "完成。之后用 infra/bin/q1py 运行数据处理与部署脚本。"
