#!/bin/bash
# 单段动作跟踪策略（验证工具链、做技能）：未来 10 帧参考 + 躯干相对朝向，每 15 分钟整条评测，达标即停，最长 3 小时。
#   scripts/train/single.sh <run 名> <npz 目录> [<片段名> ...] [-- sm-train 其他参数]
set -e
cd "$(dirname "$0")/../.."
export SUPERMIMIC_ROBOT=q1 PYTHONUNBUFFERED=1
run="$1"; npz="$2"; shift 2
clips=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do clips+=("$1"); shift; done
[ "$1" = "--" ] && shift
mkdir -p logs
exec .venv/bin/sm-train --run-name "$run" --data.npz-dir "$npz" ${clips[@]:+--data.clips "${clips[@]}"} \
  --env.num-envs 4096 --env.anchor-obs --env.anchor-ori-only --env.future-frames 10 \
  --schedule.max-hours 3 --schedule.eval-interval-min 15 "$@"
