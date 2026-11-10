#!/bin/bash
# 基础动作模型第一阶段：E_ref + Decoder 从零训练（只跟踪完整参考）。
#   scripts/train/base_ref.sh <run 名> <npz 目录> [sm-train 其他参数]
# 观测口径与第二阶段相同（未来 4 帧参考 + 躯干相对朝向，不给相对位置），第二阶段才能直接续训。
set -e
cd "$(dirname "$0")/../.."
export SUPERMIMIC_ROBOT=q1 PYTHONUNBUFFERED=1
run="$1"; npz="$2"; shift 2
mkdir -p logs
exec .venv/bin/sm-train --run-name "$run" --data.npz-dir "$npz" --env.num-envs 4096 \
  --env.anchor-obs --env.anchor-ori-only --env.future-frames 4 \
  --schedule.total-iterations 50000 --schedule.eval-interval-min 60 --schedule.no-stop-on-pass \
  --schedule.checkpoint-interval 2000 --schedule.numbered-checkpoint-interval 10000 "$@"
