#!/bin/bash
# 基础动作模型第二阶段：加入 E_tele（蒸馏 + 自由命令，Decoder 冻结），E_ref + Decoder 继续训练。
#   scripts/train/base_dual.sh <run 名> <npz 目录> <第一阶段检查点> [sm-train 其他参数]
set -e
cd "$(dirname "$0")/../.."
export SUPERMIMIC_ROBOT=q1 PYTHONUNBUFFERED=1
run="$1"; npz="$2"; ckpt="$3"; shift 3
mkdir -p logs
exec .venv/bin/sm-train --run-name "$run" --data.npz-dir "$npz" --resume "$ckpt" --env.num-envs 4096 \
  --env.anchor-obs --env.anchor-ori-only --env.future-frames 4 \
  --tele.enabled --tele.var-lin 0.05 --tele.var-yaw 0.25 --tele.base-only-prob 0.25 \
  --wm-trainer.lr 1e-4 --policy-trainer.lr 1e-4 --policy-trainer.critic-lr 1e-4 --schedule.lr-from-resume \
  --schedule.total-iterations 145000 --schedule.max-hours 6.5 --schedule.eval-interval-min 60 --schedule.no-stop-on-pass \
  --schedule.checkpoint-interval 2000 --schedule.numbered-checkpoint-interval 12000 "$@"
