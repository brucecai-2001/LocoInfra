#!/bin/bash
# 俯卧起身 + 站立趴下技能策略：跟踪 apps/ref/skills/ 里随仓库发布的 getup_facedown / liedown_facedown，两段共用一个策略。
#   [Q1_PD=vendor_getup] scripts/train/getup.sh <run 名> [sm-train 其他参数]
# Q1_PD 是 PD 增益预设（src/supermimic/robots/q1.py）：default 髋膝 kp 80；vendor_getup 为低增益（髋膝 30，weights/getup_vpd_ft 用它）。
# 检查点会记录预设，之后评测、导出也要带同一个 Q1_PD，否则直接报错。
# 只训一段：追加 --data.clips getup_facedown（后给的参数覆盖前面的）。
# 第二轮建议加多余接触与撞击惩罚续训（不加时策略会用膝盖砸地借力）：
#   scripts/train/getup.sh getup_v2 --resume logs/getup_v1/ckpt_latest.pt --reward.w-extra-contact -2 --reward.w-impact -5 --schedule.max-hours 1.5
set -e
cd "$(dirname "$0")/../.."
export SUPERMIMIC_ROBOT=q1 PYTHONUNBUFFERED=1 Q1_PD="${Q1_PD:-default}"
run="$1"; shift
mkdir -p logs
exec .venv/bin/sm-train --run-name "$run" --data.npz-dir ../../apps/ref/skills --data.clips getup_facedown liedown_facedown \
  --env.num-envs 4096 --env.anchor-obs --env.anchor-ori-only --env.future-frames 10 \
  --schedule.max-hours 3 --schedule.eval-interval-min 15 "$@"
