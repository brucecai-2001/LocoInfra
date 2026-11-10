# SuperMimic-Q1

LocoInfra 的训练、评测与导出部分：[SuperMimic](https://arxiv.org/abs/2610.28480)（可微世界模型 + 短视野 BPTT）的 Q1 版本，包含双 Encoder 基础动作模型。基于 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（Yuxi Liu，MIT 许可，见 `LICENSE`）改写，底座为 [mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）。`SUPERMIMIC_ROBOT=q1|g1` 切换机器人。

文档见仓库根目录：[训练与评测](../../docs/training.md)、[导出与部署](../../docs/deployment.md)。

```
src/supermimic/
  config.py          全部训练配置（tyro 命令行）
  robot.py           机器人规格与维度（由关节数推出）；robots/q1.py 是 Q1 的 mjlab 配置
  data/              any4hdmi qpos → motion.npz、回放池、仿真记录器
  envs/              mjlab 环境：多片段动作命令、按回报采样起点、随机化与仿真上下文 xi、奖励、终止
  kinematics/        可微正运动学
  world_model/       动力学 MLP、终止头、解析奖励 / 观测头、世界模型训练
  imagination/       想象 rollout、策略目标、BPTT 策略训练
  policy/            E_ref、E_tele、Decoder、critic
  tele/              槽位注册表、参考 → 命令标签、自由命令奖励
  training/          主循环、学习率、多卡
  eval/              整条评测、单段动作验收、报告
scripts/
  train/             训练启动脚本（单段动作、基础模型两阶段）
  data/              any4hdmi → motion.npz（含并行与留出集划分）
  eval_*.py          评测；export_*.py 导出权重包；play.py / render_policy.py 回放
tests/               正运动学、解析头、环境、世界模型、想象、数据转换、遥操作的数值与冒烟测试
```

```bash
uv sync --group dev && uv pip install onnx onnxruntime
SUPERMIMIC_ROBOT=q1 env -u PYTHONPATH .venv/bin/pytest tests -q
```
