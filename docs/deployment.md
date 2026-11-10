# 导出与部署

> 一句话：把训练好的检查点导出成一个自包含的"权重包"（ONNX 网络 + 配置 + 仿真场景），再用一套与训练代码完全独立的轻量运行时，在仿真或机器人上以 50 Hz 推理。
>
> 导出脚本在 `infra/SuperMimic-Q1/scripts/export_*.py`，运行时在 `runtime/`。运行时只依赖 `numpy`、`mujoco`、`onnxruntime`、`pyyaml`，不 import 任何训练代码——它能跑通，就证明导出的东西可以独立部署。

## 1. 导出权重包

### 1.1 基础动作模型

```bash
cd infra/SuperMimic-Q1
SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/export_base_package.py \
    --ckpt logs/base_dual/ckpt_latest.pt --out ../jobs/base_dual/artifact_base \
    --eval-dir logs/base_dual/eval --code-commit $(git rev-parse --short HEAD)
```

| 文件 | 输入 → 输出 | 用途 |
|---|---|---|
| `policy_ref.onnx` | `obs`(720) + `body_ref`(222) → `action`(22)、`joint_target`(22)、`z`(32) | 照着参考做 |
| `policy_tele.onnx` | `obs`(720) + `slots`(19) + `mask`(6) → `action`、`joint_target`、`z` | 听命令做 |
| `decoder.onnx` | `obs`(720) + `z`(32) → `action`、`joint_target` | 两种用法切换时插值 `z` |
| `deploy.yaml` | — | 部署需要知道的一切（见下） |
| `scene.xml` | — | 训练用的仿真场景（地面 + Q1，碰撞与执行器与训练完全相同），网格引用 `q1/q1_v3/meshes` |
| `meta.json`、评测文件 | — | 来源追溯：检查点、代码版本、评测结论 |

三个 ONNX 里的 Decoder 权重完全相同。导出时每个图都在 512 组随机输入上与 PyTorch 对比，最大误差 ≥ 1e-5 直接中止。`joint_target = 站姿 + 0.25 × action` 已在图内算好，可直接作为 PD 目标。只用最基础的算子（opset 17），机器人板上的 ONNX Runtime 1.18 能跑。加 `--copy-ckpt` 会把完整检查点一起拷进包里，方便接着训。

### 1.2 单段动作策略

```bash
SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/export_package.py \
    --ckpt logs/walk1_s1/ckpt_latest.pt --motion-npz data/npz/walk1_s1/walk1_s1.npz \
    --motion-dir ../jobs/walk1_s1/check/motion --eval-dir logs/walk1_s1/eval \
    --out ../jobs/walk1_s1/artifact --code-commit $(git rev-parse --short HEAD)
```

包里是一个 `policy.onnx`（`obs`(720) + `goal`(546) → `action`、`joint_target`）、`deploy.yaml`、提前算好的每帧目标特征 `deploy_motion.npz`、参考动作本身和评测报告。`scripts/deploy_sim_check.py --package <包>` 只用包里的文件做一次闭环检查。

## 2. `deploy.yaml`：网络输入怎么拼

部署代码必须和训练拼法**完全一致**，否则网络"看到"的东西就错了。`deploy.yaml` 把这些全写清楚了：关节顺序、默认站姿、PD 参数（kp、kd、力矩上限、额定力矩、增益预设名）、动作缩放、控制频率、观测布局、`body_ref` 布局、槽位注册表和训练时的命令范围、仿真参数。网络内部已做好归一化，外面直接喂物理量。

**`obs`（720 维）**：5 种量，每种连续放 10 帧（从旧到新）：

| 项 | 维数 | 单位 |
|---|---|---|
| 关节角 − 默认站姿 | 22 | rad |
| 关节角速度 | 22 | rad/s |
| 骨盆 IMU 角速度（骨盆系） | 3 | rad/s |
| 骨盆系下的重力方向 | 3 | 单位向量 |
| 上一拍的 `action` | 22 | — |

刚启动时 10 帧都填当前帧，上一拍 `action` 填 0。

**`body_ref`（222 维）**：机器人正在做参考第 \(i\) 帧时，取第 \(i+1\) 到 \(i+4\) 帧（超出末尾重复最后一帧），每帧 54 维（关节角 − 站姿、关节速度、重力方向、骨盆角速度、航向系骨盆线速度、骨盆高度），最后接 6 维"参考躯干相对机器人躯干的朝向"。IMU 不知道绝对航向，所以开始播放时先把参考整体转到与机器人当前朝向一致（取使 \(R_z(\psi)R_\text{ref}\) 最接近 \(R_\text{robot}\) 的 \(\psi\)，趴着时也稳定）。

**`slots`（19 维）+ `mask`（6 维）**：按槽位注册表顺序拼接（`base` 3、`height` 1、`torso` 3、`hand_L` 3、`hand_R` 3、`feet` 6），不填的槽位写默认值、掩码置 0。`feet` 是预留槽位，必须保持不填。

## 3. 运行时 `runtime/`

| 文件 | 职责 |
|---|---|
| `package.py` | 读 `deploy.yaml`，暴露关节名、站姿、PD、控制周期、槽位表、图路径；同时认基础模型包和单段动作包 |
| `policy.py` | `ObsHistory`（10 帧历史，首帧填满）；`TeleGoal`（按槽位名设值 / 屏蔽）；`BaseModel`（三个 ONNX 会话 + `z` 插值切换）；`SkillModel`（单段动作包） |
| `body_ref.py` | 由参考 `motion.npz` 现算每帧 54 维特征，按 `future_frames` 取未来帧，开始播放时对齐航向 |
| `backend.py` | `Backend` 接口（读传感器、写 PD 目标、切阻尼、换增益）与 `MuJoCoBackend`（载入包内 `scene.xml`） |
| `session.py` | `Session`：后端 + 观测历史 + 上一拍动作，控制循环的公共部分 |
| `selfcheck.py` | 闭环自检（见第 4 节） |

最小用法（在仓库根目录用 `infra/bin/q1py` 运行）：

```python
import numpy as np
from runtime import Package, Session, TeleGoal
from runtime.body_ref import BodyRef

pkg = Package.load("infra/jobs/base_dual/artifact_base")
run = Session(pkg)                       # 默认 MuJoCoBackend；接真机时传入自己的 Backend
run.robot.reset()
run.begin()

# 听命令做：以 0.5 m/s 前进 4 秒
goal = TeleGoal(pkg)
goal.set("base", [0.5, 0.0, 0.0])
for _ in range(200):
    run.apply(*run.model.act_tele(run.obs(), goal))

# 切到照着参考做：25 步内把 z 平滑插到 E_ref
ref = BodyRef(pkg, "path/to/motion.npz")
run.model.start_blend(25)
ref.start(run.robot.torso_quat())
for i in range(ref.T - 1):
    run.apply(*run.model.act_ref(run.obs(), ref.goal(i, run.robot.torso_quat())))
```

`act_tele` / `act_ref` 返回 `(action, joint_target)`；`Session.apply` 把 `joint_target` 交给后端执行一个控制周期，再把新的观测推进历史。

**两种用法之间切换**：`start_blend(n)` 记下切换前最后的 `z`，之后 \(n\) 步内把它与新接口输出的 `z` 线性插值，经 `decoder.onnx` 解码。两个 Encoder 输出在同一个潜空间，这正是共享 Decoder 的意义。

## 4. 部署侧闭环自检

```bash
infra/bin/q1py -m runtime.selfcheck --package infra/jobs/base_dual/artifact_base \
    --clips a.npz b.npz --switch-clip a.npz --blend 0 10 25
```

只用权重包里的文件和 `runtime/`，在 CPU MuJoCo（训练场景，脚摩擦 1.0）里：

- `ref`：每段参考从首帧起用 `policy_ref.onnx` 整条播放，骨盆高度 / 倾角偏离参考超出训练终止阈值算摔倒，报告关节误差；
- `switch`：命令站立 2 s → 插值切到 `E_ref` 播放参考 → 再插值回命令站立 3 s，全程不摔算通过。

结果写到包里的 `runtime_selfcheck.json`。如果自检结果和训练侧评测一致，就说明不只是网络权重对了，连 `deploy.yaml` 里的拼法（帧顺序、历史排列、航向对齐）也写对了。我们的模型在部署侧逐段复现了训练侧的摔倒与否（包括某段在第 8 步就摔的同一个片段）。

## 5. 接到真机

Q1 真机后端是 `runtime.PrimeBotBackend`（PrimeBot SDK 的 ROS 2 话题，用法见 [应用：手柄遥操作](apps.md#53-真机q1--primebot-sdk)）。接其他机器人时实现一个 `Backend` 子类即可，控制循环和推理不用改：

| 方法 | 要做的事 |
|---|---|
| `reset()` | 从当前姿态平滑插值到默认站姿（或指定姿态），之后保持 |
| `sense()` | 返回 `Sensors`：关节角、关节速度（按 `deploy.yaml` 的关节顺序）、骨盆 IMU 角速度（骨盆系）、骨盆系重力方向 |
| `step(joint_target)` | 下发 PD 目标（kp / kd 取 `deploy.yaml`），并等到下一个 20 ms 周期 |
| `damp()` | 急停：切阻尼（kp = 0） |
| `set_pd(kp, kd)` | 换增益（技能策略与基础模型增益不同时用） |
| `torso_quat()` | 躯干朝向：骨盆 IMU 姿态 + 腰关节正运动学（`body_ref` 的航向对齐和朝向项要用） |

我们在 Q1 上的经验（供参考）：

- 底层下发频率要比策略高（我们用约 500 Hz 零阶保持策略目标），每帧位置增量限幅（≤ 0.2 rad）；
- Python 单进程里 ROS 收发与推理会互抢 GIL，收、发拆到独立的实时子进程、经共享内存交换后才稳定（推理平均 11 ms、p99 16 ms）；
- 必须有保护：关节或 IMU 断流超过 0.1 s、策略超时、策略进程退出、初始插值时关节偏离指令过大，都立即切阻尼；
- 上真机前先实测电机刚度 kp——模型对 kp 最敏感（kp × 1.2 时跟踪误差约 × 1.35）；
- 吊带保护下先做"只保持站姿"的测试，再逐级放开。

> ⚠️ 真机运行有摔倒、损坏设备和伤人的风险。请阅读 README 中的免责声明，并在具备安全防护（吊带、急停、空旷场地）的条件下进行。

## 6. 技能策略

基础动作模型只覆盖站立状态下的动作。基础模型覆盖不到的动作（如倒地后起身）可以单独训练成单段动作策略。俯卧起身 / 站立趴下的参考已随仓库发布，训练与使用见 [应用：起身 / 趴下技能](apps.md#52-起身--趴下技能)；其他技能：

1. 准备一条 any4hdmi 格式的参考轨迹，转成 `motion.npz`；
2. 用 `scripts/train/single.sh` 训练（贴地类动作可加 `--reward.w-extra-contact` / `--reward.w-impact` 多余接触与撞击惩罚；需要不同增益时在 `PD_PRESETS` 里加一组，用 `Q1_PD` 选择）；
3. 用 `export_package.py` 导出，运行时用 `SkillModel` 播放；技能与基础模型之间切换时对关节目标做 0.4 s 线性过渡，并用 `set_pd` 切换增益。
