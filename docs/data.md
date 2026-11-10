# 数据处理

> 一句话：把人体动捕数据换算成 Q1 能做的动作，查出并修掉做不了的部分，最后合并成训练用的动作库。
>
> 代码在 `infra/q1infra/`（重定向、可行性检查、轨迹优化、查看器、建库）和 `infra/SuperMimic-Q1/scripts/data/`（转训练格式）。整体流程见 [概览](overview.md)。

## 0. 环境

数据处理只需要 CPU，笔记本即可。

```bash
bash infra/setup.sh          # 克隆 GMR / any4hdmi / mjlab 到 infra/.deps（固定 commit），建 infra/.venv（Python 3.11，需要 uv）
cd infra                     # 下文命令都在 infra/ 下运行
bin/q1py -m q1infra.retarget --help
```

`bin/q1py` 是带好 `PYTHONPATH` 的 Python；`bin/q1view` 用来开轨迹查看器（macOS 上会自动用 `mjpython`）。在多核 GPU 训练机上批量建库时可用 `bin/q1py_server`，它复用训练环境（需在训练环境里额外 `uv pip install -e ../.deps/GMR -e ../.deps/any4hdmi`）。

外部依赖都锁定在固定版本：[GMR](https://github.com/YanjieZe/GMR) `25f56db`、[any4hdmi](https://github.com/EGalahad/any4hdmi) `bf83602`、[mjlab](https://github.com/mujocolab/mjlab) `8bb2bc9`。

## 1. 支持的输入

| 来源 | 是什么 | 怎么读 |
|---|---|---|
| LAFAN1 BVH | Ubisoft 的动捕数据集：走、跑、跳、舞、打斗 | 直接读（GMR 的 BVH 读取器） |
| SMPL-X（如 AMASS SMPL-X 版） | 学术界通用的参数化人体模型 | 用官方人体模型算出 22 个身体关节的位置和朝向，手指、下巴、眼睛不用；需要 SMPL-X 模型文件 |
| SMPL+H（AMASS 原版） | 比 SMPL-X 少脸和手的细节 | 身体关节树与 SMPL-X 相同，旋转直接用；体型统一按 SMPL-X 标准体型算：首次运行时用 `SMPLX_NEUTRAL` 算出 22 个静止关节点，缓存为模型目录下的 `smplx_neutral_rest_joints.npz` |
| 单目视频 | 普通手机或摄像机拍的视频 | GVHMR 重建出 SMPL-X 后按 SMPL-X 处理，见 [应用](apps.md#3-视频学动作) |
| 一句话描述 | 文字 | HY-Motion 1.0 生成 SMPL-H 后按 SMPL+H 处理，见 [应用](apps.md#4-一句话生成动作) |

数据集本身不随仓库分发，请按各自许可自行下载：

- LAFAN1：[ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)，解压到 `infra/data/lafan1/`；
- AMASS：[amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/)（需注册）；
- SMPL-X 模型（SMPL-X / SMPL+H 输入、视频与文本应用需要；只用 LAFAN1 BVH 时不需要）：[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)，放到 `infra/data/body_models/smplx/`。

## 2. 重定向：把人的动作换算到 Q1

```bash
bin/q1py -m q1infra.retarget --bvh data/lafan1/walk1_subject1.bvh --job walk1_s1 --end-sec 30 --render
bin/q1py -m q1infra.retarget --smplx <AMASS .npz> --job <任务名> --render
```

输出到 `infra/jobs/<任务名>/`：`retarget/`（any4hdmi 格式的 Q1 轨迹）、`retarget_report.json`（跟踪误差、贴地、接近限位比例）、`retarget_preview.mp4`（加 `--render` 时）。

**通俗地说**：先把人"缩小"成 Q1 的身材，再让 Q1 的骨盆、躯干、膝、脚、肘、手尽量对准人的对应部位。

**具体做法**（`retarget/gmr_q1.py`）：

1. **读人体动作**，得到每帧每根骨头的全局位置和朝向；源帧率高于 60 Hz 时整数步长抽帧。
2. **按比例缩放**（`HumanScaler`）：
   - 根的位移整体乘"骨盆到踝"的长度比（约 0.45）；
   - 髋、肩这种"锚点"不按人体骨长缩放，而是用人体骨盆 / 胸腔的朝向去驱动 **Q1 自己的**骨盆→髋、骨盆→肩偏移。人和 Q1 在这两处的偏移方向差很多，等比缩放会把脚和手放错；
   - 其余骨头按"Q1 骨长 / 人体骨长"缩放（腿 0.38–0.45、上臂 0.44、前臂 0.50）。
3. **逐帧逆运动学**：用 GMR 求一组关节角，让 Q1 的骨盆、躯干、大腿、膝、脚对准位置和朝向，上臂只对准朝向，肘和手只对准位置。配置在 `q1_ik_config.json`（BVH）和 `q1_smplx_ik_config.json`（SMPL-X）。
4. **贴地**：整段统一上下平移，让脚底最低点的 5% 分位数落在地面上（单帧的穿地 / 悬空留给可行性检查）。
5. **重采样到 50 Hz**：位置和关节线性插值，根四元数 Slerp。
6. 可选：`--flat-feet` 把着地脚放平（深蹲、仆步等 Q1 踝关节行程不够的动作），`--min-pelvis-height` 给骨盆设下限。

**SMPL-X 的朝向偏移是算出来的，不是手调的**（`retarget/smplx_config.py`）。原理：让人和 Q1 摆同一个姿势，两者对应部位朝向之差就是要补的偏移 \(R_\text{offset}=R_\text{human}^{-1}R_\text{Q1}\)。

- SMPL-X 零姿态下所有关节的全局朝向一致，所以让 Q1 先摆 T 字形，就能算出骨盆、躯干、膝、肩的偏移。用同样方法算 G1 机器人，结果与 GMR 作者手调的值一致，说明方法可靠；
- SMPL-X 零姿态其实不是标准 T 字形（上臂下垂约 16°、大腿外张约 8°、小腿内收约 6°），所以上臂、大腿、小腿各再乘一个最小旋转，对齐到人体实际的骨段方向；
- 静止姿势确定不了上臂绕自身轴的扭转（也就是肘往哪弯），取 −25°，这是在 ACCAD 动作上扫出来的手臂误差最小点。

## 3. 统一格式：any4hdmi

重定向之后的所有步骤都只读写 [any4hdmi](https://github.com/EGalahad/any4hdmi) 格式，各步骤不用关心动作原来从哪来：一个 `manifest.json` 加一堆 `motions/**/*.npz`，每个 npz 只存一个数组 `qpos`（每帧 29 个数：根位置 3 + 根四元数 wxyz 4 + 22 个关节角）。

| manifest 字段 | 内容 |
|---|---|
| `format_version` | 2 |
| `mjcf` | 用的哪个 Q1 模型 |
| `timestep` | 0.02（50 Hz） |
| `qpos_names` | 29 个数各是什么 |
| `source` | 来源类型、源文件哈希、原始帧率、缩放比例等，便于追溯 |

格式里放不下的东西（哪只脚着地、可行性检查结果、人工改过哪些帧）放在任务目录的其他文件里。

## 4. 可行性检查

```bash
bin/q1py -m q1infra.feasibility jobs/walk1_s1/retarget          # 写 jobs/walk1_s1/check/
bin/q1py -m q1infra.feasibility traj.npz                        # 也接受含 qpos / fps 的 npz
```

**通俗地说**：重定向出来的动作经常有毛病——手插进身体、支撑脚悬空或打滑、某些瞬间电机力矩不够、物理上根本站不住。这一步只**检查并出报告**，不改动作。

**场景**（`feasibility/scene.py`）规定哪里是地面、身体哪些部位允许碰东西。目前只有平地 `FLAT`：只允许两只脚着地。换场景（台阶、坐椅子）只需新建一个 `Scene`。

| 检查项 | 判定规则 | 发现问题怎么办 |
|---|---|---|
| 关节限位 | 不能超出模型限位 | 优化时夹回 |
| 穿模 | 脚压进地面 > 1.5 cm，或其他部位与环境 / 自身穿透 > 3 mm | 优化修 |
| 不允许的接触 | 脚以外的部位离地 < 1 cm（跪、爬、躺） | 截掉或换动作 |
| 着地 / 滑脚 / 悬空 | 脚离局部地面 3 cm 内且水平速度 < 0.25 m/s 算"着地"；着地时移动 > 2 cm/s 算滑脚；着地却离地 > 2 cm 算悬空 | 优化：钉脚、贴地 |
| 平滑 | 关节转速不超电机上限；角加速度 > 150 rad/s² 算突变 | 优化修 |
| 力矩 | 逆动力学算出所需关节力矩（摩擦系数 0.8），对比峰值和额定 | 只报告 |
| 平衡 | 地面接触力解释不了身体受到的力（差值 > 体重 10%），说明物理上站不住 | 只报告 |

**力矩和平衡怎么算**（`feasibility/check.py`）：

1. 差分求速度和加速度：根用 `mj_differentiatePos`（四元数不能直接相减），再用 Savitzky–Golay 滤波（窗 9、二阶）得平滑加速度；
2. `mj_rne` 做逆动力学，得广义力 \(\tau_\text{id}=M\ddot q+\text{偏置力}\)（前 6 维属于浮动根）；
3. 根部这 6 维力必须由地面接触力承担。每个接触点的摩擦锥线性化成 4 条棱，接触力是棱的非负组合 \(\lambda\ge 0\)，解带正则的有界最小二乘：
   \[\min_{\lambda\ge 0}\ \lVert J_\text{root}\lambda-\tau_\text{id}[{:}6]\rVert^2+0.05^2\lVert\lambda\rVert^2\]
4. 残差 \(\lVert\tau_\text{id}[{:}3]-J\lambda\rVert\) 超过体重 10% 记"不平衡"（腾空、质心出支撑面、需要超出摩擦的力）；关节力矩 \(=\tau_\text{id}[6{:}]-J_\text{joint}\lambda\)，对照 `q1/motor_id/prior.npz` 里的峰值 / 额定。

**结论分三档**：

- `infeasible`：不允许的接触超过 1% 的帧，或力矩达标率 < 80%，或平衡达标率 < 70%（阈值来自实测：力矩 89%、平衡 86% 的走路动作能训成）；
- `needs_fix`：问题能靠优化修好，或者训练时策略能自己消化；
- `pass`。

报告列出不可行时段，`annotations.npz` 逐帧保存支撑脚、接触、关节力矩、平衡残差、脚离地距离，供优化和建库使用。

## 5. 轨迹优化

```bash
bin/q1py -m q1infra.feasibility.optimize jobs/walk1_s1/retarget   # 写 jobs/walk1_s1/check/motion/ 和 optimize_report.json
```

**通俗地说**：在尽量不改变动作样子的前提下，把整段动作一起微调，修掉穿模、悬空和滑脚。

**具体做法**（`feasibility/optimize.py`）：变量是全部帧的 qpos，用 Levenberg–Marquardt 解非线性最小二乘（解析雅可比、稀疏正规方程）。哪只脚在什么时候着地取自检查结果。

| 残差项 | 作用 |
|---|---|
| `keep` | 别离原动作太远（根部、手改关键帧权重 ×3） |
| `acc` | 相邻三帧二阶差分，动作平滑不抖 |
| `vel` | 关节每帧变化不超电机转速 |
| `pin` | 支撑脚在一段支撑期内水平位置固定（去滑脚）；脚移开 > 4 cm 算新的一步 |
| `ground` | 支撑脚最低点贴到地面（去悬空） |
| `clear` | 不自我穿透；脚以外部位离地 ≥ 1 cm；脚压地 ≤ 1 cm |

`pin` / `ground` 的权重在每段支撑首尾 10 帧线性渐变，避免触地 / 离地瞬间跳变。关节限位是硬约束，每步积分后夹回。

优化只管几何、不管力学，所以前后各做一次检查：力矩或平衡达标率下降超过 3 个百分点，或某关节力矩峰值超限且涨了 10% 以上，就判 `rejected`（仍写出，但建议继续用原轨迹）。

## 6. 轨迹查看器

```bash
bin/q1view jobs/walk1_s1/retarget              # 只读
bin/q1view jobs/walk1_s1/check/motion --edit   # 可手改、优化、存盘
```

3D 画面里穿模的部位标红、1 cm 内接近的标橙、力矩超标的关节标黄；底部时间条标出问题帧。手改时拖关节滑条，当前帧成为"关键帧"，改动按 smoothstep 权重平滑摊到前后关键帧之间；打开"钉脚"时改骨盆脚不会跟着动。终端命令 `check` 重新检查、`opt` 联合优化、`save` 存盘，`h` 看帮助。

## 7. 建动作库

```bash
bin/q1py -m q1infra.dataset --name loco_v1 --bvh 'data/lafan1/walk*.bvh' \
    --smplx 'data/amass/ACCAD/*Walking*/*.npz' --exclude 'stagei|Cal' -j 8
```

把一批源动作逐段做重定向和可行性检查（每段一个任务目录 `infra/dataset_jobs/<任务>/`，已做过的直接复用），按规则截掉不合适的部分，合并成一个 any4hdmi 动作库 `infra/datasets/<名字>/`（`manifest.json`、`motions/`、`index.json` 记录每段的来源、截取区间和丢弃原因）。进程崩溃（如 MuJoCo 段错误）会自动重建进程池继续。

截取规则：

| 情况 | 处理 |
|---|---|
| 跪、爬、躺（脚以外部位着地） | 前后各多截 0.5 s |
| 连续 ≥ 1.5 s 站不住（如坐椅子） | 截掉；更短的通常是逆动力学噪声，保留 |
| 双脚同时离地 > 4 cm 持续 ≥ 0.4 s（台阶、平台、大跳） | 前后各多截 0.3 s。CMU 等数据集文件名不写动作名，靠这条和上两条拦住楼梯、坐椅子 |
| 片段开头不稳（骨盆倾斜 ≥ 30°、双脚都离地 ≥ 2 cm，或关节角速度 ≥ 20 rad/s） | 起点后挪到第一个稳的帧；前 2 s 都找不到就丢 |
| 截完不足 2 s | 丢掉 |
| 力矩达标率 < 80%（冲刺、大跳） | 整段丢掉 |

"开头要稳"这条是吃过亏才加的：训练和评测都把机器人直接摆到片段首帧，投掷、挥臂这类动作如果从发力中间开始，机器人一上来就处于失衡状态，几步内必倒。

`--high-dynamic` 是放宽版规则（跳跃腾空保留、力矩只截持续超峰值 1.25 倍的时段），用来建只给 `E_ref` 用的高动态子集；`--novel-vs` 只输出相对已有动作库新增的部分。

## 8. 转成训练格式 `motion.npz`

训练不直接读 any4hdmi，而是先把动作重采样到 50 Hz，用 MuJoCo 正运动学算出每个刚体的位姿和速度，存成 `motion.npz`（`joint_pos / joint_vel` 每帧 22 维，`body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w` 每帧覆盖全部刚体）。这样训练时不用反复计算。这一步在训练环境里跑（见 [训练与评测](training.md#0-环境)）：

```bash
cd infra/SuperMimic-Q1
SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/data/any4hdmi_to_npz.py \
    --datasets ../jobs/walk1_s1/check/motion --output-dir data/npz/walk1_s1
# 大动作库：多进程，并按固定种子留出 3% 做留出集（写到 <output-dir>_holdout）
SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/data/any4hdmi_to_npz_parallel.py \
    --datasets ../datasets/amass_v1 ../datasets/base_v2 --output-dir data/npz/mix_v1 --workers 20 --holdout 0.03
```

`scripts/tele_coverage.py` 可以审计一个动作库的遥操作标签分布（多少帧能只用命令复现、各类命令各占多少小时），见 [训练与评测](training.md#24-从参考动作生成命令标签)。

## 9. 我们用的动作库

| 动作库 | 内容 | 规模 |
|---|---|---|
| `base_v2` | LAFAN1 低动态部分（去掉打斗、跳跃、冲刺、摔倒爬起、地面动作）+ ACCAD 走 / 跑 / 站 / 手势 | 579 段，2.7 h |
| `amass_v1` | AMASS 14 个子集（BMLrub、CMU、EKUT、Eyes_Japan、HUMAN4D、HumanEva、KIT、HDM05、MoSh、SFU、TotalCapture、Transitions、BMLmovi、DanceDB），按文件名去掉坐、楼梯、攀爬、爬行、躺、跪、跑步机、侧手翻、倒立、游泳、骑车、俯卧撑等 | 10177 段源动作（39.9 h）→ 9503 段，32.1 h |
| `mix_v1` | `base_v2` + `amass_v1` 转成 `motion.npz` | 训练集 9587 段、留出集 302 段，约 33 h |

经验与注意事项：

- **解耦数据很重要**：如果数据里下蹲总伴随弯腰，模型就会学成"一蹲就弯腰"。理想情况下需要专门的"只蹲不弯腰""高度不变只倾斜躯干""站着只动手""边走边动手"等动作，现有公开数据覆盖有限；
- **起点安稳**规则会丢掉部分未优化的长走路片段（重定向后脚普遍悬空 2–3 cm），重建前可以先做轨迹优化，或放宽这一项；
- 动作首尾各补 1 s 站姿（让机器人从站立平稳进入、做完回到站立）目前还没实现。
