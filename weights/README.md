# 发布权重

> ⚠️ **仅限非商业用途。** 本目录下的权重以 CC BY-NC 4.0 发布，并同时受训练数据 LAFAN1（CC BY-NC-ND 4.0）与 AMASS（仅限非商业科研）许可的约束，**不适用**仓库代码的 MIT 许可。详见 [`LICENSE`](LICENSE)。
>
> *Non-commercial use only. The weights are released under CC BY-NC 4.0 and are additionally subject to the LAFAN1 (CC BY-NC-ND 4.0) and AMASS (non-commercial research only) licenses. They are not covered by the MIT license of the code.*

| 目录 | 内容 |
|---|---|
| `dual_v3b/` | 基础动作模型（双 Encoder），在约 33 h 动作库（AMASS 14 个子集 + LAFAN1）上训练，单卡累计约 35 GPU 小时 |
| `getup_vpd_ft/` | 起身 / 趴下技能策略（单段动作包），跟踪 `apps/ref/skills/` 的 `getup_facedown` / `liedown_facedown`，两段共用一个策略；低增益预设 `Q1_PD=vendor_getup`（髋膝 kp 30），约 5 GPU 小时 |

## dual_v3b

包内文件：`policy_ref.onnx` / `policy_tele.onnx` / `decoder.onnx`（完整参考 / 命令两种用法与共用 Decoder）、`deploy.yaml`（网络输入拼法）、`scene.xml`（训练场景）、`meta.json`（训练记录），以及留出集、命令、鲁棒性和部署侧闭环自检的评测报告。格式说明见 [导出与部署](../docs/deployment.md)，各项数字见 [实验结果](../docs/results.md#23-基础动作模型)。

```bash
infra/bin/q1py -m runtime.selfcheck --package weights/dual_v3b --clips <motion.npz> --switch-clip <motion.npz>
infra/bin/q1py -m apps.track_web --package weights/dual_v3b
infra/bin/q1py -m apps.text2motion --package weights/dual_v3b
```

## getup_vpd_ft

包内文件：`policy.onnx`（`obs`(720) + `goal`(546) → `action`、`joint_target`）、`deploy.yaml`（含技能用的 PD 增益）、`deploy_motion.npz`（预先算好的每帧目标特征）、`motion.npz`（起身参考）、起身与趴下两份验收报告（`report*.html` / `report*.json`）和回放视频。

| 验收项 | 起身 | 趴下 |
|---|---|---|
| 整条确定性播放 | 回报 0.972，不摔 | 回报 0.975，不摔 |
| 全量随机化 1000 回合 | 0 摔倒，回报 0.937 | 0 摔倒，回报 0.944 |
| 关节误差（全程平均） | 0.053 rad | 0.049 rad |
| 力矩峰值 / 上限 | 0.45 | 0.52 |
| 推搡 0.5 / 1.0 / 1.5 m/s 摔倒（每档 64） | 0 / 7 / 34 | 2 / 1 / 11 |

手柄遥操作默认使用它（见 [应用：手柄遥操作](../docs/apps.md#5-手柄遥操作)）：

```bash
infra/bin/q1py -m apps.teleop                         # 仿真：L2+A 起身，L2+B 趴下
```

起身 / 趴下过程中对推搡不鲁棒（1.5 m/s 时约一半摔倒），真机上务必用吊带兜住。

权重只在仿真中验证。上真机前请阅读 [免责声明](../README.md#免责声明) 和 [接到真机](../docs/deployment.md#5-接到真机)。
