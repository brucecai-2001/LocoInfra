# Q1 机器人资产

| 路径 | 内容 |
|---|---|
| `q1_v3/urdf/q1_v3.urdf` | 原始 URDF |
| `q1_v3/xml/q1_v3.xml` | 原始 MJCF |
| `q1_v3/xml/q1_v3_full_range.xml` | 关节限位放到实测全行程的 MJCF |
| `q1_v3/xml/q1_v3_mc.xml` | 带转子惯量的 MJCF，电机先验的证据之一 |
| `q1_v3/xml/q1_v3_collision.xml` | **训练与工具链统一使用的模型**，由 `build_collision.py` 生成，不要手改 |
| `q1_v3/meshes/` | 网格 |
| `motor_id/` | 电机参数先验与辨识工具，产物 `prior.npz` 被可行性检查和训练使用 |

`q1_v3_collision.xml` 相对原始模型的改动（`python q1_v3/xml/build_collision.py` 重新生成）：

- 全身外形用网格碰撞，每只脚底加 4 个小球作为接触点；
- 关节限位取实测全行程两端各收 1°（原始模型每端收约 0.1 rad，参考轨迹会贴住限位）；
- 髋 / 膝峰值力矩从原始模型的 36 保守改为 35 N·m。

`q1_v3/` 下的机器人模型文件来自厂商开源发布，相关权利归原权利人所有。
