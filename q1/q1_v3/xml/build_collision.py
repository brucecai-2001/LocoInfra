#!/usr/bin/env python3
"""
生成训练用的 Q1 碰撞版模型 q1_v3_collision.xml(原始 q1_v3.xml 不动)。

原模型的碰撞体是一组粗圆柱: 小腿下段和踝部没有碰撞, 小腿可以穿过障碍物而仿真不报。这里改为:
  * 全身用外形网格(MuJoCo 自动取凸包)做碰撞, 与检查参考轨迹时用的碰撞体一致; 原来的圆柱全部关掉
  * 手部保留原来的两个方块(手没有外形网格)
  * 脚底四角各加一个小球, 比网格脚底低 2 mm, 落地时由四个球先接触 -> 每只脚 4 个接触点
    (只用网格时, 脚底和曲面只有 1-2 个接触点, 站不住也算不出压力中心)
  * 同一条腿相邻关节之间、骨盆与髋、躯干与肩不算碰撞(由关节限位保证不穿, 凸包近似会误报)
  * 关节限位取 q1_v3_full_range.xml 两端各收 1°; q1_v3.xml 原来每端收约 0.1 rad,
    参考轨迹会贴住限位。髋膝峰值力矩从 36 保守改为 35 N·m(见 q1/motor_id/README.md)
用法: python build_collision.py   (在本目录生成 q1_v3_collision.xml)
"""
from __future__ import annotations

import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "q1_v3.xml")
FULL_RANGE = os.path.join(HERE, "q1_v3_full_range.xml")
OUT = os.path.join(HERE, "q1_v3_collision.xml")
RANGE_MARGIN = np.deg2rad(1.0)
TAU_MAX = {"hip_pitch": 35.0, "hip_roll": 35.0, "hip_yaw": 35.0, "knee": 35.0}

LEG_CHAIN = ["hip_pitch_link", "hip_roll_link", "hip_yaw_link", "knee_link", "ankle_pitch_link", "ankle_roll_link"]
FOOT_R = 0.008            # 脚底小球半径
FOOT_DROP = 0.002         # 小球比网格脚底低多少
FOOT_INSET = 0.004        # 小球离脚底轮廓边缘往里收多少


def sole_corners(m, d, body):
    """脚底网格最低一层(3 mm 内)的点在 body 系下的 x/y 范围和最低 z。"""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
            V = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pts.append((V - d.xpos[bid]) @ d.xmat[bid].reshape(3, 3))
    P = np.vstack(pts)
    zmin = P[:, 2].min()
    low = P[P[:, 2] < zmin + 0.003]
    return low[:, 0].min(), low[:, 0].max(), low[:, 1].min(), low[:, 1].max(), zmin


def build():
    spec = mujoco.MjSpec.from_file(SRC)
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    full_spec = mujoco.MjSpec.from_file(FULL_RANGE)
    full = {j.name: np.array(j.range) for j in full_spec.joints if j.name}
    for j in spec.joints:
        if j.name in full:
            j.range = full[j.name] + [RANGE_MARGIN, -RANGE_MARGIN]
        key = j.name.replace("left_", "").replace("right_", "").replace("_joint", "")
        if key in TAU_MAX:
            j.actfrcrange = [-TAU_MAX[key], TAU_MAX[key]]
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.group == 1:      # 外形网格: 开碰撞
            g.contype, g.conaffinity = 1, 1
        elif g.type == mujoco.mjtGeom.mjGEOM_BOX and g.parent.name.endswith("wrist_yaw_link"):
            continue                                                   # 手部方块保留
        elif g.contype or g.conaffinity:                               # 原来的圆柱 / 重复网格: 关碰撞, 不显示
            g.contype, g.conaffinity = 0, 0
            g.group = 4
    for side in ("left", "right"):
        body = f"{side}_ankle_roll_link"
        x0, x1, y0, y1, zmin = sole_corners(m0, d0, body)
        b = spec.body(body)
        z = zmin - FOOT_DROP + FOOT_R
        for i, (x, y) in enumerate([(x0, y0), (x0, y1), (x1, y0), (x1, y1)]):
            sx = x + FOOT_R + FOOT_INSET if x == x0 else x - FOOT_R - FOOT_INSET
            sy = y + FOOT_R + FOOT_INSET if y == y0 else y - FOOT_R - FOOT_INSET
            b.add_geom(name=f"{side}_sole_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[FOOT_R, 0, 0],
                       pos=[sx, sy, z], contype=1, conaffinity=1, condim=3, group=3,
                       rgba=[1.0, 0.4, 0.1, 1.0], density=0)
        print(f"{side} 脚底小球: x [{x0 + FOOT_R + FOOT_INSET:.3f}, {x1 - FOOT_R - FOOT_INSET:.3f}] "
              f"y ±{(y1 - y0) / 2 - FOOT_R - FOOT_INSET:.3f}  球底 z {z - FOOT_R:.4f}(网格脚底 {zmin:.4f})")
    for s in ("left", "right"):
        for i, a in enumerate(LEG_CHAIN):
            for bn in LEG_CHAIN[i + 2:]:
                spec.add_exclude(bodyname1=f"{s}_{a}", bodyname2=f"{s}_{bn}")
        spec.add_exclude(bodyname1="pelvis", bodyname2=f"{s}_hip_roll_link")
        spec.add_exclude(bodyname1="pelvis", bodyname2=f"{s}_hip_yaw_link")
        spec.add_exclude(bodyname1="waist_roll_link", bodyname2=f"{s}_hip_pitch_link")
        spec.add_exclude(bodyname1="torso_link", bodyname2=f"{s}_shoulder_roll_link")
        spec.add_exclude(bodyname1="torso_link", bodyname2=f"{s}_shoulder_yaw_link")
    spec.add_exclude(bodyname1="pelvis", bodyname2="torso_link")
    spec.modelname = "q1_v3_collision"
    return spec


def main():
    spec = build()
    spec.compile()
    xml = spec.to_xml()
    with open(OUT, "w") as f:
        f.write("<!-- 由 build_collision.py 从 q1_v3.xml 生成, 不要手改。训练用碰撞版: 全身外形网格碰撞 + 脚底四点 -->\n")
        f.write(xml)
    m = mujoco.MjModel.from_xml_path(OUT)
    ncol = sum(1 for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g])
    print(f"写入 {OUT}: nq={m.nq} nbody={m.nbody} 碰撞几何体 {ncol} 个, 排除对 {m.nexclude} 个")


if __name__ == "__main__":
    main()
