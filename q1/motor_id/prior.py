"""
无真机电机参数辨识(档 1): 汇总本地证据 → 执行器先验 + 随机化范围, 写 prior.npz。

证据(全部来自 q1/q1_v3 下的公开模型文件):
  E1 URDF <limit effort velocity>        q1/q1_v3/urdf/q1_v3.urdf
  E2 MJCF q1_v3_mc.xml                    ctrlrange 36/22, armature 0.004, damping 0, frictionloss 0
  E3 MJCF q1_v3.xml actuatorfrcrange     36 / 22
  E4 整机质量(MJCF 惯性)                  由本脚本求和

推断链:
  峰值力矩 τ_max   = E1–E3 一致的 36 / 22; 髋膝腰保守取 35, 踝臂 22
  空载转速 ω_max   = E1 velocity 36 / 32 rad/s(URDF velocity 字段通常填电机空载转速 / 减速比)
  转子惯量 armature = E2 0.004 kg·m²(关节侧); 随机化 ×0.5–2
  持续力矩 τ_cont  = 公开资料未给, 按峰值 / 额定 ≈ 3 估计(actuator.PEAK_TO_RATED), 约 11.7 / 7.3 N·m
  摩擦             模型给 0 → 名义 0, 随机化上界按 τ_max 的 1.5–2%(库仑) + 小粘滞

运行:
  python prior.py            打印证据表与推断, 写 prior.npz
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from joints import ALL                                  # noqa: E402
from actuator import NOMINAL, DR_RANGES, joint_class, for_joints  # noqa: E402

G = 9.81


# ------------------------------------------------------------------ 证据
def read_urdf_limits():
    import re
    path = os.path.join(ROOT, "q1", "q1_v3", "urdf", "q1_v3.urdf")
    txt = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'<joint\s+name="([a-z_]+)"\s+type="revolute">(.*?)</joint>', txt, re.S):
        lim = re.search(r'<limit[^>]*effort="([\d.]+)"[^>]*velocity="([\d.]+)"', m.group(2))
        if lim:
            out[m.group(1)] = (float(lim.group(1)), float(lim.group(2)))
    return out


def read_mjcf():
    import re
    path = os.path.join(ROOT, "q1", "q1_v3", "xml", "q1_v3_mc.xml")
    txt = open(path, encoding="utf-8").read()
    arm = float(re.search(r'armature="([\d.]+)"', txt).group(1))
    damp = float(re.search(r'damping="([\d.]+)"', txt).group(1))
    fric = float(re.search(r'frictionloss="([\d.]+)"', txt).group(1))
    live = re.sub(r"<!--.*?-->", "", txt, flags=re.S)
    ctrl = {m.group(1): float(m.group(2)) for m in re.finditer(r'joint="([a-z_]+)"\s+ctrlrange="-([\d.]+)', live)}
    mass = sum(float(x) for x in re.findall(r'mass="([\d.]+)"', txt))
    return arm, damp, fric, ctrl, mass


def evidence():
    urdf = read_urdf_limits()
    arm, damp, fric, ctrl, mass = read_mjcf()
    print("== 证据表(本地文件) ==")
    print(f"整机质量(MJCF 惯性求和) {mass:.2f} kg, 体重 {mass*G:.0f} N")
    print(f"q1_v3_mc.xml 关节默认: armature={arm} kg·m², damping={damp}, frictionloss={fric}")
    print(f"{'关节':<28s}{'URDF effort':>12s}{'URDF vel':>10s}{'MJCF ctrl':>11s}{'类':>4s}")
    for n in ALL:
        e, v = urdf[n]
        print(f"{n:<28s}{e:12.0f}{v:10.0f}{ctrl.get(n, float('nan')):11.0f}{joint_class(n):>4s}")
    # 一致性
    ok = all(abs(urdf[n][0] - ctrl[n]) < 1e-6 for n in ALL)
    print(f"URDF effort 与 MJCF ctrlrange {'完全一致' if ok else '存在不一致'}")
    return urdf, arm, mass


# ------------------------------------------------------------------ 物理推断
def inference(arm, mass):
    print("\n== 推断 ==")
    for c, p in NOMINAL.items():
        corner = p.tau_max * p.omega_max * p.omega_knee_frac
        print(f"类 {c}: τ_max {p.tau_max:.0f} N·m, ω_max {p.omega_max:.0f} rad/s({p.omega_max*60/2/np.pi:.0f} rpm 关节侧), "
              f"恒力矩区到 {p.omega_knee_frac*p.omega_max:.0f} rad/s, 拐点机械功率 {corner:.0f} W, τ_cont≈{p.tau_cont:.1f} N·m")
    print(f"转子惯量: 关节侧 {arm} kg·m²; 若减速比 N=6/8/10 → 电机侧 {arm/36*1e6:.0f}/{arm/64*1e6:.0f}/{arm/100*1e6:.0f} g·cm²"
          f"(≈ 60–80 mm 无框力矩电机量级, 与 36 N·m 峰值自洽)")
    # 反射惯量 vs 连杆惯量: 说明 armature 不可忽略
    print(f"  对比: 小腿+脚约 0.85 kg、质心距膝 0.2 m → 连杆惯量 ≈ {0.85*0.2**2:.3f} kg·m², 反射转子惯量占 {arm/(0.85*0.2**2+arm)*100:.0f}% → 训练模型必须带 armature")
    # 行走功率 → 持续力矩校核
    v, cot = 1.2, 1.0
    p_mech = cot * mass * G * v
    print(f"行走校核: 1.2 m/s, 运输成本取 1.0 → 全身机械功率 ≈ {p_mech:.0f} W, 分摊到 12 个腿关节 ≈ {p_mech/12:.0f} W/关节;"
          f" 若典型角速度 3 rad/s → 均方根力矩 ≈ {p_mech/12/3:.1f} N·m, 低于 τ_cont 估计 {NOMINAL['A'].tau_cont:.0f} / {NOMINAL['B'].tau_cont:.1f} → 自洽")
    # 单腿静态: 膝屈 60° 站立
    m_above_knee = mass - 2 * (1.14 + 0.56 + 0.28)
    shank = 0.22
    knee_static = m_above_knee * G * shank * np.sin(np.deg2rad(30))      # 近似: 膝屈 60° 时小腿与铅垂夹角 30°
    print(f"静态校核: 膝屈 60° 双脚站立, 膝上质量 {m_above_knee:.1f} kg → 单膝 ≈ {knee_static/2:.1f} N·m(双支撑)/{knee_static:.1f}(单支撑), "
          f"占 τ_cont {knee_static/NOMINAL['A'].tau_cont*100:.0f}% → 单腿久站会热限")
    print("DR 范围:")
    for c, r in DR_RANGES.items():
        print(f"  类 {c}: " + ", ".join(f"{k} {lo:g}–{hi:g}" for k, (lo, hi) in r.items()))


def save(path=os.path.join(HERE, "prior.npz")):
    ps = for_joints(ALL)
    cls = [joint_class(n) for n in ALL]
    fields = ["tau_max", "tau_cont", "omega_max", "omega_knee_frac", "armature", "friction_c", "friction_b"]
    data = {f: np.array([getattr(ps[n], f) for n in ALL]) for f in fields}
    for f in fields:
        if f == "tau_cont":
            continue
        data[f"dr_low_{f}"] = np.array([DR_RANGES[c].get(f, (getattr(NOMINAL[c], f),) * 2)[0] for c in cls])
        data[f"dr_high_{f}"] = np.array([DR_RANGES[c].get(f, (getattr(NOMINAL[c], f),) * 2)[1] for c in cls])
    np.savez(path, names=np.array(ALL), cls=np.array(cls), source=np.array("prior:urdf+mjcf_mc, rated=peak/3, no hardware"), **data)
    print(f"\n已写 {os.path.relpath(path, ROOT)} ({len(ALL)} 关节, 字段 {', '.join(fields)} + dr_low_/dr_high_)")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    _, arm, mass = evidence()
    inference(arm, mass)
    save()


if __name__ == "__main__":
    main()
