"""介绍视频的素材采集：用发布权重在 MuJoCo 里真实跑三段闭环仿真，把每个控制周期的状态存成 npz。

    infra/bin/q1py promo/intro/capture.py

三段都是策略闭环的物理仿真（不是运动学回放）：
  ref    基础模型 E_ref 直接跟踪参考库里的一段走路（同时记下参考，渲染时画成"影子"）
  tele   基础模型 E_tele 只听命令：前进、转弯、下蹲、双手
  skill  手柄遥操作状态机（脚本手柄）：趴姿 → 起身技能 → 行走 → 趴下技能
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from apps import ref as reflib  # noqa: E402
from apps.teleop.command import fill_goal, ramp  # noqa: E402
from apps.teleop.gamepad import Edges, ScriptPad  # noqa: E402
from apps.teleop.machine import Teleop  # noqa: E402
from runtime import MuJoCoBackend, Package, Session, TeleGoal  # noqa: E402
from runtime.body_ref import BodyRef  # noqa: E402

BASE = PROJECT / "weights" / "dual_v3b"
SKILL = PROJECT / "weights" / "getup_vpd_ft"
WALK_REF = PROJECT / "apps" / "ref" / "track_web" / "web_20261110_151204_walk1_subject1.npz"
OUT = PROJECT / "infra" / "jobs" / "promo_intro"

# 双手目标（相对头、躯干航向系，m）；默认值来自 deploy.yaml 的槽位注册表
HAND_DEFAULT = np.array([[0.02, 0.135, -0.30], [0.02, -0.135, -0.30]])
HAND_POSES = {
    "forward": np.array([[0.22, 0.12, -0.22], [0.22, -0.12, -0.22]]),
    "side": np.array([[0.04, 0.30, -0.20], [0.04, -0.30, -0.20]]),
    "up": np.array([[0.14, 0.14, -0.02], [0.14, -0.14, -0.02]]),
}


def take_ref() -> dict:
    pkg = Package.load(BASE)
    run = Session(pkg)
    ref = BodyRef(pkg, WALK_REF)
    sim: MuJoCoBackend = run.robot
    d = sim.data
    sim.set_state(ref.root_pos[0], ref.root_quat[0], ref.joint_pos[0], ref.root_lin_vel[0], ref.root_ang_vel_w[0], ref.joint_vel[0])
    run.begin()
    ref.start(sim.torso_quat())
    qpos, ghost, err = [d.qpos.copy()], [], [0.0]
    g0 = d.qpos.copy()
    ghost.append(g0)
    for i in range(ref.T - 1):
        run.apply(*run.model.act_ref(run.obs(), ref.goal(i, sim.torso_quat())))
        f = i + 1
        g = d.qpos.copy()
        g[sim.fq : sim.fq + 3] = ref.root_pos[f]
        g[sim.fq + 3 : sim.fq + 7] = ref.root_quat[f]
        g[sim.q_adr] = ref.joint_pos[f]
        qpos.append(d.qpos.copy())
        ghost.append(g)
        err.append(float(np.abs(d.qpos[sim.q_adr] - ref.joint_pos[f]).mean()))
    err = np.asarray(err)
    print(f"[ref] {ref.T} 帧  关节误差均值 {err[1:].mean():.4f} rad  最终骨盆高 {sim.pelvis_height():.3f} m  upright {sim.upright():.2f}")
    return {"qpos": np.asarray(qpos), "ghost": np.asarray(ghost), "joint_err": err}


def take_tele() -> dict:
    pkg = Package.load(BASE)
    run = Session(pkg)
    sim: MuJoCoBackend = run.robot
    goal = TeleGoal(pkg)
    head_off = pkg.registry["head_offset_in_torso"]
    dt = pkg.control_dt
    sim.reset()
    run.begin()

    # (时长 s, [vx, vy, wz, dh, pitch, roll], 双手姿态名或 None, 标签)
    plan = [
        (1.2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, "stand"),
        (3.6, [0.6, 0.0, 0.0, 0.0, 0.0, 0.0], None, "forward"),
        (3.4, [0.5, 0.0, 0.9, 0.0, 0.0, 0.0], None, "turn"),
        (1.6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, "stop"),
        (2.2, [0.0, 0.0, 0.0, -0.10, 0.0, 0.0], None, "squat"),
        (1.6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, "rise"),
        (2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "forward", "hands_forward"),
        (2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "side", "hands_side"),
        (2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "up", "hands_up"),
        (1.6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, "end"),
    ]
    cmd = np.zeros(6)
    hand_t = HAND_DEFAULT.copy()
    rec = {k: [] for k in ("qpos", "cmd", "vel", "height", "hand_tgt_w", "hand_w", "hand_on", "hand_err", "seg")}
    for si, (sec, target, hands, label) in enumerate(plan):
        for _ in range(int(round(sec / dt))):
            cmd = ramp(cmd, np.asarray(target, dtype=np.float64), dt)
            want = HAND_DEFAULT if hands is None else HAND_POSES[hands]
            hand_t += np.clip(want - hand_t, -0.5 * dt, 0.5 * dt)  # 手目标 0.5 m/s 限速
            hands_on = hands is not None or np.abs(hand_t - HAND_DEFAULT).max() > 1e-4
            fill_goal(goal, cmd)
            goal.set("hand_L", hand_t[0], valid=hands_on)
            goal.set("hand_R", hand_t[1], valid=hands_on)
            run.apply(*run.model.act_tele(run.obs(), goal))
            head, Rh = sim.head_frame(head_off)
            meas = sim.hand_positions(head_off)
            rec["qpos"].append(sim.data.qpos.copy())
            rec["cmd"].append(cmd.copy())
            rec["vel"].append(sim.base_velocity())
            rec["height"].append(sim.pelvis_height())
            rec["hand_tgt_w"].append(head + hand_t @ Rh.T)
            rec["hand_w"].append(np.stack([sim.data.xpos[b] for b in sim.hands]))
            rec["hand_on"].append(hands_on)
            rec["hand_err"].append(np.linalg.norm(meas - hand_t, axis=1).mean())
            rec["seg"].append(si)
        v = np.mean(rec["vel"][-int(1.0 / dt):], axis=0)
        print(f"[tele] {label:14s} cmd {np.round(cmd[:4], 2)}  实测 vx {v[0]:+.2f} vy {v[1]:+.2f} wz {v[2]:+.2f}  "
              f"骨盆高 {sim.pelvis_height():.3f}  手误差 {rec['hand_err'][-1] * 100:.1f} cm  upright {sim.upright():.2f}")
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["seg_names"] = np.array([p[3] for p in plan])
    out["stand_height"] = np.array(pkg.stand_height)
    return out


def take_skill() -> dict:
    base, skill = Package.load(BASE), Package.load(SKILL)
    robot = MuJoCoBackend(base)
    tele = Teleop(base, skill, robot, reflib.find("getup_facedown"), reflib.find("liedown_facedown"))
    pad = ScriptPad("idle:1.0,getup:9.6,fwd:3.0,idle:1.0,liedown:12.5", base.control_dt)
    edges = Edges()
    modes = ["LYING", "GETUP", "WALK", "STRAIGHTEN", "LIEDOWN", "DAMPED"]
    qpos, mode = [], []
    while not pad.done:
        p = pad.read()
        tele.tick(p, edges(p))
        qpos.append(robot.data.qpos.copy())
        mode.append(modes.index(tele.mode))
    mode = np.asarray(mode)
    for k, name in enumerate(modes):
        idx = np.flatnonzero(mode == k)
        if len(idx):
            print(f"[skill] {name:10s} {idx[0] * base.control_dt:6.2f} – {(idx[-1] + 1) * base.control_dt:6.2f} s")
    print(f"[skill] 结束：{tele.mode}  骨盆高 {robot.pelvis_height():.3f} m")
    return {"qpos": np.asarray(qpos), "mode": mode, "mode_names": np.array(modes)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("ref", take_ref), ("tele", take_tele), ("skill", take_skill)):
        np.savez_compressed(OUT / f"take_{name}.npz", **fn())
        print(f"  → {(OUT / f'take_{name}.npz').relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
