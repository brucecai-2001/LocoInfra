"""统一权重包的部署侧闭环自检：只用包里的文件和 runtime，CPU MuJoCo（训练场景，脚摩擦 1.0）。

    infra/bin/q1py -m runtime.selfcheck --package infra/jobs/<任务>/artifact_base --clips a.npz b.npz [--switch-clip a.npz]

ref     每段参考从首帧起用 policy_ref.onnx 整条播放：骨盆高度 / 倾角偏离参考超出训练终止阈值算摔倒；
        报告关节误差均值（rad）。与训练侧整条确定性评测口径相同。
switch  两个应用之间切换：遥操作站立 2 s → 在 blend 步内把 z 插值到 E_ref，从当前姿态起播放参考
        （航向对齐到机器人）→ 再插值回遥操作站立 3 s；全程不摔算通过。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from runtime import Package, Session, TeleGoal
from runtime.body_ref import BodyRef, quat_to_mat

HEIGHT_THR, TILT_THR = 0.15, 0.8  # 与训练终止相同（骨盆高度差 m、重力方向 z 分量差）


def play_ref(run: Session, clip: Path) -> dict:
    ref = BodyRef(run.pkg, clip)
    run.robot.set_state(ref.root_pos[0], ref.root_quat[0], ref.joint_pos[0], ref.root_lin_vel[0], ref.root_ang_vel_w[0], ref.joint_vel[0])
    run.begin()
    ref.start(run.robot.torso_quat())
    errs, fell = [], None
    g = np.array([0.0, 0.0, -1.0])
    d = run.robot.data
    for i in range(ref.T - 1):
        run.apply(*run.model.act_ref(run.obs(), ref.goal(i, run.robot.torso_quat())))
        f = i + 1
        errs.append(float(np.abs(d.qpos[run.robot.q_adr] - ref.joint_pos[f]).mean()))
        dz = abs(run.robot.pelvis_height() - ref.root_pos[f][2])
        tilt = abs((quat_to_mat(ref.root_quat[f]).T @ g)[2] - (d.xmat[run.robot.pelvis].reshape(3, 3).T @ g)[2])
        if dz > HEIGHT_THR or tilt > TILT_THR:
            fell = round(f * run.pkg.control_dt, 2)
            break
    return {"clip": clip.stem, "steps": len(errs), "of": ref.T - 1, "fell_at_s": fell, "joint_err_mean": round(float(np.mean(errs)), 4)}


def switch(run: Session, clip: Path, blend: int, ref_seconds: float) -> dict:
    goal = TeleGoal(run.pkg)
    goal.set("base", [0.0, 0.0, 0.0])
    run.robot.reset()
    run.begin()
    ref = BodyRef(run.pkg, clip)
    dt = run.pkg.control_dt
    low = []
    fell = None
    g = np.array([0.0, 0.0, -1.0])

    def check(t, f=None):
        """f: 参考帧（播放参考时按参考判定，与训练终止相同）；None：按站立判定。"""
        nonlocal fell
        h = run.robot.pelvis_height()
        low.append(h)
        if f is None:
            bad = h < run.pkg.stand_height - HEIGHT_THR or run.robot.upright() < 0.3
        else:
            tilt = abs((quat_to_mat(ref.root_quat[f]).T @ g)[2] - (run.robot.data.xmat[run.robot.pelvis].reshape(3, 3).T @ g)[2])
            bad = abs(h - ref.root_pos[f][2]) > HEIGHT_THR or tilt > TILT_THR
        if fell is None and bad:
            fell = round(t, 2)

    t = 0.0
    for _ in range(int(2.0 / dt)):
        run.apply(*run.model.act_tele(run.obs(), goal))
        t += dt
        check(t)
    run.model.start_blend(blend)
    ref.start(run.robot.torso_quat())
    n_ref = min(ref.T - 1, int(ref_seconds / dt))
    for i in range(n_ref):
        run.apply(*run.model.act_ref(run.obs(), ref.goal(i, run.robot.torso_quat())))
        t += dt
        check(t, i + 1)
    run.model.start_blend(blend)
    for _ in range(int(3.0 / dt)):
        run.apply(*run.model.act_tele(run.obs(), goal))
        t += dt
        check(t)
    return {"clip": clip.stem, "blend_steps": blend, "ref_seconds": round(n_ref * dt, 2), "fell_at_s": fell, "min_pelvis_height": round(min(low), 3)}


def main() -> None:
    p = argparse.ArgumentParser(description="统一权重包的部署侧闭环自检")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--clips", type=Path, nargs="*", default=[], help="E_ref 整条播放的参考（SuperMimic motion.npz）")
    p.add_argument("--switch-clip", type=Path, default=None, help="切换测试用的参考")
    p.add_argument("--blend", type=int, nargs="*", default=[0, 10, 25])
    p.add_argument("--ref-seconds", type=float, default=6.0)
    a = p.parse_args()
    pkg = Package.load(a.package)
    run = Session(pkg)
    out = {"ref": [], "switch": []}
    for c in a.clips:
        r = play_ref(run, c)
        out["ref"].append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    if a.switch_clip is not None:
        for b in a.blend:
            r = switch(run, a.switch_clip, b, a.ref_seconds)
            out["switch"].append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (a.package / "runtime_selfcheck.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
