"""遥操作走姿检查：E_tele 按几种速度命令走，报告脚尖朝向、髋 yaw、步宽；可选用 E_ref 播放走路参考作对照。

    infra/bin/q1py apps/teleop/gait_check.py --package infra/jobs/<任务>/artifact_base [--clips walk_a.npz ...]

脚尖朝向 = 脚（ankle_roll_link）航向 − 骨盆航向，髋 yaw 取关节角；两者都按"向外为正"（右腿取反）。
步宽 = 双脚在骨盆航向系下的横向距离。每条命令丢掉前 2 s（起步）。
人类走路参考（重定向后）：脚尖 −6 … +10°、髋 yaw −11 … +1°、步宽 15–21 cm。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.body_ref import BodyRef  # noqa: E402
from runtime import Session  # noqa: E402
from apps.teleop.command import ramp  # noqa: E402
from apps.teleop.sim import Controller  # noqa: E402

COMMANDS = {"fwd_0.3": (0.3, 0, 0), "fwd_0.6": (0.6, 0, 0), "back_0.4": (-0.4, 0, 0), "left_0.3": (0, 0.3, 0), "turn_l_1.0": (0, 0, 1.0)}


def yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


class Gait:
    def __init__(self, model: mujoco.MjModel, joint_names: list[str]):
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{s}_ankle_roll_link") for s in ("left", "right")]
        self.hy = [joint_names.index(f"{s}_hip_yaw_joint") for s in ("left", "right")]

    def frame(self, d: mujoco.MjData, pelvis: int, q_adr: np.ndarray) -> list[float]:
        hp = yaw(d.xmat[pelvis].reshape(3, 3))
        fy = [np.degrees(np.angle(np.exp(1j * (yaw(d.xmat[f].reshape(3, 3)) - hp)))) for f in self.feet]
        dl = d.xpos[self.feet[0]] - d.xpos[self.feet[1]]
        q = d.qpos[q_adr]
        return [fy[0], -fy[1], np.degrees(q[self.hy[0]]), -np.degrees(q[self.hy[1]]), -np.sin(hp) * dl[0] + np.cos(hp) * dl[1]]

    @staticmethod
    def summary(rows: list[list[float]]) -> dict:
        m = np.mean(rows, 0)
        return {"toe_out_deg": [round(m[0], 1), round(m[1], 1)], "hip_yaw_out_deg": [round(m[2], 1), round(m[3], 1)], "width_cm": round(m[4] * 100, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description="遥操作走姿检查")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--clips", type=Path, nargs="*", default=[], help="E_ref 对照用的走路参考（SuperMimic motion.npz）")
    p.add_argument("--seconds", type=float, default=8.0)
    a = p.parse_args()
    ctl = Controller(a.package)
    g = Gait(ctl.sim.model, list(ctl.pkg.joint_names))
    dt = ctl.pkg.control_dt
    out: dict = {"tele": {}, "ref": {}}
    for name, (vx, vy, wz) in COMMANDS.items():
        ctl.reset()
        cmd, rows = np.zeros(6), []
        for i in range(int(a.seconds / dt)):
            cmd = ramp(cmd, np.array([vx, vy, wz, 0, 0, 0.0]), dt)
            ctl.step(cmd)
            if i * dt > 2.0:
                rows.append(g.frame(ctl.sim.data, ctl.sim.pelvis, ctl.sim.q_adr))
        out["tele"][name] = {**Gait.summary(rows), "fell": ctl.fallen}
        print(f"E_tele {name:11s}", json.dumps(out["tele"][name], ensure_ascii=False), flush=True)
    run = Session(ctl.pkg)
    ghost = mujoco.MjData(run.robot.model)
    for c in a.clips:
        ref = BodyRef(ctl.pkg, c)
        run.robot.set_state(ref.root_pos[0], ref.root_quat[0], ref.joint_pos[0], ref.root_lin_vel[0], ref.root_ang_vel_w[0], ref.joint_vel[0])
        run.begin()
        ref.start(run.robot.torso_quat())
        sim_rows, ref_rows = [], []
        for i in range(ref.T - 1):
            run.apply(*run.model.act_ref(run.obs(), ref.goal(i, run.robot.torso_quat())))
            sim_rows.append(g.frame(run.robot.data, run.robot.pelvis, run.robot.q_adr))
            ghost.qpos[run.robot.fq : run.robot.fq + 3], ghost.qpos[run.robot.fq + 3 : run.robot.fq + 7] = ref.root_pos[i + 1], ref.root_quat[i + 1]
            ghost.qpos[run.robot.q_adr] = ref.joint_pos[i + 1]
            mujoco.mj_kinematics(run.robot.model, ghost)
            ref_rows.append(g.frame(ghost, run.robot.pelvis, run.robot.q_adr))
        out["ref"][c.stem] = {"reference": Gait.summary(ref_rows), "E_ref": Gait.summary(sim_rows)}
        print(f"E_ref  {c.stem}", json.dumps(out["ref"][c.stem], ensure_ascii=False), flush=True)
    (a.package / "gait_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
