"""轨迹查看 / 编辑的核心（不含窗口）：轨迹、关键帧、检查结果、编辑、优化、存盘。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from q1infra.feasibility.check import NEAR, SIDES, check_motion, load_input, load_robot
from q1infra.feasibility.optimize import TrajOpt
from q1infra.feasibility.scene import Scene

RED = (1.0, 0.2, 0.2, 1.0)
ORANGE = (1.0, 0.6, 0.1, 1.0)
YELLOW = (1.0, 0.9, 0.1, 1.0)
BASE = (("pel_x m", -3.0, 3.0), ("pel_y m", -3.0, 3.0), ("pel_z m", 0.0, 1.5),
        ("pel_yaw", -180.0, 180.0), ("pel_pitch", -90.0, 90.0), ("pel_roll", -90.0, 90.0))


def smoothstep(u: float) -> float:
    return u * u * (3 - 2 * u)


@dataclass
class FrameStatus:
    collide: list[tuple[str, str, float]]  # (部位, 对象, 深度 m)
    near: list[tuple[str, str, float]]
    torque_over: list[tuple[str, float, float]]  # (关节, |τ|, 上限)
    stance: tuple[bool, bool]


class TrajectoryEditor:
    def __init__(self, path: Path, scene: Scene, log=print):
        self.path = Path(path)
        self.scene = scene
        self.log = log
        name, qpos, dt, _ = load_input(self.path)[0]
        self.name, self.dt = name, dt
        self.Q = np.array(qpos, float)
        self.key = np.zeros(len(self.Q), bool)
        self.robot = load_robot(scene)
        self.m = self.robot.model
        self.d = mj.MjData(self.m)
        self.joint_names = self.robot.joint_names
        self.lo, self.hi = self.m.jnt_range[1:, 0], self.m.jnt_range[1:, 1]
        self.joint_body = [int(self.m.jnt_bodyid[j]) for j in range(1, self.m.njnt)]
        self.undo_stack: list[tuple[np.ndarray, np.ndarray]] = []
        self.report: dict | None = None
        self.ann: dict | None = None
        self.stale = True
        self.opt_thread: threading.Thread | None = None
        self.opt_result: np.ndarray | None = None
        self.run_check()

    # ---------- 基本 ----------
    @property
    def N(self) -> int:
        return len(self.Q)

    def qdiff(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        v = np.zeros(self.m.nv)
        mj.mj_differentiatePos(self.m, v, 1.0, q0, q1)
        return v

    def qinteg(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, float)
        mj.mj_integratePos(self.m, q, v, 1.0)
        q[7:] = np.clip(q[7:], self.lo, self.hi)
        return q

    def push_undo(self) -> None:
        self.undo_stack.append((self.Q.copy(), self.key.copy()))
        self.undo_stack = self.undo_stack[-30:]

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.Q, self.key = self.undo_stack.pop()
        self.stale = True
        return True

    # ---------- 检查 ----------
    def run_check(self) -> dict:
        self.report, self.ann = check_motion(self.robot, self.Q, self.dt)
        self.stale = False
        return self.report

    def frame_status(self, t: int) -> FrameStatus:
        """当前帧的接触（实时算）和力矩（取最近一次整条检查的结果）。"""
        m, d, sc = self.m, self.d, self.scene
        d.qpos[:] = self.Q[t]
        mj.mj_forward(m, d)
        env = self.robot.groups != ""
        collide, near = [], []
        for c in d.contact[: d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if env[g1] and env[g2]:
                continue
            if env[g1] or env[g2]:
                ge, gr = (g1, g2) if env[g1] else (g2, g1)
                body = self.robot.body_names[m.geom_bodyid[gr]]
                group = self.robot.groups[ge]
                if sc.allowed(body, group):
                    if -c.dist > sc.soft_max:
                        collide.append((body, group, -c.dist))
                elif c.dist < sc.contact_tol or -c.dist > sc.hard_tol:
                    collide.append((body, group, -c.dist))
                elif c.dist < NEAR:
                    near.append((body, group, -c.dist))
            elif -c.dist > sc.hard_tol:
                collide.append((self.robot.body_names[m.geom_bodyid[g1]], self.robot.body_names[m.geom_bodyid[g2]], -c.dist))
        over = []
        stance = (False, False)
        if self.ann is not None and not self.stale:
            tau = np.abs(self.ann["joint_torque"][t])
            over = [(self.joint_names[j], float(tau[j]), float(self.robot.tau_max[j])) for j in np.flatnonzero(tau > self.robot.tau_max)]
            stance = (bool(self.ann["stance_left"][t]), bool(self.ann["stance_right"][t]))
        return FrameStatus(sorted(collide, key=lambda x: -x[2]), sorted(near, key=lambda x: -x[2]), over, stance)

    def bar(self, t: int, width: int = 100) -> str:
        """缩略条：# 当前、! 不可行、x 穿模、T 力矩超限、K 关键帧、- 正常。检查结果过期时只显示 # 和 K。"""
        chars = []
        a_ = self.ann if not self.stale else None
        for i in range(width):
            a, b = i * self.N // width, max((i + 1) * self.N // width, i * self.N // width + 1)
            if a <= t < b:
                chars.append("#")
                continue
            if a_ is not None:
                if (a_["disallowed_contact"][a:b] | a_["unsupported"][a:b]).any():
                    chars.append("!")
                    continue
                if ((a_["env_penetration_m"][a:b] > self.scene.hard_tol) | (a_["self_penetration_m"][a:b] > self.scene.hard_tol)).any():
                    chars.append("x")
                    continue
                if (np.abs(a_["joint_torque"][a:b]) > self.robot.tau_max).any():
                    chars.append("T")
                    continue
            chars.append("K" if self.key[a:b].any() else "-")
        return "".join(chars)

    def colors(self, t: int, status: FrameStatus | None = None) -> dict[str, tuple]:
        """部位 -> 颜色：穿模红、接近橙、力矩超限的关节所在部位黄。"""
        st = status or self.frame_status(t)
        want: dict[str, tuple] = {}
        for j, _, _ in st.torque_over:
            want[self.robot.body_names[self.joint_body[self.joint_names.index(j)]]] = YELLOW
        for a, _, _ in st.near:
            want[a] = ORANGE
        for a, b, _ in st.collide:
            want[a] = RED
            if b in self.robot.body_names:
                want[b] = RED
        return want

    # ---------- 姿态读写 ----------
    @staticmethod
    def base_of(q: np.ndarray) -> np.ndarray:
        yaw, pitch, roll = R.from_quat(q[3:7], scalar_first=True).as_euler("ZYX", degrees=True)
        return np.array([q[0], q[1], q[2], yaw, pitch, roll])

    @staticmethod
    def set_base(q: np.ndarray, base: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[0:3] = base[0:3]
        q[3:7] = R.from_euler("ZYX", base[3:6], degrees=True).as_quat(scalar_first=True)
        return q

    def foot_pose(self, q: np.ndarray, side: str) -> np.ndarray:
        self.d.qpos[:] = q
        mj.mj_kinematics(self.m, self.d)
        b = self.robot.foot_bodies[side]
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def keep_foot(self, q: np.ndarray, side: str, T_target: np.ndarray) -> np.ndarray:
        """平移、旋转骨盆，使这只脚回到 T_target（只改根位姿）。"""
        T_foot = self.foot_pose(q, side)
        T_root = np.eye(4)
        T_root[:3, :3] = R.from_quat(q[3:7], scalar_first=True).as_matrix()
        T_root[:3, 3] = q[0:3]
        T_new = T_target @ np.linalg.inv(T_foot) @ T_root
        q = q.copy()
        q[0:3] = T_new[:3, 3]
        q[3:7] = R.from_matrix(T_new[:3, :3]).as_quat(scalar_first=True)
        return q

    # ---------- 编辑 ----------
    def apply_edit(self, t: int, q_new: np.ndarray, new_burst: bool = True) -> None:
        """第 t 帧改成 q_new，改动量平滑摊到前后最近的关键帧之间；第 t 帧成为关键帧。"""
        if new_burst:
            self.push_undo()
        v = self.qdiff(self.Q[t], q_new)
        ks = np.flatnonzero(self.key)
        prev, nxt = ks[ks < t], ks[ks > t]
        p = int(prev[-1]) if len(prev) else None
        n = int(nxt[0]) if len(nxt) else None
        lo = p + 1 if p is not None else 0
        hi = n - 1 if n is not None else self.N - 1
        for s in range(lo, hi + 1):
            if s == t:
                continue
            if s < t:
                w = 1.0 if p is None else smoothstep((s - p) / (t - p))
            else:
                w = 1.0 if n is None else smoothstep((n - s) / (n - t))
            self.Q[s] = self.qinteg(self.Q[s], w * v)
        self.Q[t] = q_new
        self.key[t] = True
        self.stale = True

    def shift(self, t: int, dxyz_cm: np.ndarray) -> None:
        q = self.Q[t].copy()
        q[0:3] += np.asarray(dxyz_cm, float) / 100.0
        self.apply_edit(t, q)

    # ---------- 优化 ----------
    def start_opt(self, iters: int) -> bool:
        if self.opt_thread is not None:
            return False
        if self.stale:
            self.run_check()
        Q, key = self.Q.copy(), self.key.copy()
        stance = {s: self.ann[f"stance_{s}"].copy() for s in SIDES}

        def run():
            try:
                self.opt_result = TrajOpt(self.robot, Q, self.dt, stance, log=self.log, key=key).run(iters)
            except Exception as e:  # noqa: BLE001
                self.log(f"[opt] 出错：{e!r}")
                self.opt_result = None

        self.opt_result = None
        self.opt_thread = threading.Thread(target=run, daemon=True)
        self.opt_thread.start()
        return True

    def poll_opt(self) -> bool:
        """优化结束则替换轨迹（可 undo），返回是否刚完成。"""
        if self.opt_thread is None or self.opt_thread.is_alive():
            return False
        self.opt_thread = None
        if self.opt_result is not None:
            self.push_undo()
            self.Q = self.opt_result
            self.run_check()
        return True

    # ---------- 存盘 ----------
    def default_save_path(self) -> Path:
        if self.path.is_dir():
            if self.path.name == "motion" and self.path.parent.name == "check":
                return self.path
            return self.path.parent / "check" / "motion"
        return self.path.with_name(f"{self.path.stem}_edit.npz")

    def save(self, out: Path | None = None) -> Path:
        out = Path(out) if out else self.default_save_path()
        if self.path.is_dir():
            from any4hdmi.core.format import load_manifest, save_motion, write_manifest

            man = load_manifest(self.path)
            motions = out / man.payload.get("motions_subdir", "motions")
            save_motion(motions / f"{self.name}.npz", self.Q)
            source = dict(man.payload.get("source", {}))
            source["edited"] = {"from": str(self.path), "keys": int(self.key.sum())}
            write_manifest(
                out, dataset_name=man.dataset_name, mjcf=man.mjcf, timestep=man.timestep,
                qpos_names=list(man.payload["qpos_names"]), num_motions=1, source=source,
                total_hours=self.N * man.timestep / 3600.0,
            )
            np.save(out / "edit_keys.npy", self.key)
        else:
            np.savez(out, qpos=self.Q, key=self.key, fps=1.0 / self.dt)
        return out

    def scan(self) -> list[str]:
        if self.stale:
            self.run_check()
        r = self.report
        lines = [f"结论 {r['verdict']}"]
        lines += [f"  不可行 {a:.2f}-{b:.2f} s" for a, b in r["infeasible_segments_s"]]
        pen = (self.ann["env_penetration_m"] > self.scene.hard_tol) | (self.ann["self_penetration_m"] > self.scene.hard_tol)
        lines += [f"  穿模 {a:.2f}-{b:.2f} s" for a, b in _runs(pen, self.dt)]
        over = (np.abs(self.ann["joint_torque"]) > self.robot.tau_max).any(1)
        lines += [f"  力矩超限 {a:.2f}-{b:.2f} s" for a, b in _runs(over, self.dt)]
        return lines


def _runs(mask: np.ndarray, dt: float) -> list[tuple[float, float]]:
    out, t, n = [], 0, len(mask)
    while t < n:
        if not mask[t]:
            t += 1
            continue
        t1 = t
        while t1 + 1 < n and mask[t1 + 1]:
            t1 += 1
        out.append((t * dt, (t1 + 1) * dt))
        t = t1 + 1
    return out
