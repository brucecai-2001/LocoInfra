"""执行后端：读传感器（关节编码器、骨盆 IMU）、写 PD 目标、推进一个控制周期。

MuJoCoBackend 用权重包里的训练场景（CPU MuJoCo）；PrimeBotBackend 走 PrimeBot SDK 的 ROS2 话题接 Q1 真机，约定同一套方法。
接其他机器人时实现同一套 Backend 方法即可（读关节编码器与骨盆 IMU、按 50 Hz 下发 PD 目标、急停切阻尼），控制循环与策略推理不用改。
"""

from __future__ import annotations

import atexit
import os
import re
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from runtime import primebot_io as io
from runtime.package import MESH_DIR, Package


@dataclass
class Sensors:
    joint_pos: np.ndarray  # rad，MJCF 关节顺序
    joint_vel: np.ndarray  # rad/s
    base_ang_vel: np.ndarray  # 骨盆 IMU 角速度，骨盆系
    projected_gravity: np.ndarray  # 骨盆系重力方向

    def obs_frame(self, q0: np.ndarray, last_action: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "joint_pos": self.joint_pos - q0,
            "joint_vel": self.joint_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "actions": last_action,
        }


class Backend:
    def reset(self) -> None:
        raise NotImplementedError

    def sense(self) -> Sensors:
        raise NotImplementedError

    def step(self, joint_target: np.ndarray) -> None:
        """写 PD 目标并推进一个控制周期（真机：等到下一个周期）。"""
        raise NotImplementedError

    def damp(self) -> None:
        """急停：切阻尼模式（kp = 0）。"""
        raise NotImplementedError

    def set_pd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """换 PD 增益（MJCF 关节顺序），下一个控制周期起生效；缺省为后端所用权重包的 deploy.yaml pd。"""
        raise NotImplementedError


class MuJoCoBackend(Backend):
    def __init__(self, pkg: Package):
        self.pkg = pkg
        self.model = m = _load_scene(pkg)
        o = pkg.spec["sim"]
        m.opt.timestep, m.opt.integrator, m.opt.cone = o["timestep"], o["integrator"], o["cone"]
        m.opt.solver, m.opt.iterations, m.opt.ls_iterations = o["solver"], o["iterations"], o["ls_iterations"]
        for g in range(m.ngeom):
            if re.fullmatch(r"robot/(left|right)_sole_\d+", m.geom(g).name or ""):
                m.geom_friction[g, 0] = o["foot_friction"]
        self.data = mujoco.MjData(m)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}") for n in pkg.joint_names]
        self.q_adr = np.array([m.jnt_qposadr[j] for j in jids])
        self.v_adr = np.array([m.jnt_dofadr[j] for j in jids])
        self.ctrl = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{n}") for n in pkg.joint_names])
        free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.fq, self.fv = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.hands = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}") for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
        self.decim = int(round(pkg.control_dt / m.opt.timestep))
        self.kp_nominal = m.actuator_gainprm[self.ctrl, 0].copy()
        self.bias_nominal = m.actuator_biasprm[self.ctrl, :3].copy()

    def reset(self) -> None:
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        m.actuator_gainprm[self.ctrl, 0] = self.kp_nominal
        m.actuator_biasprm[self.ctrl, :3] = self.bias_nominal
        d.qpos[self.fq : self.fq + 3] = (0.0, 0.0, self.pkg.stand_height + 0.01)
        d.qpos[self.fq + 3 : self.fq + 7] = (1.0, 0.0, 0.0, 0.0)
        q0 = self.pkg.default_joint_pos
        d.qpos[self.q_adr] = q0
        d.ctrl[self.ctrl] = q0
        mujoco.mj_forward(m, d)

    def sense(self) -> Sensors:
        d = self.data
        R = d.xmat[self.pelvis].reshape(3, 3)
        return Sensors(
            joint_pos=d.qpos[self.q_adr].copy(),
            joint_vel=d.qvel[self.v_adr].copy(),
            base_ang_vel=d.qvel[self.fv + 3 : self.fv + 6].copy(),
            projected_gravity=R.T @ np.array([0.0, 0.0, -1.0]),
        )

    def step(self, joint_target: np.ndarray) -> None:
        self.data.ctrl[self.ctrl] = joint_target
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)

    def damp(self) -> None:
        self.model.actuator_gainprm[self.ctrl, 0] = 0.0
        self.model.actuator_biasprm[self.ctrl, 1] = 0.0

    def set_pd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """位置执行器 force = kp (ctrl - q) - kd qd：gainprm[0] = kp，biasprm[1:3] = (-kp, -kd)。"""
        m = self.model
        m.actuator_gainprm[self.ctrl, 0] = kp
        m.actuator_biasprm[self.ctrl, 1] = -np.asarray(kp)
        m.actuator_biasprm[self.ctrl, 2] = -np.asarray(kd)

    # ---- 仿真里才有的真值与操作（显示、摔倒判定、评测）
    def set_state(self, root_pos, root_quat, joint_pos, root_lin_vel_w=None, root_ang_vel_w=None, joint_vel=None) -> None:
        """把机器人摆到给定状态（如参考首帧）；速度缺省为 0，PD 目标设为当前关节角，PD 增益恢复（解除阻尼）。"""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        self.model.actuator_gainprm[self.ctrl, 0] = self.kp_nominal
        self.model.actuator_biasprm[self.ctrl, :3] = self.bias_nominal
        d.qpos[self.fq : self.fq + 3] = root_pos
        d.qpos[self.fq + 3 : self.fq + 7] = root_quat
        d.qpos[self.q_adr] = joint_pos
        if root_lin_vel_w is not None:
            d.qvel[self.fv : self.fv + 3] = root_lin_vel_w
        if root_ang_vel_w is not None:
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, np.asarray(root_quat, dtype=np.float64))
            d.qvel[self.fv + 3 : self.fv + 6] = Rq.reshape(3, 3).T @ np.asarray(root_ang_vel_w)
        if joint_vel is not None:
            d.qvel[self.v_adr] = joint_vel
        d.ctrl[self.ctrl] = joint_pos
        mujoco.mj_forward(self.model, d)

    def torso_quat(self) -> np.ndarray:
        """躯干朝向（wxyz）；真机上由骨盆 IMU 加腰关节正运动学得到。"""
        return self.data.xquat[self.torso].copy()

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis][2])

    def upright(self) -> float:
        return float(self.data.xmat[self.pelvis].reshape(3, 3)[2, 2])

    def base_velocity(self) -> np.ndarray:
        d = self.data
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        v = d.qvel[self.fv : self.fv + 3]
        w_world = R @ d.qvel[self.fv + 3 : self.fv + 6]
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], w_world[2]])

    def head_frame(self, head_offset) -> tuple[np.ndarray, np.ndarray]:
        """头位置与躯干航向系（3×3，列为航向系轴）。"""
        R = self.data.xmat[self.torso].reshape(3, 3)
        head = self.data.xpos[self.torso] + R @ np.asarray(head_offset)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        return head, np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def hand_positions(self, head_offset) -> np.ndarray:
        """(2, 3) 双手相对头、躯干航向系（与 hand_L / hand_R 槽位同定义）。"""
        head, Rh = self.head_frame(head_offset)
        return np.stack([Rh.T @ (self.data.xpos[b] - head) for b in self.hands])


class PrimeBotBackend(Backend):
    """PrimeBot 真机后端（ROS2 / rclpy）：订阅 /aima/hal/joint/state 与 /aima/hal/imu/state，
    发布 /aima/hal/joint/command。

    前置：APP 开发者模式选「低层运控开发」并关掉高层运控；吊带兜住；source ROS2 Humble 与 aimdk_msgs。

    收、发各一个子进程（primebot_io.py，各绑一个大核、SCHED_FIFO），与策略进程经共享内存交换。
    下发进程以 cmd_hz（默认 500 Hz）持续发全身指令：SDK 要求 500–1000 Hz，低于 100 Hz 会触发电机看门狗。
    策略目标在两次 step() 之间零阶保持（与训练一致），每帧位置增量限 max_delta（SDK 上限 0.2 rad）。
    reset() 之前不发任何指令；关节或 IMU 断流超过 stale_s、策略超过 policy_timeout_s 没调 step()、
    或策略进程退出，自动切阻尼，下一次 step() 抛 RuntimeError。阻尼即 SDK 规定的 stiffness = 0、damping = 5。
    """

    DAMP_KD = io.DAMP_KD

    def __init__(self, pkg: Package, cmd_hz: float = 500.0, ramp_s: float = 3.0, max_delta: float = 0.2,
                 stale_s: float = 0.1, policy_timeout_s: float = 0.5,
                 sub_core: int | None = 4, pub_core: int | None = 5, io_prio: int | None = 70,
                 cmd_topic: str = "/aima/hal/joint/command", ramp_track_lim: float = 0.3, track_lim: float = np.inf):
        """cmd_topic 换成别的话题即演练：全流程照跑，电机收不到指令。
        任一关节实际位置与下发指令差超过限值即切阻尼：reset() 插值期间用 ramp_track_lim，之后用 track_lim
        （起身时力矩大，kp 30 下差近 1 rad 是正常的，所以运行期默认不限）。"""
        import multiprocessing as mp

        self.pkg = pkg
        self.names = list(pkg.joint_names)
        self.q0 = pkg.default_joint_pos
        self.ramp_s = ramp_s
        self.ramp_track_lim, self.track_lim = ramp_track_lim, track_lim

        # 关节限位与躯干正运动学用权重包里的训练场景
        self.model = m = _load_scene(pkg)
        self.data = mujoco.MjData(m)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}") for n in self.names]
        self.q_adr = np.array([m.jnt_qposadr[j] for j in jids])
        self.q_lo, self.q_hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.fq = m.jnt_qposadr[free]
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")

        ctx = mp.get_context("spawn")
        self.L = L = io.Layout(len(self.names))
        self._arr = ctx.Array("d", L.size)
        self._buf = io.view(self._arr)
        self._buf[L.quat] = (1.0, 0.0, 0.0, 0.0)
        self._buf[L.mode] = io.IDLE
        self._buf[L.hw_idx] = -1
        self._buf[L.track_lim] = track_lim
        self._buf[L.kp], self._buf[L.kd] = self.pkg.kp, self.pkg.kd
        self._closed = False
        self._t_next = 0.0
        self.overruns = 0  # step() 超出 50 Hz 节拍的次数（推理太慢）

        parent = os.getpid()
        self._procs = [
            ctx.Process(target=io.sub_worker, args=(self._arr, self.names, sub_core, io_prio, parent), daemon=True),
            ctx.Process(target=io.pub_worker, args=(self._arr, self.names, cmd_topic, cmd_hz, max_delta, stale_s, policy_timeout_s,
                                                    pub_core, io_prio, parent), daemon=True),
        ]
        for p in self._procs:
            p.start()
        atexit.register(self.close)

    def _fault(self) -> str | None:
        code = int(self._buf[self.L.fault][0])
        return io.FAULTS.get(code, f"故障 {code}") if code else None

    def rates(self) -> tuple[int, int]:
        """累计收到的关节状态帧数、发出的指令帧数。"""
        L = self.L
        with self._arr.get_lock():
            return int(self._buf[L.n_joint][0]), int(self._buf[L.n_pub][0])

    def worst_tracking(self) -> tuple[float, str]:
        """reset() 以来关节实际位置与下发指令的最大差（rad）及关节名。"""
        with self._arr.get_lock():
            d, k = self._buf[self.L.worst]
        return float(d), self.names[int(k)]

    # ---- Backend 接口
    def wait_sensors(self, timeout: float = 10.0) -> None:
        L, t_end = self.L, time.monotonic() + timeout
        while self._buf[L.t_joint][0] == 0 or self._buf[L.t_imu][0] == 0:
            if self._fault():
                raise RuntimeError(self._fault())
            if not all(p.is_alive() for p in self._procs):
                raise RuntimeError("ROS 收发进程退出：检查 source ROS2 Humble 与 aimdk_msgs")
            if time.monotonic() > t_end:
                raise TimeoutError("没收到 /aima/hal/joint/state 或 /aima/hal/imu/state：检查开发者模式、网线与 ROS_DOMAIN_ID")
            time.sleep(0.01)

    def reset(self, joint_pos: np.ndarray | None = None) -> None:
        """从当前姿态用 ramp_s 秒线性插值到 joint_pos（缺省为 default_joint_pos），之后保持，等第一次 step()。"""
        self.wait_sensors()
        goal = np.clip(self.q0 if joint_pos is None else np.asarray(joint_pos, dtype=np.float64), self.q_lo, self.q_hi)
        dt, L, b = self.pkg.control_dt, self.L, self._buf
        with self._arr.get_lock():
            start = b[L.q].copy()
            b[L.fault] = 0
            b[L.cmd] = start
            b[L.target] = start
            b[L.t_step] = time.monotonic()
            b[L.worst] = (0.0, 0)
            b[L.track_lim] = self.ramp_track_lim
            b[L.mode] = io.HOLD
        steps = max(1, int(round(self.ramp_s / dt)))
        t_next = time.monotonic()
        for k in range(1, steps + 1):
            self._set_target(start + (goal - start) * (k / steps))
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
        with self._arr.get_lock():
            b[L.track_lim] = self.track_lim
        self._t_next = time.monotonic()

    def sense(self) -> Sensors:
        L, b = self.L, self._buf
        with self._arr.get_lock():
            q, qd, gyro, quat = b[L.q].copy(), b[L.qd].copy(), b[L.gyro].copy(), b[L.quat].copy()
        return Sensors(joint_pos=q, joint_vel=qd, base_ang_vel=gyro, projected_gravity=_quat_to_mat(quat).T @ np.array([0.0, 0.0, -1.0]))

    def step(self, joint_target: np.ndarray) -> None:
        self._set_target(np.clip(np.asarray(joint_target, dtype=np.float64), self.q_lo, self.q_hi))
        self._t_next += self.pkg.control_dt
        dt = self._t_next - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        else:
            self.overruns += 1
            self._t_next = time.monotonic()

    def _set_target(self, target: np.ndarray) -> None:
        L, b = self.L, self._buf
        with self._arr.get_lock():
            fault, mode = self._fault(), b[L.mode][0]
            if fault:
                raise RuntimeError(f"PrimeBot 已切阻尼：{fault}")
            if mode != io.HOLD:
                raise RuntimeError("PrimeBot 不在 PD 跟踪状态，先调 reset()")
            b[L.target] = target
            b[L.t_step] = time.monotonic()

    def damp(self) -> None:
        with self._arr.get_lock():
            if self._buf[self.L.mode][0] != io.IDLE:
                self._buf[self.L.mode] = io.DAMP

    def set_pd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        with self._arr.get_lock():
            self._buf[self.L.kp], self._buf[self.L.kd] = kp, kd

    def close(self, damp_s: float = 0.5) -> None:
        """切阻尼并持续发 damp_s 秒，再停收发进程（SDK 要求退出前先进阻尼态，不能直接断流）。"""
        if self._closed:
            return
        self._closed = True
        if self._buf[self.L.mode][0] != io.IDLE:
            self.damp()
            time.sleep(damp_s)
        self._buf[self.L.stop] = 1
        for p in self._procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> "PrimeBotBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 真机可得的派生量
    def torso_quat(self) -> np.ndarray:
        """躯干朝向（wxyz）：骨盆 IMU 姿态 + 腰关节正运动学（IMU 坐标系与骨盆一致，不用加旋转）。"""
        with self._arr.get_lock():
            q, quat = self._buf[self.L.q].copy(), self._buf[self.L.quat].copy()
        d = self.data
        d.qpos[self.fq : self.fq + 3] = 0.0
        d.qpos[self.fq + 3 : self.fq + 7] = quat
        d.qpos[self.q_adr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.xquat[self.torso].copy()

    def upright(self) -> float:
        return float(-self.sense().projected_gravity[2])


def _load_scene(pkg: Package) -> mujoco.MjModel:
    xml = (pkg.root / "scene.xml").read_text()
    xml = re.sub(r"<compiler([^>]*)/>", lambda m: f'<compiler{m.group(1)} meshdir="{MESH_DIR}/"/>', xml, count=1)
    return mujoco.MjModel.from_xml_string(xml)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return R.reshape(3, 3)
