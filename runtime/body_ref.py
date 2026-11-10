"""由参考动作（SuperMimic motion.npz）拼参考输入：基础模型包按 body_ref 布局（E_ref），单段动作包按 goal 布局。

每帧特征（54）：参考关节角 − 默认站姿、关节角速度、骨盆系重力方向、骨盆角速度（骨盆系）、
航向系骨盆线速度、骨盆高度。机器人处在参考第 i 帧时喂第 i+1 … i+K 帧（超出用末帧），
再接躯干相对朝向 6 维：R = R_robot^T R_ref（参考取第 i+1 帧）的前两列按行展开。
开始播放时把参考绕 z 轴转到与机器人当前躯干航向一致（IMU 航向无绝对基准）。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from runtime.package import Package


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def yaw_of(q: np.ndarray) -> float:
    R = quat_to_mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return out


class BodyRef:
    def __init__(self, pkg: Package, motion_npz: Path, root_body: str = "pelvis", anchor_body: str = "torso_link"):
        spec = pkg.ref_spec
        self.K = int(spec["future_frames"])
        self.anchor = bool(spec.get("anchor_ori"))
        d = np.load(motion_npz)
        names = [str(b) for b in d["body_names"]]
        r, a = names.index(root_body), names.index(anchor_body)
        q = d["body_quat_w"][:, r]
        Rs = np.stack([quat_to_mat(x) for x in q])
        g = np.array([0.0, 0.0, -1.0])
        grav = np.einsum("tji,j->ti", Rs, g)
        ang_b = np.einsum("tji,tj->ti", Rs, d["body_ang_vel_w"][:, r])
        lin = d["body_lin_vel_w"][:, r]
        yaw = np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        lin_h = np.stack([c * lin[:, 0] + s * lin[:, 1], -s * lin[:, 0] + c * lin[:, 1], lin[:, 2]], axis=-1)
        self.feat = np.concatenate(
            [d["joint_pos"] - pkg.default_joint_pos, d["joint_vel"], grav, ang_b, lin_h, d["body_pos_w"][:, r, 2:3]], axis=-1
        ).astype(np.float32)
        self.anchor_quat = d["body_quat_w"][:, a]
        self.root_pos, self.root_quat = d["body_pos_w"][:, r], q
        self.joint_pos, self.joint_vel = d["joint_pos"], d["joint_vel"]
        self.root_lin_vel, self.root_ang_vel_w = lin, d["body_ang_vel_w"][:, r]
        self.T = self.feat.shape[0]
        self.align = np.array([1.0, 0.0, 0.0, 0.0])

    def start(self, robot_torso_quat: np.ndarray, frame: int = 0) -> None:
        """把参考航向对齐到机器人当前躯干航向（从第 ``frame`` 帧开始播放时调用）。

        取使 Rz(yaw) R_ref 最接近 R_robot 的 yaw，而不是比较两边 x 轴的水平投影：趴着时躯干 x 轴几乎竖直，
        投影方向由几度的倾斜决定，航向对错几十度，策略起身时会转身去纠正（真机实测多转 60°+）。"""
        M = quat_to_mat(robot_torso_quat) @ quat_to_mat(self.anchor_quat[frame]).T
        self.align = yaw_quat(float(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])))

    def goal(self, i: int, robot_torso_quat: np.ndarray) -> np.ndarray:
        idx = np.minimum(np.arange(i + 1, i + 1 + self.K), self.T - 1)
        g = self.feat[idx].reshape(-1)
        if not self.anchor:
            return g
        Rr, Rf = quat_to_mat(robot_torso_quat), quat_to_mat(quat_mul(self.align, self.anchor_quat[idx[0]]))
        return np.concatenate([g, (Rr.T @ Rf)[:, :2].reshape(-1)]).astype(np.float32)
