"""遥操作适配器的公共部分：命令 (vx, vy, wz, Δh, pitch, roll) 的范围、加速度 / 变化率限幅，以及填进 E_tele 的槽位。"""

from __future__ import annotations

import numpy as np

from runtime import TeleGoal

# 满速（m/s, m/s, rad/s）与加速度上限；高度 / 躯干设定值的范围与变化率
FWD, BACK, SIDE, TURN = 0.6, 0.4, 0.3, 1.0
# 侧移起步速度（不随 speed_scale 缩放）：E_tele 对 |vy| ≲ 0.15 只侧倾不迈步（仿真 0.1 → 实测 0.02）
SIDE_MIN = 0.2
# 运行中十字键 ← / → 调的速度系数：1.5 时各方向满速仍在 E_tele 的训练指令范围内
SPEED_RANGE, SPEED_STEP = (0.2, 1.5), 0.1
ACC = np.array([1.0, 1.0, 3.0])
DH_RANGE, DH_RATE = (-0.12, 0.0), 0.15
PITCH_RANGE, ROLL_RANGE, LEAN_RATE = (0.0, 0.45), (-0.2, 0.2), 0.5


def torso_gravity(pitch: float, roll: float) -> np.ndarray:
    """与 supermimic/tele/slots.py 相同：R = R_y(pitch) R_x(roll) 下躯干系的重力方向。"""
    return np.array([np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)])


def side_speed(lx: float, scale: float) -> float:
    """摇杆一离开死区就从 SIDE_MIN 起步，推满到 max(scale * SIDE, SIDE_MIN)。"""
    if lx == 0.0:
        return 0.0
    hi = max(scale * SIDE, SIDE_MIN)
    return float(np.sign(lx)) * (SIDE_MIN + (hi - SIDE_MIN) * abs(lx))


def ramp(cur: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    lim = np.concatenate([ACC, [DH_RATE, LEAN_RATE, LEAN_RATE]]) * dt
    return cur + np.clip(target - cur, -lim, lim)


def fill_goal(goal: TeleGoal, cmd: np.ndarray) -> None:
    """base、height、torso 三个槽位有效，双手缺省（掩码 0）。"""
    goal.set("base", cmd[:3])
    goal.set("height", [cmd[3]])
    goal.set("torso", torso_gravity(cmd[4], cmd[5]))
