"""Q1 关节名与已知力矩上限(多源对照)。辨识结果覆盖这些默认值。"""
from __future__ import annotations

# 全身铰接关节(与 q1_v3.xml 一致)
LEGS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST = ["waist_yaw_joint", "waist_roll_joint"]
ARMS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]
ALL = LEGS + WAIST + ARMS

# 关键关节: 髋 / 膝 pitch 承重最大, 踝峰值力矩最小
CRITICAL = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
]

# 辨识前的力矩上限参考; 辨识后写进 motors.npz 覆盖
#   URDF effort / MJCF actuatorfrcrange: 髋膝 36 / 踝 22
#   早期保守值: 髋膝 33 / 踝 22
# 名义值见 actuator.NOMINAL(prior.py)
EFFORT_SCHEME = {
    "scheme": 33.0,          # 早期保守值: 髋膝
    "scheme_ankle": 22.0,
    "mjcf": 36.0,            # MJCF: 髋膝
    "mjcf_ankle": 22.0,
}


def effort_guess(name: str, source: str = "scheme") -> float:
    ankle = "ankle" in name
    if source == "scheme":
        return EFFORT_SCHEME["scheme_ankle"] if ankle else EFFORT_SCHEME["scheme"]
    if source == "mjcf":
        return EFFORT_SCHEME["mjcf_ankle"] if ankle else EFFORT_SCHEME["mjcf"]
    raise KeyError(source)


def short(name: str) -> str:
    return name.replace("_joint", "")
