"""
Q1 执行器参数化模型(训练侧 / 分析脚本共用)。

模型: 关节 PD + 前馈 → 期望力矩 τ_cmd; 实际输出受"力矩–转速包络"限制
    |τ| ≤ τ_max                          |ω| ≤ ω_knee
    |τ| ≤ τ_max · (ω_max-|ω|)/(ω_max-ω_knee)   ω_knee < |ω| < ω_max     (线性降到 0)
    且同向驱动时才受限(反向制动可到 τ_max, 四象限简化)
armature(关节侧折算转子惯量)、库仑/粘滞摩擦作为 MuJoCo 关节属性给。

参数来源见 prior.py; 真机辨识(fit.py)后用 motors.npz 覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

HIP_KNEE_WAIST = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "waist_yaw", "waist_roll")
ANKLE_ARM = ("ankle_pitch", "ankle_roll", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")


def joint_class(name: str) -> str:
    key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
    if key in HIP_KNEE_WAIST:
        return "A"       # 35 N·m / 36 rad/s 模组
    if key in ANKLE_ARM:
        return "B"       # 22 N·m / 32 rad/s 模组
    raise KeyError(name)


@dataclass
class ActuatorParams:
    """单个关节的执行器参数(名义值)"""
    tau_max: float            # 峰值力矩 N·m (电机/固件钳位, 短时)
    tau_cont: float           # 持续力矩估计 N·m (热限制, 秒级以上)
    omega_max: float          # 空载转速 rad/s
    omega_knee_frac: float    # 恒力矩区上界 / omega_max
    armature: float           # 关节侧折算转子惯量 kg·m²
    friction_c: float = 0.0   # 库仑摩擦 N·m
    friction_b: float = 0.0   # 粘滞摩擦 N·m/(rad/s)

    def torque_limit(self, omega: np.ndarray, tau_cmd: np.ndarray | None = None) -> np.ndarray:
        """给定关节速度, 返回可用力矩上界(标量或数组)。tau_cmd 给了则只在驱动象限受限"""
        w = np.abs(np.asarray(omega, float))
        wk = self.omega_knee_frac * self.omega_max
        lim = np.where(w <= wk, self.tau_max,
                       self.tau_max * np.clip((self.omega_max - w) / max(self.omega_max - wk, 1e-6), 0, 1))
        if tau_cmd is not None:
            braking = np.sign(tau_cmd) * np.sign(omega) < 0
            lim = np.where(braking, self.tau_max, lim)
        return lim

    def clamp(self, tau_cmd, omega):
        lim = self.torque_limit(omega, tau_cmd)
        return np.clip(tau_cmd, -lim, lim)


# ------------------------------------------------------------------ 名义值与随机化范围(无真机先验, 见 prior.py)
# 公开资料没有额定力矩, 按一体化关节模组常见的峰值 / 额定 ≈ 3 估计
PEAK_TO_RATED = 3.0
NOMINAL = {
    # τ_max: URDF effort / MJCF ctrlrange 给 36 / 22, 髋膝腰保守取 35
    # ω_max / armature: URDF velocity + q1_v3_mc.xml armature 0.004
    "A": ActuatorParams(tau_max=35.0, tau_cont=35.0 / PEAK_TO_RATED, omega_max=36.0, omega_knee_frac=0.5, armature=0.004),
    "B": ActuatorParams(tau_max=22.0, tau_cont=22.0 / PEAK_TO_RATED, omega_max=32.0, omega_knee_frac=0.5, armature=0.004),
}

# 域随机化: (low, high), 均匀抽; 峰值力矩上界取名义值, 下界留给固件降额 / 低电压
DR_RANGES = {
    "A": dict(tau_max=(30.0, 35.0), omega_max=(28.0, 40.0), omega_knee_frac=(0.35, 0.7),
              armature=(0.002, 0.008), friction_c=(0.0, 0.6), friction_b=(0.0, 0.08)),
    "B": dict(tau_max=(18.0, 22.0), omega_max=(25.0, 36.0), omega_knee_frac=(0.35, 0.7),
              armature=(0.002, 0.008), friction_c=(0.0, 0.4), friction_b=(0.0, 0.05)),
}


def sample(cls: str, rng: np.random.Generator) -> ActuatorParams:
    base = asdict(NOMINAL[cls])
    for k, (lo, hi) in DR_RANGES[cls].items():
        base[k] = float(rng.uniform(lo, hi))
    base["tau_cont"] = base["tau_max"] * NOMINAL[cls].tau_cont / NOMINAL[cls].tau_max
    return ActuatorParams(**base)


def for_joints(names, rng: np.random.Generator | None = None) -> dict[str, ActuatorParams]:
    """按关节名返回参数字典; rng=None 给名义值"""
    out = {}
    for n in names:
        c = joint_class(n)
        out[n] = NOMINAL[c] if rng is None else sample(c, rng)
    return out


def load_npz(path: str) -> dict[str, ActuatorParams]:
    """读 prior.py / fit.py 写的 motors*.npz(字段: names, tau_max, tau_cont, omega_max, omega_knee_frac, armature, friction_c, friction_b)"""
    z = np.load(path, allow_pickle=False)
    out = {}
    for i, n in enumerate(map(str, z["names"])):
        out[n] = ActuatorParams(
            tau_max=float(z["tau_max"][i]), tau_cont=float(z["tau_cont"][i]),
            omega_max=float(z["omega_max"][i]), omega_knee_frac=float(z["omega_knee_frac"][i]),
            armature=float(z["armature"][i]),
            friction_c=float(z["friction_c"][i]) if "friction_c" in z else 0.0,
            friction_b=float(z["friction_b"][i]) if "friction_b" in z else 0.0,
        )
    return out
