"""生成 SMPL-X → Q1 的 GMR IK 配置（q1_smplx_ik_config.json）。

跟踪的 Q1 刚体、位置 / 朝向权重、刚体系位置偏移与 LAFAN 配置（q1_ik_config.json）相同，只换人体关节名；
旋转偏移按定义解析计算：人和 Q1 摆同一姿态时 offset = R_human⁻¹ · R_robot。
SMPL-X 零姿态下每个关节的全局朝向都等于身体朝向（canonical：x 左、y 上、z 前），人面朝 +x、z 朝上时
R_human = R_FACE。零姿态近似 T-pose 但不标准（上臂下垂约 16°、大腿外展约 8°），所以 Q1 先摆 T-pose
（全零关节角 + 肩 roll ±90°），再对上臂、大腿、小腿各自乘一个最小旋转，把 Q1 的骨段方向转到人体静止
姿态的骨段方向（由 SMPL-X 中性模型算出），得到与人体零姿态一致的 R_robot。

重新生成（在 infra/ 下，需要 data/body_models/smplx/SMPLX_NEUTRAL.*）：bin/q1py -m q1infra.retarget.smplx_config
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).parent
LAFAN_CONFIG = HERE / "q1_ik_config.json"
SMPLX_CONFIG = HERE / "q1_smplx_ik_config.json"

# SMPL-X canonical 轴 → 世界：x(左)→+y，y(上)→+z，z(前)→+x
R_FACE = R.from_matrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

LAFAN_TO_SMPLX = {
    "Hips": "pelvis",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFootMod": "left_ankle",
    "RightFootMod": "right_ankle",
    "Spine2": "spine3",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
}

# 朝向受跟踪的骨段：Q1 刚体 → (人体骨段起点, 终点, Q1 骨段起点, 终点)
SEGMENT_ALIGN = {
    f"{s}_{body}": (f"{s}_{h0}", f"{s}_{h1}", f"{s}_{r0}", f"{s}_{r1}")
    for s in ("left", "right")
    for body, h0, h1, r0, r1 in (
        ("shoulder_yaw_link", "shoulder", "elbow", "shoulder_roll_link", "elbow_link"),
        ("hip_yaw_link", "hip", "knee", "hip_roll_link", "knee_link"),
        ("knee_link", "knee", "ankle", "knee_link", "ankle_roll_link"),
    )
}

# 在 ACCAD 走路 + 手势 + 综合 4 段上扫描，手臂（肩、肘、腕）平均位置误差在 −20° … −30° 最小且平坦
ARM_TWIST_DEG = -25.0

SEGMENT_SCALING = {
    "root": {"human": ["pelvis", "left_ankle"], "robot": ["pelvis", "left_ankle_roll_link"]},
    "anchors": {
        "left_hip": [["pelvis", "pelvis", "left_hip_roll_link"]],
        "right_hip": [["pelvis", "pelvis", "right_hip_roll_link"]],
        "left_shoulder": [["pelvis", "pelvis", "torso_link"], ["spine3", "torso_link", "left_shoulder_roll_link"]],
        "right_shoulder": [["pelvis", "pelvis", "torso_link"], ["spine3", "torso_link", "right_shoulder_roll_link"]],
    },
    "segments": [
        {"bones": ["left_knee", "right_knee"], "human": ["left_hip", "left_knee"], "robot": ["left_hip_roll_link", "left_knee_link"]},
        {"bones": ["left_ankle", "right_ankle", "left_foot", "right_foot"], "human": ["left_knee", "left_ankle"], "robot": ["left_knee_link", "left_ankle_roll_link"]},
        {
            "bones": ["spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"],
            "human": ["pelvis", "left_shoulder"],
            "robot": ["pelvis", "left_shoulder_roll_link"],
        },
        {"bones": ["left_elbow", "right_elbow"], "human": ["left_shoulder", "left_elbow"], "robot": ["left_shoulder_roll_link", "left_elbow_link"]},
        {"bones": ["left_wrist", "right_wrist"], "human": ["left_elbow", "left_wrist"], "robot": ["left_elbow_link", "left_wrist_yaw_link"]},
    ],
}


def q1_tpose(mjcf: Path) -> tuple[dict[str, R], dict[str, np.ndarray]]:
    model = mj.MjModel.from_xml_path(str(mjcf))
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    for joint, angle in (("left_shoulder_roll_joint", np.pi / 2), ("right_shoulder_roll_joint", -np.pi / 2)):
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = angle
    mj.mj_forward(model, data)
    rot = {model.body(i).name: R.from_quat(data.xquat[i], scalar_first=True) for i in range(model.nbody)}
    pos = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}
    return rot, pos


def smplx_rest_joints(model_dir: Path) -> dict[str, np.ndarray]:
    """SMPL-X 中性模型零姿态、零体型的关节位置，转到 Q1 的世界朝向（面朝 +x、z 朝上）。"""
    import smplx
    from smplx.joint_names import JOINT_NAMES

    f = next(model_dir / f"SMPLX_NEUTRAL.{e}" for e in ("npz", "pkl") if (model_dir / f"SMPLX_NEUTRAL.{e}").exists())
    body = smplx.SMPLX(model_path=str(f), ext=f.suffix[1:], use_pca=False)
    j = body().joints[0].detach().numpy()
    return {n: R_FACE.apply(j[i]) for i, n in enumerate(JOINT_NAMES[:22])}


def _swing(a: np.ndarray, b: np.ndarray) -> R:
    """把方向 a 转到方向 b 的最小旋转。"""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R.identity()
    return R.from_rotvec(axis / s * np.arctan2(s, float(np.dot(a, b))))


def make_config(mjcf: Path, model_dir: Path, arm_twist_deg: float = ARM_TWIST_DEG) -> dict:
    base = json.loads(LAFAN_CONFIG.read_text())
    robot_rot, robot_pos = q1_tpose(mjcf)
    human = smplx_rest_joints(model_dir)
    matched = dict(robot_rot)
    for body, (h0, h1, r0, r1) in SEGMENT_ALIGN.items():
        d_human = human[h1] - human[h0]
        rot = _swing(robot_pos[r1] - robot_pos[r0], d_human) * robot_rot[body]
        if "shoulder" in body:
            # 绕上臂轴的扭转：静止姿态定不了肘的屈曲平面，按动作数据标定（左右镜像）
            sign = 1.0 if body.startswith("left") else -1.0
            rot = R.from_rotvec(np.radians(sign * arm_twist_deg) * d_human / np.linalg.norm(d_human)) * rot
        matched[body] = rot
    cfg = {
        "robot_root_name": base["robot_root_name"],
        "human_root_name": "pelvis",
        "ground_height": base["ground_height"],
        "human_height_assumption": base["human_height_assumption"],
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "segment_scaling": SEGMENT_SCALING,
        "human_scale_table": {LAFAN_TO_SMPLX[k]: v for k, v in base["human_scale_table"].items()},
    }
    for table in ("ik_match_table1", "ik_match_table2"):
        cfg[table] = {}
        for body, (human_name, pos_w, rot_w, pos_offset, _rot) in base[table].items():
            offset = (R_FACE.inv() * matched[body]).as_quat(scalar_first=True)
            cfg[table][body] = [LAFAN_TO_SMPLX[human_name], pos_w, rot_w, pos_offset, [round(float(x), 8) for x in offset]]
    return cfg


def main() -> None:
    from q1infra.retarget.gmr_q1 import Q1_MJCF, SMPLX_MODEL_DIR

    SMPLX_CONFIG.write_text(json.dumps(make_config(Q1_MJCF, SMPLX_MODEL_DIR), indent=4, ensure_ascii=False) + "\n")
    print("wrote", SMPLX_CONFIG)


if __name__ == "__main__":
    main()
