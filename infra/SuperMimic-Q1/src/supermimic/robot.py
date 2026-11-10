"""Robot selection (env var ``SUPERMIMIC_ROBOT`` = ``g1`` | ``q1``, default ``g1``).

Every dimension that used to be hard-coded for the 29-DoF G1 is derived from ``NJ`` here:

    qpos NQ = 7 + NJ, qvel NV = 6 + NJ, state STATE_DIM = NQ + NV, world-model delta STATE_DIM - 1,
    proprioception OBS_DIM = 3 NJ + 6, goal feature per frame GOAL_FEAT_DIM = 2 NJ + 10,
    sim context XI_DIM = 1 (foot friction) + NJ (encoder bias) + 3 (COM offset) + XI_EXTRA_NAMES
    (per-environment physics scales and latencies, see ``envs/dr.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco

ROBOT_NAME = os.environ.get("SUPERMIMIC_ROBOT", "g1").lower()
# Q1 PD-gain preset (env var ``Q1_PD``, presets in robots/q1.py). Checkpoints record it; evaluation and
# export refuse a checkpoint trained under another preset (check_pd_preset).
PD_PRESET = os.environ.get("Q1_PD", "default") if ROBOT_NAME == "q1" else "default"


def check_pd_preset(ck: dict, strict: bool = True) -> None:
    trained = ck.get("pd_preset", "default")
    if trained == PD_PRESET:
        return
    msg = f"checkpoint trained with PD preset {trained!r}, current Q1_PD={PD_PRESET!r}"
    if strict:
        raise RuntimeError(msg + f": run with Q1_PD={trained}")
    print(f"[WARN] {msg} (continuing training under the current gains)")

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# MJCF order of q1_v3_collision.xml (waist roll before yaw, unlike the G1).
Q1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
)

# Same link names on both robots.
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    xml_path: Callable[[], Path]
    robot_cfg: Callable[[], object]  # -> mjlab EntityCfg
    action_scale: Callable[[], dict[str, float]]
    foot_geom_regex: str
    com_body: str
    com_ranges: dict[int, tuple[float, float]]
    root_height_threshold: float  # pelvis height deviation that terminates an episode (m)
    root_tilt_threshold: float = 0.8
    tracked_bodies: tuple[str, ...] = TRACKED_BODIES
    anchor_body: str = "torso_link"
    root_body: str = "pelvis"
    viewer_distance: float = 2.8
    nconmax: int = 35
    njmax: int = 250
    rated_torque: Callable[[], dict[str, float]] | None = None  # regex -> N·m; None: use the effort limit
    hand_bodies: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _g1_xml() -> Path:
    import mjlab

    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _g1_cfg():
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _g1_action_scale():
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    return G1_ACTION_SCALE


def _q1_xml() -> Path:
    from supermimic.robots import q1

    return q1.Q1_XML


def _q1_cfg():
    from supermimic.robots import q1

    return q1.get_q1_robot_cfg()


def _q1_action_scale():
    from supermimic.robots import q1

    return q1.Q1_ACTION_SCALE


def _q1_rated_torque():
    from supermimic.robots import q1

    return q1.Q1_RATED_TORQUE


SPECS: dict[str, RobotSpec] = {
    "g1": RobotSpec(
        name="g1",
        joint_names=G1_JOINT_NAMES,
        xml_path=_g1_xml,
        robot_cfg=_g1_cfg,
        action_scale=_g1_action_scale,
        foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
        com_body="torso_link",
        com_ranges={0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
        root_height_threshold=0.25,
    ),
    "q1": RobotSpec(
        name="q1",
        joint_names=Q1_JOINT_NAMES,
        xml_path=_q1_xml,
        robot_cfg=_q1_cfg,
        action_scale=_q1_action_scale,
        foot_geom_regex=r"^(left|right)_sole_[0-3]$",
        com_body="torso_link",
        com_ranges={0: (-0.015, 0.015), 1: (-0.02, 0.02), 2: (-0.02, 0.02)},
        # Pelvis stands at ~0.43 m (G1 0.79 m with a 0.25 m threshold).
        root_height_threshold=0.15,
        viewer_distance=1.6,
        nconmax=60,
        njmax=300,
        rated_torque=_q1_rated_torque,
    ),
}

if ROBOT_NAME not in SPECS:
    raise ValueError(f"SUPERMIMIC_ROBOT={ROBOT_NAME!r}; expected one of {sorted(SPECS)}")
SPEC = SPECS[ROBOT_NAME]

NJ = len(SPEC.joint_names)
NQ = 7 + NJ
NV = 6 + NJ
STATE_DIM = NQ + NV
DELTA_DIM = STATE_DIM - 1
OBS_DIM = 3 * NJ + 6
GOAL_FEAT_DIM = 2 * NJ + 10
XI_BASE_DIM = 1 + NJ + 3
# Extra context entries with their nominal (no randomization) values.
XI_EXTRA_NAMES: tuple[str, ...] = (
    "mass_scale", "kp_scale", "kd_scale", "effort_scale", "armature_scale", "joint_friction", "action_delay", "obs_delay",
)
XI_EXTRA_NOMINAL: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
XI_DIM = XI_BASE_DIM + len(XI_EXTRA_NAMES)


def load_robot_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Standalone CPU MjModel of the selected robot; asserts the hinge order matches ``SPEC.joint_names``."""
    model = mujoco.MjModel.from_xml_path(str(xml_path or SPEC.xml_path()))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
    assert tuple(names[1:]) == SPEC.joint_names, f"joint order mismatch: {names[1:]}"
    assert model.nq == NQ and model.nv == NV
    return model
