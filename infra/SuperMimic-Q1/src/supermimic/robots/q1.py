"""Kabutack Q1 (22 DoF) for mjlab, built from ``q1/q1_v3/xml/q1_v3_collision.xml``.

Motor numbers come from ``q1/motor_id/prior.npz``: peak torque 35 N·m (hip / knee / waist, the
public URDF / MJCF give 36) and 22 N·m (ankle / arm), reflected inertia 0.004 kg·m².
Default PD gains: hip / knee 80, waist 60, ankle 40, arm 30 N·m/rad. ``Q1_PD=vendor_getup`` (see
``supermimic.robot.PD_PRESET``) selects the vendor's get-up / lie-down gains instead; further gain sets can be
added to ``PD_PRESETS``.

Collisions: the XML collides every hull mesh with everything; for training the hulls only
touch the ground (no self-collision, the convex hulls overlap near joints) and each foot
contacts through its four sole spheres with condim 3.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from supermimic.robot import PD_PRESET

Q1_XML = Path(
    os.environ.get("Q1_XML", Path(__file__).resolve().parents[5] / "q1" / "q1_v3" / "xml" / "q1_v3_collision.xml")
)

ARMATURE = 0.004
SOLE_REGEX = r"^(left|right)_sole_[0-3]$"
HULL_REGEX = r"^.*_hull_\d+$"


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(Q1_XML))
    # The scene (floor, lights, sky) comes from mjlab; keep only the robot.
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    for mat in list(spec.materials):
        spec.delete(mat)
    for tex in list(spec.textures):
        spec.delete(tex)
    # Name the unnamed collision geoms so CollisionCfg can select them.
    counts: dict[str, int] = {}
    for g in spec.geoms:
        if g.name or (g.contype == 0 and g.conaffinity == 0):
            continue
        body = g.parent.name
        g.name = f"{body}_hull_{counts.get(body, 0)}"
        counts[body] = counts.get(body, 0) + 1
    return spec


def _act(names: tuple[str, ...], kp: float, kd: float, effort: float) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=names, stiffness=kp, damping=kd, effort_limit=effort, armature=ARMATURE
    )


LEGS = (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint")
ANKLES = (".*_ankle_pitch_joint", ".*_ankle_roll_joint")
ARMS = (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint")

PD_PRESETS: dict[str, tuple[BuiltinPositionActuatorCfg, ...]] = {
    "default": (
        _act(LEGS, 80.0, 3.0, 35.0),
        _act(ANKLES, 40.0, 1.5, 22.0),
        _act(("waist_roll_joint", "waist_yaw_joint"), 60.0, 2.5, 35.0),
        _act(ARMS, 30.0, 1.0, 22.0),
    ),
    # Low get-up / lie-down gains of the Q1 vendor; weights/getup_vpd_ft is trained with them.
    "vendor_getup": (
        _act(LEGS, 30.0, 1.5, 35.0),
        _act(ANKLES, 20.0, 1.5, 22.0),
        _act(("waist_roll_joint",), 80.0, 3.5, 35.0),
        _act(("waist_yaw_joint",), 30.0, 1.5, 35.0),
        _act(ARMS, 20.0, 1.5, 22.0),
    ),
}
if PD_PRESET not in PD_PRESETS:
    raise ValueError(f"Q1_PD={PD_PRESET!r}; expected one of {sorted(PD_PRESETS)}")
Q1_ACTUATORS = PD_PRESETS[PD_PRESET]

Q1_ARTICULATION = EntityArticulationInfoCfg(actuators=Q1_ACTUATORS, soft_joint_pos_limit_factor=0.9)

# Slightly crouched stance; also the action offset (use_default_offset=True).
STAND_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.445),
    joint_pos={
        ".*_hip_pitch_joint": -0.2,
        ".*_knee_joint": 0.4,
        ".*_ankle_pitch_joint": -0.2,
        ".*_elbow_joint": 0.3,
        "left_shoulder_roll_joint": 0.1,
        "right_shoulder_roll_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

GROUND_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(SOLE_REGEX, HULL_REGEX),
    contype=0,
    conaffinity=1,
    condim={SOLE_REGEX: 3, ".*": 1},
    priority={SOLE_REGEX: 1, ".*": 0},
    friction={SOLE_REGEX: (0.6,)},
)


def get_q1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=STAND_KEYFRAME,
        collisions=(GROUND_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=Q1_ARTICULATION,
    )


# Continuous (rated) torque, N·m: not published, estimated as 1/3 of peak (q1/motor_id/actuator.py).
Q1_RATED_TORQUE: dict[str, float] = {
    r".*_(hip_.*|knee)_joint": 35.0 / 3,
    r"waist_.*_joint": 35.0 / 3,
    r".*_ankle_.*_joint": 22.0 / 3,
    r".*_(shoulder_.*|elbow)_joint": 22.0 / 3,
}

# A unit action moves the PD target 0.25 rad from the stance (G1's 0.25·τmax/Kp is 0.35–0.55 rad
# with its stiff-by-armature gains; Q1's gains are set from hardware practice instead).
Q1_ACTION_SCALE: dict[str, float] = {".*": 0.25}
