"""Slot registry of the teleoperation encoder ``E_tele`` (docs/training.md).

Every slot is a physical target in Q1 scale and Q1 frames; device adapters (PICO, joystick, video)
convert their raw inputs into slots outside the model. Slots are only appended, never edited or
reordered (a removed slot stays as a permanently masked entry), so older weight packages keep
loading. ``deploy.yaml`` and the ONNX input layout are generated from this table.

Frames:
    heading frame   the torso's yaw-only frame (x forward, z up); the robot has no neck, so the
                    headset's yaw frame maps onto it
    head            ``head_link``, rigidly mounted on ``torso_link`` at ``HEAD_OFFSET``
    pelvis height   above the ground; ``H_STAND`` is the standing pelvis height of the motion library

Network input per slot: ``(value - default) / scale``, multiplied by the slot's mask bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.kinematics.rotations import heading_quat, projected_gravity, quat_apply, quat_apply_inverse

REGISTRY_VERSION = 1
HEAD_OFFSET = (0.0, 0.0, 0.2625)  # head_link in torso_link (q1_v3 xml)
H_STAND = 0.434  # m, median pelvis height of mix_v1 (= the baseline's standing virtual reference)
HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class Slot:
    name: str
    dim: int
    unit: str
    desc: str
    default: tuple[float, ...]
    scale: tuple[float, ...]
    reserved: bool = False  # no source yet: always masked in training and deployment


SLOTS: tuple[Slot, ...] = (
    Slot("base", 3, "m/s, m/s, rad/s", "骨盆在航向系的平面速度 vx（前）/ vy（左），世界系偏航角速度 wz（逆时针为正）",
         (0.0, 0.0, 0.0), (1.0, 0.5, 1.5)),
    Slot("height", 1, "m", "骨盆离地高度相对标准站姿 H_STAND 的偏移 Δh（下蹲为负）",
         (0.0,), (0.1,)),
    Slot("torso", 3, "1", "躯干（torso_link）坐标系下的单位重力方向，只含俯仰 / 侧倾（偏航无关）；前倾 x > 0，右倾 y < 0",
         (0.0, 0.0, -1.0), (0.3, 0.3, 0.1)),
    # Defaults: median of the tele_ok frames of mix_v1 (scripts/tele_coverage.py), mirrored.
    Slot("hand_L", 3, "m", "左手（left_wrist_yaw_link 原点）相对头，躯干航向系",
         (0.02, 0.135, -0.30), (0.15, 0.15, 0.15)),
    Slot("hand_R", 3, "m", "右手（right_wrist_yaw_link 原点）相对头，躯干航向系",
         (0.02, -0.135, -0.30), (0.15, 0.15, 0.15)),
    Slot("feet", 6, "m", "左、右脚（*_ankle_roll_link 原点）相对头，躯干航向系；预留给腿部 Tracker",
         (-0.045, 0.105, -0.76, -0.045, -0.105, -0.76), (0.15, 0.15, 0.15, 0.15, 0.15, 0.15), reserved=True),
)

SLOT_NAMES = tuple(s.name for s in SLOTS)
NUM_SLOTS = len(SLOTS)
SLOT_DIM = sum(s.dim for s in SLOTS)
GOAL_DIM = SLOT_DIM + NUM_SLOTS  # policy / critic goal: slots then the mask bits


def _offsets() -> dict[str, slice]:
    out, i = {}, 0
    for s in SLOTS:
        out[s.name] = slice(i, i + s.dim)
        i += s.dim
    return out


SLICE = _offsets()


def default_vector(device=None) -> torch.Tensor:
    return torch.tensor([v for s in SLOTS for v in s.default], device=device)


def scale_vector(device=None) -> torch.Tensor:
    return torch.tensor([v for s in SLOTS for v in s.scale], device=device)


def reserved_mask(device=None) -> torch.Tensor:
    """(NUM_SLOTS,) bool, True for slots that are always masked."""
    return torch.tensor([s.reserved for s in SLOTS], device=device)


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """(…, NUM_SLOTS) -> (…, SLOT_DIM): every slot's bit repeated over its dims."""
    reps = torch.tensor([s.dim for s in SLOTS], device=mask.device)
    return torch.repeat_interleave(mask, reps, dim=-1)


def split_goal(goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(…, GOAL_DIM) -> slots (…, SLOT_DIM), mask (…, NUM_SLOTS)."""
    return goal[..., :SLOT_DIM], goal[..., SLOT_DIM:]


def normalized(goal: torch.Tensor) -> torch.Tensor:
    """Critic input: masked normalized slots followed by the mask bits (…, GOAL_DIM)."""
    slots, mask = split_goal(goal)
    x = (slots - default_vector(goal.device)) / scale_vector(goal.device)
    return torch.cat([x * expand_mask(mask), mask], dim=-1)


def head_position(torso_pos: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    off = torch.tensor(HEAD_OFFSET, dtype=torso_pos.dtype, device=torso_pos.device).expand_as(torso_pos)
    return torso_pos + quat_apply(torso_quat, off)


def pose_slots(root_pos: torch.Tensor, torso_pos: torch.Tensor, torso_quat: torch.Tensor, hands: torch.Tensor, feet: torch.Tensor) -> dict[str, torch.Tensor]:
    """Instantaneous height / torso / hand / feet slots from world poses: root_pos (…, 3), torso pose
    (…, 3) / (…, 4), hands and feet (…, 2, 3)."""
    head = head_position(torso_pos, torso_quat)
    hq = heading_quat(torso_quat)[..., None, :].expand(*hands.shape[:-1], 4)
    rel_hands = quat_apply_inverse(hq, hands - head[..., None, :])
    rel_feet = quat_apply_inverse(hq, feet - head[..., None, :])
    return {
        "height": root_pos[..., 2:3] - H_STAND,
        "torso": projected_gravity(torso_quat),
        "hand_L": rel_hands[..., 0, :],
        "hand_R": rel_hands[..., 1, :],
        "feet": rel_feet.reshape(*rel_feet.shape[:-2], 6),
    }


def assemble(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-slot tensors in registry order (…, SLOT_DIM)."""
    return torch.cat([parts[s.name] for s in SLOTS], dim=-1)


def torso_gravity(pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Torso slot of a lean (rad): R = R_y(pitch) R_x(roll), forward lean pitch > 0, right lean roll > 0."""
    sp, cp, sr, cr = torch.sin(pitch), torch.cos(pitch), torch.sin(roll), torch.cos(roll)
    return torch.stack([sp, -sr * cp, -cr * cp], dim=-1)


def registry_yaml() -> dict:
    """Slot table for deploy.yaml."""
    rows = []
    for s in SLOTS:
        sl = SLICE[s.name]
        rows.append({
            "name": s.name, "offset": sl.start, "dim": s.dim, "unit": s.unit, "desc": s.desc,
            "default": list(s.default), "reserved": s.reserved,
        })
    return {
        "version": REGISTRY_VERSION, "slot_dim": SLOT_DIM, "num_slots": NUM_SLOTS, "slots": rows,
        "h_stand": H_STAND, "head_offset_in_torso": list(HEAD_OFFSET),
        "mask": "每个槽位 1 位，1 = 有效、0 = 缺失；缺失槽位的数值随意（网络内部乘掩码），预留槽位恒为 0",
    }
