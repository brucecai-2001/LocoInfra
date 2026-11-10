"""可行性检查的场景：机器人 + 环境、哪些环境几何体是支撑面、哪个部位允许贴哪类支撑面。平地只允许脚着地。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco as mj
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
Q1_COLLISION_XML = PROJECT_ROOT / "q1" / "q1_v3" / "xml" / "q1_v3_collision.xml"
FEET = frozenset({"left_ankle_roll_link", "right_ankle_roll_link"})


@dataclass
class Scene:
    name: str
    build_spec: Callable[[], mj.MjSpec]
    # 每个 geom 的环境类别："" 表示机器人；支撑面如 "floor"；不能碰的环境用 "rest"
    env_groups: Callable[[mj.MjModel], np.ndarray]
    # (部位名, 环境类别) -> 是否允许接触
    allowed: Callable[[str, str], bool]
    support_groups: frozenset[str] = field(default_factory=lambda: frozenset({"floor"}))
    soft_max: float = 0.015  # 允许接触最多压入
    hard_tol: float = 0.003  # 其余接触容差
    contact_tol: float = 0.01  # 不允许的部位离支撑面 / 环境小于该距离记为"不允许的接触"

    def build_model(self) -> mj.MjModel:
        return self.build_spec().compile()


def _flat_groups(m: mj.MjModel) -> np.ndarray:
    return np.array(["floor" if m.geom(g).name == "floor" else "" for g in range(m.ngeom)])


FLAT = Scene(
    name="flat",
    build_spec=lambda: mj.MjSpec.from_file(str(Q1_COLLISION_XML)),
    env_groups=_flat_groups,
    allowed=lambda body, group: body in FEET and group == "floor",
)

