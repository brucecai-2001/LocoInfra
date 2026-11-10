"""权重包：deploy.yaml + ONNX + 训练场景。两种：

基础模型包（export_base_package.py）：policy_ref / policy_tele / decoder 三个图，参考布局在 ``body_ref``，带 scene.xml。
单段动作包（export_package.py，如起身 / 趴下技能）：一个 policy.onnx，输入 (obs, goal)，参考布局在 ``goal``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
MESH_DIR = PROJECT / "q1" / "q1_v3" / "meshes"


@dataclass
class Package:
    root: Path
    spec: dict

    @classmethod
    def load(cls, root: Path) -> "Package":
        root = Path(root)
        return cls(root, yaml.safe_load((root / "deploy.yaml").read_text()))

    @property
    def is_base(self) -> bool:
        return "graphs" in self.spec

    @property
    def joint_names(self) -> list[str]:
        return self.spec["joint_names"]

    @property
    def default_joint_pos(self) -> np.ndarray:
        return np.asarray(self.spec["default_joint_pos"], dtype=np.float64)

    @property
    def action_scale(self) -> np.ndarray:
        return np.asarray(self.spec["action_scale"], dtype=np.float64)

    @property
    def kp(self) -> np.ndarray:
        return np.asarray(self.spec["pd"]["kp"], dtype=np.float64)

    @property
    def kd(self) -> np.ndarray:
        return np.asarray(self.spec["pd"]["kd"], dtype=np.float64)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.spec["control_hz"])

    @property
    def stand_height(self) -> float:
        return float(self.spec["stand_height"])

    @property
    def ref_spec(self) -> dict:
        """参考输入的布局（future_frames、anchor_ori）：基础模型包的 E_ref 在 body_ref，单段动作包在 goal。"""
        return self.spec["body_ref"] if self.is_base else self.spec["goal"]

    @property
    def registry(self) -> dict:
        return self.spec["tele"]["registry"]

    def slot(self, name: str) -> dict:
        return next(s for s in self.registry["slots"] if s["name"] == name)

    def graph(self, name: str) -> Path:
        return self.root / self.spec["graphs"][name]["file"]
