"""控制循环的公共部分：后端 + 观测历史 + 上一步 action。各应用每个控制周期调一次 ``apply``。"""

from __future__ import annotations

import numpy as np

from runtime.backend import Backend, MuJoCoBackend
from runtime.package import Package
from runtime.policy import BaseModel, ObsHistory


class Session:
    def __init__(self, pkg: Package, robot: Backend | None = None):
        self.pkg = pkg
        self.robot = robot if robot is not None else MuJoCoBackend(pkg)
        self.model = BaseModel(pkg)
        self.hist = ObsHistory(pkg)
        self.q0 = pkg.default_joint_pos

    def begin(self) -> None:
        """回合开始：last action 清零，历史填满当前帧。"""
        self.last_action = np.zeros(len(self.q0), dtype=np.float32)
        self.hist.reset(self.robot.sense().obs_frame(self.q0, self.last_action))
        self.model.z_prev = None

    def obs(self) -> np.ndarray:
        return self.hist.vector()

    def apply(self, action, target) -> None:
        self.robot.step(target)
        self.last_action = np.asarray(action, dtype=np.float32)
        self.hist.push(self.robot.sense().obs_frame(self.q0, self.last_action))
