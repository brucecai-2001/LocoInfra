"""部署运行时：读权重包、拼观测 / 参考 / 槽位、ONNX 推理（含 z 插值切换）、后端（MuJoCo / PrimeBot 真机）、控制循环。

只依赖权重包里的文件（deploy.yaml、*.onnx、scene.xml），与训练代码无关。
"""

from runtime.backend import Backend, MuJoCoBackend, PrimeBotBackend, Sensors
from runtime.package import Package
from runtime.policy import BaseModel, ObsHistory, SkillModel, TeleGoal
from runtime.session import Session

__all__ = [
    "Backend", "BaseModel", "MuJoCoBackend", "ObsHistory", "Package", "PrimeBotBackend", "Sensors", "Session", "SkillModel", "TeleGoal",
]
