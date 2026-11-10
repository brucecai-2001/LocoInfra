"""观测历史、遥操作槽位与 ONNX 推理（按 deploy.yaml 拼输入）。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from runtime.package import Package


class ObsHistory:
    """按项分块（term-major）的 10 帧本体历史；首帧填满全部历史。"""

    def __init__(self, pkg: Package):
        self.terms = [t["name"] for t in pkg.spec["obs"]["terms"]]
        self.H = int(pkg.spec["obs"]["history"])
        self.hist: dict[str, deque] = {}

    def reset(self, frame: dict[str, np.ndarray]) -> None:
        self.hist = {k: deque([frame[k].astype(np.float32)] * self.H, maxlen=self.H) for k in self.terms}

    def push(self, frame: dict[str, np.ndarray]) -> None:
        for k in self.terms:
            self.hist[k].append(frame[k].astype(np.float32))

    def vector(self) -> np.ndarray:
        return np.concatenate([np.concatenate(list(self.hist[k])) for k in self.terms])[None].astype(np.float32)


class TeleGoal:
    """遥操作输入：槽位物理量 + 掩码，布局取自 deploy.yaml 的槽位注册表。缺失槽位掩码为 0，预留槽位恒为 0。"""

    def __init__(self, pkg: Package):
        reg = pkg.registry
        self.slots = {s["name"]: s for s in reg["slots"]}
        self.names = [s["name"] for s in reg["slots"]]
        self.values = np.concatenate([np.asarray(s["default"], dtype=np.float32) for s in reg["slots"]])
        self.mask = np.zeros(reg["num_slots"], dtype=np.float32)

    def set(self, name: str, value=None, valid: bool = True) -> None:
        s = self.slots[name]
        if s["reserved"]:
            raise ValueError(f"slot {name} is reserved")
        if value is not None:
            self.values[s["offset"] : s["offset"] + s["dim"]] = np.asarray(value, dtype=np.float32).reshape(s["dim"])
        self.mask[self.names.index(name)] = float(valid)

    def get(self, name: str) -> np.ndarray:
        s = self.slots[name]
        return self.values[s["offset"] : s["offset"] + s["dim"]]

    def drop(self, name: str) -> None:
        self.mask[self.names.index(name)] = 0.0


def _session(path: Path):
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


class BaseModel:
    """基础动作模型的三个图。``act_tele`` / ``act_ref`` 返回 (action, joint_target)；``blend`` > 0 时，
    输出用最近一次的 z 与本次 z 线性插值后经 decoder.onnx 解码（两个应用之间切换时用）。"""

    def __init__(self, pkg: Package):
        self.tele = _session(pkg.graph("tele"))
        self.ref = _session(pkg.graph("ref"))
        self.decoder = _session(pkg.graph("decoder"))
        self.z_prev: np.ndarray | None = None
        self.blend_steps = 0
        self.blend_left = 0
        self.z_from: np.ndarray | None = None

    def start_blend(self, steps: int) -> None:
        """从下一步起在 ``steps`` 步内由当前 z 过渡到新接口的 z。"""
        self.blend_steps = self.blend_left = int(steps)
        self.z_from = None if self.z_prev is None else self.z_prev.copy()

    def _finish(self, obs, action, target, z):
        if self.blend_left > 0 and self.z_from is not None:
            w = 1.0 - self.blend_left / max(self.blend_steps, 1)
            z = ((1.0 - w) * self.z_from + w * z).astype(np.float32)
            action, target = self.decoder.run(None, {"obs": obs, "z": z})
            self.blend_left -= 1
        self.z_prev = z
        return action[0], target[0]

    def act_tele(self, obs: np.ndarray, goal: TeleGoal):
        a, q, z = self.tele.run(None, {"obs": obs, "slots": goal.values[None], "mask": goal.mask[None]})
        return self._finish(obs, a, q, z)

    def act_ref(self, obs: np.ndarray, body_ref: np.ndarray):
        a, q, z = self.ref.run(None, {"obs": obs, "body_ref": body_ref[None].astype(np.float32)})
        return self._finish(obs, a, q, z)


class SkillModel:
    """跟踪参考的技能策略（起身、趴下等）：单段动作包的 policy.onnx（输入 obs、goal）；
    给基础模型包时用它的 policy_ref.onnx（输入 obs、body_ref）。参考输入由 BodyRef(同一个包) 拼。"""

    def __init__(self, pkg: Package):
        self.pkg = pkg
        if pkg.is_base:
            self.sess, self.key = _session(pkg.graph("ref")), "body_ref"
        else:
            self.sess, self.key = _session(pkg.root / pkg.spec["policy"]["file"]), "goal"

    def act(self, obs: np.ndarray, goal: np.ndarray):
        out = self.sess.run(None, {"obs": obs, self.key: goal[None].astype(np.float32)})
        return out[0][0], out[1][0]
