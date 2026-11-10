"""遥操作活动记录：每个控制周期一行（时间戳、状态、手柄、命令、关节），状态切换与按键另记事件。

一次启动一个目录 ``<root>/rec_<开始时间>/``：
    meta.json        关节名、控制周期、状态名表、权重包路径与其中 .onnx 的 sha256、启动参数
    events.jsonl     事件，每行一个 JSON：state（状态 / 提示变化，含急停、倾倒、后端故障的原因）、buttons（按下沿）、
                     pad（手柄连上 / 断开）、speed（速度系数）、end
    steps_NNNNN.npz  每 CHUNK 个控制周期一块（50 Hz 下 10 s），字段见 FIELDS；不在跟踪时 joint_target 为 NaN

控制循环里只往预分配的数组里抄一行；写盘在后台线程（真机上切回普通调度、挪到小核，不跟策略进程抢核），
异常退出最多丢最后一块。只保留最近 KEEP 次启动。

查看 / 导出（Mac 或机器人上都行）：
    python -m apps.teleop.recorder <rec 目录>              摘要：时长、各状态时间线、事件
    python -m apps.teleop.recorder <rec 目录> --csv out.csv   导出每周期一行的 CSV
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from apps.teleop.gamepad import BUTTONS

CHUNK = 500
KEEP = 30
MODES = ("LYING", "GETUP", "WALK", "STRAIGHTEN", "LIEDOWN", "DAMPED")
AXES = ("lx", "ly", "rx", "ry", "lt", "rt")
WRITER_CORES = {0, 1, 2, 3}  # A53 小核；策略进程在大核


def _fields(nj: int) -> dict[str, tuple[tuple[int, ...], type]]:
    return {
        "t": ((), np.float64),  # Unix 时间（s）
        "t_rel": ((), np.float64),  # 启动以来（s，单调时钟）
        "mode": ((), np.int8),  # MODES 的下标
        "ref_frame": ((), np.int16),  # 起身 / 趴下 / 上车参考的当前帧，其余状态 -1
        "pad_connected": ((), np.bool_),
        "pad_axes": ((len(AXES),), np.float32),  # AXES 顺序：摇杆 [-1, 1]（上 / 左为正），扳机 [0, 1]
        "pad_buttons": ((), np.uint16),  # 第 k 位 = BUTTONS[k] 按住
        "cmd": ((6,), np.float32),  # E_tele 命令 vx vy wz dh（m/s, m/s, rad/s, m）及两个保留位
        "speed": ((), np.float32),
        "joint_pos": ((nj,), np.float32),  # rad，deploy.yaml joint_names 顺序
        "joint_vel": ((nj,), np.float32),
        "joint_target": ((nj,), np.float32),  # 本周期下发的 PD 目标；不在跟踪为 NaN
        "kp": ((nj,), np.float32),
        "base_ang_vel": ((3,), np.float32),  # 骨盆 IMU 角速度，骨盆系
        "projected_gravity": ((3,), np.float32),
    }


def _sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _background() -> None:
    """后台线程继承了创建者的绑核和调度策略：真机上改回普通调度、挪到小核。"""
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except (AttributeError, OSError):
        pass
    try:
        all_cores = set(range(os.cpu_count() or 1))
        os.sched_setaffinity(0, (WRITER_CORES & all_cores) or all_cores)
    except (AttributeError, OSError):
        pass


class Recorder:
    def __init__(self, root: Path, joint_names: list[str], dt: float, packages: dict[str, Path | None], args: dict):
        root.mkdir(parents=True, exist_ok=True)
        old = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("rec_"))
        for d in old[: max(0, len(old) - (KEEP - 1))]:
            shutil.rmtree(d, ignore_errors=True)
        self.dir = root / time.strftime("rec_%Y%m%d_%H%M%S")
        self.dir.mkdir()
        self.nj = len(joint_names)
        self.spec = _fields(self.nj)
        self.t0 = time.monotonic()
        meta = {
            "start": time.strftime("%Y-%m-%d %H:%M:%S"), "control_dt": dt, "joint_names": list(joint_names),
            "modes": MODES, "pad_axes": AXES, "pad_buttons": BUTTONS, "chunk": CHUNK, "args": args,
            "packages": {k: None if p is None else {"path": str(p), "onnx_sha256": {f.name: _sha256(f) for f in sorted(Path(p).glob("*.onnx"))}}
                         for k, p in packages.items()},
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        self._q: queue.Queue = queue.Queue()
        self._events = open(self.dir / "events.jsonl", "a", buffering=1, encoding="utf-8")
        self._thread = threading.Thread(target=self._writer, name="teleop-recorder", daemon=True)
        self._thread.start()
        self._new_chunk()
        self.n_chunks = 0
        self.prev: tuple | None = None
        self.prev_pad: bool | None = None
        self.prev_speed: float | None = None
        print(f"[teleop] 记录到 {self.dir}", flush=True)

    def _new_chunk(self) -> None:
        self.buf = {k: np.zeros((CHUNK, *shape), dtype=dt) for k, (shape, dt) in self.spec.items()}
        self.n = 0

    def _writer(self) -> None:
        _background()
        while True:
            item = self._q.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "chunk":
                    path, data = payload
                    tmp = path.with_name(path.name + ".tmp")
                    with open(tmp, "wb") as f:
                        np.savez(f, **data)
                    tmp.rename(path)
                else:
                    self._events.write(payload + "\n")
            except OSError as e:
                print(f"[teleop] 记录写盘失败：{e}", flush=True)

    def event(self, kind: str, **fields) -> None:
        self._q.put(("event", json.dumps({"t": round(time.time(), 3), "t_rel": round(time.monotonic() - self.t0, 3),
                                          "kind": kind, **fields}, ensure_ascii=False)))

    def step(self, tele, pad, edges: set[str]) -> None:
        """每个控制周期 tick 之后调一次。"""
        state = (tele.mode, tele.msg)
        if state != self.prev:
            self.event("state", mode=tele.mode, msg=tele.msg)
            self.prev = state
        if edges:
            self.event("buttons", down=sorted(edges), l2=pad.l2, r2=pad.r2, mode=tele.mode)
        if pad.connected != self.prev_pad:
            self.event("pad", connected=pad.connected)
            self.prev_pad = pad.connected
        if tele.speed != self.prev_speed:
            self.event("speed", speed=tele.speed)
            self.prev_speed = tele.speed

        s = tele.robot.sense()
        b, k = self.buf, self.n
        b["t"][k] = time.time()
        b["t_rel"][k] = time.monotonic() - self.t0
        b["mode"][k] = MODES.index(tele.mode)
        b["ref_frame"][k] = tele.i if tele.mode in ("GETUP", "LIEDOWN") else -1
        b["pad_connected"][k] = pad.connected
        b["pad_axes"][k] = [getattr(pad, a) for a in AXES]
        b["pad_buttons"][k] = sum(1 << j for j, name in enumerate(BUTTONS) if pad.buttons[name])
        b["cmd"][k] = tele.cmd
        b["speed"][k] = tele.speed
        b["joint_pos"][k] = s.joint_pos
        b["joint_vel"][k] = s.joint_vel
        b["joint_target"][k] = np.nan if tele.target is None else tele.target
        b["kp"][k] = np.nan if tele.kp is None else tele.kp
        b["base_ang_vel"][k] = s.base_ang_vel
        b["projected_gravity"][k] = s.projected_gravity
        self.n += 1
        if self.n == CHUNK:
            self._flush()

    def _flush(self) -> None:
        if self.n == 0:
            return
        data = {k: v[: self.n] for k, v in self.buf.items()}
        self._q.put(("chunk", (self.dir / f"steps_{self.n_chunks:05d}.npz", data)))
        self.n_chunks += 1
        self._new_chunk()

    def close(self, why: str = "") -> None:
        self._flush()
        self.event("end", why=why)
        self._q.put(None)
        self._thread.join(timeout=5.0)
        self._events.close()


# ---- 读取
def load(rec: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    meta = json.loads((rec / "meta.json").read_text())
    chunks = [np.load(p) for p in sorted(rec.glob("steps_*.npz"))]
    steps = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].files} if chunks else {}
    ev_path = rec / "events.jsonl"
    events = [json.loads(ln) for ln in ev_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if ev_path.exists() else []
    return meta, steps, events


def summary(rec: Path) -> None:
    meta, s, events = load(rec)
    modes = meta["modes"]
    print(f"{rec}  开始 {meta['start']}  控制周期 {meta['control_dt'] * 1000:.0f} ms")
    for k, v in meta["packages"].items():
        print(f"  {k:8s} {v['path'] if v else '-'}" + ("".join(f"  {n} {h[:12]}" for n, h in v["onnx_sha256"].items()) if v else ""))
    if not s:
        print("  没有控制周期数据")
    else:
        t = s["t_rel"]
        print(f"  {len(t)} 个周期，{t[-1] - t[0]:.1f} s；实测周期中位 {np.median(np.diff(t)) * 1000:.1f} ms，最长 {np.diff(t).max() * 1000:.0f} ms")
        m = s["mode"]
        cut = np.flatnonzero(np.diff(m)) + 1
        print("  状态时间线：")
        for a, z in zip(np.r_[0, cut], np.r_[cut, len(m)]):
            print(f"    {t[a]:8.2f} – {t[z - 1]:8.2f} s  {modes[m[a]]}")
    print("  事件：")
    for e in events:
        rest = {k: v for k, v in e.items() if k not in ("t", "t_rel", "kind")}
        print(f"    {e['t_rel']:8.2f} s  {time.strftime('%H:%M:%S', time.localtime(e['t']))}  {e['kind']:7s} {json.dumps(rest, ensure_ascii=False)}")


def to_csv(rec: Path, out: Path) -> None:
    meta, s, _ = load(rec)
    names, modes = meta["joint_names"], meta["modes"]
    cols, data = [], []
    for k, v in s.items():
        if k == "mode":
            continue
        if v.ndim == 1:
            cols.append(k)
            data.append(v.astype(np.float64))
        else:
            sub = names if v.shape[1] == len(names) else meta["pad_axes"] if k == "pad_axes" else [str(j) for j in range(v.shape[1])]
            cols += [f"{k}.{n}" for n in sub]
            data += [v[:, j].astype(np.float64) for j in range(v.shape[1])]
    with open(out, "w", encoding="utf-8") as f:
        f.write("mode," + ",".join(cols) + "\n")
        for r in range(len(s["t"])):
            f.write(modes[s["mode"][r]] + "," + ",".join(f"{d[r]:.6g}" for d in data) + "\n")
    print(f"{out}: {len(s['t'])} 行 × {len(cols) + 1} 列")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="查看 / 导出遥操作记录")
    p.add_argument("rec", type=Path, help="rec_<时间> 目录")
    p.add_argument("--csv", type=Path, default=None, help="导出每周期一行的 CSV")
    a = p.parse_args()
    summary(a.rec) if a.csv is None else to_csv(a.rec, a.csv)
