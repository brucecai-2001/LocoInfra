"""轨迹查看器：一个 MuJoCo 窗口 + 终端命令。macOS 用 bin/q1view 启动（mjpython）。

  cd infra && bin/q1view jobs/<任务>/retarget                 # 原始重定向
  cd infra && bin/q1view jobs/<任务>/check/motion             # 优化 / 手改后的轨迹

窗口：
  右侧 Control 面板  TIME 时间轴（0–1）、PIN 钉脚（0 不钉 / 1 左 / 2 右）、骨盆 6 个、22 个关节（°）
  三维              穿模部位红、1 cm 内接近橙、力矩超限的关节所在部位黄
  左上              当前帧：穿模、接近、力矩超限、支撑脚
  底部缩略条        # 当前  ! 不可行  x 穿模  T 力矩超限  K 关键帧
  空格              播放 / 暂停

编辑（--edit）：拖关节滑条，当前帧成为关键帧，改动量平滑摊到前后关键帧之间；钉脚时骨盆自动跟着动，脚不离原位。
改完 check 重新检查，opt 联合优化（手改的帧权重更高），save 存盘。默认只读：可以看、可以试改，不存盘、不优化。

终端命令：n / p 前后一帧，nk / pk 前后关键帧，f <帧>，t <秒>，play [倍速]，stop，mv dx [dy dz]（cm，整帧平移），
  key / unkey，check，scan（列出不可行 / 穿模 / 力矩超限时段），opt [迭代]，undo，save [路径]，h，q。
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from q1infra.feasibility.scene import FLAT
from q1infra.viewer.editor import BASE, TrajectoryEditor

I_TIME, I_PIN, I_BASE = 0, 1, 2
I_JOINT = I_BASE + len(BASE)
HELP = __doc__[__doc__.index("终端命令："):]


def short(n: str) -> str:
    return n.replace("_joint", "").replace("left_", "L ").replace("right_", "R ")


class Viewer:
    def __init__(self, ed: TrajectoryEditor, readonly: bool):
        self.ed = ed
        self.readonly = readonly
        spec = ed.scene.build_spec()
        anyj = ed.joint_names[0]
        ctl = [("TIME 0-1", 0.0, 1.0), ("PIN 0/1L/2R", 0.0, 2.0), *BASE]
        ctl += [(short(n), np.degrees(lo), np.degrees(hi)) for n, lo, hi in zip(ed.joint_names, ed.lo, ed.hi)]
        for name, lo, hi in ctl:  # 只当滑条用：从不 mj_step，执行器不出力
            a = spec.add_actuator(name=name, target=anyj, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.ctrlrange = [lo, hi]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.gainprm[0] = 0.0
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.body_geoms: dict[str, list[int]] = {}
        for g in range(self.m.ngeom):
            if self.m.geom_group[g] in (0, 1, 2) and self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                bn = self.m.body(self.m.geom_bodyid[g]).name
                self.body_geoms.setdefault(bn, []).append(g)
        self.rgba0 = self.m.geom_rgba.copy()
        self.matid0 = self.m.geom_matid.copy()
        self.colored: set[str] = set()
        self.t = 0
        self.pin = 0
        self.pin_T = None
        self.playing = False
        self.speed = 1.0
        self.play_t0 = 0.0
        self.edit_burst_until = 0.0
        self.cmds: queue.Queue[str] = queue.Queue()
        self.msg = ""
        self.goto(0)
        self.push()  # 滑条先写成当前帧的值，否则第一次读到的全 0 会被当成手动编辑

    def log(self, s: str) -> None:
        self.msg = s
        print(s, flush=True)

    # ---------- 帧 ----------
    def goto(self, t: int) -> None:
        self.t = int(np.clip(t, 0, self.ed.N - 1))
        if self.pin:
            self.pin_T = self.ed.foot_pose(self.ed.Q[self.t], "left" if self.pin == 1 else "right")
        self.status = self.ed.frame_status(self.t)
        self.write_ctrl()

    def write_ctrl(self) -> None:
        q = self.ed.Q[self.t]
        c = np.zeros(self.m.nu)
        c[I_TIME] = self.t / max(self.ed.N - 1, 1)
        c[I_PIN] = self.pin
        c[I_BASE:I_JOINT] = self.ed.base_of(q)
        c[I_JOINT:] = np.degrees(q[7:])
        self.ctrl_out = c
        self.written = c.copy()
        self.ctrl_dirty = True

    def read_ui(self, c: np.ndarray) -> None:
        ch = np.abs(c - self.written) > 1e-6
        if not ch.any():
            return
        self.playing = False
        if ch[I_TIME]:
            self.goto(round(float(c[I_TIME]) * (self.ed.N - 1)))
            return
        if ch[I_PIN]:
            self.pin = int(round(float(c[I_PIN])))
            self.pin_T = self.ed.foot_pose(self.ed.Q[self.t], "left" if self.pin == 1 else "right") if self.pin else None
            self.log(f"钉脚：{['不钉', '左脚', '右脚'][self.pin]}")
            self.write_ctrl()
            return
        if self.ed.opt_thread is not None:
            self.log("优化进行中，先别改")
            self.write_ctrl()
            return
        q = self.ed.Q[self.t].copy()
        if ch[I_BASE:I_JOINT].any():
            q = self.ed.set_base(q, c[I_BASE:I_JOINT])
            if self.pin:
                self.pin_T = self.ed.foot_pose(q, "left" if self.pin == 1 else "right")
        if ch[I_JOINT:].any():
            q[7:] = np.clip(np.radians(c[I_JOINT:]), self.ed.lo, self.ed.hi)
            if self.pin:
                q = self.ed.keep_foot(q, "left" if self.pin == 1 else "right", self.pin_T)
        now = time.monotonic()
        self.ed.apply_edit(self.t, q, new_burst=now > self.edit_burst_until)
        self.edit_burst_until = now + 0.3
        self.status = self.ed.frame_status(self.t)
        self.write_ctrl()

    # ---------- 命令 ----------
    def command(self, line: str) -> bool:
        a = line.split()
        if not a:
            return True
        c, arg = a[0], a[1:]
        ed = self.ed
        if c in ("h", "help"):
            print(HELP)
        elif c == "n":
            self.goto(self.t + 1)
        elif c == "p":
            self.goto(self.t - 1)
        elif c in ("nk", "pk"):
            ks = np.flatnonzero(ed.key)
            ks = ks[ks > self.t] if c == "nk" else ks[ks < self.t][::-1]
            if len(ks):
                self.goto(int(ks[0]))
        elif c == "f" and arg:
            self.goto(int(arg[0]))
        elif c == "t" and arg:
            self.goto(round(float(arg[0]) / ed.dt))
        elif c == "play":
            self.speed = float(arg[0]) if arg else 1.0
            self.playing = True
            self.play_t0 = time.monotonic() - self.t * ed.dt / self.speed
        elif c == "stop":
            self.playing = False
        elif c == "mv" and arg:
            dx, dy, dz = (list(map(float, arg)) + [0.0, 0.0])[:3]
            ed.shift(self.t, np.array([dx, dy, dz]))
            self.goto(self.t)
        elif c == "key":
            ed.key[self.t] = True
        elif c == "unkey":
            ed.key[self.t] = False
        elif c == "check":
            r = ed.run_check()
            self.log(f"检查：{r['verdict']}，不可行时段 {len(r['infeasible_segments_s'])} 段")
            self.goto(self.t)
        elif c == "scan":
            print("\n".join(ed.scan()))
            self.goto(self.t)
        elif c == "opt":
            if self.readonly:
                self.log("只读模式不优化（用 --edit 启动）")
            elif ed.start_opt(int(arg[0]) if arg else 25):
                self.log("[opt] 开始（后台），完成后自动替换轨迹，undo 可撤回")
        elif c == "undo":
            if ed.undo():
                self.goto(self.t)
                self.log("已撤销")
        elif c == "save":
            if self.readonly:
                self.log("只读模式不存盘（用 --edit 启动）")
            else:
                self.log(f"已存 {ed.save(Path(arg[0]) if arg else None)}")
        elif c in ("q", "quit", "exit"):
            return False
        else:
            print("未知命令，h 看帮助")
        return True

    def tick(self, ctrl: np.ndarray) -> bool:
        if self.ed.poll_opt():
            self.log(f"[opt] 完成：{self.ed.report['verdict']}（undo 可撤回）")
            self.goto(self.t)
        self.read_ui(ctrl)
        while not self.cmds.empty():
            if not self.command(self.cmds.get()):
                return False
        if self.playing:
            t = int((time.monotonic() - self.play_t0) / self.ed.dt * self.speed)
            if t >= self.ed.N:
                self.play_t0, t = time.monotonic(), 0
            if t != self.t:
                self.goto(t)
        return True

    # ---------- 显示 ----------
    def push(self) -> None:
        if self.ctrl_dirty:
            self.d.ctrl[:] = self.ctrl_out
            self.ctrl_dirty = False
        self.d.qpos[: self.ed.m.nq] = self.ed.Q[self.t]
        mujoco.mj_kinematics(self.m, self.d)
        want = self.ed.colors(self.t, self.status)
        for b in self.colored - set(want):
            for g in self.body_geoms.get(b, []):
                self.m.geom_rgba[g] = self.rgba0[g]
                self.m.geom_matid[g] = self.matid0[g]
        for b, col in want.items():
            for g in self.body_geoms.get(b, []):
                self.m.geom_rgba[g] = col
                self.m.geom_matid[g] = -1
        self.colored = set(want)

    def hud(self) -> tuple[tuple[str, str], tuple[str, str]]:
        ed, st = self.ed, self.status
        fmt = lambda xs: ", ".join(f"{short(a)}x{short(b)} {dd * 1000:.0f}mm" for a, b, dd in xs[:3]) or "-"  # noqa: E731
        rows = [
            ("time", f"{self.t * ed.dt:6.2f} / {(ed.N - 1) * ed.dt:.2f} s   frame {self.t}/{ed.N - 1}"
             + ("   [KEY]" if ed.key[self.t] else "") + ("   PLAY" if self.playing else "")),
            ("pin", ["-", "left foot", "right foot"][self.pin]),
            ("collide", fmt(st.collide)),
            ("near", fmt(st.near)),
            ("torque>max", ", ".join(f"{short(j)} {v:.0f}/{lim:.0f}" for j, v, lim in st.torque_over[:3]) or "-"),
            ("stance", ("L" if st.stance[0] else "-") + ("R" if st.stance[1] else "-")),
            ("check", "stale (run check)" if ed.stale else ed.report["verdict"]),
        ]
        if ed.opt_thread is not None:
            rows.append(("optimizing", self.msg[:90]))
        top = ("\n".join(a for a, _ in rows), "\n".join(b for _, b in rows))
        bottom = (f"{ed.bar(self.t)}\n# now  ! infeasible  x collide  T torque  K key   keys {int(ed.key.sum())}", "")
        return top, bottom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="any4hdmi 数据集目录，或含 qpos / fps 的 npz")
    ap.add_argument("--edit", action="store_true", help="允许存盘和优化；默认只读")
    a = ap.parse_args()
    ed = TrajectoryEditor(a.input, FLAT)
    v = Viewer(ed, readonly=not a.edit)
    ed.log = v.log
    mode = "只读" if v.readonly else f"编辑，save 存到 {ed.default_save_path()}"
    print(f"载入 {a.input}：{ed.N} 帧 {ed.N * ed.dt:.1f} s，检查结论 {ed.report['verdict']}；{mode}\n{HELP}", flush=True)

    def reader():
        while True:
            try:
                v.cmds.put(input())
            except EOFError:
                return

    threading.Thread(target=reader, daemon=True).start()

    def on_key(k: int) -> None:
        if k == 32:
            v.cmds.put("stop" if v.playing else "play")

    with mujoco.viewer.launch_passive(v.m, v.d, key_callback=on_key, show_left_ui=False) as win:
        q0 = ed.Q[0]
        win.cam.lookat[:] = [q0[0], q0[1], 0.4]
        win.cam.distance, win.cam.azimuth, win.cam.elevation = 2.0, 135, -15
        while win.is_running():
            t0 = time.time()
            with win.lock():
                ctrl = v.d.ctrl.copy()
            if not v.tick(ctrl):
                break
            (tl, tr), (b1, b2) = v.hud()
            with win.lock():
                v.push()
            win.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, tl, tr),
                           (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, b1, b2)])
            win.sync()
            time.sleep(max(0.005, 1 / 60 - (time.time() - t0)))
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
