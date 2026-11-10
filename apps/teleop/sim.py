"""Q1 遥控仿真：与真机同一套状态机（machine.py 的 Teleop），手柄 + 键盘合并成同一个 Pad 输入，CPU MuJoCo。

    LYING ──L2+A──▶ GETUP ──▶ WALK（E_tele：vx、vy、yaw、高度）──L2+B──▶ STRAIGHTEN ──▶ LIEDOWN ──▶ LYING
    任意状态 BACK 急停切阻尼；行走中倾倒自动切阻尼；START 重置。没有技能包时从站姿开始，只能行走。

手柄可以在窗口打开后再插（每 0.5 s 检测一次）；接上后断开会和真机一样切阻尼。没有手柄时键盘同样可用：

    infra/bin/q1py apps/teleop/sim.py
    infra/bin/q1py apps/teleop/sim.py --script 'idle:1,getup:12,fwd:3,turn_l:2,squat:1,rise:1,liedown:14' --video out.mp4

手柄（Xbox 布局）：
    L2 + A  起身            L2 + B  站直后趴下      BACK   急停切阻尼       START  重置
    左摇杆  vx / vy         右摇杆左右  yaw         十字键 ↑ / ↓  高度      Y      高度回正
键盘（等价于手柄对应输入）：
    G       L2 + A          L       L2 + B          Space  BACK             R      START
    W / S   前进 / 后退     A / D   左移 / 右移     Q / E  左转 / 右转
    X / Z   起身 / 下蹲     C       高度回正        + / -  速度档位 ±0.1    Esc    退出
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps import ref  # noqa: E402
from apps.teleop.__main__ import DEFAULT_BASE, DEFAULT_SKILL, load_skill  # noqa: E402
from apps.teleop.command import fill_goal  # noqa: E402
from apps.teleop.gamepad import Edges, GlfwPad, Pad, ScriptPad  # noqa: E402
from apps.teleop.machine import Teleop  # noqa: E402
from runtime import BaseModel, MuJoCoBackend, ObsHistory, Package, TeleGoal  # noqa: E402

ARROW_SCALE = 1.5  # 速度箭头：1 m/s 画 1.5 m
PAD_SCAN_S = 0.5


class Controller:
    """不经状态机：从站姿直接用 E_tele 跟命令（gait_check.py、touchdown_stats.py 等步态分析用）。"""

    def __init__(self, package: Path):
        self.pkg = Package.load(package)
        self.sim = MuJoCoBackend(self.pkg)
        self.model = BaseModel(self.pkg)
        self.hist = ObsHistory(self.pkg)
        self.goal = TeleGoal(self.pkg)
        self.q0 = self.pkg.default_joint_pos
        self.reset()

    def reset(self) -> None:
        self.sim.reset()
        self.last_action = np.zeros(len(self.q0), dtype=np.float32)
        self.hist.reset(self.sim.sense().obs_frame(self.q0, self.last_action))
        self.fallen = False
        self.dh = 0.0

    def step(self, cmd: np.ndarray) -> None:
        """cmd = (vx, vy, wz, Δh, pitch, roll)：base、height、torso 三个槽位有效，双手缺省（掩码 0）。"""
        fill_goal(self.goal, cmd)
        action, target = self.model.act_tele(self.hist.vector(), self.goal)
        self.sim.step(target)
        self.last_action = action
        self.hist.push(self.sim.sense().obs_frame(self.q0, self.last_action))
        self.dh = cmd[3]
        if self.sim.pelvis_height() < self.pkg.stand_height + cmd[3] - 0.15 or self.sim.upright() < 0.3:
            self.fallen = True


class Display:
    """实测速度低通（步态内速度起伏大，约 0.3 s 时间常数）+ 场景里的速度箭头。
    绿色：命令；橙色：实测（均在骨盆下方地面附近，航向系 → 世界系，1 m/s 画 ARROW_SCALE m）。"""

    def __init__(self, tele: Teleop, tau: float = 0.3):
        self.tele, self.sim = tele, tele.robot
        self.alpha = tele.dt / (tau + tele.dt)
        self.meas = np.zeros(4)

    def update(self) -> np.ndarray:
        m = np.concatenate([self.sim.base_velocity(), [self.sim.pelvis_height() - self.tele.base.stand_height]])
        self.meas += self.alpha * (m - self.meas)
        return self.meas

    def add_arrows(self, scn: mujoco.MjvScene) -> None:
        d = self.sim.data
        R = d.xmat[self.sim.pelvis].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        base = d.xpos[self.sim.pelvis].copy()
        base[2] = 0.05
        for v, rgba, dz in ((self.tele.cmd, (0.2, 0.9, 0.3, 0.9), 0.0), (self.meas, (1.0, 0.55, 0.1, 0.9), 0.03)):
            vec = ARROW_SCALE * np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])
            if np.linalg.norm(vec) < 0.02 or scn.ngeom >= scn.maxgeom:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
            a = base + [0, 0, dz]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, a, a + vec)
            scn.ngeom += 1

    def rows(self) -> list[tuple[str, str, str]]:
        c, m, t = self.tele.cmd, self.meas, self.tele
        rows = [("mode", t.mode, f"{t.t_mode:.1f} s"), ("", "cmd", "meas")]
        return rows + [("vx m/s", f"{c[0]:+.2f}", f"{m[0]:+.2f}"), ("vy m/s", f"{c[1]:+.2f}", f"{m[1]:+.2f}"),
                       ("wz rad/s", f"{c[2]:+.2f}", f"{m[2]:+.2f}"), ("dh m", f"{c[3]:+.2f}", f"{m[3]:+.2f}")]


def draw_text(img: np.ndarray, rows) -> np.ndarray:
    import cv2

    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (6, 6), (266, 14 + 22 * len(rows)), (20, 20, 20), -1)
    for i, (a, b, c) in enumerate(rows):
        y = 28 + 22 * i
        cv2.putText(img, a, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, b, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 230, 80), 1, cv2.LINE_AA)
        cv2.putText(img, c, (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 40), 1, cv2.LINE_AA)
    return img


class Inputs:
    """手柄（热插拔）与键盘合并成一个 Pad：摇杆相加后限幅，按键取或；connected 只看手柄。"""

    def __init__(self, glfw):
        self.glfw = glfw
        self.held: set[int] = set()
        self.gamepad: GlfwPad | None = None
        self.t_scan = 0.0

    def _scan(self) -> None:
        now = time.perf_counter()
        if self.gamepad is not None or now - self.t_scan < PAD_SCAN_S:
            return
        self.t_scan = now
        try:
            self.gamepad = GlfwPad()
            print(f"[teleop] 手柄已连接：{self.gamepad.name}", flush=True)
        except FileNotFoundError:
            pass

    @property
    def source(self) -> str:
        return self.gamepad.name if self.gamepad is not None else "keyboard (no gamepad)"

    def read(self) -> Pad:
        self._scan()
        p = self.gamepad.read() if self.gamepad is not None else Pad()
        g = self.glfw
        k = lambda key: float(key in self.held)  # noqa: E731
        p.ly = float(np.clip(p.ly + k(g.KEY_W) - k(g.KEY_S), -1.0, 1.0))
        p.lx = float(np.clip(p.lx + k(g.KEY_A) - k(g.KEY_D), -1.0, 1.0))
        p.rx = float(np.clip(p.rx + k(g.KEY_Q) - k(g.KEY_E), -1.0, 1.0))
        keys = {"UP": (g.KEY_X,), "DOWN": (g.KEY_Z,), "Y": (g.KEY_C,), "BACK": (g.KEY_SPACE,), "START": (g.KEY_R,),
                "A": (g.KEY_G,), "B": (g.KEY_L,)}
        for btn, ks in keys.items():
            p.buttons[btn] = p.buttons[btn] or any(x in self.held for x in ks)
        if g.KEY_G in self.held or g.KEY_L in self.held:
            p.lt = 1.0
        return p


def run_script(tele: Teleop, script: str, video: Path | None) -> None:
    """无界面：ScriptPad 的 '动作:秒' 序列（同 python -m apps.teleop --script）；逐段打印跟踪结果，可选录 mp4。"""
    pad = ScriptPad(script, tele.dt)
    sim: MuJoCoBackend = tele.robot
    renderer = cam = None
    frames = []
    if video is not None:
        renderer = mujoco.Renderer(sim.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.pelvis
        cam.distance, cam.elevation, cam.azimuth = 2.0, -15.0, 135.0
    disp = Display(tele)
    edges = Edges()
    seg, meas, n = None, [], 0
    while not pad.done:
        if pad.segment != seg:
            if meas:
                m = np.mean(meas, 0)
                print(f"           {seg}: 实测 vx {m[0]:+.2f} vy {m[1]:+.2f} wz {m[2]:+.2f} dh {m[3]:+.3f}", flush=True)
            seg, meas = pad.segment, []
            print(f"[script] {seg}", flush=True)
        p = pad.read()
        tele.tick(p, edges(p))
        disp.update()
        if tele.mode == "WALK" and tele.t_mode > 1.5:
            meas.append(np.concatenate([sim.base_velocity(), [sim.pelvis_height() - tele.base.stand_height]]))
        if renderer is not None and n % 2 == 0:
            renderer.update_scene(sim.data, cam)
            disp.add_arrows(renderer.scene)
            frames.append(draw_text(renderer.render(), disp.rows()))
        n += 1
    print(f"[script] 结束：{tele.mode}  骨盆高 {sim.pelvis_height():.3f} m  upright {sim.upright():.2f}", flush=True)
    if frames:
        import imageio

        imageio.mimsave(video, frames, fps=int(round(0.5 / tele.dt)))
        print("video:", video)


def run_interactive(tele: Teleop) -> None:
    import glfw

    glfw.init()
    win = glfw.create_window(1280, 800, "Q1 遥控仿真（手柄 / 键盘）", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    sim: MuJoCoBackend = tele.robot
    m = sim.model
    scn = mujoco.MjvScene(m, maxgeom=2000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sim.pelvis
    cam.distance, cam.elevation, cam.azimuth = 2.2, -15.0, 135.0
    opt = mujoco.MjvOption()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    inputs = Inputs(glfw)

    def on_key(_w, key, _sc, action, mods):
        if action == glfw.PRESS:
            inputs.held.add(key)
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                tele.adjust_speed(1)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                tele.adjust_speed(-1)
        elif action == glfw.RELEASE:
            inputs.held.discard(key)

    drag = {"x": None, "y": None}

    def on_mouse_move(_w, x, y):
        if drag["x"] is not None and glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            cam.azimuth -= 0.3 * (x - drag["x"])
            cam.elevation = float(np.clip(cam.elevation - 0.3 * (y - drag["y"]), -80, 10))
        drag["x"], drag["y"] = x, y

    def on_scroll(_w, _dx, dy):
        cam.distance = float(np.clip(cam.distance * (0.9 if dy > 0 else 1.1), 0.6, 8.0))

    glfw.set_key_callback(win, on_key)
    glfw.set_cursor_pos_callback(win, on_mouse_move)
    glfw.set_scroll_callback(win, on_scroll)
    disp = Display(tele)
    edges = Edges()
    print(f"[teleop] 输入：{inputs.source}", flush=True)
    t_next = time.perf_counter()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        p = inputs.read()
        tele.tick(p, edges(p))
        disp.update()

        w, h = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, sim.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        disp.add_arrows(scn)
        mujoco.mjr_render(vp, scn, con)
        rows = disp.rows()
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           "\n".join(r[0] for r in rows), "\n".join(f"{r[1]}  {r[2]}" for r in rows), con)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp,
                           "green: command\norange: measured\nspeed\ninput", f"\n\n{tele.speed:.2f}\n{inputs.source}", con)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
            "pad: L2+A get up  L2+B lie down  L-stick vx/vy  R-stick yaw  D-pad up/down height, left/right speed  "
            "Y height 0  BACK damp  START reset\n"
            "keys: G get up  L lie down  W/S A/D Q/E move  X/Z height  C height 0  Space damp  R reset  +/- speed  Esc quit",
            None, con,
        )
        glfw.swap_buffers(win)
        t_next += tele.dt
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            t_next = time.perf_counter()
    glfw.terminate()


def main() -> None:
    p = argparse.ArgumentParser(description="Q1 遥控仿真（与真机同一状态机：起身 → E_tele 行走 → 趴下）")
    p.add_argument("--package", type=Path, default=DEFAULT_BASE, help="基础模型权重包（export_base_package.py）")
    p.add_argument("--skill", type=Path, default=None, help=f"起身 / 趴下技能包，默认 {DEFAULT_SKILL}；不存在时从站姿开始，只能行走")
    p.add_argument("--getup-ref", default="getup_facedown", help="参考库里的起身参考（名字或 motion.npz 路径）")
    p.add_argument("--liedown-ref", default="liedown_facedown", help="参考库里的趴下参考")
    p.add_argument("--speed-scale", type=float, default=1.0, help="满速缩放（与真机参数一致）")
    p.add_argument("--script", default=None, help="无界面脚本模式，如 'idle:1,fwd:3,turn_l:2'")
    p.add_argument("--video", type=Path, default=None, help="脚本模式下录 mp4")
    a = p.parse_args()

    base = Package.load(a.package)
    tele = Teleop(base, load_skill(a.skill, real=False), MuJoCoBackend(base), ref.find(a.getup_ref),
                  ref.find(a.liedown_ref), speed_scale=a.speed_scale)
    if a.script:
        run_script(tele, a.script, a.video)
    else:
        run_interactive(tele)


if __name__ == "__main__":
    main()
