"""手柄输入，统一成 Xbox 布局的 ``Pad``：摇杆 [-1, 1]（上 / 左为正）、扳机 [0, 1]、按键布尔。

EvdevPad   Linux /dev/input/event*（不依赖第三方库，真机上用；Q1 运控板内核没开 joydev，没有 js*）。
           键 / 轴码按北通 BTP-KP20D 接收器 BFM 模式实测（标准 evdev 手柄码）；
           其他手柄先用 ``python -m apps.teleop.gamepad`` 看实际编码，再改 EVDEV_AXES / EVDEV_BUTTONS。
JoydevPad  Linux /dev/input/js*。轴 / 键号按 xpad 驱动（Xbox 及兼容手柄的 X 模式）。
GlfwPad    GLFW 的 SDL 标准映射（Mac / 桌面仿真）。
ScriptPad  无手柄：按脚本逐帧给出手柄状态（自测用）。
AutoPad    真机用：包一层 EvdevPad / JoydevPad，没插或断开时给 connected = False 的空 Pad，定期重新找设备。
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct
import threading
import time
from dataclasses import dataclass, field

BUTTONS = ("A", "B", "X", "Y", "LB", "RB", "BACK", "START", "UP", "DOWN", "LEFT", "RIGHT")
JOYDEV_AXES = {"lx": 0, "ly": 1, "lt": 2, "rx": 3, "ry": 4, "rt": 5, "hat_x": 6, "hat_y": 7}
JOYDEV_BUTTONS = {"A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5, "BACK": 6, "START": 7}
EVDEV_AXES = {"lx": 0x00, "ly": 0x01, "rx": 0x02, "ry": 0x05, "rt": 0x09, "lt": 0x0A, "hat_x": 0x10, "hat_y": 0x11}
EVDEV_BUTTONS = {"A": 0x130, "B": 0x131, "X": 0x133, "Y": 0x134, "LB": 0x136, "RB": 0x137, "BACK": 0x13A, "START": 0x13B}
# 扳机同时报数字键；模拟量可能按不满（实测 LT 峰值只有 0.34），按下判定取两者的大值
EVDEV_TRIGGER_KEYS = {"lt": 0x138, "rt": 0x139}
TRIGGER_ON = 0.5
DEADZONE = 0.12


@dataclass
class Pad:
    lx: float = 0.0  # 左摇杆，左为正
    ly: float = 0.0  # 左摇杆，上为正
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0  # L2
    rt: float = 0.0  # R2
    buttons: dict = field(default_factory=lambda: dict.fromkeys(BUTTONS, False))
    connected: bool = True

    @property
    def l2(self) -> bool:
        return self.lt > TRIGGER_ON

    @property
    def r2(self) -> bool:
        return self.rt > TRIGGER_ON


def deadzone(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else (v - DEADZONE * (1 if v > 0 else -1)) / (1.0 - DEADZONE)


class Edges:
    """按键按下沿：本帧按下、上一帧没按。"""

    def __init__(self):
        self.prev = dict.fromkeys(BUTTONS, False)

    def __call__(self, pad: Pad) -> set[str]:
        down = {b for b in BUTTONS if pad.buttons[b] and not self.prev[b]}
        self.prev = dict(pad.buttons)
        return down


def _ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("E") << 8) | nr


EV_FMT = "llHHi"  # struct input_event：timeval、type、code、value
EV_SIZE = struct.calcsize(EV_FMT)
EV_KEY, EV_ABS = 0x01, 0x03
EVIOCGRAB = _ioc(1, 0x90, 4)


class EvdevPad:
    """后台线程读 input_event；轴按设备报告的 [min, max] 归一化（摇杆 [-1, 1]、扳机 [0, 1]）；断开时 connected = False。

    打开后 EVIOCGRAB 独占：厂商遥控程序也读同一设备，且它的组合键（LT+A、LT+B 等）和这里重叠。"""

    def __init__(self, device: str | None = None):
        devs = [device] if device else sorted(glob.glob("/dev/input/by-id/*event-joystick"))
        if not devs or not os.path.exists(devs[0]):
            raise FileNotFoundError("没找到手柄（/dev/input/by-id/*event-joystick）：检查 USB 接收器或蓝牙配对，或用 --pad-device 指定")
        self.path = devs[0]
        self.f = open(self.path, "rb", buffering=0)
        try:
            fcntl.ioctl(self.f, EVIOCGRAB, 1)
            self.grabbed = True
        except OSError:
            self.grabbed = False
        self.ranges: dict[int, tuple[int, int]] = {}
        self.raw: dict[int, int] = {}
        for code in EVDEV_AXES.values():
            buf = bytearray(24)
            try:
                fcntl.ioctl(self.f, _ioc(2, 0x40 + code, 24), buf)  # EVIOCGABS
            except OSError:
                continue
            value, lo, hi = struct.unpack("6i", buf)[:3]
            if hi > lo:  # 接收器上电先以 NS 模式枚举，有的轴报 min == max
                self.ranges[code], self.raw[code] = (lo, hi), value
        self.btns: dict[int, bool] = {}
        self.connected = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        try:
            while True:
                ev = self.f.read(EV_SIZE)
                if len(ev) < EV_SIZE:
                    break
                _, _, typ, code, value = struct.unpack(EV_FMT, ev)
                with self.lock:
                    if typ == EV_KEY:
                        self.btns[code] = bool(value)
                    elif typ == EV_ABS:
                        self.raw[code] = value
        except (OSError, ValueError):
            pass
        self.connected = False

    def close(self) -> None:
        self.f.close()

    def _norm(self, code: int, unipolar: bool) -> float:
        if code not in self.ranges:
            return 0.0
        lo, hi = self.ranges[code]
        v = self.raw.get(code, lo if unipolar else (lo + hi) / 2)
        return (v - lo) / (hi - lo) if unipolar else (2.0 * v - lo - hi) / (hi - lo)

    @property
    def axes(self) -> dict[int, float]:
        return {c: self._norm(c, c in (EVDEV_AXES["lt"], EVDEV_AXES["rt"])) for c in self.ranges}

    def read(self) -> Pad:
        with self.lock:
            a = {k: self._norm(c, k in ("lt", "rt")) for k, c in EVDEV_AXES.items()}
            bt = dict(self.btns)
        p = Pad(lx=-deadzone(a["lx"]), ly=-deadzone(a["ly"]), rx=-deadzone(a["rx"]), ry=-deadzone(a["ry"]),
                lt=max(a["lt"], float(bt.get(EVDEV_TRIGGER_KEYS["lt"], False))),
                rt=max(a["rt"], float(bt.get(EVDEV_TRIGGER_KEYS["rt"], False))), connected=self.connected)
        for name, c in EVDEV_BUTTONS.items():
            p.buttons[name] = bt.get(c, False)
        p.buttons.update(UP=a["hat_y"] < -0.5, DOWN=a["hat_y"] > 0.5, LEFT=a["hat_x"] < -0.5, RIGHT=a["hat_x"] > 0.5)
        return p


class JoydevPad:
    """后台线程读 js_event（u32 时间、s16 值、u8 类型、u8 编号）；断开时 connected = False。"""

    def __init__(self, device: str | None = None):
        devs = [device] if device else sorted(glob.glob("/dev/input/js*"))
        if not devs or not os.path.exists(devs[0]):
            raise FileNotFoundError("没找到手柄（/dev/input/js*）：检查 USB 接收器或蓝牙配对，或用 --pad-device 指定")
        self.path = devs[0]
        self.axes: dict[int, float] = {}
        self.btns: dict[int, bool] = {}
        self.connected = True
        self.lock = threading.Lock()
        self.f = open(self.path, "rb", buffering=0)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        try:
            while True:
                ev = self.f.read(8)
                if len(ev) < 8:
                    break
                _, value, typ, num = struct.unpack("IhBB", ev)
                with self.lock:
                    if typ & 0x01:
                        self.btns[num] = bool(value)
                    elif typ & 0x02:
                        self.axes[num] = value / 32767.0
        except (OSError, ValueError):
            pass
        self.connected = False

    def close(self) -> None:
        self.f.close()

    def read(self) -> Pad:
        with self.lock:
            ax, bt = dict(self.axes), dict(self.btns)
        a = lambda k, d=0.0: ax.get(JOYDEV_AXES[k], d)  # noqa: E731
        p = Pad(lx=-deadzone(a("lx")), ly=-deadzone(a("ly")), rx=-deadzone(a("rx")), ry=-deadzone(a("ry")),
                lt=(a("lt", -1.0) + 1.0) / 2.0, rt=(a("rt", -1.0) + 1.0) / 2.0, connected=self.connected)
        for name, i in JOYDEV_BUTTONS.items():
            p.buttons[name] = bt.get(i, False)
        hx, hy = a("hat_x"), a("hat_y")
        p.buttons.update(UP=hy < -0.5, DOWN=hy > 0.5, LEFT=hx < -0.5, RIGHT=hx > 0.5)
        return p


class GlfwPad:
    """调用方负责 glfw.init() 与每帧 glfw.poll_events()（仿真窗口里已有）。"""

    MAP = {"A": "A", "B": "B", "X": "X", "Y": "Y", "LB": "LEFT_BUMPER", "RB": "RIGHT_BUMPER", "BACK": "BACK", "START": "START",
           "UP": "DPAD_UP", "DOWN": "DPAD_DOWN", "LEFT": "DPAD_LEFT", "RIGHT": "DPAD_RIGHT"}
    # GLFW 内置映射库里没有的手柄（SDL gamecontrollerdb 格式，按实测编号写）
    EXTRA_MAPPINGS = (
        "030000005e040000e002000000010000,Xbox Wireless Controller,a:b0,b:b1,x:b2,y:b3,leftshoulder:b4,rightshoulder:b5,"
        "back:b6,start:b7,guide:b8,leftstick:b9,rightstick:b10,dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,"
        "leftx:a0,lefty:a1,lefttrigger:a2,rightx:a3,righty:a4,righttrigger:a5,platform:Mac OS X,",  # 蓝牙连 Mac
    )

    def __init__(self):
        import glfw

        self.glfw = glfw
        glfw.update_gamepad_mappings("\n".join(self.EXTRA_MAPPINGS) + "\n")
        self.jid = next((j for j in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1) if glfw.joystick_is_gamepad(j)), None)
        if self.jid is None:
            raise FileNotFoundError("GLFW 没识别到手柄（需要 SDL 标准映射的手柄，如 Xbox / PS / 8BitDo X 模式）")
        self.name = glfw.get_gamepad_name(self.jid)

    def read(self) -> Pad:
        g = self.glfw
        s = g.get_gamepad_state(self.jid)
        if not s:
            return Pad(connected=False)
        ax = s.axes
        p = Pad(lx=-deadzone(ax[g.GAMEPAD_AXIS_LEFT_X]), ly=-deadzone(ax[g.GAMEPAD_AXIS_LEFT_Y]),
                rx=-deadzone(ax[g.GAMEPAD_AXIS_RIGHT_X]), ry=-deadzone(ax[g.GAMEPAD_AXIS_RIGHT_Y]),
                lt=(ax[g.GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) / 2.0, rt=(ax[g.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) / 2.0)
        for name, key in self.MAP.items():
            p.buttons[name] = bool(s.buttons[getattr(g, f"GAMEPAD_BUTTON_{key}")])
        return p


class ScriptPad:
    """'动作:秒' 逗号分隔。组合键动作（getup = L2+A、liedown = L2+B、damp = BACK、reset = START）
    只在段首按一帧，其余是摇杆 / 十字键在整段内保持。脚本放完 ``done`` 为 True。"""

    STICKS = {
        "idle": {}, "fwd": {"ly": 1.0}, "back": {"ly": -1.0}, "left": {"lx": 1.0}, "right": {"lx": -1.0},
        "turn_l": {"rx": 1.0}, "turn_r": {"rx": -1.0}, "fwd_turn_l": {"ly": 1.0, "rx": 0.6},
        "squat": {"DOWN": True}, "rise": {"UP": True}, "faster": {"RIGHT": True}, "slower": {"LEFT": True},
    }
    COMBOS = {"getup": ("A", ("lt",)), "liedown": ("B", ("lt",)), "damp": ("BACK", ()),
              "reset": ("START", ())}

    def __init__(self, script: str, dt: float):
        self.frames: list[tuple[str, int]] = []
        for seg in script.split(","):
            name, sec = seg.strip().split(":")
            if name not in self.STICKS and name not in self.COMBOS:
                raise ValueError(f"未知脚本动作 {name}（{sorted(self.STICKS) + sorted(self.COMBOS)}）")
            self.frames.append((name, int(round(float(sec) / dt))))
        self.seg, self.k = 0, 0

    @property
    def done(self) -> bool:
        return self.seg >= len(self.frames)

    @property
    def segment(self) -> str:
        return "" if self.done else self.frames[self.seg][0]

    def read(self) -> Pad:
        p = Pad()
        if self.done:
            return p
        name, n = self.frames[self.seg]
        if name in self.COMBOS:
            if self.k == 0:
                btn, triggers = self.COMBOS[name]
                p.buttons[btn] = True
                for t in triggers:
                    setattr(p, t, 1.0)
        else:
            for k, v in self.STICKS[name].items():
                if k in p.buttons:
                    p.buttons[k] = v
                else:
                    setattr(p, k, v)
        self.k += 1
        if self.k >= n:
            self.seg, self.k = self.seg + 1, 0
        return p


def open_pad(kind: str, device: str | None = None):
    if kind == "evdev" or (kind == "auto" and (glob.glob("/dev/input/by-id/*event-joystick") or (device or "").startswith("/dev/input/event"))):
        return EvdevPad(device)
    if kind == "joydev" or (kind == "auto" and glob.glob("/dev/input/js*")):
        return JoydevPad(device)
    return GlfwPad()


class AutoPad:
    """没插或断开时 read() 给 connected = False 的空 Pad，每 retry_s 重新找一次设备
    （接收器拔插、重新配对后 event 编号会变，所以每次都按 by-id / js* 重新 glob；刚枚举出来时权限可能还没设好，打开失败就下次再试）。"""

    def __init__(self, kind: str = "auto", device: str | None = None, retry_s: float = 1.0):
        if kind == "glfw":
            raise ValueError("AutoPad 只支持 evdev / joydev")
        self.kind, self.device, self.retry_s = kind, device, retry_s
        self.pad: EvdevPad | JoydevPad | None = None
        self.t_retry = 0.0
        self._try_open()

    def _open(self) -> EvdevPad | JoydevPad:
        if self.kind == "evdev" or (self.kind == "auto" and (self.device or "").startswith("/dev/input/event")):
            return EvdevPad(self.device)
        if self.kind == "joydev" or self.device:
            return JoydevPad(self.device)
        if glob.glob("/dev/input/by-id/*event-joystick"):
            return EvdevPad()
        return JoydevPad()

    def _try_open(self) -> None:
        self.t_retry = time.monotonic() + self.retry_s
        try:
            self.pad = self._open()
        except OSError:
            return
        name = os.path.basename(os.path.realpath(self.pad.path)) if self.pad.path.startswith("/dev/input/by-id/") else ""
        print(f"[pad] 已连接 {self.pad.path}{f'（{name}）' if name else ''}{'（已独占）' if getattr(self.pad, 'grabbed', False) else ''}", flush=True)

    def _drop(self, why: str) -> None:
        print(f"[pad] {self.pad.path} {why}", flush=True)
        try:
            self.pad.close()
        except OSError:
            pass
        self.pad = None

    @property
    def path(self) -> str | None:
        return self.pad.path if self.pad else None

    def read(self) -> Pad:
        if self.pad is not None and not self.pad.connected:
            self._drop("断开")
        if self.pad is None and time.monotonic() >= self.t_retry:
            self._try_open()
        if self.pad is None:
            return Pad(connected=False)
        try:
            return self.pad.read()
        except Exception as e:  # noqa: BLE001  读手柄出错不能拖垮控制循环，当断开处理、稍后重找
            self._drop(f"读取出错（{type(e).__name__}: {e}），当断开处理")
            return Pad(connected=False)


if __name__ == "__main__":
    import sys
    import time

    dev = sys.argv[1] if len(sys.argv) > 1 else None
    pad = JoydevPad(dev) if (dev or "").startswith("/dev/input/js") or not (dev or glob.glob("/dev/input/by-id/*event-joystick")) \
        else EvdevPad(dev)
    print(f"{pad.path}：按键和摇杆的原始编号（evdev 为十六进制码，Ctrl-C 退出）")
    fmt = hex if isinstance(pad, EvdevPad) else int
    last = None
    while True:
        with pad.lock:
            cur = ({fmt(k): round(v, 2) for k, v in pad.axes.items() if abs(v) > 0.1}, sorted(fmt(k) for k, v in pad.btns.items() if v))
        if cur != last:
            print("axes", cur[0], "buttons", cur[1], flush=True)
            last = cur
        time.sleep(0.05)
