"""Q1 手柄遥操作：启动时趴在平地上，L2+A 起身，之后基础模型 E_tele 跟手柄的 vx、vy、yaw、高度，L2+B 站直再趴下。

起身 / 趴下用单独训练的技能策略（export_package.py 导出的单段动作包，默认发布的 weights/getup_vpd_ft）跟踪参考库
apps/ref/skills/ 里的 getup_facedown / liedown_facedown；自己训练见 docs/apps.md#52-起身--趴下技能。
行走用基础模型包的 policy_tele.onnx。状态机见 machine.py。
没有技能包时仿真从站姿开始，只能行走（L2+A 回到站姿，L2+B 不可用）；真机必须给技能包。

真机（APP 开发者模式选「低层运控开发」、关掉高层运控；先吊带；source ROS2 Humble 与 aimdk_msgs；手柄插在跑本脚本的机器上）：
    infra/bin/q1py -m apps.teleop --backend real --speed-scale 0.5
仿真（Mac 上 GLFW 读手柄、开窗口；无手柄用 --script 自测）：
    infra/bin/q1py -m apps.teleop
    infra/bin/q1py -m apps.teleop --script 'idle:1,getup:12,fwd:3,turn_l:2,squat:1,rise:1,liedown:14'

手柄（Xbox 布局）：
    L2 + A  起身（趴着 / 阻尼时）         L2 + B  站直后趴下（行走时）        BACK  急停切阻尼
    左摇杆  前后 vx / 左右 vy             右摇杆左右  yaw                     十字键 ↑ / ↓  起身 / 下蹲，Y 高度回正
    十字键 ← / →  速度系数 −/+ 0.1（0.2 ~ 1.5，语音播报）                     START   仿真里重置

每个阶段有语音提示（--voice，见 voice.py）：真机默认机器人扬声器，仿真窗口默认本机，--script 默认不出声。
真机启动时没插手柄也照常起来，提示连接并每秒重新找设备；厂商运控 mc 在跑（不是「低层运控开发」模式）则拒绝启动。
开机自启见 apps/teleop/autostart/。
活动记录（recorder.py：每周期时间戳、状态、手柄、命令、关节位置 / 速度 / 目标，状态切换与按键事件）：真机默认记到
/robot/persist/log/kabutack_teleop/rec/，仿真加 --record 才记；--no-record 关掉。查看：python -m apps.teleop.recorder <rec 目录>。
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from apps import ref  # noqa: E402
from apps.teleop.gamepad import AutoPad, Edges, ScriptPad  # noqa: E402
from apps.teleop.machine import Teleop  # noqa: E402
from apps.teleop.voice import open_voice  # noqa: E402
from runtime import MuJoCoBackend, Package  # noqa: E402

DEFAULT_BASE = PROJECT / "weights" / "dual_v3b"
DEFAULT_SKILL = PROJECT / "weights" / "getup_vpd_ft"
REC_REAL = Path("/robot/persist/log/kabutack_teleop/rec")
REC_SIM = PROJECT / "infra" / "jobs" / "teleop_rec"
DRY_TOPIC = "/kabutack/dry_run/joint/command"
POLICY_CORE, POLICY_PRIO = 6, 60  # A72 大核（0–3 是 A53）；收发进程占 core 4、5，FIFO 70
EXIT_VENDOR_MC = 3  # autostart/start_teleop 见到它不重启


def _interrupt(signum, frame):
    # stop-app 对整个进程组发 SIGTERM，start_teleop 还会再转发一次：只响应第一次，别打断切阻尼
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def run_real(tele: Teleop, pad: AutoPad, sensor_timeout: float) -> None:
    # 后台启动时 SIGINT 可能被继承为忽略；SIGTERM 默认直接杀进程、不切阻尼
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    edges = Edges()
    t_print = 0.0
    try:
        try:
            tele.robot.wait_sensors(sensor_timeout)
        except (TimeoutError, RuntimeError) as e:
            print(f"[teleop] {e}", flush=True)
            tele.voice.say("没有收到关节状态，遥操作退出")
            raise SystemExit(1)
        print(f"[teleop] 手柄 {pad.path or '未连接'}  upright {tele.robot.upright():+.2f}", flush=True)
        while True:
            p = pad.read()
            tele.tick(p, edges(p))
            t_print += tele.dt
            if t_print >= 1.0 and tele.mode not in ("LYING", "DAMPED"):
                t_print = 0.0
                n_joint, n_pub = tele.robot.rates()
                print("[teleop] " + "  ".join(f"{k} {v}" for k, v in tele.status() if k and k != "mode")
                      + f"  upright {tele.robot.upright():+.2f}  超拍 {tele.robot.overruns}  收 / 发累计 {n_joint} / {n_pub}", flush=True)
    except KeyboardInterrupt:
        print("[teleop] Ctrl-C / SIGTERM：切阻尼退出", flush=True)
        tele.voice.say("遥操作退出，已切阻尼")
    finally:
        tele.robot.close()
        tele.voice.close()


def run_script(tele: Teleop, pad: ScriptPad) -> None:
    edges = Edges()
    seg, meas = None, []
    while not pad.done:
        if pad.segment != seg:
            if meas and tele.mode == "WALK":
                m = sum(meas) / len(meas)
                print(f"           {seg}: 实测 vx {m[0]:+.2f} vy {m[1]:+.2f} wz {m[2]:+.2f}  骨盆高 {tele.robot.pelvis_height():.3f} m", flush=True)
            seg, meas = pad.segment, []
            print(f"[script] {seg}", flush=True)
        p = pad.read()
        tele.tick(p, edges(p))
        if tele.mode == "WALK" and tele.t_mode > 1.0:
            meas.append(tele.robot.base_velocity())
    print(f"[script] 结束：{tele.mode}  骨盆高 {tele.robot.pelvis_height():.3f} m  upright {tele.robot.upright():.2f}", flush=True)


def run_window(tele: Teleop) -> None:
    import glfw
    import mujoco

    from apps.teleop.gamepad import GlfwPad

    glfw.init()
    pad = GlfwPad()
    print(f"[teleop] 手柄：{pad.name}", flush=True)
    win = glfw.create_window(1280, 800, "Q1 手柄遥操作（仿真）", None, None)
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
    glfw.set_key_callback(win, lambda w, key, sc, action, mods: key == glfw.KEY_ESCAPE and glfw.set_window_should_close(w, True))
    glfw.set_scroll_callback(win, lambda w, dx, dy: setattr(cam, "distance", min(8.0, max(0.6, cam.distance * (0.9 if dy > 0 else 1.1)))))
    edges = Edges()
    t_next = time.perf_counter()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        p = pad.read()
        tele.tick(p, edges(p))
        w, h = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, sim.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, con)
        rows = tele.status()
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows), con)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                           "L2+A get up   L2+B lie down   L-stick vx/vy   R-stick yaw   D-pad height   Y height 0   "
                           "BACK damp   START reset   Esc quit",
                           None, con)
        glfw.swap_buffers(win)
        t_next += tele.dt
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            t_next = time.perf_counter()
    glfw.terminate()
    tele.voice.close()


def load_skill(path: Path | None, real: bool) -> Package | None:
    """起身 / 趴下技能包：显式给了必须存在；缺省找 DEFAULT_SKILL，没有时仿真只能行走，真机退出。"""
    root = path or DEFAULT_SKILL
    if (root / "deploy.yaml").exists():
        return Package.load(root)
    if path is not None or real:
        sys.exit(f"起身 / 趴下技能包不存在：{root}（训练与导出见 docs/apps.md#5-手柄遥操作）")
    print("[teleop] 没有起身 / 趴下技能包（--skill）：从站姿开始，只能行走", flush=True)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Q1 手柄遥操作（起身 → E_tele 行走 → 趴下）")
    p.add_argument("--backend", choices=("sim", "real"), default="sim")
    p.add_argument("--package", type=Path, default=DEFAULT_BASE, help="基础模型权重包（export_base_package.py）")
    p.add_argument("--skill", type=Path, default=None,
                   help=f"起身 / 趴下技能包（export_package.py），默认 {DEFAULT_SKILL.relative_to(PROJECT)}；不存在时仿真只能行走")
    p.add_argument("--getup-ref", default="getup_facedown", help="参考库里的起身参考（名字或 motion.npz 路径）")
    p.add_argument("--liedown-ref", default="liedown_facedown", help="参考库里的趴下参考")
    p.add_argument("--speed-scale", type=float, default=1.0, help="满速缩放（真机初次联调建议 0.5）")
    p.add_argument("--pad", choices=("auto", "evdev", "joydev", "glfw"), default="auto",
                   help="auto：有 /dev/input/by-id/*event-joystick 用 evdev，其次 /dev/input/js* 用 joydev，否则 GLFW")
    p.add_argument("--pad-device", default=None, help="手柄设备，如 /dev/input/event0 或 /dev/input/js0")
    p.add_argument("--script", default=None, help="仿真无界面自测，如 'idle:1,getup:12,fwd:3,liedown:14'")
    p.add_argument("--dry-run", action="store_true", help=f"真机演练：全流程照跑，指令发到 {DRY_TOPIC}，电机收不到")
    p.add_argument("--voice", choices=("auto", "robot", "local", "off"), default="auto",
                   help="阶段语音提示：robot 机器人扬声器（PlayTts），local 本机 say / espeak-ng；auto 真机 robot、仿真窗口 local、--script off")
    p.add_argument("--pad-lost", choices=("damp", "liedown"), default="damp",
                   help="动作中手柄断开：damp 立即切阻尼；liedown 起身 / 趴下照常放完，行走中自动站直趴下")
    p.add_argument("--sensor-timeout", type=float, default=10.0, help="真机启动时等关节 / IMU 状态的秒数（开机自启时设长些）")
    p.add_argument("--record", type=Path, nargs="?", const=REC_SIM, default=None,
                   help=f"活动记录目录（每次启动建一个 rec_<时间>/）；真机缺省 {REC_REAL}，仿真缺省不记、只写 --record 则记到 {REC_SIM.relative_to(PROJECT)}")
    p.add_argument("--no-record", action="store_true", help="不记录")
    a = p.parse_args()
    if a.voice == "auto":
        a.voice = "robot" if a.backend == "real" else "off" if a.script else "local"

    if a.backend == "real":
        from runtime.primebot_io import vendor_mc_running

        # 厂商运控在跑时既不能下发关节指令，也不能独占手柄（厂商遥控程序也读这个手柄）
        if vendor_mc_running():
            voice = open_voice(a.voice)
            voice.say("高层运控正在运行，遥操作不启动")
            voice.close()
            print("[teleop] 厂商运控 mc_app_main 在运行：APP 开发者模式选「低层运控开发」后重启机器人", flush=True)
            sys.exit(EXIT_VENDOR_MC)

    base = Package.load(a.package)
    skill = load_skill(a.skill, a.backend == "real")
    getup_ref, liedown_ref = ref.find(a.getup_ref), ref.find(a.liedown_ref)

    if a.backend == "real":
        from runtime import PrimeBotBackend
        from runtime.primebot_io import realtime

        if a.script:
            sys.exit("--script 只用于仿真")
        if a.pad == "glfw":
            sys.exit("真机手柄只支持 evdev / joydev")
        pad = AutoPad(a.pad, a.pad_device)
        voice = open_voice(a.voice)  # 先于 realtime()，免得语音进程继承策略进程的绑核与 FIFO
        # 演练时电机不跟指令，插值期间的跟踪偏差保护会必然触发
        robot = PrimeBotBackend(base, **({"cmd_topic": DRY_TOPIC, "ramp_track_lim": float("inf")} if a.dry_run else {}))
        print(f"[teleop] 策略进程 {realtime(POLICY_CORE, POLICY_PRIO)}" + (f"；演练：指令发到 {DRY_TOPIC}" if a.dry_run else ""), flush=True)
    elif a.dry_run:
        sys.exit("--dry-run 只用于 --backend real")
    else:
        robot = MuJoCoBackend(base)
        if a.voice == "robot":
            sys.exit("--voice robot 只用于 --backend real")
        voice = open_voice(a.voice)
    tele = Teleop(base, skill, robot, getup_ref, liedown_ref, speed_scale=a.speed_scale, voice=voice, pad_lost=a.pad_lost)
    rec_root = None if a.no_record else a.record or (REC_REAL if a.backend == "real" else None)
    if rec_root is not None:
        from apps.teleop.recorder import Recorder

        tele.recorder = Recorder(rec_root, base.joint_names, base.control_dt,
                                 {"base": a.package, "skill": None if skill is None else skill.root},
                                 {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})

    why = "正常退出"
    try:
        if a.backend == "real":
            return run_real(tele, pad, a.sensor_timeout)
        if a.script:
            return run_script(tele, ScriptPad(a.script, base.control_dt))
        run_window(tele)
    except BaseException as e:
        why = f"{type(e).__name__}: {e}"
        raise
    finally:
        if tele.recorder is not None:
            tele.recorder.close(why)


if __name__ == "__main__":
    main()
