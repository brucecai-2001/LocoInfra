"""PrimeBot 真机 ROS2 收发进程。运控板上 rclpy 单进程收 1 kHz 状态、发 500 Hz 指令再加推理会互相抢 GIL
（实测下发只到 226 Hz、状态断流 > 100 ms），所以收、发各占一个进程、一个大核、SCHED_FIFO，
与策略进程经共享内存（multiprocessing.Array，带锁）交换数据。

只依赖 numpy 与 rclpy / aimdk_msgs；由 backend.PrimeBotBackend 以 spawn 方式启动。
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np

IDLE, HOLD, DAMP = 0.0, 1.0, 2.0
FAULTS = {1: "/aima/hal/joint/state 关节与权重包不一致", 2: "关节 / IMU 状态断流", 3: "策略超时未调用 step()", 4: "策略进程退出",
          5: "关节实际位置偏离指令过大"}
DAMP_KD = 5.0


class Layout:
    """共享内存里各字段的切片（float64）。"""

    def __init__(self, n: int):
        self.n = n
        off = 0
        for name, dim in (("q", n), ("qd", n), ("gyro", 3), ("quat", 4), ("t_joint", 1), ("t_imu", 1),
                          ("mode", 1), ("target", n), ("cmd", n), ("kp", n), ("kd", n), ("t_step", 1),
                          ("fault", 1), ("stop", 1), ("n_pub", 1), ("n_joint", 1), ("hw_idx", n), ("track_lim", 1), ("worst", 2)):
            setattr(self, name, slice(off, off + dim))
            off += dim
        self.size = off


def view(arr) -> np.ndarray:
    return np.frombuffer(arr.get_obj(), dtype=np.float64)


def realtime(core: int | None, prio: int | None) -> str:
    """把当前进程绑到 core、设 SCHED_FIFO prio；失败只返回说明，不抛异常。"""
    msg = []
    if core is not None:
        try:
            os.sched_setaffinity(0, {core})
            msg.append(f"core {core}")
        except OSError as e:
            msg.append(f"绑核 {core} 失败（{e}）")
    if prio:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
            msg.append(f"FIFO {prio}")
        except (OSError, PermissionError) as e:
            msg.append(f"SCHED_FIFO 失败（{e}）")
    return "，".join(msg)


def vendor_mc_running() -> bool:
    """厂商运控 mc_app_main 在跑（标准模式或没选「低层运控开发」）时，再下发关节指令会和它冲突。"""
    for pid in os.listdir("/proc"):
        try:
            with open(f"/proc/{pid}/comm") as f:
                if f.read().strip() == "mc_app_main":
                    return True
        except (OSError, ValueError):
            continue
    return False


def _qos():
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                      reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def init_ros():
    """ROS 子进程里初始化 rclpy。Ctrl-C、process_manager 的 stop-app 会对整个进程组发 SIGINT / SIGTERM，
    rclpy 默认的信号处理会立刻关掉上下文，下发进程就赶在主进程切阻尼之前停发了；所以不装它、两个信号都忽略，
    子进程只按主进程置的 stop 标志或主进程退出来结束。"""
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    return rclpy


def sub_worker(arr, names: list[str], core, prio, parent: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    realtime(core, prio)
    from sensor_msgs.msg import Imu

    from aimdk_msgs.msg import JointStateArray

    L, buf = Layout(len(names)), view(arr)
    rclpy = init_ros()
    node = rclpy.create_node("kabutack_primebot_sub")
    idx: list[int] | None = None

    def on_joint(msg) -> None:
        nonlocal idx
        js = msg.joints
        if idx is None:
            pos = {j.name: i for i, j in enumerate(js)}
            missing = [n for n in names if n not in pos]
            if missing or len(js) != len(names):
                print(f"[primebot] {FAULTS[1]}：缺 {missing}，状态里 {len(js)} 个关节", flush=True)
                with arr.get_lock():
                    buf[L.fault] = 1
                    if buf[L.mode][0] == HOLD:
                        buf[L.mode] = DAMP
                return
            idx = [pos[n] for n in names]
            with arr.get_lock():
                buf[L.hw_idx] = idx
        q = [js[i].position for i in idx]
        qd = [js[i].velocity for i in idx]
        with arr.get_lock():
            buf[L.q], buf[L.qd], buf[L.t_joint] = q, qd, time.monotonic()
            buf[L.n_joint] += 1

    def on_imu(msg) -> None:
        w, o = msg.angular_velocity, msg.orientation
        with arr.get_lock():
            buf[L.gyro] = (w.x, w.y, w.z)
            buf[L.quat] = (o.w, o.x, o.y, o.z)
            buf[L.t_imu] = time.monotonic()

    node.create_subscription(JointStateArray, "/aima/hal/joint/state", on_joint, _qos())
    node.create_subscription(Imu, "/aima/hal/imu/state", on_imu, _qos())
    try:
        while buf[L.stop][0] == 0 and os.getppid() == parent:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def pub_worker(arr, names: list[str], topic: str, cmd_hz: float, max_delta: float, stale_s: float, policy_timeout_s: float,
               core, prio, parent: int) -> None:
    """以 cmd_hz 持续下发全身指令：hold 时按 max_delta 限幅逼近 target；状态断流、策略超时、策略进程退出都切阻尼。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    realtime(core, prio)
    from aimdk_msgs.msg import JointCommand, JointCommandArray

    L, buf = Layout(len(names)), view(arr)
    rclpy = init_ros()
    node = rclpy.create_node("kabutack_primebot_pub")
    pub = node.create_publisher(JointCommandArray, topic, _qos())
    msg, slot = None, None  # 按 /aima/hal/joint/state 的关节顺序下发（同 SDK 示例），收到第一帧状态后建
    period, seq = 1.0 / cmd_hz, 0
    orphan_until = None
    t_next = time.monotonic()
    try:
        while buf[L.stop][0] == 0:
            now = time.monotonic()
            with arr.get_lock():
                if orphan_until is None and os.getppid() != parent:
                    orphan_until = now + 0.5
                    buf[L.fault] = 4
                    if buf[L.mode][0] == HOLD:
                        buf[L.mode] = DAMP
                if buf[L.mode][0] == HOLD:
                    tj, ti = buf[L.t_joint][0], buf[L.t_imu][0]
                    if tj == 0 or ti == 0 or now - tj > stale_s or now - ti > stale_s:
                        buf[L.fault], buf[L.mode] = 2, DAMP
                    elif now - buf[L.t_step][0] > policy_timeout_s:
                        buf[L.fault], buf[L.mode] = 3, DAMP
                    else:
                        dev = np.abs(buf[L.q] - buf[L.cmd])
                        k = int(np.argmax(dev))
                        if dev[k] > buf[L.worst][0]:
                            buf[L.worst] = (dev[k], k)
                        if dev[k] > buf[L.track_lim][0]:
                            buf[L.fault], buf[L.mode] = 5, DAMP
                        else:
                            buf[L.cmd] = buf[L.cmd] + np.clip(buf[L.target] - buf[L.cmd], -max_delta, max_delta)
                mode = buf[L.mode][0]
                cmd, kp, kd = buf[L.cmd].tolist(), buf[L.kp].tolist(), buf[L.kd].tolist()
                if msg is None and buf[L.n_joint][0] > 0:
                    slot = [int(x) for x in buf[L.hw_idx]]
                    msg = JointCommandArray()
                    msg.joints = [JointCommand() for _ in names]
                    for i, n in enumerate(names):
                        msg.joints[slot[i]].name = n
            if orphan_until is not None and now > orphan_until:
                break
            if mode != IDLE and msg is not None:
                damp = mode == DAMP
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.sequence = seq
                seq += 1
                for i in range(len(names)):
                    j = msg.joints[slot[i]]
                    j.position = cmd[i]
                    j.velocity = 0.0
                    j.effort = 0.0
                    j.stiffness = 0.0 if damp else kp[i]
                    j.damping = DAMP_KD if damp else kd[i]
                pub.publish(msg)
                with arr.get_lock():
                    buf[L.n_pub] += 1
            t_next += period
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.monotonic()
    finally:
        node.destroy_node()
        rclpy.shutdown()
