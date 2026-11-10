"""遥操作阶段语音提示。新提示打断正在播的旧提示；播报失败只打印，不影响控制。

    robot：真机扬声器，调 /aimdk_5Fmsgs/srv/PlayTts（无网络时只能播已缓存过的文本，固定提示词播过一次即缓存）。
           rclpy 跑在单独的 spawn 子进程里，不占策略进程的 GIL，并退回普通调度、绑到小核。
    local：跑本脚本的机器上播（macOS say，Linux espeak-ng）。
    off：  不出声。
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys

TTS_SERVICE = "/aimdk_5Fmsgs/srv/PlayTts"
TTS_DOMAIN = "kabutack_teleop"
LITTLE_CORES = (0, 1, 2, 3)  # A53；A72 大核 4–7 留给收发与策略进程
SERVICE_WAIT_S = 5.0  # 开机自启时 audio_manager 可能还没起来
CALL_TRIES, CALL_TIMEOUT_S = 3, 2.0


class Silent:
    def say(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass


class LocalVoice(Silent):
    def __init__(self):
        self._p: subprocess.Popen | None = None
        if sys.platform == "darwin" and shutil.which("say"):
            voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
            self.cmd = ["say", "-v", "Tingting"] if "Tingting" in voices else ["say"]
        elif shutil.which("espeak-ng"):
            self.cmd = ["espeak-ng", "-v", "cmn"]
        else:
            self.cmd = None
            print("[voice] 本机没有 say / espeak-ng，语音提示关闭", flush=True)

    def say(self, text: str) -> None:
        if self.cmd is None:
            return
        self.close()
        self._p = subprocess.Popen(self.cmd + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self) -> None:
        if self._p is not None and self._p.poll() is None:
            self._p.terminate()


def _tts_worker(q, parent: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # 进程组收到停止信号时还要播完“已切阻尼”，由主进程发 None 结束
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        os.sched_setaffinity(0, set(LITTLE_CORES) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    from aimdk_msgs.msg import CommonRequest, TtsPriorityLevel
    from aimdk_msgs.srv import PlayTts

    from runtime.primebot_io import init_ros

    rclpy = init_ros()
    node = rclpy.create_node("kabutack_teleop_tts")
    cli = node.create_client(PlayTts, TTS_SERVICE)
    seq = 0
    try:
        while os.getppid() == parent:
            try:
                items = [q.get(timeout=0.2)]
            except queue.Empty:
                continue
            while True:  # 积压时只播最新一条
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
            texts = [t for t in items if t is not None]
            if texts and not cli.service_is_ready() and not cli.wait_for_service(timeout_sec=SERVICE_WAIT_S):
                print(f"[voice] {TTS_SERVICE} 不可用，丢弃：{texts[-1]}", flush=True)
            elif texts:
                req = PlayTts.Request()
                req.header = CommonRequest()
                req.header.header.stamp = node.get_clock().now().to_msg()
                req.tts_req.text = texts[-1]
                req.tts_req.domain = TTS_DOMAIN
                req.tts_req.trace_id = f"{TTS_DOMAIN}_{seq}"
                req.tts_req.is_interrupted = True
                req.tts_req.priority_weight = 0
                req.tts_req.priority_level.value = TtsPriorityLevel.INTERACTION_L6
                seq += 1
                for _ in range(CALL_TRIES):  # 刚发现服务时第一个请求可能丢（SDK 示例同样重试）
                    fut = cli.call_async(req)
                    rclpy.spin_until_future_complete(node, fut, timeout_sec=CALL_TIMEOUT_S)
                    if fut.done():
                        break
                    cli.remove_pending_request(fut)
                resp = fut.result() if fut.done() else None
                if resp is None or not resp.tts_resp.is_success:
                    why = "超时" if resp is None else resp.tts_resp.error_message
                    print(f"[voice] 播报失败（{why}）：{texts[-1]}", flush=True)
            if None in items:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


class RobotVoice(Silent):
    """须在策略进程调 realtime() 之前创建：子进程会继承父进程的绑核与调度策略。"""

    def __init__(self):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._q = ctx.Queue()
        self._p = ctx.Process(target=_tts_worker, args=(self._q, os.getpid()), daemon=True)
        self._p.start()

    def say(self, text: str) -> None:
        self._q.put_nowait(text)

    def close(self) -> None:
        if not self._p.is_alive():
            return
        self._q.put_nowait(None)
        self._p.join(timeout=SERVICE_WAIT_S + CALL_TRIES * CALL_TIMEOUT_S + 1.0)
        if self._p.is_alive():
            self._p.terminate()


def open_voice(kind: str) -> Silent:
    return {"robot": RobotVoice, "local": LocalVoice, "off": Silent}[kind]()
