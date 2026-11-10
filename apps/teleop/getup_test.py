"""真机单独测起身（不接手柄、不接 E_tele）：趴在平地上，技能策略跟踪起身参考，站起后继续喂参考末帧站住；
回车 → 技能策略跟踪趴下参考 → 切阻尼。Ctrl-C 随时切阻尼。全程关节、IMU、策略输出录进 npz。

在运控板上跑（APP 开发者模式选「低层运控开发」、关掉高层运控；吊带放松到不碍起身但能兜住）：
    source /opt/ros/humble/setup.bash && source /robot/software/interface/local_setup.bash
    python3 -m apps.teleop.getup_test --dry-run     # 演练：全流程照跑，指令发到假话题，电机不动
    python3 -m apps.teleop.getup_test --hold-only   # 只保持当前姿态 6 s，确认下发指令时机器人不乱动
    python3 -m apps.teleop.getup_test               # 起身 → 站住 → 回车趴下
    python3 -m apps.teleop.getup_test --no-liedown  # 站住时回车直接切阻尼（吊带要兜住）
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from apps import ref  # noqa: E402
from runtime import ObsHistory, Package, PrimeBotBackend, SkillModel  # noqa: E402
from runtime.body_ref import BodyRef  # noqa: E402
from runtime.primebot_io import realtime  # noqa: E402
from apps.teleop.__main__ import DEFAULT_BASE, DEFAULT_SKILL, DRY_TOPIC, POLICY_CORE, POLICY_PRIO  # noqa: E402
from apps.teleop.machine import FALL_UPRIGHT, LYING_UPRIGHT  # noqa: E402


class Log:
    def __init__(self):
        self.rows: dict[str, list] = {k: [] for k in ("t", "phase", "frame", "q", "qd", "gyro", "grav", "target", "action", "infer_ms")}

    def add(self, phase: int, frame: int, s, target, action, infer_ms: float) -> None:
        for k, v in (("t", time.monotonic()), ("phase", phase), ("frame", frame), ("q", s.joint_pos), ("qd", s.joint_vel),
                     ("gyro", s.base_ang_vel), ("grav", s.projected_gravity), ("target", target), ("action", action),
                     ("infer_ms", infer_ms)):
            self.rows[k].append(v)

    def save(self, path: Path, **meta) -> None:
        if not self.rows["t"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in self.rows.items()}, **{k: np.asarray(v) for k, v in meta.items()})
        print(f"[getup] 记录 {len(self.rows['t'])} 帧 → {path}", flush=True)


class Runner:
    def __init__(self, skill: Package, robot: PrimeBotBackend, log: Log):
        self.skill, self.robot, self.log = skill, robot, log
        self.model = SkillModel(skill)
        self.hist = ObsHistory(skill)
        self.q0 = skill.default_joint_pos
        self.last_action = np.zeros(len(self.q0), dtype=np.float32)
        self._rates = None

    def begin(self) -> None:
        self.last_action[:] = 0.0
        self.hist.reset(self.robot.sense().obs_frame(self.q0, self.last_action))

    def step(self, r: BodyRef, i: int, phase: int) -> None:
        t0 = time.perf_counter()
        action, target = self.model.act(self.hist.vector(), r.goal(i, self.robot.torso_quat()))
        infer_ms = (time.perf_counter() - t0) * 1e3
        self.robot.step(target)
        self.last_action = np.asarray(action, dtype=np.float32)
        s = self.robot.sense()
        self.hist.push(s.obs_frame(self.q0, self.last_action))
        self.log.add(phase, i, s, target, action, infer_ms)

    def play(self, r: BodyRef, name: str, phase: int) -> None:
        r.start(self.robot.torso_quat())
        dt = self.skill.control_dt
        n = r.T - 1
        print(f"[getup] {name} {n * dt:.1f} s", flush=True)
        for i in range(n):
            self.step(r, i, phase)
            if i % int(round(1.0 / dt)) == 0:
                err = np.abs(self.robot.sense().joint_pos - r.joint_pos[min(i + 1, r.T - 1)]).max()
                print(f"[getup]   t {i * dt:4.1f} s  upright {self.robot.upright():+.2f}  关节偏差最大 {err:.2f} rad"
                      f"  推理 {self.log.rows['infer_ms'][-1]:.1f} ms  超拍 {self.robot.overruns}", flush=True)
                n_joint, n_pub = self.robot.rates()
                if self._rates is not None:
                    el = time.monotonic() - self._rates[2]
                    print(f"[getup]           收状态 {(n_joint - self._rates[0]) / el:.0f} Hz  发指令 {(n_pub - self._rates[1]) / el:.0f} Hz", flush=True)
                self._rates = (n_joint, n_pub, time.monotonic())


def wait_enter(flag: threading.Event) -> None:
    try:
        input()
    except EOFError:
        return
    flag.set()


def main() -> None:
    p = argparse.ArgumentParser(description="真机单独测起身（技能策略跟踪起身参考）")
    p.add_argument("--package", type=Path, default=DEFAULT_BASE, help="基础模型包：只用它的 scene.xml 做关节限位与躯干正运动学")
    p.add_argument("--skill", type=Path, default=DEFAULT_SKILL, help="起身 / 趴下技能包")
    p.add_argument("--getup-ref", default="getup_facedown")
    p.add_argument("--liedown-ref", default="liedown_facedown")
    p.add_argument("--no-liedown", action="store_true", help="站住后回车直接切阻尼，不播趴下")
    p.add_argument("--max-stand-s", type=float, default=60.0, help="站住最多多久后自动趴下（或切阻尼）")
    p.add_argument("--ramp-s", type=float, default=3.0, help="从当前姿态插值到起身首帧的时间")
    p.add_argument("--dry-run", action="store_true", help=f"演练：全流程照跑（站住 2 s），指令发到 {DRY_TOPIC}，电机收不到")
    p.add_argument("--core", type=int, default=POLICY_CORE, help="策略进程绑的核（收发进程在 core 4、5）")
    p.add_argument("--prio", type=int, default=POLICY_PRIO, help="策略进程 SCHED_FIFO 优先级，0 不设")
    p.add_argument("--hold-only", action="store_true", help="只保持当前姿态：从当前姿态 reset、再保持 3 s，报告关节偏离指令的最大值后切阻尼")
    p.add_argument("--yes", action="store_true", help="不等回车确认直接开始")
    p.add_argument("--log", type=Path, default=None, help="记录文件，默认 q1/getup/data/real_policy_<时刻>.npz")
    a = p.parse_args()

    base, skill = Package.load(a.package), Package.load(a.skill)
    if base.joint_names != skill.joint_names or not np.allclose(base.default_joint_pos, skill.default_joint_pos):
        sys.exit("基础模型包与技能包的关节顺序 / 默认站姿不同")
    getup, liedown = BodyRef(skill, ref.find(a.getup_ref)), BodyRef(skill, ref.find(a.liedown_ref))
    print(f"[getup] 技能包 {a.skill}（PD {skill.spec['pd'].get('preset', '?')}），起身 {getup.T} 帧、趴下 {liedown.T} 帧", flush=True)

    robot = PrimeBotBackend(base, ramp_s=a.ramp_s, **({"cmd_topic": DRY_TOPIC} if a.dry_run else {}))
    robot.set_pd(skill.kp, skill.kd)
    print(f"[getup] 策略进程 {realtime(a.core, a.prio)}", flush=True)
    log = Log()
    run = Runner(skill, robot, log)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = a.log or PROJECT / "q1" / "getup" / "data" / f"real_policy_{stamp}.npz"
    phase = 0
    try:
        robot.wait_sensors()
        s = robot.sense()
        up = robot.upright()
        dq = s.joint_pos - getup.joint_pos[0]
        k = int(np.argmax(np.abs(dq)))
        print(f"[getup] 骨盆重力方向 {np.round(s.projected_gravity, 2)}  upright {up:+.2f}", flush=True)
        print(f"[getup] 当前关节与起身首帧最大差 {abs(dq[k]):.2f} rad（{skill.joint_names[k]}），插值 {a.ramp_s:.1f} s", flush=True)

        if up > LYING_UPRIGHT and not a.dry_run:
            sys.exit(f"[getup] upright {up:+.2f} > {LYING_UPRIGHT}：机器人不是趴着，不起身")
        if a.dry_run:
            print(f"[getup] 演练：指令发到 {DRY_TOPIC}，电机收不到", flush=True)
        elif not a.yes:
            input("[getup] 确认机器人趴在平地、周围 1 m 无人、吊带放松，回车开始（Ctrl-C 取消）")

        if a.hold_only:
            q_now = robot.sense().joint_pos
            robot.reset(q_now)
            for _ in range(int(round(3.0 / skill.control_dt))):
                robot.step(q_now)
            d, j = robot.worst_tracking()
            moved = np.abs(robot.sense().joint_pos - q_now)
            k = int(np.argmax(moved))
            robot.damp()
            print(f"[getup] 保持当前姿态 6 s：关节偏离指令最大 {d:.3f} rad（{j}），相对起点最大移动 {moved[k]:.3f} rad"
                  f"（{skill.joint_names[k]}），已切阻尼", flush=True)
            return

        robot.reset(getup.joint_pos[0])
        d, j = robot.worst_tracking()
        print(f"[getup] 插值到首帧完成：关节偏离指令最大 {d:.3f} rad（{j}）", flush=True)
        run.begin()
        phase = 1
        run.play(getup, "起身", phase)

        phase = 2
        enter = threading.Event()
        threading.Thread(target=wait_enter, args=(enter,), daemon=True).start()
        print(f"[getup] 站住（参考末帧）：回车{'切阻尼' if a.no_liedown else '趴下'}，Ctrl-C 切阻尼；最多 {a.max_stand_s:.0f} s", flush=True)
        dt, i_end, t_stand = skill.control_dt, getup.T - 1, 0.0
        max_stand = 2.0 if a.dry_run else a.max_stand_s
        while not enter.is_set() and t_stand < max_stand:
            run.step(getup, i_end, phase)
            t_stand += dt
            if robot.upright() < FALL_UPRIGHT and not a.dry_run:
                robot.damp()
                print(f"[getup] 倾倒（upright {robot.upright():+.2f}），已切阻尼", flush=True)
                return
            if int(round(t_stand / dt)) % int(round(2.0 / dt)) == 0:
                print(f"[getup]   站住 {t_stand:4.1f} s  upright {robot.upright():+.2f}  角速度 {np.linalg.norm(robot.sense().base_ang_vel):.2f} rad/s", flush=True)

        if a.no_liedown:
            print("[getup] 切阻尼", flush=True)
            return
        phase = 3
        run.play(liedown, "趴下", phase)
        robot.damp()
        print("[getup] 趴下完成，已切阻尼", flush=True)
    except KeyboardInterrupt:
        print("[getup] Ctrl-C：切阻尼退出", flush=True)
    except RuntimeError as e:
        print(f"[getup] 后端故障（已切阻尼）：{e}", flush=True)
    finally:
        robot.close()
        st = np.asarray(log.rows["t"])
        if len(st) > 1:
            ms = np.asarray(log.rows["infer_ms"])
            n_joint, n_pub = robot.rates()
            el = st[-1] - st[0]
            print(f"[getup] 统计：{len(st)} 步 {el:.1f} s  超拍 {robot.overruns}  推理平均 {ms.mean():.1f} / p99 {np.percentile(ms, 99):.1f} ms"
                  f"  全程收状态 {n_joint} 帧  发指令 {n_pub} 帧", flush=True)
        log.save(log_path, phase_names=np.array(["ramp", "getup", "stand", "liedown"]), joint_names=np.array(skill.joint_names),
                 overruns=robot.overruns, last_phase=phase)


if __name__ == "__main__":
    main()
