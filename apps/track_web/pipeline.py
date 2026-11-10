"""网页任务的流水线，一个任务一个子进程（崩溃不影响网页服务）：

    python -m apps.track_web.pipeline <任务目录>

读 <任务目录>/job.json（输入文件、选项、视频提取后端），逐步执行并回写状态：

prepare   探测时长，截到 [start, end]，超过 MAX_SECONDS 拒绝
extract   视频 → 本机 GVHMR（CUDA / MPS / CPU）→ SMPL-X；动作文件跳过
retarget  人体动作 → Q1（GMR）
optimize  可行性检查 + 轨迹优化（动力学变差时退回原轨迹）
convert   → motion.npz（50 Hz，含全部刚体位姿 / 速度），存进参考库 apps/ref/track_web/<任务 id>.npz
track     基础模型 E_ref 直接跟踪（未微调），录对照视频与逐步误差（跟踪结果补进参考库的 .json）

输入类型：视频（mp4 / mov / …）、LAFAN1 BVH、SMPL-X / AMASS npz、Q1 qpos npz（qpos + fps）、
SuperMimic motion.npz（直接跟踪）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

os.environ.setdefault("SUPERMIMIC_ROBOT", "q1")

MAX_SECONDS = 10.0
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
STEPS = ("prepare", "extract", "retarget", "optimize", "convert", "track")
OPT_ITERS = 25
FEAS_KEYS = ("joint_limit", "penetration", "foot_slip", "foot_tilt", "smoothness", "torque", "balance")
PROJECT = Path(__file__).resolve().parents[2]
GVHMR_INFER = PROJECT / "infra" / "q1infra" / "video" / "gvhmr_infer.py"


class Job:
    def __init__(self, root: Path):
        self.root = root
        self.data = json.loads((root / "job.json").read_text())

    @property
    def opts(self) -> dict:
        return self.data["options"]

    def save(self) -> None:
        tmp = self.root / "job.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        os.replace(tmp, self.root / "job.json")

    def _step(self, name: str) -> dict:
        return next(s for s in self.data["steps"] if s["name"] == name)

    def skip(self, name: str, msg: str = "") -> None:
        self._step(name).update(state="skipped", msg=msg)
        self.save()

    def note(self, name: str, msg: str) -> None:
        self._step(name)["msg"] = msg
        self.save()

    @contextmanager
    def step(self, name: str):
        s = self._step(name)
        s.update(state="running", t0=time.time(), msg="")
        self.save()
        print(f"===== {name} =====", flush=True)
        try:
            yield s
        except Exception as e:
            s.update(state="failed", t1=time.time(), msg=s.get("msg") or str(e))
            raise
        s.update(state="done", t1=time.time())
        self.save()


class UserError(Exception):
    """输入本身的问题（时长、格式、质量门槛），直接把消息显示给用户。"""


def window(total: float, start: float | None, end: float | None) -> tuple[float, float]:
    a = max(0.0, start or 0.0)
    b = total if end is None else min(end, total)
    if b - a < 0.5:
        raise UserError(f"截取区间 [{a:.2f}, {b:.2f}] s 太短（素材总长 {total:.2f} s）")
    if b - a > MAX_SECONDS + 0.05:
        raise UserError(f"动作时长 {b - a:.1f} s 超过 {MAX_SECONDS:.0f} s：请填写起止时间，截取 {MAX_SECONDS:.0f} s 以内的一段")
    return a, b


def npz_kind(path: Path) -> str:
    keys = set(np.load(path, allow_pickle=True).files)
    if {"joint_pos", "body_pos_w"} <= keys:
        return "motion"
    if "qpos" in keys:
        return "q1"
    if "trans" in keys and ({"root_orient", "pose_body"} <= keys or "poses" in keys):
        return "smplx"
    raise UserError(f"无法识别的 npz（键：{sorted(keys)}）：支持 SMPL-X / AMASS、Q1 qpos（qpos + fps）、SuperMimic motion.npz")


def bvh_seconds(path: Path) -> float:
    n = ft = None
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("Frames:"):
                n = int(s.split(":")[1])
            elif s.startswith("Frame Time:"):
                ft = float(s.split(":")[1])
                break
    if n is None or ft is None:
        raise UserError("BVH 缺少 Frames / Frame Time 头")
    return n * ft


def smplx_fps(d) -> float:
    key = next((k for k in ("mocap_frame_rate", "mocap_framerate", "fps") if k in d), None)
    if key is None:
        raise UserError("SMPL-X npz 没有帧率字段（mocap_frame_rate / fps）")
    return float(np.asarray(d[key]).reshape(-1)[0])


def feas_summary(r: dict) -> dict:
    return {
        "verdict": r["verdict"],
        "pass_rate": {k: round(r[k]["pass_rate"], 3) for k in FEAS_KEYS},
        "hover_rate": round(r["foot_slip"]["hover_rate"], 3),
        "infeasible_segments_s": r["infeasible_segments_s"],
    }


# ---------------------------------------------------------------- 各步骤


def prepare(job: Job) -> dict:
    src = job.root / "upload" / job.data["input"]["file"]
    ext = src.suffix.lower()
    o = job.opts
    if ext in VIDEO_EXT:
        import imageio_ffmpeg

        _, total = imageio_ffmpeg.count_frames_and_secs(str(src))
        a, b = window(total, o.get("start"), o.get("end"))
        clip = job.root / "upload" / "clip.mp4"
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", str(src), "-t", f"{b - a:.3f}", "-an",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(clip)],
            check=True,
        )
        return {"kind": "video", "path": str(clip), "total_s": total, "window_s": [a, b]}
    if ext == ".bvh":
        total = bvh_seconds(src)
        a, b = window(total, o.get("start"), o.get("end"))
        return {"kind": "bvh", "path": str(src), "total_s": total, "window_s": [a, b]}
    if ext != ".npz":
        raise UserError(f"不支持的文件类型 {ext}：上传视频、.bvh 或 .npz")
    kind = npz_kind(src)
    d = np.load(src, allow_pickle=True)
    if kind == "smplx":
        total = len(d["trans"]) / smplx_fps(d)
        a, b = window(total, o.get("start"), o.get("end"))
        return {"kind": kind, "path": str(src), "total_s": total, "window_s": [a, b]}
    if kind == "q1":
        fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 50.0
        total = len(d["qpos"]) / fps
        a, b = window(total, o.get("start"), o.get("end"))
        out = job.root / "upload" / "q1_qpos.npz"
        np.savez(out, qpos=np.asarray(d["qpos"], np.float64)[int(round(a * fps)) : int(round(b * fps))], fps=fps)
        return {"kind": kind, "path": str(out), "total_s": total, "window_s": [a, b]}
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    T = len(d["joint_pos"])
    total = T / fps
    a, b = window(total, o.get("start"), o.get("end"))
    i0, i1 = int(round(a * fps)), int(round(b * fps))
    out = job.root / "artifact" / "motion.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: (v[i0:i1] if v.ndim and v.shape[0] == T else v) for k, v in ((k, d[k]) for k in d.files)})
    return {"kind": kind, "path": str(out), "total_s": total, "window_s": [a, b]}


def extract(job: Job, video: Path) -> Path:
    import torch

    from q1infra.retarget.gmr_q1 import SMPLX_MODEL_DIR
    from q1infra.video import convert

    cfg, o = job.data["extract"], job.opts
    vdir = job.root / "video"
    args = (["--static-cam"] if o.get("static_cam") else []) + (["--f-mm", str(int(o["f_mm"]))] if o.get("f_mm") else [])
    if (vdir / "hmr4d_results.pt").exists():
        print(f"[extract] 复用已有 GVHMR 结果 {vdir}", flush=True)
    elif cfg["backend"] == "local":
        print(f"[extract] GVHMR（{cfg['device']}）：{cfg['gvhmr_root']}", flush=True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # GVHMR 用它自己的环境
        subprocess.run([cfg["gvhmr_python"], str(GVHMR_INFER), str(video), str(vdir), "--device", cfg["device"], *args],
                       cwd=cfg["gvhmr_root"], env=env, check=True)
    else:
        raise UserError("本机没有配置 GVHMR，不能处理视频：启动时用 --gvhmr-root / --gvhmr-python 指定，"
                        "或者直接上传动作文件（BVH / SMPL-X npz）")

    vis = convert.visibility(torch.load(vdir / "vitpose.pt", map_location="cpu").numpy())
    gate = convert.gate(vis, None, None)
    npz = vdir / "clip_smplx.npz"
    conv = convert.gvhmr_to_amass(vdir / "hmr4d_results.pt", npz, SMPLX_MODEL_DIR)
    report = {"gvhmr": json.loads((vdir / "meta.json").read_text()), "gate": gate, "smplx": conv}
    (vdir / "video_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    job.data["result"]["video_gate"] = gate
    msg = f"全身可见 {gate['visible_frac']:.0%}，最长连续不可见 {gate['longest_invisible_s']:.1f} s"
    job.note("extract", msg)
    print(f"[extract] {msg}", flush=True)
    if not gate["passed"] and not o.get("force"):
        raise UserError(f"视频质量门槛未通过（{msg}；要求全身可见 ≥ 90%、时长 ≥ 2 s）。换一段全身入镜的视频，或勾选“门槛不过也继续”")
    return npz


def retarget(job: Job, kind: str, src: Path, win: list[float]) -> Path:
    import shutil

    out = job.root / "retarget"
    if (out / "manifest.json").exists() and (job.root / "retarget_report.json").exists():
        print("[retarget] 复用已有重定向结果", flush=True)
        rep = json.loads((job.root / "retarget_report.json").read_text())
    else:
        for d in ("retarget", "input", ".gmr_tmp"):
            shutil.rmtree(job.root / d, ignore_errors=True)
        rep = _retarget(job, kind, src, win)
    te = rep["tracking_error_m"]
    job.data["result"]["retarget"] = {
        "duration_s": rep["duration_s"],
        "tracking_error_cm": {b: round(v["mean"] * 100, 1) for b, v in te.items()},
        "near_joint_limit": {k: round(v, 3) for k, v in rep["near_joint_limit_ratio"].items()},
    }
    mean_cm = np.mean([v["mean"] for v in te.values()]) * 100
    job.note("retarget", f"{rep['duration_s']:.1f} s，关键点平均误差 {mean_cm:.1f} cm")
    return out


def _retarget(job: Job, kind: str, src: Path, win: list[float]) -> dict:
    from q1infra.retarget.gmr_q1 import retarget_bvh, retarget_smplx

    kw = {"flat_feet": True, "min_pelvis_height": job.opts.get("min_pelvis_h")}
    if kind == "video":
        rep = retarget_smplx(src, job.root, source_extra={"input_type": "video_gvhmr", "video": job.data["input"]["file"]}, **kw)
    elif kind == "bvh":
        rep = retarget_bvh(src, job.root, win[0], win[1], **kw)
    else:
        rep = retarget_smplx(src, job.root, win[0], win[1], **kw)
    return rep


def optimize(job: Job, src: Path) -> tuple[np.ndarray, float]:
    from q1infra.feasibility.check import check_path, load_input
    from q1infra.feasibility.optimize import optimize_path
    from q1infra.feasibility.scene import FLAT

    use = src
    if job.opts.get("optimize", True):
        r = next(iter(optimize_path(src, FLAT, OPT_ITERS, log=lambda s: print(s, flush=True)).values()))
        before, after = feas_summary(r["before"]), feas_summary(r["after"])
        if not r["rejected"]:
            use = Path(r["output"])
        job.data["result"]["feasibility"] = {"before": before, "after": after, "optimized": not r["rejected"]}
        job.note("optimize", f"{before['verdict']} → {after['verdict']}" + ("（动力学变差，退回原轨迹）" if r["rejected"] else ""))
    else:
        r = next(iter(check_path(src, FLAT).values()))
        job.data["result"]["feasibility"] = {"before": feas_summary(r), "after": None, "optimized": False}
        job.note("optimize", f"只检查：{r['verdict']}")
    _, qpos, dt, _ = load_input(use)[0]
    return qpos, dt


def convert_motion(job: Job, qpos: np.ndarray, dt: float) -> Path:
    from q1infra.retarget.gmr_q1 import Q1_MJCF
    from supermimic.data.motion_convert import convert_qpos_to_npz
    from supermimic.robot import load_robot_model

    out = job.root / "artifact" / "motion.npz"
    # 显式给 MJCF：默认路径要经 supermimic.robots.q1 导入 mjlab（只在训练环境里有）
    info = convert_qpos_to_npz(qpos, out, 1.0 / dt, 50.0, load_robot_model(Q1_MJCF), source=str(job.root))
    job.note("convert", f"{info['output_frames']} 帧 @ 50 Hz")
    return out


def save_ref(job: Job, motion: Path, source: str) -> Path:
    """生成的参考存进参考库 apps/ref/<source>/<任务 id>.npz；跟踪结果在 track_step 里补进 .json。"""
    from apps.ref import save

    res = job.data["result"]
    meta = {"job": str(job.root), "input": job.data["input"], "options": job.opts}
    meta |= {k: res[k] for k in ("generate", "retarget", "feasibility") if k in res}
    ref = save(motion, source, job.data["id"], meta)
    res["ref"] = str(ref.relative_to(PROJECT))
    print(f"[ref] {res['ref']}", flush=True)
    job.save()
    return ref


def track_step(job: Job, motion: Path, human: Path | None, human_label: str = "人体视频", ref: Path | None = None) -> None:
    from apps.ref import update_meta
    from apps.track_web.track import track

    with job.step("track") as s:
        res = track(Path(job.data["package"]), motion, job.root / "tracking.mp4", human, log=lambda m: print(m, flush=True),
                    human_label=human_label)
        (job.root / "metrics.json").write_text(json.dumps(res, ensure_ascii=False))
        job.data["result"]["track"] = {k: v for k, v in res.items() if k != "curve"}
        if ref is not None:
            update_meta(ref, track={k: v for k, v in res.items() if k not in ("curve", "panels")})
        s["msg"] = ("未摔倒" if res["fell_at_s"] is None else f"摔倒 @ {res['fell_at_s']:.2f} s") + f"，关节误差 {res['joint_err_mean']:.3f} rad"


def run(job: Job) -> None:
    with job.step("prepare") as s:
        prep = prepare(job)
        job.data["result"]["input"] = prep
        a, b = prep["window_s"]
        s["msg"] = f"{prep['kind']}，截取 {a:.2f}–{b:.2f} s（素材 {prep['total_s']:.1f} s）"
    kind, src = prep["kind"], Path(prep["path"])

    if kind == "video":
        with job.step("extract"):
            src = extract(job, src)
    else:
        job.skip("extract", "动作文件，不需要")

    if kind in ("video", "bvh", "smplx"):
        with job.step("retarget"):
            src = retarget(job, kind, src, prep["window_s"])
    else:
        job.skip("retarget", "已是 Q1 轨迹")

    if kind == "motion":
        job.skip("optimize", "直接跟踪 motion.npz")
        job.skip("convert", "")
        motion = src
    else:
        with job.step("optimize"):
            qpos, dt = optimize(job, src)
        with job.step("convert"):
            motion = convert_motion(job, qpos, dt)

    ref = save_ref(job, motion, "track_web")
    human = job.root / "video" / "0_input_video.mp4"
    if not human.exists() and kind == "video":
        human = Path(prep["path"])
    track_step(job, motion, human if kind == "video" else None, ref=ref)


def main(run_fn=run) -> None:
    job = Job(Path(sys.argv[1]).resolve())
    job.data.update(state="running", started=time.time(), error=None)
    job.save()
    try:
        run_fn(job)
        job.data.update(state="done", finished=time.time())
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        user = isinstance(e, UserError)
        msg = str(e) if user else f"{type(e).__name__}: {e}"
        if isinstance(e, subprocess.CalledProcessError):
            msg = f"外部命令失败（退出码 {e.returncode}）：{' '.join(map(str, e.cmd))[:300]}；详见日志"
        job.data.update(state="failed", finished=time.time(), error=msg)
        job.save()
        sys.exit(1)
    job.save()


if __name__ == "__main__":
    main()
