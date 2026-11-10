"""文本生成动作的流水线，一个任务一个子进程：

    python -m apps.text2motion.pipeline <任务目录>

generate  文本 → GPU 上的 HY-Motion 服务 → SMPL-H 人体动作（中文或勾选改写时先改写成英文）→ AMASS SMPL+H npz、骨架视频
retarget  人体动作 → Q1（GMR，SMPL+H 分支）
optimize  可行性检查 + 轨迹优化（动力学变差时退回原轨迹）
convert   → motion.npz（50 Hz），存进参考库 apps/ref/text2motion/<任务 id>.npz
track     基础模型 E_ref 直接跟踪（未微调），录对照视频与逐步误差（跟踪结果补进参考库的 .json）

后四步与动作跟踪网页（apps.track_web.pipeline）完全相同。重试时复用已生成的人体动作（同一种子）和重定向结果。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("SUPERMIMIC_ROBOT", "q1")

from apps.track_web.pipeline import MAX_SECONDS, Job, UserError, convert_motion, main, optimize, retarget, save_ref, track_step  # noqa: E402

STEPS = ("generate", "retarget", "optimize", "convert", "track")
GEN_TIMEOUT = 600  # 服务刚启动时第一次请求可能还在加载


def request_motion(url: str, req: dict) -> bytes:
    r = urllib.request.Request(f"{url}/generate", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=GEN_TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", "")
        except ValueError:
            msg = ""
        if e.code == 400:
            raise UserError(f"HY-Motion 拒绝了这条描述：{msg}") from e
        raise RuntimeError(f"HY-Motion 服务出错（HTTP {e.code}）：{msg}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise UserError(f"连不上 HY-Motion 服务 {url}（{getattr(e, 'reason', e)}）：在 GPU 机器上启动 hymotion_server.py，"
                        "服务在别的机器时用 ssh -L 8770:127.0.0.1:8770 转发端口") from e


def generate(job: Job) -> tuple[Path, Path]:
    from apps.text2motion.human_view import grounded, render_skeleton
    from q1infra.text2motion.convert import body_joints, hymotion_to_amass

    o, gdir = job.opts, job.root / "generate"
    gdir.mkdir(exist_ok=True)
    raw = gdir / "hymotion.npz"
    if raw.exists():
        print(f"[generate] 复用已生成的人体动作 {raw}", flush=True)
    else:
        url = job.data["t2m"]["url"]
        print(f"[generate] {url}：{o['text']!r}，时长 {o.get('duration') or '自动'}，seed {o['seed']}", flush=True)
        req = {"text": o["text"], "duration": o.get("duration"), "seed": o["seed"], "cfg_scale": o.get("cfg_scale"),
               "rewrite": o.get("rewrite", False), "max_duration": MAX_SECONDS}
        raw.write_bytes(request_motion(url, req))
    d = np.load(raw)
    meta = {k: d[k].item() for k in ("text", "original_text", "duration", "seed", "cfg_scale", "model", "gen_s")}
    meta["rewritten"] = meta["text"] != meta["original_text"]
    print(f"[generate] {meta['model']}：{meta['text']!r}，{meta['duration']:.2f} s，生成 {meta['gen_s']:.1f} s", flush=True)

    amass = gdir / "human_smplh.npz"
    meta |= hymotion_to_amass(raw, amass)
    pos, parents, fps = body_joints(amass)
    pos = grounded(pos)
    meta["height_range_m"] = [round(float(pos[..., 2].min()), 3), round(float(pos[..., 2].max()), 3)]
    human = gdir / "human.mp4"
    render_skeleton(pos, parents, fps, human)
    job.data["result"]["generate"] = meta
    job.note("generate", f"{meta['duration_s']:.1f} s · seed {meta['seed']}" + (f" · {meta['text']}" if meta["rewritten"] else ""))
    return amass, human


def run(job: Job) -> None:
    with job.step("generate"):
        src, human = generate(job)
    with job.step("retarget"):
        src = retarget(job, "smplx", src, [None, None])
    with job.step("optimize"):
        qpos, dt = optimize(job, src)
    with job.step("convert"):
        motion = convert_motion(job, qpos, dt)
    ref = save_ref(job, motion, "text2motion")
    track_step(job, motion, human, "HY-Motion 生成的人体", ref=ref)


if __name__ == "__main__":
    main(run)
