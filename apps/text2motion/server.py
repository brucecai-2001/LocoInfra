"""文本生成动作网页的服务端：沿用动作跟踪网页的任务队列、文件下载和前端，只换提交入口和流水线。

任务目录 infra/jobs/t2m_<时间>_<名字>/。POST /api/jobs 的请求体是 JSON：
    text、duration（秒，空 = 5 s）、seed（空 = 随机，写进任务以便重试复现）、cfg_scale、rewrite、optimize
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from apps.text2motion.pipeline import STEPS
from apps.track_web.server import MAX_SECONDS, App, num

MAX_TEXT = 400
DEFAULT_DURATION = 5.0  # 不估时长：基础版 Qwen3-8B 估得不准（见 hymotion_server.py）


class TextApp(App):
    NAME = "text2motion"
    PREFIX = "t2m"
    PIPELINE = "apps.text2motion.pipeline"
    STEPS = STEPS
    FILES = App.FILES + ("generate/human.mp4", "generate/human_smplh.npz", "generate/hymotion.npz")

    def __init__(self, package: Path, jobs_dir: Path, t2m_url: str, device: dict):
        self.t2m_url = t2m_url.rstrip("/")
        super().__init__(package, jobs_dir, {"backend": "off"}, device)

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.t2m_url}/health", timeout=3) as r:
                return {"ok": True, **json.loads(r.read())}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(getattr(e, "reason", e))}

    def env(self) -> dict:
        return super().env() | {"t2m": {"url": self.t2m_url, **self.health()}}

    def summary(self) -> str:
        h = self.health()
        return f"HY-Motion {self.t2m_url}（" + (f"{h['model']} · {h['gpu']}" if h["ok"] else f"连不上：{h['error']}") + "）"

    def job_config(self) -> dict:
        return {"package": str(self.package), "t2m": {"url": self.t2m_url}}

    def post_job(self, req: BaseHTTPRequestHandler, q: dict) -> tuple[dict, int]:
        try:
            body = json.loads(req.rfile.read(int(req.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return {"error": "请求不是 JSON"}, 400
        text = " ".join(str(body.get("text") or "").split())
        if not text:
            return {"error": "请输入动作描述"}, 400
        if len(text) > MAX_TEXT:
            return {"error": f"描述太长（{len(text)} 字），请控制在 {MAX_TEXT} 字以内"}, 400
        dur = num(body, "duration")
        if dur is not None and not 1.0 <= dur <= MAX_SECONDS:
            return {"error": f"时长需在 1–{MAX_SECONDS:.0f} s 之间（留空为 {DEFAULT_DURATION:.0f} s）"}, 400
        seed = num(body, "seed")
        opts = {
            "text": text, "duration": dur or DEFAULT_DURATION, "seed": int(seed) if seed is not None else random.randint(0, 99999),
            "cfg_scale": num(body, "cfg_scale") or 5.0, "rewrite": bool(body.get("rewrite")), "optimize": body.get("optimize", True) is not False,
        }
        jid, root = self.new_root(re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "text")
        return {"id": self.submit(jid, root, {"text": text, "original_name": text}, opts)}, 200
