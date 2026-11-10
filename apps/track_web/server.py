"""网页服务（只用标准库）：上传动作 → 排队 → 子进程跑 pipeline → 浏览器看进度与跟踪结果。

任务目录 infra/jobs/web_<时间>_<名字>/，与其他任务同层级（any4hdmi manifest 里的相对路径依赖这一层级）。
任务按提交顺序一个一个跑（CPU MuJoCo / GVHMR 都不适合并发）。
"""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from apps.track_web.pipeline import MAX_SECONDS, STEPS, VIDEO_EXT

PROJECT = Path(__file__).resolve().parents[2]
INFRA = PROJECT / "infra"
INDEX = Path(__file__).with_name("index.html")
MAX_UPLOAD = 1 << 30
ALLOWED_EXT = VIDEO_EXT | {".bvh", ".npz"}


def pipeline_env() -> dict:
    env = dict(os.environ)
    paths = [PROJECT, INFRA, INFRA / ".deps" / "GMR", INFRA / ".deps" / "any4hdmi" / "src", INFRA / "SuperMimic-Q1" / "src"]
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in paths] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("SUPERMIMIC_ROBOT", "q1")
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    if sys.platform.startswith("linux") and not env.get("DISPLAY"):
        env.setdefault("MUJOCO_GL", "egl")
    return env


class App:
    """任务队列 + 状态。子类换 PREFIX（任务目录前缀）、PIPELINE（流水线模块）、STEPS 和 post_job（提交入口）。"""

    NAME = "track"
    PREFIX = "web"
    PIPELINE = "apps.track_web.pipeline"
    STEPS = STEPS
    FILES = ("tracking.mp4", "metrics.json", "artifact/motion.npz", "upload/clip.mp4", "pipeline.log")

    def __init__(self, package: Path, jobs_dir: Path, extract: dict, device: dict):
        self.package = package.resolve()
        self.jobs_dir = jobs_dir.resolve()
        self.extract = extract
        self.device = device
        self.queue: queue.Queue[str] = queue.Queue()
        self.current: str | None = None
        self.lock = threading.Lock()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for j in self.list_jobs():
            if j["state"] in ("queued", "running"):
                self._fail(self.jobs_dir / j["id"], "服务重启，任务中断；请重新提交")
        threading.Thread(target=self._worker, daemon=True).start()

    def env(self) -> dict:
        """/api/env：网页据此显示设备信息、切换输入方式。"""
        return {
            "app": self.NAME, "device": self.device, "extract": dict(self.extract), "package": str(self.package),
            "package_name": f"{self.package.parent.name}/{self.package.name}", "max_seconds": MAX_SECONDS,
            "current": self.current, "queued": list(self.queue.queue),
        }

    def summary(self) -> str:
        return f"视频提取 {self.extract['backend']}"

    def job_config(self) -> dict:
        """新建和重试时写进 job.json 的运行配置（服务启动参数可能变了，重试时以当前为准）。"""
        return {"package": str(self.package), "extract": self.extract}

    # ---- 任务
    def list_jobs(self) -> list[dict]:
        out = []
        for d in sorted(self.jobs_dir.glob(f"{self.PREFIX}_*"), reverse=True):
            f = d / "job.json"
            if f.exists():
                try:
                    j = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                out.append({k: j.get(k) for k in ("id", "created", "state", "error", "input")} | {"track": j.get("result", {}).get("track")})
        return out

    def new_root(self, stem: str) -> tuple[str, Path]:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:24] or "clip"
        with self.lock:
            base = f"{self.PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}_{stem}"
            jid, n = base, 1
            while (self.jobs_dir / jid).exists():
                n += 1
                jid = f"{base}_{n}"
            root = self.jobs_dir / jid
            root.mkdir(parents=True)
        return jid, root

    def submit(self, jid: str, root: Path, inp: dict, opts: dict) -> str:
        job = {
            "id": jid, "created": time.time(), "state": "queued", "error": None, "input": inp, "options": opts,
            **self.job_config(), "steps": [{"name": s, "state": "pending", "msg": ""} for s in self.STEPS], "result": {},
        }
        (root / "job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False))
        self.queue.put(jid)
        return jid

    def create(self, filename: str, body, length: int, opts: dict) -> str:
        jid, root = self.new_root(Path(filename).stem)
        (root / "upload").mkdir()
        dst = root / "upload" / f"source{Path(filename).suffix.lower()}"  # 不用原名：pipeline 会在同目录写 clip.mp4 等派生文件
        left = length
        with dst.open("wb") as f:
            while left > 0:
                chunk = body.read(min(left, 1 << 20))
                if not chunk:
                    raise ConnectionError("上传中断")
                f.write(chunk)
                left -= len(chunk)
        return self.submit(jid, root, {"file": dst.name, "original_name": filename, "bytes": length}, opts)

    def post_job(self, req: BaseHTTPRequestHandler, q: dict) -> tuple[dict, int]:
        """POST /api/jobs：请求体直接是文件，选项在查询串里。"""
        name = q.get("filename", "")
        if Path(name).suffix.lower() not in ALLOWED_EXT:
            return {"error": f"不支持的文件类型：{name}（视频 / .bvh / .npz）"}, 400
        length = int(req.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return {"error": "文件为空或超过 1 GB"}, 400
        opts = {
            "start": num(q, "start"), "end": num(q, "end"), "f_mm": num(q, "f_mm"), "min_pelvis_h": num(q, "min_pelvis_h"),
            "static_cam": q.get("static_cam") == "1", "optimize": q.get("optimize", "1") == "1", "force": q.get("force") == "1",
        }
        try:
            return {"id": self.create(name, req.rfile, length, opts)}, 200
        except ConnectionError as e:
            return {"error": str(e)}, 400

    def job(self, jid: str) -> dict | None:
        f = self.jobs_dir / jid / "job.json"
        if not re.fullmatch(rf"{self.PREFIX}_[A-Za-z0-9_]+", jid) or not f.exists():
            return None
        j = json.loads(f.read_text())
        if j["state"] == "queued":
            j["queue_pos"] = list(self.queue.queue).index(jid) + 1 if jid in self.queue.queue else None
        j["files"] = {n: f"/files/{jid}/{n}" for n in self.FILES if (self.jobs_dir / jid / n).exists()}
        return j

    def retry(self, jid: str) -> str | None:
        """重新排队：pipeline 会复用已有的中间结果（GVHMR、重定向、生成的人体动作），其余步骤重跑。"""
        root = self.jobs_dir / jid
        with self.lock:
            j = json.loads((root / "job.json").read_text())
            if j["state"] in ("queued", "running"):
                return "任务正在排队或运行"
            j.update(state="queued", error=None, result={}, steps=[{"name": s, "state": "pending", "msg": ""} for s in self.STEPS],
                     **self.job_config())
            (root / "job.json").write_text(json.dumps(j, indent=1, ensure_ascii=False))
            self.queue.put(jid)
        return None

    def _fail(self, root: Path, msg: str) -> None:
        j = json.loads((root / "job.json").read_text())
        j.update(state="failed", error=msg, finished=time.time())
        for s in j["steps"]:
            if s["state"] == "running":
                s["state"] = "failed"
        (root / "job.json").write_text(json.dumps(j, indent=1, ensure_ascii=False))

    def _worker(self) -> None:
        while True:
            jid = self.queue.get()
            root = self.jobs_dir / jid
            self.current = jid
            with (root / "pipeline.log").open("w") as log:
                p = subprocess.run([sys.executable, "-m", self.PIPELINE, str(root)], cwd=PROJECT, env=pipeline_env(),
                                   stdout=log, stderr=subprocess.STDOUT)
            if p.returncode != 0 and json.loads((root / "job.json").read_text())["state"] != "failed":
                self._fail(root, f"流水线进程异常退出（退出码 {p.returncode}），详见日志")
            self.current = None


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Q1TrackWeb/1.0"

        def log_message(self, fmt, *args):
            if "/api/jobs/" not in self.path or self.command != "GET":
                sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _file(self, path: Path, ctype: str | None = None):
            size = path.stat().st_size
            ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and size:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            if ctype == "application/octet-stream":
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            u = urlparse(self.path)
            p = unquote(u.path)
            if p in ("/", "/index.html"):
                return self._file(INDEX, "text/html; charset=utf-8")
            if p == "/api/env":
                return self._json(app.env())
            if p == "/api/jobs":
                return self._json(app.list_jobs())
            m = re.fullmatch(r"/api/jobs/([^/]+)(/log)?", p)
            if m:
                j = app.job(m.group(1))
                if j is None:
                    return self._json({"error": "任务不存在"}, 404)
                if m.group(2):
                    f = app.jobs_dir / m.group(1) / "pipeline.log"
                    text = f.read_text(errors="replace")[-20000:] if f.exists() else ""
                    return self._json({"log": text})
                return self._json(j)
            m = re.fullmatch(rf"/files/({app.PREFIX}_[A-Za-z0-9_]+)/(.+)", p)
            if m:
                root = (app.jobs_dir / m.group(1)).resolve()
                f = (root / m.group(2)).resolve()
                if root in f.parents and f.is_file():
                    return self._file(f, "text/plain; charset=utf-8" if f.suffix == ".log" else None)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            m = re.fullmatch(r"/api/jobs/([^/]+)/retry", u.path)
            if m:
                if app.job(m.group(1)) is None:
                    return self._json({"error": "任务不存在"}, 404)
                err = app.retry(m.group(1))
                return self._json({"error": err} if err else {"id": m.group(1)}, 409 if err else 200)
            if u.path != "/api/jobs":
                return self._json({"error": "not found"}, 404)
            self._json(*app.post_job(self, {k: v[0] for k, v in parse_qs(u.query).items()}))

    return Handler


def num(q: dict, k: str) -> float | None:
    try:
        return float(q[k]) if str(q.get(k, "")).strip() else None
    except (TypeError, ValueError):
        return None


def detect_device() -> dict:
    info = {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "cuda": False, "gpu": None, "mps": False}
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnxruntime as ort

        info["onnx_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        info["onnx_providers"] = None
    return info


def serve(app: App, host: str, port: int) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1", "") else host
    print(f"[{app.PIPELINE.split('.')[1]}] http://{shown}:{port}  权重包 {app.package}  {app.summary()}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
