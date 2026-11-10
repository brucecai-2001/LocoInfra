"""HY-Motion 1.0 文本生成人体动作服务。在 GPU 机器上、HY-Motion 自己的环境里常驻（工作目录 = HY-Motion 仓库根目录，权重在 ckpts/ 下）：

    cd $HY_ROOT && $HY_ROOT/.venv/bin/python <LocoInfra>/infra/q1infra/text2motion/hymotion_server.py [--port 8770] [--lite]

模型（DiT + Qwen3-8B / CLIP 文本编码器，约 26 GB 显存）启动时加载一次，之后每次请求只做采样。
只用标准库 HTTP；一次只算一个请求（GPU 锁）。

GET  /health    → 模型、设备、显存、最长时长
POST /generate  → 请求 JSON：
        text          动作描述（英文最好；含中文或 rewrite=true 时先改写成英文）
        duration      秒；不填用 5 s
        max_duration  时长上限（秒），默认模型上限 12 s
        seed、cfg_scale（默认 5.0）、rewrite
    返回 npz（Y 朝上、30 fps、已贴地）：
        poses (T,156)     SMPL-H 轴角：根 + 21 个身体关节 + 双手平均姿态
        trans (T,3)       根平移（SMPL 约定：骨盆 = 骨盆静止位置 + trans）
        text / original_text / duration / seed / cfg_scale / fps / model / gen_s
    出错时返回 JSON {"error": ...}（输入问题 400，其余 500）。

中文描述的改写复用已经加载的 Qwen3-8B 文本编码器（它本身就是完整的因果语言模型），用官方改写模板
REWRITE_AND_INFER_TIME_PROMPT_FORMAT 生成英文描述（模板顺带给的时长不用）；官方改写模型 Text2MotionPrompter
（Qwen3-30B，61 GB）不用。
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

FPS = 30
# 基础版 Qwen3-8B 估不准时长（官方模板下几乎总答 60 / 120 帧，按秒问又偏大数倍），不填时用固定值
DEFAULT_S = 5.0
CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


class BadRequest(Exception):
    pass


class HYMotion:
    def __init__(self, model_dir: Path):
        sys.path.insert(0, str(Path.cwd()))
        from hymotion.utils.t2m_runtime import T2MRuntime

        t0 = time.time()
        rt = T2MRuntime(config_path=str(model_dir / "config.yml"), ckpt_name=str(model_dir / "latest.ckpt"),
                        device_ids=[0], disable_prompt_engineering=True)
        self.pipe = rt.pipelines[0].eval()
        self.device = next(self.pipe.parameters()).device
        self.model = model_dir.name
        self.max_s = self.pipe.train_frames / FPS
        self.min_s = min(self.pipe.train_frames, 20) / FPS
        self.lock = threading.Lock()
        print(f"[hymotion] {self.model} 加载完成 {time.time() - t0:.0f} s，设备 {self.device}，最长 {self.max_s:.1f} s", flush=True)

    def health(self) -> dict:
        return {
            "model": self.model, "device": str(self.device), "gpu": torch.cuda.get_device_name(self.device),
            "mem_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 1), "fps": FPS,
            "min_duration": round(self.min_s, 2), "max_duration": round(self.max_s, 2), "busy": self.lock.locked(),
        }

    @torch.no_grad()
    def rewrite(self, text: str) -> tuple[float, str]:
        """官方改写模板 → (时长 s, 英文描述)。"""
        from hymotion.prompt_engineering.model_constants import REWRITE_AND_INFER_TIME_PROMPT_FORMAT

        te = self.pipe.text_encoder
        tok, llm = te.llm_tokenizer, te.llm_text_encoder
        msgs = [{"role": "user", "content": REWRITE_AND_INFER_TIME_PROMPT_FORMAT.format(text)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok([s], return_tensors="pt").to(self.device)
        out = llm.generate(**ids, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None)
        resp = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", resp, re.S)
        try:
            r = json.loads(m.group()) if m else {}
            return float(r["duration"]) / FPS, str(r["short_caption"]).strip()
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"改写输出无法解析：{resp[:300]}") from e

    @torch.no_grad()
    def generate(self, req: dict) -> dict:
        from hymotion.pipeline.body_model import construct_smpl_data_dict

        text = str(req.get("text") or "").strip()
        if not text:
            raise BadRequest("text 为空")
        dur = req.get("duration")
        seed = int(req["seed"]) if req.get("seed") is not None else random.randint(0, 99999)
        cfg = float(req.get("cfg_scale") or 5.0)
        max_s = min(float(req.get("max_duration") or self.max_s), self.max_s)
        with self.lock:
            t0 = time.time()
            prompt, est = text, None
            if req.get("rewrite") or CJK.search(text):
                est, prompt = self.rewrite(text)
                print(f"[hymotion] 改写：{text!r} → {prompt!r}（模板估计 {est:.2f} s，不用）", flush=True)
            dur = min(max(float(dur if dur is not None else DEFAULT_S), self.min_s), max_s)
            out = self.pipe.generate(prompt, [seed], dur, cfg_scale=cfg)
            smpl = construct_smpl_data_dict(out["rot6d"][0].clone(), out["transl"][0].clone())
            gen_s = time.time() - t0
        print(f"[hymotion] {prompt!r} {dur:.2f} s seed={seed} cfg={cfg} → {smpl['poses'].shape[0]} 帧，{gen_s:.1f} s", flush=True)
        return {
            "poses": smpl["poses"].astype(np.float32), "trans": smpl["trans"].astype(np.float32),
            "text": prompt, "original_text": text, "duration": dur, "estimated_duration": est if est is not None else -1.0,
            "seed": seed, "cfg_scale": cfg, "fps": FPS, "model": self.model, "gen_s": gen_s,
        }


def make_handler(hy: HYMotion):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HYMotionQ1/1.0"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", code)

        def do_GET(self):
            if self.path == "/health":
                return self._json(hy.health())
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/generate":
                return self._json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                res = hy.generate(req)
            except (BadRequest, json.JSONDecodeError, ValueError) as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            buf = io.BytesIO()
            np.savez(buf, **res)
            self._send(buf.getvalue(), "application/octet-stream")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="HY-Motion 1.0 文本生成动作服务")
    p.add_argument("--host", default="127.0.0.1", help="默认只听本机，其他机器经 ssh -L 转发访问")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--lite", action="store_true", help="用 HY-Motion-1.0-Lite（0.46B）")
    a = p.parse_args()
    hy = HYMotion(Path("ckpts/tencent") / ("HY-Motion-1.0-Lite" if a.lite else "HY-Motion-1.0"))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(hy))
    srv.daemon_threads = True
    print(f"[hymotion] 监听 http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
