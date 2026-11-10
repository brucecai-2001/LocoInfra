"""参考库：各应用要播放的参考轨迹（SuperMimic motion.npz，50 Hz）统一放在这里，每条旁边一个同名 .json 记来源。

    apps/ref/skills/        技能参考，随仓库发布：getup_facedown（俯卧起身）、liedown_facedown（站立趴下）
    apps/ref/track_web/     动作跟踪网页生成（文件名 = 任务 id）
    apps/ref/text2motion/   文本生成动作网页生成（文件名 = 任务 id）

    infra/bin/q1py -m apps.ref            # 列出参考库

除 skills/ 外数据不入库（见 .gitignore）。按名字找参考用 ``find``：给路径直接用，给名字在各子目录里找。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

REF_ROOT = Path(__file__).resolve().parent
SOURCES = ("skills", "track_web", "text2motion")


def _info(npz: Path) -> dict:
    d = np.load(npz)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    n = int(d["joint_pos"].shape[0])
    return {"frames": n, "fps": fps, "duration_s": round(n / fps, 2)}


def save(motion: Path, source: str, name: str, meta: dict | None = None) -> Path:
    """把 motion.npz 存进 apps/ref/<source>/<name>.npz（同名覆盖），旁边写 <name>.json。"""
    if source not in SOURCES:
        raise ValueError(f"未知来源 {source}（{SOURCES}）")
    dst = REF_ROOT / source / f"{name}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(motion, dst)
    info = {"name": name, "source": source, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), **_info(dst), **(meta or {})}
    dst.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    return dst


def update_meta(ref: Path, **fields) -> None:
    f = Path(ref).with_suffix(".json")
    info = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps(info | fields, indent=1, ensure_ascii=False))


def find(name: str | Path) -> Path:
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    hits = [REF_ROOT / s / f"{p.stem}.npz" for s in SOURCES if (REF_ROOT / s / f"{p.stem}.npz").exists()]
    if not hits:
        raise FileNotFoundError(f"参考库里没有 {name}（{REF_ROOT}/{{{','.join(SOURCES)}}}/）")
    return hits[0]


def list_refs() -> list[dict]:
    out = []
    for s in SOURCES:
        for f in sorted((REF_ROOT / s).glob("*.npz")):
            j = f.with_suffix(".json")
            info = json.loads(j.read_text()) if j.exists() else {"name": f.stem, "source": s, **_info(f)}
            out.append(info | {"path": str(f)})
    return out
