"""GVHMR 结果 → AMASS 格式 SMPL-X npz（重定向的 --smplx 输入），以及视频质量门槛。

GVHMR 的 ``smpl_params_global`` 在重力对齐、Y 朝上的世界系里（SMPL-X 中性体型，10 维 betas，30 fps）。
AMASS 是 Z 朝上：整个人绕 x 轴转 +90°。SMPL-X 的根旋转绕的是骨盆静止位置 J0（随体型变），所以
    root_orient' = R · root_orient,     trans' = R (J0 + trans) − J0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# COCO-17：肩、髋、膝、踝都可信才算全身可见
BODY_KPTS = (5, 6, 11, 12, 13, 14, 15, 16)
KPT_CONF = 0.5
GATE_VISIBLE = 0.90
GATE_SECONDS = (2.0, 60.0)
FPS = 30.0
Y_UP_TO_Z_UP = Rotation.from_euler("x", 90, degrees=True)


def pelvis_rest(betas: np.ndarray, model_dir: Path) -> np.ndarray:
    import smplx
    import torch

    from q1infra.retarget.gmr_q1 import _smplx_model_file

    f = _smplx_model_file(model_dir, "neutral")
    body = smplx.SMPLX(model_path=str(f), gender="neutral", use_pca=False, num_betas=len(betas), ext=f.suffix[1:])
    with torch.no_grad():
        j = body(betas=torch.as_tensor(betas, dtype=torch.float32)[None]).joints[0, 0]
    return j.numpy().astype(np.float64)


def visibility(vitpose: np.ndarray) -> np.ndarray:
    """(T,) 每帧是否全身可见。vitpose (T, 17, 3)：x, y, 置信度。"""
    return (vitpose[:, BODY_KPTS, 2] > KPT_CONF).all(axis=1)


def gate(vis: np.ndarray, start_sec: float | None, end_sec: float | None) -> dict:
    a = int(round((start_sec or 0.0) * FPS))
    b = len(vis) if end_sec is None else min(len(vis), int(round(end_sec * FPS)))
    v = vis[a:b]
    dur = len(v) / FPS
    frac = float(v.mean()) if len(v) else 0.0
    # 最长的连续不可见段，便于定位
    worst, run = 0, 0
    for x in v:
        run = 0 if x else run + 1
        worst = max(worst, run)
    ok = frac >= GATE_VISIBLE and GATE_SECONDS[0] <= dur <= GATE_SECONDS[1]
    return {
        "window_s": [a / FPS, b / FPS], "duration_s": dur, "visible_frac": frac, "longest_invisible_s": worst / FPS,
        "threshold": {"visible_frac": GATE_VISIBLE, "duration_s": list(GATE_SECONDS)}, "passed": bool(ok),
    }


def gvhmr_to_amass(pred_path: Path, out_npz: Path, model_dir: Path) -> dict:
    import torch

    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    sp = {k: v.numpy().astype(np.float64) for k, v in pred["smpl_params_global"].items()}
    go, body_pose, transl = sp["global_orient"], sp["body_pose"], sp["transl"]
    betas = sp["betas"].mean(0) if sp["betas"].ndim == 2 else sp["betas"]
    j0 = pelvis_rest(betas, model_dir)
    root = (Y_UP_TO_Z_UP * Rotation.from_rotvec(go)).as_rotvec()
    trans = Y_UP_TO_Z_UP.apply(j0 + transl) - j0
    np.savez(
        out_npz,
        root_orient=root.astype(np.float32), pose_body=body_pose[:, :63].astype(np.float32), trans=trans.astype(np.float32),
        betas=betas.astype(np.float32), gender="neutral", mocap_frame_rate=np.float32(FPS), surface_model_type="smplx",
    )
    return {"frames": len(go), "fps": FPS, "num_betas": len(betas), "pelvis_rest_m": j0.round(4).tolist(), "height_range_m": [float(trans[:, 2].min()), float(trans[:, 2].max())]}
