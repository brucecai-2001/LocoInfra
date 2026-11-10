"""HY-Motion 输出 → AMASS 格式 SMPL+H npz（重定向的 SMPL-X 输入走 SMPL+H 分支）。

HY-Motion 把训练数据统一拟合到中性 SMPL-H 骨架，22 个身体关节的树和顺序与 SMPL-X 相同；输出 Y 朝上、30 fps。
AMASS 是 Z 朝上：整个人绕 x 轴转 +90°，旋转绕骨盆静止位置 J0（与 video/convert.py 相同）：
    root_orient' = R · root_orient,     trans' = R (J0 + trans) − J0
J0 取重定向 SMPL+H 分支所用的 SMPL-X 标准体型骨盆；HY-Motion 自己的 J0 与它只差一个常量平移，重定向贴地时被消掉。
HY-Motion 网页预览用的木头人骨架静止姿态是标准 T 字形，与 SMPL-H 不同（上臂差约 16°），所以这里不用它。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

Y_UP_TO_Z_UP = Rotation.from_euler("x", 90, degrees=True)


def hymotion_to_amass(raw: Path, out_npz: Path) -> dict:
    """raw：生成服务返回的 npz。写 out_npz（poses (T,156)、trans、mocap_framerate，无 surface_model_type）。"""
    from q1infra.retarget.gmr_q1 import smplx_rest_joints

    d = np.load(raw)
    poses, trans, fps = d["poses"].astype(np.float64), d["trans"].astype(np.float64), float(d["fps"])
    j0 = np.load(smplx_rest_joints())["joints"][0]
    poses[:, :3] = (Y_UP_TO_Z_UP * Rotation.from_rotvec(poses[:, :3])).as_rotvec()
    trans = Y_UP_TO_Z_UP.apply(j0 + trans) - j0
    np.savez(out_npz, poses=poses.astype(np.float32), trans=trans.astype(np.float32), betas=np.zeros(16, np.float32),
             gender="neutral", mocap_framerate=np.float32(fps))
    return {"frames": len(poses), "fps": fps, "duration_s": round(len(poses) / fps, 2)}


def body_joints(amass_npz: Path) -> tuple[np.ndarray, list[int], float]:
    """与重定向看到的完全相同的人体：22 个身体关节的世界位置 (T,22,3)、父关节表、帧率。"""
    from q1infra.retarget.gmr_q1 import SMPLX_MODEL_DIR, load_smplx_motion

    human, _ = load_smplx_motion(amass_npz, SMPLX_MODEL_DIR, None, None)
    names = list(human.rest_pos)
    parents = [-1 if human.parents[n] is None else names.index(human.parents[n]) for n in names]
    pos = np.array([[f[n][0] for n in names] for f in human.frames])
    return pos, parents, human.fps
