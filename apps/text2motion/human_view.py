"""生成的人体动作 → 骨架视频（对照视频的第一栏）。

相机与 track.py 的机器人画面一致：跟踪骨盆，方位角 160°、俯角 12°、竖直视场 45°；人约是 Q1 的两倍高，距离按比例放大，
两栏里人和机器人大小相近、朝向相同（重定向不改朝向）。只用 PIL 画，不依赖人体模型文件。
"""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

from apps.track_web.track import H, ROBOT_W

AZIMUTH, ELEVATION, FOVY = 160.0, -12.0, 45.0
DISTANCE = 4.4  # 机器人画面 2.2 m × 身高比约 2
SS = 2  # 超采样抗锯齿
FOOT_JOINT_Z = 0.04  # SMPL 最低的关节（踝 / 脚趾根）在鞋底以上约 4 cm
LEFT = {1, 4, 7, 10, 13, 16, 18, 20}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21}
HEAD = 15
COLORS = {"l": (91, 156, 255), "r": (242, 135, 92), "c": (220, 224, 232)}
BG_TOP, BG_BOTTOM, GRID = (38, 44, 56), (18, 20, 26), (70, 78, 94)


def grounded(pos: np.ndarray) -> np.ndarray:
    """预览贴地：最低关节（5% 分位）放到 FOOT_JOINT_Z。"""
    pos = pos.copy()
    pos[..., 2] -= np.percentile(pos[..., 2].min(1), 5) - FOOT_JOINT_Z
    return pos


def _camera(azimuth: float, elevation: float):
    az, el = np.radians(azimuth), np.radians(elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return fwd, right, np.cross(right, fwd)


def render_skeleton(pos: np.ndarray, parents: list[int], fps: float, out_mp4: Path) -> None:
    """pos (T,22,3)：Z 朝上的世界坐标（米）。"""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p") as wr:
        for img in skeleton_frames(pos, parents):
            wr.append_data(img)


def skeleton_frames(pos: np.ndarray, parents: list[int], size: tuple[int, int] = (ROBOT_W, H), azimuth: float = AZIMUTH,
                    elevation: float = ELEVATION, distance: float = DISTANCE):
    """逐帧生成骨架画面 (h, w, 3)。"""
    fwd, right, up = _camera(azimuth, elevation)
    w, h = size[0] * SS, size[1] * SS
    focal = (h / 2) / np.tan(np.radians(FOVY / 2))
    bg = Image.fromarray(np.linspace(BG_TOP, BG_BOTTOM, h).astype(np.uint8)[:, None].repeat(w, 1))
    side = lambda j: "l" if j in LEFT else "r" if j in RIGHT else "c"  # noqa: E731

    def proj(p, look):
        d = p - look
        z = d @ fwd + distance
        return np.stack([w / 2 + focal * (d @ right) / z, h / 2 - focal * (d @ up) / z], -1), z

    for p in pos:
        look = p[0].copy()
        img = bg.copy()
        g = ImageDraw.Draw(img)
        # 地面网格（1 m），只画相机前方的线段
        cx, cy = np.floor(look[:2])
        for k in range(-6, 7):
            for a, b in (([cx + k, cy - 6, 0], [cx + k, cy + 6, 0]), ([cx - 6, cy + k, 0], [cx + 6, cy + k, 0])):
                seg = np.linspace(a, b, 25)
                uv, z = proj(seg, look)
                for i in range(len(seg) - 1):
                    if z[i] > 0.3 and z[i + 1] > 0.3:
                        g.line([tuple(uv[i]), tuple(uv[i + 1])], fill=GRID, width=SS)
        uv, z = proj(p, look)
        for j in np.argsort(-z):  # 远的先画
            if parents[j] >= 0:
                g.line([tuple(uv[parents[j]]), tuple(uv[j])], fill=COLORS[side(j)], width=6 * SS)
        for j in range(len(p)):
            r = (focal * 0.08 / z[j]) if j == HEAD else 4 * SS
            g.ellipse([uv[j][0] - r, uv[j][1] - r, uv[j][0] + r, uv[j][1] + r], fill=COLORS[side(j)])
        yield np.asarray(img.resize(size, Image.LANCZOS))
