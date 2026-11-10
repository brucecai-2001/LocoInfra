"""MuJoCo 离屏"摄影棚"：把训练场景换成白模 + 白地面 + 斜向主光，输出明度、分割（哪个连杆）、地面阴影和相机投影。

风格化在 style.py 里做；这里只负责拿到干净的几何信息。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from runtime.package import MESH_DIR  # noqa: E402

SCENE = PROJECT / "weights" / "dual_v3b" / "scene.xml"

# 连杆明暗分组：关节处的连杆略深，画面上有"分件"的层次
DARK_LINKS = ("hip_roll", "knee", "ankle_pitch", "shoulder_roll", "elbow", "waist_roll")
MID_LINKS = ("ankle_roll", "shoulder_pitch", "hip_pitch")


@dataclass
class Shot:
    lum: np.ndarray  # (H, W) float32，机器人与地面的明度 0–1
    ids: np.ndarray  # (H, W) int32，机器人 geom id，非机器人为 -1
    floor: np.ndarray  # (H, W) bool
    cam_pos: np.ndarray
    cam_R: np.ndarray  # 列为 right / up / forward
    fx: float  # 像素焦距

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """世界坐标 (..., 3) → 像素 (..., 2) 与相机前向深度 (...)。"""
        d = np.asarray(pts, dtype=np.float64) - self.cam_pos
        x, y, z = d @ self.cam_R[:, 0], d @ self.cam_R[:, 1], d @ self.cam_R[:, 2]
        zc = np.maximum(z, 1e-6)
        h, w = self.lum.shape
        return np.stack([w / 2 + self.fx * x / zc, h / 2 - self.fx * y / zc], axis=-1), z


class Stage:
    def __init__(self, width: int = 1920, height: int = 1080, light_dir=(0.45, 0.55, -1.0)):
        xml = SCENE.read_text()
        xml = re.sub(r"<compiler([^>]*)/>", lambda m: f'<compiler{m.group(1)} meshdir="{MESH_DIR}/"/>', xml, count=1)
        xml = re.sub(r'offwidth="\d+" offheight="\d+"', f'offwidth="{width}" offheight="{height}"', xml)
        xml = re.sub(r"<headlight[^>]*/>", '<headlight ambient="0.42 0.42 0.42" diffuse="0 0 0" specular="0 0 0"/>', xml)
        ld = np.asarray(light_dir, dtype=np.float64)
        ld /= np.linalg.norm(ld)
        xml = re.sub(r'<light name="sun"[^>]*/>',
                     f'<light name="sun" pos="0 0 3" dir="{ld[0]:.4f} {ld[1]:.4f} {ld[2]:.4f}" type="directional" '
                     'diffuse="0.58 0.58 0.58" specular="0 0 0" castshadow="true"/>'
                     '<light name="fill" pos="0 0 3" dir="-0.6 -0.3 -0.4" type="directional" '
                     'diffuse="0.16 0.16 0.16" specular="0 0 0" castshadow="false"/>', xml)
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        self.width, self.height = width, height
        self.terrain = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        m.geom_matid[self.terrain] = -1
        m.geom_rgba[self.terrain] = (1, 1, 1, 1)
        for g in range(m.ngeom):
            if g == self.terrain:
                continue
            body = m.body(m.geom_bodyid[g]).name
            v = 0.62 if any(k in body for k in DARK_LINKS) else 0.80 if any(k in body for k in MID_LINKS) else 1.0
            m.geom_rgba[g] = (v, v, v, 1)
            m.geom_matid[g] = -1
        m.vis.map.shadowclip = 2.0  # 阴影贴图只盖机器人周围，边缘更清晰
        m.vis.quality.offsamples = 8
        self.data = mujoco.MjData(m)
        self.rend = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.fq = m.jnt_qposadr[free]
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        self.fovy = float(m.vis.global_.fovy)
        self.fx = 0.5 * height / np.tan(np.deg2rad(self.fovy) / 2)

    def camera(self, lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    def pose(self, qpos: np.ndarray) -> mujoco.MjData:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def shoot(self, qpos: np.ndarray, cam: mujoco.MjvCamera, shading: bool = True) -> Shot:
        """shading=False 时只出分割（给影子 / 残影用，省一遍着色渲染）。"""
        d = self.pose(qpos)
        r = self.rend
        r.update_scene(d, cam, self.opt)
        if shading:
            rgb = r.render()
            lum = rgb[..., 1].astype(np.float32) / 255.0  # 白模：三通道相同
        else:
            lum = np.zeros((self.height, self.width), dtype=np.float32)
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        floor = is_geom & (seg[..., 0] == self.terrain)
        ids = np.where(is_geom & ~floor, seg[..., 0], -1).astype(np.int32)
        gl = r.scene.camera
        pos = (np.asarray(gl[0].pos) + np.asarray(gl[1].pos)) / 2
        fwd = np.asarray(gl[0].forward, dtype=np.float64)
        up = np.asarray(gl[0].up, dtype=np.float64)
        right = np.cross(fwd, up)
        return Shot(lum, ids, floor, pos, np.stack([right, up, fwd], axis=1), self.fx)

    def joint_anchors(self) -> np.ndarray:
        """当前姿态下 22 个转动关节的世界坐标。"""
        m = self.model
        return np.array([self.data.xanchor[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
