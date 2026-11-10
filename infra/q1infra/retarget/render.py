"""离屏渲染 any4hdmi 动作（Mac 上没有 mjpython 也能看），可叠加人体目标点。"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np

from any4hdmi.core.format import load_manifest, load_motion


def render_dataset(
    dataset_root: Path,
    out_path: Path,
    keypoints_npz: Path | None = None,
    width: int = 960,
    height: int = 540,
    stills: list[float] | None = None,
) -> list[Path]:
    manifest = load_manifest(dataset_root)
    motion_files = sorted((manifest.root / manifest.payload["motions_subdir"]).rglob("*.npz"))
    qpos = load_motion(motion_files[0]).astype(np.float64)
    model = mj.MjModel.from_xml_path(str(manifest.mjcf_path))
    data = mj.MjData(model)
    fps = 1.0 / manifest.timestep

    kp, kp_fps = None, None
    if keypoints_npz is not None and keypoints_npz.is_file():
        dbg = np.load(keypoints_npz)
        kp, kp_fps = dbg["keypoints"], float(dbg["fps"])

    renderer = mj.Renderer(model, height=height, width=width)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -15.0
    opt = mj.MjvOption()

    still_frames = {int(round(t * fps)) for t in (stills or [])}
    still_paths = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        cam.lookat[:] = [q[0], q[1], 0.4]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        if kp is not None:
            scn = renderer.scene
            for p in kp[min(len(kp) - 1, int(round(i / fps * kp_fps)))]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mj.mjv_initGeom(
                    scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                    p.astype(np.float64), np.eye(3).flatten(), np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32),
                )
                scn.ngeom += 1
        img = renderer.render()
        writer.append_data(img)
        if i in still_frames:
            p = out_path.with_name(f"{out_path.stem}_t{i / fps:05.1f}s.png")
            imageio.imwrite(p, img)
            still_paths.append(p)
    writer.close()
    renderer.close()
    return still_paths
