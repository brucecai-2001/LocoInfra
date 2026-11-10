"""基础模型 E_ref 在 MuJoCo 里直接跟踪一段参考（motion.npz），录对照视频并记录逐步误差。

画面：人体（视频输入时是原视频，文本生成时是人体骨架渲染）| Q1 参考（运动学回放）| E_ref 物理仿真。摔倒判定与训练终止相同。
"""

from __future__ import annotations

from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image

from runtime import Package, Session
from runtime.body_ref import BodyRef, quat_to_mat
from runtime.selfcheck import HEIGHT_THR, TILT_THR

H = 480  # 画面高
ROBOT_W = 400  # 机器人面板宽
VIDEO_FPS = 25  # 每 2 个控制步一帧（控制 50 Hz）


def _human_frames(video: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(video)
    fps = float(reader.get_meta_data()["fps"])
    frames = []
    w = None
    for f in reader:
        if w is None:
            w = int(round(f.shape[1] * H / f.shape[0] / 16)) * 16
            w = min(max(w, 160), 720)
        frames.append(np.asarray(Image.fromarray(f).resize((w, H))))
    reader.close()
    return frames, fps


def track(package: Path, motion: Path, out_mp4: Path, human_video: Path | None = None, log=print, human_label: str = "人体视频") -> dict:
    pkg = Package.load(package)
    run = Session(pkg)
    ref = BodyRef(pkg, motion)
    m, d = run.robot.model, run.robot.data
    dt = pkg.control_dt
    human, human_fps = _human_frames(human_video) if human_video else ([], 0.0)

    ghost = mujoco.MjData(m)
    run.robot.set_state(ref.root_pos[0], ref.root_quat[0], ref.joint_pos[0], ref.root_lin_vel[0], ref.root_ang_vel_w[0], ref.joint_vel[0])
    run.begin()
    ref.start(run.robot.torso_quat())
    rend = mujoco.Renderer(m, H, ROBOT_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = run.robot.pelvis
    cam.distance, cam.elevation, cam.azimuth = 2.2, -12.0, 160.0

    g = np.array([0.0, 0.0, -1.0])
    stride = int(round(1.0 / (VIDEO_FPS * dt)))
    t_s, err, err_j, h_ref, h_sim = [], [], [], [], []
    fell = None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4, fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p") as w:
        for i in range(ref.T - 1):
            run.apply(*run.model.act_ref(run.obs(), ref.goal(i, run.robot.torso_quat())))
            f = i + 1
            e = np.abs(d.qpos[run.robot.q_adr] - ref.joint_pos[f])
            tilt = abs((quat_to_mat(ref.root_quat[f]).T @ g)[2] - (d.xmat[run.robot.pelvis].reshape(3, 3).T @ g)[2])
            dz = abs(run.robot.pelvis_height() - ref.root_pos[f][2])
            if fell is None and (dz > HEIGHT_THR or tilt > TILT_THR):
                fell = round(f * dt, 2)
                log(f"[track] 摔倒 @ {fell:.2f} s（骨盆高度差 {dz:.3f} m，倾角差 {tilt:.2f}）")
            t_s.append(f * dt)
            err.append(float(e.mean()))
            err_j.append(e)
            h_ref.append(float(ref.root_pos[f][2]))
            h_sim.append(run.robot.pelvis_height())
            if i % stride:
                continue
            ghost.qpos[run.robot.fq : run.robot.fq + 3] = ref.root_pos[f]
            ghost.qpos[run.robot.fq + 3 : run.robot.fq + 7] = ref.root_quat[f]
            ghost.qpos[run.robot.q_adr] = ref.joint_pos[f]
            mujoco.mj_forward(m, ghost)
            rend.update_scene(ghost, cam)
            panels = [rend.render()]
            rend.update_scene(d, cam)
            panels.append(rend.render())
            if human:
                panels.insert(0, human[min(int(f * dt * human_fps), len(human) - 1)])
            w.append_data(np.concatenate(panels, axis=1))
            if i % (stride * 50) == 0:
                log(f"[track] {f * dt:4.1f} / {(ref.T - 1) * dt:.1f} s  关节误差 {err[-1]:.3f} rad")
    rend.close()

    err_a, err_j = np.asarray(err), np.asarray(err_j)
    n_ok = len(err) if fell is None else max(1, int(round(fell / dt)))
    per_joint = err_j[:n_ok].mean(0)
    order = np.argsort(-per_joint)
    return {
        "package": str(package),
        "steps": len(err),
        "duration_s": round(len(err) * dt, 2),
        "fell_at_s": fell,
        "joint_err_mean": round(float(err_a[:n_ok].mean()), 4),
        "joint_err_mean_full": round(float(err_a.mean()), 4),
        "joint_err_p95": round(float(np.percentile(err_a[:n_ok], 95)), 4),
        "worst_joints": [{"name": pkg.joint_names[j], "err": round(float(per_joint[j]), 4)} for j in order[:6]],
        "panels": ([human_label] if human else []) + ["Q1 参考（运动学回放）", "E_ref 物理仿真跟踪"],
        "curve": {
            "t": [round(x, 3) for x in t_s],
            "joint_err": [round(x, 4) for x in err],
            "pelvis_h_ref": [round(x, 4) for x in h_ref],
            "pelvis_h_sim": [round(x, 4) for x in h_sim],
        },
    }
