"""Headless replay of a trained policy on one clip -> mp4 (robot solid, reference tracked bodies as red spheres).

  MUJOCO_GL=egl SUPERMIMIC_ROBOT=q1 uv run python scripts/render_policy.py \
      --ckpt logs/q1_walk/ckpt_latest.pt --npz-dir data/npz/q1_walk --clip lafan_walk1_subject1 --seconds 30

Deterministic evaluation settings (no noise, no DR), starting at the clip's first frame (or --start-s).
``--interface tele``: E_tele only gets the slots derived from the reference (``--slots``, default every
non-reserved slot); the hand targets are drawn as green spheres placed relative to the robot's own head,
as a headset operator would command them.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robot import SPEC


@torch.no_grad()
def main(
    ckpt: Path,
    npz_dir: Path,
    clip: str,
    seconds: float = 30.0,
    start_s: float = 0.0,
    out: Path | None = None,
    width: int = 960,
    height: int = 540,
    device: str = "cuda:0",
    interface: str = "ref",
    slots: str = "all",
) -> None:
    from supermimic.kinematics.rotations import heading_quat, quat_apply
    from supermimic.tele import slots as S

    from eval_library import mask_of  # noqa: E402  (same directory)

    files = [str(f) for f in discover_npz(npz_dir, names=[clip])]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, num_envs=1, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
            action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=False, seed=0,
            anchor_obs=cfg["env"].get("anchor_obs", False), anchor_ori_only=cfg["env"].get("anchor_ori_only", False),
        ),
        device=device,
    )
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    start = int(lib.clip_start[0]) + int(round(start_s * 50))
    cmd.forced_frame[:] = start
    obs, _ = env.reset()
    origin = env.scene.env_origins[0].cpu().numpy()
    robot = env.scene["robot"]
    q_adr = np.concatenate([robot.indexing.free_joint_q_adr.cpu().numpy(), robot.indexing.joint_q_adr.cpu().numpy()])

    qs, refs, kps, rews, fell_at = [], [], [], [], None
    n_steps = int(seconds * 50)
    torso = cmd.tele_idx[0]
    if interface == "tele":
        cmd.ensure_tele_labels()
        mask = mask_of(slots).to(device)[None]
    for t in range(n_steps):
        if interface == "tele":
            target = lib.tele_slots[cmd.frame]
            act = policy(obs["policy"], torch.cat([target, mask], dim=-1), "tele")
            # Hand targets as the operator would see them: relative to the robot's own head.
            tq = cmd.robot_body_quat_w[:, torso]
            head = S.head_position(cmd.robot_body_pos_w[:, torso], tq)
            hq = heading_quat(tq)
            pts = [head] + [head + quat_apply(hq, target[:, S.SLICE[h]]) for h in ("hand_L", "hand_R") if mask[0, S.SLOT_NAMES.index(h)] > 0]
            kps.append(np.stack([(p[0] - env.scene.env_origins[0]).cpu().numpy() for p in pts]))
        else:
            act = policy(obs["policy"], obs["goal"])
        obs, rew, term, tout, _ = env.step(act)
        q = env.sim.data.qpos[0].cpu().numpy()[q_adr].astype(np.float64)
        q[0:3] -= origin
        qs.append(q)
        refs.append((cmd.body_pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
        rews.append(float(rew[0]))
        if bool(term[0]) and fell_at is None:
            fell_at = t
            break
        if bool(cmd.clip_end[0]):
            break
    env.close()

    norm_ret = sum(rews) / (0.1 * max(len(rews), 1))
    print(f"{clip}: {len(qs)} steps ({len(qs) / 50:.1f}s), normalized return {norm_ret:.3f}, " + ("FELL at %.1fs" % (fell_at / 50) if fell_at is not None else "no fall"))

    model = mujoco.MjModel.from_xml_path(str(SPEC.xml_path()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -15.0
    out = out or ckpt.parent / f"play_{interface}_{clip}_{ckpt.stem}.mp4"
    writer = imageio.get_writer(out, fps=50, codec="libx264", quality=7, macro_block_size=8)

    def sphere(scn, p, r, rgba):
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, np.float64), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
            scn.ngeom += 1

    for i, (q, ref) in enumerate(zip(qs, refs)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [q[0], q[1], 0.4]
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        for p in ref:
            sphere(scn, p, 0.015, (1.0, 0.2, 0.2, 0.8))
        if kps:
            for p in kps[i]:
                sphere(scn, p, 0.03, (0.2, 0.9, 0.3, 0.6))
        writer.append_data(renderer.render())
    writer.close()
    print("video:", out)


if __name__ == "__main__":
    tyro.cli(main)
