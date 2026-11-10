"""Closed-loop check of an exported package with nothing but the package files.

The policy runs through onnxruntime; observations and goals are assembled in numpy following
``deploy.yaml`` (no SuperMimic / mjlab code on the control path). Physics is CPU MuJoCo on the
training scene (the mjlab-built model: floor, ground-only collisions, position actuators,
armature), started on the first reference frame with foot friction 1.0, like the benchmark.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/deploy_sim_check.py --package ../jobs/<job>/artifact
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import tyro
import yaml


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def yaw_of(q: np.ndarray) -> float:
    R = quat_to_mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def main(package: Path, seconds: float | None = None, device: str = "cuda:0") -> None:
    import onnxruntime as ort

    from supermimic.envs import make_env
    from supermimic.eval.benchmark import deterministic_env_args
    from supermimic.robot import SPEC

    spec = yaml.safe_load((package / "deploy.yaml").read_text())
    dm = np.load(package / "deploy_motion.npz")
    sess = ort.InferenceSession(str(package / "policy.onnx"), providers=["CPUExecutionProvider"])

    # The training scene's compiled model (CPU copy), with the benchmark's foot friction.
    env = make_env(deterministic_env_args([str(package / "motion.npz")], 1), device=device)
    model = env.sim.mj_model
    robot = env.scene["robot"]
    sole_ids = [int(robot.indexing.geom_ids[i]) for i in robot.find_geoms(SPEC.foot_geom_regex)[0]]
    assert tuple(robot.joint_names) == tuple(spec["joint_names"])
    jid = robot.indexing.joint_ids.cpu().tolist()
    q_adr = robot.indexing.joint_q_adr.cpu().numpy()
    v_adr = robot.indexing.joint_v_adr.cpu().numpy()
    free_q = int(robot.indexing.free_joint_q_adr[0])
    free_v = int(robot.indexing.free_joint_v_adr[0])
    torso = int(robot.indexing.body_ids[robot.body_names.index(SPEC.anchor_body)])
    pelvis = int(robot.indexing.body_ids[robot.body_names.index(SPEC.root_body)])
    env.close()
    model.geom_friction[sole_ids, 0] = 1.0
    data = mujoco.MjData(model)

    names = spec["joint_names"]
    ctrl_of = np.zeros(len(names), dtype=int)
    for c in range(model.nu):
        ctrl_of[jid.index(int(model.actuator_trnid[c, 0]))] = c
    assert list(q_adr) == [model.jnt_qposadr[j] for j in jid]
    q0 = np.array(spec["default_joint_pos"])
    H = spec["obs"]["history"]
    K = spec["goal"]["future_frames"]
    decim = int(round(1.0 / spec["control_hz"] / model.opt.timestep))

    # Start on the first reference frame.
    T = dm["goal_feat"].shape[0]
    d0 = np.load(package / "motion.npz")
    bn = [str(b) for b in d0["body_names"]]
    r = bn.index(SPEC.root_body)
    data.qpos[free_q : free_q + 3] = dm["root_pos_w"][0]
    data.qpos[free_q + 3 : free_q + 7] = dm["root_quat_w"][0]
    data.qpos[q_adr] = dm["joint_pos"][0]
    R0 = quat_to_mat(dm["root_quat_w"][0])
    data.qvel[free_v : free_v + 3] = d0["body_lin_vel_w"][0, r]
    data.qvel[free_v + 3 : free_v + 6] = R0.T @ d0["body_ang_vel_w"][0, r]
    data.qvel[v_adr] = d0["joint_vel"][0]
    mujoco.mj_forward(model, data)

    # Heading alignment of the reference (no absolute yaw on hardware).
    dyaw = yaw_of(data.xquat[torso]) - yaw_of(dm["anchor_quat_w"][0])
    align = yaw_quat(dyaw)

    hist = {t["name"]: deque(maxlen=H) for t in spec["obs"]["terms"]}
    last_action = np.zeros(len(names), dtype=np.float32)

    def sense():
        quat = data.qpos[free_q + 3 : free_q + 7]
        R = quat_to_mat(quat)
        return {
            "joint_pos": data.qpos[q_adr] - q0,
            "joint_vel": data.qvel[v_adr].copy(),
            "base_ang_vel": data.qvel[free_v + 3 : free_v + 6].copy(),
            "projected_gravity": R.T @ np.array([0.0, 0.0, -1.0]),
            "actions": last_action.copy(),
        }

    def push_obs(first: bool):
        o = sense()
        for k, dq in hist.items():
            for _ in range(H if first else 1):
                dq.append(o[k].astype(np.float32))

    push_obs(first=True)
    steps = T - 1 if seconds is None else min(T - 1, int(seconds * spec["control_hz"]))
    errs, fell_at = [], None
    for i in range(steps):
        obs = np.concatenate([np.concatenate(list(hist[t["name"]])) for t in spec["obs"]["terms"]])[None]
        idx = np.minimum(np.arange(i + 1, i + 1 + K), T - 1)
        goal = dm["goal_feat"][idx].reshape(-1)
        if spec["goal"].get("anchor_ori"):
            ref_q = quat_mul(align, dm["anchor_quat_w"][idx[0]])
            Rr, Rf = quat_to_mat(data.xquat[torso]), quat_to_mat(ref_q)
            goal = np.concatenate([goal, (Rr.T @ Rf)[:, :2].reshape(-1)])
        action, target = sess.run(None, {"obs": obs.astype(np.float32), "goal": goal[None].astype(np.float32)})
        data.ctrl[ctrl_of] = target[0]
        for _ in range(decim):
            mujoco.mj_step(model, data)
        last_action = action[0]
        push_obs(first=False)
        f = i + 1
        errs.append(np.abs(data.qpos[q_adr] - dm["joint_pos"][f]).mean())
        dz = abs(data.xpos[pelvis][2] - dm["root_pos_w"][f][2])
        g = np.array([0.0, 0.0, -1.0])
        tilt = abs((quat_to_mat(dm["root_quat_w"][f]).T @ g)[2] - (quat_to_mat(data.xquat[pelvis]).T @ g)[2])
        if dz > SPEC.root_height_threshold or tilt > SPEC.root_tilt_threshold:
            fell_at = f / spec["control_hz"]
            break
    res = {"steps": len(errs), "of": steps, "fell_at_s": fell_at, "joint_err_mean": float(np.mean(errs))}
    print(json.dumps(res))
    (package / "deploy_sim_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
