"""Export a trained single-motion policy as a deployable weight package.

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/export_package.py \
      --ckpt logs/<run>/ckpt_latest.pt --motion-npz data/npz/<set>/<clip>.npz \
      --motion-dir ../jobs/<job>/check/motion --eval-dir logs/<run>/eval \
      --out ../jobs/<job>/artifact --code-commit <git sha>

Package:
  policy.onnx        goal encoder + policy: (obs, goal) -> (action, joint_target); checked against PyTorch
  deploy.yaml        joint order, stance, PD gains, action scale, observation / goal layout, 50 Hz
  deploy_motion.npz  per-frame goal features (T, 2 NJ + 10) and reference torso quaternion, ready to feed
  motion.npz, motion/  the training reference (SuperMimic npz and the any4hdmi source)
  ckpt.pt            full checkpoint (world model, critic, optimizers) for continued training
  report.html, report.json, play_*.mp4   evaluation (copied from --eval-dir)
  meta.json          source-motion hashes, code commit, training mode / time / seed, ONNX parity
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml

from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robot import GOAL_FEAT_DIM, NJ, OBS_DIM, PD_PRESET, SPEC
from supermimic.world_model.heads import OBS_TERMS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_joint(mapping: dict[str, float], default: float | None = None) -> list[float]:
    out = []
    for j in SPEC.joint_names:
        v = next((val for pat, val in mapping.items() if re.fullmatch(pat, j)), default)
        if v is None:
            raise KeyError(j)
        out.append(float(v))
    return out


def robot_params() -> dict:
    cfg = SPEC.robot_cfg()
    kp, kd, effort = {}, {}, {}
    for a in cfg.articulation.actuators:
        for pat in a.target_names_expr:
            kp[pat], kd[pat], effort[pat] = a.stiffness, a.damping, a.effort_limit
    return {
        "kp": per_joint(kp),
        "kd": per_joint(kd),
        "effort_limit": per_joint(effort),
        "rated_torque": per_joint(SPEC.rated_torque()) if SPEC.rated_torque else per_joint(effort),
        "action_scale": per_joint(SPEC.action_scale()),
        "default_joint_pos": per_joint(cfg.init_state.joint_pos, 0.0),
    }


class ExportPolicy(torch.nn.Module):
    def __init__(self, policy, default_joint_pos, action_scale):
        super().__init__()
        self.policy = policy
        self.register_buffer("q0", torch.tensor(default_joint_pos, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(action_scale, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        a = self.policy(obs, goal)
        return a, self.q0 + self.scale * a


def deploy_motion(npz: Path, q0: list[float]) -> dict[str, np.ndarray]:
    d = np.load(npz)
    names = [str(n) for n in d["body_names"]]
    r, a = names.index(SPEC.root_body), names.index(SPEC.anchor_body)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    ref = RefState(t(d["body_pos_w"][:, r]), t(d["body_quat_w"][:, r]), t(d["joint_pos"]), t(d["body_lin_vel_w"][:, r]), t(d["body_ang_vel_w"][:, r]), t(d["joint_vel"]))
    return {
        "goal_feat": goal_features(ref, t(q0)).numpy().astype(np.float32),
        "anchor_quat_w": d["body_quat_w"][:, a].astype(np.float32),
        "root_pos_w": d["body_pos_w"][:, r].astype(np.float32),
        "root_quat_w": d["body_quat_w"][:, r].astype(np.float32),
        "joint_pos": d["joint_pos"].astype(np.float32),
        "fps": np.array([50.0], dtype=np.float32),
    }


def deploy_yaml(cfg: dict, p: dict, goal_dim: int, n_frames: int) -> dict:
    e = cfg["env"]
    K = e["future_frames"]
    H = cfg["policy"]["history"]
    anchor = goal_dim - K * GOAL_FEAT_DIM
    terms = {
        "joint_pos": ("rad", "关节编码器角度减默认站姿 q - default_joint_pos"),
        "joint_vel": ("rad/s", "关节角速度"),
        "base_ang_vel": ("rad/s", "骨盆 IMU 角速度，骨盆坐标系"),
        "projected_gravity": ("1", "骨盆坐标系下的单位重力方向 R_pelvis^T [0, 0, -1]"),
        "actions": ("1", "上一控制周期策略输出的 action（不是关节目标），回合开始为 0"),
    }
    return {
        "robot": SPEC.name,
        "model": "q1/q1_v3/xml/q1_v3_collision.xml",
        "control_hz": 50,
        "joint_names": list(SPEC.joint_names),
        "default_joint_pos": p["default_joint_pos"],
        "action_scale": p["action_scale"],
        "pd": {"preset": PD_PRESET, "kp": p["kp"], "kd": p["kd"], "effort_limit": p["effort_limit"], "rated_torque": p["rated_torque"]},
        "policy": {
            "file": "policy.onnx",
            "inputs": {"obs": [None, H * OBS_DIM], "goal": [None, goal_dim]},
            "outputs": {"action": [None, NJ], "joint_target": [None, NJ]},
            "joint_target": "default_joint_pos + action_scale * action（rad），作为 PD 目标",
            "normalization": "无：观测缩放已在网络内部，直接喂原始量",
        },
        "obs": {
            "history": H,
            "layout": "按项分块（term-major）：每一项连续放 history 帧，帧顺序从旧到新；回合开始时所有历史帧都填第一帧",
            "terms": [{"name": n, "dim": d, "unit": terms[n][0], "desc": terms[n][1]} for n, d in OBS_TERMS],
        },
        "goal": {
            "future_frames": K,
            "frame_offset": "机器人处在参考第 i 帧（首帧 i = 0）时，喂第 i+1 … i+K 帧，超出末帧用末帧",
            "per_frame_features": [
                {"name": "joint_pos", "dim": NJ, "desc": "参考关节角减 default_joint_pos"},
                {"name": "joint_vel", "dim": NJ, "desc": "参考关节角速度"},
                {"name": "projected_gravity", "dim": 3, "desc": "参考骨盆坐标系下的重力方向"},
                {"name": "root_ang_vel_b", "dim": 3, "desc": "参考骨盆角速度，骨盆坐标系"},
                {"name": "root_lin_vel_h", "dim": 3, "desc": "参考骨盆线速度，参考航向坐标系"},
                {"name": "root_height", "dim": 1, "desc": "参考骨盆高度（m）"},
            ],
            "precomputed": "deploy_motion.npz: goal_feat (T, %d)，已按上面顺序算好" % GOAL_FEAT_DIM,
            "anchor_ori": (
                {
                    "dim": anchor,
                    "desc": "接在 K 帧之后：参考躯干（torso_link）相对机器人躯干的旋转 R = R_robot^T R_ref 的前两列，按行展开 [R00, R01, R10, R11, R20, R21]；"
                    "参考用第 i+1 帧（与第一帧目标同帧）；R_robot 由骨盆 IMU 姿态与腰关节角正运动学得到；"
                    "开始播放时把参考整体绕 z 轴旋转，使首帧参考躯干航向与机器人当前躯干航向一致（IMU 航向无绝对基准）",
                    "precomputed": "deploy_motion.npz: anchor_quat_w (T, 4) wxyz",
                }
                if anchor
                else None
            ),
        },
        "motion": {"file": "motion.npz", "frames": n_frames, "fps": 50},
        "start": "从参考首帧姿态（deploy_motion.npz: joint_pos[0]）缓慢插值到位后开始，last action = 0",
    }


def main(
    ckpt: Path,
    motion_npz: Path,
    out: Path,
    motion_dir: Path | None = None,
    eval_dir: Path | None = None,
    code_commit: str = "unknown",
    job: str | None = None,
    opset: int = 17,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu")
    p = robot_params()
    goal_dim = cfg["policy"]["goal_dim"]
    obs_dim = cfg["policy"]["history"] * OBS_DIM
    model = ExportPolicy(policy, p["default_joint_pos"], p["action_scale"]).eval()

    # ---- ONNX + parity against PyTorch (CPU fp32)
    onnx_path = out / "policy.onnx"
    ex_obs, ex_goal = torch.zeros(1, obs_dim), torch.zeros(1, goal_dim)
    torch.onnx.export(
        model, (ex_obs, ex_goal), str(onnx_path), input_names=["obs", "goal"], output_names=["action", "joint_target"],
        dynamic_axes={"obs": {0: "batch"}, "goal": {0: "batch"}, "action": {0: "batch"}, "joint_target": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dm = deploy_motion(motion_npz, p["default_joint_pos"])
    g = torch.Generator().manual_seed(0)
    n = 512
    # Realistic goals (actual reference windows) plus random observations at the scale of real ones.
    T = dm["goal_feat"].shape[0]
    K = cfg["env"]["future_frames"]
    starts = torch.randint(0, T, (n,), generator=g)
    idx = torch.minimum(starts[:, None] + torch.arange(1, K + 1)[None], torch.tensor(T - 1))
    goal = torch.as_tensor(dm["goal_feat"])[idx].reshape(n, -1)
    if goal_dim > goal.shape[1]:
        goal = torch.cat([goal, torch.randn(n, goal_dim - goal.shape[1], generator=g) * 0.3], -1)
    scale = torch.cat([torch.full((cfg["policy"]["history"] * d,), s) for (_, d), s in zip(OBS_TERMS, (0.2, 2.0, 0.5, 0.3, 0.5))])
    obs = torch.randn(n, obs_dim, generator=g) * scale
    with torch.no_grad():
        a_t, q_t = model(obs, goal)
    a_o, q_o = sess.run(None, {"obs": obs.numpy(), "goal": goal.numpy()})
    err_a = float(np.abs(a_o - a_t.numpy()).max())
    err_q = float(np.abs(q_o - q_t.numpy()).max())
    print(f"ONNX vs PyTorch on {n} inputs: max |Δaction| {err_a:.2e}, max |Δjoint_target| {err_q:.2e}")
    assert max(err_a, err_q) < 1e-5, "ONNX parity check failed"

    # ---- deploy spec and motion
    np.savez(out / "deploy_motion.npz", **dm)
    (out / "deploy.yaml").write_text(yaml.safe_dump(deploy_yaml(cfg, p, goal_dim, T), allow_unicode=True, sort_keys=False, width=200, default_flow_style=None))
    shutil.copy2(motion_npz, out / "motion.npz")
    if motion_dir is not None:
        shutil.copytree(motion_dir, out / "motion", dirs_exist_ok=True)
    shutil.copy2(ckpt, out / "ckpt.pt")
    report = None
    if eval_dir is not None:
        for f in eval_dir.iterdir():
            if f.name in ("report.html", "report.json") or f.suffix == ".mp4":
                shutil.copy2(f, out / f.name)
        if (eval_dir / "report.json").exists():
            report = json.loads((eval_dir / "report.json").read_text())

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    source = None
    if motion_dir is not None and (motion_dir / "manifest.json").exists():
        source = json.loads((motion_dir / "manifest.json").read_text()).get("source")
    meta = {
        "job": job,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "robot": SPEC.name,
        "source_motion": {
            "input_file": (source or {}).get("input_file"),
            "input_sha256": (source or {}).get("input_sha256"),
            "motion_npz_sha256": sha256(motion_npz),
            "any4hdmi_manifest_sha256": sha256(motion_dir / "manifest.json") if motion_dir is not None else None,
        },
        "code_commit": code_commit,
        "training": {
            "mode": "resume" if cfg.get("resume") else "scratch",
            "resumed_from": cfg.get("resume"),
            "seed": cfg["seed"],
            "iteration": ck.get("iteration"),
            "train_hours": ck.get("train_hours"),
            "num_envs": cfg["env"]["num_envs"],
            "eval_history": ck.get("eval_history"),
        },
        "evaluation": {"passed": report["passed"], "checks": report["checks"]} if report else None,
        "onnx": {"opset": opset, "parity_max_abs_err": max(err_a, err_q), "parity_inputs": n},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    print("package:", out)
    for f in sorted(out.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:8.2f} MB" if f.is_file() else f"  {f.name}/")


if __name__ == "__main__":
    tyro.cli(main)
