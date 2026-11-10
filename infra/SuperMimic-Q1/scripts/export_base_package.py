"""Export the base motion model (E_ref + E_tele + shared decoder) as one deployable weight package
(docs/deployment.md).

  SUPERMIMIC_ROBOT=q1 .venv/bin/python scripts/export_base_package.py \
      --ckpt logs/<run>/ckpt_latest.pt --out ../jobs/<run>/artifact_base --eval-dir logs/<run>/eval --code-commit <git sha>

Package:
  policy_ref.onnx    E_ref + decoder:  (obs, body_ref)     -> (action, joint_target, z)   video / full-reference tracking
  policy_tele.onnx   E_tele + decoder: (obs, slots, mask)  -> (action, joint_target, z)   teleoperation
  decoder.onnx       decoder alone:    (obs, z)            -> (action, joint_target)      z blending when switching apps
                     All three carry the same decoder weights; every graph is checked against PyTorch.
  deploy.yaml        joint order, stance, PD, action scale, observation layout, body_ref layout, slot registry
                     (generated from supermimic/tele/slots.py), training command ranges, 50 Hz
  scene.xml          the training scene (floor + Q1, training collisions / actuators) for CPU MuJoCo; meshes
                     resolve against q1/q1_v3/meshes (``meshdir`` left empty)
  ckpt.pt            full checkpoint (optional, --copy-ckpt)
  *.json             evaluation results copied from --eval-dir
  meta.json          code commit, training run / iterations / time, registry version, ONNX parity
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml

from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robot import NJ, OBS_DIM, PD_PRESET, SPEC
from supermimic.tele import slots as S
from supermimic.world_model.heads import OBS_TERMS

from export_package import deploy_yaml as single_motion_yaml  # noqa: E402  (same directory)
from export_package import robot_params  # noqa: E402

OBS_TERM_DESC = {
    "joint_pos": ("rad", "关节编码器角度减默认站姿 q - default_joint_pos"),
    "joint_vel": ("rad/s", "关节角速度"),
    "base_ang_vel": ("rad/s", "骨盆 IMU 角速度，骨盆坐标系"),
    "projected_gravity": ("1", "骨盆坐标系下的单位重力方向 R_pelvis^T [0, 0, -1]"),
    "actions": ("1", "上一控制周期策略输出的 action（不是关节目标），回合开始为 0"),
}


class _Out(torch.nn.Module):
    def __init__(self, policy, q0, scale):
        super().__init__()
        self.policy = policy
        self.register_buffer("q0", torch.tensor(q0, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def act(self, z, obs):
        a = self.policy.decoder(z, obs)
        return a, self.q0 + self.scale * a


class RefGraph(_Out):
    def forward(self, obs, body_ref):
        z = self.policy.latent(obs, body_ref, "ref")
        return (*self.act(z, obs), z)


class TeleGraph(_Out):
    def forward(self, obs, slots, mask):
        z = self.policy.latent(obs, torch.cat([slots, mask], dim=-1), "tele")
        return (*self.act(z, obs), z)


class DecoderGraph(_Out):
    def forward(self, obs, z):
        return self.act(z, obs)


def scene_xml(device: str) -> tuple[str, dict]:
    """The training scene as MJCF (mjlab's spec) plus the simulator options that live outside it."""
    from supermimic.envs import make_env
    from supermimic.eval.benchmark import deterministic_env_args

    ck_npz = next(Path("data/npz").rglob("*.npz"))
    env = make_env(deterministic_env_args([str(ck_npz)], 1), device=device)
    xml = env.scene.spec.to_xml()
    m = env.sim.mj_model
    opt = {
        "timestep": float(m.opt.timestep), "integrator": int(m.opt.integrator), "cone": int(m.opt.cone),
        "solver": int(m.opt.solver), "iterations": int(m.opt.iterations), "ls_iterations": int(m.opt.ls_iterations),
        "foot_friction": 1.0,
    }
    env.close()
    # Older MuJoCo releases reject the empty nested <default/> that MjSpec writes.
    xml = re.sub(r"\n\s*<default/>", "", xml)
    return xml, opt


def export(model: torch.nn.Module, args: tuple, names_in: list[str], names_out: list[str], path: Path, opset: int) -> None:
    torch.onnx.export(
        model, args, str(path), input_names=names_in, output_names=names_out,
        dynamic_axes={n: {0: "batch"} for n in names_in + names_out}, opset_version=opset, dynamo=False,
    )


def parity(model: torch.nn.Module, path: Path, feeds: dict[str, torch.Tensor]) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(*feeds.values())
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    return float(max(np.abs(g - r.numpy()).max() for g, r in zip(got, ref)))


def deploy_yaml(cfg: dict, p: dict, sim_opt: dict, goal_dim: int) -> dict:
    H = cfg["policy"]["history"]
    tc = cfg["tele"]
    ref_goal = single_motion_yaml(cfg, p, goal_dim, 0)["goal"]
    ref_goal.pop("precomputed", None)
    if ref_goal.get("anchor_ori"):
        ref_goal["anchor_ori"].pop("precomputed", None)
    io = lambda **kw: {k: [None, v] for k, v in kw.items()}  # noqa: E731
    return {
        "robot": SPEC.name,
        "model": "scene.xml（训练场景；网格在 q1/q1_v3/meshes）",
        "control_hz": 50,
        "joint_names": list(SPEC.joint_names),
        "default_joint_pos": p["default_joint_pos"],
        "action_scale": p["action_scale"],
        "pd": {"preset": PD_PRESET, "kp": p["kp"], "kd": p["kd"], "effort_limit": p["effort_limit"], "rated_torque": p["rated_torque"]},
        "obs": {
            "history": H,
            "layout": "按项分块（term-major）：每一项连续放 history 帧，帧顺序从旧到新；回合开始时所有历史帧都填第一帧",
            "terms": [{"name": n, "dim": d, "unit": OBS_TERM_DESC[n][0], "desc": OBS_TERM_DESC[n][1]} for n, d in OBS_TERMS],
        },
        "graphs": {
            "ref": {"file": "policy_ref.onnx", "inputs": io(obs=H * OBS_DIM, body_ref=goal_dim), "outputs": io(action=NJ, joint_target=NJ, z=cfg["policy"]["latent_dim"])},
            "tele": {"file": "policy_tele.onnx", "inputs": io(obs=H * OBS_DIM, slots=S.SLOT_DIM, mask=S.NUM_SLOTS), "outputs": io(action=NJ, joint_target=NJ, z=cfg["policy"]["latent_dim"])},
            "decoder": {"file": "decoder.onnx", "inputs": io(obs=H * OBS_DIM, z=cfg["policy"]["latent_dim"]), "outputs": io(action=NJ, joint_target=NJ)},
            "joint_target": "default_joint_pos + action_scale * action（rad），作为 PD 目标",
            "normalization": "无：观测、body_ref、槽位的缩放都在网络内部，直接喂原始物理量",
            "switch": "两个应用切换时，在若干帧内对两个图输出的 z 线性插值，再喂 decoder.onnx（或经站姿过渡）",
        },
        "body_ref": ref_goal,
        "tele": {
            "registry": S.registry_yaml(),
            "train_range": {
                "vx": list(tc["vx_range"]), "vy": list(tc["vy_range"]), "wz": list(tc["wz_range"]), "height": list(tc["height_range"]),
                "torso_pitch_rad": list(tc["pitch_range"]), "torso_roll_rad": list(tc["roll_range"]),
                "height_rate_m_s": tc["height_rate"], "hands": "动作库里 tele_ok 帧的手位置（可达范围）",
            },
            "adapter": "设备到槽位的映射、死区、低通、限幅、高度变化率限制都在应用层；网络只接收 Q1 尺度、Q1 坐标系下的物理目标",
        },
        "sim": sim_opt,
        "stand_height": S.H_STAND,
        "start": "从默认站姿开始（骨盆高度 stand_height），last action = 0；遥操作默认 mask 只开 base、其余槽位取默认值。"
        "骨盆低于 stand_height + Δh - 0.15 m 或倾角过大视为摔倒",
    }


def main(ckpt: Path, out: Path, eval_dir: Path | None = None, code_commit: str = "unknown", copy_ckpt: bool = False, device: str = "cuda:0", opset: int = 17) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu")
    assert "tele" in policy.encoders, "checkpoint has no E_tele"
    p = robot_params()
    goal_dim = cfg["policy"]["goal_dim"]
    obs_dim = cfg["policy"]["history"] * OBS_DIM
    zd = cfg["policy"]["latent_dim"]
    q0, sc = p["default_joint_pos"], p["action_scale"]

    g = torch.Generator().manual_seed(0)
    n = 512
    obs_scale = torch.cat([torch.full((cfg["policy"]["history"] * d,), s) for (_, d), s in zip(OBS_TERMS, (0.2, 2.0, 0.5, 0.3, 0.5))])
    obs = torch.randn(n, obs_dim, generator=g) * obs_scale
    body_ref = torch.randn(n, goal_dim, generator=g) * 0.3
    slots = S.default_vector() + torch.randn(n, S.SLOT_DIM, generator=g) * S.scale_vector()
    mask = (torch.rand(n, S.NUM_SLOTS, generator=g) > 0.3).float() * (~S.reserved_mask()).float()
    z = torch.rand(n, zd, generator=g) * 2 - 1

    graphs = {
        "policy_ref.onnx": (RefGraph(policy, q0, sc), {"obs": obs, "body_ref": body_ref}, ["action", "joint_target", "z"]),
        "policy_tele.onnx": (TeleGraph(policy, q0, sc), {"obs": obs, "slots": slots, "mask": mask}, ["action", "joint_target", "z"]),
        "decoder.onnx": (DecoderGraph(policy, q0, sc), {"obs": obs, "z": z}, ["action", "joint_target"]),
    }
    errs = {}
    for name, (model, feeds, outs) in graphs.items():
        model.eval()
        export(model, tuple(v[:1] for v in feeds.values()), list(feeds), outs, out / name, opset)
        errs[name] = parity(model, out / name, feeds)
        print(f"{name}: ONNX vs PyTorch on {n} inputs, max abs err {errs[name]:.2e}")
        assert errs[name] < 1e-5, f"ONNX parity check failed for {name}"

    xml, sim_opt = scene_xml(device)
    (out / "scene.xml").write_text(xml)
    (out / "deploy.yaml").write_text(yaml.safe_dump(deploy_yaml(cfg, p, sim_opt, goal_dim), allow_unicode=True, sort_keys=False, width=200, default_flow_style=None))
    if copy_ckpt:
        shutil.copy2(ckpt, out / "ckpt.pt")
    if eval_dir is not None:
        for f in eval_dir.glob("*.json"):
            shutil.copy2(f, out / f.name)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "robot": SPEC.name,
        "code_commit": code_commit,
        "slot_registry_version": S.REGISTRY_VERSION,
        "training": {"run": cfg["run_name"], "resumed_from": cfg.get("resume"), "iteration": ck.get("iteration"), "train_hours": ck.get("train_hours"),
                     "seed": cfg["seed"], "data": cfg["data"], "eval_history": ck.get("eval_history")},
        "onnx": {"opset": opset, "parity_max_abs_err": errs, "parity_inputs": n},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    print("package:", out)
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    tyro.cli(main)
