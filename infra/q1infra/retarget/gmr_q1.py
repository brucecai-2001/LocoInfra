"""人体动作（LAFAN1 BVH / SMPL-X）→ Q1 的 GMR 重定向，输出 any4hdmi 数据集。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from general_motion_retargeting import params as gmr_params
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

from any4hdmi.core.format import write_manifest
from any4hdmi.scripts.preprocess.gmr import convert_dataset


def qpos_names_from_model(model: mj.MjModel) -> list[str]:
    """any4hdmi 的 qpos 命名（根 7 维 root_* + 各铰链关节名）。any4hdmi 自带的版本拿 numpy 整数和
    mjtJoint 枚举比较，MuJoCo 3.16 起两者不再相等，这里按整数比较。"""
    names: list[str] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt == int(mj.mjtJoint.mjJNT_FREE):
            names += ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"]
        elif jt in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            names.append(model.joint(j).name)
        else:
            raise ValueError(f"unsupported joint type {jt} for {model.joint(j).name}")
    assert len(names) == model.nq
    return names

INFRA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = INFRA_ROOT.parent
Q1_MJCF = PROJECT_ROOT / "q1" / "q1_v3" / "xml" / "q1_v3_collision.xml"
IK_CONFIGS = {
    "bvh_lafan1": Path(__file__).with_name("q1_ik_config.json"),
    "smplx": Path(__file__).with_name("q1_smplx_ik_config.json"),
}
ROBOT_NAME = "kabutack_q1"
OUTPUT_FPS = 50
SOLE_GEOM_PATTERN = re.compile(r"^(left|right)_sole_\d+$")
# SMPL-X 人体模型文件（SMPLX_NEUTRAL / MALE / FEMALE .npz 或 .pkl）默认目录
SMPLX_MODEL_DIR = INFRA_ROOT / "data" / "body_models" / "smplx"
SMPLX_BODY_JOINTS = 22  # 骨盆 + 21 个身体关节；Q1 无手腕、手指，手、下颌、眼睛不用
SIDES = ("left", "right")
# 着地脚放平：脚踝目标离估计地面 < 2 cm 完全放平、> 5 cm 不动，中间线性；水平速度、偏航角速度超过
# 下面的值后线性退出（到 2 倍为 0）——转身时脚以前掌 / 脚跟为轴拧转，强行放平会整只脚贴地滑转
FLAT_HEIGHT = (0.02, 0.05)
FLAT_SPEED = 0.15
FLAT_YAW_RATE = 1.0
FLAT_ROT_W = 100.0  # 完全放平时脚的朝向权重（配置里是 10）
LOCAL_GROUND_S = 0.5
LIFT_PASSES = 4  # 骨盆下限：按 IK 结果的骨盆高度差补抬高量，重解的最多轮数
LIFT_TOL = 0.005


def register_q1() -> None:
    gmr_params.ROBOT_XML_DICT[ROBOT_NAME] = Q1_MJCF
    for src, path in IK_CONFIGS.items():
        gmr_params.IK_CONFIG_DICT.setdefault(src, {})[ROBOT_NAME] = path
    gmr_params.ROBOT_BASE_DICT[ROBOT_NAME] = "pelvis"
    gmr_params.VIEWER_CAM_DISTANCE_DICT[ROBOT_NAME] = 1.5


class Q1Retargeting(GeneralMotionRetargeting):
    """着地脚的朝向目标改成水平（只保留偏航）并加大朝向权重。

    Q1 踝关节行程比人小（pitch −0.77…+0.42、roll ±0.33 rad），深蹲、仆步这类动作按人体朝向跟踪时，
    IK 会让脚侧翻 / 踮脚去凑膝、髋位置，物理上站不住。放平后 IK 改为抬高骨盆、收窄步幅。
    每帧调用 retarget 前设置 self.flat（人体脚名 → 放平程度 0..1）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat: dict[str, float] = {}
        self._rot_w = [
            (tasks, {e[0]: e[2] for e in table.values()})
            for tasks, table in ((self.human_body_to_task1, self.ik_match_table1), (self.human_body_to_task2, self.ik_match_table2))
        ]

    def update_targets(self, human_data, offset_to_ground=False):
        super().update_targets(human_data, offset_to_ground)
        for name, w in self.flat.items():
            pos, quat = self.scaled_human_data[name]
            rot = R.from_quat(quat, scalar_first=True)
            x = rot.apply([1.0, 0.0, 0.0])
            level = R.from_euler("z", np.arctan2(x[1], x[0]))
            target = Slerp([0.0, 1.0], R.concatenate([rot, level]))(w) if w < 1.0 else level
            for tasks, rot_w in self._rot_w:
                if name in tasks:
                    tasks[name].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target.as_quat(scalar_first=True)), pos))
                    tasks[name].set_orientation_cost(rot_w[name] + w * (FLAT_ROT_W - rot_w[name]))


@dataclass
class HumanMotion:
    frames: list[dict[str, list[np.ndarray]]]
    parents: dict[str, str | None]
    rest_pos: dict[str, np.ndarray]
    fps: float


def _bvh_frame_time(path: Path) -> float:
    with path.open() as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                return float(line.split(":")[1])
    raise ValueError(f"No 'Frame Time' in {path}")


def load_lafan_motion(path: Path, start_sec: float | None, end_sec: float | None) -> HumanMotion:
    fps = 1.0 / _bvh_frame_time(path)
    frames, _ = load_bvh_file(str(path), format="lafan1")
    start = 0 if start_sec is None else int(round(start_sec * fps))
    end = len(frames) if end_sec is None else min(len(frames), int(round(end_sec * fps)))
    if end - start < 2:
        raise ValueError(f"Clip [{start}, {end}) is empty (total {len(frames)} frames)")

    anim = read_bvh(str(path))
    parents: dict[str, str | None] = {}
    rest_pos: dict[str, np.ndarray] = {}
    for i, bone in enumerate(anim.bones):
        p = anim.parents[i]
        parents[bone] = None if p < 0 else anim.bones[p]
        offset = anim.offsets[i] / 100.0
        rest_pos[bone] = offset if p < 0 else rest_pos[anim.bones[p]] + offset
    return HumanMotion(frames=frames[start:end], parents=parents, rest_pos=rest_pos, fps=fps)


def _smplx_model_file(model_dir: Path, gender: str) -> Path:
    for g in (gender, "neutral"):
        for ext in ("npz", "pkl"):
            f = model_dir / f"SMPLX_{g.upper()}.{ext}"
            if f.exists():
                return f
    raise FileNotFoundError(
        f"{model_dir} 下没有 SMPLX_{gender.upper()} / SMPLX_NEUTRAL 的 .npz 或 .pkl；"
        "从 https://smpl-x.is.tue.mpg.de 注册下载后放到这里"
    )


def smplx_rest_joints(model_dir: Path = SMPLX_MODEL_DIR) -> Path:
    """SMPLX_NEUTRAL 零姿态、零体型的 22 个身体关节点（模型坐标系）及父关节表、关节名。

    受 SMPL-X 许可约束不随仓库分发：首次调用时用用户自己下载的模型算出，缓存在模型目录下。
    """
    cache = model_dir / "smplx_neutral_rest_joints.npz"
    if cache.exists():
        return cache
    import smplx
    from smplx.joint_names import JOINT_NAMES

    model_file = _smplx_model_file(model_dir, "neutral")
    body = smplx.SMPLX(model_path=str(model_file), gender="neutral", use_pca=False, ext=model_file.suffix[1:])
    joints = body().joints[0, :SMPLX_BODY_JOINTS].detach().numpy().astype(np.float64)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, joints=joints, parents=body.parents[:SMPLX_BODY_JOINTS].numpy().astype(np.int64),
             names=np.array(JOINT_NAMES[:SMPLX_BODY_JOINTS]))
    tmp.replace(cache)
    return cache


def load_smplx_motion(
    path: Path, model_dir: Path, start_sec: float | None, end_sec: float | None, max_fps: float = 60.0, fps: float | None = None
) -> tuple[HumanMotion, dict]:
    """AMASS 格式的 SMPL-X npz（Z 朝上、米）→ 22 个身体关节的全局位置与朝向。

    参数键：root_orient (T,3) + pose_body (T,63)，或 poses (T,165)；trans (T,3)；betas；gender；
    mocap_frame_rate（缺失时必须给 fps）。源帧率高于 max_fps 时按整数步长抽帧（之后 qpos 再插值到 50 Hz）。
    """
    d = np.load(path, allow_pickle=True)
    # AMASS 的 SMPL+H 版（poses (T,156)，无 surface_model_type）：22 个身体关节的树和顺序与 SMPL-X 相同，
    # 旋转直接用；体型参数空间不同，改用 SMPL-X 标准体型（之后按 Q1 骨长逐段缩放，体型差异基本被吸收）。
    smplh = "pose_body" not in d and "poses" in d and d["poses"].shape[1] == 156 and "surface_model_type" not in d
    smt = str(d["surface_model_type"]) if "surface_model_type" in d else "smplx"
    if not smplh and (smt != "smplx" or ("poses" in d and "pose_body" not in d and d["poses"].shape[1] != 165)):
        raise ValueError(f"{path.name} 既不是 SMPL-X 也不是 AMASS SMPL+H（surface_model_type={smt}）")
    if "pose_body" in d:
        root_orient, pose_body = d["root_orient"], d["pose_body"][:, :63]
    else:
        root_orient, pose_body = d["poses"][:, :3], d["poses"][:, 3:66]
    trans = d["trans"]
    if fps is None:
        key = next((k for k in ("mocap_frame_rate", "mocap_framerate", "fps") if k in d), None)
        if key is None:
            raise ValueError(f"{path.name} 没有帧率字段，用 --smplx-fps 指定")
        fps = float(d[key])
    gender = str(d["gender"]) if "gender" in d else "neutral"
    gender = gender.replace("b'", "").replace("'", "").lower()
    betas = np.asarray(d["betas"], dtype=np.float32).reshape(-1)
    if smplh:
        betas = np.zeros(10, dtype=np.float32)

    n = len(trans)
    start = 0 if start_sec is None else int(round(start_sec * fps))
    end = n if end_sec is None else min(n, int(round(end_sec * fps)))
    stride = max(1, int(fps // max_fps))
    idx = np.arange(start, end, stride)
    if len(idx) < 2:
        raise ValueError(f"Clip [{start}, {end}) is empty (total {n} frames)")
    fps /= stride

    local = np.concatenate([root_orient[idx][:, None], pose_body[idx].reshape(-1, 21, 3)], axis=1)  # (T, 22, 3)
    if smplh:
        # 标准体型下身体关节位置只是静止关节点的正运动学（姿态混合形状只改网格、不改关节）
        model_file = smplx_rest_joints(model_dir)
        rj = np.load(model_file)
        rest, parents, names = rj["joints"], rj["parents"].tolist(), [str(x) for x in rj["names"]]
        rot = R.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(len(idx), SMPLX_BODY_JOINTS, 3, 3)
        G = np.empty_like(rot)
        pos = np.empty((len(idx), SMPLX_BODY_JOINTS, 3))
        for j in range(SMPLX_BODY_JOINTS):
            if j == 0:
                G[:, 0], pos[:, 0] = rot[:, 0], rest[0] + trans[idx]
            else:
                p = parents[j]
                G[:, j] = G[:, p] @ rot[:, j]
                pos[:, j] = pos[:, p] + G[:, p] @ (rest[j] - rest[p])
    else:
        import smplx
        import torch
        from smplx.joint_names import JOINT_NAMES

        model_file = _smplx_model_file(model_dir, gender)
        body = smplx.SMPLX(model_path=str(model_file), gender=gender, use_pca=False, num_betas=len(betas), ext=model_file.suffix[1:])
        parents = body.parents[:SMPLX_BODY_JOINTS].tolist()
        names = list(JOINT_NAMES[:SMPLX_BODY_JOINTS])
        b = torch.as_tensor(betas)[None]

        def joints(go, bp, tr):
            k = len(go)
            z = lambda m: torch.zeros(k, m)  # noqa: E731
            with torch.no_grad():
                out = body(betas=b.expand(k, -1), global_orient=go, body_pose=bp, transl=tr,
                           left_hand_pose=z(45), right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3),
                           expression=z(body.num_expression_coeffs))
            return out.joints[:, :SMPLX_BODY_JOINTS].numpy()

        rest = joints(torch.zeros(1, 3), torch.zeros(1, 63), torch.zeros(1, 3))[0]
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32))  # noqa: E731
        pos = np.concatenate([joints(t(root_orient[c]), t(pose_body[c]), t(trans[c])) for c in np.array_split(idx, max(1, len(idx) // 256))])

    frames = []
    for f in range(len(idx)):
        rots: list[R] = []
        frame = {}
        for j, name in enumerate(names):
            rot = R.from_rotvec(local[f, j]) if j == 0 else rots[parents[j]] * R.from_rotvec(local[f, j])
            rots.append(rot)
            frame[name] = [pos[f, j], rot.as_quat(scalar_first=True)]
        frames.append(frame)
    motion = HumanMotion(
        frames=frames,
        parents={n: (None if j == 0 else names[parents[j]]) for j, n in enumerate(names)},
        rest_pos={n: rest[j] for j, n in enumerate(names)},
        fps=fps,
    )
    info = {"gender": gender, "num_betas": len(betas), "model_file": model_file.name, "source_stride": stride, "source_model": "smplh" if smplh else "smplx"}
    return motion, info


@dataclass
class HumanScaler:
    """人体 → Q1 尺寸的位置变换。

    - 根：位置整体乘 root_scale（根高度、水平位移都按"骨盆到踝"长度比缩放）
    - 锚点骨（髋、肩）：不按人体骨长缩放，而是用人体骨盆 / 胸腔的朝向驱动 Q1 自己的
      骨盆→髋、骨盆→腰→肩偏移；人体和 Q1 在这两处的偏移方向差别大，等比缩放会把脚和手放错
    - 其余骨：相对父骨的向量乘该段的长度比
    """

    root_scale: float
    bone_scales: dict[str, float]
    anchors: dict[str, list[tuple[str, R, np.ndarray]]]
    foot_mod: dict[str, tuple[str, str]]  # 新骨名 → (取位置的骨, 取朝向的骨)

    def scale_frame(
        self, frame: dict[str, list[np.ndarray]], parents: dict[str, str | None]
    ) -> dict[str, list[np.ndarray]]:
        """parents 的键顺序保证父节点先于子节点；朝向不变。"""
        out: dict[str, list[np.ndarray]] = {}
        for bone, parent in parents.items():
            pos, quat = np.asarray(frame[bone][0]), np.asarray(frame[bone][1])
            if parent is None:
                out[bone] = [pos * self.root_scale, quat]
            elif bone in self.anchors:
                hops = self.anchors[bone]
                p = out[hops[0][0]][0].copy()
                for human_bone, rot_offset, vec in hops:
                    p += (R.from_quat(frame[human_bone][1], scalar_first=True) * rot_offset).apply(vec)
                out[bone] = [p, quat]
            else:
                seg = pos - np.asarray(frame[parent][0])
                out[bone] = [out[parent][0] + seg * self.bone_scales.get(bone, 1.0), quat]
        for name, (pos_bone, rot_bone) in self.foot_mod.items():
            out[name] = [out[pos_bone][0], out[rot_bone][1]]
        return out


def build_scaler(cfg: dict, human: HumanMotion, model: mj.MjModel) -> HumanScaler:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    robot_pos = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}
    robot_rot = {
        model.body(i).name: R.from_quat(data.xquat[i], scalar_first=True) for i in range(model.nbody)
    }

    def ratio(spec: dict) -> float:
        h0, h1 = spec["human"]
        r0, r1 = spec["robot"]
        human_len = np.linalg.norm(human.rest_pos[h1] - human.rest_pos[h0])
        robot_len = np.linalg.norm(robot_pos[r1] - robot_pos[r0])
        return float(robot_len / human_len)

    spec = cfg["segment_scaling"]
    bone_scales = {bone: ratio(seg) for seg in spec["segments"] for bone in seg["bones"]}

    rot_offsets = {
        (entry[0], robot_body): R.from_quat(entry[4], scalar_first=True)
        for robot_body, entry in cfg["ik_match_table1"].items()
    }
    anchors = {
        bone: [
            (hb, rot_offsets[(hb, r0)], robot_rot[r0].inv().apply(robot_pos[r1] - robot_pos[r0]))
            for hb, r0, r1 in hops
        ]
        for bone, hops in spec.get("anchors", {}).items()
    }
    foot_mod = {k: tuple(v) for k, v in spec.get("foot_mod", {}).items()}
    return HumanScaler(root_scale=ratio(spec["root"]), bone_scales=bone_scales, anchors=anchors, foot_mod=foot_mod)


def flat_weights(scaled: list[dict[str, list[np.ndarray]]], feet: list[str], fps: float) -> list[dict[str, float]]:
    """每帧各脚的放平程度 0..1。地面取 LOCAL_GROUND_S 窗口内两脚踝目标的最低高度（视频提取的人体常有
    整体高度漂移，如蹲下时骨盆高度不变、双脚一起上浮，全局地面会把后半段误判为腾空）。"""
    pos = {f: np.array([fr[f][0] for fr in scaled]) for f in feet}
    win = max(1, int(round(LOCAL_GROUND_S * fps)))
    ground = minimum_filter1d(np.minimum(*(pos[f][:, 2] for f in feet)), win, mode="nearest")
    lo, hi = FLAT_HEIGHT
    flat: list[dict[str, float]] = [{} for _ in scaled]
    for f in feet:
        h = pos[f][:, 2] - ground
        v = np.linalg.norm(np.gradient(pos[f][:, :2], axis=0), axis=1) * fps
        rot = R.from_quat(np.array([fr[f][1] for fr in scaled]), scalar_first=True)
        wz = np.abs((rot[1:] * rot[:-1].inv()).as_rotvec()[:, 2]) * fps
        yaw_rate = np.maximum(np.r_[wz[:1], wz], np.r_[wz, wz[-1:]])
        w = (np.clip((hi - h) / (hi - lo), 0.0, 1.0) * np.clip(2.0 - v / FLAT_SPEED, 0.0, 1.0)
             * np.clip(2.0 - yaw_rate / FLAT_YAW_RATE, 0.0, 1.0))
        for t in np.flatnonzero(w > 0):
            flat[t][f] = float(w[t])
    return flat


def lift_frames(scaled: list[dict[str, list[np.ndarray]]], lift: np.ndarray, knees: list[str], feet: list[str]) -> list[dict]:
    """抬高 lift[t]：脚以外全部 +Δ，膝 +Δ/2，脚不动，让 IK 把腿伸直而不是悬空。"""
    out = []
    for fr, d in zip(scaled, lift):
        if d <= 0:
            out.append(fr)
            continue
        up = {name: [p + [0.0, 0.0, d], q] for name, (p, q) in fr.items()}
        for name in knees:
            up[name][0] = up[name][0] - [0.0, 0.0, d / 2]
        for name in feet:
            up[name][0] = fr[name][0]
        out.append(up)
    return out


def _sole_bottom_z(model: mj.MjModel, data: mj.MjData, sole_ids: list[int]) -> float:
    return float(min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in sole_ids))


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    t_src = np.arange(len(qpos)) / src_fps
    t_dst = np.minimum(np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_fps), t_src[-1])
    out = np.empty((len(t_dst), qpos.shape[1]))
    for j in [0, 1, 2, *range(7, qpos.shape[1])]:
        out[:, j] = np.interp(t_dst, t_src, qpos[:, j])
    rots = R.from_quat(qpos[:, 3:7], scalar_first=True)
    out[:, 3:7] = Slerp(t_src, rots)(t_dst).as_quat(scalar_first=True)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_new_job(job_dir: Path) -> Path:
    retarget_dir = job_dir / "retarget"
    if retarget_dir.exists():
        raise FileExistsError(f"{retarget_dir} 已存在；retarget/ 只写一次，重跑请先删除任务目录")
    return retarget_dir


def retarget_bvh(
    bvh_path: Path,
    job_dir: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    link_input: bool = False,
    flat_feet: bool = False,
    min_pelvis_height: float | None = None,
) -> dict:
    """LAFAN1 BVH → Q1 qpos（源帧率）→ 50 Hz → any4hdmi，写入 job_dir/{input,retarget}。"""
    bvh_path, job_dir = bvh_path.resolve(), job_dir.resolve()
    _check_new_job(job_dir)
    human = load_lafan_motion(bvh_path, start_sec, end_sec)
    source = {"input_type": "bvh_lafan1", "input_fps": human.fps}
    return _retarget(human, "bvh_lafan1", bvh_path, job_dir, start_sec, end_sec, source, link_input, flat_feet, min_pelvis_height)


def retarget_smplx(
    smplx_path: Path,
    job_dir: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model_dir: Path = SMPLX_MODEL_DIR,
    fps: float | None = None,
    link_input: bool = False,
    source_extra: dict | None = None,
    flat_feet: bool = False,
    min_pelvis_height: float | None = None,
) -> dict:
    """SMPL-X（AMASS npz）→ Q1 qpos → 50 Hz → any4hdmi，写入 job_dir/{input,retarget}。
    source_extra 覆盖 / 追加 manifest 的 source 字段（如视频输入的 input_type 与提取信息）。"""
    smplx_path, job_dir = smplx_path.resolve(), job_dir.resolve()
    _check_new_job(job_dir)
    human, info = load_smplx_motion(smplx_path, model_dir.resolve(), start_sec, end_sec, fps=fps)
    source = {"input_type": "smplx_amass", "input_fps": human.fps * info["source_stride"], "smplx": info, **(source_extra or {})}
    return _retarget(human, "smplx", smplx_path, job_dir, start_sec, end_sec, source, link_input, flat_feet, min_pelvis_height)


def _retarget(
    human: HumanMotion, src: str, input_path: Path, job_dir: Path, start_sec: float | None, end_sec: float | None, source: dict,
    link_input: bool = False, flat_feet: bool = False, min_pelvis_height: float | None = None,
) -> dict:
    """link_input：input/ 里放源文件的符号链接而不是副本（批量建库时源数据很大）。
    flat_feet：着地脚朝向放平（见 Q1Retargeting）。深蹲 / 仆步必需；LAFAN 走、跑上跟踪误差持平或略好，
    但优化后滑脚通过率下降约 10 个百分点，所以建库默认不开；min_pelvis_height：骨盆比较低那只脚的脚底至少高多少
    （m，Q1 站立约 0.42），不够的帧抬高目标后整段重解，最多 LIFT_PASSES 轮。"""
    retarget_dir = _check_new_job(job_dir)
    register_q1()
    cfg = json.loads(IK_CONFIGS[src].read_text())
    model = mj.MjModel.from_xml_path(str(Q1_MJCF))
    scaler = build_scaler(cfg, human, model)

    tracked = {}
    for table in ("ik_match_table1", "ik_match_table2"):
        for body, (human_name, pos_w, *_rest) in cfg[table].items():
            if pos_w > 0:
                tracked[body] = human_name
    tracked_ids = {b: model.body(b).id for b in tracked}
    pelvis_id = model.body("pelvis").id
    sole_ids = [g for g in range(model.ngeom) if SOLE_GEOM_PATTERN.match(model.geom(g).name or "")]
    feet = [cfg["ik_match_table1"][f"{s}_ankle_roll_link"][0] for s in SIDES]
    knees = [cfg["ik_match_table1"][f"{s}_knee_link"][0] for s in SIDES]
    scaled_frames = [scaler.scale_frame(frame, human.parents) for frame in human.frames]
    flat = flat_weights(scaled_frames, feet, human.fps) if flat_feet else [{} for _ in scaled_frames]
    keypoint_names = sorted(set(tracked.values()))

    next_lift = np.zeros(len(scaled_frames))
    for _ in range(LIFT_PASSES if min_pelvis_height is not None else 1):
        lift = next_lift
        retargeter = Q1Retargeting(src_human=src, tgt_robot=ROBOT_NAME, actual_human_height=None, verbose=False)
        model, data = retargeter.model, retargeter.configuration.data
        qpos_src, pos_err, sole_z, keypoints, pelvis_h = [], {b: [] for b in tracked}, [], [], []
        for scaled, flat_t in zip(tqdm(lift_frames(scaled_frames, lift, knees, feet), desc="GMR → Q1"), flat):
            retargeter.flat = flat_t
            qpos_src.append(retargeter.retarget(scaled))
            target = retargeter.scaled_human_data
            for body, human_name in tracked.items():
                pos_err[body].append(np.linalg.norm(data.xpos[tracked_ids[body]] - target[human_name][0]))
            sole_z.append(_sole_bottom_z(model, data, sole_ids))
            pelvis_h.append(data.xpos[pelvis_id][2] - sole_z[-1])
            keypoints.append([target[n][0] for n in keypoint_names])
        if min_pelvis_height is None:
            break
        short = min_pelvis_height - np.asarray(pelvis_h)
        if short.max() < LIFT_TOL:
            break
        next_lift = np.maximum(0.0, lift + short)
    qpos_src = np.asarray(qpos_src)
    keypoints = np.asarray(keypoints)

    # 脚底贴地：让 5% 分位的"最低脚底点"落在 z = 0（单帧的穿地 / 悬空留给可行性检查处理）
    ground_shift = -float(np.percentile(sole_z, 5))
    qpos_src[:, 2] += ground_shift
    keypoints[..., 2] += ground_shift
    sole_z = np.asarray(sole_z) + ground_shift

    qpos = resample_qpos(qpos_src, human.fps, OUTPUT_FPS)

    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if link_input:
        (input_dir / input_path.name).symlink_to(input_path)
    else:
        shutil.copy2(input_path, input_dir / input_path.name)

    qpos_names = qpos_names_from_model(model)
    source = {
        **source,
        "input_file": input_path.name,
        "input_sha256": _sha256(input_path),
        "clip_sec": [start_sec, end_sec],
        "retarget": f"GMR + infra/q1infra/retarget/{IK_CONFIGS[src].name}",
        "root_scale": scaler.root_scale,
        "bone_scales": scaler.bone_scales,
        "ground_shift_m": ground_shift,
        "flat_feet": flat_feet,
        "min_pelvis_height_m": min_pelvis_height,
    }
    gmr_dir = job_dir / ".gmr_tmp"
    if gmr_dir.exists():
        shutil.rmtree(gmr_dir)
    (gmr_dir / "motions").mkdir(parents=True)
    np.savez(
        gmr_dir / "motions" / f"{job_dir.name}.npz",
        root_pos=qpos[:, :3],
        root_rot=qpos[:, [4, 5, 6, 3]],
        dof_pos=qpos[:, 7:],
    )
    write_manifest(
        gmr_dir,
        dataset_name=job_dir.name,
        mjcf=Q1_MJCF,
        timestep=1.0 / OUTPUT_FPS,
        qpos_names=qpos_names,
        num_motions=1,
        source=source,
        total_hours=len(qpos) / OUTPUT_FPS / 3600.0,
    )
    convert_dataset(gmr_dir, retarget_dir)
    shutil.rmtree(gmr_dir)

    jnt_lo = model.jnt_range[1:, 0]
    jnt_hi = model.jnt_range[1:, 1]
    margin = np.radians(1.0)
    near_limit = (qpos_src[:, 7:] < jnt_lo + margin) | (qpos_src[:, 7:] > jnt_hi - margin)
    joint_names = qpos_names[7:]
    report = {
        "job": job_dir.name,
        "input_type": source["input_type"],
        "frames_src": len(qpos_src),
        "frames_50hz": len(qpos),
        "duration_s": len(qpos) / OUTPUT_FPS,
        "root_scale": scaler.root_scale,
        "bone_scales": scaler.bone_scales,
        "ground_shift_m": ground_shift,
        "tracking_error_m": {
            b: {"mean": float(np.mean(e)), "max": float(np.max(e))} for b, e in pos_err.items()
        },
        "sole_z_m": {
            "min": float(sole_z.min()),
            "p50": float(np.median(sole_z)),
            "frames_below_-1cm": int((sole_z < -0.01).sum()),
        },
        "near_joint_limit_ratio": {
            n: float(near_limit[:, i].mean()) for i, n in enumerate(joint_names) if near_limit[:, i].any()
        },
        "flat_feet_ratio": {n: float(np.mean([f.get(n, 0.0) > 0.5 for f in flat])) for n in sorted({k for f in flat for k in f})},
        "pelvis_lift_m": {"max": float(lift.max()), "frames_ratio": float((lift > 0).mean())},
        "pelvis_above_sole_min_m": float(np.min(pelvis_h)),
    }
    (job_dir / "retarget_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(
        job_dir / "retarget_debug.npz",
        fps=human.fps,
        keypoint_names=np.array(keypoint_names),
        keypoints=keypoints.astype(np.float32),
    )
    return report
