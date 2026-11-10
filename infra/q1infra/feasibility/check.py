"""Q1 轨迹可行性检查（只报告，不改轨迹）。

逐帧检查（场景见 scene.py，默认平地只允许脚着地）：

- 关节限位：碰撞版模型的限位（厂商区间两端各收 1°），容差 0.1°。
- 穿模：允许接触压入 > 1.5 cm；其余与环境的接触、自碰撞 > 3 mm。
- 不允许的接触：不允许的部位离支撑面或环境 < 场景的 contact_tol（平地 1 cm，如膝、手、躯干着地）；
  1 cm 内但未达到 contact_tol 的单独记为"接近"。
- 触地 / 滑脚 / 悬空：脚离支撑面的距离减去 1 s 内双脚的最小距离（局部地面）< 3 cm、且平面速度 < 0.25 m/s
  记为触地；触地时速度 > 2 cm/s 为滑脚，离支撑面 > 2 cm 为悬空。其他允许部位离支撑面 < 1 cm 记为触地。
- 脚掌倾角：触地脚的脚掌法向与竖直方向夹角 > 10° 为不平（侧翻、踮脚着地，Q1 踝关节撑不住）。
- 平滑：关节速度不超电机上限；关节加速度 > 150 rad/s² 为尖峰。
- 力矩与平衡：逆动力学（M q̈ + 偏置力），触地部位的接触力（摩擦锥 μ = 0.8 内，带正则）承担浮动根的
  6 维广义力；接触解释不了的根力（腾空、质心出支撑面、需要超出摩擦的力）> 体重 10% 为不平衡。关节力矩 = 逆动力学 − 接触力贡献，对照峰值 / 额定力矩；
  首尾各 2 帧（差分边界）不计。
- 不可行时段：不允许的接触或不平衡持续的时间段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from q1infra.feasibility.scene import PROJECT_ROOT, Scene

MOTOR_PRIOR = PROJECT_ROOT / "q1" / "motor_id" / "prior.npz"

FOOT_MARGIN = 0.06
NEAR = 0.01
STANCE_REL = 0.03
STANCE_SPEED = 0.25
HOVER = 0.02
SLIP_SPEED = 0.02
FOOT_TILT = np.radians(10.0)
LOCAL_GROUND_S = 1.0
ACC_SPIKE = 150.0
LIMIT_TOL = np.radians(0.1)
UNSUPPORTED_FRAC = 0.10
FORCE_REG = 0.05
FRICTION = 0.8
EDGE_FRAMES = 2
SEGMENT_GAP_S = 0.5
SEGMENT_MIN_S = 0.2
SIDES = ("left", "right")
GROUND_BASIS = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # 列：法向 z、切向 x、切向 y


@dataclass
class Robot:
    scene: Scene
    model: mj.MjModel
    groups: np.ndarray
    body_names: list[str]
    joint_names: list[str]
    sole_geoms: dict[str, list[int]]
    foot_bodies: dict[str, int]
    tau_max: np.ndarray
    tau_cont: np.ndarray
    omega_max: np.ndarray


def load_robot(scene: Scene) -> Robot:
    m = scene.build_model()
    groups = scene.env_groups(m)
    joint_names = [m.joint(j).name for j in range(1, m.njnt)]
    feet = {s: m.body(f"{s}_ankle_roll_link").id for s in SIDES}
    # 检测余量：让 1 cm 内未压入的接近也生成接触（只做运动学）。平面的接触只认平面一侧的余量，环境也要设。
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_margin[g] = NEAR
    prior = np.load(MOTOR_PRIOR, allow_pickle=True)
    idx = [list(prior["names"]).index(n) for n in joint_names]
    return Robot(
        scene, m, groups, [m.body(b).name for b in range(m.nbody)], joint_names,
        {s: [m.geom(f"{s}_sole_{i}").id for i in range(4)] for s in SIDES}, feet,
        prior["tau_max"][idx], prior["tau_cont"][idx], prior["omega_max"][idx],
    )


def finite_differences(model: mj.MjModel, qpos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """qvel（中心差分，自由关节用 mj_differentiatePos）与平滑后的 qacc。"""
    T, nv = len(qpos), model.nv
    step = np.zeros((T - 1, nv))
    for t in range(T - 1):
        mj.mj_differentiatePos(model, step[t], dt, qpos[t], qpos[t + 1])
    qvel = np.empty((T, nv))
    qvel[1:-1] = 0.5 * (step[:-1] + step[1:])
    qvel[0], qvel[-1] = step[0], step[-1]
    win = min(9, T - (1 - T % 2))
    qacc = savgol_filter(qvel, win, 2, deriv=1, delta=dt, axis=0) if win >= 5 else np.gradient(qvel, dt, axis=0)
    return qvel, qacc


def _sole_bottoms(robot: Robot, data: mj.MjData, side: str) -> np.ndarray:
    m = robot.model
    return np.array([data.geom_xpos[g] - [0, 0, m.geom_size[g][0]] for g in robot.sole_geoms[side]])


def _foot_gap(robot: Robot, data: mj.MjData, side: str, support: list[int]) -> float:
    """脚底四个小球到允许支撑面的最小距离（负为压入），超出 FOOT_MARGIN 记为 FOOT_MARGIN。"""
    m = robot.model
    best = FOOT_MARGIN
    fromto = np.zeros(6)
    for sg in robot.sole_geoms[side]:
        for eg in support:
            if m.geom_type[eg] != mj.mjtGeom.mjGEOM_PLANE:
                gap = np.linalg.norm(data.geom_xpos[sg] - data.geom_xpos[eg]) - m.geom_rbound[sg] - m.geom_rbound[eg]
                if gap > best:
                    continue
            best = min(best, mj.mj_geomDistance(m, data, sg, eg, best, fromto))
    return best


def _contact_basis(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    return np.stack([n, t1, np.cross(n, t1)], axis=1)


def _pyramid(basis: np.ndarray) -> np.ndarray:
    """摩擦锥线性化：法向 ± μ·切向的 4 条棱（3×4），非负组合即满足摩擦约束。"""
    n, t1, t2 = basis[:, 0], basis[:, 1], basis[:, 2]
    edges = [n + FRICTION * t1, n - FRICTION * t1, n + FRICTION * t2, n - FRICTION * t2]
    return np.stack([e / np.linalg.norm(e) for e in edges], axis=1)


def _contact_force_split(robot: Robot, data: mj.MjData, tau_id: np.ndarray, points: list[tuple[np.ndarray, int, np.ndarray]]):
    """接触力（摩擦锥内，带正则）分担浮动根广义力。返回 (关节力矩, 根残差力 N)。"""
    m = robot.model
    if not points:
        return tau_id[6:], float(np.linalg.norm(tau_id[0:3]))
    jacp = np.zeros((3, m.nv))
    n = 4 * len(points)
    Jt = np.zeros((m.nv, n))
    for i, (p, b, basis) in enumerate(points):
        mj.mj_jac(m, data, jacp, None, p, b)
        Jt[:, 4 * i : 4 * i + 4] = jacp.T @ _pyramid(basis)
    A = np.vstack([Jt[:6], FORCE_REG * np.eye(n)])
    rhs = np.concatenate([tau_id[:6], np.zeros(n)])
    lam = lsq_linear(A, rhs, bounds=(0.0, np.inf), method="bvls").x
    resid = tau_id[:6] - Jt[:6] @ lam
    return tau_id[6:] - Jt[6:] @ lam, float(np.linalg.norm(resid[0:3]))


def _segments(mask: np.ndarray, dt: float) -> list[list[float]]:
    segs: list[list[float]] = []
    for t in np.flatnonzero(mask):
        if segs and t * dt - segs[-1][1] <= SEGMENT_GAP_S:
            segs[-1][1] = t * dt
        else:
            segs.append([t * dt, t * dt])
    return [[round(a, 2), round(b + dt, 2)] for a, b in segs if b + dt - a >= SEGMENT_MIN_S]


def check_motion(robot: Robot, qpos: np.ndarray, dt: float) -> tuple[dict, dict]:
    m, sc = robot.model, robot.scene
    data = mj.MjData(m)
    T, nj = len(qpos), len(robot.joint_names)
    qvel, qacc = finite_differences(m, qpos, dt)
    weight = float(m.body_subtreemass[m.body("pelvis").id] * -m.opt.gravity[2])
    is_env = robot.groups != ""
    foot_ids = set(robot.foot_bodies.values())
    foot_support = [g for g in np.flatnonzero(is_env) if sc.allowed("left_ankle_roll_link", robot.groups[g])]

    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    limit_excess = np.maximum(lo - qpos[:, 7:], qpos[:, 7:] - hi) - LIMIT_TOL

    foot_dist = {s: np.full(T, FOOT_MARGIN) for s in SIDES}
    foot_xy = {s: np.zeros((T, 2)) for s in SIDES}
    foot_tilt = {s: np.zeros(T) for s in SIDES}
    other_support: list[list[tuple[np.ndarray, int, np.ndarray]]] = [[] for _ in range(T)]
    support_pen = np.zeros(T)
    env_pen = np.zeros(T)
    env_pen_what: list[tuple[str, str] | None] = [None] * T
    self_pen = np.zeros(T)
    self_pair: list[tuple[str, str] | None] = [None] * T
    disallowed = np.zeros(T, bool)
    disallowed_what: dict[str, int] = {}
    near = np.zeros(T, bool)
    near_what: dict[str, int] = {}

    for t in range(T):
        data.qpos[:] = qpos[t]
        mj.mj_forward(m, data)
        for s in SIDES:
            foot_xy[s][t] = _sole_bottoms(robot, data, s)[:, :2].mean(0)
            foot_dist[s][t] = _foot_gap(robot, data, s, foot_support)
            foot_tilt[s][t] = np.arccos(np.clip(data.xmat[robot.foot_bodies[s]][8], -1.0, 1.0))
        hit_this_frame: set[str] = set()
        near_this_frame: set[str] = set()
        for c in data.contact[: data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if is_env[g1] and is_env[g2]:
                continue
            if is_env[g1] or is_env[g2]:
                ge, gr = (g1, g2) if is_env[g1] else (g2, g1)
                group, body = robot.groups[ge], m.geom_bodyid[gr]
                bname = robot.body_names[body]
                if sc.allowed(bname, group):
                    if body in foot_ids:
                        pass
                    elif c.dist < NEAR:
                        normal = np.array(c.frame[:3]) * (1.0 if ge == g1 else -1.0)  # 由环境指向机器人
                        other_support[t].append((np.array(c.pos), body, _contact_basis(normal)))
                    support_pen[t] = max(support_pen[t], -c.dist)
                else:
                    tag = f"{bname.replace('_link', '')}@{group}"
                    if c.dist < sc.contact_tol:
                        disallowed[t] = True
                        hit_this_frame.add(tag)
                    elif c.dist < NEAR:
                        near[t] = True
                        near_this_frame.add(tag)
                    if -c.dist > env_pen[t]:
                        env_pen[t], env_pen_what[t] = -c.dist, (bname, group)
            elif c.dist < 0 and -c.dist > self_pen[t]:
                self_pen[t] = -c.dist
                self_pair[t] = (robot.body_names[m.geom_bodyid[g1]], robot.body_names[m.geom_bodyid[g2]])
        for k in hit_this_frame:
            disallowed_what[k] = disallowed_what.get(k, 0) + 1
        for k in near_this_frame:
            near_what[k] = near_what.get(k, 0) + 1

    speed = {s: np.linalg.norm(np.gradient(foot_xy[s], dt, axis=0), axis=1) for s in SIDES}
    win = max(1, int(round(LOCAL_GROUND_S / dt)))
    local_ground = np.clip(minimum_filter1d(np.minimum(foot_dist["left"], foot_dist["right"]), win, mode="nearest"), 0.0, None)
    stance = {s: (foot_dist[s] - local_ground < STANCE_REL) & (speed[s] < STANCE_SPEED) for s in SIDES}

    tau = np.zeros((T, nj))
    resid = np.zeros(T)
    tau_id = np.zeros(m.nv)
    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        data.qacc[:] = qacc[t]
        mj.mj_kinematics(m, data)
        mj.mj_comPos(m, data)
        mj.mj_comVel(m, data)
        mj.mj_rne(m, data, 1, tau_id)
        pts = [(p, robot.foot_bodies[s], GROUND_BASIS) for s in SIDES if stance[s][t] for p in _sole_bottoms(robot, data, s)]
        if other_support[t]:
            mj.mj_forward(m, data)
            pts += other_support[t]
        tau[t], resid[t] = _contact_force_split(robot, data, tau_id, pts)
    tau[:EDGE_FRAMES] = 0.0
    tau[-EDGE_FRAMES:] = 0.0

    jvel, jacc = qvel[:, 6:], qacc[:, 6:]
    over_vel = np.abs(jvel) > robot.omega_max
    acc_spike = np.abs(jacc) > ACC_SPIKE
    over_peak = np.abs(tau) > robot.tau_max
    over_rated = np.abs(tau) > robot.tau_cont
    unsupported = resid > UNSUPPORTED_FRAC * weight
    slip_frames = int(sum((stance[s] & (speed[s] > SLIP_SPEED)).sum() for s in SIDES))
    stance_frames = int(sum(stance[s].sum() for s in SIDES))
    stance_speed = np.concatenate([speed[s][stance[s]] for s in SIDES]) if stance_frames else np.zeros(1)
    stance_dist = np.concatenate([foot_dist[s][stance[s]] for s in SIDES]) if stance_frames else np.zeros(1)
    stance_tilt = np.concatenate([foot_tilt[s][stance[s]] for s in SIDES]) if stance_frames else np.zeros(1)
    pen_bad = (support_pen > sc.soft_max) | (env_pen > sc.hard_tol) | (self_pen > sc.hard_tol)

    def worst(mask_2d: np.ndarray, value: np.ndarray) -> dict | None:
        if not mask_2d.any():
            return None
        t, j = np.unravel_index(np.argmax(np.where(mask_2d, value, -np.inf)), value.shape)
        return {"t_s": round(t * dt, 2), "joint": robot.joint_names[j], "value": round(float(value[t, j]), 3)}

    def per_joint(mask_2d: np.ndarray, scale: float = 1.0) -> dict:
        counts = mask_2d.sum(0) * scale
        return {robot.joint_names[j]: round(float(counts[j]), 2) for j in np.argsort(-counts) if counts[j] > 0}

    te, ts = int(np.argmax(env_pen)), int(np.argmax(self_pen))
    report = {
        "scene": sc.name,
        "frames": T,
        "duration_s": round(T * dt, 2),
        "joint_limit": {
            "pass_rate": round(float(1 - (limit_excess > 0).any(1).mean()), 4),
            "frames_per_joint": per_joint(limit_excess > 0),
            "worst_deg": worst(limit_excess > 0, np.degrees(limit_excess + LIMIT_TOL)),
        },
        "penetration": {
            "pass_rate": round(float(1 - pen_bad.mean()), 4),
            "support_overpress_frames": int((support_pen > sc.soft_max).sum()),
            "support_max_depth_mm": round(float(support_pen.max()) * 1e3, 1),
            "env_frames": int((env_pen > sc.hard_tol).sum()),
            "env_worst": {"t_s": round(te * dt, 2), "body": env_pen_what[te][0], "group": env_pen_what[te][1], "depth_mm": round(env_pen[te] * 1e3, 1)} if env_pen[te] > sc.hard_tol else None,
            "self_frames": int((self_pen > sc.hard_tol).sum()),
            "self_worst": {"t_s": round(ts * dt, 2), "bodies": self_pair[ts], "depth_mm": round(self_pen[ts] * 1e3, 1)} if self_pen[ts] > sc.hard_tol else None,
        },
        "contact": {
            "disallowed_frames": int(disallowed.sum()),
            "disallowed_by_part": dict(sorted(disallowed_what.items(), key=lambda kv: -kv[1])[:8]),
            "near_frames": int(near.sum()),
            "near_by_part": dict(sorted(near_what.items(), key=lambda kv: -kv[1])[:8]),
            "stance_ratio": {s: round(float(stance[s].mean()), 3) for s in SIDES},
            "no_foot_support_ratio": round(float((~(stance["left"] | stance["right"])).mean()), 4),
            "root_height_min_m": round(float(qpos[:, 2].min()), 3),
        },
        "foot_slip": {
            "pass_rate": round(1 - slip_frames / max(stance_frames, 1), 4),
            "stance_speed_p50_cm_s": round(float(np.percentile(stance_speed, 50)) * 100, 1),
            "stance_speed_p95_cm_s": round(float(np.percentile(stance_speed, 95)) * 100, 1),
            "hover_rate": round(float(np.mean(stance_dist > HOVER)), 4),
            "stance_gap_p50_cm": round(float(np.percentile(stance_dist, 50)) * 100, 1),
        },
        "foot_tilt": {
            "pass_rate": round(float(1 - (stance_tilt > FOOT_TILT).sum() / max(stance_frames, 1)), 4),
            "stance_tilt_p50_deg": round(float(np.degrees(np.percentile(stance_tilt, 50))), 1),
            "stance_tilt_p95_deg": round(float(np.degrees(np.percentile(stance_tilt, 95))), 1),
        },
        "smoothness": {
            "pass_rate": round(float(1 - (over_vel | acc_spike).any(1).mean()), 4),
            "over_velocity_limit_frames": per_joint(over_vel),
            "acc_spike_frames": per_joint(acc_spike),
            "worst_acc": worst(acc_spike, np.abs(jacc)),
        },
        "torque": {
            "pass_rate": round(float(1 - over_peak.any(1).mean()), 4),
            "peak_nm": {robot.joint_names[j]: round(float(np.abs(tau[:, j]).max()), 1) for j in np.argsort(-np.abs(tau).max(0))[:6]},
            "peak_ratio": {robot.joint_names[j]: round(float(np.abs(tau[:, j]).max() / robot.tau_max[j]), 3) for j in range(nj)},
            "over_peak_frames": per_joint(over_peak),
            "over_rated_s": per_joint(over_rated, dt),
            "worst": worst(over_peak, np.abs(tau)),
        },
        "balance": {
            "pass_rate": round(float(1 - unsupported.mean()), 4),
            "unsupported_frames": int(unsupported.sum()),
            "residual_p95_n": round(float(np.percentile(resid, 95)), 1),
            "body_weight_n": round(weight, 1),
        },
        "infeasible_segments_s": _segments(disallowed | unsupported, dt),
    }
    report["verdict"] = verdict(report)
    annotations = {
        "stance_left": stance["left"], "stance_right": stance["right"],
        "disallowed_contact": disallowed, "unsupported": unsupported,
        "joint_torque": tau.astype(np.float32), "root_residual_n": resid.astype(np.float32),
        "foot_gap_m": np.stack([foot_dist[s] for s in SIDES], 1).astype(np.float32),
        "env_penetration_m": env_pen.astype(np.float32), "self_penetration_m": self_pen.astype(np.float32),
    }
    return report, annotations



def verdict(r: dict) -> str:
    """infeasible：动作超出 Q1 能力，应截掉不可行时段、放慢或换动作——不允许的接触 > 1% 帧，或力矩、平衡
    大面积不满足（通过率 < 80% / 70%；跟踪策略不必逐帧复现参考力矩，力矩通过率 89%、平衡 86% 的走路
    已训练成功）；needs_fix：轨迹优化可修或训练可吸收的问题；pass：可直接训练。"""
    if (
        r["contact"]["disallowed_frames"] > 0.01 * r["frames"]
        or r["torque"]["pass_rate"] < 0.80
        or r["balance"]["pass_rate"] < 0.70
    ):
        return "infeasible"
    checks = ("joint_limit", "penetration", "foot_slip", "foot_tilt", "smoothness", "torque", "balance")
    if min(r[k]["pass_rate"] for k in checks) < 0.99 or r["foot_slip"]["hover_rate"] > 0.2:
        return "needs_fix"
    return "pass"


def load_input(path: Path) -> list[tuple[str, np.ndarray, float, Path]]:
    """any4hdmi 数据集目录，或含 qpos / fps 的 npz。返回 [(名字, qpos, dt, 输出目录)]。"""
    if path.is_dir():
        from any4hdmi.core.format import load_manifest, load_motion

        man = load_manifest(path)
        motions = sorted((man.root / man.payload.get("motions_subdir", "motions")).rglob("*.npz"))
        return [(p.stem, load_motion(p).astype(np.float64), man.timestep, path.parent / "check") for p in motions]
    d = np.load(path, allow_pickle=True)
    return [(path.stem, np.asarray(d["qpos"], np.float64), 1.0 / float(d["fps"]), path.parent / f"{path.stem}_check")]


def check_path(path: Path, scene: Scene) -> dict[str, dict]:
    robot = load_robot(scene)
    reports = {}
    items = load_input(path)
    for name, qpos, dt, out_dir in items:
        if qpos.shape[1] != robot.model.nq:
            raise ValueError(f"{name}: qpos 宽度 {qpos.shape[1]} 与模型 nq {robot.model.nq} 不符")
        report, ann = check_motion(robot, qpos, dt)
        reports[name] = report
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if len(items) == 1 else f"_{name}"
        np.savez_compressed(out_dir / f"annotations{suffix}.npz", **ann)
        (out_dir / f"report{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return reports


def summary_line(name: str, r: dict) -> str:
    def pct(k: str) -> str:
        return f"{r[k]['pass_rate'] * 100:5.1f}%"

    segs = r["infeasible_segments_s"]
    seg_s = sum(b - a for a, b in segs)
    return (
        f"{name:22s} {r['verdict']:10s} | 限位 {pct('joint_limit')} 穿模 {pct('penetration')} 滑脚 {pct('foot_slip')} "
        f"悬空 {r['foot_slip']['hover_rate'] * 100:4.0f}% 脚平 {pct('foot_tilt')} 平滑 {pct('smoothness')} 力矩 {pct('torque')} 平衡 {pct('balance')} | "
        f"不允许接触 {r['contact']['disallowed_frames']} 帧 | 不可行时段 {len(segs)} 段 {seg_s:.1f}s"
    )
