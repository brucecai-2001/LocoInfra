"""整条轨迹联合优化：在尽量贴近原轨迹的前提下修正可行性检查报出的可修问题。

变量：每帧完整 qpos（根 6 + 关节 22，在切空间更新）。残差：

  keep       贴近优化前的轨迹（手改过的关键帧权重 ×3）
  acc        相邻三帧二阶差分（平滑，压掉加速度尖峰）
  vel        关节单帧变化不超过电机转速上限
  pin        支撑脚在一个支撑段内脚底中心水平位置固定（去滑脚）；脚水平走开 > 4 cm 另起一段
  ground     支撑脚最低的脚底点贴到允许的支撑面（去悬空）
             pin / ground 的权重在支撑段首尾 10 帧内线性升降，根部的 acc 权重更高，避免触地 / 离地时跳变
  clear      自碰撞距离 ≥ 0；不允许的部位离环境 ≥ 场景的 contact_tol（至少 2 mm）；允许接触压入 ≤ 1 cm

关节限位作为硬约束（每步夹回）。支撑标记取自可行性检查。Levenberg–Marquardt，解析雅可比
（接触法向 × 点雅可比），稀疏正规方程。

  cd infra && bin/q1py -m q1infra.feasibility.optimize jobs/<任务>/retarget      # 写 jobs/<任务>/check/motion/
  cd infra && bin/q1py -m q1infra.feasibility.optimize traj.npz                  # 写 traj_opt.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from q1infra.feasibility.check import SIDES, Robot, check_motion, load_input, load_robot, summary_line
from q1infra.feasibility.scene import FLAT, Scene

W = dict(keep=1.0, root_keep=3.0, acc=10.0, root_acc=100.0, vel=100.0, clear=300.0, pin=50.0, ground=100.0)
CLEAR_SELF = 0.0
CLEAR_ENV_MIN = 0.002
CLEAR_SUPPORT = -0.01
PIN_SPLIT = 0.04
PIN_MIN = 8
PASS_RATE_DROP = 0.03
KEY_WEIGHT = 3.0
RAMP = 10  # 支撑段首尾的帧数内 pin / ground 权重线性升降，避免触地 / 离地瞬间的跳变


class TrajOpt:
    def __init__(self, robot: Robot, qpos: np.ndarray, dt: float, stance: dict[str, np.ndarray], w: dict | None = None, log=print,
                 key: np.ndarray | None = None):
        self.robot, self.m = robot, robot.model
        self.nv = self.m.nv
        self.Q0 = np.asarray(qpos, float)
        self.N = len(self.Q0)
        self.dt = dt
        self.w = {**W, **(w or {})}
        self.log = log
        self.d = mj.MjData(self.m)
        self.is_env = robot.groups != ""
        self.foot_ids = {robot.foot_bodies[s]: s for s in SIDES}
        self.foot_support = [g for g in np.flatnonzero(self.is_env) if robot.scene.allowed("left_ankle_roll_link", robot.groups[g])]
        self.lo, self.hi = self.m.jnt_range[1:, 0], self.m.jnt_range[1:, 1]
        self.clear_env = max(robot.scene.contact_tol, CLEAR_ENV_MIN)
        self.vmax = robot.omega_max * dt
        wk = np.full(self.nv, self.w["keep"])
        wk[:6] = self.w["root_keep"]
        frame_w = np.where(np.asarray(key, bool), KEY_WEIGHT, 1.0) if key is not None else np.ones(self.N)
        self.wk = (frame_w[:, None] * wk[None, :]).ravel()
        self.J_keep = sp.diags(self.wk)
        D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(self.N - 2, self.N))
        wa = np.full(self.nv, self.w["acc"])
        wa[:6] = self.w["root_acc"]
        self.wa = np.tile(wa, self.N - 2)
        self.J_acc = (sp.diags(self.wa) @ sp.kron(D2, sp.eye(self.nv))).tocsr()
        self.stance = stance
        self.ramp = {s: self._ramp(stance[s]) for s in SIDES}
        self.pin, self.pin_targets = self._pins(self.Q0)

    # ---------- 工具 ----------
    def qdiff(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        v = np.zeros(self.nv)
        mj.mj_differentiatePos(self.m, v, 1.0, q0, q1)
        return v

    def qinteg(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, float)
        mj.mj_integratePos(self.m, q, v, 1.0)
        q[7:] = np.clip(q[7:], self.lo, self.hi)
        return q

    def _fwd(self, q: np.ndarray) -> mj.MjData:
        self.d.qpos[:] = q
        mj.mj_forward(self.m, self.d)
        return self.d

    def _sole_center(self, d: mj.MjData, s: str) -> np.ndarray:
        return np.mean([d.geom_xpos[g] for g in self.robot.sole_geoms[s]], axis=0)

    def _lowest_sole(self, d: mj.MjData, s: str) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        """最低脚底点到允许支撑面的距离、脚底上的最近点、以及距离增大的方向（支撑面 → 脚）。"""
        m = self.m
        best, point, normal = np.inf, None, None
        fromto = np.zeros(6)
        for sg in self.robot.sole_geoms[s]:
            for eg in self.foot_support:
                if m.geom_type[eg] != mj.mjtGeom.mjGEOM_PLANE:
                    rough = np.linalg.norm(d.geom_xpos[sg] - d.geom_xpos[eg]) - m.geom_rbound[sg] - m.geom_rbound[eg]
                    if rough > min(best, 0.1):
                        continue
                dist = mj.mj_geomDistance(m, d, sg, eg, 0.1, fromto)
                if dist < best:
                    v = fromto[:3] - fromto[3:]
                    nv_ = np.linalg.norm(v)
                    if nv_ < 1e-9:
                        n = np.array([0.0, 0.0, 1.0])
                    else:
                        n = v / nv_ if dist >= 0 else -v / nv_
                    best, point, normal = dist, fromto[:3].copy(), n
        return best, point, normal

    @staticmethod
    def _ramp(on: np.ndarray) -> np.ndarray:
        """支撑段内的权重：离段首 / 段尾 k 帧处为 (k + 1) / RAMP，段中为 1，非支撑为 0。"""
        n = len(on)
        w = np.zeros(n)
        t = 0
        while t < n:
            if not on[t]:
                t += 1
                continue
            t1 = t
            while t1 + 1 < n and on[t1 + 1]:
                t1 += 1
            k = np.arange(t1 - t + 1)
            w[t : t1 + 1] = np.minimum(1.0, (np.minimum(k, k[::-1]) + 1) / RAMP)
            t = t1 + 1
        return w

    def _pins(self, Q: np.ndarray):
        pos = {s: np.zeros((self.N, 3)) for s in SIDES}
        for t in range(self.N):
            d = self._fwd(Q[t])
            for s in SIDES:
                pos[s][t] = self._sole_center(d, s)
        pin = {s: np.full(self.N, -1) for s in SIDES}
        targets = []
        for s in SIDES:
            on = self.stance[s]
            t = 0
            while t < self.N:
                if not on[t]:
                    t += 1
                    continue
                t1 = t
                while t1 + 1 < self.N and on[t1 + 1] and np.linalg.norm(pos[s][t1 + 1][:2] - pos[s][t][:2]) < PIN_SPLIT:
                    t1 += 1
                if t1 - t + 1 >= PIN_MIN:
                    pin[s][t : t1 + 1] = len(targets)
                    targets.append(np.median(pos[s][t : t1 + 1, :2], axis=0))
                t = t1 + 1
        return pin, np.array(targets) if targets else np.zeros((0, 2))

    def _kind(self, c) -> tuple[str, int, int] | None:
        g1, g2 = c.geom1, c.geom2
        if self.is_env[g1] and self.is_env[g2]:
            return None
        if self.is_env[g1] or self.is_env[g2]:
            ge, gr = (g1, g2) if self.is_env[g1] else (g2, g1)
            body = self.m.geom_bodyid[gr]
            ok = self.robot.scene.allowed(self.robot.body_names[body], self.robot.groups[ge])
            return ("support" if ok else "env"), ge, gr
        return "self", g1, g2

    # ---------- 残差 / 雅可比 ----------
    def _eval(self, Q: np.ndarray, need_J: bool):
        N, nv, m = self.N, self.nv, self.m
        r_keep = self.wk * np.concatenate([self.qdiff(self.Q0[t], Q[t]) for t in range(N)])
        Dq = np.array([self.qdiff(Q[t], Q[t + 1]) for t in range(N - 1)])
        r_acc = self.wa * (Dq[1:] - Dq[:-1]).ravel()
        r_x: list[float] = []
        rows, cols, vals = [], [], []
        jp = np.zeros((3, nv))
        ar = np.arange(nv)

        def add_row(value: float, t: int, grad: np.ndarray | None):
            k = len(r_x)
            r_x.append(value)
            if need_J and grad is not None:
                rows.append(np.full(nv, k))
                cols.append(t * nv + ar)
                vals.append(grad)

        over = np.abs(Dq[:, 6:]) - self.vmax
        for t, j in zip(*np.nonzero(over > 0)):
            k = len(r_x)
            r_x.append(self.w["vel"] * over[t, j])
            if need_J:
                sgn = np.sign(Dq[t, 6 + j])
                rows.append(np.array([k, k]))
                cols.append(np.array([t * nv + 6 + j, (t + 1) * nv + 6 + j]))
                vals.append(self.w["vel"] * sgn * np.array([-1.0, 1.0]))

        st = dict(clear_frames=0, worst_mm=0.0, slide_cm=0.0, hover_cm=0.0)
        wc, wp, wg = self.w["clear"], self.w["pin"], self.w["ground"]
        for t in range(N):
            d = self._fwd(Q[t])
            bad = False
            for c in d.contact[: d.ncon]:
                kind = self._kind(c)
                if kind is None:
                    continue
                kname = kind[0]
                if kname == "support" and m.geom_bodyid[kind[2]] in self.foot_ids:
                    target = CLEAR_SUPPORT
                else:
                    target = {"self": CLEAR_SELF, "env": self.clear_env, "support": CLEAR_SUPPORT}[kname]
                if c.dist >= target:
                    continue
                bad = True
                st["worst_mm"] = max(st["worst_mm"], (target - c.dist) * 1e3)
                grad = None
                if need_J:
                    n = np.array(c.frame[:3])
                    grad = np.zeros(nv)
                    for gg, sign in ((c.geom1, -1.0), (c.geom2, 1.0)):
                        if self.is_env[gg]:
                            continue
                        mj.mj_jac(m, d, jp, None, c.pos, m.geom_bodyid[gg])
                        grad += sign * (n @ jp)
                    grad = -wc * grad
                add_row(wc * (target - c.dist), t, grad)
            st["clear_frames"] += bad
            for s in SIDES:
                ramp = self.ramp[s][t]
                if ramp <= 0:
                    continue
                body = self.robot.foot_bodies[s]
                gap, point, normal = self._lowest_sole(d, s)
                if point is not None and np.isfinite(gap):
                    st["hover_cm"] = max(st["hover_cm"], gap * 100)
                    grad = None
                    if need_J:
                        mj.mj_jac(m, d, jp, None, point, body)
                        grad = ramp * wg * (normal @ jp)
                    add_row(ramp * wg * gap, t, grad)
                pi = self.pin[s][t]
                if pi >= 0:
                    p = self._sole_center(d, s)
                    e = p[:2] - self.pin_targets[pi]
                    st["slide_cm"] = max(st["slide_cm"], float(np.linalg.norm(e)) * 100)
                    if need_J:
                        mj.mj_jac(m, d, jp, None, p, body)
                    for a in range(2):
                        add_row(ramp * wp * e[a], t, ramp * wp * jp[a] if need_J else None)
        r = np.concatenate([r_keep, r_acc, np.array(r_x)])
        if not need_J:
            return r, None, st
        Jx = sp.csr_matrix(
            (np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [], np.concatenate(cols) if cols else [])),
            shape=(len(r_x), N * nv),
        )
        return r, sp.vstack([self.J_keep, self.J_acc, Jx]).tocsc(), st

    # ---------- 求解 ----------
    def run(self, iters: int = 25, tol: float = 1e-3) -> np.ndarray:
        Q = self.Q0.copy()
        lam = 1e-3
        t0 = time.time()
        r, J, st = self._eval(Q, True)
        cost = float(r @ r)
        self.log(f"[opt] {self.N} 帧，支撑段 {len(self.pin_targets)} 个 | 开始：碰撞 {st['clear_frames']} 帧 最深 {st['worst_mm']:.0f} mm，"
                 f"滑移 {st['slide_cm']:.1f} cm，悬空 {st['hover_cm']:.1f} cm")
        I = sp.identity(self.N * self.nv, format="csc")
        for it in range(iters):
            A = (J.T @ J).tocsc()
            g = J.T @ r
            while True:
                delta = -spsolve(A + lam * I, g).reshape(self.N, self.nv)
                Qn = np.array([self.qinteg(Q[t], delta[t]) for t in range(self.N)])
                rn, _, _ = self._eval(Qn, False)
                cn = float(rn @ rn)
                if cn < cost:
                    lam = max(lam * 0.3, 1e-6)
                    break
                lam *= 5.0
                if lam > 1e4:
                    break
            if lam > 1e4:
                self.log("[opt] 步长收不下去，停止")
                break
            rel = (cost - cn) / max(cost, 1e-12)
            Q, cost = Qn, cn
            r, J, st = self._eval(Q, True)
            self.log(f"[opt] 迭代 {it + 1}: cost {cost:.4g} | 碰撞 {st['clear_frames']} 帧 最深 {st['worst_mm']:.0f} mm，"
                     f"滑移 {st['slide_cm']:.1f} cm，悬空 {st['hover_cm']:.1f} cm ({time.time() - t0:.0f} s)")
            if rel < tol:
                break
        self.stats = st
        return Q


def dynamics_regressions(before: dict, after: dict) -> list[str]:
    """优化不含动力学项，可能让力矩 / 平衡变差。判为变差：力矩或平衡通过率下降 > 3 个百分点
    （训练对照中下降 1–2 个百分点不影响可训性），或某关节峰值超出上限且比优化前高 10% 以上。"""
    out = []
    for k in ("torque", "balance"):
        if after[k]["pass_rate"] < before[k]["pass_rate"] - PASS_RATE_DROP:
            out.append(f"{k} 通过率 {before[k]['pass_rate']:.3f} → {after[k]['pass_rate']:.3f}")
    rb, ra = before["torque"]["peak_ratio"], after["torque"]["peak_ratio"]
    for j, v in ra.items():
        if v > 1.0 and v > 1.1 * rb.get(j, 0.0):
            out.append(f"{j} 力矩峰值 {rb.get(j, 0.0) * 100:.0f}% → {v * 100:.0f}% 上限")
    return out


def optimize_path(path: Path, scene: Scene, iters: int, log=print) -> dict:
    robot = load_robot(scene)
    results = {}
    for name, qpos, dt, out_dir in load_input(path):
        before, ann = check_motion(robot, qpos, dt)
        stance = {"left": ann["stance_left"], "right": ann["stance_right"]}
        q_opt = TrajOpt(robot, qpos, dt, stance, log=log).run(iters)
        after, _ = check_motion(robot, q_opt, dt)
        regress = dynamics_regressions(before, after)
        log("前 " + summary_line(name, before))
        log("后 " + summary_line(name, after))
        if regress:
            log("[opt] rejected：动力学变差（" + "；".join(regress) + "），建议继续用原轨迹")
        out = _write(path, name, q_opt, dt, out_dir, before, after, regress)
        results[name] = {"before": before, "after": after, "rejected": bool(regress), "output": str(out)}
    return results


def _write(path: Path, name: str, q_opt: np.ndarray, dt: float, out_dir: Path, before: dict, after: dict, regress: list[str]) -> Path:
    summary = {"accepted": not regress, "dynamics_regressions": regress, "before": before, "after": after}
    if path.is_dir():
        from any4hdmi.core.format import load_manifest, save_motion, write_manifest

        man = load_manifest(path)
        ds = out_dir / "motion"
        if ds.exists():
            shutil.rmtree(ds)
        save_motion(ds / man.payload.get("motions_subdir", "motions") / f"{name}.npz", q_opt)
        source = dict(man.payload.get("source", {}))
        source["trajopt"] = {"from": str(path), "verdict_before": before["verdict"], "verdict_after": after["verdict"], "accepted": not regress}
        write_manifest(
            ds, dataset_name=man.dataset_name, mjcf=man.mjcf, timestep=man.timestep,
            qpos_names=list(man.payload["qpos_names"]), num_motions=1, source=source,
            total_hours=len(q_opt) * man.timestep / 3600.0,
        )
        (out_dir / "optimize_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return ds
    out = path.with_name(f"{path.stem}_opt.npz")
    extra = {k: v for k, v in np.load(path, allow_pickle=True).items() if k not in ("qpos", "fps")}
    np.savez(out, qpos=q_opt, fps=1.0 / dt, **extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "optimize_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1 轨迹优化（修正可行性检查的可修问题）")
    ap.add_argument("inputs", type=Path, nargs="+", help="any4hdmi 数据集目录，或含 qpos / fps 的 npz")
    ap.add_argument("--iters", type=int, default=25)
    a = ap.parse_args()
    for p in a.inputs:
        for name, res in optimize_path(p, FLAT, a.iters).items():
            print(f"{name} -> {res['output']}{'（rejected）' if res['rejected'] else ''}", flush=True)


if __name__ == "__main__":
    main()
