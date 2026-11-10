#!/usr/bin/env python3
"""
电机辨识离线拟合: 读真机被动录数的 npz(字段见 synth_demo: names, t, q, qd, tau, 可选 motor_temp 等), 对每个关节估计
  - 力矩–转速包络(分速度箱的 |τ| 95/99 分位)
  - 库仑 + 粘滞摩擦 τ ≈ c·sign(qd) + b·qd  (低力矩段)
  - 峰值 / "持续"力矩(全序列 99 分位 vs 滑动 1 s 均值的 95 分位)
  - 温度–力矩关系(有升温数据时)
并把结果写成 motors.npz, 供训练侧 MOTORS / 域随机化读取。

不需要 ROS2, 本机可跑。合成自测: python3 fit.py --synth

用法:
  python3 fit.py data/passive_*.npz [--out data/motors.npz] [--plot data/envelope.png]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from joints import CRITICAL, effort_guess, short  # noqa: E402


def load_recordings(paths):
    """合并多个 npz; 要求关节名集合一致。返回 dict(names, t, q, qd, tau, ...拼接后的)"""
    chunks = [dict(np.load(p, allow_pickle=False)) for p in paths]
    names = [str(n) for n in chunks[0]["names"]]
    for c, p in zip(chunks[1:], paths[1:]):
        assert list(map(str, c["names"])) == names, f"关节名不一致: {p}"
    out = {"names": names}
    for k in ("q", "qd", "tau", "coil_temp", "motor_temp", "motor_vol"):
        if k in chunks[0]:
            out[k] = np.concatenate([c[k] for c in chunks], 0)
    # 时间戳拼接: 各段接在上一段末尾
    ts = []
    offset = 0.0
    for c in chunks:
        ts.append(c["t"] + offset)
        offset = float(ts[-1][-1]) + 1.0 / float(c.get("hz", 500))
    out["t"] = np.concatenate(ts)
    out["hz"] = float(np.median(1.0 / np.diff(out["t"]))) if len(out["t"]) > 1 else 0.0
    return out


def envelope(qd, tau, n_bins=12, abs_qd_max=None):
    """分速度箱的 |τ| 分位。返回 bins_edge (B+1,), p95 (B,), p99 (B,), count (B,)"""
    aqd, atau = np.abs(qd), np.abs(tau)
    vmax = abs_qd_max or max(float(np.percentile(aqd, 99)), 1e-3)
    edges = np.linspace(0, vmax, n_bins + 1)
    p95 = np.full(n_bins, np.nan)
    p99 = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, int)
    for i in range(n_bins):
        m = (aqd >= edges[i]) & (aqd < edges[i + 1])
        cnt[i] = int(m.sum())
        if cnt[i] >= 20:
            p95[i] = np.percentile(atau[m], 95)
            p99[i] = np.percentile(atau[m], 99)
    return edges, p95, p99, cnt


def friction(qd, tau, tau_cap_frac=0.3, effort_ref=33.0):
    """低力矩段线性回归 τ ≈ c·sign(qd) + b·qd。返回 c, b, r2, n"""
    m = (np.abs(tau) < tau_cap_frac * effort_ref) & (np.abs(qd) > 0.05)
    if m.sum() < 50:
        return np.nan, np.nan, np.nan, int(m.sum())
    x = np.column_stack([np.sign(qd[m]), qd[m]])
    y = tau[m]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return float(coef[0]), float(coef[1]), float(1 - ss_res / ss_tot), int(m.sum())


def sustained(tau, hz, window_s=1.0):
    """滑动 window_s 均值的 95 分位 ≈ 持续力矩下界(被动数据里能见到的)"""
    w = max(1, int(round(window_s * hz)))
    if len(tau) < w:
        return float(np.percentile(np.abs(tau), 95))
    ker = np.ones(w) / w
    sm = np.convolve(np.abs(tau), ker, mode="valid")
    return float(np.percentile(sm, 95))


def fit_one(name, qd, tau, hz, motor_temp=None):
    ref = effort_guess(name, "scheme")
    edges, p95, p99, cnt = envelope(qd, tau)
    c, b, r2, n_fric = friction(qd, tau, effort_ref=ref)
    peak = float(np.percentile(np.abs(tau), 99))
    peak_max = float(np.abs(tau).max())
    sust = sustained(tau, hz)
    out = dict(
        name=name, effort_ref=ref,
        peak_p99=peak, peak_max=peak_max, sustained_1s_p95=sust,
        friction_c=c, friction_b=b, friction_r2=r2, friction_n=n_fric,
        env_edges=edges, env_p95=p95, env_p99=p99, env_count=cnt,
        qd_p99=float(np.percentile(np.abs(qd), 99)),
        qd_max=float(np.abs(qd).max()),
        tau_rms=float(np.sqrt(np.mean(tau ** 2))),
    )
    if motor_temp is not None and np.nanmax(motor_temp) - np.nanmin(motor_temp) > 3:
        # 粗看: 温度升幅 vs |τ| 滑动均值的相关
        out["temp_rise"] = float(np.nanmax(motor_temp) - np.nanmin(motor_temp))
        out["temp_max"] = float(np.nanmax(motor_temp))
    return out


def report(fits, paths):
    print(f"录数: {paths}")
    print(f"{'关节':<28s} {'峰值p99':>8s} {'峰值max':>8s} {'持续1s':>8s} {'模型上限':>8s} {'占上限':>6s}  "
          f"{'|qd|p99':>7s}  摩擦 c/b (R²)")
    for f in fits:
        ratio = f["peak_p99"] / f["effort_ref"] if f["effort_ref"] else np.nan
        mark = " ◀" if f["name"] in CRITICAL else ""
        fr = (f"{f['friction_c']:+.2f}/{f['friction_b']:+.3f} ({f['friction_r2']:.2f})"
              if np.isfinite(f["friction_c"]) else "—")
        print(f"{short(f['name']):<28s} {f['peak_p99']:7.1f}  {f['peak_max']:7.1f}  "
              f"{f['sustained_1s_p95']:7.1f}  {f['effort_ref']:7.1f}  {ratio*100:5.0f}%  "
              f"{f['qd_p99']:6.2f}  {fr}{mark}")
    print("\n◀ = 关键关节。峰值接近/超过模型上限 → 模型里的力矩上限可能偏保守或偏乐观, 看占上限列。")
    print("持续1s ≪ 峰值 → 录到的动作里尖峰短, 看不出持续额定; 持续≈峰值 → 接近持续额定。")


def save_motors(fits, path, meta):
    names = [f["name"] for f in fits]
    # 训练侧直接用的标量
    peak = np.array([f["peak_p99"] for f in fits])
    sust = np.array([f["sustained_1s_p95"] for f in fits])
    fric_c = np.array([f["friction_c"] for f in fits])
    fric_b = np.array([f["friction_b"] for f in fits])
    # 力矩–转速包络: 统一到同一速度网格(取各关节 vmax 的最大)
    vmax = max(f["env_edges"][-1] for f in fits)
    grid = np.linspace(0, vmax, 13)
    env = np.full((len(fits), len(grid) - 1), np.nan)
    for i, f in enumerate(fits):
        # 把各关节自己的箱映射到统一网格(最近箱)
        mid_src = 0.5 * (f["env_edges"][:-1] + f["env_edges"][1:])
        mid_dst = 0.5 * (grid[:-1] + grid[1:])
        for j, m in enumerate(mid_dst):
            k = int(np.argmin(np.abs(mid_src - m)))
            if f["env_count"][k] >= 20:
                env[i, j] = f["env_p95"][k]
    np.savez_compressed(
        path, names=np.array(names),
        peak_p99=peak, sustained_1s_p95=sust,
        friction_c=fric_c, friction_b=fric_b,
        omega_edges=grid, torque_envelope_p95=env,
        effort_scheme=np.array([effort_guess(n, "scheme") for n in names]),
        effort_mjcf=np.array([effort_guess(n, "mjcf") for n in names]),
        **{f"meta_{k}": np.array(v) for k, v in meta.items()},
    )
    print(f"saved {path}")


def plot_envelope(fits, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未装, 跳过作图")
        return
    crit = [f for f in fits if f["name"] in CRITICAL]
    if not crit:
        crit = fits[:6]
    cols = 3
    rows = int(np.ceil(len(crit) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, f in zip(axes.flat, crit):
        mid = 0.5 * (f["env_edges"][:-1] + f["env_edges"][1:])
        m = np.isfinite(f["env_p95"])
        ax.plot(mid[m], f["env_p95"][m], "o-", label="|τ| p95")
        ax.plot(mid[m], f["env_p99"][m], "s--", label="|τ| p99", alpha=0.7)
        ax.axhline(f["effort_ref"], color="r", ls=":", label=f"scheme limit {f['effort_ref']:.0f}")
        ax.set_title(short(f["name"]), fontsize=9)
        ax.set_xlabel("|qd| (rad/s)")
        ax.set_ylabel("|τ| (N·m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(crit):]:
        ax.axis("off")
    fig.suptitle("Torque–speed envelope from passive recording")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def synth_demo(path):
    """合成一段假数据, 验证拟合管线(无真机时用)"""
    rng = np.random.default_rng(0)
    names = CRITICAL
    hz, T = 500.0, 30.0
    n = int(hz * T)
    t = np.arange(n) / hz
    q = np.zeros((n, len(names)))
    qd = np.zeros_like(q)
    tau = np.zeros_like(q)
    for j, name in enumerate(names):
        lim = effort_guess(name, "scheme")
        # 正弦运动 + 摩擦 + 偶尔贴近上限的尖峰
        w = 1.5 + 0.3 * j
        qd[:, j] = 2.0 * np.sin(w * t) + 0.3 * rng.normal(size=n)
        q[:, j] = np.cumsum(qd[:, j]) / hz
        tau[:, j] = (0.4 * lim * np.sin(w * t + 0.3)
                     + 1.5 * np.sign(qd[:, j]) + 0.3 * qd[:, j]
                     + 0.5 * rng.normal(size=n))
        # 每 5 s 一个尖峰到 0.95*lim
        for k in range(0, n, int(5 * hz)):
            tau[k:k + 10, j] = 0.95 * lim * np.sign(qd[k, j] + 1e-6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, names=np.array(names), t=t, q=q, qd=qd, tau=tau,
        coil_temp=np.full_like(q, 35), motor_temp=np.full_like(q, 40),
        motor_vol=np.full_like(q, 48), hz=np.array(hz),
        source=np.array("synth"),
    )
    print(f"synth -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="被动录数 npz, 可多个/通配")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "motors.npz"))
    ap.add_argument("--plot", default=os.path.join(HERE, "data", "envelope.png"))
    ap.add_argument("--synth", action="store_true", help="生成合成数据自测拟合管线")
    args = ap.parse_args()

    if args.synth:
        p = os.path.join(HERE, "data", "synth_passive.npz")
        synth_demo(p)
        paths = [p]
    else:
        paths = []
        for pat in args.inputs:
            paths.extend(sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else []))
        if not paths:
            raise SystemExit("请给出至少一个 npz, 或用 --synth 自测")

    data = load_recordings(paths)
    print(f"合计 {len(data['t'])} 帧 @ ~{data['hz']:.0f} Hz, 时长 {data['t'][-1]:.1f}s, "
          f"{len(data['names'])} 关节")
    fits = []
    for j, name in enumerate(data["names"]):
        mt = data["motor_temp"][:, j] if "motor_temp" in data else None
        fits.append(fit_one(name, data["qd"][:, j], data["tau"][:, j], data["hz"], mt))
    # 关键关节排前面打印
    fits_sorted = sorted(fits, key=lambda f: (0 if f["name"] in CRITICAL else 1, f["name"]))
    report(fits_sorted, paths)
    save_motors(fits, args.out, meta={"hz": data["hz"], "duration_s": float(data["t"][-1]),
                                       "n_frames": len(data["t"]), "sources": ",".join(paths)})
    if args.plot:
        plot_envelope(fits_sorted, args.plot)


if __name__ == "__main__":
    main()
