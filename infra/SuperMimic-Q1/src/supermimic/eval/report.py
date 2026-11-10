"""``report.html`` (self-contained except for the replay videos) and the playback renderer."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

from supermimic.robot import SPEC


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'


def _clip_figures(c: dict, limit: dict, rated: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = c["traces"]
    t = np.arange(len(tr["reward"])) * 0.02
    joints = list(SPEC.joint_names)
    short = [j.replace("_joint", "").replace("left_", "L_").replace("right_", "R_") for j in joints]
    figs = []

    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, tr["reward"], lw=0.8)
    ax[0].set_ylabel("step reward / max")
    ax[0].set_ylim(min(0.0, min(tr["reward"])), 1.05)
    ax[1].plot(t, tr["joint_err"], lw=0.8, color="tab:orange")
    ax[1].axhline(0.1, ls="--", color="gray", lw=0.8)
    ax[1].set_ylabel("mean |q - q_ref| (rad)")
    ax[2].plot(t, tr["root_err"], lw=0.8, color="tab:green")
    ax[2].set_ylabel("root position error (m)")
    ax[2].set_xlabel("time (s)")
    fig.suptitle(f"{c['name']}: per-step curves")
    figs.append(_png(fig))

    E = np.asarray(tr["joint_err_frames"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    im = ax[0].imshow(E.T, aspect="auto", origin="lower", extent=[0, t[-1] if len(t) else 1, -0.5, len(joints) - 0.5], cmap="magma", vmin=0, vmax=max(0.2, float(np.percentile(E, 99))))
    ax[0].set_yticks(range(len(joints)), short, fontsize=6)
    ax[0].set_xlabel("time (s)")
    ax[0].set_title("per-frame joint error (rad)")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    ax[1].barh(range(len(joints)), E.mean(0), color="tab:orange")
    ax[1].set_yticks(range(len(joints)), short, fontsize=6)
    ax[1].axvline(0.1, ls="--", color="gray", lw=0.8)
    ax[1].set_title("mean (rad)")
    figs.append(_png(fig))

    tau = np.asarray(tr["torque"])
    load = np.abs(tau).max(0) / np.array([limit[j] for j in joints])
    top = np.argsort(-load)[:6]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for a, j in zip(axs.flat, top):
        a.plot(t, tau[:, j], lw=0.6)
        for s in (1, -1):
            a.axhline(s * rated[joints[j]], ls="--", color="orange", lw=0.7)
            a.axhline(s * limit[joints[j]], ls="--", color="red", lw=0.7)
        a.set_title(short[j], fontsize=8)
    fig.suptitle("joint torque (N·m) of the most loaded joints — orange: rated, red: peak limit")
    figs.append(_png(fig))
    return figs


def _table(rows: list[list], head: list[str]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def _mark(p) -> str:
    return '<span class="na">—</span>' if p is None else ('<span class="ok">通过</span>' if p else '<span class="bad">未通过</span>')


def write_html(res: dict, out: Path, title: str, meta: dict, videos: dict[str, str] | None = None) -> None:
    det = res["deterministic"]
    parts = [
        f"<h1>{html.escape(title)}</h1>",
        "<p>" + " · ".join(f"{html.escape(k)}: <code>{html.escape(str(v))}</code>" for k, v in meta.items()) + "</p>",
        f"<h2>结论：{_mark(res['passed'])}</h2>",
        _table([[html.escape(c["name"]), html.escape(c["value"]), html.escape(c["threshold"]), _mark(c["passed"])] for c in res["checks"]], ["指标", "结果", "验收线", ""]),
    ]
    for c in det["clips"]:
        parts.append(f"<h2>整条确定性播放：{html.escape(c['name'])}</h2>")
        parts.append(
            f"<p>{c['steps']}/{c['frames']} 帧，{'<b class=bad>摔倒</b>' if c['fell'] else '未摔倒'}；归一化回报 {c['norm_return']:.3f}；"
            f"关节误差 {c['joint_err_mean']:.3f} rad；根位置误差 {c['root_err_mean']:.3f} m；PD 力矩需求峰值 / 上限 {c['torque_peak_ratio']:.2f}（超限 {c['torque_saturated_frac']:.2%} 关节·步）；"
            f"超额定累计 {c['over_rated_total_s']:.1f} s" + (f"（可行性估算 {c['feasibility_over_rated_total_s']:.1f} s）" if "feasibility_over_rated_total_s" in c else "") + "</p>"
        )
        if videos and c["name"] in videos:
            parts.append(f'<video src="{html.escape(videos[c["name"]])}" controls width="960"></video>')
        parts += _clip_figures(c, det["torque_limit"], det["torque_rated"])
        terms = sorted(c["terms"].items())
        parts.append("<details><summary>每步奖励分项</summary>" + _table([[k, f"{v:.5f}"] for k, v in terms], ["项", "均值 / 步"]) + "</details>")
        over = c["over_rated_s"]
        feas = c.get("feasibility_over_rated_s", {})
        if over or feas:
            names = sorted(set(over) | set(feas))
            parts.append("<details><summary>超额定时间（s）</summary>" + _table([[n, over.get(n, 0.0), feas.get(n, 0.0)] for n in names], ["关节", "策略", "可行性估算"]) + "</details>")
    dr = res["randomized"]
    parts.append("<h2>全量随机化</h2>")
    parts.append(f"<p>{dr['episodes']} 个单回合环境，随机起点、每回合 ≤ 10 s；质量 / PD / 力矩上限 / armature / 关节摩擦 / 脚摩擦 / 零偏 / 质心 / 动作与观测延迟、传感器噪声、初始扰动全开（不加动作噪声）。"
                 f"摔倒 {dr['falls']}（{dr['fall_rate']:.2%}），平均回报 {dr['mean_norm_return']:.3f}，平均回合 {dr['mean_episode_s']:.1f} s。</p>")
    ps = res["push"]
    parts.append("<h2>推力</h2>")
    parts.append(f"<p>在参考根速度 &lt; {ps['slow_speed_mps']} m/s 的帧（{ps['slow_frames']} 帧）随机选点，前 1 s 正常跟踪，然后给骨盆加一个水平方向随机的速度，{ps['recover_s']:.0f} s 内不触发终止算恢复。</p>")
    parts.append(_table([[k, v["pushes"], v["falls"], f"{v['recovered_rate']:.0%}"] for k, v in ps["by_speed"].items()], ["速度 (m/s)", "次数", "摔倒", "恢复率"]))
    parts.append("<h2>换求解器参数</h2>")
    parts.append(_table([[k, html.escape(str(v["settings"])), f"{v['mean_norm_return']:.3f}", v["failed"], f"{v['drop']:+.1%}"] for k, v in res["solver"].items()], ["设置", "参数", "回报", "摔倒", "相对确定性下降"]))
    css = (
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;color:#222}"
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 10px;font-size:14px}"
        "th{background:#f3f3f3}.ok{color:#18794e;font-weight:600}.bad{color:#c62828;font-weight:600}.na{color:#888}img{max-width:100%}"
    )
    out.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body>' + "\n".join(parts) + "</body></html>")


def render_video(qpos: np.ndarray, ref_bodies: np.ndarray, out: Path, width: int = 960, height: int = 540) -> None:
    """Robot solid, reference tracked bodies as red spheres (needs MUJOCO_GL=egl on a headless box)."""
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SPEC.xml_path()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -15.0
    writer = imageio.get_writer(out, fps=50, codec="libx264", quality=7, macro_block_size=8)
    for q, ref in zip(qpos, ref_bodies):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [q[0], q[1], 0.4]
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        for p in ref:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]), p.astype(np.float64), np.eye(3).flatten(), np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32))
            scn.ngeom += 1
        writer.append_data(renderer.render())
    writer.close()
