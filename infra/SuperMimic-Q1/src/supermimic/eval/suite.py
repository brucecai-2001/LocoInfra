"""Acceptance evaluation of a single-motion policy (capability doc, section 5).

Five experiments, all in the true simulator (never in imagination):

* ``deterministic``  every clip once from its first frame, no noise, no randomization; per-step traces of
                     reward, joint-position error and joint torques (peak vs limit, time above rated);
* ``randomized``     N one-episode environments with the full training randomization (physics, latency,
                     sensor noise, reset noise; no action noise), random start frames, 10 s episodes;
* ``push``           horizontal velocity kicks during standing / slow segments of the reference;
* ``solver``         the deterministic run again under other MuJoCo solver settings.

``run_suite`` returns a JSON-able dict; ``report.py`` turns it into ``report.html``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from supermimic.envs import make_env
from supermimic.envs.dr import actuator_model, policy_obs
from supermimic.eval.benchmark import BenchmarkRunner, deterministic_env_args, summarize
from supermimic.policy.nets import Policy
from supermimic.robot import SPEC

MAX_STEP_REWARD = 0.1


@dataclass
class Criteria:
    det_return: float = 0.80  # deterministic normalized return, with 0 falls
    dr_fall_rate: float = 0.01  # randomized episodes
    joint_err_rad: float = 0.10  # whole-clip mean |q - q_ref|
    solver_drop: float = 0.10  # relative return drop under other solver settings
    over_rated_margin: float = 1.2  # time above rated torque <= margin x feasibility estimate (+0.2 s)


@dataclass
class SuiteCfg:
    dr_episodes: int = 1000
    dr_episode_s: float = 10.0
    push_speeds: tuple[float, ...] = (0.5, 1.0, 1.5)
    push_per_speed: int = 64
    push_slow_speed: float = 0.3  # reference root speed (m/s) below which a frame counts as standing / slow
    push_recover_s: float = 3.0
    solver_variants: dict = field(
        default_factory=lambda: {
            "elliptic_cone": {"mujoco_overrides": {"cone": "elliptic"}},
            "cg_solver": {"mujoco_overrides": {"solver": "cg"}},
            "converged": {"mujoco_overrides": {"iterations": 50, "ls_iterations": 50}},
            "dt_2.5ms": {"sim_timestep": 0.0025, "decimation": 8},
        }
    )
    criteria: Criteria = field(default_factory=Criteria)


def _env_kw(cfg: dict) -> dict:
    e = cfg["env"]
    return dict(
        future_frames=e["future_frames"], anchor_obs=e.get("anchor_obs", False), anchor_ori_only=e.get("anchor_ori_only", False),
        anchor_pos_noise=e.get("anchor_pos_noise", 0.25),
    )


# ---------------------------------------------------------------------- deterministic traces


@torch.no_grad()
def deterministic(policy: Policy, cfg: dict, npz_files: list[str], device: str, feasibility_over_rated: dict | None = None) -> dict:
    n = len(npz_files)
    env = make_env(deterministic_env_args(npz_files, n, **_env_kw(cfg)), device=device)
    try:
        robot = env.scene["robot"]
        act = actuator_model(env)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=device)]
        obs, _ = env.reset()
        origin = env.scene.env_origins
        q_adr = torch.cat([robot.indexing.free_joint_q_adr, robot.indexing.joint_q_adr]).long()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fell = torch.zeros(n, dtype=torch.bool, device=device)
        action_term = env.action_manager.get_term("joint_pos")
        tr = {k: [] for k in ("reward", "joint_err", "torque", "demand", "root_err", "qpos", "ref_bodies", "alive", "state_ok")}
        names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(n, len(names), device=device)
        max_steps = int(lib.clip_len.max())
        for _ in range(max_steps):
            q_ref = cmd.joint_pos.clone()
            ref_bodies = cmd.body_pos_w - origin[:, None]
            obs, rew, term, tout, _ = env.step(policy(policy_obs(env, obs), obs["goal"]))
            # Values of this step are only meaningful for envs still playing their clip.
            tr["alive"].append(alive.clone())
            tr["reward"].append(rew.clone())  # the env reuses its reward buffer
            tr["joint_err"].append((robot.data.joint_pos - q_ref).abs())
            tr["torque"].append(robot.data.qfrc_actuator.clone())
            # PD torque before the force-range clip (what the motor would have to deliver).
            tr["demand"].append(act.kp * (action_term._processed_actions - robot.data.joint_pos) - act.kd * robot.data.joint_vel)
            tr["root_err"].append(torch.norm(cmd.robot_root_pos_w - cmd.root_pos_w, dim=-1))
            qp = env.sim.data.qpos[:, q_adr].clone()
            qp[:, 0:3] -= origin
            tr["qpos"].append(qp)
            tr["ref_bodies"].append(ref_bodies)
            term_sum += env.reward_manager.term_rewards * alive[:, None]
            fell |= term & alive
            alive &= ~(term | tout)
            # On the step an episode ends the post-step state is already the auto-reset one.
            tr["state_ok"].append(alive.clone())
            if not alive.any():
                break
        A = torch.stack(tr["alive"]).cpu().numpy()  # (T, n) env was playing during the step (reward valid)
        S = torch.stack(tr["state_ok"]).cpu().numpy()  # (T, n) and the post-step state is still that episode's
        st = {k: torch.stack(v).cpu().numpy() for k, v in tr.items() if k not in ("alive", "state_ok")}
    finally:
        env.close()

    effort = act.effort.cpu().numpy()
    rated = act.rated.cpu().numpy()
    joints = list(SPEC.joint_names)
    clips = []
    for i in range(n):
        m, s = A[:, i], S[:, i]
        steps = int(m.sum())
        tau = st["torque"][s, i]
        demand = np.abs(st["demand"][s, i])
        over = np.abs(tau) > rated
        over_s = over.sum(0) * 0.02
        jerr = st["joint_err"][s, i]
        c = {
            "name": lib.clips[i].name,
            "frames": lib.clips[i].length,
            "steps": steps,
            "fell": bool(fell[i]),
            "norm_return": float(st["reward"][m, i].sum() / (MAX_STEP_REWARD * max(steps, 1))),
            "terms": {k: float(term_sum[i, j]) / max(steps, 1) for j, k in enumerate(names)},
            "joint_err_mean": float(jerr.mean()),
            "joint_err_per_joint": dict(zip(joints, jerr.mean(0).round(4).tolist())),
            "root_err_mean": float(st["root_err"][s, i].mean()),
            "torque_peak": dict(zip(joints, np.abs(tau).max(0).round(2).tolist())),
            "torque_demand_peak": dict(zip(joints, demand.max(0).round(2).tolist())),
            "torque_peak_ratio": float((demand.max(0) / effort).max()),
            "torque_saturated_frac": float((demand > effort).mean()),
            "over_rated_s": {j: round(float(v), 2) for j, v in zip(joints, over_s) if v > 0},
            "over_rated_total_s": float(over_s.sum()),
            "traces": {
                "reward": (st["reward"][s, i] / MAX_STEP_REWARD).round(4).tolist(),
                "joint_err": jerr.mean(1).round(4).tolist(),
                "root_err": st["root_err"][s, i].round(4).tolist(),
                "joint_err_frames": jerr.round(4).tolist(),
                "torque": tau.round(2).tolist(),
            },
            "_qpos": st["qpos"][s, i],
            "_ref_bodies": st["ref_bodies"][s, i],
        }
        if feasibility_over_rated is not None:
            feas = float(sum(feasibility_over_rated.values()))
            c["feasibility_over_rated_s"] = feasibility_over_rated
            c["feasibility_over_rated_total_s"] = feas
        clips.append(c)
    return {"clips": clips, "torque_limit": dict(zip(joints, effort.tolist())), "torque_rated": dict(zip(joints, rated.tolist()))}


# ---------------------------------------------------------------------- randomized episodes


@torch.no_grad()
def randomized(policy: Policy, cfg: dict, npz_files: list[str], device: str, n: int, episode_s: float) -> dict:
    from supermimic.envs import SuperMimicEnvCfgArgs

    e = cfg["env"]
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files, num_envs=n, future_frames=e["future_frames"], episode_length_s=episode_s,
        action_noise_std=0.0, obs_noise=True, reset_noise=True, domain_randomization=True, physics_dr=True,
        sampling_mode="uniform", record_transitions=False, seed=1,
        anchor_obs=e.get("anchor_obs", False), anchor_ori_only=e.get("anchor_ori_only", False),
        anchor_pos_noise=e.get("anchor_pos_noise", 0.25),
    )
    env = make_env(args, device=device)
    try:
        obs, _ = env.reset()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fell = torch.zeros(n, dtype=torch.bool, device=device)
        ret = torch.zeros(n, device=device)
        steps = torch.zeros(n, device=device)
        for _ in range(int(episode_s / env.step_dt) + 2):
            obs, rew, term, tout, _ = env.step(policy(policy_obs(env, obs), obs["goal"]))
            ret += rew * alive
            steps += alive.float()
            fell |= term & alive
            alive &= ~(term | tout)
            if not alive.any():
                break
    finally:
        env.close()
    norm = (ret / (MAX_STEP_REWARD * steps.clamp_min(1))).cpu().numpy()
    return {
        "episodes": n,
        "falls": int(fell.sum()),
        "fall_rate": float(fell.float().mean()),
        "mean_norm_return": float(norm.mean()),
        "mean_episode_s": float(steps.mean() * 0.02),
    }


# ---------------------------------------------------------------------- push recovery


@torch.no_grad()
def push(policy: Policy, cfg: dict, npz_files: list[str], device: str, s: SuiteCfg) -> dict:
    lead, recover = 50, int(s.push_recover_s / 0.02)
    n = len(s.push_speeds) * s.push_per_speed
    env = make_env(deterministic_env_args(npz_files, n, **_env_kw(cfg)), device=device)
    try:
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        # Candidate push frames: inside a clip with 1 s before and the recovery window after.
        frames = torch.arange(lib.num_frames, device=device)
        clip = lib.frame_clip
        ok = (frames - lib.clip_start[clip] >= lead) & (lib.clip_last[clip] - frames >= recover)
        speed = torch.norm(lib.ref(frames).root_lin_vel_w[:, :2], dim=-1)
        slow = ok & (speed < s.push_slow_speed)
        if int(slow.sum()) < 10:  # no standing segment: take the slowest 20 % of the valid frames
            thr = torch.quantile(speed[ok], 0.2)
            slow = ok & (speed <= thr)
        cand = frames[slow]
        g = torch.Generator(device=device).manual_seed(0)
        pick = cand[torch.randint(0, len(cand), (n,), device=device, generator=g)]
        cmd.forced_frame[:] = pick - lead
        obs, _ = env.reset()
        v_push = torch.tensor(s.push_speeds, device=device).repeat_interleave(s.push_per_speed)
        ang = torch.rand(n, device=device, generator=g) * 2 * np.pi
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fell = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(lead + recover):
            if t == lead:
                v = robot.data.root_link_vel_w.clone()
                v[:, 0] += v_push * torch.cos(ang)
                v[:, 1] += v_push * torch.sin(ang)
                robot.write_root_link_velocity_to_sim(v)
            obs, rew, term, tout, _ = env.step(policy(policy_obs(env, obs), obs["goal"]))
            fell |= term & alive
            alive &= ~(term | tout)
    finally:
        env.close()
    out = {"slow_frames": int(slow.sum()), "slow_speed_mps": s.push_slow_speed, "recover_s": s.push_recover_s, "by_speed": {}}
    for k, sp in enumerate(s.push_speeds):
        sl = slice(k * s.push_per_speed, (k + 1) * s.push_per_speed)
        out["by_speed"][f"{sp:.1f}"] = {"pushes": s.push_per_speed, "falls": int(fell[sl].sum()), "recovered_rate": float(1 - fell[sl].float().mean())}
    out["all_recovered"] = all(v["falls"] == 0 for v in out["by_speed"].values())
    return out


# ---------------------------------------------------------------------- solver variants


def solver(policy: Policy, cfg: dict, npz_files: list[str], device: str, s: SuiteCfg, base_return: float) -> dict:
    out = {}
    for name, kw in s.solver_variants.items():
        r = BenchmarkRunner(npz_files, device, **_env_kw(cfg), **kw)
        try:
            sm = summarize(r.run(policy))
        finally:
            r.close()
        drop = (base_return - sm["mean_norm_return"]) / max(base_return, 1e-6)
        out[name] = {"settings": kw, "mean_norm_return": sm["mean_norm_return"], "failed": sm["failed"], "drop": drop}
    return out


# ---------------------------------------------------------------------- everything


def run_suite(policy: Policy, cfg: dict, npz_files: list[str], device: str = "cuda:0", s: SuiteCfg | None = None, feasibility_over_rated: dict | None = None, log=print) -> dict:
    s = s or SuiteCfg()
    c = s.criteria
    log("[suite] deterministic")
    det = deterministic(policy, cfg, npz_files, device, feasibility_over_rated)
    clips = det["clips"]
    det_ret = float(np.mean([x["norm_return"] for x in clips]))
    det_falls = sum(x["fell"] for x in clips)
    log(f"[suite]   return {det_ret:.3f}, falls {det_falls}/{len(clips)}")
    log(f"[suite] randomized x{s.dr_episodes}")
    dr = randomized(policy, cfg, npz_files, device, s.dr_episodes, s.dr_episode_s)
    log(f"[suite]   falls {dr['falls']}/{dr['episodes']}, return {dr['mean_norm_return']:.3f}")
    log("[suite] push")
    ps = push(policy, cfg, npz_files, device, s)
    log(f"[suite]   {ps['by_speed']}")
    log("[suite] solver variants")
    sv = solver(policy, cfg, npz_files, device, s, det_ret)
    log("[suite]   " + ", ".join(f"{k} {v['mean_norm_return']:.3f} ({v['drop']:+.1%})" for k, v in sv.items()))

    joint_err = float(np.mean([x["joint_err_mean"] for x in clips]))
    peak_ratio = max(x["torque_peak_ratio"] for x in clips)
    sat = max(x["torque_saturated_frac"] for x in clips)
    over_total = sum(x["over_rated_total_s"] for x in clips)
    checks = [
        {"name": "整条确定性播放", "value": f"回报 {det_ret:.3f}，摔倒 {det_falls}/{len(clips)}", "threshold": f"≥ {c.det_return:.2f}，0 摔倒", "passed": det_ret >= c.det_return and det_falls == 0},
        {"name": f"全量随机化 {dr['episodes']} 回合", "value": f"摔倒 {dr['falls']}（{dr['fall_rate']:.1%}），回报 {dr['mean_norm_return']:.3f}", "threshold": f"摔倒 ≤ {c.dr_fall_rate:.0%}", "passed": dr["fall_rate"] <= c.dr_fall_rate},
        {"name": "推力 " + " / ".join(ps["by_speed"]) + " m/s", "value": "，".join(f"{k}: {v['falls']}/{v['pushes']} 摔" for k, v in ps["by_speed"].items()), "threshold": "全部恢复", "passed": ps["all_recovered"]},
        {"name": "关节力矩峰值（PD 需求，限幅前）", "value": f"峰值 / 上限 = {peak_ratio:.2f}，超限 {sat:.2%} 关节·步", "threshold": "≤ 1", "passed": peak_ratio <= 1.0},
    ]
    if feasibility_over_rated is not None:
        feas = float(sum(feasibility_over_rated.values()))
        checks.append({"name": "超额定累计时间", "value": f"{over_total:.1f} s（可行性估算 {feas:.1f} s）", "threshold": f"≤ {c.over_rated_margin:.1f} × 估算 + 0.2 s", "passed": over_total <= c.over_rated_margin * feas + 0.2})
    else:
        checks.append({"name": "超额定累计时间", "value": f"{over_total:.1f} s（无可行性估算）", "threshold": "需可行性报告", "passed": None})
    checks.append({"name": "关节位置误差（全程平均）", "value": f"{joint_err:.3f} rad", "threshold": f"≤ {c.joint_err_rad:.2f} rad", "passed": joint_err <= c.joint_err_rad})
    worst = max(sv.values(), key=lambda v: v["drop"])
    checks.append({"name": "换 MuJoCo 求解器参数", "value": "，".join(f"{k} {v['drop']:+.1%}" for k, v in sv.items()), "threshold": f"回报下降 < {c.solver_drop:.0%}", "passed": worst["drop"] < c.solver_drop and all(v["failed"] == 0 for v in sv.values())})
    passed = all(ch["passed"] is not False for ch in checks)
    return {
        "passed": passed,
        "checks": checks,
        "deterministic": {"mean_norm_return": det_ret, "falls": det_falls, "joint_err_mean": joint_err, **det},
        "randomized": dr,
        "push": ps,
        "solver": sv,
    }
