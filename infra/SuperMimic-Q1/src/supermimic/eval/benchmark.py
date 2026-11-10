"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    hand_err_m: float | None = None  # tele interface: mean hand-position error relative to the head (m)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    """Policy of a checkpoint; three-encoder checkpoints load as their E_ref + decoder."""
    from supermimic.policy.nets import migrate_policy_state, policy_cfg_from_dict

    from supermimic.robot import check_pd_preset

    ck = torch.load(path, map_location=device, weights_only=False)
    check_pd_preset(ck)
    policy = Policy(policy_cfg_from_dict(ck["config"]["policy"])).to(device)
    policy.load_state_dict(migrate_policy_state(ck["policy"])[0])
    policy.eval()
    return policy, ck["config"]


def deterministic_env_args(npz_files: list[str], num_envs: int, future_frames: int = 10, anchor_obs: bool = False, anchor_ori_only: bool = False, **overrides) -> SuperMimicEnvCfgArgs:
    """Benchmark protocol: whole clips from their first frame, no noise, no randomization."""
    kw = dict(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        anchor_ori_only=anchor_ori_only,
    )
    kw.update(overrides)
    return SuperMimicEnvCfgArgs(**kw)


class BenchmarkRunner:
    """Keeps one deterministic evaluation env alive so it can be re-run (e.g. during training)."""

    def __init__(self, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, **env_kw):
        self.num_envs = min(num_envs or 1024, len(npz_files))
        self.device = device
        self.env = make_env(deterministic_env_args(npz_files, self.num_envs, **env_kw), device=device)

    @torch.no_grad()
    def run(self, policy: Policy, verbose: bool = False, interface: str = "ref", slot_mask: torch.Tensor | None = None) -> list[ClipResult]:
        """``interface="tele"``: E_tele reads the reference-derived slots (supermimic/tele/labels.py) with
        ``slot_mask`` (NUM_SLOTS,), default every non-reserved slot; ``hand_err_m`` is then the mean hand
        position error relative to the head (heading frame)."""
        from supermimic.envs.dr import policy_obs
        from supermimic.tele import slots as S
        from supermimic.tele.interface import slots_from_bodies

        env, num_envs, device = self.env, self.num_envs, self.device
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        kp = interface == "tele"
        if kp:
            cmd.ensure_tele_labels()
            m = (~S.reserved_mask(device)).float() if slot_mask is None else slot_mask.to(device).float()
            mask = m.expand(num_envs, -1)
            d = cmd.robot.data
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        hand_sum = torch.zeros(num_envs, device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            if kp:
                target = lib.tele_slots[cmd.frame]
                act = policy(policy_obs(env, obs), torch.cat([target, mask], dim=-1), "tele")
            else:
                act = policy(policy_obs(env, obs), obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            if kp:
                rob = slots_from_bodies(d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_b, cmd.robot_body_pos_w, cmd.robot_body_quat_w, cmd.tele_idx)
                e = rob - target
                hand_sum += 0.5 * (e[:, S.SLICE["hand_L"]].norm(dim=-1) + e[:, S.SLICE["hand_R"]].norm(dim=-1))
            steps += 1
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            hand_err_m=float(hand_sum[e]) / max(st, 1) if kp else None,
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                hand_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]

    def close(self) -> None:
        self.env.close()


def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, anchor_ori_only: bool = False, **env_kw) -> list[ClipResult]:
    runner = BenchmarkRunner(npz_files, device, num_envs, future_frames=future_frames, anchor_obs=anchor_obs, anchor_ori_only=anchor_ori_only, **env_kw)
    try:
        return runner.run(policy, verbose)
    finally:
        runner.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    if results and results[0].hand_err_m is not None:
        out["hand_err_m"] = sum(r.hand_err_m * r.steps for r in results) / max(tot_steps, 1)
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
