"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.dr import actuator_model, policy_obs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.robot import PD_PRESET, check_pd_preset
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robot import GOAL_FEAT_DIM, load_robot_model
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel

NAN_DEBUG = bool(os.environ.get("SM_NAN_DEBUG"))


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        anchor_dim = (6 if cfg.env.anchor_ori_only else 9) if cfg.env.anchor_obs else 0
        cfg.policy.goal_dim = cfg.env.future_frames * GOAL_FEAT_DIM + anchor_dim
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.anchor_ori_only = cfg.env.anchor_ori_only
        cfg.policy_trainer.anchor_pos_noise = cfg.env.anchor_pos_noise
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        tc = cfg.tele if cfg.tele.enabled else None
        if tc is not None:
            cfg.policy.tele_hidden = tuple(tc.encoder_hidden)
            cfg.policy.tele_embed = tc.embed
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                anchor_ori_only=cfg.env.anchor_ori_only,
                anchor_pos_noise=cfg.env.anchor_pos_noise,
                physics_dr=cfg.env.physics_dr,
                reward_cfg=cfg.reward,
                seed=cfg.seed + 1000 * self.rank,
                tele=tc,
            ),
            device=self.device,
        )
        self.eval_files = [str(f) for f in files] if cfg.data.library is None else []
        self.evaluator = None
        self.eval_history: list[dict] = []
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_robot_model(), device=self.device)
        heads = AnalyticHeads(
            fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
            reward_cfg=cfg.reward, actuator=actuator_model(env),
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        fd = self.wm.spec.feat_dim
        self.critic = Critic(cfg.critic, feat_dim=fd, pcfg=cfg.policy).to(self.device)
        from supermimic.tele.slots import GOAL_DIM as TELE_GOAL_DIM

        mk_critic = lambda: Critic(cfg.critic, feat_dim=fd, goal_dim=TELE_GOAL_DIM, pcfg=cfg.policy).to(self.device)  # noqa: E731
        self.distill_critic = mk_critic() if tc is not None else None
        self.free_critic = mk_critic() if tc is not None else None
        for m in (self.wm, self.policy, self.critic, self.distill_critic, self.free_critic):
            if m is not None:
                broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.free_mask, self.distill_mask = self.command.free_mask, self.command.distill_mask
        tele = None
        if tc is not None:
            self.wm_trainer.virtual_ref = self.free_mask
            tele = dict(
                cfg=tc, distill_critic=self.distill_critic, free_critic=self.free_critic, distill_mask=self.distill_mask, free_mask=self.free_mask,
                soft_limits=heads.soft_limits, effort_weights=(cfg.reward.w_torque, cfg.reward.w_power, cfg.reward.w_over_rated),
                body_idx=self.command.tele_idx,
            )
            if self.is_main:
                print(f"[INFO] tele: {int(self.distill_mask.sum())} distill + {int(self.free_mask.sum())} free / {env.num_envs} envs, stand height {tc.stand_height:.3f} m")
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer, tele=tele)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps({**asdict(cfg), "pd_preset": PD_PRESET}, indent=2, default=str))
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        self.hours_before = 0.0  # training time of the checkpoint we resumed from
        self.t_start = time.time()
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
                "pd_preset": PD_PRESET,
                "train_hours": self.hours_before + (time.time() - self.t_start) / 3600,
                "eval_history": self.eval_history,
                **(
                    {
                        "distill_critic": self.distill_critic.state_dict(), "distill_critic_opt": self.policy_trainer.distill_critic_opt.state_dict(),
                        "free_critic": self.free_critic.state_dict(), "free_critic_opt": self.policy_trainer.free_critic_opt.state_dict(),
                        "tele_updates": self.policy_trainer.tele_updates,
                    }
                    if self.distill_critic is not None
                    else {}
                ),
            },
            self.log_dir / name,
        )

    @staticmethod
    def _pad_xi_inputs(sd: dict, model: torch.nn.Module, key: str, tail: int) -> bool:
        """The sim context xi changed length since the checkpoint (xi is followed by ``tail`` other inputs in
        the first layer ``key``): new trailing entries get zero input columns; trailing entries the code no
        longer has (e.g. scene entries of a scene branch, 0 after normalisation at their nominal
        values) are dropped. Either way the loaded function is unchanged at nominal context."""
        w, target = sd[key], model.state_dict()[key]
        if w.shape == target.shape:
            return False
        extra = target.shape[1] - w.shape[1]
        assert w.shape[0] == target.shape[0], (key, w.shape, target.shape)
        cut = w.shape[1] - tail
        if extra > 0:
            sd[key] = torch.cat([w[:, :cut], torch.zeros(w.shape[0], extra, dtype=w.dtype, device=w.device), w[:, cut:]], dim=1)
            print(f"[INFO] {key}: {extra} new xi inputs (zero-initialised)")
        else:
            sd[key] = torch.cat([w[:, : cut + extra], w[:, cut:]], dim=1)
            print(f"[INFO] {key}: {-extra} trailing xi inputs dropped")
        return True

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if self.is_main:
            check_pd_preset(ck, strict=False)
        wm_padded = self._pad_xi_inputs(ck["wm"], self.wm, "dynamics.mlp.net.0.weight", tail=0)
        critic_padded = self._pad_xi_inputs(ck["critic"], self.critic, "mlp.net.0.weight", tail=4)
        self.wm.load_state_dict(ck["wm"])
        from supermimic.policy.nets import migrate_policy_state, tele_from_keypoint

        policy_sd, kp_sd = migrate_policy_state(ck["policy"])
        migrated = any(k.startswith(("encoders.motion.", "encoders.velocity.", "encoders.keypoint.")) for k in ck["policy"])  # parameter order changed
        missing, unexpected = self.policy.load_state_dict(policy_sd, strict=False)
        assert not unexpected, unexpected
        self.critic.load_state_dict(ck["critic"])
        if not wm_padded and "wm_opt" in ck:
            self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        if missing:
            # A checkpoint gaining E_tele: seed it from the baseline keypoint encoder if there is one; fresh optimizer state.
            assert all(k.startswith("encoders.tele.") for k in missing), missing
            if kp_sd is not None and "tele" in self.policy.encoders:
                tele_from_keypoint(self.policy.encoders["tele"], kp_sd)
                print("[INFO] E_tele seeded from the checkpoint's keypoint encoder")
            print(f"[INFO] new policy parameters (not in checkpoint): {len(missing)} tensors")
        elif "policy_opt" in ck and not migrated:
            self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        if not critic_padded and "critic_opt" in ck:
            self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        for name in ("distill_critic", "free_critic"):
            critic = getattr(self, name)
            if critic is not None and name in ck:
                padded = self._pad_xi_inputs(ck[name], critic, "mlp.net.0.weight", tail=4)
                critic.load_state_dict(ck[name])
                if not padded:
                    getattr(self.policy_trainer, f"{name}_opt").load_state_dict(ck[f"{name}_opt"])
        if "tele_updates" in ck:
            self.policy_trainer.tele_updates = int(ck["tele_updates"])
        if ck["bin_return"].shape == self.command.bin_return.shape:
            self.command.bin_return.copy_(ck["bin_return"])
            self.command.bin_memory.copy_(ck["bin_memory"])
        else:
            print("[INFO] motion library changed since the checkpoint: reset-sampler statistics start fresh")
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        self.hours_before = float(ck.get("train_hours", 0.0))
        self.eval_history = list(ck.get("eval_history", []))
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ periodic evaluation

    def evaluate(self) -> dict:
        """Deterministic whole-clip benchmark of the current policy (main rank only)."""
        from supermimic.eval.benchmark import BenchmarkRunner, summarize

        cfg = self.cfg
        if self.evaluator is None:
            self.evaluator = BenchmarkRunner(
                self.eval_files, self.device, future_frames=cfg.env.future_frames,
                anchor_obs=cfg.env.anchor_obs, anchor_ori_only=cfg.env.anchor_ori_only, reward_cfg=cfg.reward,
            )
        self.policy.eval()
        s = summarize(self.evaluator.run(self.policy))
        if self.distill_critic is not None:
            k = summarize(self.evaluator.run(self.policy, interface="tele"))
            s["tele"] = {key: k[key] for key in ("failed", "mean_norm_return", "hand_err_m") if key in k}
        self.policy.train()
        s.pop("terms", None)
        s.update(iteration=self.iteration, hours=self.hours_before + (time.time() - self.t_start) / 3600)
        s["passed"] = s["failed"] == 0 and s["mean_norm_return"] >= cfg.schedule.stop_return
        self.eval_history.append(s)
        (self.log_dir / "eval_history.json").write_text(json.dumps(self.eval_history, indent=1))
        it = self.iteration
        self.writer.add_scalar("eval/mean_norm_return", s["mean_norm_return"], it)
        self.writer.add_scalar("eval/failed", s["failed"], it)
        eval_msg = ""
        if "tele" in s:
            k = s["tele"]
            self.writer.add_scalar("eval/tele_mean_norm_return", k["mean_norm_return"], it)
            self.writer.add_scalar("eval/tele_failed", k["failed"], it)
            eval_msg = f" | tele return {k['mean_norm_return']:.3f} failed {k['failed']}/{s['clips']} hand err {100 * k.get('hand_err_m', float('nan')):.1f} cm"
        print(f"[EVAL] it {it} | {s['hours']:.2f} h | return {s['mean_norm_return']:.3f} | failed {s['failed']}/{s['clips']}{eval_msg} | {'PASS' if s['passed'] else 'not yet'}", flush=True)
        return s

    def _nan_watch(self, m_wm: dict, m_pi: dict) -> None:
        keys = ("wm/loss", "wm/grad_norm", "pi/imag_objective", "pi/policy_grad_norm", "pi/action_abs_mean", "pi/imag_survival_H", "pi/critic_loss", "pi/critic_value_mean")
        m = {**{f"wm/{k}": v for k, v in m_wm.items()}, **{f"pi/{k}": v for k, v in m_pi.items()}}
        print(f"[nan-debug] upd {self.num_updates} " + " ".join(f"{k}={m.get(k, float('nan')):.4g}" for k in keys), flush=True)
        bad = {k: v for k, v in m.items() if not math.isfinite(v)}
        bad_params = [f"{name}:{n}" for name, mod in (("wm", self.wm), ("policy", self.policy), ("critic", self.critic)) for n, p in mod.named_parameters() if not torch.isfinite(p).all()]
        if bad or bad_params:
            print(f"[nan-debug] FIRST NON-FINITE at update {self.num_updates}: metrics {sorted(bad)} params {bad_params[:8]}", flush=True)
            s = self.buffer.s[self.buffer.valid] if hasattr(self.buffer, "s") else None
            if s is not None:
                print(f"[nan-debug] buffer states: finite {bool(torch.isfinite(s).all())} max|s| {float(s.abs().max()):.3g}", flush=True)
            raise SystemExit(3)

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        o = policy_obs(self.env, obs)
        if self.distill_critic is None:
            return self.policy(o, obs["goal"])
        m, t = self._ref_ids, self._tele_ids
        a = torch.empty(o.shape[0], self.policy.decoder.mlp.net[-1].out_features, device=o.device)
        a[m] = self.policy(o[m], obs["goal"][m])
        if t.numel():
            a[t] = self.policy(o[t], self.command.tele_goal[t], "tele")
        return a

    @torch.no_grad()
    def _free_error(self) -> torch.Tensor:
        """(6,) mean |robot slot - command| over the free envs (simulator, current step), valid slots only:
        vx, vy, wz, height, torso gravity (norm), mean hand position error (norm)."""
        from supermimic.tele import slots as S
        from supermimic.tele.interface import slots_from_bodies

        c = self.command
        ids = self._free_ids
        d = c.robot.data
        robot = slots_from_bodies(
            d.root_link_pos_w[ids] - self.env.scene.env_origins[ids], d.root_link_quat_w[ids], d.root_link_lin_vel_w[ids], d.root_link_ang_vel_b[ids],
            c.robot_body_pos_w[ids], c.robot_body_quat_w[ids], c.tele_idx,
        )
        slots, mask = S.split_goal(c.tele_goal[ids])
        e = robot - slots
        m = lambda n: mask[:, S.SLOT_NAMES.index(n)]  # noqa: E731
        wavg = lambda x, w: (x * w).sum() / w.sum().clamp_min(1)  # noqa: E731
        b = S.SLICE["base"]
        hand = 0.5 * (e[:, S.SLICE["hand_L"]].norm(dim=-1) + e[:, S.SLICE["hand_R"]].norm(dim=-1))
        return torch.stack([
            wavg(e[:, b.start].abs(), m("base")), wavg(e[:, b.start + 1].abs(), m("base")), wavg(e[:, b.start + 2].abs(), m("base")),
            wavg(e[:, S.SLICE["height"].start].abs(), m("height")), wavg(e[:, S.SLICE["torso"]].norm(dim=-1), m("torso")),
            wavg(hand, m("hand_L") * m("hand_R")),
        ])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = self.t_start = time.time()
        start_iteration = self.iteration
        next_eval_s = sch.eval_interval_min * 60 if sch.eval_interval_min else None
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        self._ref_ids = (~self.free_mask & ~self.distill_mask).nonzero().squeeze(-1)
        self._tele_ids = (self.free_mask | self.distill_mask).nonzero().squeeze(-1)
        self._free_ids = self.free_mask.nonzero().squeeze(-1)
        distill_returns: list[float] = []
        term_counts: dict[str, torch.Tensor] = {}
        free_err = torch.zeros(6, device=self.device)
        free_eps = free_falls = 0
        free_len_sum = 0.0
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if self.distill_critic is not None:
                free_err += self._free_error()
                vd = done & self.free_mask
                if vd.any():
                    free_eps += int(vd.sum())
                    free_falls += int((term & self.free_mask).sum())
                    free_len_sum += float(self.ep_len[vd].sum())
                    self.ep_reward_sum[vd] = 0.0
                    self.ep_len[vd] = 0.0
                done = done & ~self.free_mask
                kd = done & self.distill_mask
                if kd.any():
                    distill_returns.extend((self.ep_reward_sum[kd] / (self.max_step_reward * self.ep_len[kd].clamp_min(1))).tolist())
                    self.ep_reward_sum[kd] = 0.0
                    self.ep_len[kd] = 0.0
                done = done & ~self.distill_mask
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - start_iteration >= sch.warmup_iterations:  # the replay buffer is empty after a resume too
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr_it, lr_total = (self.iteration - start_iteration, sch.total_iterations - start_iteration) if sch.lr_from_resume else (self.iteration, sch.total_iterations)
                    lr = cosine_lr(lr_it, lr_total, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    if NAN_DEBUG:
                        self._nan_watch(m_wm, m_pi)
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    tele_msg = ""
                    if self.distill_critic is not None:
                        e = (free_err / sch.log_interval).tolist()
                        for k, x in zip(("vx", "vy", "wz", "height", "torso", "hand"), e):
                            self.writer.add_scalar(f"free/sim_abs_err_{k}", x, it)
                        if free_eps:
                            self.writer.add_scalar("free/sim_fall_rate", free_falls / free_eps, it)
                            self.writer.add_scalar("free/sim_episode_len", free_len_sum / free_eps, it)
                        g = lambda k: agg.get(k, float("nan")) / max(agg_n, 1)  # noqa: E731
                        kr = sum(distill_returns) / len(distill_returns) if distill_returns else float("nan")
                        if distill_returns:
                            self.writer.add_scalar("distill/train_episode_norm_return", kr, it)
                        tele_msg = (
                            f" | free err v {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f} h {e[3]:.3f} torso {e[4]:.2f} hand {e[5]:.3f} falls {free_falls}/{free_eps} "
                            f"imag {g('policy/free/imag_return_undisc'):.3f} align {g('policy/free/align_loss'):.3f}"
                            f" | distill train_ret {kr:.3f} (n={len(distill_returns)}) imag {g('policy/distill/imag_return_undisc'):.3f} "
                            f"align {g('policy/distill/align_loss'):.3f} actor {g('policy/tele/actor_on'):.0f}"
                        )
                    if term_counts:
                        tc = {k: int(v) for k, v in term_counts.items()}
                        for k, v in tc.items():
                            self.writer.add_scalar(f"term/{k}", v, it)
                        tele_msg += " | term " + " ".join(f"{k} {v}" for k, v in tc.items())
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg + tele_msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                free_err.zero_()
                distill_returns = []
                term_counts = {}
                free_eps = free_falls = 0
                free_len_sum = 0.0
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                every = sch.checkpoint_interval if sch.numbered_checkpoint_interval is None else sch.numbered_checkpoint_interval
                if every and self.iteration % every == 0:
                    self.save(f"ckpt_{self.iteration:07d}.pt")
            if next_eval_s is not None and self.iteration % sch.log_interval == 0 and self.eval_files:
                due = float(time.time() - t_start >= next_eval_s)
                if self.world > 1:
                    due = allreduce_mean(due)
                if due > 0:
                    next_eval_s += sch.eval_interval_min * 60
                    passed = 0.0
                    if self.is_main:
                        passed = float(self.evaluate()["passed"])
                        self.save("ckpt_latest.pt")
                    if self.world > 1:
                        passed = allreduce_mean(passed)
                    if sch.stop_on_pass and passed > 0:
                        if self.is_main:
                            print("[INFO] evaluation passed, stopping")
                        break
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                        if next_eval_s is not None and self.eval_files:
                            self.evaluate()
                    break
        self.save("ckpt_latest.pt")
        if self.evaluator is not None:
            self.evaluator.close()
        if self.writer is not None:
            self.writer.flush()
        finalize()
