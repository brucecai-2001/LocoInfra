"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5).

Every imagined batch is split by batch type (docs/training.md):
    ref      starts from E_ref envs; trains E_ref and the shared decoder on the tracking reward
    distill  starts from distill envs (tele_ok rows); E_tele on reference-derived slots, tracking reward
    free     starts from free-command envs; E_tele on held / reference commands, slot-tracking reward
The two tele batches update E_tele only: the return gradient passes through the decoder, whose weights
are held fixed there. Each batch type has its own critic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine, imagine_free
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False
    anchor_ori_only: bool = False
    anchor_pos_noise: float = 0.25
    freeze_ref: bool = False  # E_ref and the decoder stay fixed (only E_tele trains)


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg, tele=None):
        """``tele``: optional dict(cfg=TeleCfg, distill_critic=Critic, free_critic=Critic, distill_mask=(E,) bool,
        free_mask=(E,) bool, soft_limits=(NJ, 2), effort_weights=(3,), body_idx=tracked_indices)."""
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import FreeImagStep, ImagStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)
        self.tele = tele
        self.tele_updates = 0
        if tele is not None:
            tc = tele["cfg"]
            self.distill_critic, self.free_critic = tele["distill_critic"], tele["free_critic"]
            self.distill_critic_opt = torch.optim.Adam(self.distill_critic.parameters(), lr=cfg.critic_lr)
            self.free_critic_opt = torch.optim.Adam(self.free_critic.parameters(), lr=cfg.critic_lr)
            self.free_step_fn = FreeImagStep(wm, lib.default_joint_pos, tc, tele["soft_limits"], tele["effort_weights"], tele["body_idx"], compile_mode=cfg.compile)
        self.ref_params = [p for n, p in policy.named_parameters() if not n.startswith("encoders.tele.")]
        self.tele_params = list(policy.encoders["tele"].parameters()) if tele is not None else []

    @property
    def critics(self) -> list[Critic]:
        return [self.critic] + ([self.distill_critic, self.free_critic] if self.tele is not None else [])

    def _downstream_backward(self, own: list, loss_fn) -> None:
        """Backward of a tele batch's loss with every other policy parameter held fixed
        (the gradient still flows through the shared decoder into ``own``)."""
        own_ids = {id(p) for p in own}
        frozen = [p for p in self.policy.parameters() if id(p) not in own_ids and p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        try:
            loss_fn().backward()
        finally:
            for p in frozen:
                p.requires_grad_(True)

    def set_lr(self, lr: float) -> None:
        opts = [self.opt, self.critic_opt] + ([self.distill_critic_opt, self.free_critic_opt] if self.tele is not None else [])
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def _clip(self, params) -> torch.Tensor:
        gn = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(params, self.cfg.grad_clip_value)
        return gn

    def _fit_critic(self, critic: Critic, opt, ro: ImagRollout, feat, xi_norm, prefix: str, metrics: dict) -> None:
        cfg = self.cfg
        with torch.no_grad():
            feat_d, goals_d, e_root_d = feat.detach(), ro.goals.detach(), ro.e_root.detach()
            H1, n, _ = feat_d.shape
            xi_rep = xi_norm[None].expand(H1, n, -1)
            values = critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1)).reshape(H1, n)
            targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
            X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
            X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
            X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
            X_er = e_root_d[:-1].reshape(-1, 4)
            Y = targets.reshape(-1)
        N = Y.shape[0]
        mb = N // cfg.critic_minibatches
        last = torch.zeros((), device=Y.device)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(N, device=Y.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                pred = critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                closs = torch.nn.functional.mse_loss(pred, Y[idx])
                opt.zero_grad(set_to_none=True)
                closs.backward()
                allreduce_grads(critic)
                cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_clip)
                if torch.isfinite(cgn):
                    opt.step()
                last = closs.detach()
        metrics[f"{prefix}critic_loss"] = float(last)
        metrics[f"{prefix}critic_value_mean"] = float(values.mean())
        metrics[f"{prefix}critic_target_mean"] = float(targets.mean())

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        tele = self.tele
        n_dis = n_free = 0
        if tele is not None:
            tc = tele["cfg"]
            n_dis = int(round(cfg.batch_size * tc.distill_batch_fraction))
            n_free = int(round(cfg.batch_size * tc.free_batch_fraction))
            start_d = buffer.sample_starts(n_dis, cfg.history, env_mask=tele["distill_mask"], frame_ok=self.lib.tele_ok)
            start_f = buffer.sample_starts(n_free, cfg.history, env_mask=tele["free_mask"])
            ref_mask = ~(tele["distill_mask"] | tele["free_mask"])
        else:
            ref_mask = None
        start = buffer.sample_starts(cfg.batch_size - n_dis - n_free, cfg.history, env_mask=ref_mask)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        mk = dict(obs_noise=cfg.obs_noise, anchor_obs=cfg.anchor_obs, anchor_ori_only=cfg.anchor_ori_only, anchor_pos_noise=cfg.anchor_pos_noise)
        tele_actor = tele is not None and self.tele_updates >= tele["cfg"].critic_warmup_updates

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        for c in self.critics:
            c.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        if cfg.freeze_ref:
            with torch.no_grad():
                ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, step_fn=self.step_fn, **mk)
        else:
            ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, step_fn=self.step_fn, **mk)
        feat, goals, e_root = self._critic_inputs(ro)
        v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1]) if cfg.use_critic else None
        J = policy_objective(ro, v_H, cfg.gamma)
        if not cfg.freeze_ref:
            (-J / cfg.horizon).backward()

        if tele is not None:
            tc = tele["cfg"]
            xi_d, xi_f = self.wm.spec.normalize_xi(start_d.xi), self.wm.spec.normalize_xi(start_f.xi)
            box = {}

            def distill_loss():
                r = imagine(self.wm, self.policy, self.lib, start_d, cfg.horizon, cfg.future_frames, step_fn=self.step_fn, tele=dict(align=tc.align_weight > 0), **mk)
                f, g, e = self._critic_inputs(r)
                v = self.distill_critic(f[-1], g[-1], xi_d, e[-1]) if cfg.use_critic else None
                box.update(ro_d=r, feat_d=f, J_d=policy_objective(r, v, cfg.gamma))
                return -box["J_d"] / cfg.horizon + tc.align_weight * r.align_loss

            def free_loss():
                r = imagine_free(self.wm, self.policy, self.lib, start_f, cfg.horizon, self.free_step_fn, tc.stand_height, future_frames=cfg.future_frames, align=tc.align_weight > 0, **mk)
                f, g, e = self._critic_inputs(r)
                v = self.free_critic(f[-1], g[-1], xi_f, e[-1]) if cfg.use_critic else None
                box.update(ro_f=r, feat_f=f, J_f=policy_objective(r, v, cfg.gamma))
                return -box["J_f"] / cfg.horizon + tc.align_weight * r.align_loss

            if tele_actor:
                self._downstream_backward(self.tele_params, distill_loss)
                self._downstream_backward(self.tele_params, free_loss)
            else:
                with torch.no_grad():
                    distill_loss()
                    free_loss()
            ro_d, feat_d, ro_f, feat_f = box["ro_d"], box["feat_d"], box["ro_f"], box["feat_f"]

        allreduce_grads(self.policy)
        norms = []
        if not cfg.freeze_ref:
            norms.append(self._clip(self.ref_params))
        if tele_actor:
            norms.append(self._clip(self.tele_params))
        # A single non-finite imagined rollout would otherwise turn every policy weight into NaN.
        skipped = not all(bool(torch.isfinite(g)) for g in norms)
        if norms and not skipped:
            self.opt.step()
        self.wm.requires_grad_(True)
        for c in self.critics:
            c.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(norms[0]) if norms and not skipped else 0.0,
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
            "skipped_nonfinite": float(skipped),
        }
        if tele is not None:
            for p, r in (("distill/", ro_d), ("free/", ro_f)):
                metrics.update({
                    f"{p}imag_return_undisc": float((r.survival[:-1] * r.rewards).sum(0).mean().detach()),
                    f"{p}imag_survival_H": float(r.survival[-1].mean().detach()),
                    f"{p}align_loss": float(r.align_loss.detach()),
                })
            metrics["tele/grad_norm"] = float(norms[-1]) if tele_actor and not skipped else 0.0
            metrics["tele/actor_on"] = float(tele_actor)
            self.tele_updates += 1
        if skipped:
            return metrics
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v
        if cfg.use_critic:
            self._fit_critic(self.critic, self.critic_opt, ro, feat, xi_norm, "", metrics)
        if tele is not None:
            for k, v in ro_f.reward_terms.items():
                metrics[f"free/imag_r/{k}"] = v
            if cfg.use_critic:
                self._fit_critic(self.distill_critic, self.distill_critic_opt, ro_d, feat_d, xi_d, "distill/", metrics)
                self._fit_critic(self.free_critic, self.free_critic_opt, ro_f, feat_f, xi_f, "free/", metrics)
        return metrics
