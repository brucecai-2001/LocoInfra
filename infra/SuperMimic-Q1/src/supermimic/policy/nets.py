"""Base motion model (docs/training.md): two goal encoders and one shared decoder, plus the critic.

    z = E_ref(body_ref)                      video / full-reference tracking
    z = E_tele(slots, mask, o_{t-9:t})       teleoperation (supermimic/tele/slots.py)
    a = D(z, o_{t-9:t})

The decoder is trained by the ``ref`` batches only; ``tele`` batches update E_tele through the
frozen decoder (imagination/policy_trainer.py).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from supermimic.robot import GOAL_FEAT_DIM, NJ, OBS_DIM, STATE_DIM, XI_DIM
from supermimic.tele import slots as S
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import OBS_TERMS

GOAL_DIM = 10 * GOAL_FEAT_DIM  # G1: 680
LATENT_DIM = 32
ACT_DIM = NJ
INTERFACES = ("ref", "tele")


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos NJ, joint_vel NJ, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # body_ref: K future frames x 54 (+6 / +9 anchor terms)
    # Teleoperation encoder; None = no E_tele.
    tele_hidden: tuple[int, ...] | None = None
    tele_embed: int = 64  # per-slot projection width (>= 13 for the keypoint warm start)


def policy_cfg_from_dict(d: dict) -> PolicyCfg:
    """PolicyCfg from a checkpoint config, including three-encoder checkpoints (velocity / keypoint
    fields are dropped; their keypoint encoder can seed E_tele, see ``tele_from_keypoint``)."""
    names = {f.name for f in fields(PolicyCfg)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return PolicyCfg(**kw)


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = (NJ, NJ, 3, 3, 3, 1)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    k, extra = divmod(goal_dim, GOAL_FEAT_DIM)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


class RefEncoder(nn.Module):
    """E_ref: z = tanh(MLP(body_ref)); body_ref = K future reference frames (+ anchor orientation)."""

    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class TeleEncoder(nn.Module):
    """E_tele: every slot i is projected (W_i x_i + e_i, e_i its type embedding), multiplied by its
    mask bit and sum-pooled; the pooled vector, the mask bits and the scaled observation history go
    through an MLP to z. Input ``goal`` = [slots (SLOT_DIM), mask (NUM_SLOTS)] in physical units.
    A new slot only appends a projection (the trunk's input width does not change)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        E = cfg.tele_embed
        self.proj = nn.ModuleList(nn.Linear(s.dim, E) for s in S.SLOTS)
        self.type_emb = nn.Parameter(torch.zeros(S.NUM_SLOTS, E))
        self.mlp = MLP(E + S.NUM_SLOTS + cfg.history * OBS_DIM, cfg.tele_hidden, cfg.latent_dim)
        self.register_buffer("offset", S.default_vector())
        self.register_buffer("in_scale", 1.0 / S.scale_vector())
        self.register_buffer("obs_scale", _obs_scale_vector(cfg))
        self.bounds = [(S.SLICE[s.name].start, S.SLICE[s.name].stop) for s in S.SLOTS]

    def forward(self, goal: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        x = (goal[..., : S.SLOT_DIM] - self.offset) * self.in_scale
        m = goal[..., S.SLOT_DIM :]
        pooled = None
        for i, (a, b) in enumerate(self.bounds):
            e = m[..., i : i + 1] * (self.proj[i](x[..., a:b]) + self.type_emb[i])
            pooled = e if pooled is None else pooled + e
        return torch.tanh(self.mlp(torch.cat([pooled, m, obs_hist * self.obs_scale], dim=-1)))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int | None = None):
        super().__init__()
        obs_dim = obs_dim or cfg.history * OBS_DIM
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = D(E_i(goal), o_hist), i in ``ref`` (goal = body_ref) / ``tele`` (goal = slots + mask)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"ref": RefEncoder(cfg)})
        if cfg.tele_hidden is not None:
            self.encoders["tele"] = TeleEncoder(cfg)
        self.decoder = Decoder(cfg)

    def latent(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "ref") -> torch.Tensor:
        if interface == "tele":
            return self.encoders["tele"](goal, obs_hist)
        return self.encoders[interface](goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "ref") -> torch.Tensor:
        return self.decoder(self.latent(obs_hist, goal, interface), obs_hist)


# Old keypoint goal (13, supermimic keypoint interface of the three-encoder baseline):
#   head height, gravity in the head (= torso) frame, left / right hand relative to the head in the
#   torso heading frame, walking command; each divided by these scales.
_KP_SCALE = (0.5, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.5, 1.5)
_KP_COLS = {"height": [0], "torso": [1, 2, 3], "hand_L": [4, 5, 6], "hand_R": [7, 8, 9], "base": [10, 11, 12]}
HEAD_ABOVE_PELVIS = 0.38  # m, head_link above the pelvis in the default stance (approximate for the warm start)


def migrate_policy_state(sd: dict) -> tuple[dict, dict | None]:
    """Three-encoder checkpoint -> (state with ``encoders.motion`` renamed to ``encoders.ref`` and the
    velocity / keypoint encoders removed, keypoint encoder state or None)."""
    out, kp = {}, {}
    for k, v in sd.items():
        if k.startswith("encoders.motion."):
            out["encoders.ref." + k[len("encoders.motion.") :]] = v
        elif k.startswith("encoders.keypoint."):
            kp[k[len("encoders.keypoint.") :]] = v
        elif k.startswith("encoders.velocity."):
            continue
        else:
            out[k] = v
    return out, (kp or None)


@torch.no_grad()
def tele_from_keypoint(tele: TeleEncoder, kp: dict) -> None:
    """Seed E_tele from the baseline keypoint encoder. The first 13 pooled dims carry the old normalized
    keypoint goal minus its value at the slot defaults (head height from the pelvis height offset, the
    rest are the same quantities); the defaults are folded into the trunk's first bias. So with every
    slot valid E_tele reproduces the keypoint encoder, and a masked slot reads as its default (upright,
    natural hands, zero command). The trunk copies the old MLP with zero columns for the remaining
    pooled dims and the mask bits."""
    E = tele.type_emb.shape[1]
    assert E >= 13, "tele_embed must be >= 13 for the keypoint warm start"
    w0 = kp["mlp.net.0.weight"]
    n0 = tele.mlp.net[0]
    assert n0.weight.shape[0] == w0.shape[0], "tele_hidden must match the keypoint encoder"
    n0.weight.zero_()
    n0.weight[:, :13] = w0[:, :13]
    n0.weight[:, E + S.NUM_SLOTS :] = w0[:, 13:]
    u_def = torch.zeros(13, dtype=w0.dtype, device=w0.device)
    for i, layer in enumerate(tele.mlp.net):
        if i > 0 and isinstance(layer, nn.Linear):
            layer.weight.copy_(kp[f"mlp.net.{i}.weight"])
            layer.bias.copy_(kp[f"mlp.net.{i}.bias"])
    tele.type_emb.zero_()
    for s, lin in zip(S.SLOTS, tele.proj):
        lin.weight[:13].zero_()
        lin.bias[:13].zero_()
        cols = _KP_COLS.get(s.name)
        if cols is None:
            continue
        dflt = torch.tensor(s.default)
        if s.name == "height":  # head height ≈ H_STAND + HEAD_ABOVE_PELVIS + Δh
            dflt = dflt + S.H_STAND + HEAD_ABOVE_PELVIS
        for j, c in enumerate(cols):
            lin.weight[c, j] = s.scale[j] / _KP_SCALE[c]
            u_def[c] = dflt[j] / _KP_SCALE[c]
    n0.bias.copy_(kp["mlp.net.0.bias"] + w0[:, :13] @ u_def)


class Critic(nn.Module):
    """V(canonical state, goal, xi, root error 4) -> scalar (one per batch type)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = STATE_DIM, goal_dim: int | None = None, xi_dim: int = XI_DIM, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
