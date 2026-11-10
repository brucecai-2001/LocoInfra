"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048))."""

from __future__ import annotations

import torch
import torch.nn as nn

from supermimic.robot import DELTA_DIM, NJ, STATE_DIM, XI_DIM


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a_t, a_{t-1}, xi_norm) -> (…, DELTA_DIM). Zero-initialised last layer
    (identity dynamics at init). a_{t-1} matters under command latency (the old set-point is held
    for the first substeps of the step)."""

    def __init__(self, feat_dim: int = STATE_DIM, act_dim: int = NJ, xi_dim: int = XI_DIM, hidden=(2048, 2048, 2048), out_dim: int = DELTA_DIM):
        super().__init__()
        self.mlp = MLP(feat_dim + 2 * act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, prev_action, xi_norm], dim=-1))
