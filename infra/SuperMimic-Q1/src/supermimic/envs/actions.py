"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).

Command latency (``envs/dr.py``): for the first ``lag`` physics substeps of a control step
the previous set-point stays applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self._processed_actions = torch.zeros_like(self._raw_actions) + self._offset
        self._prev_processed = self._processed_actions.clone()
        self._lag = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._substep = 0

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        from supermimic.envs.dr import action_lag_substeps

        self._prev_processed = self._processed_actions.clone()
        self._lag = action_lag_substeps(self._env)
        self._substep = 0
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self) -> None:
        """Called every physics substep: the first ``lag`` substeps still hold the previous set-point."""
        held = (self._substep < self._lag)[:, None]
        processed = torch.where(held, self._prev_processed, self._processed_actions)
        self._substep += 1
        target = processed - self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0
        rest = torch.zeros_like(self._processed_actions) + self._offset
        self._processed_actions[env_ids] = rest[env_ids]
        self._prev_processed[env_ids] = rest[env_ids]


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)
