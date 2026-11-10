"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.tele.interface import TeleCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    anchor_ori_only: bool = False  # with anchor_obs: only the anchor orientation (6), no position (needs global localization on hardware)
    anchor_pos_noise: float = 0.25  # m, uniform half-width on the anchor-relative position (sim and imagination)
    physics_dr: bool = True  # mass / PD / torque limit / armature / joint friction / latency (envs/dr.py)


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations (ckpt_latest.pt)
    numbered_checkpoint_interval: int | None = None  # ckpt_<it>.pt every this many iterations; None: checkpoint_interval, 0: never
    max_hours: float | None = None
    # Deterministic whole-clip evaluation every `eval_interval_min` minutes (None: off); with
    # stop_on_pass, training stops at the first evaluation with 0 falls and return >= stop_return.
    eval_interval_min: float | None = None
    stop_on_pass: bool = True
    stop_return: float = 0.80
    # With --resume: run the cosine schedule from the resumed iteration to total_iterations (new LR cycle).
    lr_from_resume: bool = False


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    reward: TrackingRewardCfg = field(default_factory=TrackingRewardCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    tele: TeleCfg = field(default_factory=TeleCfg)
    resume: Path | None = None
