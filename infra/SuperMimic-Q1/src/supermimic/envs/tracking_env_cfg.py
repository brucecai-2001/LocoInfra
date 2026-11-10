"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2, hand orientation down-weighted,
    torque / power / over-rated-torque penalties;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
    physics scales and latencies of ``envs/dr.py`` (part of the sim context xi);
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs import dr as sm_dr
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import env_rewards as sm_rew
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp.rewards import TrackingRewardCfg, body_ori_weights
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.robot import SPEC
from supermimic.tele.interface import TeleCfg

class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


TRACKED_BODIES = SPEC.tracked_bodies
ANCHOR_BODY = SPEC.anchor_body
ROOT_BODY = SPEC.root_body
FOOT_GEOM_REGEX = SPEC.foot_geom_regex


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = SPEC.root_height_threshold
    root_tilt_threshold: float = SPEC.root_tilt_threshold
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    anchor_ori_only: bool = False  # with anchor_obs: orientation only (6)
    anchor_pos_noise: float = 0.25  # m, uniform half-width on the anchor-relative position
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)
    tele: TeleCfg | None = None  # teleoperation envs (supermimic/tele/interface.py)
    # Physics randomization on top of the paper's (envs/dr.py); active with domain_randomization.
    physics_dr: bool = True
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    kp_scale_range: tuple[float, float] = (0.8, 1.2)
    kd_scale_range: tuple[float, float] = (0.8, 1.2)
    effort_scale_range: tuple[float, float] = (0.8, 1.2)
    armature_scale_range: tuple[float, float] = (0.8, 1.2)
    joint_friction_range: tuple[float, float] = (0.0, 0.3)  # N·m (the model has none)
    action_delay_s: tuple[float, float] = (0.0, 0.02)
    obs_delay_s: tuple[float, float] = (0.0, 0.02)
    # Paper randomization ranges (robustness tests pin them to nominal values).
    foot_friction_range: tuple[float, float] = (0.3, 1.2)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    com_ranges: dict | None = None  # None: SPEC.com_ranges
    reward_cfg: TrackingRewardCfg = field(default_factory=TrackingRewardCfg)
    # Simulator settings (evaluation re-runs with other solver parameters; step_dt stays 0.02 s).
    sim_timestep: float = 0.005
    decimation: int = 4
    mujoco_overrides: dict = field(default_factory=dict)  # extra MujocoCfg fields (solver, cone, impratio, ...)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {"anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-a.anchor_pos_noise, a.anchor_pos_noise))}
                    if a.anchor_obs and not a.anchor_ori_only
                    else {}
                ),
                **(
                    {"anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05))}
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=SPEC.action_scale(),
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    cmd_kw = dict(a.extra_command_kwargs)
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            tele=a.tele,
            **cmd_kw,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(SPEC.com_body,)),
                "operation": "add",
                "ranges": SPEC.com_ranges if a.com_ranges is None else a.com_ranges,
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": a.encoder_bias_range},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": a.foot_friction_range,
                "shared_random": True,
            },
        )
        if a.physics_dr:
            events["mass_scale"] = EventTermCfg(mode="startup", func=sm_dr.mass_scale, params={"scale_range": a.mass_scale_range})
            events["armature_scale"] = EventTermCfg(mode="startup", func=sm_dr.armature_scale, params={"scale_range": a.armature_scale_range})
            events["actuator_scale"] = EventTermCfg(
                mode="reset",
                func=sm_dr.actuator_scale,
                params={"kp_range": a.kp_scale_range, "kd_range": a.kd_scale_range, "effort_range": a.effort_scale_range},
            )
            events["joint_friction"] = EventTermCfg(mode="reset", func=sm_dr.joint_friction, params={"friction_range": a.joint_friction_range})
            events["latency"] = EventTermCfg(mode="reset", func=sm_dr.latency, params={"action_delay_s": a.action_delay_s, "obs_delay_s": a.obs_delay_s})
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2 + effort terms)
    rc = a.reward_cfg
    ori_w = tuple(body_ori_weights(TRACKED_BODIES, SPEC.hand_bodies, rc.hand_ori_weight).tolist())
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=rc.w_root_pos, params={"command_name": "motion", "std": rc.std_root_pos}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=rc.w_root_ori, params={"command_name": "motion", "std": rc.std_root_ori}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=rc.w_body_pos, params={"command_name": "motion", "std": rc.std_body_pos}),
        "motion_body_ori": RewardTermCfg(func=sm_rew.motion_relative_body_orientation_error_exp_weighted, weight=rc.w_body_ori, params={"command_name": "motion", "std": rc.std_body_ori, "weights": ori_w}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=rc.w_body_lin_vel, params={"command_name": "motion", "std": rc.std_body_lin_vel}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=rc.w_body_ang_vel, params={"command_name": "motion", "std": rc.std_body_ang_vel}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=rc.w_action_rate),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "torque": RewardTermCfg(func=sm_rew.joint_torque, weight=rc.w_torque),
        "power": RewardTermCfg(func=sm_rew.joint_power, weight=rc.w_power),
        "over_rated": RewardTermCfg(func=sm_rew.torque_over_rated, weight=rc.w_over_rated),
    }
    if rc.w_extra_contact:
        rewards["extra_contact"] = RewardTermCfg(func=sm_rew.extra_contact, weight=rc.w_extra_contact, params={"command_name": "motion", "cfg": rc})
    if rc.w_impact:
        rewards["impact"] = RewardTermCfg(func=sm_rew.contact_impact, weight=rc.w_impact, params={"command_name": "motion", "cfg": rc})

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }
    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": SPEC.com_body})

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=a.num_envs, entities={"robot": SPEC.robot_cfg()},
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=SPEC.anchor_body, distance=SPEC.viewer_distance, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(
            nconmax=SPEC.nconmax, njmax=SPEC.njmax,
            mujoco=MujocoCfg(**{"timestep": a.sim_timestep, "iterations": 10, "ls_iterations": 20, **a.mujoco_overrides}),
        ),
        decimation=a.decimation,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
