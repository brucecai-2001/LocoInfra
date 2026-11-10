"""Slot registry, E_tele masking and the keypoint warm start (SUPERMIMIC_ROBOT=q1)."""

from __future__ import annotations

import math

import pytest
import torch

from supermimic.robot import OBS_DIM, ROBOT_NAME

pytestmark = pytest.mark.skipif(ROBOT_NAME != "q1", reason="Q1 slot registry")


def test_layout():
    from supermimic.tele import slots as S

    assert S.SLOT_DIM == sum(s.dim for s in S.SLOTS) and S.GOAL_DIM == S.SLOT_DIM + S.NUM_SLOTS
    assert S.SLICE["base"] == slice(0, 3) and S.SLICE["feet"].stop == S.SLOT_DIM
    m = torch.tensor([[1.0, 0, 1, 0, 1, 0]])
    e = S.expand_mask(m)
    assert e.shape == (1, S.SLOT_DIM) and e[0, S.SLICE["height"]].sum() == 0 and e[0, S.SLICE["torso"]].sum() == 3


def test_torso_gravity_matches_quaternion():
    from mjlab.utils.lab_api.math import quat_from_euler_xyz

    from supermimic.kinematics.rotations import projected_gravity
    from supermimic.tele.slots import torso_gravity

    p, r = torch.tensor([0.4, -0.1, 0.0]), torch.tensor([0.0, 0.2, -0.25])
    q = quat_from_euler_xyz(r, p, torch.zeros(3))  # R = Rz Ry Rx
    assert torch.allclose(projected_gravity(q), torso_gravity(p, r), atol=1e-6)
    assert torso_gravity(torch.tensor(0.3), torch.tensor(0.0))[0] > 0  # forward lean: x > 0
    assert torso_gravity(torch.tensor(0.0), torch.tensor(0.3))[1] < 0  # right lean: y < 0


def test_masked_slot_has_no_effect():
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.tele import slots as S

    torch.manual_seed(0)
    pol = Policy(PolicyCfg(tele_hidden=(64, 64)))
    obs = torch.randn(4, 10 * OBS_DIM)
    g = torch.cat([S.default_vector().expand(4, -1), torch.ones(4, S.NUM_SLOTS)], dim=-1)
    g[:, S.SLOT_DIM + S.SLOT_NAMES.index("hand_L")] = 0.0
    g2 = g.clone()
    g2[:, S.SLICE["hand_L"]] += 5.0
    assert torch.allclose(pol.latent(obs, g, "tele"), pol.latent(obs, g2, "tele"))


def test_keypoint_warm_start_reproduces_keypoint_encoder():
    from supermimic.policy.nets import _KP_SCALE, HEAD_ABOVE_PELVIS, Policy, PolicyCfg, _obs_scale_vector, tele_from_keypoint
    from supermimic.tele import slots as S
    from supermimic.world_model.dynamics import MLP

    torch.manual_seed(0)
    cfg = PolicyCfg(tele_hidden=(64, 64))
    kp = MLP(13 + 10 * OBS_DIM, (64, 64), cfg.latent_dim)
    pol = Policy(cfg)
    tele_from_keypoint(pol.encoders["tele"], {f"mlp.{k}": v for k, v in kp.state_dict().items()})  # encoder-relative keys

    n = 8
    slots = S.default_vector().expand(n, -1).clone()
    slots[:, S.SLICE["base"]] = torch.randn(n, 3) * 0.3
    slots[:, S.SLICE["height"]] = -0.05
    slots[:, S.SLICE["torso"]] = S.torso_gravity(torch.rand(n) * 0.4, torch.rand(n) * 0.2 - 0.1)
    slots[:, S.SLICE["hand_L"]] += torch.randn(n, 3) * 0.05
    slots[:, S.SLICE["hand_R"]] += torch.randn(n, 3) * 0.05
    mask = (~S.reserved_mask()).float().expand(n, -1)
    obs = torch.randn(n, 10 * OBS_DIM)
    z_new = pol.latent(obs, torch.cat([slots, mask], dim=-1), "tele")

    head_h = slots[:, S.SLICE["height"]] + S.H_STAND + HEAD_ABOVE_PELVIS
    kp_goal = torch.cat([head_h, slots[:, S.SLICE["torso"]], slots[:, S.SLICE["hand_L"]], slots[:, S.SLICE["hand_R"]], slots[:, S.SLICE["base"]]], dim=-1)
    z_old = torch.tanh(kp(torch.cat([kp_goal / torch.tensor(_KP_SCALE), obs * _obs_scale_vector(cfg)], dim=-1)))
    assert torch.allclose(z_new, z_old, atol=1e-5), (z_new - z_old).abs().max()
    assert math.isfinite(float(z_new.detach().sum()))

    # Every slot masked == the keypoint encoder at the slot defaults (standing, natural hands, zero command).
    z_masked = pol.latent(obs, torch.cat([slots, torch.zeros_like(mask)], dim=-1), "tele")
    d = S.default_vector().expand(n, -1)
    kp_def = torch.cat([d[:, S.SLICE["height"]] + S.H_STAND + HEAD_ABOVE_PELVIS, d[:, S.SLICE["torso"]], d[:, S.SLICE["hand_L"]], d[:, S.SLICE["hand_R"]], d[:, S.SLICE["base"]]], dim=-1)
    z_def = torch.tanh(kp(torch.cat([kp_def / torch.tensor(_KP_SCALE), obs * _obs_scale_vector(cfg)], dim=-1)))
    assert torch.allclose(z_masked, z_def, atol=1e-5), (z_masked - z_def).abs().max()
