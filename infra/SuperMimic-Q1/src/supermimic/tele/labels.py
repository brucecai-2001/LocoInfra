"""Reference -> teleoperation labels (docs/training.md), shared by every data source.

For every frame of a motion library:
    slots     (F, SLOT_DIM)  what an operator would have commanded: ``base`` is the library's
                             reference command (25-frame centred mean), ``height`` / ``torso`` are
                             smoothed the same way, hands / feet are the instantaneous FK
    tele_ok   (F,)           the frame can be reproduced from the slots alone (no feet target): false
                             near high feet (kicks), flight phases, crossed feet, fast motion and in
                             dance clips; such frames stay in the E_ref batches only
Per 1 s sampler bin:
    tele_ok_bin  (num_bins,) all frames of the bin are tele_ok
    category     (num_bins,) bit set of the active command channels (CATEGORY_BITS), used to balance
                             the E_tele starts over the command space instead of over clip time
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.tele import slots as S

CATEGORY_BITS = ("walk", "turn", "squat", "lean", "hands")


@dataclass
class TeleLabelCfg:
    window: int = 25  # frames of the centred mean for height / torso (0.5 s, as the base command)
    dilate: int = 25  # frames a rejected frame spreads to on each side
    high_foot: float = 0.12  # m above the clip's ground level
    flight_height: float = 0.03  # m, both feet above it ...
    flight_frames: int = 5  # ... for this many frames
    fast_speed: float = 1.2  # m/s planar
    fast_yaw: float = 2.0  # rad/s
    cross_margin: float = 0.0  # m, left foot y - right foot y below it (heading frame) = crossed
    reject_names: str = r"(?i)dance"
    # Category thresholds (bin means).
    walk_speed: float = 0.15
    turn_rate: float = 0.3
    squat_dh: float = -0.05
    lean_angle: float = 0.2
    hand_offset: float = 0.12


def _clip_mean(lib: MotionLibrary, x: torch.Tensor, window: int) -> torch.Tensor:
    """Centred ``window``-frame mean of x (F, D) inside each clip (clipped at the clip ends)."""
    xd = x.double()
    csum = torch.cat([torch.zeros(1, x.shape[1], dtype=xd.dtype, device=x.device), torch.cumsum(xd, 0)], 0)
    f = torch.arange(lib.num_frames, device=x.device)
    clip = lib.frame_clip
    lo = torch.maximum(f - window // 2, lib.clip_start[clip])
    hi = torch.minimum(f + window // 2, lib.clip_last[clip]) + 1
    return ((csum[hi] - csum[lo]) / (hi - lo)[:, None].to(xd.dtype)).to(x.dtype)


def _dilate(bad: torch.Tensor, lib: MotionLibrary, r: int) -> torch.Tensor:
    """Spread rejected frames by r frames on each side, without crossing clip boundaries."""
    c = torch.cat([torch.zeros(1, device=bad.device), torch.cumsum(bad.float(), 0)])
    f = torch.arange(lib.num_frames, device=bad.device)
    clip = lib.frame_clip
    lo = torch.maximum(f - r, lib.clip_start[clip])
    hi = torch.minimum(f + r, lib.clip_last[clip]) + 1
    return (c[hi] - c[lo]) > 0


@torch.no_grad()
def compute_labels(lib: MotionLibrary, cfg: TeleLabelCfg | None = None, chunk: int = 1 << 19) -> dict[str, torch.Tensor]:
    cfg = cfg or TeleLabelCfg()
    names = lib.body_names
    ti = names.index("torso_link")
    hi_ = [names.index(b) for b in S.HAND_BODIES]
    fi = [names.index(b) for b in S.FOOT_BODIES]
    dev = lib.device
    F = lib.num_frames
    pose = {k: [] for k in ("height", "torso", "hand_L", "hand_R", "feet")}
    foot_z = []
    for s in range(0, F, chunk):
        fr = torch.arange(s, min(s + chunk, F), device=dev)
        r, b = lib.ref(fr), lib.bodies(fr)
        p = S.pose_slots(r.root_pos_w, b.pos_w[:, ti], b.quat_w[:, ti], b.pos_w[:, hi_], b.pos_w[:, fi])
        for k in pose:
            pose[k].append(p[k])
        foot_z.append(b.pos_w[:, fi, 2])
    pose = {k: torch.cat(v, 0) for k, v in pose.items()}
    foot_z = torch.cat(foot_z, 0)  # (F, 2)

    torso = _clip_mean(lib, pose["torso"], cfg.window)
    torso = torso / torso.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    parts = {
        "base": lib.ref_cmd.float(),
        "height": _clip_mean(lib, pose["height"], cfg.window),
        "torso": torso,
        "hand_L": pose["hand_L"],
        "hand_R": pose["hand_R"],
        "feet": pose["feet"],
    }
    slots = S.assemble(parts)

    # Ground level of each clip: 5th percentile of the lower ankle height.
    low = foot_z.min(-1).values
    ground = torch.empty(lib.num_clips, device=dev)
    for c in range(lib.num_clips):
        st, ln = int(lib.clip_start[c]), int(lib.clip_len[c])
        ground[c] = torch.quantile(low[st : st + ln], 0.05)
    lift = foot_z - ground[lib.frame_clip][:, None]
    high = lift.max(-1).values > cfg.high_foot
    air = (lift.min(-1).values > cfg.flight_height).float()
    k = cfg.flight_frames
    run = torch.nn.functional.avg_pool1d(air[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0, :F]
    flight = _dilate(run > 1 - 1e-6, lib, k // 2)
    cmd = parts["base"]
    fast = (cmd[:, :2].norm(dim=-1) > cfg.fast_speed) | (cmd[:, 2].abs() > cfg.fast_yaw)
    crossed = (parts["feet"][:, 1] - parts["feet"][:, 4]) < cfg.cross_margin
    rx = re.compile(cfg.reject_names)
    clip_bad = torch.tensor([bool(rx.search(c.name)) for c in lib.clips], device=dev)
    bad = high | flight | fast | crossed | clip_bad[lib.frame_clip]
    tele_ok = ~_dilate(bad, lib, cfg.dilate)

    # Per-bin aggregation.
    fb = lib.bin_of_frame(torch.arange(F, device=dev))
    cnt = torch.zeros(lib.num_bins, device=dev).index_add_(0, fb, torch.ones(F, device=dev))
    ok_bin = torch.zeros(lib.num_bins, device=dev).index_add_(0, fb, tele_ok.float()) >= cnt - 0.5
    mean = torch.zeros(lib.num_bins, slots.shape[1], device=dev).index_add_(0, fb, slots) / cnt.clamp_min(1)[:, None]
    m = {n: mean[:, S.SLICE[n]] for n in S.SLOT_NAMES}
    dflt = S.default_vector(dev)
    tilt = torch.acos((-m["torso"][:, 2] / m["torso"].norm(dim=-1).clamp_min(1e-6)).clamp(-1, 1))
    hand_dev = torch.maximum(
        (m["hand_L"] - dflt[S.SLICE["hand_L"]]).norm(dim=-1), (m["hand_R"] - dflt[S.SLICE["hand_R"]]).norm(dim=-1)
    )
    bits = (
        (m["base"][:, :2].norm(dim=-1) > cfg.walk_speed),
        (m["base"][:, 2].abs() > cfg.turn_rate),
        (m["height"][:, 0] < cfg.squat_dh),
        (tilt > cfg.lean_angle),
        (hand_dev > cfg.hand_offset),
    )
    category = sum(b.long() << i for i, b in enumerate(bits))
    reasons = {"high_foot": high, "flight": flight, "fast": fast, "crossed": crossed, "clip_name": clip_bad[lib.frame_clip]}
    return {"slots": slots, "tele_ok": tele_ok, "tele_ok_bin": ok_bin, "category": category, "reasons": reasons}


def attach(lib: MotionLibrary, cfg: TeleLabelCfg | None = None) -> None:
    """Compute the labels once and keep them on the library (``lib.tele_slots`` …)."""
    out = compute_labels(lib, cfg)
    lib.tele_slots = out["slots"]
    lib.tele_ok = out["tele_ok"]
    lib.tele_ok_bin = out["tele_ok_bin"]
    lib.tele_category = out["category"]


def category_name(c: int) -> str:
    on = [n for i, n in enumerate(CATEGORY_BITS) if (c >> i) & 1]
    return "+".join(on) if on else "stand"
