"""Test clips of the selected robot: G1 uses the LAFAN1 retargets in data/npz/lafan1, Q1 the walking set
in data/npz/q1_walk (names prefixed ``lafan_``)."""

from __future__ import annotations

from pathlib import Path

from supermimic.robot import ROBOT_NAME

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / ("data/npz/lafan1" if ROBOT_NAME == "g1" else "data/npz/q1_walk")


def clips(names: list[str]) -> list[Path]:
    """The named clips, or (when this robot's set lacks them) as many of its clips as were asked for."""
    from supermimic.envs.mdp.motion_library import discover_npz

    prefix = "" if ROBOT_NAME == "g1" else "lafan_"
    files = discover_npz(NPZ, names=[prefix + n for n in names])
    rest = [f for f in discover_npz(NPZ) if f not in files]
    return (files + rest)[: len(names)]
