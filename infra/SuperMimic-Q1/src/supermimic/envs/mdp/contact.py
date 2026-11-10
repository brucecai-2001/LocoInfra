"""Body-to-floor clearance from collision-shell vertices, and the two contact penalties built on it.

The world model has no geometry, so contact is expressed through the tracked bodies' poses: each body
carries a fixed set of points on its convex collision shell (body frame); its clearance is the lowest of
those points after rotating with the body quaternion. The lowest point of a convex shell is a vertex, so
with all hull vertices this is exact; meshes are down-sampled (farthest-point) to ``max_verts`` points,
boxes contribute their 8 corners. One implementation serves the simulator rewards and the imagination.

  extra_contact  sum_b gate_b * (relu(margin - c_b) / margin)^2
                 gate_b = clamp((c_ref_b - ref_free) / ref_free_band, 0, 1): only bodies that are clearly
                 off the floor in the reference (e.g. a knee 7 cm up) are penalised for touching it.
  impact         sum_b near_b * relu(-v_b - impact_speed)^2,  near_b = clamp(1 - c_b / impact_band, 0, 1)
                 v_b: vertical velocity of the body's lowest point (origin velocity + w x r).
Feet (ankle links) are excluded from both.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

EXCLUDED_BODY_SUBSTR = ("ankle",)


def _farthest_points(p: np.ndarray, k: int) -> np.ndarray:
    if len(p) <= k:
        return p
    idx = [int(np.argmin(p[:, 2]))]
    d = np.linalg.norm(p - p[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(p - p[i], axis=1))
    return p[idx]


def contact_vertex_table(model: mujoco.MjModel, body_names: tuple[str, ...], max_verts: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """-> verts (B, V, 3) in each body frame (padded by repetition), mask (B,) of bodies that are checked."""
    from scipy.spatial import ConvexHull

    per_body = []
    for name in body_names:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            R = R.reshape(3, 3)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                v = v[ConvexHull(v).vertices]
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                s = model.geom_size[g]
                v = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
            else:
                continue
            pts.append(model.geom_pos[g] + v @ R.T)
        per_body.append(np.concatenate(pts) if pts else np.zeros((1, 3)))
    mask = [bool(len(p) > 1) and not any(s in n for s in EXCLUDED_BODY_SUBSTR) for p, n in zip(per_body, body_names)]
    per_body = [_farthest_points(p, max_verts) for p in per_body]
    V = max(len(p) for p in per_body)
    verts = np.stack([np.concatenate([p, np.repeat(p[:1], V - len(p), 0)]) for p in per_body])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(mask)


def _rot(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion (…, 4) -> rotation matrix (…, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def clearance(pos: torch.Tensor, quat: torch.Tensor, verts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest shell point of each body: height above z = 0 (…, B) and its world offset from the body origin (…, B, 3)."""
    R = _rot(quat)  # (…, B, 3, 3)
    r = torch.einsum("...bij,bvj->...bvi", R, verts)  # (…, B, V, 3)
    i = r[..., 2].argmin(-1, keepdim=True)  # (…, B, 1)
    r_low = torch.gather(r, -2, i[..., None].expand(*i.shape, 3)).squeeze(-2)
    return pos[..., 2] + r_low[..., 2], r_low


def contact_penalties(
    robot_pos: torch.Tensor, robot_quat: torch.Tensor, robot_lin_vel: torch.Tensor, robot_ang_vel: torch.Tensor,
    ref_pos: torch.Tensor, ref_quat: torch.Tensor, verts: torch.Tensor, mask: torch.Tensor, cfg,
) -> dict[str, torch.Tensor]:
    """Unweighted penalties (N,) for world-frame tracked-body states of the robot and the reference."""
    c, r_low = clearance(robot_pos, robot_quat, verts)
    c_ref, _ = clearance(ref_pos, ref_quat, verts)
    m = mask.to(c.dtype)
    gate = ((c_ref - cfg.contact_ref_free) / cfg.contact_ref_free_band).clamp(0.0, 1.0)
    extra = (gate * ((cfg.contact_margin - c).clamp(min=0.0) / cfg.contact_margin) ** 2 * m).sum(-1)
    v_low = robot_lin_vel[..., 2] + torch.cross(robot_ang_vel, r_low, dim=-1)[..., 2]
    near = (1.0 - c / cfg.impact_band).clamp(0.0, 1.0)
    impact = (near * (-v_low - cfg.impact_speed).clamp(min=0.0) ** 2 * m).sum(-1)
    return {"extra_contact": extra, "impact": impact}
