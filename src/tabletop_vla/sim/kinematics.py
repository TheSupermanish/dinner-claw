"""Contact-gap IK for the five-axis SO-101; scratch state only."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Fixed finger inner face is x=-7.9 mm. Center of a 50 mm object is x=17.1 mm.
# Derived from the pinned collision mesh vertices, not the upstream tip marker.
GRASP_POINT = np.array([0.0171, -0.000218, -0.090])

# Deterministic extra restarts: fixed seed, precomputed unit cube, scaled per
# call into the arm's joint ranges. Appended AFTER the 5 legacy starts so any
# episode whose IK already accepted within the first 5 is numerically
# unaffected (the loop breaks before reaching the extras).
RESTART_SEED = 12345
N_EXTRA_STARTS = 16
_EXTRA_UNIT: np.ndarray | None = None


@dataclass
class Solution:
    targets: np.ndarray
    position_error: float
    axis_error: float
    achieved: np.ndarray | None = field(default=None)

    @property
    def accepted(self):
        return self.position_error < 0.003 and self.axis_error < 0.04


@dataclass(frozen=True)
class GripperGeometry:
    """Measured world-space geometry of the two collision-pad inner faces."""

    midpoint: np.ndarray
    closing_axis: np.ndarray
    separation: float
    fixed_face: np.ndarray
    moving_face: np.ndarray


def gripper_geometry(model, data, arm: str) -> GripperGeometry:
    """Measure the current jaw gap from MuJoCo collision boxes.

    The SO-101 has one stationary pad and one articulated pad, so the useful
    grasp point changes as the gripper moves. ``closing_axis`` points from the
    stationary pad toward the moving pad. This function reads the live physics
    state and never changes it.
    """
    if arm not in {"left", "right"}:
        raise ValueError("Unknown arm")
    fixed_id = model.geom(f"{arm}_fixed_pad").id
    moving_id = model.geom(f"{arm}_moving_pad").id
    centers = data.geom_xpos[[fixed_id, moving_id]].copy()
    delta = centers[1] - centers[0]
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        raise ValueError("Jaw collision-pad centers overlap")
    axis = delta / distance

    def support_radius(geom_id: int) -> float:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        return float(np.abs(rotation.T @ axis) @ model.geom_size[geom_id])

    fixed_face = centers[0] + support_radius(fixed_id) * axis
    moving_face = centers[1] - support_radius(moving_id) * axis
    separation = float(np.dot(moving_face - fixed_face, axis))
    return GripperGeometry(
        midpoint=(fixed_face + moving_face) / 2,
        closing_axis=axis,
        separation=separation,
        fixed_face=fixed_face,
        moving_face=moving_face,
    )


def _extra_units() -> np.ndarray:
    global _EXTRA_UNIT
    if _EXTRA_UNIT is None:
        _EXTRA_UNIT = np.random.default_rng(RESTART_SEED).uniform(0.0, 1.0, (N_EXTRA_STARTS, 5))
    return _EXTRA_UNIT


def _as_unit(vec, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite xyz for {name}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"Expected non-zero {name}")
    return arr / norm


def solve_pose(model, data, arm, target, approach=(0, 0, -1), closing=None, initial=None):
    """Point the fingers along ``approach`` while placing the contact gap at target.

    ``approach`` is the desired unit axis of the achieved finger direction
    (``-gripper_z`` in world). ``closing`` optionally gives the desired unit
    axis of the gripper closing direction (gripper local x in world); when it
    is None the legacy single-component redundancy regularization is kept
    exactly. ``Solution.achieved`` reports the achieved approach axis so a
    caller can see what it actually got. Extra restarts are deterministic
    (fixed seed) so repeated runs reproduce exactly.
    """
    if arm not in {"left", "right"}:
        raise ValueError("Unknown arm")
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Expected finite xyz")
    approach_u = _as_unit(approach, "approach")
    closing_u = None if closing is None else _as_unit(closing, "closing")
    joints = [model.joint(f"{arm}_{name}") for name in JOINTS]
    qids = [int(j.qposadr[0]) for j in joints]
    vids = [int(j.dofadr[0]) for j in joints]
    bid = model.body(f"{arm}_gripper").id
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    starts = [data.qpos[qids] if initial is None else np.asarray(initial)]
    starts += [np.array([0, lift, elbow, 0, 0.0])
               for lift, elbow in ((-0.5, 1), (0.5, -1), (1, -1), (-1, 1))]
    lows = np.array([j.range[0] for j in joints], dtype=float)
    highs = np.array([j.range[1] for j in joints], dtype=float)
    units = _extra_units()
    for row in units:
        starts.append(lows + row * (highs - lows))
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    best = None
    for start in starts:
        scratch.qpos[qids] = start
        for _ in range(180):
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            rot = scratch.xmat[bid].reshape(3, 3)
            point = scratch.xpos[bid] + rot @ GRASP_POINT
            axis = -rot[:, 2]
            pe = target - point
            ae = approach_u - axis
            if np.linalg.norm(pe) < 0.0005 and np.linalg.norm(ae) < 0.005:
                break
            mujoco.mj_jac(model, scratch, jp, jr, point, bid)
            ja = np.cross(jr[:, vids].T, axis).T
            closing_jac = np.cross(jr[:, vids].T, rot[:, 0]).T
            if closing_u is None:
                jac = np.vstack([jp[:, vids], 0.12 * ja, 0.04 * closing_jac[0]])
                error = np.r_[pe, 0.12 * ae, -0.04 * rot[0, 0]]
            else:
                cerr = closing_u - rot[:, 0]
                jac = np.vstack([jp[:, vids], 0.12 * ja, 0.04 * closing_jac])
                error = np.r_[pe, 0.12 * ae, 0.04 * cerr]
            dq = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(5), jac.T @ error)
            for k, joint in enumerate(joints):
                scratch.qpos[qids[k]] = np.clip(
                    scratch.qpos[qids[k]] + np.clip(dq[k], -0.12, 0.12), *joint.range
                )
        mujoco.mj_kinematics(model, scratch)
        rot = scratch.xmat[bid].reshape(3, 3)
        achieved = -rot[:, 2].copy()
        pos_error = float(np.linalg.norm(target - scratch.xpos[bid] - rot @ GRASP_POINT))
        axis_error = float(np.linalg.norm(achieved - approach_u))
        candidate = Solution(scratch.qpos[qids].copy(), pos_error, axis_error, achieved)
        if best is None or pos_error + 0.1 * axis_error < best.position_error + 0.1 * best.axis_error:
            best = candidate
        if candidate.accepted:
            break
    return best


def solve_down(model, data, arm, target, initial=None):
    """Point the fingers down while placing the contact gap at target."""
    return solve_pose(model, data, arm, target, approach=(0, 0, -1), closing=None, initial=initial)
