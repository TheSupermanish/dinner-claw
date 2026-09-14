"""Contact-gap IK for the five-axis SO-101; scratch state only."""

from dataclasses import dataclass

import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Fixed finger inner face is x=-7.9 mm. Center of a 50 mm object is x=17.1 mm.
# Derived from the pinned collision mesh vertices, not the upstream tip marker.
GRASP_POINT = np.array([0.0171, -0.000218, -0.090])


@dataclass
class Solution:
    targets: np.ndarray
    position_error: float
    axis_error: float

    @property
    def accepted(self):
        return self.position_error < 0.003 and self.axis_error < 0.04


def solve_down(model, data, arm, target, initial=None):
    """Point the fingers down while placing the contact gap at target."""
    if arm not in {"left", "right"}:
        raise ValueError("Unknown arm")
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Expected finite xyz")
    joints = [model.joint(f"{arm}_{name}") for name in JOINTS]
    qids = [int(j.qposadr[0]) for j in joints]
    vids = [int(j.dofadr[0]) for j in joints]
    bid = model.body(f"{arm}_gripper").id
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    starts = [data.qpos[qids] if initial is None else np.asarray(initial)]
    starts += [np.array([0, lift, elbow, 0, 0.0])
               for lift, elbow in ((-0.5, 1), (0.5, -1), (1, -1), (-1, 1))]
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
            ae = np.array([0.0, 0.0, -1.0]) - axis
            if np.linalg.norm(pe) < 0.0005 and np.linalg.norm(ae) < 0.005:
                break
            mujoco.mj_jac(model, scratch, jp, jr, point, bid)
            ja = np.cross(jr[:, vids].T, axis).T
            closing_jac = np.cross(jr[:, vids].T, rot[:, 0]).T
            jac = np.vstack([jp[:, vids], 0.12 * ja, 0.04 * closing_jac[0]])
            error = np.r_[pe, 0.12 * ae, -0.04 * rot[0, 0]]
            dq = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(5), jac.T @ error)
            for k, joint in enumerate(joints):
                scratch.qpos[qids[k]] = np.clip(
                    scratch.qpos[qids[k]] + np.clip(dq[k], -0.12, 0.12), *joint.range
                )
        mujoco.mj_kinematics(model, scratch)
        rot = scratch.xmat[bid].reshape(3, 3)
        pos_error = float(np.linalg.norm(target - scratch.xpos[bid] - rot @ GRASP_POINT))
        axis_error = float(np.linalg.norm(-rot[:, 2] - [0, 0, -1]))
        candidate = Solution(scratch.qpos[qids].copy(), pos_error, axis_error)
        if best is None or pos_error + 0.1 * axis_error < best.position_error + 0.1 * best.axis_error:
            best = candidate
        if candidate.accepted:
            break
    return best
