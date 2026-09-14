"""Left-arm physical plate placement: best-effort rim grasp with strict gates.

Shape mirrors ``cutlery.py`` (hover, descend, close, lift, carry, release, verify;
closed-loop jaw correction; bilateral-contact, sustained-lift and stable-release
gates; open-gripper negative control). The workcell parks the plate BEFORE the
episode and is declared in the result; nothing is welded, teleported or moved
during execution.

Measured 2026-09-14 (probes /tmp/probe_plate.py, /tmp/probe2.py, /tmp/probe3.py,
/tmp/probe4.py): the 200 mm disc cannot be spanned by the 56 mm jaw opening, and
the only candidate rim-thickness pinch needs a vertical closing axis that the IK
rejects everywhere (axis error 0.136 > 0.04 gate). Every IK-accepted
horizontal-closing pose drives the 32 mm-tall fixed pad into the plate side and
the table at 63-177 N and shoves the plate 19-149 mm without ever achieving
bilateral finger contact or lift. This teacher therefore FAILS honestly at its
grasp/lift gates; see outputs/plate-place-40-49.json. That 0/10 with the numbers
is the result the brief asked for if the plate cannot be grasped.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.kinematics import JOINTS, gripper_geometry, solve_pose
from tabletop_vla.sim.reach import _cartesian

# Most IK-tolerant plate start found by the probe sweep (seed 30): every
# approach/height variant accepts here, unlike the table centre which the left
# arm cannot reach at all. Still distinct from plate_mat [-0.20, 0.12].
PLATE_START = np.array([-0.28, 0.05, 0.752])
PLATE_JITTER = 0.006
# Plate rim radius is 0.10 m; aim 10 mm inside the east edge at top-face level.
RIM_DX = 0.09
RIM_DZ = 0.008
MAT = np.array([-0.20, 0.12, 0.738])
GRASP_TILT, GRASP_AZ = 20, 98
MAT_TILT, MAT_AZ = 30, 106
MAT_PHASES = frozenset({"Carry to mat", "Descend to mat", "Release", "Retract",
                        "Verify placement"})
ALIGNED_PHASES = frozenset({"Hover above rim", "Descend to rim", "Close on rim",
                            "Carry to mat", "Descend to mat", "Release"})
OPEN_GRIP = 0.85


def prepare_plate_workcell(sim, seed):
    """Park the plate at a reachable start BEFORE the episode, never during motion."""
    rng = np.random.default_rng(seed + 8100000)
    start = PLATE_START.copy()
    start[:2] += rng.uniform(-PLATE_JITTER, PLATE_JITTER, 2)
    address = sim.model.jnt_qposadr[sim.model.body_jntadr[sim.model.body("plate").id]]
    sim.data.qpos[address:address + 3] = start
    sim.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    sim.variation["plate_workcell"] = {"plate_start": start.tolist()}


def _waypoint(model, data, arm, target, tilt, az, initial, closing=None):
    """Nearby approach angles are tried before giving up; chaining `initial` keeps
    the arm on one IK branch so the joint-space path between waypoints stays short."""
    for delta_t in (0, -10, 10, -20, 20):
        for delta_a in (0, -18, 18, -36, 36):
            tilt_candidate = tilt + delta_t
            if not 0 <= tilt_candidate <= 80:
                continue
            solution = solve_pose(model, data, arm, target,
                                  approach=_cartesian(tilt_candidate, az + delta_a),
                                  closing=closing, initial=initial)
            if solution.accepted:
                return solution
    return solve_pose(model, data, arm, target, approach=_cartesian(tilt, az),
                      closing=closing, initial=initial)


def _jaw_correct(model, data, arm, target, tilt, az, initial, closing, grip_q):
    """Solve, measure where the live jaw gap actually lands, then re-solve once.

    `solve_pose` aims the static GRASP_POINT constant, but the SO-101 has one fixed
    and one moving pad, so the real jaw midpoint moves with the opening (about
    11 mm at the approach opening, measured 2026-09-14). Same idea as the cutlery
    teacher; written out here because that helper is private to its module.
    """
    target = np.asarray(target, dtype=float)
    solution = _waypoint(model, data, arm, target, tilt, az, initial, closing)
    if not solution.accepted:
        return solution
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for name, value in zip(JOINTS, solution.targets, strict=True):
        scratch.qpos[int(model.joint(f"{arm}_{name}").qposadr[0])] = value
    scratch.qpos[int(model.joint(f"{arm}_gripper").qposadr[0])] = grip_q
    mujoco.mj_forward(model, scratch)
    delta = target - gripper_geometry(model, scratch, arm).midpoint
    if float(np.linalg.norm(delta)) < 1e-4:
        return solution
    corrected = _waypoint(model, data, arm, target + delta, tilt, az, initial, closing)
    return corrected if corrected.accepted else solution


class PlatePlace:
    """Best-effort left-arm plate rim grasp, carry to plate_mat, and release."""

    def __init__(self, sim, close_gripper=True):
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.close_gripper = close_gripper
        self.status, self.reason = "running", None
        self.started = float(self.data.time)
        self.phase, self.phase_started = -1, self.started
        self.body = self.model.body("plate").id
        self.plate_geom = self.model.geom("plate_geom").id
        self.fingers = {self.model.body("left_gripper").id,
                        self.model.body("left_moving_jaw_so101_v1").id}
        self.actuators = [self.model.actuator(f"left_{j}").id for j in JOINTS]
        self.grip = self.model.actuator("left_gripper").id
        self.grip_q = int(self.model.joint("left_gripper").qposadr[0])
        # Table top plus plate half-height: the plate rests at 0.743 m.
        self.rest_z = 0.735 + float(self.model.geom_size[self.plate_geom][1])
        self.mat = MAT.copy()
        self.start = self.data.xpos[self.body].copy()
        self.solution = None
        self.bilateral_s = self.current_contact = 0.0
        self.peak_lift = self.lift_run = self.sustained_lift = self.stable_release = 0.0
        self.events = []
        self.phases = (
            ("Settle", 0.6),
            ("Hover above rim", 1.5),
            ("Descend to rim", 2.0),
            ("Close on rim", 3.0),
            ("Lift clear of table", 2.0),
            ("Raise to transit height", 1.6),
            ("Carry to mat", 2.2),
            ("Descend to mat", 1.8),
            ("Release", 2.0),
            ("Retract", 2.0),
            ("Verify placement", 1.4),
        )

    # ---- shared helpers -------------------------------------------------

    def contacts(self):
        """Bodies pressing on the plate with a force-bearing contact."""
        bodies = set()
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.body in (a, b):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if force[0] > 0.02:
                    bodies.add(int(b if a == self.body else a))
        return bodies

    def fail(self, reason):
        self.status, self.reason = "failed", reason
        self.sim.running = False

    def cancel(self):
        self.status, self.reason = "cancelled", "Manual control or another task"

    @property
    def lift(self):
        return float(self.data.xpos[self.body][2] - self.rest_z)

    # ---- placement phases -----------------------------------------------

    def rim(self):
        """Where the jaw gap should sit at the plate rim, in world coordinates."""
        plate = self.data.xpos[self.body].copy()
        return np.array([plate[0] + RIM_DX, plate[1], plate[2] + RIM_DZ])

    def target_for(self, name):
        rim = self.rim()
        if name == "Hover above rim":
            return rim + [0, 0, 0.04]
        if name in {"Descend to rim", "Close on rim"}:
            return rim + [0, 0, -0.002]
        if name == "Lift clear of table":
            return rim + [0, 0, 0.06]
        if name == "Raise to transit height":
            return np.array([rim[0], rim[1], 0.86])
        if name == "Carry to mat":
            aim = self.mat + self.body_to_rim()
            return np.array([aim[0], aim[1], 0.86])
        if name in {"Descend to mat", "Release"}:
            return self.mat + self.body_to_rim() + [0, 0, 0.012]
        if name == "Retract":
            return np.array([self.mat[0], self.mat[1], 0.86])
        return None

    def body_to_rim(self):
        """The jaws hold the rim, but the placement gate scores the body centre,
        which sits about 90 mm away. Aim the rim so the BODY lands on the mat."""
        return np.array([RIM_DX, 0.0, 0.0])

    def axis_for(self, name):
        """Preferred approach angle per phase; `_waypoint` searches around it."""
        return (MAT_TILT, MAT_AZ) if name in MAT_PHASES else (GRASP_TILT, GRASP_AZ)

    def gripping(self, name):
        return self.close_gripper and name in {
            "Close on rim", "Lift clear of table", "Raise to transit height",
            "Carry to mat", "Descend to mat"}

    def enter(self, phase):
        self.phase, self.phase_started = phase, float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        name = self.phases[phase][0]
        self.events.append({"phase": name, "time_s": self.phase_started})
        if name == "Settle":
            self.end[self.actuators[0]] = OPEN_GRIP
            self.end[self.grip] = OPEN_GRIP
            return
        target = self.target_for(name)
        if target is None:
            return
        # Jaws must close across the rim radially, not along its tangent.
        closing = (1.0, 0.0, 0.0) if name in ALIGNED_PHASES else None
        tilt, az = self.axis_for(name)
        if closing is None:
            solution = _waypoint(self.model, self.data, "left", target,
                                 tilt, az, self.solution, closing=closing)
        else:
            grip_q = (float(self.data.qpos[self.grip_q]) if self.gripping(name)
                      else OPEN_GRIP)
            solution = _jaw_correct(self.model, self.data, "left", target, tilt, az,
                                    self.solution, closing, grip_q)
        if not solution.accepted:
            self.fail(f"Unreachable {name}: IK error {solution.position_error:.4f} m, "
                      f"axis {solution.axis_error:.4f}")
            return
        self.solution = solution.targets
        self.end[self.actuators] = solution.targets
        if not self.gripping(name):
            self.end[self.grip] = OPEN_GRIP

    def before_step(self):
        if self.status != "running":
            return
        if self.phase < 0:
            self.enter(0)
        elapsed = float(self.data.time - self.phase_started)
        duration = self.phases[self.phase][1]
        if elapsed >= duration - 1e-9:
            name = self.phases[self.phase][0]
            if name == "Close on rim" and self.bilateral_s < 0.15:
                self.fail("No sustained two-jaw grasp of the plate rim")
                return
            if name == "Lift clear of table" and self.sustained_lift < 0.2:
                self.fail(f"Plate never left the table: peak lift {self.peak_lift:.4f} m")
                return
            if self.phase == len(self.phases) - 1:
                if self.sustained_lift >= 0.4 and self.stable_release >= 0.3:
                    self.status, self.sim.running = "succeeded", False
                else:
                    self.fail(f"Lift {self.sustained_lift:.2f} s, "
                              f"stable release {self.stable_release:.2f} s")
                return
            self.enter(self.phase + 1)
            if self.status != "running":
                return
            elapsed, duration = 0.0, self.phases[self.phase][1]
        blend = min(elapsed / (0.8 * duration), 1)
        blend = blend * blend * (3 - 2 * blend)
        self.data.ctrl[:] = (1 - blend) * self.begin + blend * self.end
        if self.gripping(self.phases[self.phase][0]):
            self.data.ctrl[self.grip] = np.clip(
                self.data.qpos[self.grip_q] - 0.004,
                *self.model.actuator_ctrlrange[self.grip])

    def after_step(self):
        if self.status != "running":
            return
        bodies = self.contacts()
        held = self.fingers <= bodies
        if held:
            self.current_contact += self.model.opt.timestep
            self.bilateral_s = max(self.bilateral_s, self.current_contact)
        else:
            self.current_contact = 0.0
        self.peak_lift = max(self.peak_lift, self.lift)
        if held and self.lift > 0.025:
            self.lift_run += self.model.opt.timestep
            self.sustained_lift = max(self.sustained_lift, self.lift_run)
        else:
            self.lift_run = 0.0
        position = self.data.xpos[self.body]
        velocity = self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[self.body]]:][:6]
        if (np.linalg.norm(position[:2] - self.mat[:2]) < 0.03
                and not (self.fingers & bodies) and bodies
                and float(np.linalg.norm(velocity)) < 0.02):
            self.stable_release += self.model.opt.timestep
        else:
            self.stable_release = 0.0

    def state(self):
        return {"name": "plate-place",
                "controller": "Left-arm scripted physical plate rim grasp",
                "status": self.status, "reason": self.reason, "attempt": 1,
                "phase": self.phases[max(self.phase, 0)][0],
                "elapsed_s": float(self.data.time - self.started),
                "progress": min(float(self.data.time - self.started)
                                / sum(t for _, t in self.phases), 1),
                "bilateral_contact_s": self.bilateral_s,
                "peak_lift_m": self.peak_lift,
                "sustained_lift_s": self.sustained_lift,
                "stable_release_s": self.stable_release,
                "placement_error_m": float(np.linalg.norm(
                    self.data.xpos[self.body][:2] - self.mat[:2])),
                "workcell": {"plate_start": [float(v) for v in self.start]},
                "events": self.events,
                "limits": "Scripted plate teacher in its own workcell; "
                          "not a learned policy and not the full dinner-table sequence"}


def main():
    from tabletop_vla.sim.runtime import Simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="40:50")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-gripper", action="store_true",
                        help="Negative control: the grasp gate must reject this")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing evidence is never overwritten")
    start, stop = map(int, args.seeds.split(":"))
    if not 0 <= start < stop <= 1000001 or stop - start > 1000:
        parser.error("Invalid seed range")
    sim, results = Simulation(), []
    try:
        for seed in range(start, stop):
            sim.reset(seed)
            prepare_plate_workcell(sim, seed)
            sim.task = PlatePlace(sim, close_gripper=not args.open_gripper)
            sim.running = True
            for _ in range(60):
                sim.step(500)
                if sim.task.status != "running":
                    break
            if sim.task.status == "running":
                sim.task.fail("Episode timeout")
            result = {"seed": seed, **sim.task.state(), "variation": sim.variation,
                      "physics_warnings": sim.data.warning.number.tolist()}
            results.append(result)
            print(json.dumps({k: result[k] for k in (
                "seed", "status", "reason", "peak_lift_m", "sustained_lift_s",
                "stable_release_s", "placement_error_m")}), flush=True)
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"task": "plate-place",
         "successes": sum(r["status"] == "succeeded" for r in results),
         "episodes": len(results), "negative_control": args.open_gripper,
         "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
