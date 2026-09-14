"""Left-arm cutlery retrieval from a drawer this same episode physically opened.

No reset between the drawer skill and the retrieval skill, no weld, no teleport, and
no direct drive of the drawer joint. Success comes from independent postconditions:
force-bearing bilateral jaw contact, a sustained lift, and a stable release on the mat.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.drawer import DrawerOpen, prepare_workcell
from tabletop_vla.sim.kinematics import JOINTS, gripper_geometry, solve_pose
from tabletop_vla.sim.reach import _cartesian

# Measured 2026-09-14 (scratch probes, see the commit message): a constant-height
# traverse at this altitude clears the drawer handle and front wall with zero contact.
# Routes that dipped below it swept the gripper through the handle at 18-85 N and shoved
# the drawer shut, dragging the cutlery back inside.
TRAVERSE_Z = 0.86
TRAVERSE_Y = (-0.12, -0.08, -0.04, 0.00, 0.04)
MATS = {"fork": np.array([-0.32, 0.10, 0.738]), "spoon": np.array([0.32, 0.10, 0.738])}
GRASP_TILT, GRASP_AZ = 20, 98
# Measured 2026-09-14: the mat side of the table wants a different wrist azimuth than the
# drawer side. fork_mat is accepted from z 0.74 to 0.86 at azimuth 106, and not at 98.
MAT_TILT, MAT_AZ = 30, 106
# Phases on the mat side of the table, and phases whose jaws must stay aligned with the
# handle. Matching on substrings silently sent "Release" to the drawer-side azimuth and
# failed it on a 0.0068 axis-error margin, so the sets are explicit.
MAT_PHASES = frozenset({"Carry to mat", "Descend to mat", "Release", "Retract",
                        "Verify placement"})
# Constraining the closing axis during the lift and the raise over-constrained a 5-DOF
# arm and pushed "Close on handle" back over the axis gate, so those two stay free.
ALIGNED_PHASES = frozenset({"Hover above handle", "Descend to handle", "Close on handle",
                            "Carry to mat", "Descend to mat", "Release"})
OPEN_GRIP = 0.85


def _waypoint(model, data, arm, target, tilt, az, initial, closing=None):
    """Nearby approach angles are tried before giving up; a waypoint is not worth
    failing over a two-degree wrist angle. Chaining `initial` keeps the arm on one IK
    branch so the joint-space path between waypoints stays short."""
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


def _jaw_aim(model, data, arm, target, tilt, az, initial, closing, grip_q):
    """Solve, measure where the live jaw gap actually lands, then re-solve once.

    `solve_pose` aims the static GRASP_POINT constant, but the SO-101 has one fixed and
    one moving pad, so the real jaw midpoint moves with the opening. Measured
    2026-09-14: at the approach opening the midpoint sits about 11 mm from the constant,
    which is enough to close the jaws above a 16 mm cutlery handle instead of around it.
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


class CutleryRetrieve:
    """Opens the drawer with the verified teacher, then retrieves one piece of cutlery."""

    def __init__(self, sim, obj="fork", close_gripper=True):
        if obj not in MATS:
            raise ValueError("Only fork and spoon are implemented")
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.obj, self.close_gripper = obj, close_gripper
        self.status, self.reason = "running", None
        self.started = float(self.data.time)
        self.drawer = DrawerOpen(sim, close_gripper=True)
        self.stage = "drawer"
        self.phase, self.phase_started = -1, self.started
        self.body = self.model.body(obj).id
        self.handle_geom = self.model.geom(f"{obj}_handle").id
        self.fingers = {self.model.body("left_gripper").id,
                        self.model.body("left_moving_jaw_so101_v1").id}
        self.actuators = [self.model.actuator(f"left_{j}").id for j in JOINTS]
        self.grip = self.model.actuator("left_gripper").id
        self.grip_q = int(self.model.joint("left_gripper").qposadr[0])
        self.floor_z = float(self.data.geom_xpos[self.model.geom("drawer_floor").id][2])
        self.mat = MATS[obj]
        self.solution = None
        self.bilateral_s = self.current_contact = 0.0
        self.peak_lift = self.lift_run = self.sustained_lift = self.stable_release = 0.0
        self.events = []
        self.phases = (
            ("Fold clear of handle", 1.6),
            *[(f"Traverse y={y:+.2f}", 1.1) for y in TRAVERSE_Y],
            ("Traverse over cutlery", 1.3),
            ("Hover above handle", 1.5),
            ("Descend to handle", 2.0),
            ("Close on handle", 3.0),
            ("Lift clear of drawer", 2.0),
            ("Raise to transit height", 1.6),
            ("Carry to mat", 2.2),
            ("Descend to mat", 1.8),
            ("Release", 2.0),
            ("Retract", 2.0),
            ("Verify placement", 1.4),
        )

    # ---- shared helpers -------------------------------------------------

    def contacts(self):
        """Bodies pressing on the cutlery with a force-bearing contact."""
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
        self.drawer.status = "cancelled"

    @property
    def lift(self):
        return float(self.data.xpos[self.body][2] - self.floor_z)

    # ---- retrieval phases -----------------------------------------------

    def target_for(self, name):
        """Where the jaw gap should sit for this phase, in world coordinates."""
        handle = self.data.geom_xpos[self.handle_geom].copy()
        if name == "Traverse over cutlery":
            return np.array([handle[0], handle[1], TRAVERSE_Z])
        if name.startswith("Traverse y="):
            return np.array([handle[0], float(name.split("=")[1]), TRAVERSE_Z])
        if name == "Hover above handle":
            return handle + [0, 0, 0.04]
        if name in {"Descend to handle", "Close on handle"}:
            # Measured: aiming at the handle centre leaves the jaws ~7 mm high, gripping
            # the top corner and slipping. The arm does not fully reach its commanded
            # pose, so aim below the centre and let contact stop the descent.
            return handle + [0, 0, -0.005]
        if name == "Lift clear of drawer":
            return handle + [0, 0, 0.06]
        if name == "Raise to transit height":
            return np.array([handle[0], handle[1], TRAVERSE_Z])
        if name == "Carry to mat":
            aim = self.mat + self.body_to_handle()
            return np.array([aim[0], aim[1], TRAVERSE_Z])
        if name in {"Descend to mat", "Release"}:
            return self.mat + self.body_to_handle() + [0, 0, 0.012]
        if name == "Retract":
            return np.array([self.mat[0], self.mat[1], TRAVERSE_Z])
        return None

    def body_to_handle(self):
        """The jaws hold the handle, but the placement gate scores the body centre, which
        sits about 32 mm away. Aim the handle so the BODY lands on the mat."""
        offset = self.data.geom_xpos[self.handle_geom] - self.data.xpos[self.body]
        return np.array([offset[0], offset[1], 0.0])

    def axis_for(self, name):
        """Preferred approach angle per phase; `_waypoint` searches around it."""
        return (MAT_TILT, MAT_AZ) if name in MAT_PHASES else (GRASP_TILT, GRASP_AZ)

    def gripping(self, name):
        return self.close_gripper and name in {
            "Close on handle", "Lift clear of drawer", "Raise to transit height",
            "Carry to mat", "Descend to mat"}

    def enter(self, phase):
        self.phase, self.phase_started = phase, float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        name = self.phases[phase][0]
        self.events.append({"phase": name, "time_s": self.phase_started})
        if name == "Fold clear of handle":
            self.end[self.actuators[0]] = OPEN_GRIP
            self.end[self.grip] = OPEN_GRIP
            return
        target = self.target_for(name)
        if target is None:
            return
        # Jaws must close across the handle's 14 mm width, not along its length.
        closing = (1.0, 0.0, 0.0) if name in ALIGNED_PHASES else None
        # Only the phases that must land the jaws on something pay for the extra solve.
        tilt, az = self.axis_for(name)
        if closing is None:
            solution = _waypoint(self.model, self.data, "left", target,
                                 tilt, az, self.solution, closing=closing)
        else:
            grip_q = (float(self.data.qpos[self.grip_q]) if self.gripping(name)
                      else OPEN_GRIP)
            solution = _jaw_aim(self.model, self.data, "left", target, tilt, az,
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
        if self.stage == "drawer":
            self.drawer.before_step()
            if self.drawer.status == "failed":
                self.fail(f"Drawer stage failed: {self.drawer.reason}")
            elif self.drawer.status == "succeeded":
                self.stage, self.sim.running = "retrieve", True
                self.solution = None
                self.enter(0)
            return
        if self.phase < 0:
            self.enter(0)
        elapsed = float(self.data.time - self.phase_started)
        duration = self.phases[self.phase][1]
        if elapsed >= duration - 1e-9:
            name = self.phases[self.phase][0]
            if name == "Close on handle" and self.bilateral_s < 0.15:
                self.fail("No sustained two-jaw grasp of the cutlery handle")
                return
            if name == "Lift clear of drawer" and self.sustained_lift < 0.2:
                self.fail(f"Cutlery never left the drawer: peak lift {self.peak_lift:.4f} m")
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
        if self.stage == "drawer":
            self.drawer.after_step()
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
        name = (self.drawer.phases[max(self.drawer.phase, 0)][0] if self.stage == "drawer"
                else self.phases[max(self.phase, 0)][0])
        total = sum(t for _, t in self.phases) + sum(t for _, t in self.drawer.phases)
        return {"name": f"{self.obj}-retrieve",
                "controller": "Left-arm scripted physical cutlery retrieval",
                "status": self.status, "reason": self.reason, "attempt": 1,
                "stage": self.stage, "phase": name,
                "elapsed_s": float(self.data.time - self.started),
                "progress": min(float(self.data.time - self.started) / total, 1),
                "drawer_travel_m": self.drawer.travel,
                "bilateral_contact_s": self.bilateral_s,
                "peak_lift_m": self.peak_lift,
                "sustained_lift_s": self.sustained_lift,
                "stable_release_s": self.stable_release,
                "placement_error_m": float(np.linalg.norm(
                    self.data.xpos[self.body][:2] - self.mat[:2])),
                "events": self.events,
                "limits": "Scripted teacher chained after the drawer skill in one episode; "
                          "not a learned policy and not the full dinner-table sequence"}


def main():
    from tabletop_vla.sim.runtime import Simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="fork", choices=sorted(MATS))
    parser.add_argument("--seeds", default="30:40")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-gripper", action="store_true",
                        help="Negative control: the retrieval gate must reject this")
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
            prepare_workcell(sim, seed)
            sim.task = CutleryRetrieve(sim, args.object, close_gripper=not args.open_gripper)
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
        {"task": f"{args.object}-retrieve",
         "successes": sum(r["status"] == "succeeded" for r in results),
         "episodes": len(results), "negative_control": args.open_gripper,
         "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
