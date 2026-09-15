"""Bimanual spoon handoff: left-arm giver to right-arm receiver, strict five-gate scoring.

The giver stage reuses ``CutleryRetrieve(sim, "spoon")`` unchanged: it opens the drawer
in the same episode, grasps the spoon handle, and lifts it. Its carry to ``spoon_mat``
is unreachable for the left arm (IK error 0.244 m), which is why the handoff exists.
Takeover happens exactly when the giver fails at "Carry to mat" while still holding
the spoon with a sustained lift; any other giver failure fails gate 1 honestly.

Measured 2026-09-14 (stage-1 probes, seed 30, real physics; full transcript in
.agent-logs/impl-handoff.md):

- The spoon handle is 14 x 36 x 16 mm. Each gripper carries a moving pad 13 x 34 x
  15 mm. In the scratch state with both jaw midpoints on the handle, the two
  moving-pad centres sit 3.3 mm apart (9.7 mm of overlap) and each pad's projected
  y-footprint covers 22-25 mm of the 36 mm handle. Two grippers need 44-50 mm.
- Left carries the held spoon to the handoff point (handle lands 25 mm off even with
  jaw-aim correction). Right reaches the handoff point with a free wrist (pe
  0.87 mm, ae 0.0368) but every grasp-aligned closing is IK-rejected there (closeX
  pe 2.20 mm ae 0.0803; closeY pe 1.35 mm ae 0.0833). The live held handle is
  rejected at every approach angle (best of 49 axes pe 81 mm).
- Driving the right arm to the held spoon collides arm-to-arm: left_moving_pad
  against right_moving_pad at 18.1-24.3 N plus neck contacts at 15.4-15.7 N, shoves
  the left arm 50-75 mm off station, and never achieves right bilateral contact.
- The blade (16 x 100 x 6 mm, flush with the surface) is IK-accepted at 0.77 mm but
  the pads strike the table at 13.7-25.3 N with no bilateral contact and no lift,
  the same failure mode as the fork blade (7.6 N skid).

SUPERSEDED: the "every grasp-aligned closing is IK-rejected" finding above was
measured at the old free-wrist handoff point and does not generalise. Re-measuring
with the closing axis constrained found the midline point recorded at HANDOFF below,
which both arms accept. The handle length finding stands and has been fixed in
build_scene.py (36 mm to 56 mm). Sequential touching is still never scored: gates 3-5
strictly require gate 2 first, in order.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.cutlery import UNREACHED, CutleryRetrieve
from tabletop_vla.sim.drawer import prepare_workcell
from tabletop_vla.sim.kinematics import JOINTS, gripper_geometry, solve_pose
from tabletop_vla.sim.reach import _cartesian

# The only point both arms accept (outputs/reach-map-refined-seed30.json):
# left tilt 30 az 45 (pe 0.86 mm), right tilt 30 az 135 (pe 0.87 mm).
# CORRECTED 2026-09-14 after an independent re-measurement. The earlier point
# [0.0, -0.05, 0.85] came from a reach map that let the wrist roll freely, and at that
# point every grasp-aligned closing really is rejected, which is what the first probe
# found. But a free-wrist point is the wrong thing to search for: two grippers holding
# one object BOTH need a defined closing direction. Searching with the closing axis
# constrained finds the midline instead, verified for both arms at three heights:
#   [0.00, +0.02, 0.80] close_y  left t30/a45 pe 0.00065 ae 0.0251
#                                right t30/a135 pe 0.00063 ae 0.0249
#   [0.00, +0.02, 0.85] close_y  left t45/a45 pe 0.00130 ae 0.0313
#                                right t45/a135 pe 0.00126 ae 0.0312
# Closing along x is rejected for the left arm at all three heights, so the spoon must
# be presented with its long axis along world x and both jaws closing along world y.
HANDOFF = np.array([0.0, 0.02, 0.85])
GIVE_TILT, GIVE_AZ = 45, 45
RECEIVE_TILT, RECEIVE_AZ = 45, 135
HANDOFF_CLOSING = (0.0, 1.0, 0.0)
RESTING_Z = 0.738
# Right-arm reachable only at z <= 0.76 (0.74/0.76 accepted, 0.78+ rejected), so the
# receiver carries low instead of at the cutlery TRAVERSE_Z of 0.86.
RECEIVE_CARRY_Z = 0.76
OPEN_GRIP = 0.85
# Explicit phase sets. Substring matching once sent "Release" to the wrong wrist
# azimuth in cutlery.py, so every set lists full phase names.
LEFT_PHASES = frozenset({"Carry to handoff", "Settle at handoff", "Right approach",
                         "Right close"})
RIGHT_PHASES = frozenset({"Right approach", "Right close", "Giver release",
                          "Receiver lift", "Carry to mat", "Descend to mat",
                          "Release", "Retract", "Verify ownership"})
ALIGNED_PHASES = frozenset({"Carry to handoff", "Right approach", "Right close",
                            "Carry to mat", "Descend to mat", "Release"})
LEFT_GRIPPING = frozenset({"Carry to handoff", "Settle at handoff", "Right approach",
                           "Right close"})
RIGHT_GRIPPING = frozenset({"Right close", "Giver release", "Receiver lift",
                            "Carry to mat", "Descend to mat"})


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


def _jaw_aim(model, data, arm, target, tilt, az, initial, closing, grip_q):
    """Solve, measure where the live jaw gap actually lands, then re-solve once.

    `solve_pose` aims the static GRASP_POINT constant, but the SO-101 has one fixed
    and one moving pad, so the real jaw midpoint moves with the opening (about
    11 mm at the approach opening, measured 2026-09-14). Same correction as the
    cutlery teacher; written out here because that helper is private to its module.
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


class SpoonHandoff:
    """Left arm retrieves the spoon, both arms meet at HANDOFF, right arm places it.

    Gates in order, each measured independently: (1) giver holds, (2) receiver
    contacts while the giver still holds, (3) giver releases with the receiver
    still holding, (4) receiver lifts held only by the right jaws, (5) sustained
    receiver ownership with no further left contact. `close_gripper=False` is the
    negative control: the RIGHT gripper is held open (the giver still closes, so a
    score would have to come through the receiver gates) and the run must score 0.
    """

    def __init__(self, sim, close_gripper=True):
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.close_gripper = close_gripper
        self.status, self.reason = "running", None
        self.started = float(self.data.time)
        self.giver = CutleryRetrieve(sim, "spoon", close_gripper=True)
        self.stage = "giver"
        self.phase, self.phase_started = -1, self.started
        self.body = self.model.body("spoon").id
        self.handle_geom = self.model.geom("spoon_handle").id
        self.left_fingers = {self.model.body("left_gripper").id,
                             self.model.body("left_moving_jaw_so101_v1").id}
        self.right_fingers = {self.model.body("right_gripper").id,
                              self.model.body("right_moving_jaw_so101_v1").id}
        self.left_actuators = [self.model.actuator(f"left_{j}").id for j in JOINTS]
        self.right_actuators = [self.model.actuator(f"right_{j}").id for j in JOINTS]
        self.left_grip = self.model.actuator("left_gripper").id
        self.right_grip = self.model.actuator("right_gripper").id
        self.left_q = int(self.model.joint("left_gripper").qposadr[0])
        self.right_q = int(self.model.joint("right_gripper").qposadr[0])
        site = self.model.site("spoon_mat")
        self.mat = np.array([float(site.pos[0]), float(site.pos[1]), RESTING_Z])
        self.left_solution, self.right_solution = None, None
        self.gates = {"giver_holds": False, "receiver_contacts": False,
                      "giver_releases": False, "receiver_lifts": False,
                      "sustained_ownership": False}
        self.simultaneous_s = self.simultaneous_run = 0.0
        self.receiver_only_run = self.receiver_only_s = 0.0
        self.release_z = self.peak_rise = 0.0
        self.left_contact_after_release = False
        self.release_time = 0.0
        self.stable_release = 0.0
        self.events = []
        self.phases = (
            ("Carry to handoff", 2.2),
            ("Settle at handoff", 1.0),
            ("Right approach", 2.0),
            ("Right close", 3.0),
            ("Giver release", 2.0),
            ("Receiver lift", 2.0),
            ("Carry to mat", 2.2),
            ("Descend to mat", 1.8),
            ("Release", 2.0),
            ("Retract", 2.0),
            ("Verify ownership", 1.4),
        )

    # ---- shared helpers -------------------------------------------------

    def contacts(self):
        """Bodies pressing on the spoon with a force-bearing contact."""
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
        self.giver.status = "cancelled"

    @property
    def left_holds(self):
        return self.left_fingers <= self.contacts()

    @property
    def right_holds(self):
        return self.right_fingers <= self.contacts()

    # ---- handoff geometry ------------------------------------------------

    def target_for(self, name):
        handle = self.data.geom_xpos[self.handle_geom].copy()
        if name == "Carry to handoff":
            return HANDOFF.copy()
        if name in {"Settle at handoff", "Right close", "Giver release"}:
            return handle.copy()
        if name == "Right approach":
            return handle + [0, 0, 0.04]
        if name == "Receiver lift":
            return handle + [0, 0, 0.06]
        if name == "Carry to mat":
            return np.array([self.mat[0], self.mat[1], RECEIVE_CARRY_Z])
        if name in {"Descend to mat", "Release"}:
            return self.mat + [0, 0, 0.012]
        if name == "Retract":
            return np.array([self.mat[0], self.mat[1], RECEIVE_CARRY_Z])
        return None

    def axis_for(self, name):
        if name in {"Carry to handoff", "Settle at handoff"}:
            return (GIVE_TILT, GIVE_AZ)
        return (RECEIVE_TILT, RECEIVE_AZ)

    def enter(self, phase):
        self.phase, self.phase_started = phase, float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        name = self.phases[phase][0]
        self.events.append({"phase": name, "time_s": self.phase_started})
        if name == "Settle at handoff":
            return
        target = self.target_for(name)
        if target is None:
            return
        # Jaws must close across the handle's 14 mm width, not along its length.
        closing = HANDOFF_CLOSING if name in ALIGNED_PHASES else None
        tilt, az = self.axis_for(name)
        if name in LEFT_PHASES and name not in RIGHT_PHASES:
            grip_q = float(self.data.qpos[self.left_q])
            solution = _jaw_aim(self.model, self.data, "left", target, tilt, az,
                                self.left_solution, closing, grip_q)
            if not solution.accepted:
                self.fail(f"{UNREACHED} {name} within the search window: "
                          f"position error {solution.position_error:.4f} m, "
                          f"axis error {solution.axis_error:.4f}")
                return
            self.left_solution = solution.targets
            self.end[self.left_actuators] = solution.targets
        else:
            grip_q = (float(self.data.qpos[self.right_q]) if self.close_gripper
                      else OPEN_GRIP)
            initial = self.right_solution
            solution = _jaw_aim(self.model, self.data, "right", target, tilt, az,
                                initial, closing, grip_q)
            if not solution.accepted:
                self.fail(f"{UNREACHED} {name} within the search window "
                          f"(gate 2: receiver cannot reach a grasp): "
                          f"IK error {solution.position_error:.4f} m, "
                          f"axis {solution.axis_error:.4f}")
                return
            self.right_solution = solution.targets
            self.end[self.right_actuators] = solution.targets
        if name not in RIGHT_GRIPPING or not self.close_gripper:
            self.end[self.right_grip] = OPEN_GRIP

    def take_over(self):
        """Start the handoff stage from the giver's held spoon; the giver's own
        "Carry to mat" is unreachable for the left arm, so its failure there while
        holding is the expected handover point, not a defect."""
        self.gates["giver_holds"] = True
        self.stage = "handoff"
        self.sim.running = True
        self.left_solution = self.giver.solution
        self.events.append({"phase": "Takeover from giver",
                            "time_s": float(self.data.time),
                            "bilateral_s": self.giver.bilateral_s,
                            "sustained_lift_s": self.giver.sustained_lift})
        self.enter(0)

    def before_step(self):
        if self.status != "running":
            return
        if self.stage == "giver":
            self.giver.before_step()
            if self.giver.status == "failed":
                if (self.giver.reason.startswith(f"{UNREACHED} Carry to mat")
                        and self.giver.bilateral_s >= 0.15
                        and self.giver.sustained_lift >= 0.4):
                    self.take_over()
                else:
                    self.fail(f"Giver stage failed (gate 1: giver holds): "
                              f"{self.giver.reason}")
            elif self.giver.status == "succeeded":
                self.fail("Giver stage placed the spoon alone (gate 1: left arm "
                          "cannot reach spoon_mat; unexpected success)")
            return
        if self.phase < 0:
            self.enter(0)
        elapsed = float(self.data.time - self.phase_started)
        duration = self.phases[self.phase][1]
        if elapsed >= duration - 1e-9:
            name = self.phases[self.phase][0]
            if name == "Settle at handoff" and not self.left_holds:
                self.fail("Gate 1 failed at handoff: giver lost the spoon in transit")
                return
            if name == "Right close" and self.simultaneous_s < 0.15:
                self.fail("Gate 2 failed: no sustained receiver bilateral contact "
                          f"while the giver holds (simultaneous {self.simultaneous_s:.2f} s)")
                return
            if name == "Giver release":
                if not self.gates["giver_releases"]:
                    self.fail("Gate 3 failed: giver never left contact while the "
                              "receiver held")
                    return
                if not self.right_holds:
                    self.fail("Gate 3 failed: receiver dropped the spoon on release")
                    return
            # Contact duration alone is NOT a lift. `receiver_lifts` was computed from
            # a 20 mm rise since the giver let go (see the rise check below) and then
            # never consulted, so a receiver that gripped the spoon and held it exactly
            # where the giver left it scored a handoff. AGENTS.md forbids counting
            # sequential contact by two arms as a transfer; this is that rule, enforced.
            if name == "Receiver lift" and (self.receiver_only_s < 0.4
                                            or not self.gates["receiver_lifts"]):
                self.fail(f"Gate 4 failed: no sustained receiver-only lift "
                          f"(receiver-only contact {self.receiver_only_s:.2f} s of 0.40, "
                          f"peak rise after release {self.peak_rise * 1000:.1f} mm of 20.0)")
                return
            if self.phase == len(self.phases) - 1:
                if (self.receiver_only_s >= 0.4 and self.gates["receiver_lifts"]
                        and self.stable_release >= 0.3
                        and not self.left_contact_after_release):
                    self.gates["sustained_ownership"] = True
                    self.status, self.sim.running = "succeeded", False
                else:
                    self.fail(f"Gate 5 failed: receiver-only {self.receiver_only_s:.2f} s, "
                              f"peak rise after release {self.peak_rise * 1000:.1f} mm, "
                              f"stable release {self.stable_release:.2f} s, "
                              f"left contact after release: "
                              f"{self.left_contact_after_release}")
                return
            self.enter(self.phase + 1)
            if self.status != "running":
                return
            elapsed, duration = 0.0, self.phases[self.phase][1]
        blend = min(elapsed / (0.8 * duration), 1)
        blend = blend * blend * (3 - 2 * blend)
        self.data.ctrl[:] = (1 - blend) * self.begin + blend * self.end
        name = self.phases[self.phase][0]
        if name in LEFT_GRIPPING:
            self.data.ctrl[self.left_grip] = np.clip(
                self.data.qpos[self.left_q] - 0.004,
                *self.model.actuator_ctrlrange[self.left_grip])
        elif name == "Giver release":
            self.data.ctrl[self.left_grip] = OPEN_GRIP
        if name in RIGHT_GRIPPING and self.close_gripper:
            self.data.ctrl[self.right_grip] = np.clip(
                self.data.qpos[self.right_q] - 0.004,
                *self.model.actuator_ctrlrange[self.right_grip])
        else:
            self.data.ctrl[self.right_grip] = OPEN_GRIP

    def after_step(self):
        if self.status != "running":
            return
        if self.stage == "giver":
            self.giver.after_step()
            return
        bodies = self.contacts()
        name = self.phases[max(self.phase, 0)][0]
        left, right = self.left_holds, self.right_holds
        if left and right:
            self.simultaneous_run += self.model.opt.timestep
            self.simultaneous_s = max(self.simultaneous_s, self.simultaneous_run)
        else:
            self.simultaneous_run = 0.0
        if name in {"Right close"} and left and right:
            self.gates["receiver_contacts"] = True
        if self.gates["giver_releases"] and (bodies & self.left_fingers):
            # Any left touch after a genuine release breaks receiver ownership.
            # Contacts while the fingers are still leaving (before the first
            # no-left instant) do not latch this flag.
            self.left_contact_after_release = True
        if name == "Giver release" and not (bodies & self.left_fingers) and right:
            self.gates["giver_releases"] = True
            self.release_time = float(self.data.time)
            if self.release_z == 0.0:
                self.release_z = float(self.data.xpos[self.body][2])
        if self.gates["giver_releases"] and right and not (bodies & self.left_fingers):
            self.receiver_only_run += self.model.opt.timestep
            self.receiver_only_s = max(self.receiver_only_s, self.receiver_only_run)
            rise = float(self.data.xpos[self.body][2] - self.release_z)
            self.peak_rise = max(self.peak_rise, rise)
            if rise >= 0.02:
                self.gates["receiver_lifts"] = True
        else:
            if name in {"Receiver lift", "Carry to mat", "Descend to mat", "Release",
                        "Retract", "Verify ownership"}:
                self.receiver_only_run = 0.0
        position = self.data.xpos[self.body]
        velocity = self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[self.body]]:][:6]
        on_table = abs(float(position[2]) - float(self.mat[2])) < 0.006
        if (np.linalg.norm(position[:2] - self.mat[:2]) < 0.03 and on_table
                and not (self.left_fingers & bodies) and not (self.right_fingers & bodies)
                and bodies and float(np.linalg.norm(velocity)) < 0.02):
            self.stable_release += self.model.opt.timestep
        else:
            self.stable_release = 0.0

    def state(self):
        if self.stage == "giver":
            name = (self.giver.drawer.phases[max(self.giver.drawer.phase, 0)][0]
                    if self.giver.stage == "drawer"
                    else self.giver.phases[max(self.giver.phase, 0)][0])
        else:
            name = self.phases[max(self.phase, 0)][0]
        total = sum(t for _, t in self.phases) + sum(
            t for _, t in self.giver.phases) + sum(t for _, t in self.giver.drawer.phases)
        return {"name": "spoon-handoff",
                "controller": "Left-arm giver plus right-arm receiver spoon handoff",
                "status": self.status, "reason": self.reason, "attempt": 1,
                "stage": self.stage, "phase": name,
                "elapsed_s": float(self.data.time - self.started),
                "progress": min(float(self.data.time - self.started) / total, 1),
                "gates": dict(self.gates),
                "giver_bilateral_s": self.giver.bilateral_s,
                "giver_sustained_lift_s": self.giver.sustained_lift,
                "simultaneous_bilateral_s": self.simultaneous_s,
                "receiver_only_s": self.receiver_only_s,
                "peak_rise_after_release_m": self.peak_rise,
                "left_contact_after_release": self.left_contact_after_release,
                "stable_release_s": self.stable_release,
                "placement_error_m": float(np.linalg.norm(
                    self.data.xpos[self.body][:2] - self.mat[:2])),
                "events": self.events,
                "limits": "Scripted bimanual teacher in the drawer workcell; not a learned "
                          "policy and not the full dinner-table sequence"}


def main():
    from tabletop_vla.sim.runtime import Simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="50:60")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-gripper", action="store_true",
                        help="Negative control: the RIGHT gripper is held open, so the "
                             "receiver gates must reject every episode")
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
            sim.task = SpoonHandoff(sim, close_gripper=not args.open_gripper)
            sim.running = True
            for _ in range(80):
                sim.step(500)
                if sim.task.status != "running":
                    break
            if sim.task.status == "running":
                sim.task.fail("Episode timeout")
            result = {"seed": seed, **sim.task.state(), "variation": sim.variation,
                      "physics_warnings": sim.data.warning.number.tolist()}
            results.append(result)
            print(json.dumps({k: result[k] for k in (
                "seed", "status", "reason", "stage", "gates",
                "simultaneous_bilateral_s", "receiver_only_s")}), flush=True)
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"task": "spoon-handoff",
         "successes": sum(r["status"] == "succeeded" for r in results),
         "episodes": len(results), "negative_control": args.open_gripper,
         "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
