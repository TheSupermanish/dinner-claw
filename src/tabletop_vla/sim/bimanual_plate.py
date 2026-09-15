"""Two-arm plate lift: both grippers pinch opposing rim segments and carry it level.

Why two arms. A single arm pinching the rim grips 94 mm from the plate's centre of
mass, so the lift is almost pure torque and the plate pivots on its far rim instead of
rising. Measured 2026-09-15 on seeds 40-49 of the single-arm teacher: six of ten reach
38-55 mm of centre height with 0.000-0.008 s of sustained lift, because the plate was
still resting on the table the whole time. Opposing grips cancel that torque.

Measured 2026-09-15, prototype over six candidate plate centres:

    centre (-0.05, 0.00)  rise 61.3 mm  tilt 7.9 deg  pads 4/4  on table False
    centre (+0.00, 0.00)  rise 59.8 mm  tilt 8.2 deg  pads 4/4  on table False
    centre (+0.05, 0.00)  rise 59.8 mm  tilt 9.4 deg  pads 4/4  on table False

and a reach sweep that found 7 of 12 candidate centres reachable by BOTH arms, all of
the y=0.00 row. The committed single-arm plate start x=-0.28 is far outside that
overlap, which is why it has only ever been levered.

Success needs independent physical postconditions, not a finished script: force-bearing
contact from all FOUR pads, a sustained lift with the plate touching nothing else, a
levelness bound that a pivot cannot satisfy, and a stable released placement at resting
height. The workcell parks the plate and the mat BEFORE the episode and declares both;
nothing is welded, teleported, or moved during execution.
"""

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.cutlery import UNREACHED
from tabletop_vla.sim.kinematics import JOINTS, solve_pose

ARMS = ("left", "right")
# Rim segment centres sit at radius 0.094 with their tops 0.024 above the plate origin;
# the jaws close radially across one segment, so the grasp point is the segment centre.
RIM_R, RIM_Z_OFF = 0.094, 0.014
PLATE_START = np.array([0.05, 0.00, 0.752])
PLATE_MAT = np.array([-0.05, 0.00, 0.737])
PLATE_JITTER = 0.005
# Table top plus the plate base half-height.
RESTING_Z = 0.739
TRANSIT_DZ = 0.07
OPEN_GRIP = 0.85
# Preferred approach angle per arm, from the reach sweep. Every waypoint re-searches
# from here rather than reusing it: an angle accepted at rim height is not accepted
# 50 mm above it, and reusing one across configurations is the defect that has cost
# this project four separate rounds.
PREFERRED = {"left": (0, 0), "right": (10, 120)}
# A plate held by two opposing grips stayed within 9.4 degrees of level in every
# prototype run. 15 degrees leaves margin for seed variation while still rejecting the
# single-arm pivot, which tips the plate far past it.
LEVEL_LIMIT_DEG = 15.0


def prepare_bimanual_workcell(sim, seed):
    """Park the plate and its mat inside the two-arm overlap, BEFORE the episode."""
    rng = np.random.default_rng(seed + 8200000)
    start = PLATE_START.copy()
    start[:2] += rng.uniform(-PLATE_JITTER, PLATE_JITTER, 2)
    body = sim.model.body("plate").id
    address = sim.model.jnt_qposadr[sim.model.body_jntadr[body]]
    sim.data.qpos[address:address + 3] = start
    sim.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
    # The authored plate_mat is outside the overlap, and so is the single-arm
    # workcell's relocated mat at x=-0.14: the right arm cannot reach either. Both
    # ends of this carry have to be reachable by both arms or the plate cannot be
    # held level for the whole move.
    sim.model.site_pos[sim.model.site("plate_mat").id] = PLATE_MAT
    mujoco.mj_forward(sim.model, sim.data)
    sim.variation["bimanual_plate_workcell"] = {
        "plate_start": start.tolist(),
        "plate_mat_moved_into_two_arm_overlap": PLATE_MAT.tolist()}


def _axis(tilt, az):
    t, a = np.deg2rad(tilt), np.deg2rad(az)
    return np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])


def _waypoint(model, data, arm, target, preferred, initial):
    """Search an approach angle AT THIS TARGET, preferring the measured one.

    The preferred angle is tried first so an accepted pose keeps the wrist on the
    branch the sweep measured; the coarse fallback exists because the same angle is
    not accepted at every height along a phase sequence.

    A coarse grid alone rejects poses that are reachable. Measured 2026-09-15 on the
    10-degree tilt, 15-degree azimuth grid: seed 40 lost "Hover above rims" at axis
    error 0.0415 and seed 42 lost "Retract" at 0.0405, both against a 0.0400 gate and
    seed 42 with a finished 4.67 s lift already behind it. So a near miss earns a fine
    search around the coarse winner rather than a failure, the same two-stage shape
    reach.py uses.
    """
    best = None

    def attempt(tilt, az):
        nonlocal best
        solution = solve_pose(model, data, arm, target, approach=_axis(tilt, az),
                              closing=(1.0, 0.0, 0.0), initial=initial)
        if best is None or (solution.position_error + 0.1 * solution.axis_error
                            < best.position_error + 0.1 * best.axis_error):
            best = solution
            best_angle[:] = [tilt, az]
        return solution

    best_angle = [preferred[0], preferred[1]]
    coarse = itertools.chain(
        [preferred], itertools.product(range(0, 50, 10), range(0, 360, 15)))
    for tilt, az in coarse:
        if attempt(tilt, az).accepted:
            return best
    centre_tilt, centre_az = best_angle
    for dt, da in itertools.product(np.arange(-8.0, 8.5, 2.0), np.arange(-12.0, 12.5, 3.0)):
        tilt = float(np.clip(centre_tilt + dt, 0.0, 89.0))
        if attempt(tilt, (centre_az + da) % 360).accepted:
            return best
    return best


class BimanualPlateLift:
    """Both arms grasp opposing plate rims, lift level, carry, and release."""

    def __init__(self, sim, close_gripper=True):
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.close_gripper = close_gripper
        self.status, self.reason = "running", None
        self.started = float(self.data.time)
        self.phase, self.phase_started = -1, self.started
        self.body = self.model.body("plate").id
        self.pads = {arm: {self.model.body(f"{arm}_gripper").id,
                           self.model.body(f"{arm}_moving_jaw_so101_v1").id}
                     for arm in ARMS}
        self.all_pads = set().union(*self.pads.values())
        self.actuators = {arm: [self.model.actuator(f"{arm}_{j}").id for j in JOINTS]
                          for arm in ARMS}
        self.grip = {arm: self.model.actuator(f"{arm}_gripper").id for arm in ARMS}
        self.grip_q = {arm: int(self.model.joint(f"{arm}_gripper").qposadr[0])
                       for arm in ARMS}
        site = self.model.site("plate_mat")
        self.mat = np.array([float(site.pos[0]), float(site.pos[1]), RESTING_Z])
        self.start = self.data.xpos[self.body].copy()
        self.solution = {arm: None for arm in ARMS}
        self.quad_s = self.current_contact = 0.0
        self.peak_lift = self.lift_run = self.sustained_lift = 0.0
        self.stable_release = 0.0
        self.peak_tilt_deg = 0.0
        self.tilt_at_peak_lift_deg = 0.0
        self.events = []
        self.phases = (
            ("Settle", 0.6),
            ("Hover above rims", 1.6),
            ("Descend to rims", 2.0),
            ("Close on rims", 3.0),
            ("Lift clear of table", 2.2),
            ("Carry to mat", 2.6),
            ("Descend to mat", 2.0),
            ("Release", 2.0),
            ("Retract", 1.6),
            ("Verify placement", 1.4),
        )

    # -- geometry ---------------------------------------------------------------

    def rim_targets(self, centre):
        """Where each jaw gap should sit, on opposing rim segments of `centre`."""
        return {"left": np.array([centre[0] - RIM_R, centre[1], centre[2] + RIM_Z_OFF]),
                "right": np.array([centre[0] + RIM_R, centre[1], centre[2] + RIM_Z_OFF])}

    def target_for(self, name):
        plate = self.data.xpos[self.body].copy()
        if name == "Hover above rims":
            return self.rim_targets(plate + [0, 0, 0.05])
        if name in {"Descend to rims", "Close on rims"}:
            return self.rim_targets(plate - [0, 0, 0.002])
        if name == "Lift clear of table":
            return self.rim_targets(plate + [0, 0, TRANSIT_DZ])
        if name == "Carry to mat":
            # Hold the achieved lift height and translate in xy only, so the carry
            # cannot smuggle in a vertical move that the lift gate already scored.
            return self.rim_targets(np.array([self.mat[0], self.mat[1], plate[2]]))
        if name in {"Descend to mat", "Release"}:
            return self.rim_targets(self.mat + [0, 0, 0.004])
        if name == "Retract":
            return self.retract_candidates()[0]
        return None

    def retract_candidates(self):
        """Clearance heights to try for the retreat, nearest first.

        Straight up from where the jaws ACTUALLY are: aiming a retract at the nominal
        mat commands a sideways move equal to however far the plate drifted, and that
        sideways move is itself the drag that two single-arm retract rewrites failed
        to remove. Unlike every other waypoint the retract has real freedom, because
        any pose clear of the plate ends the episode equally well, so a height that
        does not solve is not a failure while another height remains. Measured
        2026-09-15 at a single 0.06 m clearance: seeds 42, 53 and 54 lost a finished
        lift to a retract miss of 0.0005 on the axis gate.
        """
        here = self.rim_targets(self.data.xpos[self.body])
        return [{arm: np.array([p[0], p[1], self.mat[2] + dz])
                 for arm, p in here.items()}
                for dz in (0.06, 0.08, 0.10, 0.05, 0.12)]

    # -- bookkeeping ------------------------------------------------------------

    def contacts(self):
        """Bodies pressing on the plate with a force-bearing contact."""
        bodies, force = set(), np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.body in (a, b):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if force[0] > 0.02:
                    bodies.add(int(b if a == self.body else a))
        return bodies

    def tilt_deg(self):
        """Angle between the plate's own z axis and world up."""
        zaxis = self.data.xmat[self.body].reshape(3, 3)[:, 2]
        return float(np.rad2deg(np.arccos(np.clip(abs(float(zaxis[2])), -1.0, 1.0))))

    @property
    def lift(self):
        return float(self.data.xpos[self.body][2]) - float(self.start[2])

    def fail(self, reason):
        self.status, self.reason = "failed", reason
        self.sim.running = False

    def cancel(self):
        self.status, self.reason = "cancelled", "Manual control or another task"

    def gripping(self, name):
        return self.close_gripper and name in {
            "Close on rims", "Lift clear of table", "Carry to mat", "Descend to mat"}

    # -- control ----------------------------------------------------------------

    def enter(self, phase):
        self.phase, self.phase_started = phase, float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        name = self.phases[phase][0]
        self.events.append({"phase": name, "time_s": self.phase_started})
        if name == "Settle":
            for arm in ARMS:
                self.end[self.grip[arm]] = OPEN_GRIP
            return
        options = (self.retract_candidates() if name == "Retract"
                   else [self.target_for(name)])
        if options[0] is None:
            return
        worst = None
        for targets in options:
            solved, failure = {}, None
            for arm in ARMS:
                solution = _waypoint(self.model, self.data, arm, targets[arm],
                                     PREFERRED[arm], self.solution[arm])
                if not solution.accepted:
                    failure = (arm, solution)
                    break
                solved[arm] = solution
            if failure is None:
                for arm, solution in solved.items():
                    self.solution[arm] = solution.targets
                    self.end[self.actuators[arm]] = solution.targets
                    if not self.gripping(name):
                        self.end[self.grip[arm]] = OPEN_GRIP
                return
            worst = failure
        arm, solution = worst
        self.fail(f"{UNREACHED} {name} within the search window for the "
                  f"{arm} arm after {len(options)} clearance "
                  f"{'heights' if len(options) > 1 else 'target'}: position error "
                  f"{solution.position_error:.4f} m, axis error "
                  f"{solution.axis_error:.4f}")

    def before_step(self):
        if self.status != "running":
            return
        if self.phase < 0:
            self.enter(0)
        elapsed = float(self.data.time - self.phase_started)
        duration = self.phases[self.phase][1]
        if elapsed >= duration - 1e-9:
            name = self.phases[self.phase][0]
            if name == "Close on rims" and self.quad_s < 0.15:
                self.fail(f"No sustained four-pad grasp of the plate rims "
                          f"(all-four contact {self.quad_s:.2f} s of 0.15)")
                return
            if name == "Lift clear of table" and self.sustained_lift < 0.2:
                self.fail(f"Plate never left the table: peak lift "
                          f"{self.peak_lift:.4f} m, peak tilt "
                          f"{self.peak_tilt_deg:.1f} deg")
                return
            if self.phase == len(self.phases) - 1:
                if self.sustained_lift >= 0.4 and self.stable_release >= 0.3:
                    self.status, self.sim.running = "succeeded", False
                else:
                    self.fail(f"Lift {self.sustained_lift:.2f} s, stable release "
                              f"{self.stable_release:.2f} s, peak tilt "
                              f"{self.peak_tilt_deg:.1f} deg")
                return
            self.enter(self.phase + 1)
            if self.status != "running":
                return
            elapsed, duration = 0.0, self.phases[self.phase][1]
        blend = min(elapsed / (0.8 * duration), 1)
        blend = blend * blend * (3 - 2 * blend)
        self.data.ctrl[:] = (1 - blend) * self.begin + blend * self.end
        if self.gripping(self.phases[self.phase][0]):
            for arm in ARMS:
                self.data.ctrl[self.grip[arm]] = np.clip(
                    self.data.qpos[self.grip_q[arm]] - 0.004,
                    *self.model.actuator_ctrlrange[self.grip[arm]])

    def after_step(self):
        if self.status != "running":
            return
        bodies = self.contacts()
        # All FOUR pads, not two. Two pads bearing force is one arm's grasp, which is
        # the configuration that levers the plate rather than lifting it.
        held = self.all_pads <= bodies
        if held:
            self.current_contact += self.model.opt.timestep
            self.quad_s = max(self.quad_s, self.current_contact)
        else:
            self.current_contact = 0.0
        tilt = self.tilt_deg()
        self.peak_lift = max(self.peak_lift, self.lift)
        # Height alone is not evidence of a lift, and neither is height plus contact:
        # a plate pivoting on its far rim raises its centre while still resting on the
        # table. Require the pads to be the only force-bearing contact AND the plate to
        # stay level, which a pivot cannot do.
        unsupported = held and 0 not in bodies
        if unsupported and self.lift > 0.025 and tilt < LEVEL_LIMIT_DEG:
            self.lift_run += self.model.opt.timestep
            if self.lift_run > self.sustained_lift:
                self.sustained_lift = self.lift_run
                self.tilt_at_peak_lift_deg = tilt
            self.peak_tilt_deg = max(self.peak_tilt_deg, tilt)
        else:
            self.lift_run = 0.0
        position = self.data.xpos[self.body]
        velocity = self.data.qvel[
            self.model.jnt_dofadr[self.model.body_jntadr[self.body]]:][:6]
        # The resting-height check is not optional: without it a plate left on top of
        # the drawer or the cabinet scores as placed. This is the check that exposed
        # the fork-on-the-open-drawer result and the single-arm plate-on-the-drawer one.
        on_table = abs(float(position[2]) - float(self.mat[2])) < 0.006
        if (float(np.linalg.norm(position[:2] - self.mat[:2])) < 0.03 and on_table
                and not (self.all_pads & bodies) and bodies
                and tilt < LEVEL_LIMIT_DEG
                and float(np.linalg.norm(velocity)) < 0.02):
            self.stable_release += self.model.opt.timestep
        else:
            self.stable_release = 0.0

    def state(self):
        return {"name": "bimanual-plate-lift",
                "controller": "Two-arm scripted physical plate rim grasp",
                "status": self.status, "reason": self.reason, "attempt": 1,
                "phase": self.phases[max(self.phase, 0)][0],
                "elapsed_s": float(self.data.time - self.started),
                "progress": min(float(self.data.time - self.started)
                                / sum(t for _, t in self.phases), 1),
                "four_pad_contact_s": self.quad_s,
                "peak_lift_m": self.peak_lift,
                "sustained_lift_s": self.sustained_lift,
                "peak_tilt_during_lift_deg": self.peak_tilt_deg,
                "tilt_at_longest_lift_deg": self.tilt_at_peak_lift_deg,
                "stable_release_s": self.stable_release,
                "placement_error_m": float(np.linalg.norm(
                    self.data.xpos[self.body][:2] - self.mat[:2])),
                "workcell": {"plate_start": [float(v) for v in self.start],
                             "plate_mat": [float(v) for v in self.mat]},
                "events": self.events,
                "limits": "Scripted two-arm teacher in its own workcell; not a learned "
                          "policy and not the full dinner-table sequence"}


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
            prepare_bimanual_workcell(sim, seed)
            sim.task = BimanualPlateLift(sim, close_gripper=not args.open_gripper)
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
                "peak_tilt_during_lift_deg", "stable_release_s",
                "placement_error_m")}), flush=True)
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"task": "bimanual-plate-lift",
         "successes": sum(r["status"] == "succeeded" for r in results),
         "episodes": len(results), "negative_control": args.open_gripper,
         "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
