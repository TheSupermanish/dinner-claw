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

from tabletop_vla.sim.cutlery import UNREACHED
from tabletop_vla.sim.kinematics import JOINTS, gripper_geometry, solve_pose
from tabletop_vla.sim.reach import _cartesian

# Most IK-tolerant plate start found by the probe sweep (seed 30): every
# approach/height variant accepts here, unlike the table centre which the left
# arm cannot reach at all. Still distinct from plate_mat [-0.20, 0.12].
PLATE_START = np.array([-0.28, 0.05, 0.752])
PLATE_JITTER = 0.006
# Plate rim radius is 0.10 m; aim 10 mm inside the east edge at top-face level.
# Measured 2026-09-15 IN THE PLATE WORKCELL (the earlier -0.09 choice was measured at the
# plate's default reset position x=-0.115, but prepare_plate_workcell parks it at x=-0.28,
# so the wrong rim side was picked). Both sides are reachable at nominal poses; what
# decides it is DRIFT. The plate is dragged about 24 mm outward during the grasp, so the
# grip must start with margin:
#   +0.09 -> grip at x ~ -0.19, drift stays well inside the envelope
#   -0.09 -> grip at x ~ -0.37, drift pushes "Raise to transit height" past the edge
# Measured angles for +0.09: hover t10/a90, close t0/a0, lift t15/a90, transit t10/a90,
# carry t40/a75, place t35/a75.
RIM_DX = 0.09
# Carry height: 0.80 is accepted for the corrected rim side; 0.86 was not tested there.
TRANSIT_Z = 0.80
RIM_DZ = 0.008
# Authored site position. The live site is read at episode start, because the plate
# workcell relocates it clear of the drawer (see prepare_plate_workcell).
MAT = np.array([-0.20, 0.12, 0.738])
RESTING_Z = 0.739
# 20/98 is what actually carries the hover, descend and close phases. The measured
# t0/a45 applies to the rim resting ON the table, not to the hover 40 mm above it;
# substituting it broke "Hover above rim" on all ten seeds.
GRASP_TILT, GRASP_AZ = 10, 90
MAT_TILT, MAT_AZ = 10, 45
# Retract is split in two so the open jaws clear the rim VERTICALLY before moving
# back. Measured 2026-09-15 from the relocated mat (release xy [-0.05, 0.0]): the
# single diagonal move (60 mm back while only 50 mm above the mat) swept the open
# jaws through the rim and dragged the placed plate 54-129 mm off its mat. The
# straight-up pose [-0.05, 0.0, 0.789] solves at t10/a45 (pe 0.00018) and the
# up-0.05-back-0.06 pose [-0.11, -0.04, 0.789] solves at t0/a0 (pe 0.00001), so
# each sub-phase carries its own measured preferred angle (the _waypoint window
# is only +/-20 tilt, +/-36 azimuth, so sharing one angle strands one of them).
RETRACT_LIFT_TILT, RETRACT_LIFT_AZ = 10, 45
RETRACT_TILT, RETRACT_AZ = 0, 0
LIFT_TILT, LIFT_AZ = 10, 90
LIFT_PHASES = frozenset({"Lift clear of table", "Raise to transit height"})
MAT_PHASES = frozenset({"Carry to mat", "Descend to mat", "Release", "Retract lift",
                        "Retract back", "Verify placement"})
# Both retract halves belong here even though the jaws are already open. Measured 2026-09-15 at
# the retract pose [-0.11, 0.12, 0.828]: with the closing axis constrained there are 8
# accepted approach angles (best t45/a75, pe 0.00096, ae 0.0142); with a FREE wrist
# there are zero. Constraining the wrist helps here rather than hurting.
ALIGNED_PHASES = frozenset({"Hover above rim", "Descend to rim", "Close on rim",
                            "Carry to mat", "Descend to mat", "Release",
                            "Retract lift", "Retract back"})
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
    # Measured 2026-09-15: the authored plate_mat at [-0.20, 0.12] is NOT clear of the
    # drawer. The plate has a 100 mm radius so it reaches y 0.22, and the drawer spans
    # y 0.167-0.333 at its default pose, so the plate came to rest ON the drawer, 22-35 mm
    # above the table. The release gate's height check is what exposed it; without that
    # check it looked like a placement-tolerance problem. Same failure the fork mat had.
    # [-0.14, 0.0] clears the drawer by the plate's own radius and solves at carry t10/a45.
    plate_mat = np.array([-0.14, 0.0, 0.737])
    sim.model.site_pos[sim.model.site("plate_mat").id] = plate_mat
    mujoco.mj_forward(sim.model, sim.data)
    sim.variation["plate_workcell"] = {"plate_start": start.tolist(),
                                       "plate_mat_moved_clear_of_drawer": plate_mat.tolist()}


def _waypoint(model, data, arm, target, tilt, az, initial, closing=None):
    """Nearby approach angles are tried before giving up; chaining `initial` keeps
    the arm on one IK branch so the joint-space path between waypoints stays short."""
    # Measured 2026-09-14: the narrow +/-20 tilt, +/-36 azimuth search left phases
    # rejected by 0.007-0.088, all near misses on the axis gate, and hand-tuning a
    # constant per phase fixed one phase while breaking another. Sweeping broadly from
    # the preferred angle outward is what reach.py does and it generalises across seeds.
    # Returning the FIRST accepted angle makes the sweep width actively harmful: a wider
    # search finds an accepted but awkward wrist earlier, and the chained `initial=` then
    # carries that branch into the next phase, which fails. Measured 2026-09-14: widening
    # the sweep alone moved failures from "Raise to transit" back to "Hover above rim".
    # So collect every accepted angle and take the one CLOSEST IN JOINT SPACE to where the
    # arm already is, which keeps the whole chain on one branch.
    reference = np.asarray(initial, dtype=float) if initial is not None else None
    if reference is None:
        reference = np.array([float(data.qpos[int(model.joint(f"{arm}_{n}").qposadr[0])])
                              for n in JOINTS])
    accepted = []
    # Kept NARROW deliberately. Measured 2026-09-14: widening to +/-40 tilt and +/-60
    # azimuth moved failures EARLIER, from "Raise to transit height" back to "Hover above
    # rim", because a wider search reaches an accepted but contorted wrist sooner and the
    # chained `initial=` carries that branch forward. Narrow sweep + nearest-branch wins.
    for delta_t in (0, -10, 10, -20, 20):
        for delta_a in (0, -18, 18, -36, 36):
            tilt_candidate = tilt + delta_t
            if not 0 <= tilt_candidate <= 80:
                continue
            solution = solve_pose(model, data, arm, target,
                                  approach=_cartesian(tilt_candidate, az + delta_a),
                                  closing=closing, initial=initial)
            if solution.accepted:
                accepted.append(solution)
    if accepted:
        return min(accepted,
                   key=lambda s: float(np.linalg.norm(np.asarray(s.targets) - reference)))
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
        site = self.model.site("plate_mat")
        self.mat = np.array([float(site.pos[0]), float(site.pos[1]), RESTING_Z])
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
            ("Retract lift", 1.2),
            ("Retract back", 1.2),
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
            return np.array([rim[0], rim[1], TRANSIT_Z])
        if name == "Carry to mat":
            aim = self.mat + self.body_to_rim()
            return np.array([aim[0], aim[1], TRANSIT_Z])
        if name in {"Descend to mat", "Release"}:
            return self.mat + self.body_to_rim() + [0, 0, 0.005]
        if name in {"Retract lift", "Retract back"}:
            # Retract from where the jaws ACTUALLY are, not from where the mat says
            # they should be. `body_to_rim()` is a fixed 90 mm offset, so a retract
            # aimed at `mat + body_to_rim()` commands a sideways move equal to however
            # far the plate drifted during the carry, and that sideways move IS the
            # drag. Measured 2026-09-15 on seeds 40-49: splitting the retract into
            # lift-then-back while still aiming at the mat left the placement error at
            # 32-135 mm (1/10), statistically the same as the single diagonal move it
            # replaced; the two runs where the retract failed to solve and therefore
            # never executed left the plate at 7-49 mm. The phase is entered exactly
            # once, so reading the live jaw midpoint here reads the release pose for
            # "Retract lift" and the already-lifted pose for "Retract back".
            here = gripper_geometry(self.model, self.data, "left").midpoint
            straight_up = np.array([here[0], here[1], self.mat[2] + 0.05])
            if name == "Retract lift":
                return straight_up
            return straight_up + [-0.06, -0.04, 0.0]
        return None

    def body_to_rim(self):
        """The jaws hold the rim, but the placement gate scores the body centre,
        which sits about 90 mm away. Aim the rim so the BODY lands on the mat."""
        return np.array([RIM_DX, 0.0, 0.0])

    def axis_for(self, name):
        """Preferred approach angle per phase; `_waypoint` searches around it.

        One axis for every phase does not work here. Measured 2026-09-14 for the -x rim:
        the pickup wants t0/a45 (pe 0.00006), the lift and transit want t10/a90
        (pe 0.00008), and the mat side wants t40/a106 (pe 0.00061). Using the grasp axis
        for the lift left "Raise to transit height" rejected on every seed, because the
        search only covers +/-20 tilt and +/-36 azimuth around its starting guess.
        """
        if name == "Retract lift":
            return (RETRACT_LIFT_TILT, RETRACT_LIFT_AZ)
        if name == "Retract back":
            return (RETRACT_TILT, RETRACT_AZ)
        if name in MAT_PHASES:
            return (MAT_TILT, MAT_AZ)
        return (GRASP_TILT, GRASP_AZ)

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
            self.fail(f"{UNREACHED} {name} within the search window: "
                      f"position error {solution.position_error:.4f} m, "
                      f"axis error {solution.axis_error:.4f}")
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
        # Height alone is not evidence that the jaws bear the weight. The world body
        # carries the table and the cabinet, and a plate resting on either while the
        # pads merely touch it is not a lift; this is the check whose absence let a
        # fork resting on the open drawer score as retrieved.
        unsupported = held and 0 not in bodies
        if unsupported and self.lift > 0.025:
            self.lift_run += self.model.opt.timestep
            self.sustained_lift = max(self.sustained_lift, self.lift_run)
        else:
            self.lift_run = 0.0
        position = self.data.xpos[self.body]
        velocity = self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[self.body]]:][:6]
        # Height is not optional in this gate. The cutlery gate shipped without it earlier
        # today and passed eleven seeds while the fork was resting on the OPEN DRAWER,
        # 60 mm above the table. Same check as scoring.py:37 and cutlery.py.
        on_table = abs(float(position[2]) - float(self.mat[2])) < 0.008
        if (np.linalg.norm(position[:2] - self.mat[:2]) < 0.03 and on_table
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
