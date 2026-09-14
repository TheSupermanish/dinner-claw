"""Ground-truth teacher with physical postconditions, never object attachments."""

from __future__ import annotations

import mujoco
import numpy as np

from tabletop_vla.sim.kinematics import JOINTS, solve_down


class CupPlace:
    phases = (
        ("Settle", 0.6), ("Approach", 2.0), ("Descend", 2.0),
        ("Close fingers", 2.0), ("Lift", 2.0), ("Hold and verify", 0.6),
        ("Carry", 2.0), ("Lower", 2.0), ("Release", 2.0),
        ("Retreat", 2.0), ("Verify placement", 0.6),
    )

    def __init__(self, sim, close_gripper=True, vision=False):
        self.sim = sim
        self.model, self.data = sim.model, sim.data
        self.arm = "right"
        self.object_id = self.model.body("mug").id
        self.fingers = {self.model.body("right_gripper").id,
                        self.model.body("right_moving_jaw_so101_v1").id}
        self.actuators = [self.model.actuator(f"right_{name}").id for name in JOINTS]
        self.grip = self.model.actuator("right_gripper").id
        self.grip_q = int(self.model.joint("right_gripper").qposadr[0])
        self.close_gripper = close_gripper
        self.vision = vision
        self.observation = None
        self.phase = -1
        self.started = float(self.data.time)
        self.attempt_started = self.started
        self.attempt = 1
        self.phase_started = self.started
        self.latched = False
        self.status = "running"
        self.reason = None
        self.initial = None
        self.target = np.array([0.27, -0.04, 0.760])
        self.half_size = np.array([0.025, 0.025, 0.004])
        if not vision:
            self.half_size = self.model.geom("mug_geom").size.copy()
            self.target[2] = 0.735 + sim.variation["cup_dimensions_m"][2] / 2
        self.lifted_seconds = 0.0
        self.current_lift_seconds = 0.0
        self.peak_lift = 0.0
        self.release_seconds = 0.0
        self.bilateral_steps = 0
        self.events = []
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()

    def contact_bodies(self):
        result = set()
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.object_id not in (a, b):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if force[0] > 0.02:
                result.add(int(b if a == self.object_id else a))
        return result

    def fail(self, reason):
        self.status = "failed"
        self.reason = reason
        self.sim.running = False

    def cancel(self):
        if self.status in {"running", "succeeded"}:
            self.status, self.reason = "cancelled", "Manual control or another task"

    def _enter(self, phase):
        self.phase = phase
        self.phase_started = float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        name = self.phases[phase][0]
        self.events.append({"phase": name, "time_s": self.phase_started})
        if phase == 0:
            if self.vision:
                # Move the arm aside before observing, so the cup is not hidden
                # below the forearm. This pose is independent of object position.
                self.end[self.actuators[0]] = 0.6
                self.end[self.grip] = 0.85
            return
        if phase == 1:
            self.initial = self.data.xpos[self.object_id].copy()
            self.pick_xy = self.initial[:2].copy()
            if self.vision:
                from tabletop_vla.perception.cup import locate_blue_cup

                try:
                    self.observation = locate_blue_cup(self.sim.rgb("top"))
                except ValueError as exc:
                    self.fail(str(exc))
                    return
                # Only pixels/calibration choose the pickup location and width.
                # self.initial is still retained for the independent lift scorer.
                self.pick_xy = np.array(self.observation["xy"])
                self.half_size[1] = np.clip(self.observation["estimated_width_y_m"] / 2, 0.022, 0.028)
        xy = (self.pick_xy if phase < 6 else self.target[:2]).copy()
        xy[1] += 0.025 - self.half_size[1]
        grasp_height = self.target[2] + 0.012
        z = {1: 0.820, 2: grasp_height, 4: 0.820, 6: 0.820, 7: grasp_height, 9: 0.820}.get(phase)
        if z is not None:
            solution = solve_down(self.model, self.data, self.arm, [*xy, z])
            if not solution.accepted:
                self.fail(f"Unreachable {name}: IK error {solution.position_error:.4f} m")
                return
            self.end[self.actuators] = solution.targets
        if phase in {1, 2, 8, 9, 10} or not self.close_gripper:
            self.end[self.grip] = 0.85

    def before_step(self):
        if self.status != "running":
            return
        if self.phase == -1:
            self._enter(0)
        elapsed = self.data.time - self.phase_started
        if elapsed >= self.phases[self.phase][1] - 1e-9:
            if self.phase == 5 and self.lifted_seconds < 0.4:
                if self.vision and self.attempt < 2:
                    # Retry from the actual failed scene, not a seeded reset. The
                    # arm moves aside and a fresh image selects the next grasp.
                    self.events.append({"phase": "Empty grasp: reobserve and retry",
                                        "time_s": float(self.data.time)})
                    self.attempt += 1
                    self.attempt_started = float(self.data.time)
                    self.initial = None
                    self.latched = False
                    self.lifted_seconds = self.current_lift_seconds = 0.0
                    self.release_seconds = 0.0
                    self._enter(0)
                    return
                self.fail("Cup did not sustain a bilateral-contact lift of at least 25 mm")
                return
            if self.phase == 10:
                if self.release_seconds < 0.3:
                    self.fail("Cup was not stably released upright on the target")
                else:
                    self.status = "succeeded"
                    self.sim.running = False
                return
            self._enter(self.phase + 1)
            elapsed = 0
        if self.status != "running":
            return
        t = min(elapsed / (self.phases[self.phase][1] * 0.7), 1.0)
        t = t * t * (3 - 2 * t)
        self.data.ctrl[:] = self.begin + (self.end - self.begin) * t
        if self.close_gripper and 3 <= self.phase <= 7:
            self.latched |= self.fingers <= self.contact_bodies()
            # Bounded closure effort using the unchanged upstream position servo.
            # Do not drive all the way closed through the object after contact.
            offset = 0.001 if self.latched else 0.002
            self.data.ctrl[self.grip] = np.clip(
                self.data.qpos[self.grip_q] - offset,
                *self.model.actuator_ctrlrange[self.grip],
            )

    def after_step(self):
        if self.status != "running" or self.initial is None:
            return
        bodies = self.contact_bodies()
        pos = self.data.xpos[self.object_id]
        upright = float(self.data.xmat[self.object_id].reshape(3, 3)[2, 2])
        bilateral = self.fingers <= bodies
        self.bilateral_steps += int(bilateral)
        lift = float(pos[2] - self.initial[2])
        self.peak_lift = max(self.peak_lift, lift)
        if 4 <= self.phase <= 6 and lift > 0.025 and bilateral and upright > 0.9 and 0 not in bodies:
            self.current_lift_seconds += self.model.opt.timestep
            self.lifted_seconds = max(self.lifted_seconds, self.current_lift_seconds)
        else:
            self.current_lift_seconds = 0.0
        if self.phase >= 9:
            jid = self.model.body_jntadr[self.object_id]
            vid = self.model.jnt_dofadr[jid]
            stable = np.linalg.norm(self.data.qvel[vid:vid + 6]) < 0.015
            released = not (self.fingers & bodies)
            placed = np.linalg.norm(pos[:2] - self.target[:2]) < 0.02
            supported = abs(pos[2] - self.target[2]) < 0.003 and 0 in bodies
            if stable and released and placed and supported and upright > 0.95:
                self.release_seconds += self.model.opt.timestep
            else:
                self.release_seconds = 0.0
        if pos[2] < 0.71:
            self.fail("Cup fell off the table")

    def state(self):
        position = self.data.xpos[self.object_id]
        return {
            "name": "camera-cup-place" if self.vision else "cup-place",
            "controller": "calibrated_color_vision_scripted" if self.vision else "ground_truth_scripted_teacher",
            "observation": self.observation,
            "status": self.status, "reason": self.reason,
            "attempt": self.attempt,
            "phase": self.phases[max(self.phase, 0)][0],
            "elapsed_s": float(self.data.time - self.started),
            "progress": min(float(self.data.time - self.attempt_started) / sum(t for _, t in self.phases), 1),
            "peak_lift_m": self.peak_lift, "bilateral_contact_steps": self.bilateral_steps,
            "sustained_lift_s": self.lifted_seconds,
            "stable_released_s": self.release_seconds,
            "placement_error_m": float(np.linalg.norm(position[:2] - self.target[:2])),
            "events": list(self.events),
            "limits": "One right-arm cup skill; optional calibrated color vision, no VLM/learned controller/bimanual task",
        }
