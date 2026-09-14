"""Scorer-only simulator state; these values must not enter policy observations."""

import mujoco
import numpy as np


class PhysicalScore:
    def __init__(self, sim):
        self.sim = sim
        self.object_id = sim.model.body("mug").id
        self.fingers = {sim.model.body("right_gripper").id,
                        sim.model.body("right_moving_jaw_so101_v1").id}
        self.expected_z = 0.735 + sim.variation["cup_dimensions_m"][2] / 2
        self.peak_lift = self.lift_run = self.sustained_lift = self.stable_release = 0.0

    def update(self):
        model, data = self.sim.model, self.sim.data
        bodies = set()
        force = np.empty(6)
        for i, contact in enumerate(data.contact):
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.object_id in (a, b):
                mujoco.mj_contactForce(model, data, i, force)
                if force[0] > 0.02:
                    bodies.add(int(b if a == self.object_id else a))
        pos = data.xpos[self.object_id]
        upright = data.xmat[self.object_id].reshape(3, 3)[2, 2]
        lift = float(pos[2] - self.expected_z)
        self.peak_lift = max(self.peak_lift, lift)
        if lift > 0.025 and upright > 0.9 and self.fingers <= bodies and 0 not in bodies:
            self.lift_run += model.opt.timestep
            self.sustained_lift = max(self.sustained_lift, self.lift_run)
        else:
            self.lift_run = 0.0
        vid = model.jnt_dofadr[model.body_jntadr[self.object_id]]
        if (np.linalg.norm(pos[:2] - [0.27, -0.04]) < 0.02 and upright > 0.95
                and abs(pos[2] - self.expected_z) < 0.003 and 0 in bodies
                and not (self.fingers & bodies) and np.linalg.norm(data.qvel[vid:vid + 6]) < 0.015):
            self.stable_release += model.opt.timestep
        else:
            self.stable_release = 0.0

    def result(self):
        return {"success": self.sustained_lift >= 0.4 and self.stable_release >= 0.3,
                "peak_lift_m": self.peak_lift, "sustained_lift_s": self.sustained_lift,
                "stable_release_s": self.stable_release,
                "placement_error_m": float(np.linalg.norm(
                    self.sim.data.xpos[self.object_id, :2] - [0.27, -0.04]))}
