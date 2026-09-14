"""Left-arm physical drawer opening; explicit workcell reset and measured pull."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.kinematics import JOINTS, solve_down


def prepare_workcell(sim, seed):
    """Configure a reachable cabinet before the episode starts, never during motion."""
    model, data = sim.model, sim.data
    rng = np.random.default_rng(seed + 7100000)
    center = np.array([-0.23, 0.165, 0.752])
    center[:2] += rng.uniform(-0.006, 0.006, 2)
    model.body_pos[model.body("drawer").id] = center
    handle = model.geom("drawer_handle").id
    model.geom_pos[handle] = [0, -0.165, 0.035]
    model.geom_size[handle] = [0.035, 0.022, 0.012]
    # Runtime geometry relocation invalidates the compiled per-body BVH. Disable
    # that optimization in this small workcell; retain real contact generation.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    for name, x in (("left", -0.03), ("right", 0.03)):
        model.geom_pos[model.geom(f"drawer_mount_{name}").id] = [x, -0.12, 0.02]
    # Measured 2026-09-14: at a 0.09 m top offset the left wrist jams on `cabinet_top`
    # (11-47 N) at every approach angle that reaches the fork, so cutlery retrieval was
    # geometrically impossible. A 0.18 m offset is the smallest swept value at which the
    # gripper reaches the fork with no cabinet contact (0.15 m still jams at 42.9 N).
    for name, offset in (("left", [-0.147, 0, 0.035]),
                         ("right", [0.147, 0, 0.035]),
                         ("back", [0, 0.092, 0.035]),
                         ("top", [0, 0, 0.18])):
        model.geom_pos[model.geom(f"cabinet_{name}").id] = center + offset
    for obj, position in (("plate", [0.0, 0.38, 0.752]),
                          ("fork", center + [-0.045, 0, 0.014]),
                          ("spoon", center + [0.045, 0, 0.014])):
        address = model.jnt_qposadr[model.body_jntadr[model.body(obj).id]]
        data.qpos[address:address + 3] = position
    jid = model.joint("drawer_slide").id
    vid = model.jnt_dofadr[jid]
    model.dof_damping[vid] = rng.uniform(2.5, 5.0)
    model.dof_frictionloss[vid] = rng.uniform(0.15, 0.45)
    drawer = model.body("drawer").id
    mass_scale = rng.uniform(0.8, 1.2)
    model.body_mass[drawer] *= mass_scale
    model.body_inertia[drawer] *= mass_scale
    model.geom_friction[model.geom_bodyid == drawer, 0] = rng.uniform(0.7, 1.1)
    pose = data.qpos.copy()
    mujoco.mj_setConst(model, data)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    sim.variation["drawer"] = {"center": center.tolist(), "damping": float(model.dof_damping[vid]),
                               "slide_friction": float(model.dof_frictionloss[vid]),
                               "mass_scale": mass_scale}


class DrawerOpen:
    phases = (("Settle", 0.6), ("Clear cabinet", 1.5), ("Fold outside cabinet", 2.0),
              ("Approach handle", 2.0), ("Lower to handle", 2.0),
              ("Grip handle", 2.0), ("Pull drawer", 3.0), ("Verify opening", 0.6),
              ("Release handle", 1.5), ("Retract left arm", 2.0), ("Verify released drawer", 0.6))

    def __init__(self, sim, close_gripper=True):
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.status, self.reason = "running", None
        self.started = float(self.data.time)
        self.phase, self.phase_started = -1, self.started
        self.close_gripper = close_gripper
        self.jid = self.model.joint("drawer_slide").id
        self.qid = self.model.jnt_qposadr[self.jid]
        self.vid = self.model.jnt_dofadr[self.jid]
        self.handle = self.model.geom("drawer_handle").id
        self.fingers = {self.model.body("left_gripper").id,
                        self.model.body("left_moving_jaw_so101_v1").id}
        self.actuators = [self.model.actuator(f"left_{j}").id for j in JOINTS]
        self.grip = self.model.actuator("left_gripper").id
        self.grip_q = self.model.joint("left_gripper").qposadr[0]
        self.initial = self.data.geom_xpos[self.handle].copy()
        self.start_q = float(self.data.qpos[self.qid])
        self.bilateral_s = self.current_contact = self.stable_s = self.contact_pull_m = 0.0
        self.latched = False
        self.events = []

    def contacts(self):
        bodies = set()
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if self.handle in (contact.geom1, contact.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if force[0] > 0.02:
                    other = contact.geom2 if contact.geom1 == self.handle else contact.geom1
                    bodies.add(int(self.model.geom_bodyid[other]))
        return bodies

    def fail(self, reason):
        self.status, self.reason = "failed", reason
        self.sim.running = False

    def cancel(self):
        self.status, self.reason = "cancelled", "Manual control or another task"

    def enter(self, phase):
        self.phase, self.phase_started = phase, float(self.data.time)
        self.begin = self.data.ctrl.copy()
        self.end = self.begin.copy()
        self.events.append({"phase": self.phases[phase][0], "time_s": self.phase_started})
        name = self.phases[phase][0]
        if name == "Clear cabinet":
            self.end[self.actuators[0]] = 0.85
        if name in {"Fold outside cabinet", "Approach handle", "Lower to handle", "Pull drawer", "Retract left arm"}:
            target = self.initial.copy()
            target[1] += 0.003  # Adapt the 50 mm tool center to the 44 mm handle.
            target[2] += 0.0 if name in {"Lower to handle", "Pull drawer"} else 0.024
            if name in {"Pull drawer", "Retract left arm"}:
                target[1] -= 0.10
            if name == "Fold outside cabinet":
                target[1] = -0.06
            solution = solve_down(self.model, self.data, "left", target)
            if not solution.accepted:
                self.fail(f"Unreachable {self.phases[phase][0]}: {solution.position_error:.4f} m")
                return
            self.end[self.actuators] = solution.targets
            if name == "Fold outside cabinet":
                self.end[self.actuators[0]] = 0.85
        if name not in {"Grip handle", "Pull drawer", "Verify opening"} or not self.close_gripper:
            self.end[self.grip] = 0.85

    def before_step(self):
        if self.status != "running":
            return
        if self.phase < 0:
            self.enter(0)
        elapsed = float(self.data.time - self.phase_started)
        duration = self.phases[self.phase][1]
        if elapsed >= duration - 1e-9:
            if self.phases[self.phase][0] == "Grip handle" and self.bilateral_s < 0.15:
                self.fail("No sustained two-finger handle grasp")
                return
            if self.phases[self.phase][0] == "Verify opening" and (self.travel < 0.085 or self.contact_pull_m < 0.06):
                self.fail("Drawer did not open far enough through a verified handle pull")
                return
            if self.phase == len(self.phases) - 1:
                if self.stable_s >= 0.3:
                    self.status, self.sim.running = "succeeded", False
                else:
                    self.fail("Drawer was not stably open after release")
                return
            self.enter(self.phase + 1)
            elapsed, duration = 0.0, self.phases[self.phase][1]
        if self.status != "running":
            return
        blend = min(elapsed / (0.8 * duration), 1)
        blend = blend * blend * (3 - 2 * blend)
        self.data.ctrl[:] = (1 - blend) * self.begin + blend * self.end
        if self.close_gripper and self.phases[self.phase][0] in {"Grip handle", "Pull drawer", "Verify opening"}:
            if self.fingers <= self.contacts():
                self.latched = True
            offset = 0.001 if self.latched else 0.002
            self.data.ctrl[self.grip] = np.clip(self.data.qpos[self.grip_q] - offset,
                                               *self.model.actuator_ctrlrange[self.grip])

    @property
    def travel(self):
        return float(self.data.qpos[self.qid] - self.start_q)

    def after_step(self):
        if self.status != "running":
            return
        bodies = self.contacts()
        if self.fingers <= bodies:
            self.current_contact += self.model.opt.timestep
            self.bilateral_s = max(self.bilateral_s, self.current_contact)
            self.contact_pull_m = max(self.contact_pull_m, self.travel)
        else:
            self.current_contact = 0.0
        if self.travel >= 0.085 and not (bodies & self.fingers) and abs(self.data.qvel[self.vid]) < 0.005:
            self.stable_s += self.model.opt.timestep
        else:
            self.stable_s = 0.0

    def state(self):
        return {"name": "drawer-open", "controller": "Left-arm scripted physical handle pull",
                "status": self.status, "reason": self.reason, "attempt": 1,
                "phase": self.phases[max(self.phase, 0)][0],
                "elapsed_s": float(self.data.time - self.started),
                "progress": min(float(self.data.time - self.started) / sum(t for _, t in self.phases), 1),
                "drawer_travel_m": self.travel, "contact_pull_m": self.contact_pull_m,
                "bilateral_contact_s": self.bilateral_s, "stable_released_s": self.stable_s,
                "events": self.events,
                "limits": "Separate reachable workcell; scripted teacher, not learned drawer policy or full sequence"}


def main():
    from tabletop_vla.sim.runtime import Simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-gripper", action="store_true")
    args = parser.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    if not 0 <= start < stop <= 1000001 or stop - start > 1000:
        parser.error("Invalid seed range")
    sim, results = Simulation(), []
    try:
        for seed in range(start, stop):
            sim.start_task(seed, "drawer-open", close_gripper=not args.open_gripper)
            for _ in range(20):
                sim.step(500)
                if sim.task.status != "running":
                    break
            if sim.task.status == "running":
                sim.task.fail("Episode timeout")
            result = {"seed": seed, **sim.task.state(), "variation": sim.variation,
                      "physics_warnings": sim.data.warning.number.tolist()}
            results.append(result)
            print(json.dumps({k: result[k] for k in ("seed", "status", "reason", "drawer_travel_m", "contact_pull_m")}), flush=True)
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"task": "drawer-open", "successes": sum(r["status"] == "succeeded" for r in results),
                                      "episodes": len(results), "negative_control": args.open_gripper,
                                      "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
