"""Local simulation state, real actuator control, and position-only IK diagnostics."""

from __future__ import annotations

import io
from importlib.util import find_spec

import mujoco
import numpy as np
from PIL import Image

from tabletop_vla.sim.build_scene import PROJECT, build_scene


class Simulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(build_scene()))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.policy_renderer = None
        self.compiled_policy = None
        self.learned_available = (find_spec("openvino") is not None and
                                  (PROJECT / "outputs/act-openvino/act-fp32.xml").is_file() and
                                  (PROJECT / "outputs/act-openvino/act-fp32.bin").is_file())
        self.original_mass = self.model.body_mass.copy()
        self.original_inertia = self.model.body_inertia.copy()
        self.original_friction = self.model.geom_friction.copy()
        self.original_light = self.model.light_diffuse.copy()
        self.original_geom_size = self.model.geom_size.copy()
        self.original_geom_pos = self.model.geom_pos.copy()
        self.original_rgba = self.model.geom_rgba.copy()
        self.original_iquat = self.model.body_iquat.copy()
        self.original_ipos = self.model.body_ipos.copy()
        self.original_body_pos = self.model.body_pos.copy()
        self.original_damping = self.model.dof_damping.copy()
        self.original_frictionloss = self.model.dof_frictionloss.copy()
        self.original_disableflags = self.model.opt.disableflags
        self.reset(0)

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 1000000:
            raise ValueError("Seed must be an integer between 0 and 1000000")
        self.seed = seed
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[:] = self.original_mass
        self.model.body_inertia[:] = self.original_inertia
        self.model.geom_friction[:] = self.original_friction
        self.model.light_diffuse[:] = self.original_light * rng.uniform(0.8, 1.15)
        self.model.geom_size[:] = self.original_geom_size
        self.model.geom_pos[:] = self.original_geom_pos
        self.model.geom_rgba[:] = self.original_rgba
        self.model.body_iquat[:] = self.original_iquat
        self.model.body_ipos[:] = self.original_ipos
        self.model.body_pos[:] = self.original_body_pos
        self.model.dof_damping[:] = self.original_damping
        self.model.dof_frictionloss[:] = self.original_frictionloss
        self.model.opt.disableflags = self.original_disableflags
        for obj in ("plate", "mug", "bottle", "fork", "spoon"):
            body = self.model.body(obj)
            jid = int(body.jntadr[0])
            address = int(self.model.jnt_qposadr[jid])
            jitter = 0.004 if obj in {"fork", "spoon"} else 0.012
            self.data.qpos[address : address + 2] += rng.uniform(-jitter, jitter, 2)
            scale = rng.uniform(0.8, 1.2)
            self.model.body_mass[body.id] *= scale
            self.model.body_inertia[body.id] *= scale
            geoms = self.model.geom_bodyid == body.id
            self.model.geom_friction[geoms, 0] = rng.uniform(0.6, 1.2)
        # Bounded rectangular-cup shape variation. Preserve mass while transforming
        # the inertia tensor consistently with the scaled compound geometry.
        scale = rng.uniform([0.90, 0.95, 0.98], [1.10, 1.05, 1.02])
        cup = self.model.body("mug").id
        cup_geoms = self.model.geom_bodyid == cup
        self.model.geom_size[cup_geoms] *= scale
        self.model.geom_pos[cup_geoms] *= scale
        # Wall boxes at 90/270 degrees store their local x/y sizes rotated.
        for k in (1, 3):
            gid = self.model.geom(f"mug_wall_{k}").id
            self.model.geom_size[gid] = self.original_geom_size[gid] * scale[[1, 0, 2]]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.model.body_iquat[cup])
        rotation = rotation.reshape(3, 3)
        tensor = rotation @ np.diag(self.model.body_inertia[cup]) @ rotation.T
        covariance = np.eye(3) * np.trace(tensor) / 2 - tensor
        covariance = np.diag(scale) @ covariance @ np.diag(scale)
        tensor = np.eye(3) * np.trace(covariance) - covariance
        values, vectors = np.linalg.eigh(tensor)
        if np.linalg.det(vectors) < 0:
            vectors[:, 0] *= -1
        self.model.body_inertia[cup] = values
        mujoco.mju_mat2Quat(self.model.body_iquat[cup], vectors.flatten())
        self.model.body_ipos[cup] *= scale
        cup_q = self.model.jnt_qposadr[self.model.body_jntadr[cup]]
        self.data.qpos[cup_q + 2] = 0.735 + 0.025 * scale[2] + 0.0003
        for name in ("floor", "table"):
            gid = self.model.geom(name).id
            self.model.geom_rgba[gid, :3] *= rng.uniform(0.75, 1.2, 3)
        self.variation = {"cup_dimensions_m": (0.05 * scale).tolist(),
                          "background": "seeded table/floor colors"}
        randomized_qpos = self.data.qpos.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = randomized_qpos
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = 0
        self.last_motion = None
        self.steps = 0
        self.running = False
        self.demo = None
        self.task = None
        return self.state()

    def start_task(self, seed, name="cup-place", close_gripper=True):
        if name not in {"cup-place", "camera-cup-place", "learned-cup-place",
                        "learned-cup-safe-place", "drawer-open",
                        "fork-retrieve", "spoon-retrieve", "plate-place-left"}:
            raise ValueError("Only the implemented cup tasks are currently executable")
        if name.startswith("learned-"):
            if not close_gripper:
                raise ValueError("Open-gripper control is supported only for the scripted teacher")
            if not self.learned_available:
                raise ValueError("Install the Intel extra, then train and export the ACT checkpoint first")
        from tabletop_vla.sim.manipulation import CupPlace

        self.reset(seed)
        if name == "drawer-open":
            from tabletop_vla.sim.drawer import DrawerOpen, prepare_workcell

            prepare_workcell(self, seed)
            self.task = DrawerOpen(self, close_gripper=close_gripper)
        elif name.endswith("-retrieve"):
            from tabletop_vla.sim.cutlery import CutleryRetrieve
            from tabletop_vla.sim.drawer import prepare_workcell

            prepare_workcell(self, seed)
            self.task = CutleryRetrieve(self, name.split("-")[0], close_gripper=close_gripper)
        elif name == "plate-place-left":
            from tabletop_vla.sim.plate import PlatePlace, prepare_plate_workcell

            prepare_plate_workcell(self, seed)
            self.task = PlatePlace(self, close_gripper=close_gripper)
        elif name.startswith("learned-"):
            from tabletop_vla.sim.learned import LearnedCup

            self.task = LearnedCup(self, safe_finish=name == "learned-cup-safe-place")
        else:
            self.task = CupPlace(self, close_gripper=close_gripper, vision=name == "camera-cup-place")
        self.running = True
        return self.state()

    def playback(self, running):
        if not isinstance(running, bool):
            raise TypeError("running must be true or false")
        self.running = running
        return self.state()

    def start_demo(self, seed):
        self.reset(seed)
        self.demo = {
            "status": "running",
            "phase": "Raise both arms",
            "progress": 0.0,
            "elapsed_s": 0.0,
            "duration_s": 8.0,
            "description": "Motor motion demonstration; no object grasping",
            "peak_tip_displacement_m": {"left": 0.0, "right": 0.0},
        }
        self.demo_initial_tips = {
            arm: self.data.site(f"{arm}_gripperframe").xpos.copy() for arm in ("left", "right")
        }
        self.demo_poses = np.array(
            [
                [0, 0, 0, 0, 0, 0] * 2,
                [0, -0.1, -0.3, -0.15, 0, 0.6] * 2,
                [0.25, -0.1, -0.3, -0.15, 0.4, 0.6, -0.25, -0.1, -0.3, -0.15, -0.4, 0.6],
                [0, -0.1, -0.3, -0.15, 0, 0.1] * 2,
                [0, 0, 0, 0, 0, 0] * 2,
            ],
            dtype=float,
        )
        self.running = True
        return self.state()

    def _demo_targets(self):
        if not self.demo or self.demo["status"] != "running":
            return
        elapsed = float(self.data.time)
        phase = min(int(elapsed / 2), 3)
        blend = np.clip((elapsed - phase * 2) / 2, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        self.data.ctrl[:] = (1 - blend) * self.demo_poses[phase] + blend * self.demo_poses[
            phase + 1
        ]
        self.demo.update(
            phase=["Raise both arms", "Rotate outward", "Close grippers", "Return home"][phase],
            elapsed_s=elapsed,
            progress=min(elapsed / 8, 1.0),
        )
        for arm in ("left", "right"):
            tip = self.data.site(f"{arm}_gripperframe").xpos
            travel = float(np.linalg.norm(tip - self.demo_initial_tips[arm]))
            self.demo["peak_tip_displacement_m"][arm] = max(
                self.demo["peak_tip_displacement_m"][arm], travel
            )
        if elapsed >= 8:
            self.demo.update(status="completed", phase="Motion demo complete", progress=1.0)
            self.running = False

    def cancel_demo(self):
        if self.task:
            self.task.cancel()
        if self.demo and self.demo["status"] == "running":
            self.demo["status"] = "cancelled"
            self.demo["phase"] = "Manual control"

    def step(self, count=25, observer=None):
        if not isinstance(count, int) or not 1 <= count <= 500:
            raise ValueError("Physics steps must be between 1 and 500")
        processed = 0
        for _ in range(count):
            self._demo_targets()
            active_task = self.task and self.task.status == "running"
            if self.task:
                self.task.before_step()
                if active_task and self.task.status != "running":
                    break
            if observer is not None:
                observer(self)
            mujoco.mj_step(self.model, self.data)
            processed += 1
            if self.task:
                self.task.after_step()
                if active_task and self.task.status != "running":
                    break
        self.steps += processed
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Simulation state became non-finite; reset required")
        return self.state()

    def control(self, name, value):
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        if name not in names or not np.isfinite(value):
            raise ValueError("Unknown actuator or non-finite target")
        actuator = self.model.actuator(name)
        low, high = self.model.actuator_ctrlrange[actuator.id]
        if not low <= value <= high:
            raise ValueError("Joint target exceeds the official actuator limits")
        self.cancel_demo()
        self.data.ctrl[actuator.id] = value
        return self.state()

    def reach(self, arm, target):
        """Compute joint targets in scratch data; never teleport the live robot."""
        if arm not in {"left", "right"}:
            raise ValueError("Arm must be left or right")
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target must contain three finite coordinates")
        if np.any(target < [-0.5, -0.4, 0.8]) or np.any(target > [0.5, 0.4, 1.35]):
            raise ValueError("Reach diagnostics are limited to the space above the table")
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        joints = [
            self.model.joint(f"{arm}_{name}")
            for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        qids = [int(j.qposadr[0]) for j in joints]
        vids = [int(j.dofadr[0]) for j in joints]
        sid = self.model.site(f"{arm}_gripperframe").id
        jac = np.zeros((3, self.model.nv))
        for _ in range(160):
            mujoco.mj_forward(self.model, scratch)
            error = target - scratch.site_xpos[sid]
            if np.linalg.norm(error) < 0.002:
                break
            mujoco.mj_jacSite(self.model, scratch, jac, None, sid)
            j = jac[:, vids]
            delta = j.T @ np.linalg.solve(j @ j.T + 0.0004 * np.eye(3), error)
            for k, joint in enumerate(joints):
                scratch.qpos[qids[k]] = np.clip(
                    scratch.qpos[qids[k]] + np.clip(delta[k], -0.08, 0.08), *joint.range
                )
        mujoco.mj_forward(self.model, scratch)
        residual = float(np.linalg.norm(target - scratch.site_xpos[sid]))
        accepted = residual < 0.008
        if accepted:
            for joint, qid in zip(joints, qids):
                self.control(joint.name, float(scratch.qpos[qid]))
        self.last_motion = {
            "arm": arm,
            "target": target.tolist(),
            "ik_residual_m": residual,
            "accepted": accepted,
            "kind": "position_only_diagnostic",
        }
        return self.state()

    def state(self):
        joints = []
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            joints.append(
                {
                    "name": self.model.actuator(i).name,
                    "actual": float(self.data.qpos[self.model.jnt_qposadr[jid]]),
                    "target": float(self.data.ctrl[i]),
                    "range": self.model.actuator_ctrlrange[i].tolist(),
                }
            )
        motion = dict(self.last_motion) if self.last_motion else None
        if motion:
            site = self.data.site(f"{motion['arm']}_gripperframe").xpos
            motion["actual_error_m"] = float(np.linalg.norm(np.array(motion["target"]) - site))
        return {
            "running": self.running,
            "demo": self.demo,
            "task": self.task.state() if self.task else None,
            "seed": self.seed,
            "variation": self.variation,
            "time": float(self.data.time),
            "steps": self.steps,
            "joints": joints,
            "contacts": int(self.data.ncon),
            "last_motion": motion,
            "end_effectors": {
                arm: self.data.site(f"{arm}_gripperframe").xpos.tolist()
                for arm in ("left", "right")
            },
            "objects": {
                name: self.data.body(name).xpos.tolist()
                for name in ("plate", "mug", "bottle", "fork", "spoon")
            },
            "capabilities": {
                "physics": True,
                "manual_control": True,
                "vision_model": False,
                "learned_policy": self.learned_available,
                "learned_policy_status": "experimental" if self.learned_available else "not installed",
                "task_execution": True,
                "task_scope": ["cup-place", "camera-cup-place", "drawer-open"] + (
                    ["learned-cup-place", "learned-cup-safe-place"] if self.learned_available else []),
                "calibrated_color_vision": True,
                "intel_benchmark": False,
            },
        }

    def rgb(self, camera="overview"):
        if camera not in {"overview", "top"}:
            raise ValueError("Unknown camera")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=600, width=900)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def frame(self, camera="overview"):
        out = io.BytesIO()
        Image.fromarray(self.rgb(camera)).save(out, format="JPEG", quality=85)
        return out.getvalue()

    def close(self):
        if self.renderer:
            self.renderer.close()
        if self.policy_renderer:
            self.policy_renderer.close()
