"""Experimental ACT/OpenVINO task: RGB and joint observations, no teacher actions."""

import time
from collections import deque

import mujoco
import numpy as np
from PIL import Image

from tabletop_vla.sim.build_scene import PROJECT
from tabletop_vla.sim.kinematics import GRASP_POINT, solve_down
from tabletop_vla.sim.scoring import PhysicalScore

MODEL_PATH = PROJECT / "outputs/act-openvino/act-fp32.xml"


class LearnedCup:
    def __init__(self, sim, blind=False, safe_finish=False):
        import openvino as ov

        if not MODEL_PATH.is_file():
            raise ValueError("Train and export ACT first; no learned checkpoint is installed")
        self.sim = sim
        if sim.compiled_policy is None:
            sim.compiled_policy = ov.Core().compile_model(str(MODEL_PATH), "CPU", {
                "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY",
            })
        if sim.policy_renderer is None:
            sim.policy_renderer = mujoco.Renderer(sim.model, height=480, width=720)
        self.compiled = sim.compiled_policy
        self.renderer = sim.policy_renderer
        self.qids = sim.model.jnt_qposadr[sim.model.actuator_trnid[:, 0]]
        self.actions = deque()
        self.started = float(sim.data.time)
        self.next_action = self.started
        self.status, self.reason = "running", None
        self.scorer = PhysicalScore(sim)
        self.clipped = self.calls = 0
        self.latencies = []
        self.blind = blind
        self.safe_finish = safe_finish
        self.retreat_target = None
        self.duration = 21 if safe_finish else 18

    def fail(self, reason):
        self.status, self.reason = "failed", reason
        self.sim.running = False

    def cancel(self):
        if self.status in {"running", "succeeded"}:
            self.status, self.reason = "cancelled", "Manual control or another task"

    def before_step(self):
        if self.status != "running":
            return
        sim = self.sim
        elapsed = float(sim.data.time - self.started)
        if elapsed >= self.duration - 1e-9:
            result = self.scorer.result()
            self.status = "succeeded" if result["success"] else "failed"
            if not result["success"]:
                self.reason = "Learned rollout did not satisfy lift plus stable released-placement gates"
            sim.running = False
            return
        if self.safe_finish and elapsed >= 18 - 1e-9:
            # A separately labelled post-policy safety primitive. It uses robot
            # kinematics only; no cup pose, task phase, or scoring state is input.
            if self.retreat_target is None:
                body = sim.data.body("right_gripper")
                target = body.xpos + body.xmat.reshape(3, 3) @ GRASP_POINT
                target[2] += 0.01
                solution = solve_down(sim.model, sim.data, "right", target)
                if not solution.accepted:
                    self.fail("Post-policy retreat is unreachable; stopped")
                    return
                self.retreat_start = sim.data.ctrl.copy()
                self.retreat_target = self.retreat_start.copy()
                self.retreat_target[6:11] = solution.targets
                self.retreat_target[11] = 0.85
            blend = np.clip((elapsed - 18) / 1.5, 0, 1)
            blend = blend * blend * (3 - 2 * blend)
            sim.data.ctrl[:] = (1 - blend) * self.retreat_start + blend * self.retreat_target
            return
        if sim.data.time + 1e-9 < self.next_action:
            return
        self.next_action += 0.05
        if not self.actions:
            self.renderer.update_scene(sim.data, camera="overview")
            rgb = np.array(Image.fromarray(self.renderer.render()).resize((224, 224)))
            if self.blind:
                rgb[:] = 0
            observation = {"rgb": rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255,
                           "proprio": sim.data.qpos[self.qids].astype(np.float32)[None]}
            started = time.perf_counter()
            output = self.compiled(observation)[self.compiled.output(0)][0].copy()
            self.latencies.append((time.perf_counter() - started) * 1000)
            self.calls += 1
            if output.shape != (50, 12) or not np.isfinite(output).all():
                self.fail("Invalid learned action chunk")
                return
            self.actions.extend(output)
        action = self.actions.popleft()
        bounded = np.clip(action, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        self.clipped += int(np.any(action != bounded))
        sim.data.ctrl[:] = bounded

    def after_step(self):
        if self.status == "running":
            self.scorer.update()

    def state(self):
        result = self.scorer.result()
        return {"name": "learned-cup-safe-place" if self.safe_finish else "learned-cup-place",
                "controller": "LeRobot ACT / OpenVINO CPU FP32" + (
                    " + scripted post-policy retreat" if self.safe_finish else ""),
                "status": self.status, "reason": self.reason,
                "phase": ("Post-policy retreat and settle" if self.retreat_target is not None
                          else "Experimental camera-to-action policy"), "attempt": 1,
                "elapsed_s": float(self.sim.data.time - self.started),
                "progress": min(float(self.sim.data.time - self.started) / self.duration, 1),
                **result, "stable_released_s": result["stable_release_s"],
                "clipped_action_steps": self.clipped, "model_calls": self.calls,
                "inference_ms_mean": float(np.mean(self.latencies)) if self.latencies else None,
                "blind": self.blind,
                "scripted_finish": self.safe_finish,
                "limits": "Experimental single cup skill. No VLM decisions or bimanual completion."}
