"""Record aligned RGB/proprioception/action demonstrations and an honest demo video."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tabletop_vla.sim.runtime import Simulation


class Recorder:
    def __init__(self, sim, video=None):
        self.renderer = mujoco.Renderer(sim.model, height=480, width=720)
        self.images, self.states, self.actions, self.times = [], [], [], []
        self.phases = []
        self.next_frame = 0.0
        self.video = video
        self.qids = sim.model.jnt_qposadr[sim.model.actuator_trnid[:, 0]]

    def __call__(self, sim):
        if sim.data.time + 1e-9 < self.next_frame:
            return
        self.next_frame += 0.05  # 20 Hz action observations; physics remains 500 Hz.
        self.renderer.update_scene(sim.data, camera="overview")
        frame = Image.fromarray(self.renderer.render())
        self.images.append(np.array(frame.resize((224, 224))))
        self.states.append(sim.data.qpos[self.qids].copy())
        self.actions.append(sim.data.ctrl.copy())
        self.times.append(float(sim.data.time))
        self.phases.append(sim.task.state()["phase"])
        if self.video:
            result = sim.task.state()
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 720, 58), fill=(12, 20, 25))
            draw.text((12, 8), f"SEED {sim.seed} | CUP PICK & PLACE | {result['phase']} | "
                      f"{sim.data.time:.2f}s", fill="white")
            draw.text((12, 27), result["controller"] + " / single cup, not full challenge",
                      fill=(150, 220, 200))
            draw.text((12, 43), f"Lift {result['peak_lift_m'] * 100:.1f} cm | "
                      f"Target error {result['placement_error_m'] * 1000:.1f} mm", fill="white")
            self.video.stdin.write(np.asarray(frame).tobytes())

    def final_card(self, sim, result):
        if self.video:
            self.renderer.update_scene(sim.data, camera="overview")
            frame = Image.fromarray(self.renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 720, 78), fill=(12, 20, 25))
            draw.text((12, 8), f"SEED {sim.seed}: {result['status'].upper()} / single-cup skill only", fill="white")
            draw.text((12, 27), result["controller"], fill=(150, 220, 200))
            draw.text((12, 44), f"Lift {result['peak_lift_m'] * 100:.1f} cm / "
                      f"Placement error {result['placement_error_m'] * 1000:.1f} mm", fill="white")
            draw.text((12, 61), result["reason"] or "Physical contact, lift and released-placement gates passed", fill="white")
            for _ in range(40):
                self.video.stdin.write(np.asarray(frame).tobytes())

    def save(self, output, result, variation):
        # Store failed episodes too, explicitly labelled, so data filtering is auditable.
        np.savez_compressed(output, rgb=np.stack(self.images),
                            proprio=np.array(self.states, dtype=np.float32),
                            action=np.array(self.actions, dtype=np.float32),
                            time=np.array(self.times), phase=np.array(self.phases),
                            metadata=json.dumps({"result": result, "variation": variation,
                                                 "camera": "overview", "rate_hz": 20,
                                                 "instruction": "Pick up the cup and place it on its marker"}))

    def close(self):
        self.renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1000:1010")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--task", default="cup-place", choices=(
        "cup-place", "camera-cup-place", "learned-cup-place", "learned-cup-safe-place"))
    args = parser.parse_args()
    if args.task.startswith("learned-") and args.split != "test":
        parser.error("Learned rollout recordings must be labelled test, not teacher demonstrations")
    start, stop = map(int, args.seeds.split(":"))
    if not 0 <= start < stop <= 1000001 or stop - start > 1000:
        parser.error("Use 1–1000 valid seeds")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new one to preserve recordings")
    if args.video and (args.video.exists() or not shutil.which("ffmpeg")):
        parser.error("Video path must be new and ffmpeg must be installed")
    args.output.mkdir(parents=True)
    video = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video = subprocess.Popen([
            "ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "720x480", "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(args.video),
        ], stdin=subprocess.PIPE)
    sim = Simulation()
    results = []
    try:
        for seed in range(start, stop):
            sim.start_task(seed, args.task)
            recorder = Recorder(sim, video)
            try:
                for _ in range(40):
                    sim.step(500, observer=recorder)
                    if sim.task.status != "running":
                        break
                if sim.task.status == "running":
                    sim.task.fail("Recording timeout")
                result = {"seed": seed, **sim.task.state()}
                result["file"] = f"seed-{seed}.npz"
                recorder.save(args.output / result["file"], result, sim.variation)
                recorder.final_card(sim, result)
                results.append(result)
                print(f"Seed {seed}: {result['status']}, {len(recorder.images)} aligned frames", flush=True)
            finally:
                recorder.close()
    finally:
        sim.close()
        if video:
            video.stdin.close()
            if video.wait(timeout=30) != 0:
                raise RuntimeError("Video encoding failed")
        (args.output / "manifest.json").write_text(json.dumps({
            "split": args.split, "task": args.task,
            "controller": results[0]["controller"] if results else None,
            "observation": "RGB 224x224 uint8 + 12 joint positions",
            "action": "12 upstream position-actuator targets, radians",
            "alignment": "Observation and issued target sampled before the same physics step",
            "training_filter": "Use succeeded episodes only; do not train on validation/test seeds",
            "episodes": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
