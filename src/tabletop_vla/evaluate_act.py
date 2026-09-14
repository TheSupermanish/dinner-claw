"""Closed-loop ACT evaluation. Simulator object state is scorer-only, never policy input."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from PIL import Image

from tabletop_vla.sim.runtime import Simulation
from tabletop_vla.sim.scoring import PhysicalScore
from tabletop_vla.train_act import IMAGE_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Training output directory containing best/")
    parser.add_argument("--seeds", default="2000:2010")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--action-steps", type=int, help="Override executed chunk length for diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    if not 0 <= start < stop <= 1000001 or stop - start > 1000:
        parser.error("Invalid seed range")
    training = json.loads((args.checkpoint / "training-report.json").read_text())
    if set(range(start, stop)) & set(training["train_seeds"] + training["validation_seeds"]):
        parser.error("Evaluation seeds overlap training/validation")
    torch.set_num_threads(4)
    policy = ACTPolicy.from_pretrained(args.checkpoint / "best", local_files_only=True)
    policy.to(args.device).eval()
    if args.action_steps is not None:
        if not 1 <= args.action_steps <= policy.config.chunk_size:
            parser.error("Action steps must fit within trained chunk size")
        policy.config.n_action_steps = args.action_steps
        policy.reset()
    with np.load(args.checkpoint / "normalization.npz") as stored:
        stats = {key: stored[key] for key in stored.files}
    sim = Simulation()
    renderer = mujoco.Renderer(sim.model, height=480, width=720)
    qids = sim.model.jnt_qposadr[sim.model.actuator_trnid[:, 0]]
    results = []
    try:
        for seed in range(start, stop):
            sim.reset(seed)
            policy.reset()
            score = PhysicalScore(sim)
            clipped = 0
            latencies = []
            trace = []
            for frame_index in range(360):
                renderer.update_scene(sim.data, camera="overview")
                image = np.array(Image.fromarray(renderer.render()).resize((224, 224)))
                if args.blind:
                    image[:] = 0
                rgb = image.astype(np.float32).transpose(2, 0, 1) / 255
                rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array(
                    [0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
                proprio = (sim.data.qpos[qids].astype(np.float32) - stats["state_mean"]) / stats["state_std"]
                observation = {IMAGE_KEY: torch.from_numpy(rgb)[None].to(args.device),
                               "observation.state": torch.from_numpy(proprio)[None].to(args.device)}
                started = time.perf_counter()
                with torch.inference_mode():
                    action = policy.select_action(observation).cpu().numpy()[0]
                latencies.append((time.perf_counter() - started) * 1000)
                action = action * stats["action_std"] + stats["action_mean"]
                if not np.isfinite(action).all():
                    raise RuntimeError("Non-finite policy action")
                bounded = np.clip(action, sim.model.actuator_ctrlrange[:, 0],
                                  sim.model.actuator_ctrlrange[:, 1])
                clipped += int(np.any(action != bounded))
                sim.data.ctrl[:] = bounded
                if frame_index % 20 == 0:
                    trace.append({"time": float(sim.data.time),
                                  "actual": sim.data.qpos[qids].tolist(),
                                  "target": bounded.tolist()})
                for _ in range(25):
                    mujoco.mj_step(sim.model, sim.data)
                    score.update()
            result = {"seed": seed, **score.result(), "clipped_action_steps": clipped,
                      "select_action_ms_mean_including_queue": float(np.mean(latencies)),
                      "physics_warnings": sim.data.warning.number.tolist()}
            result["trace"] = trace
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "trace"}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "controller": "LeRobot ACT, RGB + 12 joint positions only",
                "blind": args.blind, "checkpoint": str(args.checkpoint),
                "successes": sum(r["success"] for r in results), "episodes": len(results),
                "results": results,
            }, indent=2) + "\n")
    finally:
        renderer.close()
        sim.close()


if __name__ == "__main__":
    main()
