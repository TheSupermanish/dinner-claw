"""Reproducible per-seed rollouts, labelled by the actual controller used."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import mujoco

from tabletop_vla.sim.build_scene import OUTPUT, PROJECT
from tabletop_vla.sim.runtime import Simulation


def run_episode(sim, seed, close_gripper=True, task="cup-place", blind=False):
    if blind and not task.startswith("learned-"):
        raise ValueError("Blind ablation is available only for learned policies")
    sim.start_task(seed, task, close_gripper=close_gripper)
    if blind:
        sim.task.blind = True
    for _ in range(40):
        sim.step(500)
        if sim.task.status != "running":
            break
    if sim.task.status == "running":
        sim.task.fail("Episode timeout")
    return {"seed": seed, "variation": sim.variation, **sim.task.state(),
            "physics_warnings": sim.data.warning.number.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:10", help="Inclusive start:exclusive end")
    parser.add_argument("--output", type=Path, default=Path("outputs/cup-evaluation.json"))
    parser.add_argument("--open-gripper", action="store_true", help="Negative-control ablation")
    parser.add_argument("--blind", action="store_true", help="Replace learned-policy RGB with zeros")
    parser.add_argument("--task", default="cup-place", choices=(
        "cup-place", "camera-cup-place", "learned-cup-place", "learned-cup-safe-place"))
    args = parser.parse_args()
    if args.open_gripper and args.task.startswith("learned-"):
        parser.error("Open-gripper negative control is supported only for scripted tasks")
    if args.blind and not args.task.startswith("learned-"):
        parser.error("Blind ablation is available only for learned policies")
    start, stop = map(int, args.seeds.split(":"))
    if not 0 <= start < stop <= 1000001 or stop - start > 1000:
        parser.error("Use 1–1000 seeds between 0 and 1000000")
    provenance = None
    if args.task.startswith("learned-"):
        export_dir = PROJECT / "outputs/act-openvino"
        exported = json.loads((export_dir / "export-report.json").read_text())
        checkpoint = Path(exported["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT / checkpoint
        training = json.loads((checkpoint / "training-report.json").read_text())
        if set(range(start, stop)) & set(training["train_seeds"] + training["validation_seeds"]):
            parser.error("Evaluation seeds overlap training/validation")
        provenance = {"checkpoint_sha256": exported["checkpoint_sha256"],
                      "model_xml_sha256": hashlib.sha256((export_dir / "act-fp32.xml").read_bytes()).hexdigest(),
                      "model_weights_sha256": hashlib.sha256((export_dir / "act-fp32.bin").read_bytes()).hexdigest(),
                      "train_seeds": training["train_seeds"],
                      "validation_seeds": training["validation_seeds"],
                      "hardware_qualification": False}
    sim = Simulation()
    scene_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    results = []
    try:
        for seed in range(start, stop):
            result = run_episode(sim, seed, not args.open_gripper, args.task, args.blind)
            results.append(result)
            print(f"Seed {seed}: {result['status']} | lift {result['peak_lift_m']:.3f} m | "
                  f"placement error {result['placement_error_m']:.4f} m", flush=True)
    finally:
        sim.close()
    success = sum(r["status"] == "succeeded" for r in results)
    report = {
        "task": args.task, "controller": results[0]["controller"],
        "scope": "Single right-arm miniature square cup. Not full challenge evidence.",
        "learned_policy": args.task.startswith("learned-"),
        "scripted_finish": args.task == "learned-cup-safe-place",
        "blind": args.blind, "model_provenance": provenance,
        "randomization": ["xy placement", "mass", "friction", "lighting",
                          "rectangular cup aspect ratio", "table/floor colors"],
        "uncovered_variation": ["round cup or arbitrary topology", "cluttered backgrounds"],
        "negative_control": args.open_gripper,
        "successes": success, "episodes": len(results), "success_rate": success / len(results),
        "mujoco_version": mujoco.__version__, "platform": platform.platform(),
        "scene_sha256": scene_hash,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{success}/{len(results)} succeeded. Report: {args.output}")


if __name__ == "__main__":
    main()
