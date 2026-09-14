"""Export the actual trained ACT, including normalization, with numerical parity checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from lerobot.policies.act.modeling_act import ACTPolicy


class ExportedACT(torch.nn.Module):
    def __init__(self, policy, stats):
        super().__init__()
        self.policy = policy.model
        for name, value in stats.items():
            self.register_buffer(name, torch.tensor(value))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, rgb, proprio):
        batch = {"observation.images": [(rgb - self.image_mean) / self.image_std],
                 "observation.state": (proprio - self.state_mean) / self.state_std}
        return self.policy(batch)[0] * self.action_std + self.action_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episode", type=Path, required=True, help="Recorded RGB/proprio NPZ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new export directory; existing models are preserved")
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    policy = ACTPolicy.from_pretrained(args.checkpoint / "best", local_files_only=True).cpu().eval()
    with np.load(args.checkpoint / "normalization.npz") as stored:
        stats = {key: stored[key] for key in stored.files}
    wrapper = ExportedACT(policy, stats).eval()
    with np.load(args.episode, allow_pickle=False) as episode:
        frames = episode["rgb"]
        states = episode["proprio"]
    fixture = {"rgb": frames[:1].astype(np.float32).transpose(0, 3, 1, 2) / 255,
               "proprio": states[:1].astype(np.float32)}
    example = tuple(torch.from_numpy(fixture[key]) for key in ("rgb", "proprio"))
    print("Converting trained ACT with raw RGB/proprio inputs and radian action outputs", flush=True)
    converted = ov.convert_model(wrapper, example_input=example)
    for port, name in zip(converted.inputs, ("rgb", "proprio"), strict=True):
        port.get_tensor().set_names({name})
    converted.output(0).get_tensor().set_names({"action_chunk"})
    converted.reshape({"rgb": [1, 3, 224, 224], "proprio": [1, 12]})
    args.output.mkdir(parents=True)
    np.savez(args.output / "example-inputs.npz", **fixture)
    ov.save_model(converted, args.output / "act-fp32.xml", compress_to_fp16=False)
    ov.save_model(converted, args.output / "act-fp16-weights.xml", compress_to_fp16=True)
    core = ov.Core()
    results = {}
    for filename in ("act-fp32.xml", "act-fp16-weights.xml"):
        compiled = core.compile_model(str(args.output / filename), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        errors = []
        for index in np.linspace(0, len(frames) - 1, 12, dtype=int):
            rgb = frames[index:index + 1].astype(np.float32).transpose(0, 3, 1, 2) / 255
            proprio = states[index:index + 1].astype(np.float32)
            with torch.inference_mode():
                expected = wrapper(torch.from_numpy(rgb), torch.from_numpy(proprio)).numpy()
            actual = compiled({"rgb": rgb, "proprio": proprio})[compiled.output(0)]
            errors.append(float(np.max(np.abs(actual - expected))))
        results[filename] = {"max_action_error_rad": max(errors), "per_frame_max_error_rad": errors,
                             "fixture_output_parity_pass": max(errors) < 0.001}
        print(filename, results[filename]["max_action_error_rad"], flush=True)
    (args.output / "export-report.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "openvino_version": ov.__version__,
        "checkpoint_sha256": hashlib.sha256((args.checkpoint / "best/model.safetensors").read_bytes()).hexdigest(),
        "cpu_device": core.get_property("CPU", "FULL_DEVICE_NAME"),
        "parity": results, "fixture": str(args.episode),
        "behavior_rollout_parity_verified": False,
        "intel_core_ultra_demonstration_verified": False,
        "precision_note": "FP16 file compresses weights; runtime tested with FP32 compute hint",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
