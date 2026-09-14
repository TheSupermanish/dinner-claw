"""Local image+language reasoning probe. Advisory output cannot command motors."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from tabletop_vla.sim.runtime import Simulation

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--caption-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(4)
    sim = Simulation()
    try:
        sim.reset(args.seed)
        sim.control("right_shoulder_pan", 0.6)
        sim.step(400)
        frame = Image.fromarray(sim.rgb("overview"))
    finally:
        sim.close()
    args.output.mkdir(parents=True)
    frame.save(args.output / "camera.png")
    print("Loading pinned SmolVLM weights for local inference; images are not uploaded", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, token=False,
                                               trust_remote_code=False)
    dtype = torch.float16 if args.device == "mps" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, token=False, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    prompt = (
        "Describe the objects visible in this robot workspace briefly. "
        f"The requested task is: {args.instruction}\n"
        "The only implemented action is right-arm cup pick-and-place. "
        "Say which requested parts are unsupported. Do not claim any action has executed."
    )
    if args.caption_only:
        prompt = "Describe the visible objects in this image."
    messages = [{"role": "user", "content": [{"type": "image", "image": frame},
                                                {"type": "text", "text": prompt}]}]
    rendered_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=rendered_prompt, images=[frame], return_tensors="pt")
    inputs = {key: value.to(device=args.device, dtype=dtype) if value.is_floating_point()
              else value.to(args.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    answer = processor.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    result = {"model": MODEL, "revision": REVISION, "instruction": args.instruction,
              "seed": args.seed, "device": args.device, "answer": answer,
              "inference_s": time.perf_counter() - started,
              "executed": False, "role": "Untrusted advisory scene commentary; not an approved task planner",
              "prompt": prompt, "input_tokens": inputs["input_ids"].shape[-1],
              "output_tokens": generated.shape[-1] - inputs["input_ids"].shape[-1],
              "image": "camera.png"}
    (args.output / "reasoning.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
