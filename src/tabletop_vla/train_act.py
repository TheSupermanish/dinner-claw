"""Train the official LeRobot ACT implementation on local MuJoCo demonstrations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, Dataset

IMAGE_KEY = "observation.images.overview"


def read_episodes(directory, expected_split):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["split"] != expected_split:
        raise ValueError(f"Expected {expected_split} dataset, got {manifest['split']}")
    records = []
    seeds = set()
    for episode in manifest["episodes"]:
        if episode["status"] != "succeeded":
            continue
        with np.load(directory / episode["file"], allow_pickle=False) as data:
            records.append({key: data[key] for key in ("rgb", "proprio", "action")})
        seeds.add(episode["seed"])
    if not records:
        raise ValueError("No successful demonstrations")
    return records, seeds


class Episodes(Dataset):
    def __init__(self, episodes, stats, chunk=20):
        self.episodes, self.stats, self.chunk = episodes, stats, chunk
        self.indices = [(e, i) for e, episode in enumerate(episodes)
                        for i in range(len(episode["action"]))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        e, i = self.indices[index]
        episode = self.episodes[e]
        indices = np.arange(i, i + self.chunk)
        pad = indices >= len(episode["action"])
        action = episode["action"][np.minimum(indices, len(episode["action"]) - 1)]
        image = episode["rgb"][i].astype(np.float32).transpose(2, 0, 1) / 255
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return {
            IMAGE_KEY: torch.from_numpy(image),
            "observation.state": torch.from_numpy((episode["proprio"][i] - self.stats["state_mean"]) /
                                                  self.stats["state_std"]),
            "action": torch.from_numpy((action - self.stats["action_mean"]) / self.stats["action_std"]),
            "action_is_pad": torch.from_numpy(pad),
        }


def train(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output directory to preserve existing checkpoints")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    train_episodes, train_seeds = read_episodes(args.train, "train")
    val_episodes, val_seeds = read_episodes(args.validation, "validation")
    if train_seeds & val_seeds:
        raise ValueError("Training and validation seeds overlap")
    states = np.concatenate([e["proprio"] for e in train_episodes])
    actions = np.concatenate([e["action"] for e in train_episodes])
    stats = {"state_mean": states.mean(0), "state_std": states.std(0).clip(0.05),
             "action_mean": actions.mean(0), "action_std": actions.std(0).clip(0.05)}
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = ACTConfig(
        input_features={IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
        device=device, chunk_size=args.chunk_size, n_action_steps=args.action_steps, dim_model=128,
        n_heads=4, dim_feedforward=512, n_encoder_layers=2, n_decoder_layers=1,
        use_vae=False, dropout=0.0,
    )
    policy = ACTPolicy(config).to(device)
    # Freeze pretrained image features for an inexpensive first baseline. This is
    # not a claim that ten small demonstrations suffice for reliable deployment.
    for parameter in policy.model.backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=3e-5)
    loader = DataLoader(Episodes(train_episodes, stats, args.chunk_size), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    validation = DataLoader(Episodes(val_episodes, stats, args.chunk_size), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    args.output.mkdir(parents=True)
    np.savez(args.output / "normalization.npz", **stats)
    iterator = iter(loader)
    history = []
    best = float("inf")
    started = time.perf_counter()
    print(f"Training official LeRobot ACT on {device}: {len(train_episodes)} train / "
          f"{len(val_episodes)} validation episodes", flush=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        policy.train()
        policy.model.backbone.eval()
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 25 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps}: training normalized L1 {loss.item():.5f}", flush=True)
        if step % 100 == 0 or step == args.steps:
            policy.eval()
            total, count = 0.0, 0
            with torch.inference_mode():
                for sample in validation:
                    sample = {key: value.to(device) for key, value in sample.items()}
                    val_loss, _ = policy(sample)
                    size = sample["action"].shape[0]
                    total += val_loss.item() * size
                    count += size
            value = total / count
            history.append({"step": step, "train_l1": loss.item(), "validation_l1": value})
            print(f"Validation normalized L1: {value:.5f}", flush=True)
            if value < best:
                best = value
                policy.save_pretrained(args.output / "best")
            (args.output / "training-report.json").write_text(json.dumps({
                "policy": "LeRobot 0.6.0 ACT", "trained_steps": step,
                "device": device, "torch": torch.__version__, "seed": 42,
                "chunk_size": args.chunk_size, "action_steps": args.action_steps,
                "train_seeds": sorted(train_seeds), "validation_seeds": sorted(val_seeds),
                "elapsed_s": time.perf_counter() - started,
                "backbone": "frozen ImageNet ResNet18", "history": history,
                "deployment_status": "Not approved: requires independent physical rollout evaluation",
            }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=25)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
