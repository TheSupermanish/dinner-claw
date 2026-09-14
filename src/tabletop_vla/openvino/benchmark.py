from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def summarize(samples, batch_size):
    if not samples or batch_size < 1 or not np.isfinite(samples).all() or min(samples) <= 0:
        raise ValueError("Positive finite latency samples and batch size are required")
    return {
        "latency_ms_p50": float(np.percentile(samples, 50)),
        "latency_ms_p95": float(np.percentile(samples, 95)),
        "latency_ms_mean": float(np.mean(samples)),
        "sequential_requests_per_s": 1000.0 / float(np.mean(samples)),
        "sequential_samples_per_s": batch_size * 1000.0 / float(np.mean(samples)),
    }


def benchmark(model_path: Path, device: str, iterations: int, *,
              inputs_path: Path, warmup=10, batch_size=1, precision="f32") -> dict[str, object]:
    if iterations < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("Iterations/batch must be positive and warmup nonnegative")
    if not model_path.is_file() or not inputs_path.is_file():
        raise ValueError("Provide an actual exported model and recorded input NPZ fixture")
    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("Install the Intel extras first: uv sync --extra intel") from exc
    core = ov.Core()
    started = time.perf_counter()
    compiled = core.compile_model(str(model_path), device, {
        "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision,
    })
    compile_ms = (time.perf_counter() - started) * 1000
    request = compiled.create_infer_request()
    inputs = {}
    shapes = {}
    with np.load(inputs_path, allow_pickle=False) as fixture:
        for port in compiled.inputs:
            name = port.get_any_name()
            if name not in fixture:
                raise ValueError(f"Fixture missing model input {name}")
            value = fixture[name]
            if port.partial_shape.is_dynamic:
                raise ValueError("Export fixed shapes for this benchmark; dynamic inputs are unsupported")
            if tuple(value.shape) != tuple(port.shape) or value.dtype != port.element_type.to_dtype():
                raise ValueError(f"Fixture shape/dtype mismatch for {name}")
            if value.shape[0] != batch_size or not np.isfinite(value).all():
                raise ValueError(f"Invalid batch size or non-finite fixture for {name}")
            inputs[name] = value
            shapes[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    for _ in range(warmup):
        request.infer(inputs)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        request.infer(inputs)
        samples.append((time.perf_counter() - start) * 1000)
    execution_devices = list(compiled.get_property("EXECUTION_DEVICES"))
    actual_precision = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    checksums = {model_path.name: hashlib.sha256(model_path.read_bytes()).hexdigest()}
    weights = model_path.with_suffix(".bin")
    if model_path.suffix == ".xml" and weights.is_file():
        checksums[weights.name] = hashlib.sha256(weights.read_bytes()).hexdigest()
    return {
        "model": str(model_path),
        "device": device,
        "available_devices": core.available_devices,
        "iterations": iterations,
        "warmup_iterations": warmup, "batch_size": batch_size,
        "compile_time_ms": compile_ms, "openvino_version": ov.__version__,
        "execution_devices": execution_devices,
        "device_names": {item: core.get_property(item, "FULL_DEVICE_NAME") for item in execution_devices},
        "requested_precision": precision, "inference_precision_hint": actual_precision,
        "precision_note": "Runtime hint, not proof every layer uses this precision",
        "model_sha256": checksums,
        "fixture_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "inputs": shapes,
        **summarize(samples, batch_size),
        "timing_scope": "Synchronous inference only; excludes camera, preprocessing, simulation and model compile",
        "behavior_parity_verified": False,
        "core_ultra_series_2_3_demonstration_verified": False,
        "host": platform.processor() or platform.machine(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--output", type=Path, default=Path("outputs/openvino_benchmark.json"))
    args = parser.parse_args()
    result = benchmark(args.model, args.device, args.iterations, inputs_path=args.inputs,
                       warmup=args.warmup, batch_size=args.batch_size, precision=args.precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
