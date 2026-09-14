from pathlib import Path

import pytest

from tabletop_vla.openvino.benchmark import benchmark, summarize


def test_benchmark_labels_sequential_batch_throughput():
    result = summarize([10.0, 10.0, 10.0], 4)
    assert result["sequential_requests_per_s"] == 100
    assert result["sequential_samples_per_s"] == 400


def test_benchmark_rejects_invalid_counts_before_import():
    with pytest.raises(ValueError, match="positive"):
        benchmark(Path("missing.xml"), "CPU", 0, inputs_path=Path("missing.npz"))
    with pytest.raises(ValueError):
        summarize([], 1)
