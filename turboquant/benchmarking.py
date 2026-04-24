"""Benchmark helpers for FastVQ/TurboQuant."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .turboquant import TurboQuant, TurboQuantConfig
from .utils import compute_distortion

Shape = Tuple[int, ...]


def parse_shape(spec: str) -> Shape:
    """Parse a shape string like ``1024x128`` or ``1,8,4096,128``."""
    separators = "x" if "x" in spec.lower() else ","
    parts = [part.strip() for part in spec.lower().split(separators) if part.strip()]
    if not parts:
        raise ValueError("shape must contain at least one dimension")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive: {spec}")
    return shape


def parse_shapes(spec: str) -> List[Shape]:
    """Parse comma-separated shapes, preserving comma-style dimensions when needed."""
    if ";" in spec:
        return [parse_shape(part) for part in spec.split(";") if part.strip()]
    return [parse_shape(part) for part in spec.split(",") if part.strip()]


def make_dataset(shape: Shape, distribution: str = "normal", seed: int = 42) -> np.ndarray:
    """Create deterministic synthetic benchmark data."""
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        data = rng.standard_normal(shape)
    elif distribution == "uniform":
        data = rng.uniform(-1.0, 1.0, size=shape)
    elif distribution == "student":
        data = rng.standard_t(df=4, size=shape)
    else:
        raise ValueError("distribution must be 'normal', 'uniform', or 'student'")
    return data.astype(np.float32)


def _mean_trial_time(func: Any, trials: int) -> float:
    start = time.perf_counter()
    for _ in range(trials):
        func()
    return (time.perf_counter() - start) / trials


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def benchmark_array(
    data: np.ndarray,
    bit_width: int = 3,
    block_size: int = 128,
    trials: int = 5,
    rotation: str = "hadamard",
    use_qjl: bool = True,
) -> Dict[str, Any]:
    """Benchmark one array/configuration pair."""
    if trials <= 0:
        raise ValueError("trials must be positive")

    config = TurboQuantConfig(
        bit_width=bit_width,
        block_size=block_size,
        use_qjl=use_qjl,
        rotation=rotation,
    )
    quantizer = TurboQuant(config)

    quantized = quantizer.quantize(data)
    reconstructed = quantizer.dequantize(quantized)

    quantize_time = _mean_trial_time(lambda: quantizer.quantize(data), trials)
    dequantize_time = _mean_trial_time(lambda: quantizer.dequantize(quantized), trials)
    metrics = compute_distortion(data, reconstructed)
    stats = quantizer.compression_stats(quantized)

    return {
        "shape": tuple(int(dim) for dim in data.shape),
        "bit_width": bit_width,
        "block_size": block_size,
        "rotation": rotation,
        "use_qjl": use_qjl,
        "trials": trials,
        "quantize_time_ms": quantize_time * 1000,
        "dequantize_time_ms": dequantize_time * 1000,
        "throughput_mb_s": (data.nbytes / quantize_time) / (1024 * 1024),
        **stats,
        **metrics,
    }


def run_benchmark_suite(
    shapes: Sequence[Shape],
    bit_widths: Iterable[int] = (2, 3, 4),
    block_sizes: Iterable[int] = (128,),
    trials: int = 5,
    distribution: str = "normal",
    rotation: str = "hadamard",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run a synthetic benchmark grid."""
    results: List[Dict[str, Any]] = []
    for shape_index, shape in enumerate(shapes):
        data = make_dataset(shape, distribution=distribution, seed=seed + shape_index)
        for block_size in block_sizes:
            for bit_width in bit_widths:
                results.append(
                    benchmark_array(
                        data,
                        bit_width=bit_width,
                        block_size=block_size,
                        trials=trials,
                        rotation=rotation,
                    )
                )
    return results


def write_benchmark_results(results: Sequence[Dict[str, Any]], output: str) -> None:
    """Write benchmark results as JSON or CSV based on the output suffix."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_jsonable(result) for result in results]

    if path.suffix.lower() == ".csv":
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    with path.open("w") as handle:
        json.dump(rows, handle, indent=2)
