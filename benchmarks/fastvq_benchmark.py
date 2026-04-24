#!/usr/bin/env python3
"""Run FastVQ synthetic benchmarks from source checkout."""

import argparse

from turboquant.benchmarking import (
    parse_shapes,
    run_benchmark_suite,
    write_benchmark_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastVQ synthetic benchmark runner")
    parser.add_argument(
        "--shapes",
        default="1024x128,4096x128,1024x192",
        help="Comma-separated shapes; use semicolons for comma-style shapes",
    )
    parser.add_argument("--bits", default="2,3,4", help="Comma-separated bit-widths")
    parser.add_argument("--block-sizes", default="128", help="Comma-separated block sizes")
    parser.add_argument("--trials", type=int, default=5, help="Timing trials per case")
    parser.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "uniform", "student"],
        help="Synthetic data distribution",
    )
    parser.add_argument(
        "--rotation",
        default="hadamard",
        choices=["hadamard", "random"],
        help="Rotation backend",
    )
    parser.add_argument(
        "--output", default="benchmarks/results.json", help="JSON or CSV output path"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shapes = parse_shapes(args.shapes)
    bit_widths = [int(value.strip()) for value in args.bits.split(",") if value.strip()]
    block_sizes = [int(value.strip()) for value in args.block_sizes.split(",") if value.strip()]

    results = run_benchmark_suite(
        shapes=shapes,
        bit_widths=bit_widths,
        block_sizes=block_sizes,
        trials=args.trials,
        distribution=args.distribution,
        rotation=args.rotation,
    )
    write_benchmark_results(results, args.output)

    for result in results:
        shape = "x".join(str(dim) for dim in result["shape"])
        print(
            f"{shape} bits={result['bit_width']} block={result['block_size']} "
            f"q={result['quantize_time_ms']:.2f}ms "
            f"dq={result['dequantize_time_ms']:.2f}ms "
            f"ratio={result['compression_ratio']:.2f}x cosine={result['cosine_similarity']:.4f}"
        )

    print(f"Saved {len(results)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
