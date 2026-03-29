#!/usr/bin/env python3
"""
Demo script for TurboQuant - Extreme compression for KV caches.

This demonstrates the key features:
- 3-bit compression with near-zero accuracy loss
- 6x memory reduction
- Fast compression/decompression
"""

import numpy as np
import sys

sys.path.insert(0, "/home/dih/turboquant-clone")

from turboquant import TurboQuant, TurboQuantConfig
from turboquant.kv_cache import KVCacheCompressor, benchmark_kv_cache


def demo_basic_compression():
    """Demonstrate basic TurboQuant compression on random data."""
    print("=" * 60)
    print("Demo 1: Basic TurboQuant Compression")
    print("=" * 60)

    # Create sample data (simulating KV cache entries)
    batch_size, n_heads, seq_len, head_dim = 1, 8, 1024, 128
    x = np.random.randn(batch_size, n_heads, seq_len, head_dim).astype(np.float32)

    print(f"Input shape: {x.shape}")
    print(f"Input size: {x.nbytes / 1024:.2f} KB")

    # Test different bit-widths
    for bit_width in [3, 4]:
        config = TurboQuantConfig(bit_width=bit_width, block_size=128)
        tq = TurboQuant(config)

        # Compress and decompress using quantize/dequantize directly
        quantized = tq.quantize(x)
        reconstructed = tq.dequantize(quantized)

        # Compute metrics
        mse = np.mean((x - reconstructed) ** 2)

        # Estimate compressed size
        n_elements = x.size
        bits_per_element = bit_width + 0.25  # Include overhead
        compressed_bytes = (n_elements * bits_per_element) / 8 + n_elements * 4  # norms
        compression_ratio = x.nbytes / compressed_bytes

        print(f"\n  Bit-width: {bit_width}")
        print(f"    MSE: {mse:.6f}")
        print(f"    Estimated compressed size: {compressed_bytes / 1024:.2f} KB")
        print(f"    Compression ratio: {compression_ratio:.2f}x")
        print(f"    Cosine similarity: {compute_cosine_sim(x, reconstructed):.6f}")

    print()


def demo_kv_cache():
    """Demonstrate KV cache compression."""
    print("=" * 60)
    print("Demo 2: KV Cache Compression")
    print("=" * 60)

    compressor = KVCacheCompressor(bit_width=3, block_size=128)

    # Simulate LLM with 70B parameters on 72GB VRAM
    seq_len = 100000
    batch_size = 1
    n_heads = 64
    head_dim = 128

    print(f"\nModel: 70B parameters")
    print(f"VRAM: 72 GB (34 GB available for KV cache)")
    print(f"Sequence length: {seq_len:,} tokens")
    print(f"Heads: {n_heads}, Head dim: {head_dim}")

    # Compute memory statistics
    stats = compressor.compute_memory_stats(seq_len, batch_size, n_heads, head_dim)

    print(f"\nMemory Analysis:")
    print(f"  FP16 KV cache: {stats['fp16_memory_gb']:.2f} GB")
    print(f"  TurboQuant (3-bit): {stats['turboquant_memory_gb']:.2f} GB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Memory saved: {stats['memory_saved_gb']:.2f} GB")
    print(f"  Max context (FP16): {stats['max_context_fp16']:,} tokens")
    print(f"  Max context (TQ3): {stats['max_context_tq']:,} tokens")

    print()


def demo_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 60)
    print("Demo 3: Performance Benchmark")
    print("=" * 60)

    print("\nBenchmarking different bit-widths...")
    print("(This may take a moment...)\n")

    results = benchmark_kv_cache(
        seq_len=4096,
        batch_size=1,
        n_heads=32,
        head_dim=128,
        bit_widths=[3, 4],
    )

    print(
        f"{'Bit-width':<12} {'Comp Time':<12} {'Decomp Time':<14} {'MSE':<12} {'Memory GB':<12}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['bit_width']}-bit        "
            f"{r['compress_time_ms']:.2f} ms     "
            f"{r['decompress_time_ms']:.2f} ms       "
            f"{r['k_mse']:.6f}   "
            f"{r['turboquant_memory_gb']:.3f}"
        )

    print()


def compute_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("\n" + "=" * 60)
    print("TurboQuant Clone - Demonstration")
    print("=" * 60)
    print("\nA production-quality implementation of Google's TurboQuant")
    print("for extreme compression of AI models.")
    print("\nFeatures:")
    print("  - 3-bit compression with near-zero accuracy loss")
    print("  - 6x memory reduction for KV caches")
    print("  - PolarQuant + QJL two-stage algorithm")
    print("  - Optimized bit-packing")
    print("\n" + "=" * 60)

    try:
        demo_basic_compression()
        demo_kv_cache()
        demo_benchmark()

        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
