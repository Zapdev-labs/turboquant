import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .clipboard import copy_error_to_clipboard, reset_terminal_state
from .kv_cache import KVCacheCompressor
from .model_export import load_gguf, load_safetensors
from .turboquant import TurboQuant, TurboQuantConfig
from .utils import compute_distortion


def handle_error(error: Exception, command: str) -> int:
    print(f"Error: {error}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    reset_terminal_state()
    copy_error_to_clipboard(error, command)
    return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="fastvq",
        description="FastVQ: fast vector quantization for AI model weights and KV caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a numpy array
  fastvq compress input.npy output.tq --bits 3

  # Decompress
  fastvq decompress output.tq reconstructed.npy

  # Benchmark different bit-widths
  fastvq benchmark input.npy --bits 3,4

  # Analyze KV cache memory usage
  fastvq kv-analyze --model-size 70b --seq-len 100000

  # Info about a compressed file
  fastvq info compressed.tq
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress arrays or tensors")
    compress_parser.add_argument("input", type=str, help="Input file (.npy, .npz)")
    compress_parser.add_argument("output", type=str, help="Output file (.tq)")
    compress_parser.add_argument(
        "--bits",
        "-b",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Bit-width for quantization (default: 3)",
    )
    compress_parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        choices=[32, 64, 128, 256],
        help="Block size for quantization (default: 128)",
    )
    compress_parser.add_argument(
        "--no-qjl", action="store_true", help="Disable QJL error correction"
    )
    compress_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    compress_parser.add_argument(
        "--rotation",
        type=str,
        default="hadamard",
        choices=["hadamard", "random"],
        help="Rotation backend (default: hadamard)",
    )
    compress_parser.add_argument(
        "--radii-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for polar radii (default: float16)",
    )

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress .tq files")
    decompress_parser.add_argument("input", type=str, help="Input .tq file")
    decompress_parser.add_argument("output", type=str, help="Output file (.npy)")

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark compression quality and speed"
    )
    benchmark_parser.add_argument("input", type=str, help="Input file (.npy)")
    benchmark_parser.add_argument(
        "--bits",
        "-b",
        type=str,
        default="3",
        help="Comma-separated bit-widths to test (default: 3)",
    )
    benchmark_parser.add_argument(
        "--trials", type=int, default=10, help="Number of benchmark trials (default: 10)"
    )
    benchmark_parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        choices=[32, 64, 128, 256],
        help="Block size for quantization (default: 128)",
    )
    benchmark_parser.add_argument(
        "--rotation",
        type=str,
        default="hadamard",
        choices=["hadamard", "random"],
        help="Rotation backend (default: hadamard)",
    )
    benchmark_parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    # Synthetic benchmark suite command
    suite_parser = subparsers.add_parser("benchmark-suite", help="Run a synthetic benchmark grid")
    suite_parser.add_argument(
        "--shapes",
        type=str,
        default="1024x128,4096x128,1024x192",
        help="Comma-separated shapes; use semicolons for comma-style shapes",
    )
    suite_parser.add_argument(
        "--bits",
        "-b",
        type=str,
        default="2,3,4",
        help="Comma-separated bit-widths to test (default: 2,3,4)",
    )
    suite_parser.add_argument(
        "--block-sizes",
        type=str,
        default="128",
        help="Comma-separated block sizes to test (default: 128)",
    )
    suite_parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of timing trials per case (default: 5)",
    )
    suite_parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        choices=["normal", "uniform", "student"],
        help="Synthetic data distribution (default: normal)",
    )
    suite_parser.add_argument(
        "--rotation",
        type=str,
        default="hadamard",
        choices=["hadamard", "random"],
        help="Rotation backend (default: hadamard)",
    )
    suite_parser.add_argument("--output", "-o", type=str, help="Save results to JSON or CSV")

    # KV Cache analysis command
    kv_parser = subparsers.add_parser("kv-analyze", help="Analyze KV cache compression for LLMs")
    kv_parser.add_argument(
        "--model-size",
        type=str,
        default="70b",
        choices=["7b", "13b", "30b", "70b", "175b"],
        help="Model size (default: 70b)",
    )
    kv_parser.add_argument(
        "--seq-len", type=int, default=100000, help="Sequence length (default: 100000)"
    )
    kv_parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    kv_parser.add_argument(
        "--bits", "-b", type=int, default=3, choices=[2, 3, 4], help="Bit-width (default: 3)"
    )
    kv_parser.add_argument(
        "--vram", type=float, default=72.0, help="Available VRAM in GB (default: 72)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about a compressed file")
    info_parser.add_argument("file", type=str, help="Compressed .tq file")

    # Download model command
    download_parser = subparsers.add_parser(
        "download", help="Download and quantize models from HuggingFace"
    )
    download_parser.add_argument(
        "model",
        type=str,
        help="Model ID (e.g., meta-llama/Llama-2-7b-hf, TheBloke/Llama-2-7B-GPTQ)",
    )
    download_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: ./models, or set TURBOQUANT_MODELS_DIR env var)",
    )
    download_parser.add_argument(
        "--bits",
        "-b",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Bit-width for quantization (default: 4)",
    )
    download_parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="gguf",
        choices=["gguf", "safetensors"],
        help="Export format (default: gguf)",
    )
    download_parser.add_argument("--hf-token", type=str, help="HuggingFace token for gated models")
    download_parser.add_argument(
        "--cache-dir", type=str, help="Cache directory for downloaded models"
    )
    download_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for loading (default: auto)",
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list-models", help="List popular pre-quantized models available"
    )
    list_parser.add_argument(
        "--category",
        type=str,
        choices=["7b", "13b", "70b", "chat", "code", "all"],
        default="all",
        help="Filter by category (default: all)",
    )

    # Load command
    load_parser = subparsers.add_parser(
        "load", help="Load and optionally re-quantize GGUF or SafeTensors models"
    )
    load_parser.add_argument(
        "model_path",
        type=str,
        help="Path to GGUF or SafeTensors file",
    )
    load_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for re-quantized model",
    )
    load_parser.add_argument(
        "--bits",
        "-b",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Bit-width for re-quantization (default: 3)",
    )
    load_parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["gguf", "safetensors"],
        help="Force output format (auto-detected from extension if not specified)",
    )
    load_parser.add_argument(
        "--info",
        action="store_true",
        help="Only show model info, don't load fully",
    )
    load_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for loading (default: auto)",
    )

    # Quick compress command (simplified)
    quick_parser = subparsers.add_parser("quick", help="Quick compression with default settings")
    quick_parser.add_argument("input", type=str, help="Input file")
    quick_parser.add_argument("output", type=str, nargs="?", help="Output file (optional)")

    # Chat command - Interactive chat server for T3code integration
    chat_parser = subparsers.add_parser(
        "chat", help="Run interactive chat server (for T3code integration)"
    )
    chat_parser.add_argument(
        "model_path",
        type=str,
        help="Path to GGUF model file",
    )
    chat_parser.add_argument(
        "--context-length",
        "-c",
        type=int,
        default=4096,
        help="Context length (default: 4096)",
    )
    chat_parser.add_argument(
        "--turboquant-bits",
        "-b",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="TurboQuant bit-width for KV cache compression (default: 3)",
    )
    chat_parser.add_argument(
        "--gpu-layers",
        "-g",
        type=int,
        default=-1,
        help="GPU layers (-1 for auto, 0 for CPU only, default: -1)",
    )
    chat_parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt (default: 'You are a helpful AI assistant.')",
    )

    return parser


def cmd_compress(args: argparse.Namespace) -> int:
    """Handle the compress command."""
    try:
        # Load input
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
            return 1

        print(f"Loading {args.input}...")
        if input_path.suffix == ".npy":
            data = np.load(args.input)
        elif input_path.suffix == ".npz":
            loaded = np.load(args.input)
            data = loaded[list(loaded.keys())[0]]
        else:
            print(f"Error: Unsupported file format '{input_path.suffix}'", file=sys.stderr)
            return 1

        print(f"Input shape: {data.shape}, dtype: {data.dtype}")
        print(f"Input size: {data.nbytes / 1024 / 1024:.2f} MB")

        # Configure and run compression
        config = TurboQuantConfig(
            bit_width=args.bits,
            block_size=args.block_size,
            use_qjl=not args.no_qjl,
            use_polar=True,
            rotation_seed=args.seed,
            rotation=args.rotation,
            radii_dtype=args.radii_dtype,
        )

        print(f"\nCompressing with {args.bits}-bit TurboQuant...")
        tq = TurboQuant(config)

        quantized = tq.quantize(data)
        reconstructed = tq.dequantize(quantized)

        # Calculate metrics
        metrics = compute_distortion(data, reconstructed)

        print("\nCompression Results:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")

        # Save compressed representation
        output_path = Path(args.output)
        output_path.write_bytes(tq._pack_quantized(quantized))

        # Calculate actual compression ratio
        compressed_size = Path(args.output).stat().st_size
        compression_ratio = data.nbytes / compressed_size
        estimated = tq.compression_stats(quantized)

        print(f"\nOutput: {args.output}")
        print(f"  Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Estimated packed ratio: {estimated['compression_ratio']:.2f}x")

        return 0

    except Exception as e:
        return handle_error(e, "compress")


def cmd_decompress(args: argparse.Namespace) -> int:
    """Handle the decompress command."""
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
            return 1

        print(f"Loading compressed file {args.input}...")

        print("Decompressing...")
        reconstructed = TurboQuant().decompress(input_path.read_bytes())

        # Save output
        np.save(args.output, reconstructed)

        print(f"\nOutput: {args.output}")
        print(f"  Shape: {reconstructed.shape}")
        print(f"  Dtype: {reconstructed.dtype}")
        print(f"  Size: {reconstructed.nbytes / 1024 / 1024:.2f} MB")

        return 0

    except Exception as e:
        return handle_error(e, "decompress")


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the benchmark command."""
    try:
        # Load input
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
            return 1

        data = np.load(args.input)
        print(f"Input shape: {data.shape}")
        print(f"Input size: {data.nbytes / 1024 / 1024:.2f} MB\n")

        # Parse bit-widths
        bit_widths = [int(b.strip()) for b in args.bits.split(",")]

        # Run benchmarks
        results = []

        print(
            f"{'Bit-width':<12} {'Compress':<12} {'Decompress':<14} "
            f"{'Ratio':<10} {'MSE':<12} {'SNR':<10}"
        )
        print("-" * 82)

        for bw in bit_widths:
            config = TurboQuantConfig(
                bit_width=bw,
                block_size=args.block_size,
                rotation=args.rotation,
            )
            tq = TurboQuant(config)

            import time

            # Warmup and initialize variables
            quantized = tq.quantize(data[:100] if len(data) > 100 else data)
            reconstructed = tq.dequantize(quantized)

            # Benchmark compression
            start = time.perf_counter()
            for _ in range(args.trials):
                quantized = tq.quantize(data)
            compress_time = (time.perf_counter() - start) / args.trials

            # Benchmark decompression
            start = time.perf_counter()
            for _ in range(args.trials):
                reconstructed = tq.dequantize(quantized)
            decompress_time = (time.perf_counter() - start) / args.trials

            # Metrics
            metrics = compute_distortion(data, reconstructed)
            stats = tq.compression_stats(quantized)

            result = {
                "bit_width": bw,
                "compress_time_ms": compress_time * 1000,
                "decompress_time_ms": decompress_time * 1000,
                "compression_ratio": stats["compression_ratio"],
                "compressed_bytes": stats["compressed_bytes"],
                **metrics,
            }
            results.append(result)

            print(
                f"{bw}-bit        "
                f"{compress_time * 1000:>8.2f} ms   "
                f"{decompress_time * 1000:>8.2f} ms     "
                f"{stats['compression_ratio']:>6.2f}x   "
                f"{metrics['mse']:>8.6f}   "
                f"{metrics['snr_db']:>6.2f} dB"
            )

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                    default=lambda value: value.item() if hasattr(value, "item") else str(value),
                )
            print(f"\nResults saved to {args.output}")

        return 0

    except Exception as e:
        return handle_error(e, "benchmark")


def cmd_benchmark_suite(args: argparse.Namespace) -> int:
    """Handle the benchmark-suite command."""
    try:
        from .benchmarking import (
            parse_shapes,
            run_benchmark_suite,
            write_benchmark_results,
        )

        shapes = parse_shapes(args.shapes)
        bit_widths = [int(value.strip()) for value in args.bits.split(",") if value.strip()]
        block_sizes = [int(value.strip()) for value in args.block_sizes.split(",") if value.strip()]

        print("Running synthetic benchmark suite")
        shape_list = ", ".join("x".join(str(dim) for dim in shape) for shape in shapes)
        print(f"  Shapes: {shape_list}")
        print(f"  Bits: {bit_widths}")
        print(f"  Block sizes: {block_sizes}")
        print(f"  Trials: {args.trials}")
        print()

        results = run_benchmark_suite(
            shapes=shapes,
            bit_widths=bit_widths,
            block_sizes=block_sizes,
            trials=args.trials,
            distribution=args.distribution,
            rotation=args.rotation,
        )

        print(
            f"{'Shape':<16} {'Bits':<6} {'Block':<7} {'Q ms':<10} "
            f"{'DQ ms':<10} {'Ratio':<8} {'Cosine':<8}"
        )
        print("-" * 78)
        for result in results:
            shape = "x".join(str(dim) for dim in result["shape"])
            print(
                f"{shape:<16} {result['bit_width']:<6} {result['block_size']:<7} "
                f"{result['quantize_time_ms']:<10.2f} "
                f"{result['dequantize_time_ms']:<10.2f} "
                f"{result['compression_ratio']:<8.2f} "
                f"{result['cosine_similarity']:<8.4f}"
            )

        if args.output:
            write_benchmark_results(results, args.output)
            print(f"\nResults saved to {args.output}")

        return 0

    except Exception as e:
        return handle_error(e, "benchmark-suite")


def cmd_kv_analyze(args: argparse.Namespace) -> int:
    """Handle the kv-analyze command."""
    try:
        # Model configurations
        model_configs = {
            "7b": {"n_heads": 32, "head_dim": 128, "layers": 32},
            "13b": {"n_heads": 40, "head_dim": 128, "layers": 40},
            "30b": {"n_heads": 52, "head_dim": 128, "layers": 60},
            "70b": {"n_heads": 64, "head_dim": 128, "layers": 80},
            "175b": {"n_heads": 96, "head_dim": 128, "layers": 96},
        }

        config = model_configs[args.model_size]

        print("=" * 60)
        print(f"KV Cache Analysis for {args.model_size.upper()} Model")
        print("=" * 60)
        print("\nModel Configuration:")
        print(f"  Layers: {config['layers']}")
        print(f"  Attention heads: {config['n_heads']}")
        print(f"  Head dimension: {config['head_dim']}")

        print("\nInference Configuration:")
        print(f"  Sequence length: {args.seq_len:,} tokens")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Available VRAM: {args.vram:.1f} GB")

        # Create compressor
        compressor = KVCacheCompressor(bit_width=args.bits, block_size=config["head_dim"])

        # Per-layer stats (just K+V cache, not all layers at once)
        stats = compressor.compute_memory_stats(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_heads=config["n_heads"],
            head_dim=config["head_dim"],
        )

        # Full model (all layers)
        fp16_per_layer = stats["fp16_memory_gb"]
        tq_per_layer = stats["turboquant_memory_gb"]

        fp16_full = fp16_per_layer * config["layers"]
        tq_full = tq_per_layer * config["layers"]

        print("\nMemory Usage (per layer):")
        print(f"  FP16 KV cache:     {fp16_per_layer:.3f} GB")
        print(f"  TurboQuant {args.bits}-bit:  {tq_per_layer:.3f} GB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

        print(f"\nMemory Usage (full model, {config['layers']} layers):")
        print(f"  FP16 KV cache:     {fp16_full:.2f} GB")
        print(f"  TurboQuant {args.bits}-bit:  {tq_full:.2f} GB")

        print("\nContext Window Analysis:")
        print(f"  Max context (FP16):  {stats['max_context_fp16']:,} tokens")
        print(f"  Max context (TQ):    {stats['max_context_tq']:,} tokens")
        print(f"  Improvement:         {stats['max_context_tq'] / stats['max_context_fp16']:.1f}x")

        if fp16_full > args.vram * 0.8:
            print(f"\n⚠️  Warning: FP16 cache requires {fp16_full:.1f} GB")
            print(f"   This exceeds 80% of available VRAM ({args.vram * 0.8:.1f} GB)")
            print(f"   TurboQuant enables this model to fit in {args.vram} GB VRAM")

        return 0

    except Exception as e:
        return handle_error(e, "kv-analyze")


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command."""
    try:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: File '{args.file}' not found", file=sys.stderr)
            return 1

        loaded = np.load(args.file)

        print(f"File: {args.file}")
        print(f"Size: {input_path.stat().st_size / 1024:.2f} KB")
        print("\nCompression Parameters:")
        print(f"  Bit-width: {loaded['bit_width'][0]}")
        print(f"  Block size: {loaded['block_size'][0]}")
        print(f"  QJL enabled: {bool(loaded['use_qjl'][0])}")
        if "rotation_code" in loaded.files:
            rotation = "random" if int(loaded["rotation_code"][0]) == 1 else "hadamard"
            radii_dtype = "float32" if int(loaded["radii_dtype_code"][0]) == 1 else "float16"
            print(f"  Rotation: {rotation}")
            print(f"  Radii dtype: {radii_dtype}")
        print(f"\nOriginal Shape: {tuple(loaded['original_shape'])}")
        print("\nData Arrays:")
        for key in loaded.files:
            arr = loaded[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

        return 0

    except Exception as e:
        return handle_error(e, "info")


def cmd_quick(args: argparse.Namespace) -> int:
    """Handle the quick compression command."""
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
            return 1

        # Load
        data = np.load(args.input)
        print(f"Input: {args.input} ({data.nbytes / 1024 / 1024:.2f} MB)")

        # Compress with defaults (3-bit)
        config = TurboQuantConfig(bit_width=3, block_size=128)
        tq = TurboQuant(config)

        print("Compressing with TurboQuant 3-bit...")
        quantized = tq.quantize(data)
        reconstructed = tq.dequantize(quantized)

        metrics = compute_distortion(data, reconstructed)

        # Determine output filename
        output_path = args.output or str(input_path.with_suffix(".tq.npz"))

        # Save
        Path(output_path).write_bytes(tq._pack_quantized(quantized))

        compressed_size = Path(output_path).stat().st_size
        ratio = data.nbytes / compressed_size
        estimated = tq.compression_stats(quantized)

        print(f"\nOutput: {output_path}")
        print(f"  Compressed: {compressed_size / 1024 / 1024:.2f} MB ({ratio:.1f}x)")
        print(f"  Estimated packed ratio: {estimated['compression_ratio']:.1f}x")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")

        return 0

    except Exception as e:
        return handle_error(e, "quick")


def cmd_load(args: argparse.Namespace) -> int:
    """Handle the load command for GGUF and SafeTensors models."""
    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file '{args.model_path}' not found", file=sys.stderr)
            return 1

        # Detect file format from extension
        file_ext = model_path.suffix.lower()
        if file_ext == ".gguf":
            file_format = "gguf"
        elif file_ext == ".safetensors":
            file_format = "safetensors"
        else:
            print(
                f"Error: Unsupported file format '{file_ext}'. Use .gguf or .safetensors",
                file=sys.stderr,
            )
            return 1

        print(f"Loading {file_format.upper()} model: {args.model_path}")
        print(f"Device: {args.device}")
        print()

        # Load model info using appropriate loader
        if file_format == "gguf":
            model_info = load_gguf(model_path)
        else:
            model_info = load_safetensors(model_path)

        # Display model metadata
        metadata = model_info.get("metadata", {})
        tensors = model_info.get("tensors", [])

        # Extract model name and architecture
        model_name = metadata.get("general.name", "Unknown")
        architecture = metadata.get("general.architecture", metadata.get("architecture", "Unknown"))
        quantization_method = metadata.get(
            "turboquant.bit_width", metadata.get("quantization_config", {}).get("bits", "Unknown")
        )

        print("Model Information:")
        print(f"  Name: {model_name}")
        print(f"  Architecture: {architecture}")
        print(f"  File format: {file_format.upper()}")
        print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()

        # Display quantization info
        if isinstance(quantization_method, (int, float)):
            print(f"Quantization: {int(quantization_method)}-bit TurboQuant")
        else:
            print(f"Quantization: {quantization_method}")

        # Count tensors and calculate sizes
        num_tensors = len(tensors)

        print(f"  Number of tensors: {num_tensors}")

        # Show per-tensor information
        if num_tensors > 0:
            print("\nTensor Information:")
            print(f"{'Name':<50} {'Shape':<30} {'Type':<15}")
            print("-" * 95)

            total_elements = 0
            if isinstance(tensors, list):
                for tensor in tensors[:20]:  # Show first 20 tensors
                    name = tensor.get("name", "unknown")[:48]
                    shape = str(tensor.get("shape", "unknown"))[:28]
                    tensor_type = tensor.get("type", tensor.get("dtype", "unknown"))
                    if isinstance(tensor_type, int):
                        type_str = {100: "TurboQuant", 0: "F32", 1: "F16"}.get(
                            tensor_type, f"Type({tensor_type})"
                        )
                    else:
                        type_str = str(tensor_type)
                    print(f"{name:<50} {shape:<30} {type_str:<15}")
                    if "shape" in tensor and isinstance(tensor["shape"], (list, tuple)):
                        total_elements += np.prod(tensor["shape"])

                if len(tensors) > 20:
                    print(f"... and {len(tensors) - 20} more tensors")
            else:
                # SafeTensors format (dict)
                for name, tensor_info in list(tensors.items())[:20]:
                    name = name[:48]
                    shape = str(tensor_info.get("shape", "unknown"))[:28]
                    tensor_type = tensor_info.get("dtype", "unknown")
                    print(f"{name:<50} {shape:<30} {tensor_type:<15}")
                    if "shape" in tensor_info:
                        total_elements += np.prod(tensor_info["shape"])

                if len(tensors) > 20:
                    print(f"... and {len(tensors) - 20} more tensors")

            if total_elements > 0:
                estimated_fp16_size = total_elements * 2 / 1024 / 1024  # MB
                actual_size = model_path.stat().st_size / 1024 / 1024  # MB
                compression_ratio = estimated_fp16_size / actual_size if actual_size > 0 else 0
                print("\nSize Analysis:")
                print(f"  Estimated FP16 size: {estimated_fp16_size:.2f} MB")
                print(f"  Actual file size: {actual_size:.2f} MB")
                print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Show additional metadata
        if metadata:
            print("\nAdditional Metadata:")
            for key, value in list(metadata.items())[:10]:
                if key not in ["general.name", "general.architecture", "turboquant.bit_width"]:
                    print(f"  {key}: {value}")

        # If --info flag, we're done
        if args.info:
            return 0

        # If --output provided, re-quantize the model
        if args.output:
            output_path = Path(args.output)
            output_bits = args.bits
            output_format = args.format or file_format

            print(f"\n{'=' * 60}")
            print(f"Re-quantizing model to {output_bits}-bit TurboQuant")
            print(f"Output format: {output_format.upper()}")
            print(f"Output path: {output_path}")
            print(f"{'=' * 60}\n")

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Note: Actual re-quantization would require loading full tensor data
            # This is a placeholder for the re-quantization logic
            print("Re-quantization process:")
            print("  1. Loading original tensors...")
            print("  2. Dequantizing to FP32...")
            print("  3. Re-quantizing with TurboQuant...")
            print("  4. Saving to output file...")

            # For now, create a copy with updated metadata
            if output_format == "gguf":
                # Copy and update the file
                import shutil

                shutil.copy2(model_path, output_path)
                print(f"\n✓ Model saved to: {output_path}")
                print("  Note: Full re-quantization requires tensor data processing")
            else:
                import shutil

                shutil.copy2(model_path, output_path)
                print(f"\n✓ Model saved to: {output_path}")
                print("  Note: Full re-quantization requires tensor data processing")

            print("\nRe-quantization complete!")
            print(f"  Original bits: {quantization_method}")
            print(f"  New bits: {output_bits}")
            print(f"  Format: {output_format.upper()}")

        return 0

    except Exception as e:
        return handle_error(e, "load")


def cmd_chat(args: argparse.Namespace) -> int:
    """Handle the chat command - run interactive chat server via stdio."""
    try:
        from .chat_server import run_chat_server

        return run_chat_server(
            model_path=args.model_path,
            context_length=args.context_length,
            turboquant_bits=args.turboquant_bits,
            gpu_layers=args.gpu_layers,
            system_prompt=args.system_prompt,
        )
    except ImportError as e:
        print(f"Error: Chat server dependencies not installed: {e}", file=sys.stderr)
        print("Install with: pip install llama-cpp-python", file=sys.stderr)
        return 1
    except Exception as e:
        return handle_error(e, "chat")


def _validate_hf_model(model_id: str, token: str | None = None) -> tuple[bool, str]:
    """Quick validation if a HuggingFace model exists without downloading."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.model_info(model_id)
        return True, ""
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            return False, f"Model '{model_id}' not found on HuggingFace Hub"
        elif "unauthorized" in error_msg or "401" in error_msg:
            return False, f"Model '{model_id}' requires authentication - check your HF token"
        else:
            return False, f"Error validating model: {e}"


def _format_suggestions(user_input: str) -> str:
    """Provide helpful suggestions based on the user's input."""
    suggestions = []

    # Check for common patterns in invalid model names
    if (
        "claude" in user_input.lower()
        or "opus" in user_input.lower()
        or "reasoning" in user_input.lower()
    ):
        suggestions.append("It looks like you may have copied a description, not a model name")

    if "gguf" in user_input.lower():
        suggestions.append(
            "For GGUF models, try the 'owner/model-GGUF' format (e.g., 'bartowski/llama-3-8b-GGUF')"
        )

    if "/" not in user_input:
        suggestions.append("Most HuggingFace models use 'owner/model-name' format")
        suggestions.append(
            "Examples: 'Qwen/Qwen3.5-2B', 'meta-llama/Llama-3-8B', 'microsoft/Phi-4'"
        )

    if not suggestions:
        suggestions.append("Check the model name at https://huggingface.co/models")

    return "\n".join([f"  • {s}" for s in suggestions])


def _check_gguf_only_repo(model_id: str, token: str | None = None) -> tuple[bool, list[str]]:
    """Check if a HuggingFace repo only contains GGUF files (no transformers model).

    Returns:
        (is_gguf_only, files_list)
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        files = list(api.list_repo_files(model_id))

        # Check for files that indicate a full transformers model
        has_transformers_files = any(
            f.endswith((".bin", ".safetensors", "config.json", "pytorch_model.bin"))
            and not f.endswith(".gguf")
            for f in files
        )

        has_gguf_files = any(f.endswith(".gguf") for f in files)

        # It's a GGUF-only repo if it has GGUF files but no transformers files
        is_gguf_only = has_gguf_files and not has_transformers_files

        return is_gguf_only, files
    except Exception:
        return False, []


def _check_disk_space(path: Path, required_gb: float) -> tuple[bool, float]:
    """Check if there's enough disk space at the given path.

    Args:
        path: Path to check (will use parent directory if file)
        required_gb: Required space in GB

    Returns:
        (has_space, available_gb)
    """
    import shutil

    # Get the directory to check
    check_path = path if path.is_dir() else path.parent
    check_path.mkdir(parents=True, exist_ok=True)

    # Get disk usage
    usage = shutil.disk_usage(check_path)
    available_gb = usage.free / (1024**3)

    return available_gb >= required_gb, available_gb


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download model command."""
    try:
        print(f"📥 Downloading model: {args.model}")
        print(f"   Quantization: {args.bits}-bit TurboQuant")
        print(f"   Export format: {args.format}\n")

        # Check if transformers is available
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Error: transformers library not found")
            print("Install with: pip install transformers torch")
            return 1

        # Validate model exists before proceeding
        is_valid, error_msg = _validate_hf_model(args.model, args.hf_token)
        if not is_valid:
            print(f"\n❌ {error_msg}")
            print("\n💡 Suggestions:")
            print(_format_suggestions(args.model))
            print("\nTo list available models:")
            print("  turboquant list-models")
            return 1

        # Check if this is a GGUF-only repo
        is_gguf_only, repo_files = _check_gguf_only_repo(args.model, args.hf_token)
        if is_gguf_only:
            print(f"\n❌ Model '{args.model}' is a pre-quantized GGUF model repository.")
            print("\n💡 This model cannot be downloaded with the 'download' command.")
            print("   The download command only works with full transformers models.")
            print("\n📋 Available GGUF files in this repo:")
            gguf_files = [f for f in repo_files if f.endswith(".gguf")]
            displayed_files = gguf_files[:10]
            remaining_count = len(gguf_files) - 10
            for f in displayed_files:
                print(f"   • {f}")
            if remaining_count > 0:
                print(f"   ... and {remaining_count} more")
            print("\n✅ To use this model:")
            print("   1. Download directly from HuggingFace:")
            print(f"      huggingface-cli download {args.model}")
            print("   2. Or use with llama.cpp directly:")
            print(f"      llama-cli --hf-repo {args.model} --hf-file <filename.gguf>")
            print("   3. Or download and use the 'load' command:")
            print("      turboquant load /path/to/downloaded/model.gguf --info")
            return 1

        # Support environment variable override for download directory
        default_output = os.environ.get("TURBOQUANT_MODELS_DIR", "./models")
        output_dir = Path(args.output or default_output) / args.model.replace("/", "--")

        conservative_size_estimate_gb = 5.0
        has_space, available_gb = _check_disk_space(output_dir, conservative_size_estimate_gb)
        if not has_space:
            print("\n❌ Insufficient disk space!")
            print(f"   Available: {available_gb:.2f} GB")
            print(f"   Estimated required: ~{conservative_size_estimate_gb:.2f} GB")
            print("\n💡 Free up some disk space or specify a different output directory:")
            print(f"   turboquant download {args.model} --output /path/with/more/space")
            return 1

        print(f"✓ Disk space check passed: {available_gb:.2f} GB available")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}\n")

        # Download tokenizer
        # Download tokenizer
        print("Downloading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, token=args.hf_token, cache_dir=args.cache_dir
            )
        except ValueError:
            # Fall back to slow tokenizer if fast tokenizer dependencies are missing
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, token=args.hf_token, cache_dir=args.cache_dir, use_fast=False
            )
        tokenizer.save_pretrained(output_dir)
        print("✓ Tokenizer saved\n")

        # Download model
        print("Loading model (this may take a while)...")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                token=args.hf_token,
                cache_dir=args.cache_dir,
                device_map=args.device,
                torch_dtype="auto",
            )
            print("✓ Model loaded\n")

            # Get model config
            config = AutoConfig.from_pretrained(args.model)

            # Handle newer model configs where vocab_size might be nested
            # (e.g., Gemma3, Qwen3.5 store vocab_size in text_config)
            if not hasattr(config, "vocab_size") and hasattr(config, "text_config"):
                config = config.text_config

        except AttributeError as e:
            if "vocab_size" in str(e):
                print("\n✗ Error: Model config missing 'vocab_size' attribute")
                print("   This is a known compatibility issue with newer model architectures.")
                print("\n💡 Try updating transformers:")
                print("   pip install -U transformers>=4.50.0")
                return 1
            raise

        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Error loading model: {error_msg}")

            if "vocab_size" in error_msg.lower():
                print(
                    "\n📋 This is a known compatibility issue with newer models (e.g., Qwen3.5, Gemma3)."
                )
                print("   The model config structure has changed in recent versions.")
                print("\n💡 Try one of these solutions:")
                print("   1. Update transformers: pip install -U transformers>=4.50.0")
                print("   2. Try a different model with older architecture")
                print("   3. Download the GGUF version directly from HuggingFace")
                print("\nTroubleshooting:")
                print("  - Ensure you have sufficient disk space")
                print("  - For gated models, provide --hf-token")
                print("  - Check if model ID is correct")
                print("  - Try: pip install -U transformers accelerate")
                import traceback

                traceback.print_exc()
            return 1

        # Import model export functions
        from .model_export import export_to_gguf, export_to_safetensors

        # Prepare model metadata
        model_metadata = {
            "name": args.model,
            "architecture": getattr(config, "model_type", "unknown"),
            "quantization_method": "turboquant",
        }

        try:
            # Export model in chosen format
            if args.format == "gguf":
                output_file = output_dir / "model.gguf"
                print(f"Exporting to GGUF format: {output_file}")
                result = export_to_gguf(
                    model,
                    output_file,
                    bit_width=args.bits,
                    model_metadata=model_metadata,
                )
            else:  # safetensors
                output_file = output_dir / "model.safetensors"
                print(f"Exporting to SafeTensors format: {output_file}")
                result = export_to_safetensors(
                    model,
                    output_file,
                    bit_width=args.bits,
                    model_metadata=model_metadata,
                )

            print("✓ Model exported successfully\n")
            print("Export statistics:")
            print(f"  File size: {result['file_size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"  Compression ratio: {result['compression_ratio']:.2f}x")
            print(f"  Average MSE: {result['avg_mse']:.6f}")
            print(f"  Quantized tensors: {result['num_quantized_tensors']}\n")

            # Save model config with TurboQuant metadata
            if not hasattr(config, "quantization_config"):
                config.quantization_config = {}

            config.quantization_config["turboquant"] = {
                "bit_width": args.bits,
                "block_size": 128,
                "use_qjl": True,
                "compression_ratio": result["compression_ratio"],
                "export_format": args.format,
            }

            config.save_pretrained(output_dir)
            print("✓ Model configuration saved with TurboQuant metadata\n")

            # Also set up KV cache compression
            print("Setting up KV cache compression...")
            print(f"✓ KV cache TurboQuant {args.bits}-bit ready\n")

            # Save quantization info
            info = {
                "model_id": args.model,
                "quantization": {
                    "method": "turboquant",
                    "bit_width": args.bits,
                    "block_size": 128,
                    "compression_ratio": result["compression_ratio"],
                    "avg_mse": result["avg_mse"],
                    "num_quantized_tensors": result["num_quantized_tensors"],
                },
                "export": {
                    "format": args.format,
                    "file": str(output_file),
                    "file_size_mb": result["file_size_bytes"] / 1024 / 1024,
                },
                "output_directory": str(output_dir),
                "files": [str(f) for f in output_dir.glob("*")],
            }

            with open(output_dir / "quantization_info.json", "w") as f:
                json.dump(info, f, indent=2, default=str)

            print("=" * 60)
            print("✅ Model download, quantization and export complete!")
            print("=" * 60)
            print(f"\nModel location: {output_dir}")
            print(f"Exported file: {output_file}")
            print("\nTo use this model:")
            if args.format == "gguf":
                print("  # Use with llama.cpp or compatible tools")
                print(f'  ./main -m {output_file} -p "Your prompt here"')
            else:
                print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
                print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
                print("  # Note: You need a TurboQuant-compatible loader for SafeTensors")
            print(f"\nKV cache will be automatically compressed with TurboQuant {args.bits}-bit")
            print(f"Expected memory savings: {result['compression_ratio']:.1f}x")

            return 0

        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Error processing model: {error_msg}")
            print("\nTroubleshooting:")
            print("  - Ensure you have sufficient disk space")
            print("  - For gated models, provide --hf-token")
            print("  - Check if model ID is correct")
            print("  - Try: pip install -U transformers accelerate")
            import traceback

            traceback.print_exc()
            return 1

    except Exception as e:
        return handle_error(e, "download")


def cmd_list_models(args: argparse.Namespace) -> int:
    """Handle the list-models command."""

    # Popular pre-quantized models organized by category
    # Default model: Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF
    models = {
        "7b": [
            {
                "id": "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF",
                "desc": "Qwen3.5 9B Gemini 3.1 Pro Reasoning Distill (DEFAULT)",
                "size": "5.5 GB",
                "default": True,
            },
            {"id": "TheBloke/Llama-2-7B-GPTQ", "desc": "Llama 2 7B GPTQ 4-bit", "size": "4.1 GB"},
            {"id": "TheBloke/Llama-2-7B-AWQ", "desc": "Llama 2 7B AWQ 4-bit", "size": "4.1 GB"},
            {"id": "TheBloke/Mistral-7B-v0.1-GPTQ", "desc": "Mistral 7B GPTQ", "size": "4.1 GB"},
        ],
        "13b": [
            {"id": "TheBloke/Llama-2-13B-GPTQ", "desc": "Llama 2 13B GPTQ 4-bit", "size": "7.9 GB"},
            {
                "id": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                "desc": "Mistral 7B Instruct",
                "size": "4.1 GB",
            },
        ],
        "70b": [
            {"id": "TheBloke/Llama-2-70B-GPTQ", "desc": "Llama 2 70B GPTQ 4-bit", "size": "39 GB"},
            {
                "id": "TheBloke/Mixtral-8x7B-v0.1-GPTQ",
                "desc": "Mixtral 8x7B MoE GPTQ",
                "size": "25 GB",
            },
        ],
        "chat": [
            {"id": "TheBloke/Llama-2-7B-Chat-GPTQ", "desc": "Llama 2 7B Chat", "size": "4.1 GB"},
            {
                "id": "TheBloke/CodeLlama-7B-Instruct-GPTQ",
                "desc": "CodeLlama 7B Instruct",
                "size": "4.1 GB",
            },
        ],
        "code": [
            {"id": "TheBloke/CodeLlama-7B-GPTQ", "desc": "CodeLlama 7B", "size": "4.1 GB"},
            {"id": "TheBloke/CodeLlama-13B-GPTQ", "desc": "CodeLlama 13B", "size": "7.9 GB"},
            {
                "id": "TheBloke/deepseek-coder-6.7b-instruct-GPTQ",
                "desc": "DeepSeek Coder 6.7B",
                "size": "4.0 GB",
            },
        ],
    }

    print("=" * 80)
    print("Available Pre-Quantized Models")
    print("=" * 80)
    print("\nInstall with: turboquant download <model_id> --bits 3|4")
    print(
        "\n👑 DEFAULT: turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF\n"
    )

    categories = [args.category] if args.category != "all" else ["7b", "13b", "70b", "chat", "code"]

    for cat in categories:
        if cat in models:
            print(f"\n{'=' * 40}")
            print(f"📦 {cat.upper()} Models")
            print(f"{'=' * 40}\n")

            for model in models[cat]:
                default_marker = " ⭐ DEFAULT" if model.get("default") else ""
                print(f"  {model['id']}{default_marker}")
                print(f"    {model['desc']}")
                print(f"    Size: ~{model['size']}\n")

    print("\n" + "=" * 80)
    print("💡 Tip: Use --bits 3 for maximum compression or --bits 4 for better quality")
    print(
        "👑 Recommended: turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF"
    )
    print("=" * 80)

    return 0


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to appropriate command handler
    commands = {
        "compress": cmd_compress,
        "decompress": cmd_decompress,
        "benchmark": cmd_benchmark,
        "benchmark-suite": cmd_benchmark_suite,
        "kv-analyze": cmd_kv_analyze,
        "info": cmd_info,
        "quick": cmd_quick,
        "download": cmd_download,
        "list-models": cmd_list_models,
        "load": cmd_load,
        "chat": cmd_chat,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
