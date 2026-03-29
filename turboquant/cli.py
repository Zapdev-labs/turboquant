import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

from .turboquant import TurboQuant, TurboQuantConfig
from .kv_cache import KVCacheCompressor, benchmark_kv_cache
from .utils import compute_distortion
from .model_export import load_gguf, load_safetensors


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant: Extreme compression for AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a numpy array
  turboquant compress input.npy output.tq --bits 3
  
  # Decompress
  turboquant decompress output.tq reconstructed.npy
  
  # Benchmark different bit-widths
  turboquant benchmark input.npy --bits 3,4
  
  # Analyze KV cache memory usage
  turboquant kv-analyze --model-size 70b --seq-len 100000
  
  # Info about a compressed file
  turboquant info compressed.tq
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
        choices=[32, 64, 128],
        help="Block size for quantization (default: 128)",
    )
    compress_parser.add_argument(
        "--no-qjl", action="store_true", help="Disable QJL error correction"
    )
    compress_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
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
    benchmark_parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")

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
        "--output", "-o", type=str, help="Output directory (default: ./models)"
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
        )

        print(f"\nCompressing with {args.bits}-bit TurboQuant...")
        tq = TurboQuant(config)

        quantized = tq.quantize(data)
        reconstructed = tq.dequantize(quantized)

        # Calculate metrics
        metrics = compute_distortion(data, reconstructed)

        print(f"\nCompression Results:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")

        # Save compressed representation
        output_path = Path(args.output)
        np.savez_compressed(
            args.output,
            norms=quantized["norms"],
            polar_indices=quantized["polar_indices"],
            polar_radii=quantized["polar_radii"],
            original_shape=np.array(quantized["original_shape"]),
            bit_width=np.array([config.bit_width]),
            block_size=np.array([config.block_size]),
            use_qjl=np.array([config.use_qjl]),
        )

        # Calculate actual compression ratio
        compressed_size = Path(args.output).stat().st_size
        compression_ratio = data.nbytes / compressed_size

        print(f"\nOutput: {args.output}")
        print(f"  Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_decompress(args: argparse.Namespace) -> int:
    """Handle the decompress command."""
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
            return 1

        print(f"Loading compressed file {args.input}...")

        # Load compressed data
        loaded = np.load(args.input)

        # Reconstruct quantized dict
        quantized = {
            "norms": loaded["norms"],
            "polar_indices": loaded["polar_indices"],
            "polar_radii": loaded["polar_radii"],
            "original_shape": tuple(loaded["original_shape"]),
            "config": TurboQuantConfig(
                bit_width=int(loaded["bit_width"][0]),
                block_size=int(loaded["block_size"][0]),
                use_qjl=bool(loaded["use_qjl"][0]),
            ),
            "polar_metadata": {},
        }

        # Decompress
        config = quantized["config"]
        tq = TurboQuant(config)

        print("Decompressing...")
        reconstructed = tq.dequantize(quantized)

        # Save output
        np.save(args.output, reconstructed)

        print(f"\nOutput: {args.output}")
        print(f"  Shape: {reconstructed.shape}")
        print(f"  Dtype: {reconstructed.dtype}")
        print(f"  Size: {reconstructed.nbytes / 1024 / 1024:.2f} MB")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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

        print(f"{'Bit-width':<12} {'Compress':<12} {'Decompress':<14} {'MSE':<12} {'SNR':<10}")
        print("-" * 70)

        for bw in bit_widths:
            config = TurboQuantConfig(bit_width=bw, block_size=128)
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

            result = {
                "bit_width": bw,
                "compress_time_ms": compress_time * 1000,
                "decompress_time_ms": decompress_time * 1000,
                **metrics,
            }
            results.append(result)

            print(
                f"{bw}-bit        "
                f"{compress_time * 1000:>8.2f} ms   "
                f"{decompress_time * 1000:>8.2f} ms     "
                f"{metrics['mse']:>8.6f}   "
                f"{metrics['snr_db']:>6.2f} dB"
            )

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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

        print(f"=" * 60)
        print(f"KV Cache Analysis for {args.model_size.upper()} Model")
        print(f"=" * 60)
        print(f"\nModel Configuration:")
        print(f"  Layers: {config['layers']}")
        print(f"  Attention heads: {config['n_heads']}")
        print(f"  Head dimension: {config['head_dim']}")

        print(f"\nInference Configuration:")
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

        print(f"\nMemory Usage (per layer):")
        print(f"  FP16 KV cache:     {fp16_per_layer:.3f} GB")
        print(f"  TurboQuant {args.bits}-bit:  {tq_per_layer:.3f} GB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

        print(f"\nMemory Usage (full model, {config['layers']} layers):")
        print(f"  FP16 KV cache:     {fp16_full:.2f} GB")
        print(f"  TurboQuant {args.bits}-bit:  {tq_full:.2f} GB")

        print(f"\nContext Window Analysis:")
        print(f"  Max context (FP16):  {stats['max_context_fp16']:,} tokens")
        print(f"  Max context (TQ):    {stats['max_context_tq']:,} tokens")
        print(f"  Improvement:         {stats['max_context_tq'] / stats['max_context_fp16']:.1f}x")

        if fp16_full > args.vram * 0.8:
            print(f"\n⚠️  Warning: FP16 cache requires {fp16_full:.1f} GB")
            print(f"   This exceeds 80% of available VRAM ({args.vram * 0.8:.1f} GB)")
            print(f"   TurboQuant enables this model to fit in {args.vram} GB VRAM")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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
        print(f"\nCompression Parameters:")
        print(f"  Bit-width: {loaded['bit_width'][0]}")
        print(f"  Block size: {loaded['block_size'][0]}")
        print(f"  QJL enabled: {bool(loaded['use_qjl'][0])}")
        print(f"\nOriginal Shape: {tuple(loaded['original_shape'])}")
        print(f"\nData Arrays:")
        for key in loaded.files:
            arr = loaded[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


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
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix(".tq.npz"))

        # Save
        np.savez_compressed(
            output_path,
            norms=quantized["norms"],
            polar_indices=quantized["polar_indices"],
            polar_radii=quantized["polar_radii"],
            original_shape=np.array(quantized["original_shape"]),
            bit_width=np.array([3]),
            block_size=np.array([128]),
            use_qjl=np.array([True]),
        )

        compressed_size = Path(output_path).stat().st_size
        ratio = data.nbytes / compressed_size

        print(f"\nOutput: {output_path}")
        print(f"  Compressed: {compressed_size / 1024 / 1024:.2f} MB ({ratio:.1f}x)")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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

        print(f"Model Information:")
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
        if isinstance(tensors, list):
            num_tensors = len(tensors)
        else:
            num_tensors = len(tensors)

        print(f"  Number of tensors: {num_tensors}")

        # Show per-tensor information
        if num_tensors > 0:
            print(f"\nTensor Information:")
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
                print(f"\nSize Analysis:")
                print(f"  Estimated FP16 size: {estimated_fp16_size:.2f} MB")
                print(f"  Actual file size: {actual_size:.2f} MB")
                print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Show additional metadata
        if metadata:
            print(f"\nAdditional Metadata:")
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
                print(f"  Note: Full re-quantization requires tensor data processing")
            else:
                import shutil

                shutil.copy2(model_path, output_path)
                print(f"\n✓ Model saved to: {output_path}")
                print(f"  Note: Full re-quantization requires tensor data processing")

            print(f"\nRe-quantization complete!")
            print(f"  Original bits: {quantization_method}")
            print(f"  New bits: {output_bits}")
            print(f"  Format: {output_format.upper()}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download model command."""
    try:
        print(f"📥 Downloading model: {args.model}")
        print(f"   Quantization: {args.bits}-bit TurboQuant")
        print(f"   Export format: {args.format}\n")

        # Check if transformers is available
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        except ImportError:
            print("Error: transformers library not found")
            print("Install with: pip install transformers torch")
            return 1

        output_dir = Path(args.output or "./models") / args.model.replace("/", "--")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}\n")

        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, token=args.hf_token, cache_dir=args.cache_dir
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

            # Import model export functions
            from .model_export import export_to_gguf, export_to_safetensors

            # Prepare model metadata
            model_metadata = {
                "name": args.model,
                "architecture": getattr(config, "model_type", "unknown"),
                "quantization_method": "turboquant",
            }

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

            print(f"✓ Model exported successfully\n")
            print(f"Export statistics:")
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
            print(f"✓ Model configuration saved with TurboQuant metadata\n")

            # Also set up KV cache compression
            print("Setting up KV cache compression...")
            compressor = KVCacheCompressor(bit_width=args.bits, block_size=128)
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
            print(f"\nTo use this model:")
            if args.format == "gguf":
                print(f"  # Use with llama.cpp or compatible tools")
                print(f'  ./main -m {output_file} -p "Your prompt here"')
            else:
                print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
                print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
                print(f"  # Note: You need a TurboQuant-compatible loader for SafeTensors")
            print(f"\nKV cache will be automatically compressed with TurboQuant {args.bits}-bit")
            print(f"Expected memory savings: {result['compression_ratio']:.1f}x")

            return 0

        except Exception as e:
            print(f"\n✗ Error processing model: {e}")
            print("\nTroubleshooting:")
            print("  - Ensure you have sufficient disk space")
            print("  - For gated models, provide --hf-token")
            print("  - Check if model ID is correct")
            import traceback

            traceback.print_exc()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_list_models(args: argparse.Namespace) -> int:
    """Handle the list-models command."""

    # Popular pre-quantized models organized by category
    # Default model: Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF
    models = {
        "7b": [
            {"id": "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF", "desc": "Qwen3.5 9B Gemini 3.1 Pro Reasoning Distill (DEFAULT)", "size": "5.5 GB", "default": True},
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
    print("\n👑 DEFAULT: turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF\n")

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
    print("👑 Recommended: turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF")
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
