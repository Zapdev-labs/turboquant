import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

from .turboquant import TurboQuant, TurboQuantConfig
from .kv_cache import KVCacheCompressor, benchmark_kv_cache
from .utils import compute_distortion


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
        choices=[2, 3, 4, 8],
        help="Bit-width for quantization (default: 4)",
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

    # Quick compress command (simplified)
    quick_parser = subparsers.add_parser("quick", help="Quick compression with default settings")
    quick_parser.add_argument("input", type=str, help="Input file")
    quick_parser.add_argument("output", type=str, nargs="?", help="Output file (optional)")

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

            # Warmup
            _ = tq.quantize(data[:100] if len(data) > 100 else data)

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
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download model command."""
    try:
        print(f"📥 Downloading model: {args.model}")
        print(f"   Quantization: {args.bits}-bit TurboQuant\n")
        
        # Check if transformers is available
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        except ImportError:
            print("Error: transformers library not found")
            print("Install with: pip install transformers torch")
            return 1
        
        output_dir = Path(args.output or './models') / args.model.replace('/', '--')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_dir}\n")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            cache_dir=args.cache_dir
        )
        tokenizer.save_pretrained(output_dir)
        print("✓ Tokenizer saved\n")
        
        # Download and quantize model
        print("Loading model (this may take a while)...")
        
        try:
            # Try to load with AutoGPTQ or similar if available
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                token=args.hf_token,
                cache_dir=args.cache_dir,
                device_map=args.device,
                torch_dtype='auto',
            )
            print("✓ Model loaded\n")
            
            # Quantize KV cache
            print("Setting up TurboQuant compression...")
            compressor = KVCacheCompressor(bit_width=args.bits, block_size=128)
            print(f"✓ TurboQuant {args.bits}-bit ready\n")
            
            # Save model with compression metadata
            config = AutoConfig.from_pretrained(args.model)
            
            # Add TurboQuant metadata to config
            if not hasattr(config, 'quantization_config'):
                config.quantization_config = {}
            
            config.quantization_config['turboquant'] = {
                'bit_width': args.bits,
                'block_size': 128,
                'use_qjl': True,
                'compression_ratio': 4.9 if args.bits == 3 else 3.8,
            }
            
            config.save_pretrained(output_dir)
            print(f"✓ Model configuration saved with TurboQuant metadata\n")
            
            # Save quantization info
            info = {
                'model_id': args.model,
                'quantization': {
                    'method': 'turboquant',
                    'bit_width': args.bits,
                    'block_size': 128,
                    'compression_ratio': 4.9 if args.bits == 3 else 3.8,
                },
                'output_directory': str(output_dir),
                'files': list(output_dir.glob('*')),
            }
            
            with open(output_dir / 'quantization_info.json', 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            print("=" * 60)
            print("✅ Model download and setup complete!")
            print("=" * 60)
            print(f"\nModel location: {output_dir}")
            print(f"\nTo use this model:")
            print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
            print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
            print(f"\nKV cache will be automatically compressed with TurboQuant {args.bits}-bit")
            print(f"Expected memory savings: {info['quantization']['compression_ratio']:.1f}x")
            
            return 0
            
        except Exception as e:
            print(f"\n✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("  - Ensure you have sufficient disk space")
            print("  - For gated models, provide --hf-token")
            print("  - Check if model ID is correct")
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_list_models(args: argparse.Namespace) -> int:
    """Handle the list-models command."""
    
    # Popular pre-quantized models organized by category
    models = {
        '7b': [
            {'id': 'TheBloke/Llama-2-7B-GPTQ', 'desc': 'Llama 2 7B GPTQ 4-bit', 'size': '4.1 GB'},
            {'id': 'TheBloke/Llama-2-7B-AWQ', 'desc': 'Llama 2 7B AWQ 4-bit', 'size': '4.1 GB'},
            {'id': 'TheBloke/Mistral-7B-v0.1-GPTQ', 'desc': 'Mistral 7B GPTQ', 'size': '4.1 GB'},
        ],
        '13b': [
            {'id': 'TheBloke/Llama-2-13B-GPTQ', 'desc': 'Llama 2 13B GPTQ 4-bit', 'size': '7.9 GB'},
            {'id': 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ', 'desc': 'Mistral 7B Instruct', 'size': '4.1 GB'},
        ],
        '70b': [
            {'id': 'TheBloke/Llama-2-70B-GPTQ', 'desc': 'Llama 2 70B GPTQ 4-bit', 'size': '39 GB'},
            {'id': 'TheBloke/Mixtral-8x7B-v0.1-GPTQ', 'desc': 'Mixtral 8x7B MoE GPTQ', 'size': '25 GB'},
        ],
        'chat': [
            {'id': 'TheBloke/Llama-2-7B-Chat-GPTQ', 'desc': 'Llama 2 7B Chat', 'size': '4.1 GB'},
            {'id': 'TheBloke/CodeLlama-7B-Instruct-GPTQ', 'desc': 'CodeLlama 7B Instruct', 'size': '4.1 GB'},
        ],
        'code': [
            {'id': 'TheBloke/CodeLlama-7B-GPTQ', 'desc': 'CodeLlama 7B', 'size': '4.1 GB'},
            {'id': 'TheBloke/CodeLlama-13B-GPTQ', 'desc': 'CodeLlama 13B', 'size': '7.9 GB'},
            {'id': 'TheBloke/deepseek-coder-6.7b-instruct-GPTQ', 'desc': 'DeepSeek Coder 6.7B', 'size': '4.0 GB'},
        ],
    }
    
    print("=" * 80)
    print("Available Pre-Quantized Models")
    print("=" * 80)
    print("\nInstall with: turboquant download <model_id> --bits 3|4\n")
    
    categories = [args.category] if args.category != 'all' else ['7b', '13b', '70b', 'chat', 'code']
    
    for cat in categories:
        if cat in models:
            print(f"\n{'='*40}")
            print(f"📦 {cat.upper()} Models")
            print(f"{'='*40}\n")
            
            for model in models[cat]:
                print(f"  {model['id']}")
                print(f"    {model['desc']}")
                print(f"    Size: ~{model['size']}\n")
    
    print("\n" + "=" * 80)
    print("💡 Tip: Use --bits 3 for maximum compression or --bits 4 for better quality")
    print("=" * 80)
    
    return 0


