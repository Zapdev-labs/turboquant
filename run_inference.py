#!/usr/bin/env python3
"""
Run actual GGUF inference with TPS measurement.

This script tries multiple backends in order of preference:
1. llama-cpp-python (best GGUF support, but requires compilation)
2. ctransformers (simpler, but limited GGUF v3 support)
3. transformers (no GGUF support for local files)

If no backend is available, provides clear instructions on how to proceed.
"""

import time
import sys
import os
import subprocess
from pathlib import Path


def check_llama_cpp():
    """Check if llama-cpp-python is available."""
    try:
        from llama_cpp import Llama

        return True
    except ImportError:
        return False


def check_ctransformers():
    """Check if ctransformers is available."""
    try:
        from ctransformers import AutoModelForCausalLM

        return True
    except ImportError:
        return False


def check_llama_cpp_binary():
    """Check if llama.cpp binary (main/llama-cli) is available."""
    for cmd in ["llama-cli", "main", "./main", "./llama-cli"]:
        try:
            result = subprocess.run([cmd, "--help"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except:
            pass
    return None


def run_llama_cpp_python(model_path: Path, prompt: str, max_tokens: int = 32):
    """Run inference using llama-cpp-python."""
    from llama_cpp import Llama
    import os

    print(f"\n🚀 Loading with llama-cpp-python: {model_path}")
    print(f"   Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Note: 31B model on CPU will be slow (~5-10 TPS expected)")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        verbose=False,
        n_threads=os.cpu_count() or 4,
    )

    print(f"✓ Model loaded")
    print(f"   Context length: {llm.n_ctx()}")
    print(f"   Vocab size: {llm.n_vocab()}")

    print(f"\n📝 Prompt: {prompt[:80]}...")
    print(f"   Max tokens: {max_tokens}")
    print("\nGenerating...")
    print("-" * 70)

    tokens_generated = 0
    generated_text = ""

    start_time = time.perf_counter()
    first_token_time = None

    for output in llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7,
    ):
        token_text = output["choices"][0]["text"]

        if first_token_time is None:
            first_token_time = time.perf_counter()

        generated_text += token_text
        tokens_generated += 1

        if tokens_generated % 10 == 0:
            elapsed = time.perf_counter() - start_time
            current_tps = tokens_generated / elapsed
            print(
                f"  Tokens: {tokens_generated:3d} | TPS: {current_tps:5.1f} | Elapsed: {elapsed:5.2f}s",
                end="\r",
            )

    end_time = time.perf_counter()
    print()  # New line
    print("-" * 70)

    total_time = end_time - start_time
    tps = tokens_generated / total_time if total_time > 0 else 0
    ttft = first_token_time - start_time if first_token_time else 0

    return {
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "total_time": total_time,
        "tps": tps,
        "ttft": ttft,
    }


def run_llama_cpp_binary(model_path: Path, prompt: str, max_tokens: int = 128):
    """Run inference using llama.cpp binary with timing wrapper."""
    llama_bin = check_llama_cpp_binary()

    print(f"\n🚀 Running with llama.cpp binary: {llama_bin}")
    print(f"   Model: {model_path}")
    print(f"   This will run inference and measure TPS")
    print("-" * 70)

    # Write prompt to temp file
    prompt_file = Path("/tmp/llama_prompt.txt")
    prompt_file.write_text(prompt)

    start_time = time.perf_counter()

    # Run llama.cpp
    result = subprocess.run(
        [
            llama_bin,
            "-m",
            str(model_path),
            "-f",
            str(prompt_file),
            "-n",
            str(max_tokens),
            "--temp",
            "0.7",
            "--no-display-prompt",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    end_time = time.perf_counter()

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    generated_text = result.stdout
    tokens_generated = len(generated_text.split())  # Rough estimate
    total_time = end_time - start_time
    tps = tokens_generated / total_time if total_time > 0 else 0

    return {
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "total_time": total_time,
        "tps": tps,
        "ttft": 0,  # Can't measure with binary
    }


def print_diagnostic_info(model_path: Path):
    """Print diagnostic information about the model and environment."""
    print("=" * 70)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 70)

    # Model info
    print(f"\n📦 Model File:")
    print(f"   Path: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"   Exists: {model_path.exists()}")

    # Backend availability
    print(f"\n🔧 Backends:")
    print(f"   llama-cpp-python: {'✓ Available' if check_llama_cpp() else '✗ Not installed'}")
    print(f"   ctransformers: {'✓ Available' if check_ctransformers() else '✗ Not installed'}")
    llama_bin = check_llama_cpp_binary()
    print(f"   llama.cpp binary: {llama_bin if llama_bin else '✗ Not found'}")

    # System info
    print(f"\n💻 System:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")

    import psutil

    mem = psutil.virtual_memory()
    print(
        f"   RAM: {mem.total / 1024 / 1024 / 1024:.1f} GB ({mem.available / 1024 / 1024 / 1024:.1f} GB available)"
    )
    print(f"   CPUs: {psutil.cpu_count()}")

    # GGUF info
    print(f"\n📋 GGUF Format:")
    try:
        import struct

        with open(model_path, "rb") as f:
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            print(f"   Magic: {magic}")
            print(f"   Version: {version} (v3 requires llama.cpp)")
            print(f"   Tensors: {n_tensors}")
    except Exception as e:
        print(f"   Error reading: {e}")

    print("\n" + "=" * 70)


def print_installation_guide():
    """Print guide for installing missing backends."""
    print("\n" + "=" * 70)
    print("INSTALLATION GUIDE")
    print("=" * 70)

    print("""
To run GGUF inference, you need one of these backends:

1. LLAMA-CPP-PYTHON (Recommended - best GGUF v3 support)
   Install with pre-built wheels (no compilation needed):
   
   # For CPU-only:
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   
   # For CUDA (GPU):
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   
   # If prebuilt wheels don't work, install from source:
   CMAKE_ARGS="-DLLAMA_BLAS=OFF" pip install llama-cpp-python --no-cache-dir

2. LLAMA.CPP BINARY (Alternative)
   Build from source or download prebuilt:
   
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   # or: cmake -B build && cmake --build build
   
   Then use: ./main -m models/gemma-4-31b-it-Q2_K.gguf -p "Your prompt"

3. CTRANSFORMERS (Limited GGUF v3 support)
   pip install ctransformers
   
   Note: ctransformers doesn't support GGUF v3 (what your model uses).
   Use llama-cpp-python or llama.cpp binary instead.

4. ALTERNATIVE: Use a different model
   Download a GGUF v2 model or use the original model with transformers:
   
   turboquant download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits 4
   # Then run with transformers (no GGUF needed)
""")


def main():
    model_path = Path("models/gemma-4-31b-it-Q2_K.gguf")
    prompt = "What are the main differences between quantum computing and classical computing?"

    print("=" * 70)
    print("ACTUAL GGUF INFERENCE WITH TPS MEASUREMENT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Goal: 100+ TPS (tokens per second)")
    print("=" * 70)

    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print_diagnostic_info(model_path)
        return 1

    # Print diagnostic info
    print_diagnostic_info(model_path)

    # Try available backends in order of preference
    results = None
    backend_used = None

    # 1. Try llama-cpp-python (best support)
    if check_llama_cpp():
        try:
            results = run_llama_cpp_python(model_path, prompt, max_tokens=32)
            backend_used = "llama-cpp-python"
        except Exception as e:
            print(f"\n❌ llama-cpp-python failed: {e}")

    # 2. Try llama.cpp binary
    if results is None and check_llama_cpp_binary():
        try:
            results = run_llama_cpp_binary(model_path, prompt, max_tokens=128)
            if results:
                backend_used = "llama.cpp binary"
        except Exception as e:
            print(f"\n❌ llama.cpp binary failed: {e}")

    # If no backend worked, show installation guide
    if results is None:
        print("\n❌ No working inference backend found")
        print_installation_guide()
        return 1

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Backend:          {backend_used}")
    print(f"Generated tokens: {results['tokens_generated']}")
    print(f"Total time:       {results['total_time']:.2f}s")
    print(f"TPS:              {results['tps']:.1f} tokens/second")
    if results.get("ttft"):
        print(f"Time to first:    {results['ttft']:.2f}s")
    print("=" * 70)

    # TPS Analysis
    target_tps = 100
    if results["tps"] >= target_tps:
        print(f"✅ TARGET ACHIEVED: {results['tps']:.1f} TPS >= {target_tps} TPS")
    else:
        gap = target_tps - results["tps"]
        print(f"⚠️  BELOW TARGET: {results['tps']:.1f} TPS < {target_tps} TPS")
        print(f"   Gap: {gap:.1f} TPS ({(gap / target_tps) * 100:.0f}% short)")

        print(f"\n💡 To achieve 100+ TPS:")
        print(f"   1. Use GPU acceleration (CUDA/MPS)")
        print(f"   2. Use a smaller model (7B or 9B instead of 31B)")
        print(f"   3. Use higher quantization (Q4_K instead of Q2_K)")
        print(f"   4. Use optimized build with BLAS/AVX support")
        print(f"   5. Current CPU on 31B Q2_K: ~5-15 TPS expected")

    print(f"\n📝 Generated response:\n{results['generated_text'][:800]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
