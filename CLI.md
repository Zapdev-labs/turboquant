# FastVQ CLI Documentation

The `fastvq` CLI provides a comprehensive interface for compressing, analyzing, and managing quantized AI models. `turboquant` and `tq` remain available as legacy aliases.

## Installation

### From PyPI (Recommended)

```bash
pip install fastvq
```

The package supports both `fastvq` and legacy `turboquant` imports:

```python
from fastvq import TurboQuant
```

### From Source

```bash
git clone https://github.com/Zapdev-labs/turboquant
cd turboquant
pip install -e .
```

### Verify Installation

```bash
fastvq --help
turboquant --help  # Legacy alias
tq --help  # Short alias
```

## Global Options

All commands support the following:
- `--help` - Show help for any command
- Progress output with file sizes and compression metrics

## Commands

### `compress` - Compress Arrays or Tensors

Compress numpy arrays using TurboQuant quantization.

```bash
fastvq compress input.npy output.tq --bits 3
```

**Arguments:**
- `input` - Input file (.npy, .npz)
- `output` - Output file (.tq or .tq.npz)

**Options:**
- `--bits, -b` - Bit-width: 2, 3 (default), or 4
- `--block-size` - Block size: 32, 64, 128, or 256 (default: 128)
- `--no-qjl` - Disable QJL error correction
- `--seed` - Random seed for reproducibility (default: 42)
- `--rotation` - Rotation backend: hadamard (default) or random
- `--radii-dtype` - Radii storage dtype: float16 (default) or float32

**Example:**
```bash
# 3-bit compression with defaults
turboquant compress data.npy compressed.tq

# 4-bit with custom block size
turboquant compress data.npy compressed.tq --bits 4 --block-size 64

# Disable QJL for faster compression
turboquant compress data.npy compressed.tq --bits 3 --no-qjl
```

### `decompress` - Decompress .tq Files

Reconstruct arrays from compressed TurboQuant files.

```bash
turboquant decompress compressed.tq output.npy
```

**Arguments:**
- `input` - Input .tq file
- `output` - Output .npy file

### `quick` - Quick Compression with Defaults

Fast compression with sensible defaults (3-bit, block size 128).

```bash
# With explicit output
turboquant quick data.npy compressed.tq

# Auto-generate output name (data.tq.npz)
turboquant quick data.npy
```

### `benchmark` - Benchmark Compression Quality and Speed

Test different bit-widths and measure performance.

```bash
fastvq benchmark input.npy --bits 3,4 --trials 10
```

**Arguments:**
- `input` - Input .npy file

**Options:**
- `--bits, -b` - Comma-separated bit-widths (default: "3")
- `--trials` - Number of benchmark trials (default: 10)
- `--block-size` - Block size: 32, 64, 128, or 256
- `--rotation` - Rotation backend: hadamard or random
- `--output, -o` - Save results to JSON file

### `benchmark-suite` - Synthetic Benchmark Grid

Run repeatable synthetic benchmarks across shapes, bit-widths, and block sizes.

```bash
fastvq benchmark-suite --shapes 1024x128,4096x128,1024x192 --bits 2,3,4 --output benchmarks/results.json
```

Use semicolons for comma-style multidimensional shapes:

```bash
fastvq benchmark-suite --shapes "1,8,1024,128;1,8,2048,192"
```

**Example Output:**
```
Bit-width    Compress     Decompress       MSE          SNR
----------------------------------------------------------------------
3-bit          12.34 ms      8.21 ms     0.715000   3.14 dB
4-bit           9.87 ms      7.12 ms     0.193000   5.21 dB
```

### `kv-analyze` - Analyze KV Cache Compression for LLMs

Calculate memory usage and context window improvements for transformer models.

```bash
turboquant kv-analyze --model-size 70b --seq-len 100000 --bits 3
```

**Options:**
- `--model-size` - Model size: 7b, 13b, 30b, 70b (default), or 175b
- `--seq-len` - Sequence length in tokens (default: 100000)
- `--batch-size` - Batch size (default: 1)
- `--bits, -b` - Bit-width: 2, 3 (default), or 4
- `--vram` - Available VRAM in GB (default: 72.0)

**Example Output:**
```
============================================================
KV Cache Analysis for 70B Model
============================================================

Memory Usage (full model, 80 layers):
  FP16 KV cache:     3.05 GB
  TurboQuant 3-bit:  0.62 GB

Context Window Analysis:
  Max context (FP16):  1,037,597 tokens
  Max context (TQ):    5,108,173 tokens
  Improvement:         4.9x
```

### `info` - Show Information About Compressed File

Display compression parameters and metadata.

```bash
turboquant info compressed.tq
```

**Arguments:**
- `file` - Compressed .tq file

**Example Output:**
```
File: compressed.tq
Size: 1523.45 KB

Compression Parameters:
  Bit-width: 3
  Block size: 128
  QJL enabled: True

Original Shape: (1, 8, 1024, 128)

Data Arrays:
  norms: shape=(8192,), dtype=float32
  polar_indices: shape=(8192, 64), dtype=uint8
  polar_radii: shape=(8192,), dtype=float32
  original_shape: shape=(4,), dtype=int64
  bit_width: shape=(1,), dtype=int64
  block_size: shape=(1,), dtype=int64
  use_qjl: shape=(1,), dtype=bool
```

### `download` - Download and Quantize Models from HuggingFace

Download models and quantize them with TurboQuant.

```bash
turboquant download meta-llama/Llama-2-7b-hf --bits 3 --format gguf
```

**Arguments:**
- `model` - Model ID (e.g., "meta-llama/Llama-2-7b-hf")

**Options:**
- `--output, -o` - Output directory (default: ./models)
- `--bits, -b` - Bit-width: 2, 3 (default), or 4
- `--format, -f` - Export format: gguf (default) or safetensors
- `--hf-token` - HuggingFace token for gated models
- `--cache-dir` - Cache directory for downloaded models
- `--device` - Device: auto (default), cpu, cuda, or mps

**Example:**
```bash
# Download Llama 2 7B with 3-bit quantization
turboquant download meta-llama/Llama-2-7b-hf --bits 3 --format gguf

# Download with HuggingFace token for gated models
turboquant download meta-llama/Llama-2-7b-hf --hf-token YOUR_TOKEN

# Save to specific directory
turboquant download TheBloke/Llama-2-7B-GPTQ --output ./my-models --bits 4
```

### `list-models` - List Popular Pre-Quantized Models

Show available pre-quantized models from the community.

```bash
turboquant list-models --category 7b
turboquant list-models --category all  # Default
```

**Options:**
- `--category` - Filter: 7b, 13b, 70b, chat, code, or all (default)

### `load` - Load and Re-quantize Models

Load existing GGUF or SafeTensors models and optionally re-quantize them.

```bash
turboquant load model.gguf --info
turboquant load model.gguf --output requantized.gguf --bits 3
```

**Arguments:**
- `model_path` - Path to .gguf or .safetensors file

**Options:**
- `--output, -o` - Output path for re-quantized model
- `--bits, -b` - Bit-width for re-quantization: 2, 3 (default), or 4
- `--format, -f` - Force output format: gguf or safetensors
- `--info` - Only show model info, don't load fully
- `--device` - Device for loading: auto (default), cpu, cuda, or mps

**Example:**
```bash
# Just show model information
turboquant load model.gguf --info

# Re-quantize to 3-bit
turboquant load model.gguf --output model-3bit.gguf --bits 3

# Convert format while re-quantizing
turboquant load model.safetensors --output model.gguf --bits 3 --format gguf
```

### `chat` - Run Interactive Chat Server

Run an interactive chat server for GGUF models (for T3code integration).

```bash
turboquant chat model.gguf --context-length 4096 --turboquant-bits 3
```

**Arguments:**
- `model_path` - Path to GGUF model file

**Options:**
- `--context-length, -c` - Context length in tokens (default: 4096)
- `--turboquant-bits, -b` - TurboQuant bit-width for KV cache: 2, 3 (default), or 4
- `--gpu-layers, -g` - GPU layers (-1 for auto, 0 for CPU only, default: -1)
- `--system-prompt, -s` - System prompt (default: "You are a helpful AI assistant.")

**Example:**
```bash
# Basic chat with 3-bit KV cache compression
turboquant chat model.gguf

# Full GPU offload with longer context
turboquant chat model.gguf --context-length 8192 --gpu-layers 35

# CPU-only with custom system prompt
turboquant chat model.gguf --gpu-layers 0 --system-prompt "You are a coding assistant."
```

## Environment Variables

- `HF_TOKEN` - HuggingFace token for gated models (alternative to --hf-token)
- `CUDA_VISIBLE_DEVICES` - Control GPU visibility for CUDA operations

## Exit Codes

- `0` - Success
- `1` - Error (invalid arguments, file not found, processing error)

## Examples

### Complete Workflow: Download → Quantize → Chat

```bash
# 1. List available models
turboquant list-models --category 7b

# 2. Download a model with 3-bit quantization
turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF \
  --output ./models \
  --bits 3 \
  --format gguf

# 3. Analyze KV cache requirements
turboquant kv-analyze --model-size 70b --seq-len 100000 --bits 3

# 4. Chat with the model
turboquant chat ./models/Jackrong--Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF/model.gguf \
  --turboquant-bits 3 \
  --context-length 4096
```

### Batch Processing

```bash
# Compress multiple numpy files
for file in *.npy; do
    turboquant compress "$file" "compressed/${file%.npy}.tq" --bits 3
done

# Benchmark multiple configurations
turboquant benchmark data.npy --bits 2,3,4 --output results.json
```

### Memory Planning

```bash
# Check if your model fits in VRAM
turboquant kv-analyze --model-size 70b --seq-len 200000 --bits 3 --vram 48

# Result will warn if FP16 cache exceeds available VRAM
```

## Troubleshooting

### File Not Found
```
Error: Input file 'data.npy' not found
```
Ensure the input file exists and the path is correct.

### Unsupported Format
```
Error: Unsupported file format '.txt'
```
Use .npy or .npz files for compression. Convert your data first:
```python
import numpy as np
data = np.loadtxt('data.txt')
np.save('data.npy', data)
```

### HuggingFace Token Required
```
Error: Gated model requires authentication
```
Provide `--hf-token` or set `HF_TOKEN` environment variable.

### Import Error
```
Error: Chat server dependencies not installed
```
Install llama-cpp-python for chat functionality:
```bash
pip install llama-cpp-python
```

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [Algorithm Details in README.md](README.md) - Technical documentation
- `demo.py` - Interactive demonstration script
