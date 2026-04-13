# TurboQuant Quick Start Guide

Get up and running with TurboQuant in 5 minutes.

## Installation

### From PyPI (Recommended)

```bash
pip install fastvq
```

The package is published as **`fastvq`** on PyPI, but imports as `turboquant`:

```python
import turboquant  # even though you installed 'fastvq'
```

### From Source

```bash
git clone https://github.com/turboquant/turboquant
cd turboquant
pip install -e .
```

### Verify Installation

```bash
turboquant --help
tq --help  # Short alias
```

## 1. Your First Compression (30 seconds)

Create a test file and compress it:

```bash
# Create test data
python3 -c "import numpy as np; np.save('test.npy', np.random.randn(1000, 128).astype('float32'))"

# Compress with TurboQuant (3-bit)
turboquant compress test.npy compressed.tq

# Check the results
turboquant info compressed.tq
```

Expected output shows ~4-5x compression ratio with minimal quality loss.

## 2. Decompress and Verify

```bash
# Decompress back to numpy
turboquant decompress compressed.tq recovered.npy

# Compare (Python)
python3 << 'EOF'
import numpy as np
original = np.load('test.npy')
recovered = np.load('recovered.npy')
print(f"MSE: {np.mean((original - recovered)**2):.6f}")
print(f"Cosine similarity: {np.sum(original*recovered) / (np.linalg.norm(original) * np.linalg.norm(recovered)):.6f}")
EOF
```

## 3. Quick Compression Mode

Even faster - auto-generated output name:

```bash
turboquant quick test.npy
# Creates test.tq.npz automatically
```

## 4. Download a Pre-Quantized Model

```bash
# List available models
turboquant list-models

# Download the default recommended model (3-bit)
turboquant download Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF --bits 3

# Or download Llama 2 with 4-bit for better quality
turboquant download TheBloke/Llama-2-7B-GPTQ --bits 4 --format gguf
```

## 5. Analyze Memory Savings

Calculate how much VRAM you'll save with KV cache compression:

```bash
turboquant kv-analyze --model-size 70b --seq-len 100000 --bits 3
```

Key outputs:
- **Compression ratio** - How much smaller the KV cache will be
- **Max context** - How many tokens fit in your VRAM
- **Memory saved** - GB reduction from FP16 to TurboQuant

## 6. Benchmark Different Settings

Find the best bit-width for your use case:

```bash
# Create sample data
python3 -c "import numpy as np; np.save('bench.npy', np.random.randn(1024, 1024, 128).astype('float32'))"

# Benchmark all bit-widths
turboquant benchmark bench.npy --bits 2,3,4 --trials 5
```

## 7. Interactive Chat (with GGUF models)

```bash
# Install llama-cpp-python first
pip install llama-cpp-python

# Chat with your quantized model
turboquant chat ./models/model.gguf --turboquant-bits 3 --context-length 4096
```

## Common Workflows

### Workflow A: Compress Your Own Weights

```bash
# 1. Export your model weights to numpy
python3 export_my_model.py  # Your script

# 2. Compress each layer
for layer in weights/*.npy; do
    turboquant compress "$layer" "compressed/$(basename $layer .npy).tq" --bits 3
done

# 3. Package for distribution
tar czvf my-model-turboquant.tar.gz compressed/
```

### Workflow B: Convert Existing Models

```bash
# Load and re-quantize a GGUF model
turboquant load model-fp16.gguf --output model-tq3.gguf --bits 3

# Or convert SafeTensors to TurboQuant GGUF
turboquant load model.safetensors --output model.gguf --bits 3 --format gguf
```

### Workflow C: Batch Processing

```bash
# Compress all .npy files in a directory
mkdir -p compressed
for f in *.npy; do
    echo "Compressing $f..."
    turboquant quick "$f" "compressed/${f%.npy}.tq"
done
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Compress file | `turboquant compress input.npy output.tq --bits 3` |
| Quick compress | `turboquant quick input.npy` |
| Decompress | `turboquant decompress input.tq output.npy` |
| View info | `turboquant info file.tq` |
| Benchmark | `turboquant benchmark data.npy --bits 3,4` |
| KV analysis | `turboquant kv-analyze --model-size 70b` |
| Download model | `turboquant download <model-id> --bits 3` |
| List models | `turboquant list-models` |
| Load model | `turboquant load model.gguf --info` |
| Chat | `turboquant chat model.gguf` |

## Bit-Width Guide

| Bits | Use Case | Compression | Quality |
|------|----------|-------------|---------|
| 2-bit | Extreme compression | 8x | Lower |
| 3-bit | Balanced (recommended) | 5x | Good |
| 4-bit | High quality | 4x | Excellent |

## Block Size Guide

| Size | Use Case |
|------|----------|
| 32 | Small tensors, fine-grained |
| 64 | Balanced |
| 128 | Large tensors, better compression (default) |

## Next Steps

1. **Read the full CLI docs**: [CLI.md](CLI.md)
2. **Explore the API**: See `demo.py` and `README.md`
3. **Integrate into your pipeline**: Use the Python API for programmatic access

## Python API Quick Example

```python
import numpy as np
from turboquant import TurboQuant, TurboQuantConfig

# Create data
data = np.random.randn(1000, 128).astype('float32')

# Configure
config = TurboQuantConfig(bit_width=3, block_size=128)
tq = TurboQuant(config)

# Quantize
quantized = tq.quantize(data)
reconstructed = tq.dequantize(quantized)

# Check quality
mse = np.mean((data - reconstructed) ** 2)
print(f"MSE: {mse:.6f}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found` | Run `pip install -e .` to install |
| `file not found` | Check path, use absolute paths if needed |
| `HF token required` | Use `--hf-token` or set `HF_TOKEN` env var |
| `ImportError: llama_cpp` | Run `pip install llama-cpp-python` |

## Support

- Check [CLI.md](CLI.md) for detailed command reference
- Read the algorithm documentation in [README.md](README.md)
- Run `python demo.py` for interactive examples
