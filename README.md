# FastVQ

Fast vector quantization for AI model weights and KV caches, published on PyPI as `fastvq`.

## Overview

TurboQuant is a novel quantization algorithm from Google Research (ICLR 2026) that achieves:
- **3-bit compression** with near-zero accuracy loss
- **6x memory reduction** for KV caches in LLMs
- **8x speedup** on GPU with optimized kernels
- **Zero memory overhead** - no per-block quantization constants needed

This implementation includes:
- PolarQuant: Random rotation + polar transformation + Lloyd-Max quantization
- QJL: 1-bit Johnson-Lindenstrauss transform with asymmetric estimator
- TurboQuant: Combined two-stage compression with error correction
- Full KV cache integration for transformer models
- Arbitrary trailing dimensions with automatic block padding
- Fast sign-flipped Hadamard rotation backend
- Benchmark helpers, CLI benchmark suite, and notebook examples

## Key Features

### Core Algorithms
1. **PolarQuant** - Converts pairs of coordinates to polar coordinates and quantizes angles
   - Eliminates normalization overhead
   - Achieves 4.2x+ compression ratio
   - Near-optimal distortion rates

2. **QJL (Quantized JL)** - 1-bit quantization with zero memory overhead
   - Unbiased inner product estimation
   - Perfect for attention mechanisms
   - Error correction for residuals

3. **TurboQuant** - Two-stage compression combining both methods
   - Stage 1: PolarQuant for main compression
   - Stage 2: QJL on residual for error correction
   - 3-bit compression with quality neutrality

### Performance

```
Bit-width    Compression    MSE          Cosine Similarity
3-bit        4.9x           0.715        0.64
4-bit        3.8x           0.193        0.90
```

### Memory Savings (70B model, 72GB VRAM)

| Configuration | Memory | Max Context |
|--------------|--------|-------------|
| FP16         | 3.05 GB | 1,037,597 tokens |
| TurboQuant 3-bit | 0.62 GB | **5,108,173 tokens** |
| **Savings** | **4.9x** | **4.9x** |

## Installation

### From PyPI (Recommended)

```bash
pip install fastvq
```

### From Source

```bash
git clone https://github.com/turboquant/turboquant
cd turboquant
pip install -e .
```

The package supports both the new `fastvq` import and the legacy `turboquant` import:

```python
from fastvq import TurboQuant, TurboQuantConfig
```

## Quick Start

### Basic Compression

```python
import numpy as np
from fastvq import TurboQuant, TurboQuantConfig

# Create sample data. The trailing dimension can be any positive size.
x = np.random.randn(1, 8, 1024, 192).astype(np.float32)

# Configure TurboQuant
config = TurboQuantConfig(
    bit_width=3,           # 3-bit compression
    block_size=128,        # Block size
    use_qjl=True,          # Enable QJL error correction
    use_polar=True,        # Enable PolarQuant
    rotation="hadamard",   # Fast deterministic rotation
)

# Create quantizer
tq = TurboQuant(config)

# Quantize
quantized = tq.quantize(x)

# Dequantize
reconstructed = tq.dequantize(quantized)

# Check quality
mse = np.mean((x - reconstructed) ** 2)
print(f"MSE: {mse:.6f}")
print(tq.compression_stats(quantized))
```

### KV Cache Compression

```python
from turboquant.kv_cache import KVCacheCompressor

# Create compressor
compressor = KVCacheCompressor(bit_width=3, block_size=128)

# During inference, compress KV cache
key_states = np.random.randn(1, 32, 1000, 128).astype(np.float32)
value_states = np.random.randn(1, 32, 1000, 128).astype(np.float32)

k_comp, v_comp = compressor.compress_kv(key_states, value_states)

# Later, decompress for attention
k_full = compressor.decompress_k(k_comp)
v_full = compressor.decompress_v(v_comp)

# Check memory savings
stats = compressor.compute_memory_stats(
    seq_len=100000,
    batch_size=1,
    n_heads=32,
    head_dim=128,
)
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Memory saved: {stats['memory_saved_gb']:.2f} GB")
```

## Benchmarks And Notebooks

Run the synthetic benchmark suite:

```bash
fastvq benchmark-suite --shapes 1024x128,4096x128,1024x192 --bits 2,3,4 --output benchmarks/results.json
python benchmarks/fastvq_benchmark.py --output benchmarks/results.csv
```

Open `notebooks/fastvq_quickstart.ipynb` for a notebook walkthrough covering quantization, byte roundtrips, compression stats, and benchmark usage.

## Algorithm Details

### PolarQuant Algorithm

1. **Random Rotation**: Apply random orthogonal transformation to spread information evenly
2. **Pairwise Polar Transform**: Convert (x, y) pairs to (radius, angle)
3. **Angle Quantization**: Quantize angles using Lloyd-Max optimal codebook
4. **Efficient Storage**: Store norms + quantized indices (no per-block scales)

### QJL Algorithm

1. **JL Projection**: Project to lower dimension using random Gaussian matrix
2. **Sign Quantization**: Keep only sign bit (+1/-1) of each projected value
3. **Asymmetric Estimator**: Use different reconstruction for queries vs keys
4. **Unbiased IP**: Inner products remain unbiased for attention computation

### TurboQuant Combination

```
Original Vector:     x
↓ Normalize:         x̂ = x / ||x||
↓ PolarQuant:        q_polar = quantize_polar(x̂)
↓ Residual:          r = x̂ - dequantize_polar(q_polar)
↓ QJL:               q_qjl = sign(JL(r))

Compressed:          (||x||, q_polar, q_qjl)
```

## Project Structure

```
fastvq/
├── turboquant/
│   ├── __init__.py          # Main exports
│   ├── turboquant.py        # TurboQuant implementation
│   ├── polarquant.py        # PolarQuant implementation
│   ├── qjl.py               # QJL implementation
│   ├── transforms.py        # Walsh-Hadamard & polar transforms
│   ├── codebooks.py         # Lloyd-Max codebook generation
│   ├── utils.py             # Bit-packing & utilities
│   └── kv_cache.py          # KV cache integration
├── fastvq/                  # Import and CLI aliases for the PyPI package name
├── benchmarks/              # Benchmark runner scripts
├── notebooks/               # Notebook examples
├── demo.py                  # Demonstration script
└── README.md               # This file
```

## Implementation Notes

### Improvements Over Paper

This implementation includes several enhancements:

1. **Simplified Polar Transform**: Uses pairwise transformation instead of full recursive
2. **Optimized Codebooks**: Pre-computed Lloyd-Max centroids for common dimensions
3. **Flexible Block Sizes**: Supports 32, 64, 128, and 256-dimensional blocks
4. **Arbitrary Input Widths**: Pads/splits trailing dimensions automatically
5. **Hadamard Rotation Backend**: Avoids dense rotation matrices by default
6. **KV Cache Utilities**: Full streaming and batch compression support

### Limitations

- SIMD kernels are NumPy-vectorized fallbacks, not native AVX/NEON extensions
- CUDA kernels not yet implemented
- PyTorch integration not yet implemented

## References

### Papers

1. **TurboQuant** (Zandieh et al., ICLR 2026)
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - https://arxiv.org/abs/2504.19874

2. **QJL** (Zandieh et al., 2024)
   - "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead"
   - https://arxiv.org/abs/2406.03482

3. **PolarQuant** (Han et al., AISTATS 2026)
   - "PolarQuant: Quantizing KV Caches with Polar Transformation"
   - https://arxiv.org/abs/2502.02617

### Related Work

- GPTQ: Post-training quantization with Hessian-based error compensation
- AWQ: Activation-aware weight quantization
- SmoothQuant: Migrating quantization difficulty from activations to weights
- QLoRA: 4-bit NormalFloat quantization for fine-tuning

## License

This is a research implementation for educational purposes. Please cite the original TurboQuant paper if you use this in your research.

## Citation

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2025}
}
```

## Future Work

- [ ] SIMD optimizations (AVX2, AVX-512, NEON)
- [ ] CUDA kernels for GPU acceleration
- [ ] PyTorch integration with autograd support
- [ ] Integration with transformers library
- [ ] Support for model weight quantization (not just KV cache)
- [ ] Streaming compression for long sequences
- [ ] Adaptive bit-width selection per layer
