"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

A production-quality implementation of Google's TurboQuant algorithm for extreme
compression of AI models (KV cache and weights).

Key Features:
- 3-bit compression with zero accuracy loss (TQ3)
- 4-bit compression for higher quality (TQ4)
- PolarQuant: Random rotation + polar transformation + Lloyd-Max quantization
- QJL: 1-bit Johnson-Lindenstrauss transform with asymmetric estimator
- SIMD optimizations (AVX2, AVX-512, NEON)
- CUDA kernels for GPU acceleration
- PyTorch integration

Based on:
- TurboQuant paper (Zandieh et al., 2025): https://arxiv.org/abs/2504.19874
- QJL paper (Zandieh et al., 2024): https://arxiv.org/abs/2406.03482
- PolarQuant paper (Han et al., 2025): https://arxiv.org/abs/2502.02617
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"
__author__ = "TurboQuant Contributors"

from .turboquant import TurboQuant, TurboQuantConfig
from .polarquant import PolarQuant
from .qjl import QJL
from .codebooks import get_lloyd_max_codebook, generate_codebook
from .utils import quantize_decompress_benchmark, compute_mse, compute_distortion
from .model_export import (
    export_to_gguf,
    export_to_safetensors,
    load_gguf,
    load_safetensors,
    load_gguf_model,
    load_safetensors_model,
    load_model,
    quantize_model_weights,
    TurboQuantGGUFLoader,
    TurboQuantSafeTensorsLoader,
)

__all__ = [
    "TurboQuant",
    "TurboQuantConfig",
    "PolarQuant",
    "QJL",
    "get_lloyd_max_codebook",
    "generate_codebook",
    "quantize_decompress_benchmark",
    "compute_mse",
    "compute_distortion",
    "export_to_gguf",
    "export_to_safetensors",
    "load_gguf",
    "load_safetensors",
    "load_gguf_model",
    "load_safetensors_model",
    "load_model",
    "quantize_model_weights",
    "TurboQuantGGUFLoader",
    "TurboQuantSafeTensorsLoader",
]
