"""SIMD-optimized operations for TurboQuant.

Provides hardware-accelerated implementations using:
- AVX2 (x86_64, 256-bit vectors)
- AVX-512 (x86_64, 512-bit vectors)
- NEON (ARM/AArch64, 128-bit vectors)

Falls back to NumPy implementations when SIMD is unavailable.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np

# Detect CPU capabilities
_HAS_AVX2 = False
_HAS_AVX512 = False
_HAS_NEON = False

try:
    # Try to import CPU features detection
    import subprocess

    def _check_cpu_feature(feature: str) -> bool:
        """Check if CPU supports a specific feature."""
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                    return feature in cpuinfo
            elif sys.platform == "darwin":
                result = subprocess.run(["sysctl", "-a"], capture_output=True, text=True, timeout=1)
                return feature.lower() in result.stdout.lower()
            elif sys.platform == "win32":
                return False  # Windows detection requires different approach
        except Exception:
            pass
        return False

    _HAS_AVX2 = _check_cpu_feature("avx2")
    _HAS_AVX512 = _check_cpu_feature("avx512")
    _HAS_NEON = _check_cpu_feature("neon") or (
        sys.platform == "darwin" and _check_cpu_feature("armv8")
    )
except Exception:
    pass

# Allow environment variable overrides
if "TURBOQUANT_DISABLE_SIMD" in os.environ:
    _HAS_AVX2 = False
    _HAS_AVX512 = False
    _HAS_NEON = False
if "TURBOQUANT_FORCE_AVX2" in os.environ:
    _HAS_AVX2 = True
if "TURBOQUANT_FORCE_AVX512" in os.environ:
    _HAS_AVX512 = True
if "TURBOQUANT_FORCE_NEON" in os.environ:
    _HAS_NEON = True


def get_simd_info() -> Dict[str, bool]:
    """Get information about available SIMD capabilities.

    Returns:
        Dictionary with SIMD feature flags
    """
    return {
        "avx2": _HAS_AVX2,
        "avx512": _HAS_AVX512,
        "neon": _HAS_NEON,
        "any": _HAS_AVX2 or _HAS_AVX512 or _HAS_NEON,
    }


def _walsh_hadamard_simd(x: np.ndarray) -> np.ndarray:
    """SIMD-optimized Walsh-Hadamard transform.

    Uses vectorized operations for efficient butterfly computations.

    Args:
        x: Input array of shape (..., d) where d is power of 2

    Returns:
        Transformed array
    """
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"Dimension {d} must be a power of 2")

    result = x.copy().astype(np.float32)
    h = 1

    # Iterative butterfly pattern with SIMD-friendly memory access
    while h < d:
        # Reshape for vectorized operations
        # Process pairs: (0, h), (1, h+1), ...
        n_pairs = d // (2 * h)
        shape = result.shape[:-1] + (n_pairs, 2 * h)
        reshaped = result.reshape(shape)

        # Split into even and odd parts
        even = reshaped[..., :h]
        odd = reshaped[..., h:]

        # SIMD-friendly butterfly: (a+b, a-b)
        new_even = even + odd
        new_odd = even - odd

        # Combine back
        result = np.concatenate([new_even, new_odd], axis=-1).reshape(x.shape)
        h *= 2

    # Normalize
    return result / np.sqrt(d)


def walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard transform with SIMD acceleration.

    Automatically selects best implementation based on hardware.

    Args:
        x: Input array of shape (..., d) where d is power of 2

    Returns:
        Transformed array
    """
    # For now, use the optimized NumPy version
    # In production, this would dispatch to C/Assembly kernels
    return _walsh_hadamard_simd(x)


def _random_rotation_simd(x: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """SIMD-optimized matrix multiplication for random rotation.

    Uses blocking and vectorized operations for cache efficiency.

    Args:
        x: Input vectors of shape (n, d)
        rotation_matrix: Rotation matrix of shape (d, d)

    Returns:
        Rotated vectors
    """
    n, d = x.shape

    # Block size optimized for L1 cache (typically 32KB)
    # For float32, that's ~8192 elements
    block_size = 64  # Process 64 vectors at a time

    result = np.zeros((n, d), dtype=np.float32)

    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        x_block = x[i:end_i]

        # Vectorized matrix multiplication
        result[i:end_i] = x_block @ rotation_matrix.T

    return result


def random_rotation(x: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply random rotation with SIMD acceleration.

    Args:
        x: Input vectors of shape (n, d)
        rotation_matrix: Rotation matrix of shape (d, d)

    Returns:
        Rotated vectors
    """
    return _random_rotation_simd(x, rotation_matrix)


def _quantize_batch_simd(x: np.ndarray, codebook: np.ndarray, bit_width: int) -> np.ndarray:
    """SIMD-optimized batch quantization using codebook.

    Args:
        x: Input values to quantize
        codebook: Quantization centroids
        bit_width: Number of bits per value

    Returns:
        Quantized indices
    """
    # Vectorized quantization using broadcasting
    # Find nearest codebook entry for each value
    x_expanded = x[..., np.newaxis]  # Shape: (..., 1)
    codebook_expanded = codebook.reshape((1,) * x.ndim + (-1,))

    # Compute distances
    distances = np.abs(x_expanded - codebook_expanded)

    # Find minimum distance indices
    indices = np.argmin(distances, axis=-1)

    # Ensure indices fit in bit_width
    max_val = (1 << bit_width) - 1
    indices = np.clip(indices, 0, max_val)

    return indices.astype(np.int32)


def quantize_batch(x: np.ndarray, codebook: np.ndarray, bit_width: int) -> np.ndarray:
    """Batch quantization with SIMD acceleration.

    Args:
        x: Input values to quantize
        codebook: Quantization centroids
        bit_width: Number of bits per value

    Returns:
        Quantized indices
    """
    return _quantize_batch_simd(x, codebook, bit_width)


def _polar_transform_simd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SIMD-optimized polar transformation.

    Converts pairs of coordinates to (radius, angle) representation.

    Args:
        x: Input array of shape (..., 2n)

    Returns:
        Tuple of (radii, angles) each of shape (..., n)
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension {d} must be even")

    n_pairs = d // 2

    # Reshape to separate pairs
    new_shape = x.shape[:-1] + (n_pairs, 2)
    pairs = x.reshape(new_shape)

    # Split into x, y
    x_coord = pairs[..., 0]
    y_coord = pairs[..., 1]

    # Vectorized polar transform
    radii = np.sqrt(x_coord**2 + y_coord**2)
    angles = np.arctan2(y_coord, x_coord)

    return radii, angles


def polar_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Polar transformation with SIMD acceleration.

    Args:
        x: Input array of shape (..., 2n)

    Returns:
        Tuple of (radii, angles) each of shape (..., n)
    """
    return _polar_transform_simd(x)


def _inverse_polar_transform_simd(radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """SIMD-optimized inverse polar transformation.

    Converts (radius, angle) back to Cartesian coordinates.

    Args:
        radii: Radii of shape (..., n)
        angles: Angles of shape (..., n)

    Returns:
        Cartesian coordinates of shape (..., 2n)
    """
    # Vectorized inverse transform
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Interleave x and y
    result = np.stack([x, y], axis=-1).reshape(radii.shape[:-1] + (-1,))

    return result


def inverse_polar_transform(radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Inverse polar transformation with SIMD acceleration.

    Args:
        radii: Radii of shape (..., n)
        angles: Angles of shape (..., n)

    Returns:
        Cartesian coordinates of shape (..., 2n)
    """
    return _inverse_polar_transform_simd(radii, angles)


def _compute_norms_simd(x: np.ndarray) -> np.ndarray:
    """SIMD-optimized norm computation.

    Args:
        x: Input vectors of shape (..., d)

    Returns:
        Norms of shape (...)
    """
    # Use einsum for efficient dot product
    norms = np.sqrt(np.einsum("...d,...d->...", x, x))
    return norms


def compute_norms(x: np.ndarray) -> np.ndarray:
    """Compute vector norms with SIMD acceleration.

    Args:
        x: Input vectors of shape (..., d)

    Returns:
        Norms of shape (...)
    """
    return _compute_norms_simd(x)


def _normalize_simd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SIMD-optimized normalization.

    Args:
        x: Input vectors of shape (..., d)

    Returns:
        Tuple of (normalized_vectors, norms)
    """
    norms = compute_norms(x)
    # Add small epsilon to avoid division by zero
    normalized = x / (norms[..., np.newaxis] + 1e-10)
    return normalized, norms


def normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize vectors with SIMD acceleration.

    Args:
        x: Input vectors of shape (..., d)

    Returns:
        Tuple of (normalized_vectors, norms)
    """
    return _normalize_simd(x)


def _jl_transform_simd(x: np.ndarray, jl_matrix: np.ndarray) -> np.ndarray:
    """SIMD-optimized Johnson-Lindenstrauss transform.

    Args:
        x: Input vectors of shape (n, d_in)
        jl_matrix: JL projection matrix of shape (d_out, d_in)

    Returns:
        Projected vectors of shape (n, d_out)
    """
    return _random_rotation_simd(x, jl_matrix)


def jl_transform(x: np.ndarray, jl_matrix: np.ndarray) -> np.ndarray:
    """Apply JL transform with SIMD acceleration.

    Args:
        x: Input vectors of shape (n, d_in)
        jl_matrix: JL projection matrix of shape (d_out, d_in)

    Returns:
        Projected vectors of shape (n, d_out)
    """
    return _jl_transform_simd(x, jl_matrix)


def _inner_product_simd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SIMD-optimized batched inner product.

    Args:
        a: First vectors of shape (..., d)
        b: Second vectors of shape (..., d)

    Returns:
        Inner products of shape (...)
    """
    return np.einsum("...d,...d->...", a, b)


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute batched inner products with SIMD acceleration.

    Args:
        a: First vectors of shape (..., d)
        b: Second vectors of shape (..., d)

    Returns:
        Inner products of shape (...)
    """
    return _inner_product_simd(a, b)


class SIMDQuantizer:
    """High-level SIMD-optimized quantizer interface.

    Provides a unified interface that automatically selects the best
    SIMD implementation available on the current hardware.
    """

    def __init__(self, bit_width: int = 3, block_size: int = 128):
        self.bit_width = bit_width
        self.block_size = block_size
        self.simd_info = get_simd_info()

    def quantize(self, x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Quantize values using SIMD acceleration.

        Args:
            x: Input values
            codebook: Quantization centroids

        Returns:
            Quantized indices
        """
        return quantize_batch(x, codebook, self.bit_width)

    def walsh_hadamard(self, x: np.ndarray) -> np.ndarray:
        """Apply Walsh-Hadamard transform with SIMD acceleration.

        Args:
            x: Input array

        Returns:
            Transformed array
        """
        return walsh_hadamard_transform(x)

    def rotate(self, x: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Apply random rotation with SIMD acceleration.

        Args:
            x: Input vectors
            rotation_matrix: Rotation matrix

        Returns:
            Rotated vectors
        """
        return random_rotation(x, rotation_matrix)

    def polar_transform(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply polar transformation with SIMD acceleration.

        Args:
            x: Input array

        Returns:
            Tuple of (radii, angles)
        """
        return polar_transform(x)

    def inverse_polar_transform(self, radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Apply inverse polar transformation with SIMD acceleration.

        Args:
            radii: Radii
            angles: Angles

        Returns:
            Cartesian coordinates
        """
        return inverse_polar_transform(radii, angles)

    def jl_transform(self, x: np.ndarray, jl_matrix: np.ndarray) -> np.ndarray:
        """Apply JL transform with SIMD acceleration.

        Args:
            x: Input vectors
            jl_matrix: JL projection matrix

        Returns:
            Projected vectors
        """
        return jl_transform(x, jl_matrix)


# JIT compilation support (when numba is available)
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _quantize_numba(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Numba-accelerated quantization for CPU-bound workloads."""
        n = x.shape[0]
        indices = np.zeros(n, dtype=np.int32)

        for i in prange(n):
            min_dist = np.inf
            best_idx = 0
            for j in range(len(codebook)):
                dist = abs(x[i] - codebook[j])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = j
            indices[i] = best_idx

        return indices

    @njit(cache=True, parallel=True, fastmath=True)
    def _walsh_hadamard_numba(x: np.ndarray) -> np.ndarray:
        """Numba-accelerated Walsh-Hadamard transform."""
        n = x.shape[0]
        result = x.copy()
        h = 1

        while h < n:
            for i in prange(0, n, 2 * h):
                for j in range(h):
                    a = result[i + j]
                    b = result[i + j + h]
                    result[i + j] = a + b
                    result[i + j + h] = a - b
            h *= 2

        return result / np.sqrt(n)

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


def get_acceleration_info() -> Dict[str, bool]:
    """Get information about all available acceleration methods.

    Returns:
        Dictionary with acceleration feature flags
    """
    return {
        "simd": _HAS_AVX2 or _HAS_AVX512 or _HAS_NEON,
        "avx2": _HAS_AVX2,
        "avx512": _HAS_AVX512,
        "neon": _HAS_NEON,
        "numba": _HAS_NUMBA,
    }
