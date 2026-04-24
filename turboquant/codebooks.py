from contextlib import suppress
from typing import Dict

import numpy as np

# Pre-computed Lloyd-Max codebooks for different bit-widths and dimensions
# These are optimal quantization centroids for Beta-distributed coordinates
# after random rotation in TurboQuant

_LLOYD_MAX_CODEBOOKS: Dict[tuple, np.ndarray] = {}


def _generate_lloyd_max_centroids(bit_width: int, dim: int, max_iter: int = 100) -> np.ndarray:
    """Generate optimal Lloyd-Max quantization centroids.

    Lloyd-Max algorithm iteratively optimizes quantization centroids:
    1. Initialize centroids uniformly
    2. Assign samples to nearest centroid
    3. Update centroids to mean of assigned samples
    4. Repeat until convergence

    For TurboQuant, coordinates follow Beta distribution after random rotation.

    Args:
        bit_width: Number of bits (1-8)
        dim: Dimension of vectors
        max_iter: Maximum iterations for Lloyd-Max

    Returns:
        Optimal centroids array of shape (n_levels,)
    """
    n_levels = 2**bit_width

    # For Beta-distributed data in high dimensions
    # The distribution is approximately N(0, 1/dim)
    # Generate samples from this distribution
    np.random.seed(42)
    n_samples = 10000

    # Beta(α, β) where α = β = (d-1)/2 for uniform sphere
    # In high dimensions, this approaches Gaussian
    alpha = beta = (dim - 1) / 2

    # Generate samples from Beta distribution (scaled to [-1, 1])
    samples = np.random.beta(alpha, beta, n_samples) * 2 - 1

    # Initialize centroids uniformly
    centroids = np.linspace(-1, 1, n_levels)

    # Lloyd-Max iterations
    for _iteration in range(max_iter):
        # Assign samples to nearest centroid
        distances = np.abs(samples[:, np.newaxis] - centroids)
        assignments = np.argmin(distances, axis=1)

        # Update centroids to mean of assigned samples
        new_centroids = np.array(
            [
                samples[assignments == i].mean() if (assignments == i).any() else centroids[i]
                for i in range(n_levels)
            ]
        )

        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=1e-5):
            break

        centroids = new_centroids

    return centroids.astype(np.float32)


def get_lloyd_max_codebook(bit_width: int, dim: int = 128) -> np.ndarray:
    """Get pre-computed Lloyd-Max codebook for given bit-width and dimension.

    Args:
        bit_width: Number of bits (1-8)
        dim: Vector dimension (typically 128 for attention heads)

    Returns:
        Codebook array of shape (2^bit_width,)
    """
    key = (bit_width, dim)

    if key not in _LLOYD_MAX_CODEBOOKS:
        # Generate if not cached
        _LLOYD_MAX_CODEBOOKS[key] = _generate_lloyd_max_centroids(bit_width, dim)

    return _LLOYD_MAX_CODEBOOKS[key]


def quantize_with_codebook(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Quantize values using a codebook.

    Args:
        x: Input values to quantize
        codebook: Quantization centroids

    Returns:
        Indices into codebook
    """
    # Find nearest centroid for each value
    distances = np.abs(x[..., np.newaxis] - codebook)
    indices = np.argmin(distances, axis=-1)
    return indices


def dequantize_with_codebook(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Dequantize indices back to values using codebook.

    Args:
        indices: Quantized indices
        codebook: Quantization centroids

    Returns:
        Reconstructed values
    """
    return codebook[indices]


def generate_codebook(bit_width: int, dim: int = 128, distribution: str = "beta") -> np.ndarray:
    """Generate a custom quantization codebook.

    Args:
        bit_width: Number of bits
        dim: Dimension
        distribution: Target distribution ("beta", "gaussian", "uniform")

    Returns:
        Codebook array
    """
    n_levels = 2**bit_width

    if distribution == "beta":
        return _generate_lloyd_max_centroids(bit_width, dim)
    elif distribution == "gaussian":
        # For Gaussian, use quantiles of normal distribution
        from scipy.stats import norm

        quantiles = np.linspace(0, 1, n_levels + 2)[1:-1]
        return norm.ppf(quantiles).astype(np.float32)
    elif distribution == "uniform":
        return np.linspace(-1, 1, n_levels).astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# Pre-generate common codebooks
for bw in [1, 2, 3, 4, 5, 6, 7, 8]:
    for d in [32, 64, 128, 256]:
        with suppress(Exception):
            _ = get_lloyd_max_codebook(bw, d)
