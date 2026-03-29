import numpy as np
from typing import Dict
from .transforms import walsh_hadamard_transform, random_rotation_matrix
from .codebooks import (
    get_lloyd_max_codebook,
    quantize_with_codebook,
    dequantize_with_codebook,
)


class PolarQuant:
    """PolarQuant: Quantizing with Polar Transformation.

    Implements the PolarQuant algorithm from the paper which converts
    pairs of coordinates into polar coordinates (radius + angle) and
    quantizes the angles.

    Key insight: After random rotation, coordinates follow a Beta distribution,
    and the angles in polar representation have a tightly bounded,
    analytically computable distribution that eliminates the need for
    explicit normalization and quantization constants.
    """

    def __init__(
        self, bit_width: int = 3, block_size: int = 128, rotation_seed: int = 42
    ):
        self.bit_width = bit_width
        self.block_size = block_size
        self.rotation_seed = rotation_seed

        # Load pre-computed Lloyd-Max codebook for quantizing angles
        # Angles are in [-π, π], so we need a codebook covering this range
        self.codebook = self._generate_angle_codebook()

        # Initialize rotation matrix (Walsh-Hadamard or random orthogonal)
        self.rotation_matrix = self._generate_rotation_matrix()

    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate deterministic random rotation matrix.

        Uses QR decomposition of Gaussian matrix for orthogonal matrix.
        """
        return random_rotation_matrix(self.block_size, self.rotation_seed)

    def _generate_angle_codebook(self) -> np.ndarray:
        """Generate Lloyd-Max codebook for angle quantization.

        Angles are in range [-π, π], and after random rotation,
        they follow a specific distribution that allows efficient quantization.
        """
        n_levels = 2**self.bit_width
        # For simplicity, use uniform quantization over [-π, π]
        # The paper uses optimal Lloyd-Max for the Beta-derived distribution
        angles = np.linspace(-np.pi, np.pi, n_levels + 1)
        # Use midpoints as centroids
        centroids = (angles[:-1] + angles[1:]) / 2
        return centroids.astype(np.float32)

    def quantize(self, x: np.ndarray) -> Dict:
        """Quantize vectors using PolarQuant.

        Args:
            x: Input vectors of shape (n_vectors, block_size)

        Returns:
            Dictionary with quantized representation:
            - indices: Quantized angle indices
            - norms: Vector norms for reconstruction
            - metadata: Additional quantization parameters
        """
        # Store original norms
        norms = np.linalg.norm(x, axis=1, keepdims=True)

        # Normalize to unit sphere
        x_unit = x / (norms + 1e-10)

        # Step 1: Random rotation (Walsh-Hadamard or orthogonal)
        x_rotated = x_unit @ self.rotation_matrix.T

        # Step 2: Pairwise polar transformation
        # Process coordinates in pairs: (x[0], x[1]), (x[2], x[3]), ...
        # Each pair -> (radius, angle)
        n_pairs = self.block_size // 2

        radii = np.zeros((x.shape[0], n_pairs), dtype=np.float32)
        angles = np.zeros((x.shape[0], n_pairs), dtype=np.float32)

        for i in range(n_pairs):
            x1 = x_rotated[:, 2 * i]
            x2 = x_rotated[:, 2 * i + 1]

            # Polar transformation: (x, y) -> (r, θ)
            radii[:, i] = np.sqrt(x1**2 + x2**2)
            angles[:, i] = np.arctan2(x2, x1)

        # Step 3: Quantize angles using Lloyd-Max codebook
        # Normalize angles to [0, 1] for quantization
        angles_normalized = (angles + np.pi) / (2 * np.pi)
        angles_normalized = np.clip(angles_normalized, 0, 0.9999)

        n_levels = 2**self.bit_width
        indices = (angles_normalized * n_levels).astype(np.int32)

        return {
            "indices": indices,
            "norms": norms.astype(np.float32),
            "radii": radii.astype(np.float32),
            "metadata": {
                "rotation_matrix": self.rotation_matrix,
                "codebook": self.codebook,
            },
        }

    def dequantize(self, quantized: Dict) -> np.ndarray:
        """Dequantize PolarQuant representation.

        Args:
            quantized: Dictionary from quantize()

        Returns:
            Reconstructed vectors
        """
        indices = quantized["indices"]
        norms = quantized["norms"]
        radii = quantized["radii"]

        # Step 1: Dequantize angles
        n_levels = 2**self.bit_width
        angles_normalized = indices / n_levels
        angles = angles_normalized * 2 * np.pi - np.pi

        # Step 2: Inverse pairwise polar transformation
        n_pairs = self.block_size // 2
        x_rotated = np.zeros((indices.shape[0], self.block_size), dtype=np.float32)

        for i in range(n_pairs):
            r = radii[:, i]
            theta = angles[:, i]

            # Inverse: (r, θ) -> (r*cos(θ), r*sin(θ))
            x_rotated[:, 2 * i] = r * np.cos(theta)
            x_rotated[:, 2 * i + 1] = r * np.sin(theta)

        # Step 3: Inverse rotation
        x_unit = x_rotated @ self.rotation_matrix

        # Renormalize and rescale
        x_unit = x_unit / (np.linalg.norm(x_unit, axis=1, keepdims=True) + 1e-10)
        x_reconstructed = x_unit * norms

        return x_reconstructed
