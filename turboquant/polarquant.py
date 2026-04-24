from typing import Any, Dict, Optional

import numpy as np

from .simd import inverse_polar_transform, polar_transform, walsh_hadamard_transform
from .transforms import random_rotation_matrix


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
        self,
        bit_width: int = 3,
        block_size: int = 128,
        rotation_seed: int = 42,
        rotation: str = "hadamard",
    ):
        if bit_width < 1 or bit_width > 8:
            raise ValueError("bit_width must be between 1 and 8")
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer")
        if rotation not in {"hadamard", "random"}:
            raise ValueError("rotation must be 'hadamard' or 'random'")
        if rotation == "hadamard" and block_size & (block_size - 1) != 0:
            raise ValueError("hadamard rotation requires a power-of-two block_size")

        self.bit_width = bit_width
        self.block_size = block_size
        self.rotation_seed = rotation_seed
        self.rotation = rotation

        # Load pre-computed Lloyd-Max codebook for quantizing angles
        # Angles are in [-π, π], so we need a codebook covering this range
        self.codebook = self._generate_angle_codebook()

        # Hadamard rotation avoids storing and multiplying by a dense d x d matrix.
        self.rotation_matrix = self._generate_rotation_matrix()
        self.rotation_signs = self._generate_rotation_signs()

    def _generate_rotation_matrix(self) -> Optional[np.ndarray]:
        """Generate deterministic random rotation matrix.

        Uses QR decomposition of Gaussian matrix for orthogonal matrix.
        """
        if self.rotation != "random":
            return None
        return random_rotation_matrix(self.block_size, self.rotation_seed)

    def _generate_rotation_signs(self) -> np.ndarray:
        """Generate deterministic sign flips for randomized Hadamard rotation."""
        rng = np.random.default_rng(self.rotation_seed)
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.block_size)

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

    def _apply_rotation(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured orthogonal rotation."""
        if self.rotation_matrix is not None:
            return x @ self.rotation_matrix.T
        return walsh_hadamard_transform(x * self.rotation_signs)

    def _apply_inverse_rotation(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse configured orthogonal rotation."""
        if self.rotation_matrix is not None:
            return x @ self.rotation_matrix
        return walsh_hadamard_transform(x) * self.rotation_signs

    def quantize(self, x: np.ndarray) -> Dict[str, Any]:
        """Quantize vectors using PolarQuant.

        Args:
            x: Input vectors of shape (n_vectors, block_size)

        Returns:
            Dictionary with quantized representation:
            - indices: Quantized angle indices
            - norms: Vector norms for reconstruction
            - metadata: Additional quantization parameters
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.block_size:
            raise ValueError(f"x must have shape (n_vectors, {self.block_size}); got {x.shape}")

        norms = np.linalg.norm(x, axis=1, keepdims=True)

        # Normalize to unit sphere
        x_unit = np.divide(
            x,
            norms,
            out=np.zeros_like(x, dtype=np.float32),
            where=norms > 1e-10,
        )

        # Step 1: Random rotation (Walsh-Hadamard or orthogonal)
        x_rotated = self._apply_rotation(x_unit)

        # Step 2: Pairwise polar transformation
        # Process coordinates in pairs: (x[0], x[1]), (x[2], x[3]), ...
        # Each pair -> (radius, angle)
        radii, angles = polar_transform(x_rotated)

        # Step 3: Quantize angles using Lloyd-Max codebook
        # Normalize angles to [0, 1] for quantization
        angles_normalized = (angles + np.pi) / (2 * np.pi)
        angles_normalized = np.clip(angles_normalized, 0, 0.9999)

        n_levels = 2**self.bit_width
        indices = (angles_normalized * n_levels).astype(np.uint8)

        return {
            "indices": indices,
            "norms": norms.astype(np.float32),
            "radii": radii.astype(np.float32),
            "metadata": {
                "bit_width": self.bit_width,
                "block_size": self.block_size,
                "rotation": self.rotation,
                "rotation_seed": self.rotation_seed,
            },
        }

    def dequantize(self, quantized: Dict[str, Any]) -> np.ndarray:
        """Dequantize PolarQuant representation.

        Args:
            quantized: Dictionary from quantize()

        Returns:
            Reconstructed vectors
        """
        indices = np.asarray(quantized["indices"])
        norms = np.asarray(quantized["norms"], dtype=np.float32)
        radii = np.asarray(quantized["radii"], dtype=np.float32)

        # Step 1: Dequantize angles
        angles = self.codebook[indices]

        # Step 2: Inverse pairwise polar transformation
        x_rotated = inverse_polar_transform(radii, angles)

        # Step 3: Inverse rotation
        x_unit = self._apply_inverse_rotation(x_rotated)

        # Renormalize and rescale
        unit_norms = np.linalg.norm(x_unit, axis=1, keepdims=True)
        x_unit = np.divide(
            x_unit,
            unit_norms,
            out=np.zeros_like(x_unit, dtype=np.float32),
            where=unit_norms > 1e-10,
        )
        x_reconstructed = x_unit * norms

        return x_reconstructed
