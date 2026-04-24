from typing import Dict, Optional

import numpy as np


class QJL:
    """QJL: Quantized Johnson-Lindenstrauss Transform.

    Implements 1-bit quantization of the Johnson-Lindenstrauss transform
    for unbiased inner product estimation with zero memory overhead.

    Key properties:
    - Unbiased: E[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩
    - Zero overhead: No scale/zero-point storage needed
    - 1-bit per value compression

    Reference: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"
    """

    def __init__(
        self,
        block_size: int = 128,
        rotation_seed: int = 42,
        jl_dim: Optional[int] = None,
    ):
        self.block_size = block_size
        self.rotation_seed = rotation_seed
        self.jl_dim: int = jl_dim if jl_dim is not None else block_size  # Output dimension

        # Generate JL matrix
        self.jl_matrix = self._generate_jl_matrix()

    def _generate_jl_matrix(self) -> np.ndarray:
        """Generate random projection matrix for JL transform.

        Uses Gaussian random matrix scaled appropriately for JL lemma.

        Returns:
            Projection matrix of shape (jl_dim, block_size)
        """
        rng = np.random.default_rng(self.rotation_seed + 1000)
        projection = rng.standard_normal((self.jl_dim, self.block_size))

        # Scale for JL lemma preservation
        projection = projection / np.sqrt(self.jl_dim)

        return projection.astype(np.float32)

    def quantize(self, x: np.ndarray) -> Dict:
        """Quantize vectors using QJL (1-bit sign quantization).

        Args:
            x: Input vectors of shape (n_vectors, block_size)

        Returns:
            Dictionary with:
            - signs: Sign bits (+1 or -1)
            - jl_matrix: The JL projection matrix
        """
        # Apply JL transform
        x_projected = x @ self.jl_matrix.T

        signs_binary = np.greater_equal(x_projected, 0).astype(np.uint8)

        return {
            "signs": signs_binary,
            "jl_matrix": self.jl_matrix,
            "projected_shape": x_projected.shape,
        }

    def dequantize(self, quantized: Dict) -> np.ndarray:
        """Dequantize QJL representation using asymmetric estimator.

        Uses the unbiased estimator: E[⟨y, x⟩] = c · S^T · sign(Sx)
        where c is a scaling factor derived from JL properties.

        Args:
            quantized: Dictionary from quantize()

        Returns:
            Reconstructed vectors
        """
        signs_binary = quantized["signs"]
        jl_matrix = quantized["jl_matrix"]

        # Convert back to +1/-1
        signs = 2 * signs_binary.astype(np.float32) - 1

        # Asymmetric estimator for reconstruction
        # c = sqrt(π/2) / d where d is dimension
        c = np.sqrt(np.pi / 2) / self.jl_dim

        # Reconstruct: x̂ = c · S^T · signs
        x_reconstructed = c * (signs @ jl_matrix)

        return x_reconstructed

    def inner_product_estimate(self, y: np.ndarray, quantized_x: Dict) -> np.ndarray:
        """Compute unbiased inner product estimate ⟨y, x⟩.

        This is the primary use case for QJL - fast approximate inner products
        for attention mechanisms.

        Args:
            y: Query vector(s)
            quantized_x: Quantized key vector(s)

        Returns:
            Inner product estimates
        """
        signs_binary = quantized_x["signs"]
        jl_matrix = quantized_x["jl_matrix"]

        # Convert signs
        signs = 2 * signs_binary.astype(np.float32) - 1

        # Asymmetric estimator: ⟨y, x⟩ ≈ sqrt(π/2) · (Sy) · sign(Sx)
        # where Sy is the standard JL projection (not quantized)
        y_projected = y @ jl_matrix.T

        # Inner product in projected space
        inner_products = np.sqrt(np.pi / 2) * (y_projected * signs).sum(axis=-1)

        return inner_products

    def compress(self, signs: np.ndarray) -> bytes:
        """Compress sign bits to bytes using bit-packing."""
        from .utils import pack_bits

        return pack_bits(signs, 1)

    def decompress(self, data: bytes, n_bits: int) -> np.ndarray:
        """Decompress bytes back to sign bits."""
        from .utils import unpack_bits

        return unpack_bits(data, 1, n_bits)
