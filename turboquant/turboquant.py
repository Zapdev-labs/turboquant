from typing import Optional, Tuple, List, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant quantization.

    Args:
        bit_width: Number of bits per value (2, 3, or 4)
        block_size: Size of quantization blocks (32, 64, or 128)
        use_qjl: Whether to use QJL error correction on residual
        use_polar: Whether to use PolarQuant transformation
        rotation_seed: Random seed for rotation matrix generation
        dtype: Data type for computations (float32 or float16)
    """

    bit_width: int = 3
    block_size: int = 128
    use_qjl: bool = True
    use_polar: bool = True
    rotation_seed: int = 42
    dtype: str = "float32"

    def __post_init__(self):
        assert self.bit_width in [2, 3, 4], "bit_width must be 2, 3, or 4"
        assert self.block_size in [32, 64, 128], "block_size must be 32, 64, or 128"


class TurboQuant:
    """TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

    Implements the full TurboQuant algorithm combining:
    1. PolarQuant: Random rotation + polar transformation + Lloyd-Max quantization
    2. QJL: 1-bit quantization on residual for error correction

    Achieves 3-bit compression with zero accuracy loss and near-optimal distortion.
    """

    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self._init_components()

    def _init_components(self):
        """Initialize PolarQuant and QJL components."""
        from .polarquant import PolarQuant
        from .qjl import QJL

        self.polar = PolarQuant(
            bit_width=self.config.bit_width - 1
            if self.config.use_qjl
            else self.config.bit_width,
            block_size=self.config.block_size,
            rotation_seed=self.config.rotation_seed,
        )

        if self.config.use_qjl:
            self.qjl = QJL(
                block_size=self.config.block_size,
                rotation_seed=self.config.rotation_seed + 1,
            )
        else:
            self.qjl = None

    def quantize(self, x: np.ndarray) -> dict:
        """Quantize input vectors using TurboQuant.

        Args:
            x: Input array of shape (..., d) where d is the vector dimension

        Returns:
            Dictionary containing quantized representation:
            - norms: Vector norms (float32)
            - polar_indices: PolarQuant quantized indices
            - qjl_signs: QJL sign bits (optional)
            - metadata: Quantization parameters
        """
        original_shape = x.shape
        d = original_shape[-1]

        # Reshape to blocks
        n_blocks = int(np.prod(original_shape[:-1]))
        x_blocks = x.reshape(n_blocks, d)

        # Compute norms
        norms = np.linalg.norm(x_blocks, axis=1, keepdims=True)

        # Normalize
        x_unit = x_blocks / (norms + 1e-10)

        # Stage 1: PolarQuant compression
        polar_result = self.polar.quantize(x_unit)

        result = {
            "norms": norms.astype(np.float32),
            "polar_indices": polar_result["indices"],
            "polar_radii": polar_result["radii"],  # Include radii for reconstruction
            "polar_metadata": polar_result["metadata"],
            "original_shape": original_shape,
            "config": self.config,
        }

        # Stage 2: QJL on residual (if enabled)
        if self.config.use_qjl and self.qjl is not None:
            # Compute residual
            x_reconstructed = self.polar.dequantize(polar_result)
            residual = x_unit - x_reconstructed

            # Quantize residual with QJL
            qjl_result = self.qjl.quantize(residual)
            result["qjl_signs"] = qjl_result["signs"]
            result["qjl_matrix"] = qjl_result["jl_matrix"]

        return result

    def dequantize(self, quantized: dict) -> np.ndarray:
        """Dequantize TurboQuant representation back to vectors.

        Args:
            quantized: Dictionary from quantize()

        Returns:
            Reconstructed array with original shape
        """
        norms = quantized["norms"]
        polar_indices = quantized["polar_indices"]
        original_shape = quantized["original_shape"]

        # Stage 1: Dequantize PolarQuant
        polar_result = {
            "indices": polar_indices,
            "radii": quantized["polar_radii"],
            "metadata": quantized["polar_metadata"],
            "norms": norms,
        }
        x_unit = self.polar.dequantize(polar_result)

        # Stage 2: Add QJL residual (if available)
        if "qjl_signs" in quantized and self.qjl is not None:
            qjl_result = {
                "signs": quantized["qjl_signs"],
                "jl_matrix": quantized["qjl_matrix"],
            }
            residual = self.qjl.dequantize(qjl_result)
            x_unit = x_unit + residual

        # Renormalize and rescale
        x_unit = x_unit / (np.linalg.norm(x_unit, axis=1, keepdims=True) + 1e-10)
        x_reconstructed = x_unit * norms

        return x_reconstructed.reshape(original_shape)

    def compress(self, x: np.ndarray) -> bytes:
        """Compress array to bytes (full pipeline)."""
        quantized = self.quantize(x)
        return self._pack_quantized(quantized)

    def decompress(self, data: bytes) -> np.ndarray:
        """Decompress bytes back to array."""
        quantized = self._unpack_quantized(data)
        return self.dequantize(quantized)

    def _pack_quantized(self, quantized: dict) -> bytes:
        """Pack quantized representation into bytes."""
        import struct

        # Pack metadata
        config = quantized["config"]
        metadata = struct.pack(
            "<IIIII",
            config.bit_width,
            config.block_size,
            quantized["original_shape"][0],
            quantized["original_shape"][1]
            if len(quantized["original_shape"]) > 1
            else 1,
            1 if "qjl_signs" in quantized else 0,
        )

        # Pack norms
        norms_bytes = quantized["norms"].tobytes()

        # Pack polar indices (bit-packed)
        from .utils import pack_bits

        polar_bytes = pack_bits(
            quantized["polar_indices"],
            config.bit_width - 1 if config.use_qjl else config.bit_width,
        )

        # Pack QJL signs (if present)
        qjl_bytes = b""
        if "qjl_signs" in quantized:
            qjl_bytes = pack_bits(quantized["qjl_signs"], 1)

        # Combine
        header = struct.pack("<I", len(metadata)) + metadata
        return header + norms_bytes + polar_bytes + qjl_bytes

    def _unpack_quantized(self, data: bytes) -> dict:
        """Unpack bytes into quantized representation."""
        import struct
        from .utils import unpack_bits

        # Unpack header
        metadata_len = struct.unpack("<I", data[:4])[0]
        offset = 4

        metadata = struct.unpack("<IIIII", data[offset : offset + metadata_len])
        offset += metadata_len

        bit_width, block_size, shape0, shape1, has_qjl = metadata

        # Unpack norms
        n_blocks = shape0 * shape1
        norms_size = n_blocks * 4  # float32
        norms = np.frombuffer(data[offset : offset + norms_size], dtype=np.float32)
        offset += norms_size

        # Unpack polar indices
        polar_bits = bit_width - 1 if has_qjl else bit_width
        polar_indices = unpack_bits(data[offset:], polar_bits, n_blocks * block_size)
        polar_indices = polar_indices.reshape(n_blocks, block_size)

        result = {
            "norms": norms,
            "polar_indices": polar_indices,
            "polar_metadata": {},
            "original_shape": (shape0, shape1),
            "config": TurboQuantConfig(
                bit_width=bit_width, block_size=block_size, use_qjl=bool(has_qjl)
            ),
        }

        # Unpack QJL signs if present
        if has_qjl:
            offset += len(data) - offset  # Adjust based on actual data
            qjl_signs = unpack_bits(data[offset:], 1, n_blocks * block_size)
            result["qjl_signs"] = qjl_signs.reshape(n_blocks, block_size)

        return result
