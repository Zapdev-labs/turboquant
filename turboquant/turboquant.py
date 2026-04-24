import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant quantization.

    Args:
        bit_width: Number of bits per value (2, 3, or 4).
        block_size: Size of quantization blocks (32, 64, 128, or 256).
        use_qjl: Whether to use QJL error correction on the residual.
        use_polar: Whether to use PolarQuant transformation.
        rotation_seed: Random seed for deterministic rotations.
        dtype: Computation dtype name.
        rotation: ``hadamard`` for fast sign-flipped FWHT or ``random`` for dense QR
            rotation.
        radii_dtype: Storage dtype for polar radii (``float16`` or ``float32``).
        store_qjl_matrix: Include the dense JL matrix in quantized dictionaries.
    """

    bit_width: int = 3
    block_size: int = 128
    use_qjl: bool = True
    use_polar: bool = True
    rotation_seed: int = 42
    dtype: str = "float32"
    rotation: str = "hadamard"
    radii_dtype: str = "float16"
    store_qjl_matrix: bool = False

    def __post_init__(self) -> None:
        if self.bit_width not in (2, 3, 4):
            raise ValueError("bit_width must be 2, 3, or 4")
        if self.block_size not in (32, 64, 128, 256):
            raise ValueError("block_size must be 32, 64, 128, or 256")
        if self.dtype not in ("float16", "float32"):
            raise ValueError("dtype must be 'float16' or 'float32'")
        if self.rotation not in ("hadamard", "random"):
            raise ValueError("rotation must be 'hadamard' or 'random'")
        if self.radii_dtype not in ("float16", "float32"):
            raise ValueError("radii_dtype must be 'float16' or 'float32'")


class TurboQuant:
    """TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

    Implements the full TurboQuant algorithm combining:
    1. PolarQuant: Random rotation + polar transformation + angle quantization.
    2. QJL: 1-bit quantization on residual for error correction.

    Inputs may have any trailing dimension. Vectors are split into ``block_size``
    chunks and padded with zeros when needed; dequantization removes the padding.
    """

    def __init__(self, config: Optional[TurboQuantConfig] = None) -> None:
        self.config = config or TurboQuantConfig()
        self._init_components()

    def _init_components(self) -> None:
        """Initialize PolarQuant and QJL components."""
        from .polarquant import PolarQuant
        from .qjl import QJL

        polar_bits = self.config.bit_width - 1 if self.config.use_qjl else self.config.bit_width
        self.polar = PolarQuant(
            bit_width=polar_bits,
            block_size=self.config.block_size,
            rotation_seed=self.config.rotation_seed,
            rotation=self.config.rotation,
        )

        if self.config.use_qjl:
            self.qjl = QJL(
                block_size=self.config.block_size,
                rotation_seed=self.config.rotation_seed + 1,
            )
        else:
            self.qjl = None

    def _reshape_to_blocks(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, ...], int, int, int]:
        """Convert an array into 2D quantization blocks."""
        if x.ndim == 0:
            raise ValueError("input must have at least one dimension")

        original_shape = tuple(int(dim) for dim in x.shape)
        feature_dim = original_shape[-1]
        if feature_dim <= 0:
            raise ValueError("last dimension must be non-empty")

        blocks_per_vector = (feature_dim + self.config.block_size - 1) // self.config.block_size
        padded_dim = blocks_per_vector * self.config.block_size

        if padded_dim != feature_dim:
            pad_width = [(0, 0)] * x.ndim
            pad_width[-1] = (0, padded_dim - feature_dim)
            x = np.pad(x, pad_width, mode="constant")

        n_vectors = int(np.prod(original_shape[:-1], dtype=np.int64)) if x.ndim > 1 else 1
        blocks = x.reshape(n_vectors, blocks_per_vector, self.config.block_size)
        return (
            blocks.reshape(-1, self.config.block_size),
            original_shape,
            feature_dim,
            padded_dim,
            blocks_per_vector,
        )

    def _restore_from_blocks(
        self,
        blocks: np.ndarray,
        original_shape: Tuple[int, ...],
        feature_dim: int,
        padded_dim: int,
        blocks_per_vector: int,
    ) -> np.ndarray:
        """Restore padded blocks back to the original array shape."""
        n_vectors = (
            int(np.prod(original_shape[:-1], dtype=np.int64)) if len(original_shape) > 1 else 1
        )
        padded_shape = original_shape[:-1] + (padded_dim,)
        restored = blocks.reshape(n_vectors, blocks_per_vector, self.config.block_size)
        restored = restored.reshape(padded_shape)
        return restored[..., :feature_dim].reshape(original_shape)

    def quantize(self, x: np.ndarray) -> Dict[str, Any]:
        """Quantize input vectors using TurboQuant.

        Args:
            x: Input array of shape (..., d). The trailing dimension may be any
                positive integer and is padded to a whole number of blocks.

        Returns:
            Dictionary containing the quantized representation.
        """
        input_array = np.asarray(x)
        x_compute = input_array.astype(np.float32, copy=False)
        (
            x_blocks,
            original_shape,
            feature_dim,
            padded_dim,
            blocks_per_vector,
        ) = self._reshape_to_blocks(x_compute)

        norms = np.linalg.norm(x_blocks, axis=1, keepdims=True).astype(np.float32)
        x_unit = np.divide(
            x_blocks,
            norms,
            out=np.zeros_like(x_blocks, dtype=np.float32),
            where=norms > 1e-10,
        )

        polar_result = self.polar.quantize(x_unit)

        result: Dict[str, Any] = {
            "norms": norms,
            "polar_indices": polar_result["indices"],
            "polar_radii": polar_result["radii"].astype(self.config.radii_dtype),
            "polar_metadata": polar_result["metadata"],
            "original_shape": original_shape,
            "feature_dim": feature_dim,
            "padded_dim": padded_dim,
            "blocks_per_vector": blocks_per_vector,
            "original_nbytes": int(input_array.nbytes),
            "input_dtype": str(input_array.dtype),
            "config": self.config,
        }

        if self.config.use_qjl and self.qjl is not None:
            x_reconstructed = self.polar.dequantize(polar_result)
            residual = x_unit - x_reconstructed
            qjl_result = self.qjl.quantize(residual)
            result["qjl_signs"] = qjl_result["signs"]
            if self.config.store_qjl_matrix:
                result["qjl_matrix"] = qjl_result["jl_matrix"]

        return result

    def dequantize(self, quantized: Dict[str, Any]) -> np.ndarray:
        """Dequantize TurboQuant representation back to vectors."""
        quantized_config = quantized.get("config")
        if isinstance(quantized_config, TurboQuantConfig) and quantized_config != self.config:
            return TurboQuant(quantized_config).dequantize(quantized)

        norms = np.asarray(quantized["norms"], dtype=np.float32)
        original_shape = tuple(int(dim) for dim in quantized["original_shape"])
        feature_dim = int(quantized.get("feature_dim", original_shape[-1]))
        padded_dim = int(quantized.get("padded_dim", feature_dim))
        blocks_per_vector = int(
            quantized.get(
                "blocks_per_vector",
                (padded_dim + self.config.block_size - 1) // self.config.block_size,
            )
        )

        polar_result = {
            "indices": quantized["polar_indices"],
            "radii": quantized["polar_radii"],
            "metadata": quantized.get("polar_metadata", {}),
            "norms": np.ones_like(norms, dtype=np.float32),
        }
        x_unit = self.polar.dequantize(polar_result)

        if "qjl_signs" in quantized and self.qjl is not None:
            qjl_result = {
                "signs": quantized["qjl_signs"],
                "jl_matrix": quantized.get("qjl_matrix", self.qjl.jl_matrix),
            }
            residual = self.qjl.dequantize(qjl_result)
            x_unit = x_unit + residual

        unit_norms = np.linalg.norm(x_unit, axis=1, keepdims=True)
        x_unit = np.divide(
            x_unit,
            unit_norms,
            out=np.zeros_like(x_unit, dtype=np.float32),
            where=unit_norms > 1e-10,
        )
        x_blocks = x_unit * norms

        return self._restore_from_blocks(
            x_blocks,
            original_shape,
            feature_dim,
            padded_dim,
            blocks_per_vector,
        )

    def compressed_size_bytes(self, quantized: Dict[str, Any]) -> int:
        """Estimate compact serialized size without dense metadata matrices."""
        config = quantized.get("config", self.config)
        polar_bits = config.bit_width - 1 if config.use_qjl else config.bit_width
        polar_bytes = (int(np.asarray(quantized["polar_indices"]).size) * polar_bits + 7) // 8
        qjl_signs = quantized.get("qjl_signs")
        qjl_bytes = 0 if qjl_signs is None else (int(np.asarray(qjl_signs).size) + 7) // 8
        metadata_bytes = 96 + len(tuple(quantized["original_shape"])) * 8
        return (
            int(np.asarray(quantized["norms"]).nbytes)
            + int(np.asarray(quantized["polar_radii"]).nbytes)
            + polar_bytes
            + qjl_bytes
            + metadata_bytes
        )

    def compression_stats(self, quantized: Dict[str, Any]) -> Dict[str, float]:
        """Return estimated compression size and ratio for a quantized dictionary."""
        original_nbytes = float(quantized.get("original_nbytes", 0))
        if original_nbytes <= 0:
            original_shape = tuple(int(dim) for dim in quantized["original_shape"])
            original_nbytes = float(int(np.prod(original_shape, dtype=np.int64)) * 4)

        compressed_nbytes = float(self.compressed_size_bytes(quantized))
        return {
            "original_bytes": original_nbytes,
            "compressed_bytes": compressed_nbytes,
            "compression_ratio": original_nbytes / compressed_nbytes if compressed_nbytes else 0.0,
        }

    def compress(self, x: np.ndarray) -> bytes:
        """Serialize a quantized array to portable bytes."""
        quantized = self.quantize(x)
        return self._pack_quantized(quantized)

    def decompress(self, data: bytes) -> np.ndarray:
        """Deserialize bytes and reconstruct the original array shape."""
        quantized = self._unpack_quantized(data)
        config = quantized["config"]
        quantizer = self if config == self.config else TurboQuant(config)
        return quantizer.dequantize(quantized)

    def _pack_quantized(self, quantized: Dict[str, Any]) -> bytes:
        """Pack quantized representation into a compressed NPZ payload."""
        from .utils import pack_bits

        config = quantized.get("config", self.config)
        polar_bits = config.bit_width - 1 if config.use_qjl else config.bit_width
        polar_indices = np.asarray(quantized["polar_indices"])
        payload: Dict[str, Any] = {
            "norms": quantized["norms"],
            "polar_indices_packed": np.frombuffer(
                pack_bits(polar_indices, polar_bits), dtype=np.uint8
            ),
            "polar_indices_shape": np.asarray(polar_indices.shape, dtype=np.int64),
            "polar_radii": quantized["polar_radii"],
            "original_shape": np.asarray(quantized["original_shape"], dtype=np.int64),
            "bit_width": np.asarray([config.bit_width], dtype=np.int16),
            "block_size": np.asarray([config.block_size], dtype=np.int16),
            "use_qjl": np.asarray([config.use_qjl], dtype=np.bool_),
            "use_polar": np.asarray([config.use_polar], dtype=np.bool_),
            "rotation_seed": np.asarray([config.rotation_seed], dtype=np.int64),
            "rotation_code": np.asarray([1 if config.rotation == "random" else 0], dtype=np.int8),
            "radii_dtype_code": np.asarray(
                [1 if config.radii_dtype == "float32" else 0], dtype=np.int8
            ),
            "feature_dim": np.asarray([quantized["feature_dim"]], dtype=np.int64),
            "padded_dim": np.asarray([quantized["padded_dim"]], dtype=np.int64),
            "blocks_per_vector": np.asarray([quantized["blocks_per_vector"]], dtype=np.int64),
            "original_nbytes": np.asarray([quantized["original_nbytes"]], dtype=np.int64),
        }
        if "qjl_signs" in quantized:
            qjl_signs = np.asarray(quantized["qjl_signs"])
            payload["qjl_signs_packed"] = np.frombuffer(pack_bits(qjl_signs, 1), dtype=np.uint8)
            payload["qjl_signs_shape"] = np.asarray(qjl_signs.shape, dtype=np.int64)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        return buffer.getvalue()

    def _unpack_quantized(self, data: bytes) -> Dict[str, Any]:
        """Unpack bytes into a quantized representation."""
        from .utils import unpack_bits

        buffer = io.BytesIO(data)
        with np.load(buffer, allow_pickle=False) as loaded:
            rotation = "random" if int(loaded["rotation_code"][0]) == 1 else "hadamard"
            radii_dtype = "float32" if int(loaded["radii_dtype_code"][0]) == 1 else "float16"
            config = TurboQuantConfig(
                bit_width=int(loaded["bit_width"][0]),
                block_size=int(loaded["block_size"][0]),
                use_qjl=bool(loaded["use_qjl"][0]),
                use_polar=bool(loaded["use_polar"][0]),
                rotation_seed=int(loaded["rotation_seed"][0]),
                rotation=rotation,
                radii_dtype=radii_dtype,
            )

            if "polar_indices_packed" in loaded.files:
                polar_shape = tuple(int(dim) for dim in loaded["polar_indices_shape"])
                polar_bits = config.bit_width - 1 if config.use_qjl else config.bit_width
                polar_indices = unpack_bits(
                    loaded["polar_indices_packed"].tobytes(),
                    polar_bits,
                    int(np.prod(polar_shape, dtype=np.int64)),
                ).reshape(polar_shape)
            else:
                polar_indices = loaded["polar_indices"].copy()

            result: Dict[str, Any] = {
                "norms": loaded["norms"].copy(),
                "polar_indices": polar_indices,
                "polar_radii": loaded["polar_radii"].copy(),
                "original_shape": tuple(int(dim) for dim in loaded["original_shape"]),
                "feature_dim": int(loaded["feature_dim"][0]),
                "padded_dim": int(loaded["padded_dim"][0]),
                "blocks_per_vector": int(loaded["blocks_per_vector"][0]),
                "original_nbytes": int(loaded["original_nbytes"][0]),
                "polar_metadata": {},
                "config": config,
            }
            if "qjl_signs_packed" in loaded.files:
                qjl_shape = tuple(int(dim) for dim in loaded["qjl_signs_shape"])
                result["qjl_signs"] = unpack_bits(
                    loaded["qjl_signs_packed"].tobytes(),
                    1,
                    int(np.prod(qjl_shape, dtype=np.int64)),
                ).reshape(qjl_shape)
            elif "qjl_signs" in loaded.files:
                result["qjl_signs"] = loaded["qjl_signs"].copy()

        return result
