"""CUDA kernels for GPU-accelerated TurboQuant quantization.

Provides GPU implementations of core TurboQuant operations using PyTorch CUDA.
Falls back to CPU implementations when CUDA is unavailable.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# CUDA availability flag
_CUDA_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

# Allow environment override
if "TURBOQUANT_DISABLE_CUDA" in os.environ:
    _CUDA_AVAILABLE = False
if "TURBOQUANT_FORCE_CUDA" in os.environ:
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = True


def get_cuda_info() -> Dict[str, Any]:
    """Get information about CUDA availability and capabilities.

    Returns:
        Dictionary with CUDA information
    """
    info = {
        "available": _CUDA_AVAILABLE,
        "torch_available": _TORCH_AVAILABLE,
    }

    if _CUDA_AVAILABLE:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["capability"] = torch.cuda.get_device_capability(0)

        # Memory info
        mem_info = torch.cuda.mem_get_info(0)
        info["free_memory_gb"] = mem_info[0] / (1024**3)
        info["total_memory_gb"] = mem_info[1] / (1024**3)

    return info


def ensure_tensor(x: Union[np.ndarray, "torch.Tensor"], device: str = "cuda") -> "torch.Tensor":
    """Ensure input is a PyTorch tensor on the specified device.

    Args:
        x: Input array or tensor
        device: Target device ("cuda" or "cpu")

    Returns:
        PyTorch tensor on target device
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)


def ensure_numpy(x: "torch.Tensor") -> np.ndarray:
    """Convert PyTorch tensor to numpy array.

    Args:
        x: PyTorch tensor

    Returns:
        Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


class CUDAKernels:
    """CUDA-accelerated kernels for TurboQuant operations.

    Provides GPU implementations of:
    - Walsh-Hadamard transform
    - Random rotation
    - Polar transformation
    - Quantization/dequantization
    - JL transform
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize CUDA kernels.

        Args:
            device: CUDA device to use (None for auto)
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CUDA kernels")

        if device is None:
            device = "cuda" if _CUDA_AVAILABLE else "cpu"

        self.device = device
        self._rotation_cache: Dict[int, torch.Tensor] = {}
        self._jl_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_rotation_matrix(self, dim: int, seed: int) -> "torch.Tensor":
        """Get or generate rotation matrix for Walsh-Hadamard transform."""
        if dim not in self._rotation_cache:
            # Generate random orthogonal matrix using QR decomposition
            torch.manual_seed(seed)
            gaussian = torch.randn(dim, dim, device=self.device, dtype=torch.float32)
            rotation, _upper = torch.linalg.qr(gaussian)
            self._rotation_cache[dim] = rotation
        return self._rotation_cache[dim]

    def _get_jl_matrix(self, in_dim: int, out_dim: int, seed: int) -> "torch.Tensor":
        """Get or generate JL projection matrix."""
        key = (in_dim, out_dim, seed)
        if key not in self._jl_cache:
            torch.manual_seed(seed)
            projection = torch.randn(out_dim, in_dim, device=self.device, dtype=torch.float32)
            projection = projection / np.sqrt(out_dim)  # Scale for JL lemma
            self._jl_cache[key] = projection
        return self._jl_cache[key]

    def walsh_hadamard(self, x: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
        """CUDA-accelerated Walsh-Hadamard transform.

        Uses the iterative butterfly algorithm optimized for GPU.

        Args:
            x: Input tensor of shape (..., d) where d is power of 2

        Returns:
            Transformed tensor
        """
        x_t = ensure_tensor(x, self.device)
        d = x_t.shape[-1]

        if d & (d - 1) != 0:
            raise ValueError(f"Dimension {d} must be a power of 2")

        # Iterative butterfly on GPU
        h = 1
        while h < d:
            # Reshape for butterfly operation
            shape = x_t.shape[:-1] + (-1, 2 * h)
            x_reshaped = x_t.reshape(shape)

            # Split and butterfly
            even = x_reshaped[..., :h]
            odd = x_reshaped[..., h:]

            x_t = torch.cat([even + odd, even - odd], dim=-1).reshape(x_t.shape)
            h *= 2

        return x_t / np.sqrt(d)

    def random_rotation(
        self, x: Union[np.ndarray, "torch.Tensor"], rotation_matrix: "torch.Tensor"
    ) -> "torch.Tensor":
        """CUDA-accelerated random rotation.

        Args:
            x: Input vectors
            rotation_matrix: Rotation matrix

        Returns:
            Rotated vectors
        """
        x_t = ensure_tensor(x, self.device)
        rot_t = rotation_matrix.to(self.device)
        return x_t @ rot_t.T

    def polar_transform(
        self, x: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """CUDA-accelerated polar transformation.

        Args:
            x: Input array of shape (..., 2n)

        Returns:
            Tuple of (radii, angles)
        """
        x_t = ensure_tensor(x, self.device)
        d = x_t.shape[-1]

        if d % 2 != 0:
            raise ValueError(f"Last dimension {d} must be even")

        n_pairs = d // 2
        pairs = x_t.reshape(x_t.shape[:-1] + (n_pairs, 2))

        x_coord = pairs[..., 0]
        y_coord = pairs[..., 1]

        radii = torch.sqrt(x_coord**2 + y_coord**2)
        angles = torch.atan2(y_coord, x_coord)

        return radii, angles

    def inverse_polar_transform(
        self, radii: Union[np.ndarray, "torch.Tensor"], angles: Union[np.ndarray, "torch.Tensor"]
    ) -> "torch.Tensor":
        """CUDA-accelerated inverse polar transformation.

        Args:
            radii: Radii tensor
            angles: Angles tensor

        Returns:
            Cartesian coordinates
        """
        r_t = ensure_tensor(radii, self.device)
        a_t = ensure_tensor(angles, self.device)

        x = r_t * torch.cos(a_t)
        y = r_t * torch.sin(a_t)

        # Interleave
        result = torch.stack([x, y], dim=-1).reshape(r_t.shape[:-1] + (-1,))
        return result

    def quantize(
        self, x: Union[np.ndarray, "torch.Tensor"], codebook: "torch.Tensor"
    ) -> "torch.Tensor":
        """CUDA-accelerated quantization.

        Args:
            x: Input values
            codebook: Quantization centroids

        Returns:
            Quantized indices
        """
        x_t = ensure_tensor(x, self.device)
        codebook_t = codebook.to(self.device)

        # Expand dimensions for broadcasting
        x_expanded = x_t.unsqueeze(-1)
        codebook_expanded = codebook_t.view((1,) * x_t.ndim + (-1,))

        # Compute distances and find nearest
        distances = torch.abs(x_expanded - codebook_expanded)
        indices = torch.argmin(distances, dim=-1)

        return indices

    def dequantize(self, indices: "torch.Tensor", codebook: "torch.Tensor") -> "torch.Tensor":
        """CUDA-accelerated dequantization.

        Args:
            indices: Quantized indices
            codebook: Quantization centroids

        Returns:
            Dequantized values
        """
        codebook_t = codebook.to(self.device)
        return codebook_t[indices]

    def jl_transform(
        self, x: Union[np.ndarray, "torch.Tensor"], jl_matrix: "torch.Tensor"
    ) -> "torch.Tensor":
        """CUDA-accelerated JL transform.

        Args:
            x: Input vectors
            jl_matrix: JL projection matrix

        Returns:
            Projected vectors
        """
        x_t = ensure_tensor(x, self.device)
        jl_t = jl_matrix.to(self.device)
        return x_t @ jl_t.T

    def normalize(
        self, x: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """CUDA-accelerated normalization.

        Args:
            x: Input vectors

        Returns:
            Tuple of (normalized, norms)
        """
        x_t = ensure_tensor(x, self.device)
        norms = torch.norm(x_t, dim=-1, keepdim=True)
        normalized = x_t / (norms + 1e-10)
        return normalized, norms.squeeze(-1)

    def compute_norms(self, x: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
        """CUDA-accelerated norm computation.

        Args:
            x: Input vectors

        Returns:
            Norms
        """
        x_t = ensure_tensor(x, self.device)
        return torch.norm(x_t, dim=-1)

    def batch_quantize(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        block_size: int,
        bit_width: int,
        rotation_seed: int = 42,
    ) -> Dict[str, "torch.Tensor"]:
        """Full batch quantization pipeline on GPU.

        Args:
            x: Input vectors of shape (n_vectors, dim)
            block_size: Block size for quantization
            bit_width: Number of bits per value
            rotation_seed: Random seed for rotation

        Returns:
            Dictionary with quantized representation
        """
        x_t = ensure_tensor(x, self.device)
        n_vectors, dim = x_t.shape

        # Normalize
        x_norm, norms = self.normalize(x_t)

        # Random rotation
        rot_matrix = self._get_rotation_matrix(dim, rotation_seed)
        x_rotated = self.random_rotation(x_norm, rot_matrix)

        # Polar transform
        radii, angles = self.polar_transform(x_rotated)

        # Quantize angles
        n_levels = 2**bit_width
        angles_normalized = (angles + np.pi) / (2 * np.pi)
        indices = (angles_normalized * n_levels).long()
        indices = torch.clamp(indices, 0, n_levels - 1)

        return {
            "indices": indices,
            "norms": norms,
            "radii": radii,
            "rotation_matrix": rot_matrix,
        }

    def batch_dequantize(self, quantized: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Full batch dequantization pipeline on GPU.

        Args:
            quantized: Dictionary from batch_quantize()

        Returns:
            Reconstructed vectors
        """
        indices = quantized["indices"]
        norms = quantized["norms"]
        radii = quantized["radii"]
        rot_matrix = quantized["rotation_matrix"]

        bit_width = int(np.log2(indices.max().item() + 1))
        n_levels = 2**bit_width

        # Dequantize angles
        angles_normalized = indices.float() / n_levels
        angles = angles_normalized * 2 * np.pi - np.pi

        # Inverse polar transform
        x_rotated = self.inverse_polar_transform(radii, angles)

        # Inverse rotation
        x_norm = x_rotated @ rot_matrix

        # Renormalize and rescale
        x_norm = x_norm / (torch.norm(x_norm, dim=-1, keepdim=True) + 1e-10)
        x_reconstructed = x_norm * norms.unsqueeze(-1)

        return x_reconstructed


class CUDAKVCacheCompressor:
    """GPU-accelerated KV cache compressor.

    Provides efficient KV cache compression/decompression on GPU
    for transformer inference.
    """

    def __init__(
        self,
        bit_width: int = 3,
        block_size: int = 128,
        device: Optional[str] = None,
    ):
        """Initialize CUDA KV cache compressor.

        Args:
            bit_width: Number of bits per value
            block_size: Block size for quantization
            device: CUDA device (None for auto)
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CUDA KV cache compression")

        self.bit_width = bit_width
        self.block_size = block_size
        self.kernels = CUDAKernels(device)
        self.device = self.kernels.device

    def compress_kv(
        self,
        key_states: Union[np.ndarray, "torch.Tensor"],
        value_states: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, "torch.Tensor"]]:
        """Compress key and value states on GPU.

        Args:
            key_states: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            value_states: Value tensor of same shape

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        # Reshape for quantization
        batch, n_heads, seq_len, head_dim = key_states.shape

        k_reshaped = key_states.reshape(-1, head_dim)
        v_reshaped = value_states.reshape(-1, head_dim)

        k_comp = self.kernels.batch_quantize(k_reshaped, self.block_size, self.bit_width)
        v_comp = self.kernels.batch_quantize(v_reshaped, self.block_size, self.bit_width)

        # Add shape info
        k_comp["shape"] = (batch, n_heads, seq_len, head_dim)
        v_comp["shape"] = (batch, n_heads, seq_len, head_dim)

        return k_comp, v_comp

    def decompress_kv(
        self,
        k_comp: Dict[str, "torch.Tensor"],
        v_comp: Dict[str, "torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Decompress KV cache on GPU.

        Args:
            k_comp: Compressed keys
            v_comp: Compressed values

        Returns:
            Tuple of (key_states, value_states)
        """
        k_reconstructed = self.kernels.batch_dequantize(k_comp)
        v_reconstructed = self.kernels.batch_dequantize(v_comp)

        shape = k_comp["shape"]
        k_states = k_reconstructed.reshape(shape)
        v_states = v_reconstructed.reshape(shape)

        return k_states, v_states

    def compress_to_bytes(
        self,
        key_states: Union[np.ndarray, "torch.Tensor"],
        value_states: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[bytes, bytes]:
        """Compress KV cache to compact byte representation.

        Args:
            key_states: Key states
            value_states: Value states

        Returns:
            Tuple of (k_bytes, v_bytes)
        """
        k_comp, v_comp = self.compress_kv(key_states, value_states)

        # Convert to bytes (simplified - in production use bit-packing)
        k_bytes = self._compress_dict_to_bytes(k_comp)
        v_bytes = self._compress_dict_to_bytes(v_comp)

        return k_bytes, v_bytes

    def _compress_dict_to_bytes(self, comp: Dict[str, "torch.Tensor"]) -> bytes:
        """Convert compressed dict to bytes."""
        import io
        import struct

        buffer = io.BytesIO()

        # Write metadata
        indices = comp["indices"].cpu().numpy()
        norms = comp["norms"].cpu().numpy()
        radii = comp["radii"].cpu().numpy()

        # Pack shape
        shape = comp.get("shape", (1, 1, 1, 1))
        buffer.write(struct.pack("<IIII", *shape))

        # Pack tensors
        buffer.write(indices.tobytes())
        buffer.write(norms.tobytes())
        buffer.write(radii.tobytes())

        return buffer.getvalue()

    def get_memory_stats(
        self,
        seq_len: int,
        batch_size: int = 1,
        n_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict[str, Any]:
        """Compute memory usage statistics.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            n_heads: Number of attention heads
            head_dim: Head dimension

        Returns:
            Memory statistics dictionary
        """
        n_tokens = batch_size * seq_len * n_heads

        # FP16 baseline
        fp16_bytes_per_token = head_dim * 2
        fp16_total = n_tokens * fp16_bytes_per_token * 2

        # TurboQuant compressed
        bits_per_value = self.bit_width
        tq_bytes_per_token = (head_dim * bits_per_value) / 8
        tq_metadata_per_token = 4
        tq_total = n_tokens * (tq_bytes_per_token + tq_metadata_per_token) * 2

        compression_ratio = fp16_total / tq_total
        memory_saved_gb = (fp16_total - tq_total) / (1024**3)

        # GPU-specific info
        stats = {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "fp16_memory_gb": fp16_total / (1024**3),
            "turboquant_memory_gb": tq_total / (1024**3),
            "compression_ratio": compression_ratio,
            "memory_saved_gb": memory_saved_gb,
            "max_context_fp16": int(34e9 / fp16_total * seq_len),
            "max_context_tq": int(34e9 / tq_total * seq_len),
        }

        if _CUDA_AVAILABLE:
            cuda_info = get_cuda_info()
            stats["gpu_device"] = cuda_info.get("device_name", "Unknown")
            stats["gpu_memory_gb"] = cuda_info.get("total_memory_gb", 0)

        return stats


class StreamingCUDAKVCache:
    """Streaming KV cache with GPU compression for long sequences.

    Manages KV cache incrementally during autoregressive generation,
    keeping compressed representation on GPU.
    """

    def __init__(
        self,
        bit_width: int = 3,
        block_size: int = 128,
        max_seq_len: int = 262144,
        device: Optional[str] = None,
    ):
        self.compressor = CUDAKVCacheCompressor(bit_width, block_size, device)
        self.max_seq_len = max_seq_len
        self.k_cache_compressed: list = []
        self.v_cache_compressed: list = []
        self.seq_len = 0
        self.device = self.compressor.device

    def append(
        self,
        k_new: Union[np.ndarray, "torch.Tensor"],
        v_new: Union[np.ndarray, "torch.Tensor"],
    ) -> None:
        """Append new key-value pairs to the cache.

        Args:
            k_new: New key states
            v_new: New value states
        """
        k_t = ensure_tensor(k_new, self.device)
        v_t = ensure_tensor(v_new, self.device)

        # Compress new states
        k_comp, v_comp = self.compressor.compress_kv(k_t, v_t)

        # Append to cache
        self.k_cache_compressed.append(k_comp)
        self.v_cache_compressed.append(v_comp)

        # Update sequence length
        self.seq_len += k_new.shape[2] if len(k_new.shape) == 4 else 1

    def get_cache(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Get full decompressed KV cache.

        Returns:
            Tuple of (k_cache, v_cache)
        """
        if not self.k_cache_compressed:
            return torch.tensor([]), torch.tensor([])

        # Decompress all blocks
        k_list = []
        v_list = []

        for k_comp, v_comp in zip(self.k_cache_compressed, self.v_cache_compressed):
            k_dec = self.compressor.kernels.batch_dequantize(k_comp)
            v_dec = self.compressor.kernels.batch_dequantize(v_comp)
            k_list.append(k_dec)
            v_list.append(v_dec)

        # Concatenate
        k_full = torch.cat(k_list, dim=0)
        v_full = torch.cat(v_list, dim=0)

        return k_full, v_full

    def clear(self) -> None:
        """Clear the cache."""
        self.k_cache_compressed = []
        self.v_cache_compressed = []
        self.seq_len = 0

        # Clear GPU cache
        if _CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        total = 0
        for kc in self.k_cache_compressed:
            for v in kc.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.nelement()
        for vc in self.v_cache_compressed:
            for v in vc.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.nelement()
        return total


def benchmark_cuda_kernels(
    n_vectors: int = 10000, dim: int = 128, n_trials: int = 100
) -> Dict[str, Any]:
    """Benchmark CUDA kernel performance.

    Args:
        n_vectors: Number of vectors to test
        dim: Vector dimension
        n_trials: Number of timing trials

    Returns:
        Benchmark results
    """
    if not _TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    if not _CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    # Generate test data
    x = np.random.randn(n_vectors, dim).astype(np.float32)

    kernels = CUDAKernels()

    # Warmup
    _ = kernels.walsh_hadamard(x)
    torch.cuda.synchronize()

    # Benchmark Walsh-Hadamard
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_trials):
        _ = kernels.walsh_hadamard(x)
    end.record()
    torch.cuda.synchronize()
    wh_time = start.elapsed_time(end) / n_trials

    # Benchmark rotation
    rot_matrix = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    start.record()
    for _ in range(n_trials):
        _ = kernels.random_rotation(x, rot_matrix)
    end.record()
    torch.cuda.synchronize()
    rot_time = start.elapsed_time(end) / n_trials

    # Benchmark polar transform
    start.record()
    for _ in range(n_trials):
        _ = kernels.polar_transform(x)
    end.record()
    torch.cuda.synchronize()
    polar_time = start.elapsed_time(end) / n_trials

    return {
        "device": torch.cuda.get_device_name(0),
        "n_vectors": n_vectors,
        "dim": dim,
        "walsh_hadamard_ms": wh_time,
        "rotation_ms": rot_time,
        "polar_transform_ms": polar_time,
        "throughput_mvec_s": (n_vectors * n_trials) / (wh_time + rot_time + polar_time) * 1000,
    }
