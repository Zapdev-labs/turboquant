"""Model export module for TurboQuant.

Provides export functionality to convert models quantized with TurboQuant
to GGUF format (llama.cpp compatible) and SafeTensors format (HuggingFace compatible).
"""

import json
import struct
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .turboquant import TurboQuant, TurboQuantConfig
from .utils import pack_bits, compute_distortion, unpack_bits


# GGUF magic number and version
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_METADATA_VALUE_TYPE = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}

# GGUF tensor types (quantization types)
GGML_TYPE = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ4_XS": 16,
}

# TurboQuant custom type (using a high number to avoid conflicts)
GGML_TYPE_TURBOQUANT = 100


class TurboQuantGGUFWriter:
    """Writer for GGUF format files with TurboQuant quantization."""

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.tensor_data: List[bytes] = []

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata key-value pair."""
        self.metadata[key] = value

    def add_tensor(
        self,
        name: str,
        quantized_data: Dict[str, np.ndarray],
        shape: Tuple[int, ...],
        bit_width: int,
    ) -> None:
        """Add a quantized tensor to the GGUF file.

        Args:
            name: Tensor name
            quantized_data: Dictionary with quantized tensor data from TurboQuant
            shape: Original tensor shape
            bit_width: Quantization bit width
        """
        tensor_info = {
            "name": name,
            "shape": shape,
            "bit_width": bit_width,
            "quantized": quantized_data,
        }
        self.tensors.append(tensor_info)

    def _pack_metadata_value(self, value: Any) -> bytes:
        """Pack a single metadata value according to GGUF spec."""
        if isinstance(value, bool):
            return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["bool"]) + struct.pack("<?", value)
        elif isinstance(value, int):
            if 0 <= value <= 255:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["uint8"]) + struct.pack(
                    "<B", value
                )
            elif -128 <= value <= 127:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["int8"]) + struct.pack(
                    "<b", value
                )
            elif 0 <= value <= 65535:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["uint16"]) + struct.pack(
                    "<H", value
                )
            elif -32768 <= value <= 32767:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["int16"]) + struct.pack(
                    "<h", value
                )
            elif 0 <= value <= 4294967295:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["uint32"]) + struct.pack(
                    "<I", value
                )
            else:
                return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["int64"]) + struct.pack(
                    "<q", value
                )
        elif isinstance(value, float):
            return struct.pack("<I", GGUF_METADATA_VALUE_TYPE["float32"]) + struct.pack("<f", value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            return (
                struct.pack("<I", GGUF_METADATA_VALUE_TYPE["string"])
                + struct.pack("<Q", len(encoded))
                + encoded
            )
        elif isinstance(value, list):
            # Array type - simplified for string arrays (most common)
            if value and isinstance(value[0], str):
                header = struct.pack("<I", GGUF_METADATA_VALUE_TYPE["array"])
                header += struct.pack("<I", GGUF_METADATA_VALUE_TYPE["string"])
                header += struct.pack("<Q", len(value))
                for item in value:
                    encoded = item.encode("utf-8")
                    header += struct.pack("<Q", len(encoded)) + encoded
                return header
            else:
                # Fallback to JSON string for complex arrays
                encoded = json.dumps(value).encode("utf-8")
                return (
                    struct.pack("<I", GGUF_METADATA_VALUE_TYPE["string"])
                    + struct.pack("<Q", len(encoded))
                    + encoded
                )
        else:
            # Fallback to JSON string
            encoded = json.dumps(value).encode("utf-8")
            return (
                struct.pack("<I", GGUF_METADATA_VALUE_TYPE["string"])
                + struct.pack("<Q", len(encoded))
                + encoded
            )

    def _write_header(self) -> bytes:
        """Write the GGUF header."""
        header = GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION)
        header += struct.pack("<Q", len(self.tensors))
        header += struct.pack("<Q", len(self.metadata))
        return header

    def _write_metadata(self) -> bytes:
        """Write metadata section."""
        data = b""
        for key, value in self.metadata.items():
            # Write key as string
            encoded_key = key.encode("utf-8")
            data += struct.pack("<Q", len(encoded_key)) + encoded_key
            # Write value
            data += self._pack_metadata_value(value)
        return data

    def _write_tensor_info(self) -> Tuple[bytes, List[int]]:
        """Write tensor info section and return offsets."""
        data = b""
        offsets = []
        current_offset = 0

        for tensor in self.tensors:
            # Tensor name
            name_bytes = tensor["name"].encode("utf-8")
            data += struct.pack("<Q", len(name_bytes)) + name_bytes

            # Number of dimensions
            data += struct.pack("<I", len(tensor["shape"]))

            # Dimensions
            for dim in tensor["shape"]:
                data += struct.pack("<Q", dim)

            # Tensor type (use TurboQuant custom type)
            data += struct.pack("<I", GGML_TYPE_TURBOQUANT)

            # Tensor offset
            offsets.append(current_offset)
            data += struct.pack("<Q", current_offset)

            # Calculate size for next offset
            quantized = tensor["quantized"]
            tensor_size = self._calculate_tensor_size(quantized)
            # Align to 32 bytes
            tensor_size = (tensor_size + 31) // 32 * 32
            current_offset += tensor_size

        return data, offsets

    def _calculate_tensor_size(self, quantized: Dict[str, np.ndarray]) -> int:
        """Calculate the size of quantized tensor data in bytes."""
        size = 0
        # Norms (float32)
        if "norms" in quantized:
            size += int(quantized["norms"].nbytes)
        # Polar indices (packed bits)
        if "polar_indices" in quantized:
            bit_width = 3  # Default
            if "bit_width" in quantized:
                bw = quantized["bit_width"]
                if isinstance(bw, np.ndarray):
                    bit_width = int(bw.item())
                else:
                    bit_width = int(bw)
            n_values = quantized["polar_indices"].size
            size += (n_values * bit_width + 7) // 8
        # Radii (float32)
        if "polar_radii" in quantized:
            size += int(quantized["polar_radii"].nbytes)
        # QJL signs (1 bit per value)
        if "qjl_signs" in quantized:
            n_values = quantized["qjl_signs"].size
            size += (n_values + 7) // 8
        return size

    def _write_tensor_data(self) -> bytes:
        """Write tensor data section."""
        data = b""
        for tensor in self.tensors:
            quantized = tensor["quantized"]

            # Write norms
            if "norms" in quantized:
                data += quantized["norms"].tobytes()

            # Write packed polar indices
            if "polar_indices" in quantized:
                bit_width = tensor.get("bit_width", 3)
                packed = pack_bits(quantized["polar_indices"], bit_width)
                data += packed

            # Write radii
            if "polar_radii" in quantized:
                data += quantized["polar_radii"].tobytes()

            # Write QJL signs
            if "qjl_signs" in quantized:
                packed = pack_bits(quantized["qjl_signs"], 1)
                data += packed

            # Align to 32 bytes
            padding = (32 - len(data) % 32) % 32
            data += b"\x00" * padding

        return data

    def write(self, path: Union[str, Path]) -> None:
        """Write the GGUF file to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build file
        header = self._write_header()
        metadata = self._write_metadata()
        tensor_info, offsets = self._write_tensor_info()
        tensor_data = self._write_tensor_data()

        # Write to file
        with open(path, "wb") as f:
            f.write(header)
            f.write(metadata)
            f.write(tensor_info)
            f.write(tensor_data)


class SafeTensorsWriter:
    """Writer for SafeTensors format with TurboQuant quantization."""

    def __init__(self):
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value

    def add_tensor(
        self,
        name: str,
        quantized_data: Dict[str, np.ndarray],
        shape: Tuple[int, ...],
        bit_width: int,
    ) -> None:
        """Add a quantized tensor.

        Args:
            name: Tensor name
            quantized_data: Dictionary with quantized tensor data
            shape: Original tensor shape
            bit_width: Quantization bit width
        """
        self.tensors[name] = {
            "quantized": quantized_data,
            "shape": shape,
            "bit_width": bit_width,
        }

    def write(self, path: Union[str, Path]) -> None:
        """Write SafeTensors file to disk.

        SafeTensors format:
        - Header length (8 bytes, uint64, little-endian)
        - Header (JSON with tensor metadata)
        - Tensor data (concatenated)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build tensor metadata and data
        tensor_offsets = {}
        tensor_data_list = []
        current_offset = 0

        for name, tensor_info in self.tensors.items():
            quantized = tensor_info["quantized"]
            shape = tensor_info["shape"]
            bit_width = tensor_info["bit_width"]

            # Prepare tensor data
            tensor_bytes = b""

            # Store norms
            if "norms" in quantized:
                tensor_bytes += quantized["norms"].tobytes()

            # Store packed indices
            if "polar_indices" in quantized:
                packed = pack_bits(quantized["polar_indices"], bit_width)
                tensor_bytes += packed

            # Store radii
            if "polar_radii" in quantized:
                tensor_bytes += quantized["polar_radii"].tobytes()

            # Store QJL signs
            if "qjl_signs" in quantized:
                packed = pack_bits(quantized["qjl_signs"], 1)
                tensor_bytes += packed

            tensor_data_list.append(tensor_bytes)

            # Record metadata
            dtype = f"TURBOQUANT{bit_width}"
            tensor_offsets[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [current_offset, current_offset + len(tensor_bytes)],
            }

            current_offset += len(tensor_bytes)

        # Get bit width from first tensor or default to 3
        default_bit_width = 3
        if self.tensors:
            first_tensor = next(iter(self.tensors.values()))
            default_bit_width = first_tensor.get("bit_width", 3)

        # Build header
        header = {
            "__metadata__": {
                **self.metadata,
                "quantization_config": {
                    "quant_method": "turboquant",
                    "bits": default_bit_width,
                    "block_size": 128,
                },
            },
            **tensor_offsets,
        }

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = len(header_json)

        # Write file
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            for tensor_bytes in tensor_data_list:
                f.write(tensor_bytes)


def quantize_model_weights(
    model_state: Dict[str, np.ndarray],
    bit_width: int = 3,
    block_size: int = 128,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Quantize model weights using TurboQuant.

    Args:
        model_state: Dictionary mapping parameter names to numpy arrays
        bit_width: Number of bits for quantization (2, 3, or 4)
        block_size: Block size for quantization

    Returns:
        Tuple of (quantized_weights, metrics)
    """
    config = TurboQuantConfig(
        bit_width=bit_width,
        block_size=block_size,
        use_qjl=True,
        use_polar=True,
    )
    quantizer = TurboQuant(config)

    quantized_weights = {}
    metrics: Dict[str, Any] = {
        "total_original_bytes": 0,
        "total_compressed_bytes": 0,
        "mse_sum": 0.0,
    }
    weight_count = 0

    print(f"Quantizing model weights with {bit_width}-bit TurboQuant...")

    for name, param in model_state.items():
        # Only quantize floating point weights (not int tensors, etc.)
        if not np.issubdtype(param.dtype, np.floating):
            quantized_weights[name] = {"original": param, "skip": True}
            continue

        # Skip small tensors (not worth quantizing)
        if param.size < block_size:
            quantized_weights[name] = {"original": param, "skip": True}
            continue

        # Quantize the weight tensor
        quantized = quantizer.quantize(param)
        reconstructed = quantizer.dequantize(quantized)

        # Calculate per-tensor metrics
        distortion = compute_distortion(param, reconstructed)

        quantized_weights[name] = {
            "quantized": quantized,
            "shape": param.shape,
            "original_shape": param.shape,
            "bit_width": bit_width,
            "mse": distortion["mse"],
            "cosine_similarity": distortion["cosine_similarity"],
        }

        metrics["total_original_bytes"] += param.nbytes
        metrics["mse_sum"] += distortion["mse"]
        weight_count += 1

    metrics["avg_mse"] = metrics["mse_sum"] / max(weight_count, 1)
    metrics["num_quantized_tensors"] = weight_count

    return quantized_weights, metrics


def export_to_gguf(
    model,
    output_path: Union[str, Path],
    bit_width: int = 3,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export a model to GGUF format with TurboQuant quantization.

    Args:
        model: Model to export (transformers model or state dict)
        output_path: Path to write the GGUF file
        bit_width: Number of bits for quantization (2, 3, or 4)
        model_metadata: Optional metadata about the model

    Returns:
        Dictionary with export statistics and metrics
    """
    output_path = Path(output_path)

    # Extract state dict from model
    if hasattr(model, "state_dict"):
        # PyTorch model
        import torch

        state_dict = model.state_dict()
        model_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # Convert BFloat16 to float32 before numpy conversion
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                model_state[k] = v.detach().cpu().numpy()
            else:
                model_state[k] = v
    elif isinstance(model, dict):
        model_state = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Quantize weights
    quantized_weights, metrics = quantize_model_weights(model_state, bit_width=bit_width)

    # Create GGUF writer
    writer = TurboQuantGGUFWriter()

    # Add metadata
    writer.add_metadata("general.architecture", "unknown")
    writer.add_metadata(
        "general.name",
        model_metadata.get("name", "turboquant-model") if model_metadata else "turboquant-model",
    )
    writer.add_metadata("general.quantization_version", 1)
    writer.add_metadata("turboquant.version", "0.1.0")
    writer.add_metadata("turboquant.bit_width", bit_width)
    writer.add_metadata("turboquant.block_size", 128)
    writer.add_metadata("turboquant.use_qjl", True)
    writer.add_metadata("turboquant.use_polar", True)

    if model_metadata:
        for key, value in model_metadata.items():
            writer.add_metadata(f"general.{key}", value)

    # Add quantization metrics
    writer.add_metadata("turboquant.avg_mse", metrics["avg_mse"])
    writer.add_metadata("turboquant.num_tensors", metrics["num_quantized_tensors"])

    # Add tensors
    for name, weight_info in quantized_weights.items():
        if weight_info.get("skip"):
            # Non-quantized tensor (e.g., integer bias)
            continue

        quantized = weight_info["quantized"]
        shape = weight_info["shape"]

        # Prepare quantized data for storage
        quantized_data = {
            "norms": quantized["norms"],
            "polar_indices": quantized["polar_indices"],
            "polar_radii": quantized["polar_radii"],
            "bit_width": bit_width,
        }

        if "qjl_signs" in quantized:
            quantized_data["qjl_signs"] = quantized["qjl_signs"]

        writer.add_tensor(name, quantized_data, shape, bit_width)

    # Write file
    writer.write(output_path)

    file_size = output_path.stat().st_size
    compression_ratio = metrics["total_original_bytes"] / file_size if file_size > 0 else 0

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "compression_ratio": compression_ratio,
        **metrics,
    }


def export_to_safetensors(
    model,
    output_path: Union[str, Path],
    bit_width: int = 3,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export a model to SafeTensors format with TurboQuant quantization.

    Args:
        model: Model to export (transformers model or state dict)
        output_path: Path to write the SafeTensors file
        bit_width: Number of bits for quantization (2, 3, or 4)
        model_metadata: Optional metadata about the model

    Returns:
        Dictionary with export statistics and metrics
    """
    output_path = Path(output_path)

    # Extract state dict from model
    if hasattr(model, "state_dict"):
        # PyTorch model
        import torch

        state_dict = model.state_dict()
        model_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # Convert BFloat16 to float32 before numpy conversion
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                model_state[k] = v.detach().cpu().numpy()
            else:
                model_state[k] = v
    elif isinstance(model, dict):
        model_state = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Quantize weights
    quantized_weights, metrics = quantize_model_weights(model_state, bit_width=bit_width)

    # Create SafeTensors writer
    writer = SafeTensorsWriter()

    # Add metadata
    writer.add_metadata("format", "turboquant")
    writer.add_metadata("version", "0.1.0")
    writer.add_metadata("bit_width", bit_width)
    writer.add_metadata("block_size", 128)
    writer.add_metadata("use_qjl", True)
    writer.add_metadata("use_polar", True)
    writer.add_metadata("avg_mse", metrics["avg_mse"])
    writer.add_metadata("num_quantized_tensors", metrics["num_quantized_tensors"])

    if model_metadata:
        for key, value in model_metadata.items():
            writer.add_metadata(key, value)

    # Add tensors
    for name, weight_info in quantized_weights.items():
        if weight_info.get("skip"):
            # Store non-quantized tensors as-is (with metadata marker)
            continue

        quantized = weight_info["quantized"]
        shape = weight_info["shape"]

        # Prepare quantized data
        quantized_data = {
            "norms": quantized["norms"],
            "polar_indices": quantized["polar_indices"],
            "polar_radii": quantized["polar_radii"],
        }

        if "qjl_signs" in quantized:
            quantized_data["qjl_signs"] = quantized["qjl_signs"]

        writer.add_tensor(name, quantized_data, shape, bit_width)

    # Write file
    writer.write(output_path)

    file_size = output_path.stat().st_size
    compression_ratio = metrics["total_original_bytes"] / file_size if file_size > 0 else 0

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "compression_ratio": compression_ratio,
        **metrics,
    }


def load_gguf(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a GGUF file and return metadata and tensor info.

    This is primarily for verification and inspection purposes.

    Args:
        path: Path to the GGUF file

    Returns:
        Dictionary with metadata and tensor information
    """
    path = Path(path)

    with open(path, "rb") as f:
        # Read header
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF file: wrong magic number")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_metadata = struct.unpack("<Q", f.read(8))[0]

        # Read metadata
        metadata = {}
        for _ in range(n_metadata):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            value_type = struct.unpack("<I", f.read(4))[0]
            # Simplified: skip values
            if value_type == GGUF_METADATA_VALUE_TYPE["string"]:
                value_len = struct.unpack("<Q", f.read(8))[0]
                value = f.read(value_len).decode("utf-8")
            elif value_type == GGUF_METADATA_VALUE_TYPE["uint32"]:
                value = struct.unpack("<I", f.read(4))[0]
            elif value_type == GGUF_METADATA_VALUE_TYPE["int32"]:
                value = struct.unpack("<i", f.read(4))[0]
            elif value_type == GGUF_METADATA_VALUE_TYPE["float32"]:
                value = struct.unpack("<f", f.read(4))[0]
            elif value_type == GGUF_METADATA_VALUE_TYPE["bool"]:
                value = struct.unpack("<?", f.read(1))[0]
            else:
                # Skip other types
                value = None
            metadata[key] = value

        # Read tensor info
        tensors = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            tensor_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset})

    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
    }


def load_safetensors(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a SafeTensors file and return metadata.

    This is primarily for verification and inspection purposes.

    Args:
        path: Path to the SafeTensors file

    Returns:
        Dictionary with metadata and tensor information
    """
    path = Path(path)

    with open(path, "rb") as f:
        # Read header length
        header_len = struct.unpack("<Q", f.read(8))[0]

        # Read and parse header
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)

    metadata = header.get("__metadata__", {})
    tensors = {k: v for k, v in header.items() if not k.startswith("__")}

    return {"metadata": metadata, "tensors": tensors}


class TurboQuantGGUFLoader:
    """Loader for GGUF format files with TurboQuant quantization.

    Loads GGUF files and dequantizes tensors back to numpy arrays.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: List[Dict[str, Any]] = []
        self._tensor_data_offset = 0
        self._file_data: Optional[bytes] = None

    def _read_metadata_value(self, f) -> Any:
        """Read a metadata value from file according to GGUF spec."""
        value_type = struct.unpack("<I", f.read(4))[0]

        if value_type == GGUF_METADATA_VALUE_TYPE["bool"]:
            return struct.unpack("<?", f.read(1))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["uint8"]:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["int8"]:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["uint16"]:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["int16"]:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["uint32"]:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["int32"]:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["uint64"]:
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["int64"]:
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["float32"]:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["float64"]:
            return struct.unpack("<d", f.read(8))[0]
        elif value_type == GGUF_METADATA_VALUE_TYPE["string"]:
            value_len = struct.unpack("<Q", f.read(8))[0]
            return f.read(value_len).decode("utf-8")
        elif value_type == GGUF_METADATA_VALUE_TYPE["array"]:
            elem_type = struct.unpack("<I", f.read(4))[0]
            n_elems = struct.unpack("<Q", f.read(8))[0]
            if elem_type == GGUF_METADATA_VALUE_TYPE["string"]:
                return [self._read_string(f) for _ in range(n_elems)]
            else:
                # Skip other array types for simplicity
                return []
        else:
            return None

    def _read_string(self, f) -> str:
        """Read a string from file."""
        str_len = struct.unpack("<Q", f.read(8))[0]
        return f.read(str_len).decode("utf-8")

    def load(self) -> Dict[str, Any]:
        """Load the GGUF file and parse metadata and tensor info.

        Returns:
            Dictionary with metadata and tensor information
        """
        with open(self.path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF file: wrong magic number {magic}")

            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_metadata = struct.unpack("<Q", f.read(8))[0]

            # Read metadata
            self.metadata = {}
            for _ in range(n_metadata):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                self.metadata[key] = self._read_metadata_value(f)

            # Read tensor info
            self.tensor_infos = []
            for _ in range(n_tensors):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8")
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                tensor_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                self.tensor_infos.append(
                    {
                        "name": name,
                        "shape": shape,
                        "type": tensor_type,
                        "offset": offset,
                    }
                )

            # Record the start of tensor data
            self._tensor_data_offset = f.tell()

            # Read all tensor data into memory
            self._file_data = f.read()

        return {
            "version": version,
            "metadata": self.metadata,
            "tensors": self.tensor_infos,
        }

    def _get_bit_width(self) -> int:
        """Get bit width from metadata or default to 3."""
        return int(self.metadata.get("turboquant.bit_width", 3))

    def _get_block_size(self) -> int:
        """Get block size from metadata or default to 128."""
        return int(self.metadata.get("turboquant.block_size", 128))

    def _get_use_qjl(self) -> bool:
        """Get whether QJL is enabled from metadata."""
        use_qjl = self.metadata.get("turboquant.use_qjl", True)
        return bool(use_qjl) if not isinstance(use_qjl, str) else use_qjl.lower() == "true"

    def load_tensor(self, name: str) -> np.ndarray:
        """Load and dequantize a single tensor.

        Args:
            name: Name of the tensor to load

        Returns:
            Dequantized numpy array
        """
        if self._file_data is None:
            raise ValueError("Must call load() before loading tensors")

        # Find tensor info
        tensor_info = None
        for info in self.tensor_infos:
            if info["name"] == name:
                tensor_info = info
                break

        if tensor_info is None:
            raise ValueError(f"Tensor '{name}' not found in GGUF file")

        # Check if it's a TurboQuant tensor
        if tensor_info["type"] != GGML_TYPE_TURBOQUANT:
            raise ValueError(f"Tensor '{name}' is not a TurboQuant quantized tensor")

        # Get parameters
        bit_width = self._get_bit_width()
        block_size = self._get_block_size()
        use_qjl = self._get_use_qjl()
        shape = tensor_info["shape"]

        # Calculate sizes
        n_blocks = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        n_pairs = block_size // 2

        # Read tensor data
        offset = tensor_info["offset"]
        data = self._file_data[offset:]

        # Parse tensor data
        # Layout: norms (float32) + polar_indices (packed) + polar_radii (float32) + qjl_signs (packed, optional)

        # Read norms
        norms_size = n_blocks * 4  # float32
        norms = np.frombuffer(data[:norms_size], dtype=np.float32).copy()
        data_pos = norms_size

        # Read polar indices (packed bits)
        polar_bits = bit_width - 1 if use_qjl else bit_width
        n_indices = n_blocks * n_pairs
        indices_packed_size = (n_indices * polar_bits + 7) // 8
        polar_indices = unpack_bits(
            data[data_pos : data_pos + indices_packed_size], polar_bits, n_indices
        )
        polar_indices = polar_indices.reshape(n_blocks, n_pairs)
        data_pos += indices_packed_size

        # Read polar radii
        radii_size = n_blocks * n_pairs * 4  # float32
        polar_radii = np.frombuffer(data[data_pos : data_pos + radii_size], dtype=np.float32).copy()
        polar_radii = polar_radii.reshape(n_blocks, n_pairs)
        data_pos += radii_size

        # Read QJL signs if present
        qjl_signs = None
        if use_qjl:
            n_signs = n_blocks * block_size
            signs_packed_size = (n_signs + 7) // 8
            if data_pos + signs_packed_size <= len(data):
                qjl_signs = unpack_bits(data[data_pos : data_pos + signs_packed_size], 1, n_signs)
                qjl_signs = qjl_signs.reshape(n_blocks, block_size)

        # Build quantized dictionary
        quantized = {
            "norms": norms,
            "polar_indices": polar_indices,
            "polar_radii": polar_radii,
            "original_shape": shape,
            "config": TurboQuantConfig(
                bit_width=bit_width,
                block_size=block_size,
                use_qjl=use_qjl,
                use_polar=True,
            ),
            "polar_metadata": {},  # Will be regenerated by PolarQuant
        }

        if qjl_signs is not None:
            quantized["qjl_signs"] = qjl_signs
            # Generate JL matrix (deterministic based on seed)
            from .qjl import QJL

            qjl = QJL(block_size=block_size, rotation_seed=42 + 1)
            quantized["qjl_matrix"] = qjl.jl_matrix

        # Dequantize
        config = TurboQuantConfig(
            bit_width=bit_width,
            block_size=block_size,
            use_qjl=use_qjl,
            use_polar=True,
        )
        quantizer = TurboQuant(config)

        return quantizer.dequantize(quantized)

    def load_all_tensors(self) -> Dict[str, np.ndarray]:
        """Load and dequantize all tensors.

        Returns:
            Dictionary mapping tensor names to dequantized numpy arrays
        """
        result = {}
        for info in self.tensor_infos:
            try:
                result[info["name"]] = self.load_tensor(info["name"])
            except ValueError as e:
                warnings.warn(f"Could not load tensor '{info['name']}': {e}")
        return result


class TurboQuantSafeTensorsLoader:
    """Loader for SafeTensors format files with TurboQuant quantization.

    Loads SafeTensors files and dequantizes tensors back to numpy arrays.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: Dict[str, Dict[str, Any]] = {}
        self._tensor_data_offset = 0
        self._file_data: Optional[bytes] = None
        self._header: Optional[Dict] = None

    def load(self) -> Dict[str, Any]:
        """Load the SafeTensors file and parse metadata and tensor info.

        Returns:
            Dictionary with metadata and tensor information
        """
        with open(self.path, "rb") as f:
            # Read header length
            header_len = struct.unpack("<Q", f.read(8))[0]
            self._tensor_data_offset = 8 + header_len

            # Read and parse header
            header_json = f.read(header_len).decode("utf-8")
            header = json.loads(header_json)

        self.metadata = header.get("__metadata__", {})
        self.tensor_infos = {k: v for k, v in header.items() if not k.startswith("__")}
        self._header = header

        # Read tensor data
        with open(self.path, "rb") as f:
            f.seek(self._tensor_data_offset)
            self._file_data = f.read()

        return {
            "metadata": self.metadata,
            "tensors": self.tensor_infos,
        }

    def _get_bit_width(self) -> int:
        """Get bit width from metadata or default to 3."""
        # Try to get from quantization_config
        quant_config = self.metadata.get("quantization_config", {})
        if "bits" in quant_config:
            return int(quant_config["bits"])
        return int(self.metadata.get("bit_width", 3))

    def _get_block_size(self) -> int:
        """Get block size from metadata or default to 128."""
        quant_config = self.metadata.get("quantization_config", {})
        if "block_size" in quant_config:
            return int(quant_config["block_size"])
        return int(self.metadata.get("block_size", 128))

    def _get_use_qjl(self) -> bool:
        """Get whether QJL is enabled from metadata."""
        use_qjl = self.metadata.get("use_qjl", True)
        if isinstance(use_qjl, bool):
            return use_qjl
        return str(use_qjl).lower() == "true"

    def load_tensor(self, name: str) -> np.ndarray:
        """Load and dequantize a single tensor.

        Args:
            name: Name of the tensor to load

        Returns:
            Dequantized numpy array
        """
        if self._file_data is None:
            raise ValueError("Must call load() before loading tensors")

        if name not in self.tensor_infos:
            raise ValueError(f"Tensor '{name}' not found in SafeTensors file")

        tensor_info = self.tensor_infos[name]
        dtype = tensor_info.get("dtype", "")

        # Check if it's a TurboQuant tensor
        if not dtype.startswith("TURBOQUANT"):
            raise ValueError(
                f"Tensor '{name}' is not a TurboQuant quantized tensor (dtype={dtype})"
            )

        # Get parameters
        bit_width = self._get_bit_width()
        block_size = self._get_block_size()
        use_qjl = self._get_use_qjl()
        shape = tuple(tensor_info["shape"])
        data_offsets = tensor_info["data_offsets"]

        # Calculate sizes
        n_blocks = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        n_pairs = block_size // 2

        # Read tensor data
        tensor_data = self._file_data[data_offsets[0] : data_offsets[1]]

        # Parse tensor data
        # Layout: norms (float32) + polar_indices (packed) + polar_radii (float32) + qjl_signs (packed, optional)

        # Read norms
        norms_size = n_blocks * 4  # float32
        norms = np.frombuffer(tensor_data[:norms_size], dtype=np.float32).copy()
        data_pos = norms_size

        # Read polar indices (packed bits)
        polar_bits = bit_width - 1 if use_qjl else bit_width
        n_indices = n_blocks * n_pairs
        indices_packed_size = (n_indices * polar_bits + 7) // 8
        polar_indices = unpack_bits(
            tensor_data[data_pos : data_pos + indices_packed_size], polar_bits, n_indices
        )
        polar_indices = polar_indices.reshape(n_blocks, n_pairs)
        data_pos += indices_packed_size

        # Read polar radii
        radii_size = n_blocks * n_pairs * 4  # float32
        polar_radii = np.frombuffer(
            tensor_data[data_pos : data_pos + radii_size], dtype=np.float32
        ).copy()
        polar_radii = polar_radii.reshape(n_blocks, n_pairs)
        data_pos += radii_size

        # Read QJL signs if present
        qjl_signs = None
        if use_qjl:
            n_signs = n_blocks * block_size
            signs_packed_size = (n_signs + 7) // 8
            if data_pos + signs_packed_size <= len(tensor_data):
                qjl_signs = unpack_bits(
                    tensor_data[data_pos : data_pos + signs_packed_size], 1, n_signs
                )
                qjl_signs = qjl_signs.reshape(n_blocks, block_size)

        # Build quantized dictionary
        quantized = {
            "norms": norms,
            "polar_indices": polar_indices,
            "polar_radii": polar_radii,
            "original_shape": shape,
            "config": TurboQuantConfig(
                bit_width=bit_width,
                block_size=block_size,
                use_qjl=use_qjl,
                use_polar=True,
            ),
            "polar_metadata": {},
        }

        if qjl_signs is not None:
            quantized["qjl_signs"] = qjl_signs
            # Generate JL matrix (deterministic based on seed)
            from .qjl import QJL

            qjl = QJL(block_size=block_size, rotation_seed=42 + 1)
            quantized["qjl_matrix"] = qjl.jl_matrix

        # Dequantize
        config = TurboQuantConfig(
            bit_width=bit_width,
            block_size=block_size,
            use_qjl=use_qjl,
            use_polar=True,
        )
        quantizer = TurboQuant(config)

        return quantizer.dequantize(quantized)

    def load_all_tensors(self) -> Dict[str, np.ndarray]:
        """Load and dequantize all tensors.

        Returns:
            Dictionary mapping tensor names to dequantized numpy arrays
        """
        result = {}
        for name in self.tensor_infos.keys():
            try:
                result[name] = self.load_tensor(name)
            except ValueError as e:
                warnings.warn(f"Could not load tensor '{name}': {e}")
        return result


def load_gguf_model(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a GGUF model file and return full model state.

    Args:
        path: Path to the GGUF file

    Returns:
        Dictionary with:
        - metadata: Model metadata
        - tensors: Dictionary mapping tensor names to numpy arrays
    """
    loader = TurboQuantGGUFLoader(path)
    info = loader.load()
    tensors = loader.load_all_tensors()

    return {
        "metadata": info["metadata"],
        "tensors": tensors,
    }


def load_safetensors_model(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a SafeTensors model file and return full model state.

    Args:
        path: Path to the SafeTensors file

    Returns:
        Dictionary with:
        - metadata: Model metadata
        - tensors: Dictionary mapping tensor names to numpy arrays
    """
    loader = TurboQuantSafeTensorsLoader(path)
    info = loader.load()
    tensors = loader.load_all_tensors()

    return {
        "metadata": info["metadata"],
        "tensors": tensors,
    }


def load_model(path: Union[str, Path], format: Optional[str] = None) -> Dict[str, Any]:
    """Load a model file and return full model state.

    Auto-detects format from file extension if not specified.

    Args:
        path: Path to the model file
        format: Optional format override ('gguf' or 'safetensors')

    Returns:
        Dictionary with:
        - metadata: Model metadata
        - tensors: Dictionary mapping tensor names to numpy arrays
        - format: Detected or specified format
    """
    path = Path(path)

    if format is None:
        # Auto-detect from extension
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            format = "gguf"
        elif suffix == ".safetensors":
            format = "safetensors"
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{suffix}'. "
                f"Please specify format='gguf' or format='safetensors'"
            )

    if format == "gguf":
        result = load_gguf_model(path)
    elif format == "safetensors":
        result = load_safetensors_model(path)
    else:
        raise ValueError(f"Unknown format: {format}. Supported formats: 'gguf', 'safetensors'")

    result["format"] = format
    return result
