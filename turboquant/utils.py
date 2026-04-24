import numpy as np


def pack_bits(indices: np.ndarray, bits_per_value: int) -> bytes:
    """Pack integer indices into compact byte representation.

    Args:
        indices: Array of indices to pack (values 0 to 2^bits_per_value - 1)
        bits_per_value: Number of bits per index

    Returns:
        Packed bytes
    """
    indices = indices.flatten()
    n_values = len(indices)

    # Calculate total bits needed
    total_bits = n_values * bits_per_value
    total_bytes = (total_bits + 7) // 8

    # Create output array
    result = np.zeros(total_bytes, dtype=np.uint8)

    # Pack values bit by bit
    bit_pos = 0
    for val in indices:
        # Ensure value fits in bits_per_value
        val = int(val) & ((1 << bits_per_value) - 1)

        # Calculate byte position and bit offset
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8

        # Pack value across bytes if it spans boundaries
        remaining_bits = bits_per_value
        val_shift = 0

        while remaining_bits > 0:
            # Bits that fit in current byte
            bits_in_current = min(8 - bit_offset, remaining_bits)

            # Extract bits for current byte
            mask = (1 << bits_in_current) - 1
            bits_to_write = (val >> val_shift) & mask

            # Write to result
            result[byte_pos] |= bits_to_write << bit_offset

            # Update positions
            val_shift += bits_in_current
            remaining_bits -= bits_in_current
            bit_offset = 0
            byte_pos += 1

        bit_pos += bits_per_value

    return bytes(result)


def unpack_bits(data: bytes, bits_per_value: int, n_values: int) -> np.ndarray:
    """Unpack compact byte representation back to indices.

    Args:
        data: Packed bytes
        bits_per_value: Number of bits per index
        n_values: Number of values to unpack

    Returns:
        Array of indices
    """
    result = np.zeros(n_values, dtype=np.int32)

    # Convert bytes to numpy array for easier manipulation
    bytes_arr = np.frombuffer(data, dtype=np.uint8)

    # Unpack values bit by bit
    bit_pos = 0
    for i in range(n_values):
        val = 0
        remaining_bits = bits_per_value
        val_shift = 0

        while remaining_bits > 0:
            # Calculate byte position and bit offset
            byte_pos = bit_pos // 8
            bit_offset = bit_pos % 8

            # Bits that can be read from current byte
            bits_in_current = min(8 - bit_offset, remaining_bits)

            # Read bits
            if byte_pos < len(bytes_arr):
                mask = (1 << bits_in_current) - 1
                bits_read = (bytes_arr[byte_pos] >> bit_offset) & mask
                val |= bits_read << val_shift

            # Update positions
            val_shift += bits_in_current
            remaining_bits -= bits_in_current
            bit_pos += bits_in_current

        result[i] = val

    return result


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Mean Squared Error between original and reconstructed arrays.

    Args:
        original: Original array
        reconstructed: Reconstructed array

    Returns:
        MSE value
    """
    return np.mean((original - reconstructed) ** 2)


def compute_distortion(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute various distortion metrics.

    Args:
        original: Original array
        reconstructed: Reconstructed array

    Returns:
        Dictionary of metrics
    """
    mse = compute_mse(original, reconstructed)
    mae = np.mean(np.abs(original - reconstructed))
    max_error = np.max(np.abs(original - reconstructed))

    # Signal-to-Noise Ratio
    signal_power = np.mean(original**2)
    snr = 10 * np.log10(signal_power / (mse + 1e-10))

    # Cosine similarity
    cos_sim = np.sum(original * reconstructed) / (
        np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-10
    )

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "snr_db": snr,
        "cosine_similarity": cos_sim,
    }


def quantize_decompress_benchmark(quantizer, x: np.ndarray, n_trials: int = 100) -> dict:
    """Benchmark quantization and decompression performance.

    Args:
        quantizer: Quantizer object with quantize() and dequantize() methods
        x: Test data
        n_trials: Number of trials for timing

    Returns:
        Dictionary with performance metrics
    """
    import time

    # Warmup
    _ = quantizer.quantize(x)

    # Benchmark quantization
    start = time.perf_counter()
    for _ in range(n_trials):
        quantized = quantizer.quantize(x)
    quantize_time = (time.perf_counter() - start) / n_trials

    quantized = quantizer.quantize(x)

    # Benchmark dequantization
    start = time.perf_counter()
    for _ in range(n_trials):
        reconstructed = quantizer.dequantize(quantized)
    dequantize_time = (time.perf_counter() - start) / n_trials

    # Compute metrics
    metrics = compute_distortion(x, reconstructed)

    # Compression ratio
    original_bytes = x.nbytes
    compressed = quantizer.compress(x) if hasattr(quantizer, "compress") else None
    if compressed is not None:
        compressed_bytes = len(compressed)
        compression_ratio = original_bytes / compressed_bytes
    else:
        compressed_bytes = None
        compression_ratio = None

    return {
        "quantize_time_ms": quantize_time * 1000,
        "dequantize_time_ms": dequantize_time * 1000,
        "throughput_mbps": (x.nbytes / quantize_time) / (1024 * 1024),
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        **metrics,
    }


def validate_quantization(
    x: np.ndarray, reconstructed: np.ndarray, tolerance: float = 0.01
) -> bool:
    """Validate that quantization meets quality requirements.

    Args:
        x: Original data
        reconstructed: Reconstructed data
        tolerance: Maximum allowed MSE relative to signal variance

    Returns:
        True if validation passes
    """
    metrics = compute_distortion(x, reconstructed)
    signal_variance = np.var(x)

    # Check relative MSE
    relative_mse = metrics["mse"] / (signal_variance + 1e-10)

    # Check cosine similarity
    if metrics["cosine_similarity"] < 0.95:
        return False

    # Check relative error
    return not relative_mse > tolerance
