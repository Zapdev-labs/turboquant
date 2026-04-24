import numpy as np

import fastvq
from turboquant import TurboQuant, TurboQuantConfig


def test_fastvq_import_alias_exports_core_api():
    assert fastvq.TurboQuant is TurboQuant
    assert hasattr(fastvq, "run_benchmark_suite")


def test_quantize_dequantize_arbitrary_trailing_dimension():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, 3, 192)).astype(np.float32)
    quantizer = TurboQuant(TurboQuantConfig(bit_width=3, block_size=128))

    quantized = quantizer.quantize(data)
    reconstructed = quantizer.dequantize(quantized)

    assert reconstructed.shape == data.shape
    assert np.isfinite(reconstructed).all()
    assert quantized["padded_dim"] == 256
    assert quantized["blocks_per_vector"] == 2


def test_compress_decompress_roundtrip_shape():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((8, 64)).astype(np.float32)
    quantizer = TurboQuant(TurboQuantConfig(bit_width=4, block_size=64))

    payload = quantizer.compress(data)
    reconstructed = TurboQuant().decompress(payload)

    assert reconstructed.shape == data.shape
    assert np.isfinite(reconstructed).all()
