# TURBOQUANT PACKAGE

Python library for extreme AI model quantization.

## OVERVIEW

Core algorithms: TurboQuant, PolarQuant, QJL. Implements Google's research for 3-bit compression with near-zero accuracy loss. Includes KV cache integration for transformers and GGUF/SafeTensors export.

## STRUCTURE

```
turboquant/
├── __init__.py          # Public exports
├── turboquant.py        # Main TurboQuant class
├── polarquant.py        # PolarQuant algorithm
├── qjl.py               # QJL (1-bit JL transform)
├── kv_cache.py          # KV cache compression
├── model_export.py      # GGUF/SafeTensors export
├── cli.py               # Command-line interface
├── transforms.py        # Walsh-Hadamard & polar transforms
├── codebooks.py         # Lloyd-Max codebook generation
└── utils.py             # Bit-packing utilities
```

## WHERE TO LOOK

| Algorithm | File | Key Class |
|-----------|------|-----------|
| TurboQuant | `turboquant.py` | `TurboQuant`, `TurboQuantConfig` |
| PolarQuant | `polarquant.py` | `PolarQuant` |
| QJL | `qjl.py` | `QJL` |
| KV Cache | `kv_cache.py` | `KVCacheCompressor` |
| Model Export | `model_export.py` | `TurboQuantGGUFLoader` |
| CLI | `cli.py` | `main()` entry |

## CONVENTIONS

### Code Style
- **Line length:** 100 (enforced by Black and Ruff)
- **Typing:** Strict mypy - all functions must have type annotations
- **Formatting:** Black with Python 3.8+ target

### Naming
- **Classes:** PascalCase (`TurboQuant`, `TurboQuantConfig`)
- **Functions:** snake_case (`compute_mse`, `quantize_decompress_benchmark`)
- **Private:** Leading underscore (`_init_components`, `_generate_centroids`)
- **Constants:** UPPER_SNAKE_CASE

### Imports
```python
# Internal - relative
from .turboquant import TurboQuant
from .utils import compute_mse

# External - absolute
import numpy as np
from typing import Optional, Tuple
```

## ANTI-PATTERNS

- **DO NOT** use `compress()`/`decompress()` - API is `quantize()`/`dequantize()`
- **NEVER** skip type annotations - mypy enforces `disallow_untyped_defs`
- **AVOID** importing from `__init__.py` internally - use relative imports
- **DON'T** exceed 100 character lines

## COMMANDS

```bash
# Development
pip install -e ".[dev]"             # Install with dev deps

# Code Quality
black .                             # Format
ruff check .                        # Lint
mypy turboquant/                    # Type check

# CLI Usage
turboquant compress input.npy out.tq --bits 3
turboquant kv-analyze --model-size 70b
tq benchmark input.npy --bits 3,4
```

## CONFIGURATION

From `pyproject.toml`:
- **Ruff:** Selects E, F, I, N, W, UP, B, C4, SIM rules
- **MyPy:** Strict mode, Python 3.8 target
- **Black:** 100 char line length
- **Pytest:** Test pattern `test_*.py` in `tests/` (not yet created)

## DEPENDENCIES

**Runtime:**
- `numpy>=1.20.0`
- `typing-extensions>=4.0.0` (Python <3.10)

**Optional:**
- `torch>=2.0.0` (PyTorch integration)
- `pytest>=7.0.0` (testing)
- `black>=22.0.0`, `ruff>=0.1.0`, `mypy>=0.950` (dev)
