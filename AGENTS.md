# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-31
**Updated:** 2026-03-31
**Project:** FastVQ
**Type:** Python package

## OVERVIEW

FastVQ is a Python implementation of TurboQuant-style vector quantization for AI model compression. This repository contains:
- **Core Python library** (`turboquant/`) - PolarQuant, QJL, and TurboQuant algorithms
- **FastVQ alias package** (`fastvq/`) - PyPI import and CLI entrypoint aliases
- **Benchmarks** (`benchmarks/`) - Synthetic benchmark runner
- **Notebooks** (`notebooks/`) - Notebook examples
- **Test suite** (`tests/`) - Minimal pytest validation

## STRUCTURE

```
fastvq/
├── turboquant/              # Python library source (12 modules)
│   ├── __init__.py         # Public API exports
│   ├── turboquant.py       # Main algorithm
│   ├── polarquant.py       # PolarQuant implementation
│   ├── qjl.py              # QJL (Quantized JL) transform
│   ├── kv_cache.py         # KV cache integration
│   ├── model_export.py     # GGUF/SafeTensors export
│   ├── cli.py              # CLI entry point
│   └── utils.py            # Utilities
├── fastvq/                  # PyPI import/CLI alias package
├── benchmarks/              # Benchmark scripts
├── notebooks/               # Notebook examples
├── tests/                   # Test suite (minimal)
├── models/                  # Downloaded model artifacts
└── .github/workflows/       # CI/CD
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Core quantization | `turboquant/turboquant.py` | TurboQuant class |
| Polar transform | `turboquant/polarquant.py` | PolarQuant algorithm |
| QJL transform | `turboquant/qjl.py` | 1-bit JL quantization |
| KV cache | `turboquant/kv_cache.py` | Transformer cache compression |
| Model export | `turboquant/model_export.py` | GGUF/SafeTensors support |
| CLI commands | `turboquant/cli.py` | argparse subcommands |
| PyPI alias | `fastvq/` | `fastvq` import and CLI alias |
| Benchmarks | `benchmarks/`, `turboquant/benchmarking.py` | Synthetic benchmark suite |
| Notebooks | `notebooks/` | Notebook walkthroughs |

## CONVENTIONS

### Python (Global)
- **Line length:** 100 (Black, Ruff)
- **Typing:** Strict mypy (`disallow_untyped_defs = true`)
- **Naming:** PascalCase classes, snake_case functions
- **Imports:** Relative for internal (`from .module import`), absolute for third-party

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use `compress()`/`decompress()` - use `quantize()`/`dequantize()` instead
- **AVOID** adding heavy runtime dependencies; put benchmark/notebook-only deps in extras

## UNIQUE STYLES

### Package Name
- PyPI distribution and preferred import: `fastvq`
- Legacy implementation package and CLI alias remain: `turboquant`, `tq`

## COMMANDS

### Python Development
```bash
# Install
pip install -e .                    # Editable install
pip install -e ".[dev]"             # With dev dependencies
pip install -e ".[torch]"           # With PyTorch

# Lint/Format
black .                             # Format code
ruff check .                        # Lint
mypy turboquant/                    # Type check

# Test
pytest                              # Run tests (tests/ minimal)

# CLI
fastvq --help                      # Preferred command
turboquant --help                   # Full command
tq --help                           # Short alias
```

## NOTES

- **Missing tests/:** `pyproject.toml` references `tests/` but only has 1 test file
- **Python 3.8+ required:** Minimum version specified in `pyproject.toml`
- **CI/CD:** Python publishing workflow exists
