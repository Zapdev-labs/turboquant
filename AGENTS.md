# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-31
**Updated:** 2026-03-31
**Project:** TurboQuant Clone
**Type:** Multi-language monorepo (Python + Node.js)

## OVERVIEW

TurboQuant is a production-quality implementation of Google's quantization algorithm for extreme AI model compression (3-bit with near-zero loss). This monorepo contains:
- **Core Python library** (`turboquant/`) - PolarQuant, QJL, and TurboQuant algorithms
- **NPM CLI wrapper** (`npm-package/`) - Global CLI wrapper with interactive TUI
- **Test suite** (`tests/`) - Minimal pytest validation

## STRUCTURE

```
turboquant-clone/
├── turboquant/              # Python library source (12 modules)
│   ├── __init__.py         # Public API exports
│   ├── turboquant.py       # Main algorithm
│   ├── polarquant.py       # PolarQuant implementation
│   ├── qjl.py              # QJL (Quantized JL) transform
│   ├── kv_cache.py         # KV cache integration
│   ├── model_export.py     # GGUF/SafeTensors export
│   ├── cli.py              # CLI entry point
│   └── utils.py            # Utilities
├── npm-package/             # NPM CLI wrapper with TUI
│   ├── bin/                # CLI binaries (turboquant, tq)
│   ├── src/tui.tsx         # Interactive Terminal UI (React)
│   └── package.json        # NPM manifest
├── tests/                   # Test suite (minimal)
├── models/                  # Downloaded model artifacts
└── .github/workflows/       # CI/CD (NPM publish only)
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
| TUI interface | `npm-package/src/tui.tsx` | React-based terminal UI |
| NPM wrapper | `npm-package/bin/turboquant` | Node-to-Python delegation |

## CONVENTIONS

### Python (Global)
- **Line length:** 100 (Black, Ruff)
- **Typing:** Strict mypy (`disallow_untyped_defs = true`)
- **Naming:** PascalCase classes, snake_case functions
- **Imports:** Relative for internal (`from .module import`), absolute for third-party

### JavaScript/TypeScript (Global)
- **Path alias:** `@/*` maps to `./*`
- **Components:** PascalCase in PascalCase folders
- **TUI:** Bun runtime required for interactive mode

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use `compress()`/`decompress()` - use `quantize()`/`dequantize()` instead
- **DO NOT** create `tailwind.config.js` - Tailwind v4 uses CSS-based config
- **NEVER** mix Python and JS dependencies at root level
- **AVOID** adding dependencies to root - keep in respective subprojects

## UNIQUE STYLES

### NPM Wrapper Pattern
- Thin Node.js wrapper that auto-installs Python via pip
- Handles Python detection (`python3`, `python`, `py`)
- Delegates all commands to Python CLI
- **TUI Mode:** Requires Bun runtime; falls back to CLI without it

### Bun-First TUI
- Interactive UI built with React + @opentui/core
- Bun-specific dependencies (@opentui/react)
- Falls back to plain CLI if Bun unavailable

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
turboquant --help                   # Full command
tq --help                           # Short alias
```

### NPM Package Development
```bash
cd npm-package

# Install
npm install                         # Runs install.js postinstall

# Development
bun src/tui.tsx                     # Run TUI directly
npm run tui                         # Same as above

# Building
npm run build                       # Compile TypeScript
npm run test                        # Run validation tests

# Usage
tq                                  # Launch TUI (requires Bun)
turboquant --help                   # CLI mode
```

## NOTES

- **Missing tests/:** `pyproject.toml` references `tests/` but only has 1 test file
- **No root lockfile:** Each JS subproject manages deps independently
- **Python 3.8+ required:** Minimum version specified in `pyproject.toml`
- **Bun for TUI:** Interactive mode requires Bun; CLI works with Node
- **CI/CD:** Only NPM publishing workflow exists (no Python CI)
