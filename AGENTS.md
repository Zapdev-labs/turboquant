# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-29
**Project:** TurboQuant Clone
**Type:** Multi-language monorepo (Python + Node.js)

## OVERVIEW

TurboQuant is a production-quality implementation of Google's quantization algorithm for extreme AI model compression (3-bit with near-zero loss). This monorepo contains:
- **Core Python library** (`turboquant/`) - PolarQuant, QJL, and TurboQuant algorithms
- **Electron desktop app** (`electron-app/`) - Next.js 15 + React 19 + Electron 33 desktop UI
- **NPM CLI wrapper** (`npm-package/`) - Global CLI wrapper for Python library

## STRUCTURE

```
turboquant-clone/
├── turboquant/              # Python library source
│   ├── __init__.py         # Public API exports
│   ├── turboquant.py       # Main algorithm
│   ├── polarquant.py       # PolarQuant implementation
│   ├── qjl.py              # QJL (Quantized JL) transform
│   ├── kv_cache.py         # KV cache integration
│   ├── model_export.py     # GGUF/SafeTensors export
│   ├── cli.py              # CLI entry point
│   └── utils.py            # Utilities
├── electron-app/            # Desktop application
│   ├── app/                # Next.js App Router
│   ├── components/         # React components
│   ├── electron/           # Electron main process
│   └── stores/             # Zustand state
├── npm-package/             # NPM CLI wrapper
│   └── bin/                # CLI binaries
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
| Desktop UI | `electron-app/app/` | Next.js pages |
| Electron main | `electron-app/electron/main.js` | Main process |
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
- **Styling:** Tailwind v4 with CSS-based config

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use `compress()`/`decompress()` - use `quantize()`/`dequantize()` instead
- **DO NOT** create `tailwind.config.js` - Tailwind v4 uses CSS-based config
- **NEVER** mix Python and JS dependencies at root level
- **AVOID** adding dependencies to root - keep in respective subprojects

## UNIQUE STYLES

### Tailwind v4 (Electron App)
- No `tailwind.config.js` - configuration in `globals.css` via `@theme`
- Uses `@tailwindcss/postcss` instead of traditional setup
- Custom CSS variables for theming

### NPM Wrapper Pattern
- Thin Node.js wrapper that auto-installs Python via pip
- Handles Python detection (`python3`, `python`, `py`)
- Delegates all commands to Python CLI

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
pytest                              # Run tests (tests/ doesn't exist yet)

# CLI
turboquant --help                   # Full command
tq --help                           # Short alias
```

### Electron App Development
```bash
cd electron-app

# Development
bun run dev                         # Concurrent Next.js + Electron
bun run dev:next                    # Next.js dev server only
bun run dev:electron                # Electron dev (waits for Next.js)

# Building
bun run build                       # Build Next.js static export
bun run build:electron              # Build + package Electron app
bun run dist                        # Full distribution

# Quality
bun run lint                        # ESLint
bun run type-check                  # TypeScript check
```

### NPM Package
```bash
cd npm-package
npm install                         # Runs install.js postinstall
npm test                            # Runs test.js
```

## NOTES

- **Missing tests/:** `pyproject.toml` references `tests/` but directory doesn't exist
- **No root lockfile:** Each JS subproject manages deps independently
- **Dual Next.js configs:** Both `next.config.ts` and `next.config.js` exist in electron-app
- **Python 3.8+ required:** Minimum version specified in `pyproject.toml`
- **Electron targets:** macOS (DMG), Windows (NSIS), Linux (AppImage)
