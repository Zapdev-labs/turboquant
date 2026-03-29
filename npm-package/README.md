# TurboQuant NPM Package

CLI wrapper for TurboQuant Python library - install globally to use `turboquant` command from anywhere.

## Installation

### Option 1: Global Install (Recommended)
```bash
npm install -g turboquant
```

Then use anywhere:
```bash
turboquant --help
turboquant kv-analyze --model-size 70b
```

### Option 2: NPX (No Install)
```bash
npx turboquant kv-analyze --model-size 70b
```

### Option 3: From Source
```bash
cd /path/to/turboquant-clone/npm-package
npm install -g .
```

## Requirements

- Node.js 14+
- Python 3.8+ (auto-installed if missing)

## Commands

```bash
# Compression
turboquant compress input.npy output.tq --bits 3
turboquant decompress output.tq reconstructed.npy
turboquant quick input.npy

# Analysis
turboquant benchmark input.npy --bits 3,4
turboquant kv-analyze --model-size 70b --seq-len 100000

# Model Management
turboquant download TheBloke/Llama-2-7B-GPTQ --bits 4
turboquant list-models
```

## Shortcuts

```bash
tq --help              # Same as turboquant
tq quick input.npy     # Quick compression
```

## Troubleshooting

### "Python not found"
Install Python 3.8+ first:
```bash
# macOS
brew install python

# Ubuntu/Debian
sudo apt install python3 python3-pip

# Windows
# Download from python.org
```

### "Permission denied"
Use with sudo (Unix) or run as Administrator (Windows):
```bash
sudo npm install -g turboquant
```

## What This Does

This NPM package is a thin wrapper that:
1. Checks for Python 3.8+
2. Installs TurboQuant via pip (if not already installed)
3. Delegates all commands to the Python CLI

The actual quantization algorithms run in Python for performance.

## Links

- GitHub: https://github.com/zapdev-labs/turboquant
- Issues: https://github.com/zapdev-labs/turboquant/issues
