#!/bin/bash
# TurboQuant API Server Launcher
# Easy startup script for the TurboQuant API

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    🚀 TURBOQUANT API SERVER                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Default settings
HOST="${TURBOQUANT_HOST:-0.0.0.0}"
PORT="${TURBOQUANT_PORT:-8000}"
MODEL="${TURBOQUANT_MODEL:-models/gemma-4-31b-it-Q2_K.gguf}"
BITS="${TURBOQUANT_BITS:-3}"
CONTEXT="${TURBOQUANT_CONTEXT:-4096}"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "⚠️  Model not found: $MODEL"
    echo ""
    echo "Options:"
    echo "1. Download a model:"
    echo "   turboquant download freakyskittle/gemma-4-9b-it-Q2_K"
    echo ""
    echo "2. Set TURBOQUANT_MODEL to the path of your GGUF model:"
    echo "   export TURBOQUANT_MODEL=/path/to/your/model.gguf"
    echo ""
    echo "3. Continue in demo mode (simulated responses)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV}" ]; then
    if [ -d ".venv" ]; then
        echo "🔧 Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "⚠️  No virtual environment found. Run: uv venv && uv pip install -e ."
        exit 1
    fi
fi

echo "🚀 Starting TurboQuant API Server..."
echo ""
echo "Configuration:"
echo "  Host:        $HOST"
echo "  Port:        $PORT"
echo "  Model:       $MODEL"
echo "  TurboQuant:  $BITS-bit KV cache"
echo "  Context:     $CONTEXT tokens"
echo ""

# Run the server
python api_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --turboquant-bits "$BITS" \
    --context-length "$CONTEXT"
