# TurboQuant API Usage Guide

## 🚀 Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install fastvq

# From source
git clone https://github.com/turboquant/turboquant
cd turboquant
pip install -e .
```

**Note:** Package is `fastvq` on PyPI, imports as `turboquant`.

### 1. Start the API Server

```bash
# Using the startup script
./start_api.sh

# Or manually with options
python api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model models/gemma-4-31b-it-Q2_K.gguf \
    --turboquant-bits 3 \
    --context-length 4096
```

### 2. Access the API

Once running, you can access:

| Endpoint | URL | Description |
|----------|-----|-------------|
| Web UI | http://localhost:8000/ui | Interactive chat interface |
| API Docs | http://localhost:8000/docs | Auto-generated Swagger docs |
| Health | http://localhost:8000/health | Server health check |
| Chat | POST /v1/chat/completions | OpenAI-compatible chat |

### 3. Use the API

#### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# TurboQuant stats
curl http://localhost:8000/v1/turboquant/stats

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "turboquant_bits": 3,
    "stream": false
  }'

# Streaming response
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 100,
    "stream": true
  }'
```

#### Using the Python Client:

```python
from client import TurboQuantClient

# Create client
client = TurboQuantClient("http://localhost:8000")

# Check health
health = client.health()
print(f"Status: {health['status']}")

# Get TurboQuant stats
stats = client.get_turboquant_stats()
print(f"Compression: {stats['compression_ratio']:.1f}x")

# Simple generation
response = client.generate(
    prompt="What is quantum computing?",
    max_tokens=100,
    turboquant_bits=3
)
print(response)

# Chat completion with history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain neural networks"}
]

response = client.chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.7,
    stream=False
)

print(response["choices"][0]["message"]["content"])
```

## 🔧 Configuration

### Environment Variables

```bash
export TURBOQUANT_HOST=0.0.0.0        # Server host
export TURBOQUANT_PORT=8000           # Server port
export TURBOQUANT_MODEL=models/gemma-4-31b-it-Q2_K.gguf  # Model path
export TURBOQUANT_BITS=3              # TurboQuant bit-width (2, 3, or 4)
export TURBOQUANT_CONTEXT=4096        # Context length
```

### Command Line Options

```bash
python api_server.py --help

# Options:
#   --host              Host to bind to (default: 0.0.0.0)
#   --port              Port to bind to (default: 8000)
#   --model             Path to GGUF model
#   --turboquant-bits   TurboQuant bit-width: 2, 3, or 4
#   --context-length    Context length in tokens
#   --gpu-layers        Number of GPU layers (0 for CPU)
```

## 📊 OpenAI Compatibility

The API is fully compatible with OpenAI's API format:

```python
import openai

# Point to your local TurboQuant server
client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Use exactly like OpenAI
response = client.chat.completions.create(
    model="gemma-4-31b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 🌐 Network Access

To make the server accessible from other machines:

```bash
# Bind to all interfaces
python api_server.py --host 0.0.0.0 --port 8000

# Now others can connect via your IP:
# http://your-ip:8000
```

### Using ngrok for public access:

```bash
# Install ngrok
pip install pyngrok

# Create tunnel
ngrok http 8000

# Share the https URL with others
```

## 🐳 Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11

WORKDIR /app
COPY . .
RUN pip install -e ".[torch]"
RUN pip install fastapi uvicorn

EXPOSE 8000

CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t turboquant-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models turboquant-api
```

## 📈 Monitoring

### TurboQuant Stats Endpoint

```bash
# Get real-time compression stats
curl http://localhost:8000/v1/turboquant/stats

# Response:
{
  "enabled": true,
  "bit_width": 3,
  "compression_ratio": 5.33,
  "memory_saved_gb": 2.45,
  "kv_cache_entries": 1024,
  "avg_compress_time_ms": 45.2,
  "avg_decompress_time_ms": 22.1
}
```

## 🔒 Security (Production)

For production deployment:

```bash
# 1. Use a reverse proxy (nginx/traefik)
# 2. Enable HTTPS
# 3. Add API key authentication
# 4. Rate limiting

# Example with API key
export TURBOQUANT_API_KEY=your-secret-key

# Then in client:
client = TurboQuantClient(
    "http://localhost:8000",
    api_key="your-secret-key"
)
```

## 🎯 Use Cases

### 1. Local AI Assistant
```bash
# Start server
./start_api.sh

# Open http://localhost:8000/ui in browser
# Start chatting!
```

### 2. Integration with Other Apps
```python
# Any app that uses OpenAI API can use TurboQuant
# Just change the base_url!
```

### 3. Multi-User Setup
```bash
# Run on a server
python api_server.py --host 0.0.0.0 --port 8000

# Users connect via LAN/WiFi
# http://server-ip:8000
```

## 🐛 Troubleshooting

### Model not found
```bash
# Download model first
turboquant download freakyskittle/gemma-4-9b-it-Q2_K

# Or specify path
export TURBOQUANT_MODEL=/path/to/model.gguf
```

### Port already in use
```bash
# Use different port
python api_server.py --port 8001
```

### Out of memory
```bash
# Use smaller context
python api_server.py --context-length 2048

# Or use more aggressive compression
python api_server.py --turboquant-bits 2
```

## 📝 API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/ui` | Web interface |
| GET | `/v1/models` | List models |
| POST | `/v1/chat/completions` | Chat completion |
| GET | `/v1/turboquant/stats` | Compression stats |

### Request Format

```json
{
  "model": "gemma-4-31b",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "turboquant_bits": 3
}
```

### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemma-4-31b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

---

**Ready to go!** Start the server with `./start_api.sh` and open http://localhost:8000/ui 🚀
