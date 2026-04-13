#!/usr/bin/env python3
"""
TurboQuant Client
Simple Python client for the TurboQuant API.
"""

import requests
import json
from typing import Optional, List, Dict, Any, Generator


class TurboQuantClient:
    """Client for interacting with the TurboQuant API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def health(self) -> Dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
        response.raise_for_status()
        return response.json().get("data", [])

    def get_turboquant_stats(self) -> Dict[str, Any]:
        """Get TurboQuant compression statistics."""
        response = requests.get(f"{self.base_url}/v1/turboquant/stats", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemma-4-31b",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        turboquant_bits: int = 3,
    ) -> Dict[str, Any]:
        """Send a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "turboquant_bits": turboquant_bits,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._handle_stream(response)
        else:
            return response.json()

    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        turboquant_bits: int = 3,
    ) -> str:
        """Simple generate method (non-chat)."""
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            turboquant_bits=turboquant_bits,
        )

        if stream:
            return "".join(response)
        else:
            return response["choices"][0]["message"]["content"]


def demo():
    """Demo the TurboQuant client."""
    import time

    print("=" * 70)
    print("TURBOQUANT CLIENT DEMO")
    print("=" * 70)

    # Create client
    client = TurboQuantClient("http://localhost:8000")

    # Check health
    print("\n🏥 Checking API health...")
    try:
        health = client.health()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   TurboQuant enabled: {health['turboquant_enabled']}")
    except Exception as e:
        print(f"   ❌ API not available: {e}")
        print("   Start the API server with: python api_server.py")
        return 1

    # List models
    print("\n📋 Available models:")
    models = client.list_models()
    for model in models:
        status = "✅" if model["ready"] else "⏳"
        print(f"   {status} {model['id']} (owned by: {model['owned_by']})")

    # Get TurboQuant stats
    print("\n📊 TurboQuant stats:")
    try:
        stats = client.get_turboquant_stats()
        print(f"   Bit width: {stats['bit_width']}-bit")
        print(f"   Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"   Memory saved: {stats['memory_saved_gb']:.2f} GB")
        print(f"   KV cache entries: {stats['kv_cache_entries']}")
    except Exception as e:
        print(f"   ⚠️  Could not get stats: {e}")

    # Chat completion (non-streaming)
    print("\n💬 Chat completion (non-streaming):")
    prompt = "Explain quantum computing in one sentence."
    print(f"   Prompt: {prompt}")

    start = time.time()
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            stream=False,
            turboquant_bits=3,
        )
        content = response["choices"][0]["message"]["content"]
        elapsed = time.time() - start

        print(f"   Response: {content[:100]}...")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Tokens: {response['usage']['completion_tokens']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Chat completion (streaming)
    print("\n💬 Chat completion (streaming):")
    prompt = "What is the capital of France?"
    print(f"   Prompt: {prompt}")

    start = time.time()
    try:
        print("   Response: ", end="", flush=True)
        full_response = ""
        for chunk in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            stream=True,
            turboquant_bits=3,
        ):
            print(chunk, end="", flush=True)
            full_response += chunk

        elapsed = time.time() - start
        print(f"\n   Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(demo())
