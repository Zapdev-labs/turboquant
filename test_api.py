#!/usr/bin/env python3
"""
Quick test of the TurboQuant API Server
"""

import subprocess
import time
import sys


def test_api():
    print("=" * 70)
    print("TURBOQUANT API QUICK TEST")
    print("=" * 70)

    # Start server as subprocess
    print("\n🚀 Starting API server...")
    process = subprocess.Popen(
        [sys.executable, "api_server.py", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for startup
    time.sleep(5)

    try:
        import requests

        # Test health
        print("\n🏥 Testing health endpoint...")
        r = requests.get("http://localhost:8000/health")
        print(f"   Status: {r.json()}")

        # Test models
        print("\n📋 Testing models endpoint...")
        r = requests.get("http://localhost:8000/v1/models")
        print(f"   Models: {r.json()}")

        # Test TurboQuant stats
        print("\n📊 Testing TurboQuant stats...")
        r = requests.get("http://localhost:8000/v1/turboquant/stats")
        stats = r.json()
        print(f"   Bit width: {stats['bit_width']}-bit")
        print(f"   Compression: {stats['compression_ratio']:.1f}x")

        # Test chat completion
        print("\n💬 Testing chat completion...")
        r = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "gemma-4-31b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50,
                "turboquant_bits": 3,
            },
        )
        response = r.json()
        content = response["choices"][0]["message"]["content"]
        print(f"   Response: {content[:100]}...")

        print("\n" + "=" * 70)
        print("✅ API TEST PASSED!")
        print("=" * 70)
        print("\nThe API server is working correctly!")
        print("\nTo use it:")
        print("  1. Start the server: python api_server.py")
        print("  2. Open http://localhost:8000/ui in your browser")
        print("  3. Or use the client: python client.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    finally:
        # Cleanup
        process.terminate()
        process.wait()

    return 0


if __name__ == "__main__":
    sys.exit(test_api())
