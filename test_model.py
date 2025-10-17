#!/usr/bin/env python3
"""
Test script to load a model via gRPC and test inference via OpenAI API
"""

import grpc
import sys
import os

# Add proto generated files to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crates/proto/python'))

import scheduler_pb2
import scheduler_pb2_grpc
import requests
import json

def load_model_via_grpc(model_path: str, precision: str = "fp16"):
    """Load a model via the gRPC ModelHosting service"""
    channel = grpc.insecure_channel('localhost:50051')
    stub = scheduler_pb2_grpc.ModelHostingStub(channel)

    request = scheduler_pb2.LoadModelRequest(
        model_path=model_path,
        precision=precision,
        download_from_hub=False
    )

    print(f"Loading model from {model_path}...")
    response = stub.LoadModel(request)

    if response.success:
        print(f"✓ Model loaded successfully!")
        print(f"  Model ID: {response.model_id}")
        if response.allocation:
            print(f"  Devices: {response.allocation.num_devices}")
            print(f"  Device names: {', '.join(response.allocation.device_names)}")
            print(f"  Backend type: {response.allocation.backend_type}")
        return response.model_id
    else:
        print(f"✗ Failed to load model: {response.error_message}")
        return None

def test_chat_completion(model_id: str, api_key: str):
    """Test chat completion via OpenAI API"""
    url = "http://localhost:8000/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hello! What is 2+2?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    print(f"\nSending chat completion request...")
    print(f"  Prompt: {data['messages'][0]['content']}")

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Response received!")
        print(f"  Model: {result['model']}")
        print(f"  Response: {result['choices'][0]['message']['content']}")
        print(f"  Tokens: {result['usage']['total_tokens']} total ({result['usage']['prompt_tokens']} prompt + {result['usage']['completion_tokens']} completion)")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def main():
    # Configuration
    model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0"
    api_key = "sk-test-1234567890abcdef"

    print("=" * 60)
    print("GoGrid Model Loading and Inference Test")
    print("=" * 60)

    # Step 1: Load model via gRPC
    model_id = load_model_via_grpc(model_path)

    if not model_id:
        print("\nTest failed: Could not load model")
        return 1

    # Step 2: Test chat completion
    success = test_chat_completion(model_id, api_key)

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed")
    print("=" * 60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
