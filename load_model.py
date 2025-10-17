#!/usr/bin/env python3
"""
Simple gRPC client to load a model via the ModelHosting service.
"""
import grpc
import sys
import os

# Add the proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crates', 'proto'))

# Compile proto on the fly
from grpc_tools import protoc

proto_file = "crates/proto/proto/scheduler.proto"
protoc.main([
    'grpc_tools.protoc',
    '-I./crates/proto/proto',
    '--python_out=.',
    '--grpc_python_out=.',
    proto_file
])

# Import generated code
import scheduler_pb2
import scheduler_pb2_grpc

def load_model(server_url, model_path, precision="fp16"):
    """Load a model via gRPC."""
    print(f"Connecting to {server_url}...")
    channel = grpc.insecure_channel(server_url)
    stub = scheduler_pb2_grpc.ModelHostingStub(channel)

    print(f"Loading model from {model_path} with precision {precision}...")
    request = scheduler_pb2.LoadModelRequest(
        model_path=model_path,
        precision=precision,
        download_from_hub=False,
        hub_revision=""
    )

    response = stub.LoadModel(request)

    if response.success:
        print(f"✓ Model loaded successfully!")
        print(f"  Model ID: {response.model_id}")
        if response.allocation:
            print(f"  Devices: {response.allocation.num_devices}")
            print(f"  Device names: {', '.join(response.allocation.device_names)}")
            print(f"  Backend type: {response.allocation.backend_type}")
    else:
        print(f"✗ Failed to load model: {response.error_message}")
        sys.exit(1)

    channel.close()
    return response.model_id

if __name__ == "__main__":
    server_url = "localhost:50051"
    model_path = os.path.expanduser("~/GoGrid/models/TinyLlama-1.1B-Chat-v1.0")

    load_model(server_url, model_path)
