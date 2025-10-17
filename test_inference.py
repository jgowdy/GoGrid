#!/usr/bin/env python3
"""
Test inference with the loaded model via gRPC.
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

def test_inference(server_url, model_id, prompt="Hello, I am"):
    """Test inference with a loaded model."""
    print(f"Connecting to {server_url}...")
    channel = grpc.insecure_channel(server_url)
    stub = scheduler_pb2_grpc.ModelHostingStub(channel)

    print(f"Testing inference with model {model_id}...")
    print(f"Prompt: \"{prompt}\"")

    # Encode prompt to tokens using the tokenizer
    from tokenizers import Tokenizer
    tokenizer_path = os.path.expanduser("~/GoGrid/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    encoding = tokenizer.encode(prompt)
    input_tokens = encoding.ids

    print(f"Input tokens ({len(input_tokens)}): {input_tokens[:10]}..." if len(input_tokens) > 10 else f"Input tokens: {input_tokens}")

    request = scheduler_pb2.InferRequest(
        model_id=model_id,
        input_tokens=input_tokens,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9
    )

    try:
        response = stub.Infer(request)

        if response.success:
            print(f"\n✓ Inference successful!")

            # Decode output tokens
            output_text = tokenizer.decode(response.output_tokens)
            print(f"Generated text:\n{output_text}")
            print(f"\nOutput tokens: {len(response.output_tokens)}")
            print(f"New tokens: {len(response.output_tokens) - len(input_tokens)}")
            print(f"Generation time: {response.generation_time_ms}ms")
        else:
            print(f"✗ Inference failed: {response.error_message}")
            sys.exit(1)
    except grpc.RpcError as e:
        print(f"✗ RPC error: {e.code()}: {e.details()}")
        sys.exit(1)

    channel.close()

if __name__ == "__main__":
    server_url = "localhost:50051"

    # Use the model ID from the previous load
    # You can also pass this as a command line argument
    model_id = sys.argv[1] if len(sys.argv) > 1 else "d2cb8f20-b49b-4db1-a08f-a3fa5f543993"

    test_inference(server_url, model_id)
