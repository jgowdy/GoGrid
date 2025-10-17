# OpenAI-Compatible API Usage

CorpGrid now provides an OpenAI-compatible REST API for LLM inference at port 8000 (configurable via `OPENAI_API_ADDR`).

## Endpoints

### List Models
```bash
curl http://localhost:8000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "model-uuid-here",
      "object": "model",
      "created": 1234567890,
      "owned_by": "corpgrid"
    }
  ]
}
```

### Get Model Details
```bash
curl http://localhost:8000/v1/models/{model_id}
```

### Text Completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-uuid-here",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

Response:
```json
{
  "id": "cmpl-uuid",
  "object": "text_completion",
  "created": 1234567890,
  "model": "model-uuid-here",
  "choices": [
    {
      "text": " there was a...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 96,
    "total_tokens": 100
  }
}
```

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-uuid-here",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-uuid-here",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28
  }
}
```

## Using with OpenAI Python Client

```python
from openai import OpenAI

# Point to CorpGrid instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # CorpGrid doesn't require auth yet
)

# Chat completion
response = client.chat.completions.create(
    model="model-uuid-here",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)

# Text completion
response = client.completions.create(
    model="model-uuid-here",
    prompt="The meaning of life is",
    max_tokens=50
)

print(response.choices[0].text)

# List models
models = client.models.list()
for model in models.data:
    print(f"Model: {model.id}")
```

## Using with LangChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="model-uuid-here"
)

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Write a Python function to calculate fibonacci numbers.")
]

response = llm(messages)
print(response.content)
```

## Loading a Model

Before using the API, you need to load a model via gRPC:

```python
import grpc
from corpgrid_proto import scheduler_pb2, scheduler_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = scheduler_pb2_grpc.ModelHostingStub(channel)

# Load model
response = stub.LoadModel(scheduler_pb2.LoadModelRequest(
    model_path="/path/to/model",
    precision="fp16"
))

print(f"Model loaded: {response.model_id}")

# Now use this model_id in OpenAI API requests
```

## Server Configuration

Configure ports via environment variables:

```bash
export OPENAI_API_ADDR=0.0.0.0:8000  # OpenAI API (default)
export BIND_ADDR=0.0.0.0:50051        # gRPC (default)
export WEB_UI_ADDR=0.0.0.0:8080       # Web UI (default)
export METRICS_ADDR=0.0.0.0:9090      # Prometheus metrics (default)

./corpgrid-scheduler
```

## Architecture

```
┌──────────────────────────────────────────────┐
│  OpenAI API (port 8000)                      │
│  - /v1/models                                │
│  - /v1/completions                           │
│  - /v1/chat/completions                      │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  ModelHostingService                         │
│  - Automatic resource allocation             │
│  - Multi-GPU (CUDA + Metal) support          │
│  - Request queuing                           │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Inference Backends                          │
│  - HomogeneousCuda                           │
│  - HomogeneousMetal                          │
│  - HeterogeneousPipeline (CUDA+Metal)        │
└──────────────────────────────────────────────┘
```

## Notes

- **Tokenization**: Currently uses simple placeholder tokenization. Production deployments should integrate proper tokenizers (tiktoken, sentencepiece, etc.)
- **Streaming**: Not yet implemented (returns error if `stream: true`)
- **Authentication**: Not yet implemented
- **Rate Limiting**: Not yet implemented
- **Model Selection**: Use the model UUID returned from `LoadModel` gRPC call
