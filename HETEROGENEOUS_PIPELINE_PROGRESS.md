# Heterogeneous Pipeline Implementation Progress

## Overview

This document tracks the implementation of true intra-model heterogeneous pipeline parallelism for GoGrid, allowing a single model instance to distribute layers across both CUDA and Metal GPUs simultaneously.

## Completed Work

### 1. Added candle-core Dependencies

**Location**: `crates/scheduler/Cargo.toml:68-86`

Added platform-specific candle-core and candle-nn dependencies to work with tensors directly:
- Metal support on macOS
- CUDA support on Linux/Windows
- Same version pinning as mistralrs (`rev = "7511e510"`)

### 2. Real Tensor-Based Activation Data Structures

**Location**: `crates/scheduler/src/heterogeneous_pipeline.rs:472-487`

Replaced placeholder `Vec<u32>` with actual `candle_core::Tensor`:

```rust
struct Activations {
    tensor: Tensor,
    backend: corpgrid_common::GpuBackend,
}
```

This enables real tensor operations across the pipeline stages.

### 3. Real Cross-Backend Tensor Transfers

**Location**: `crates/scheduler/src/heterogeneous_pipeline.rs:288-331`

Implemented actual CPU-mediated tensor transfers between CUDA and Metal:

```rust
fn transfer_activations_cross_backend(
    &self,
    activations: Activations,
    from: corpgrid_common::GpuBackend,
    to: corpgrid_common::GpuBackend,
    target_stage: &PipelineStage,
) -> Result<Activations>
```

**How it works**:
1. Copy tensor from source GPU (CUDA/Metal) to CPU: `tensor.to_device(&CandleDevice::Cpu)`
2. Copy tensor from CPU to target GPU: `cpu_tensor.to_device(&target_stage.device)`

This is a **real implementation**, not a placeholder. When data flows from a CUDA stage to a Metal stage (or vice versa), the tensor is actually copied through CPU memory.

### 4. Model Architecture Data Structures

**Location**: `crates/scheduler/src/heterogeneous_pipeline.rs:159-189`

Created proper structures for transformer layer weights:

```rust
struct TransformerLayerWeights {
    attention_norm: Option<Tensor>,
    q_proj: Option<Tensor>,
    k_proj: Option<Tensor>,
    v_proj: Option<Tensor>,
    o_proj: Option<Tensor>,
    ffn_norm: Option<Tensor>,
    gate_proj: Option<Tensor>,
    up_proj: Option<Tensor>,
    down_proj: Option<Tensor>,
}

struct ModelEmbeddings {
    token_embedding: Option<Tensor>,
    lm_head: Option<Tensor>,
    final_norm: Option<Tensor>,
}

struct StageModel {
    layers: Vec<TransformerLayerWeights>,
    embeddings: Option<ModelEmbeddings>,
    device: CandleDevice,
    backend: corpgrid_common::GpuBackend,
}
```

These structures are ready to hold actual model weights loaded from safetensors files.

### 5. Model Configuration Loading

**Location**: `crates/scheduler/src/heterogeneous_pipeline.rs:203-250`

The executor now reads model architecture from `config.json`:
- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`

This configuration is used to validate tensor shapes and prepare for layer execution.

### 6. Compilation Verification

The entire heterogeneous pipeline module compiles successfully with only expected warnings about unused fields (these will be used once weight loading and layer execution are implemented).

## What Has Been Actually Implemented

### ✅ Real Implementations (Not Placeholders)

1. **Cross-backend tensor transfers** - Actual GPU→CPU→GPU copies using candle APIs
2. **Tensor data structures** - Real `candle_core::Tensor` types throughout
3. **Device management** - Proper `CandleDevice` creation for CUDA and Metal
4. **Pipeline stage distribution** - Layer ranges correctly calculated and assigned
5. **Configuration parsing** - Model architecture read from config.json

### ⚠️ Framework in Place (Needs Implementation)

1. **Weight loading** - Structures exist, need safetensors file reading
2. **Layer execution** - Forward pass logic needs implementation
3. **Token generation** - Sampling and autoregressive loop needs implementation

## Remaining Work

### Priority 1: Weight Loading from Safetensors

**What needs to be done:**

1. Add `safetensors` crate dependency to `Cargo.toml`
2. Implement `load_stage_model()` function in `heterogeneous_pipeline.rs`:
   ```rust
   async fn load_stage_model(
       model_path: &str,
       stage: &PipelineStage,
   ) -> Result<StageModel>
   ```

3. For each stage:
   - Open safetensors files in `model_path` (e.g., `model.safetensors`, `model-00001-of-00002.safetensors`, etc.)
   - Extract tensors for layers in range `[stage.layer_start, stage.layer_end)`
   - Tensor keys follow pattern: `model.layers.{layer_idx}.{weight_name}`
   - Load tensors directly to target device (CUDA or Metal)

4. Populate `TransformerLayerWeights` for each layer:
   - `self_attn.q_proj.weight` → `q_proj`
   - `self_attn.k_proj.weight` → `k_proj`
   - `self_attn.v_proj.weight` → `v_proj`
   - `self_attn.o_proj.weight` → `o_proj`
   - `input_layernorm.weight` → `attention_norm`
   - `post_attention_layernorm.weight` → `ffn_norm`
   - `mlp.gate_proj.weight` → `gate_proj`
   - `mlp.up_proj.weight` → `up_proj`
   - `mlp.down_proj.weight` → `down_proj`

5. For first stage only, load embeddings:
   - `model.embed_tokens.weight` → `token_embedding`
   - `model.norm.weight` → `final_norm`
   - `lm_head.weight` → `lm_head`

### Priority 2: Transformer Layer Forward Pass

**What needs to be done:**

1. Implement `execute_transformer_layer()` in `heterogeneous_pipeline.rs`:
   ```rust
   fn execute_transformer_layer(
       hidden_states: &Tensor,
       layer: &TransformerLayerWeights,
       layer_idx: usize,
   ) -> Result<Tensor>
   ```

2. Implement standard transformer decoder layer:
   - RMSNorm on input: `normed = rms_norm(hidden_states, layer.attention_norm)`
   - Self-attention:
     - Q = normed @ q_proj.T
     - K = normed @ k_proj.T
     - V = normed @ v_proj.T
     - Apply RoPE (Rotary Position Embedding) to Q and K
     - attention_output = scaled_dot_product_attention(Q, K, V)
     - attention_output = attention_output @ o_proj.T
   - Residual connection: `hidden_states = hidden_states + attention_output`
   - FFN with SwiGLU activation:
     - normed = rms_norm(hidden_states, layer.ffn_norm)
     - gate = normed @ gate_proj.T
     - up = normed @ up_proj.T
     - ffn_output = (silu(gate) * up) @ down_proj.T
   - Residual connection: `hidden_states = hidden_states + ffn_output`

3. Update `execute_stage_layers()` to call this for each layer in the stage

### Priority 3: Token Generation and Sampling

**What needs to be done:**

1. Implement `apply_final_projection()`:
   ```rust
   fn apply_final_projection(
       hidden_states: &Tensor,
       embeddings: &ModelEmbeddings,
   ) -> Result<Tensor>
   ```
   - Apply final RMS norm
   - Project to vocabulary space: `logits = normed @ lm_head.T`

2. Implement sampling function:
   ```rust
   fn sample_token(
       logits: &Tensor,
       temperature: f32,
       top_p: f32,
   ) -> Result<u32>
   ```
   - Apply temperature scaling: `logits / temperature`
   - Apply top-p (nucleus) sampling
   - Sample from resulting distribution

3. Implement autoregressive generation loop in `generate_tokens_from_activations()`:
   - Start with input sequence
   - For each new token:
     - Run full pipeline on current sequence
     - Sample next token from final logits
     - Append to sequence
     - Check for EOS or max_tokens
   - Return generated token IDs

### Priority 4: Integration with mistralrs_backend.rs

**What needs to be done:**

1. Update `MistralRsInferenceBackend::load()` in `mistralrs_backend.rs:127-220`

2. When `BackendType::HeterogeneousPipeline` is detected:
   ```rust
   BackendType::HeterogeneousPipeline => {
       // Create heterogeneous pipeline
       let pipeline = Arc::new(HeterogeneousPipeline::new(devices, model_path)?);

       // Create executor and load weights
       let executor = HeterogeneousPipelineExecutor::new(pipeline, model_path).await?;

       // Store executor instead of mistralrs::Model
       // (requires refactoring MistralRsInferenceBackend to support both)
   }
   ```

3. Refactor `MistralRsInferenceBackend` to hold either:
   - `mistralrs::Model` (for homogeneous cases)
   - `HeterogeneousPipelineExecutor` (for heterogeneous case)

4. Update `generate()` method to dispatch to appropriate backend

## Technical Architecture Summary

### Data Flow

```
Input Tokens (Vec<u32>)
    ↓
Embedding Layer (First Stage Only)
    ↓
Tensor [batch, seq_len, hidden_size]
    ↓
┌──────────────────────────────────────┐
│ Stage 1: Layers 0-15 on CUDA GPU 0  │
│   - Execute transformer layers       │
│   - Output: Tensor on CUDA           │
└──────────────────────────────────────┘
    ↓
CPU Transfer (CUDA → CPU → Metal)
    ↓
┌──────────────────────────────────────┐
│ Stage 2: Layers 16-31 on Metal GPU 0│
│   - Execute transformer layers       │
│   - Output: Tensor on Metal          │
└──────────────────────────────────────┘
    ↓
Final Layer Norm + LM Head
    ↓
Logits [batch, seq_len, vocab_size]
    ↓
Sampling (temperature, top_p)
    ↓
Output Tokens (Vec<u32>)
```

### Key Differences from Shortcuts

**Shortcut approach** (what we were doing before):
- Use only one backend per model instance
- Log that other devices are "available but not used"
- Load separate model instances on each backend type

**True heterogeneous approach** (what we're implementing now):
- Single model instance spans both CUDA and Metal
- Layers physically distributed across different backend types
- Real tensor transfers between backends via CPU
- Seamless pipeline execution across heterogeneous devices

## Testing Strategy

Once implementation is complete:

1. **Unit Tests**:
   - Test tensor transfers between CUDA and Metal
   - Verify layer forward pass correctness
   - Test sampling with different temperature/top_p values

2. **Integration Tests**:
   - Load TinyLlama-1.1B (22 layers) across 1 CUDA + 1 Metal device
   - Verify identical output to homogeneous execution
   - Measure cross-backend transfer overhead

3. **Performance Benchmarks**:
   - Compare to homogeneous CUDA-only execution
   - Compare to homogeneous Metal-only execution
   - Measure speedup from utilizing both device types

## Current Status: Compilation Success

The implementation compiles cleanly:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 49.72s
```

Warnings about unused fields are expected and will disappear once weight loading and layer execution are implemented.

## Next Steps

The recommended order of implementation:

1. **Add safetensors dependency** and implement weight loading (Priority 1)
2. **Implement transformer layer forward pass** (Priority 2)
3. **Implement token generation and sampling** (Priority 3)
4. **Integrate with mistralrs_backend.rs** (Priority 4)
5. **Test with TinyLlama-1.1B on mixed CUDA + Metal devices**

## Key Files

- `crates/scheduler/Cargo.toml:68-86` - candle dependencies
- `crates/scheduler/src/heterogeneous_pipeline.rs` - Main implementation
- `crates/scheduler/src/mistralrs_backend.rs:127-220` - Integration point
- `crates/scheduler/src/lib.rs:14` - Module export

## References

- Transformer architecture: Llama/Mistral papers
- Candle tensor library: https://github.com/huggingface/candle
- Safetensors format: https://github.com/huggingface/safetensors
- RoPE embeddings: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- SwiGLU activation: "GLU Variants Improve Transformer"
