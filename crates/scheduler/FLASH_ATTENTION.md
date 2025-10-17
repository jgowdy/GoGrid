# Flash Attention Integration - Architecture & Implementation Plan

**Status:** Phase 1 Complete (Foundation), Phase 2-4 In Progress

**Completion Estimate:** 4-6 weeks for full implementation

## Overview

Flash Attention is an optimized attention algorithm that reduces memory usage from O(N²) to O(N) and provides 2-4x speedup for long sequences (>512 tokens). This document outlines the architecture for integrating Flash Attention v2 into the heterogeneous pipeline.

## Current Implementation Status

### ✅ Phase 1: Foundation (Completed)
- Added `flash-attention` feature flag in Cargo.toml:8-13
- Added `candle-flash-attn` v0.9 dependency for Linux/Windows (CUDA only)
- Verified build compatibility (cargo check passes)
- Created architecture documentation

### 🔄 Phase 2: CUDA Integration (In Progress)
- [ ] Wrap `candle-flash-attn::flash_attn` function
- [ ] Create `flash_attention_cuda()` function in heterogeneous_pipeline.rs
- [ ] Add conditional compilation with `#[cfg(feature = "flash-attention")]`
- [ ] Update `grouped_query_attention()` to use Flash Attention when enabled
- [ ] Handle KV cache management with Flash Attention

### ⏳ Phase 3: Metal Fallback (Pending)
- [ ] Research Metal Performance Shaders for Flash Attention
- [ ] Either: Implement custom Metal kernel, or
- [ ] Use standard attention with optimization hints
- [ ] Create `flash_attention_metal()` fallback function

### ⏳ Phase 4: Testing & Validation (Pending)
- [ ] Create benchmark comparing standard vs Flash Attention
- [ ] Validate memory usage reduction
- [ ] Test with various sequence lengths (128, 512, 1024, 2048+ tokens)
- [ ] Measure speedup on CUDA hardware
- [ ] Update documentation with benchmarks

## Technical Architecture

### Flash Attention Algorithm

Flash Attention v2 improves upon standard attention through:

1. **Tiling**: Breaks attention computation into blocks that fit in SRAM
2. **Recomputation**: Recomputes attention in backward pass (saves memory)
3. **Parallelization**: Optimizes for modern GPU architectures
4. **Memory Efficiency**: O(N) memory vs O(N²) for standard attention

**Standard Attention Memory:**
```
Attention Matrix: [batch, heads, seq_len, seq_len] = O(N²)
```

**Flash Attention Memory:**
```
Tiles: [batch, heads, block_size, block_size] = O(1) per tile
Total: O(N) across all tiles
```

### Integration Points

The main integration point is the `grouped_query_attention()` function in `heterogeneous_pipeline.rs:947`.

**Current Flow:**
```rust
fn grouped_query_attention(
    q, k, v,              // Query, Key, Value tensors
    mask,                 // Causal mask
    kv_cache,             // Key-Value cache
    num_kv_heads,         // GQA head count
    num_attention_heads   // Total head count
) -> Result<(Tensor, Tensor, Tensor)>
```

**Proposed Flash Attention Flow:**
```rust
#[cfg(all(feature = "flash-attention", target_os = "linux"))]
fn flash_attention_cuda(
    q: &Tensor,           // [batch, seq_len, heads, head_dim]
    k: &Tensor,           // [batch, seq_len, kv_heads, head_dim]
    v: &Tensor,           // [batch, seq_len, kv_heads, head_dim]
    softmax_scale: f32    // 1.0 / sqrt(head_dim)
) -> Result<Tensor> {
    use candle_flash_attn::flash_attn;

    // Flash Attention expects [batch, seq_len, heads, head_dim]
    // Returns [batch, seq_len, heads, head_dim]
    flash_attn(q, k, v, softmax_scale, /*causal=*/true)
}

#[cfg(target_os = "macos")]
fn flash_attention_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor
) -> Result<Tensor> {
    // Fallback to optimized standard attention
    // TODO: Explore Metal Performance Shaders for Flash Attention
    grouped_query_attention_standard(q, k, v, mask)
}
```

### Grouped Query Attention (GQA) Support

Flash Attention natively supports GQA by accepting different head counts for Q vs K/V:

```rust
// Standard: num_attention_heads Q heads, num_kv_heads K/V heads
// Flash Attention handles this automatically by broadcasting K/V heads
flash_attn(
    q,  // [batch, seq_len, num_attention_heads, head_dim]
    k,  // [batch, seq_len, num_kv_heads, head_dim]
    v,  // [batch, seq_len, num_kv_heads, head_dim]
    softmax_scale,
    true  // causal=true for autoregressive generation
)
```

### KV Cache Integration

Flash Attention with KV caching requires special handling:

**Approach 1: Concatenate cached KV before Flash Attention**
```rust
// Concatenate past KV cache with new K/V
let k_full = if let Some(ref k_cache) = kv_cache.k {
    Tensor::cat(&[k_cache, &k], 1)?  // Concat along sequence dimension
} else {
    k.clone()
};

let v_full = if let Some(ref v_cache) = kv_cache.v {
    Tensor::cat(&[v_cache, &v], 1)?
} else {
    v.clone()
};

// Run Flash Attention on full sequences
let output = flash_attention_cuda(&q, &k_full, &v_full, softmax_scale)?;

// Update KV cache
kv_cache.k = Some(k_full);
kv_cache.v = Some(v_full);
```

**Approach 2: Incremental Flash Attention (Advanced)**
```rust
// For single-token generation (autoregressive), only compute attention for new token
// This is more complex but more efficient for long sequences
```

### Conditional Compilation Strategy

Use Rust's conditional compilation to provide Flash Attention on CUDA while maintaining compatibility:

```rust
// In heterogeneous_pipeline.rs

#[cfg(all(feature = "flash-attention", any(target_os = "linux", target_os = "windows")))]
use candle_flash_attn::flash_attn;

fn attention_implementation(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32
) -> Result<Tensor> {
    #[cfg(all(feature = "flash-attention", any(target_os = "linux", target_os = "windows")))]
    {
        // Use Flash Attention on CUDA
        flash_attn(q, k, v, softmax_scale, /*causal=*/true)
    }

    #[cfg(not(all(feature = "flash-attention", any(target_os = "linux", target_os = "windows"))))]
    {
        // Use standard attention (Metal, or CUDA without feature flag)
        standard_attention(q, k, v, mask, softmax_scale)
    }
}
```

## Metal Backend Strategy

Flash Attention is CUDA-only in `candle-flash-attn`. For Metal backend, we have three options:

### Option 1: Custom Metal Kernel (High Effort)
Implement Flash Attention algorithm in Metal Shading Language:
- **Pros**: Native performance, memory efficiency
- **Cons**: 2-3 weeks development, extensive testing, platform-specific
- **Effort**: 200-300 lines of Metal shader code + integration

### Option 2: Metal Performance Shaders (Medium Effort)
Use Apple's Metal Performance Shaders Graph framework:
- **Pros**: Maintained by Apple, optimized
- **Cons**: May not have Flash Attention specifically, API learning curve
- **Effort**: 1-2 weeks research + integration

### Option 3: Optimized Standard Attention (Low Effort - Current)
Keep current standard attention with optimizations:
- **Pros**: Already working, tested, maintainable
- **Cons**: Higher memory usage, slower for long sequences
- **Effort**: No additional work (already implemented)

**Recommendation**: Start with Option 3 (already done), research Option 2, implement Option 1 only if critical for Metal performance.

## Performance Expectations

### Memory Usage

**Standard Attention:**
```
TinyLlama-1.1B (seq_len=512, batch=1, heads=32, head_dim=64):
Attention Matrix: 1 * 32 * 512 * 512 * 4 bytes = 32 MB per layer
22 layers: ~704 MB for attention matrices alone
```

**Flash Attention:**
```
Tiles (block_size=128):
Per Tile: 1 * 32 * 128 * 128 * 4 bytes = 2 MB
Total Memory: O(seq_len) = ~8 MB per layer
22 layers: ~176 MB for attention (4x reduction)
```

### Speed

**Expected Speedup (CUDA):**
- Short sequences (<128 tokens): 0.9-1.1x (overhead from tiling)
- Medium sequences (128-512 tokens): 1.2-1.8x
- Long sequences (512-2048 tokens): 2.0-3.5x
- Very long sequences (2048+ tokens): 3.0-4.5x

**Expected Speedup (Metal with Fallback):**
- All sequences: 1.0x (no change, using standard attention)

## Usage

### Building with Flash Attention

**Linux/Windows (CUDA):**
```bash
# Enable Flash Attention feature
cargo build --release --features flash-attention

# Or add to Cargo.toml default features:
[features]
default = ["flash-attention"]
```

**macOS (Metal - Fallback):**
```bash
# Flash Attention is CUDA-only, will use standard attention fallback
cargo build --release --features flash-attention
# No effect on Metal, but keeps code compatible
```

### Runtime Detection

The pipeline will automatically use Flash Attention when:
1. `flash-attention` feature is enabled at compile time
2. Running on Linux/Windows with CUDA backend
3. Sequence length is long enough to benefit (>128 tokens recommended)

```rust
// In logs, you'll see:
INFO: Using Flash Attention v2 for CUDA backend
// or
INFO: Using standard attention (Metal fallback)
```

## Implementation Checklist

### Phase 2: CUDA Integration
- [ ] Create `flash_attention_cuda()` wrapper function
- [ ] Add conditional compilation directives
- [ ] Update `grouped_query_attention()` call site
- [ ] Test KV cache concatenation logic
- [ ] Verify GQA (Grouped Query Attention) compatibility
- [ ] Add logging for attention backend selection

### Phase 3: Testing
- [ ] Create `benchmark_flash_attention` example
- [ ] Measure memory usage (with/without Flash Attention)
- [ ] Benchmark speed across sequence lengths (128, 256, 512, 1024, 2048)
- [ ] Test with TinyLlama-1.1B
- [ ] Test with larger models (7B, 13B) when available
- [ ] Validate numerical correctness (output matches standard attention)

### Phase 4: Documentation
- [ ] Update HETEROGENEOUS_PIPELINE.md with Flash Attention section
- [ ] Add API documentation to `flash_attention_*()` functions
- [ ] Document build requirements (CUDA toolkit version)
- [ ] Add troubleshooting guide for Flash Attention issues
- [ ] Update PROJECT_STATUS.md with completion status

## Known Limitations

1. **CUDA-Only**: Flash Attention requires NVIDIA CUDA. Metal backend will use standard attention.
2. **Sequence Length Overhead**: For very short sequences (<64 tokens), Flash Attention may be slower due to tiling overhead.
3. **Memory Tradeoff**: Flash Attention saves memory during forward pass but may increase backward pass memory (not applicable for inference-only).
4. **CUDA Version**: Requires CUDA 11.4+ for `candle-flash-attn`.

## Future Enhancements

1. **Flash Attention v3**: When available, upgrade to v3 for Hopper GPUs (H100)
2. **Custom Metal Implementation**: Implement Flash Attention algorithm in Metal
3. **Adaptive Backend Selection**: Automatically choose Flash vs Standard based on sequence length
4. **Continuous Batching**: Combine Flash Attention with continuous batching for higher throughput

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Original Flash Attention
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691) - Improved algorithm
- [candle-flash-attn Documentation](https://docs.rs/candle-flash-attn) - Rust API
- [HuggingFace Candle](https://github.com/huggingface/candle) - Parent framework

## Contact

For questions or issues with Flash Attention integration:
- Review this document and HETEROGENEOUS_PIPELINE.md
- Check PROJECT_STATUS.md for implementation status
- Test with `benchmark_flash_attention` example (when available)

---

**Last Updated:** October 15, 2025
**Authors:** Claude Code
**Status:** Phase 1 Complete, Phases 2-4 In Progress
