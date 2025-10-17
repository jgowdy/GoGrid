// Benchmark for Metal-specific optimizations
// Tests RoPE caching, causal mask caching, and Metal kernel warmup

use std::sync::Arc;
use std::time::Instant;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing with timing info
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Metal Optimizations Benchmark ===\n");

    // Model path
    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";

    println!("Model: {}", model_path);
    println!("Testing: RoPE caching, Causal mask caching, Metal kernel warmup\n");

    // Create Metal device for testing
    let devices = vec![
        GpuDevice {
            agent_id: "benchmark-agent".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
    ];

    println!("=== Phase 1: Initial Load & Cold Start ===");

    let load_start = Instant::now();
    let pipeline = corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline::new(
        &devices,
        model_path,
    )?;
    let executor = Arc::new(
        corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor::new(
            Arc::new(pipeline),
            model_path,
        ).await?
    );
    let load_duration = load_start.elapsed();
    println!("Model load time: {:.2}s\n", load_duration.as_secs_f64());

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Test prompt
    let test_prompt = "The quick brown fox jumps over the lazy dog";
    let encoding = tokenizer.encode(test_prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("=== Phase 2: Cold Start (First Token - NO Warmup) ===");
    println!("This measures Metal shader compilation overhead\n");

    let first_token_start = Instant::now();
    let cold_output = executor.infer(&input_ids, 1, 0.7, 0.95).await?;
    let first_token_duration = first_token_start.elapsed();

    println!("First token latency (cold): {:.0}ms", first_token_duration.as_millis());
    println!("Generated {} token(s)\n", cold_output.len());

    println!("=== Phase 3: Warmup Test ===");
    println!("Testing Metal kernel pre-compilation\n");

    // Create a fresh executor for warmup test
    let pipeline2 = corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline::new(
        &devices,
        model_path,
    )?;
    let executor2 = Arc::new(
        corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor::new(
            Arc::new(pipeline2),
            model_path,
        ).await?
    );

    let warmup_start = Instant::now();
    executor2.warmup_metal_kernels().await?;
    let warmup_duration = warmup_start.elapsed();
    println!("Warmup time: {:.0}ms", warmup_duration.as_millis());

    let warm_first_token_start = Instant::now();
    let warm_output = executor2.infer(&input_ids, 1, 0.7, 0.95).await?;
    let warm_first_token_duration = warm_first_token_start.elapsed();

    println!("First token latency (after warmup): {:.0}ms", warm_first_token_duration.as_millis());
    println!("Generated {} token(s)", warm_output.len());

    let warmup_benefit = first_token_duration.as_millis() as i64 - warm_first_token_duration.as_millis() as i64;
    println!("Warmup benefit: {:.0}ms ({:.1}% faster)\n",
        warmup_benefit,
        (warmup_benefit as f64 / first_token_duration.as_millis() as f64) * 100.0
    );

    println!("=== Phase 4: Cache Benefit Test (Sequential Generation) ===");
    println!("Testing RoPE and causal mask caching during autoregressive generation\n");

    // Test with multiple tokens to see caching benefits
    let num_tokens_tests = vec![5, 10, 20];

    for num_tokens in num_tokens_tests {
        println!("--- Generating {} tokens ---", num_tokens);

        let gen_start = Instant::now();
        let output = executor.infer(&input_ids, num_tokens, 0.7, 0.95).await?;
        let gen_duration = gen_start.elapsed();

        let total_ms = gen_duration.as_millis();
        let tokens_per_sec = (num_tokens as f64 / gen_duration.as_secs_f64()).round();
        let ms_per_token = (total_ms as f64 / num_tokens as f64).round();

        println!("  Total time: {:.0}ms", total_ms);
        println!("  Tokens/sec: {:.1}", tokens_per_sec);
        println!("  ms/token: {:.1}", ms_per_token);
        println!("  Generated {} tokens total\n", output.len());
    }

    println!("=== Phase 5: Sequence Length Scaling Test ===");
    println!("Testing how mask caching benefits increase with sequence length\n");

    let prompts = vec![
        ("Short (4 tokens)", "The quick brown"),
        ("Medium (8 tokens)", "The quick brown fox jumps over the lazy"),
        ("Long (16 tokens)", "The quick brown fox jumps over the lazy dog and then runs through the forest"),
    ];

    for (label, prompt) in prompts {
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();

        println!("--- {} ({} input tokens) ---", label, ids.len());

        let start = Instant::now();
        let output = executor.infer(&ids, 10, 0.7, 0.95).await?;
        let duration = start.elapsed();

        let tokens_per_sec = (10.0 / duration.as_secs_f64()).round();

        println!("  Time for 10 tokens: {:.0}ms", duration.as_millis());
        println!("  Tokens/sec: {:.1}", tokens_per_sec);
        println!("  Generated {} tokens total\n", output.len());
    }

    println!("=== Phase 6: Repeated Inference (Cache Hit Test) ===");
    println!("Testing cache hit rates with repeated prompts\n");

    // Run same prompt multiple times to test cache hits
    let iterations = 5;
    let mut timings = Vec::new();

    for i in 1..=iterations {
        let start = Instant::now();
        let _ = executor.infer(&input_ids, 5, 0.7, 0.95).await?;
        let duration = start.elapsed();
        timings.push(duration);

        println!("Iteration {}: {:.0}ms", i, duration.as_millis());
    }

    let avg_time = timings.iter().map(|d| d.as_millis()).sum::<u128>() / timings.len() as u128;
    let first_time = timings[0].as_millis();
    let subsequent_avg = timings[1..].iter().map(|d| d.as_millis()).sum::<u128>() / (timings.len() - 1) as u128;

    println!("\nFirst run: {:.0}ms", first_time);
    println!("Subsequent runs avg: {:.0}ms", subsequent_avg);
    println!("Average: {:.0}ms", avg_time);

    if first_time > subsequent_avg {
        let improvement = first_time as i64 - subsequent_avg as i64;
        println!("Cache benefit: {:.0}ms ({:.1}% faster)\n",
            improvement,
            (improvement as f64 / first_time as f64) * 100.0
        );
    }

    println!("=== Benchmark Summary ===\n");

    println!("Metal Optimizations Performance:");
    println!("1. Warmup: Eliminates {:.0}ms first-token latency", warmup_benefit);
    println!("2. Caching: ~{:.1}% faster on repeated inferences",
        ((first_time as i64 - subsequent_avg as i64) as f64 / first_time as f64) * 100.0
    );
    println!("3. Throughput: ~{:.1} tokens/sec for sustained generation\n",
        (20.0 / timings.iter().map(|d| d.as_secs_f64()).sum::<f64>()) * iterations as f64
    );

    println!("Expected benefits from METAL_OPTIMIZATIONS.md:");
    println!("- First token: 200-500ms faster (actual: {:.0}ms)", warmup_benefit);
    println!("- Subsequent tokens: 10-20% faster (actual: {:.1}%)",
        ((first_time as i64 - subsequent_avg as i64) as f64 / first_time as f64) * 100.0
    );

    Ok(())
}
