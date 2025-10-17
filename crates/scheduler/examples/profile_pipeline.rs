// Comprehensive profiling tool for heterogeneous pipeline
// Measures token generation latency, throughput, and performance characteristics

use std::sync::Arc;
use std::time::Instant;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[derive(Clone)]
struct ProfileResult {
    test_name: String,
    input_tokens: usize,
    output_tokens: usize,
    total_time_ms: u128,
    first_token_ms: Option<u128>,
    tokens_per_second: f64,
    avg_token_latency_ms: f64,
    p50_token_latency_ms: f64,
    p95_token_latency_ms: f64,
    p99_token_latency_ms: f64,
}

impl ProfileResult {
    fn print(&self) {
        println!("=== {} ===", self.test_name);
        println!("  Input tokens: {}", self.input_tokens);
        println!("  Output tokens: {}", self.output_tokens);
        println!("  Total time: {}ms ({:.2}s)", self.total_time_ms, self.total_time_ms as f64 / 1000.0);
        if let Some(ttft) = self.first_token_ms {
            println!("  Time to first token (TTFT): {}ms", ttft);
        }
        println!("  Throughput: {:.2} tokens/sec", self.tokens_per_second);
        println!("  Avg latency per token: {:.1}ms", self.avg_token_latency_ms);
        println!("  P50 latency: {:.1}ms", self.p50_token_latency_ms);
        println!("  P95 latency: {:.1}ms", self.p95_token_latency_ms);
        println!("  P99 latency: {:.1}ms", self.p99_token_latency_ms);
        println!();
    }
}

fn calculate_percentile(mut values: Vec<u128>, percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort();
    let index = ((percentile / 100.0) * values.len() as f64) as usize;
    let index = index.min(values.len() - 1);
    values[index] as f64
}

async fn profile_single_inference(
    executor: &corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor,
    input_ids: &[u32],
    max_tokens: usize,
    test_name: &str,
    measure_per_token: bool,
) -> anyhow::Result<ProfileResult> {
    let start = Instant::now();

    if measure_per_token {
        // Use streaming API to measure per-token latency
        use futures::StreamExt;

        let stream = executor.infer_stream(input_ids.to_vec(), max_tokens, 0.7, 0.95);
        futures::pin_mut!(stream);

        let mut token_latencies = Vec::new();
        let mut last_time = start;
        let mut first_token_time = None;
        let mut token_count = 0;

        while let Some(token_result) = stream.next().await {
            let now = Instant::now();
            let latency = now.duration_since(last_time).as_millis();

            if first_token_time.is_none() {
                first_token_time = Some(now.duration_since(start).as_millis());
            } else {
                token_latencies.push(latency);
            }

            last_time = now;
            token_count += 1;

            token_result?;
        }

        let total_time = start.elapsed().as_millis();
        let tokens_per_second = if total_time > 0 {
            (token_count as f64 / total_time as f64) * 1000.0
        } else {
            0.0
        };

        let avg_latency = if !token_latencies.is_empty() {
            token_latencies.iter().sum::<u128>() as f64 / token_latencies.len() as f64
        } else {
            0.0
        };

        Ok(ProfileResult {
            test_name: test_name.to_string(),
            input_tokens: input_ids.len(),
            output_tokens: token_count,
            total_time_ms: total_time,
            first_token_ms: first_token_time,
            tokens_per_second,
            avg_token_latency_ms: avg_latency,
            p50_token_latency_ms: calculate_percentile(token_latencies.clone(), 50.0),
            p95_token_latency_ms: calculate_percentile(token_latencies.clone(), 95.0),
            p99_token_latency_ms: calculate_percentile(token_latencies.clone(), 99.0),
        })
    } else {
        // Batch inference without per-token measurement
        let output = executor.infer(input_ids, max_tokens, 0.7, 0.95).await?;
        let total_time = start.elapsed().as_millis();
        let token_count = output.len();

        let tokens_per_second = if total_time > 0 {
            (token_count as f64 / total_time as f64) * 1000.0
        } else {
            0.0
        };

        Ok(ProfileResult {
            test_name: test_name.to_string(),
            input_tokens: input_ids.len(),
            output_tokens: token_count,
            total_time_ms: total_time,
            first_token_ms: None,
            tokens_per_second,
            avg_token_latency_ms: total_time as f64 / token_count as f64,
            p50_token_latency_ms: 0.0,
            p95_token_latency_ms: 0.0,
            p99_token_latency_ms: 0.0,
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Heterogeneous Pipeline - Performance Profiling Tool    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Model path
    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";

    println!("Model: {}", model_path);

    // Create Metal device for testing
    let devices = vec![
        GpuDevice {
            agent_id: "profiling-agent".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
    ];

    println!("Loading pipeline...");
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
    let load_time = load_start.elapsed();
    println!("Pipeline loaded in {:.2}s\n", load_time.as_secs_f64());

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let mut all_results = Vec::new();

    // ========================================================================
    // Test 1: Warmup effect measurement
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 1: Warmup Effect (Cold vs Warm Start)               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_prompt = "The quick brown fox";
    let encoding = tokenizer.encode(test_prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Cold start
    let result = profile_single_inference(
        &executor,
        &input_ids,
        10,
        "Cold Start (no warmup)",
        true,
    ).await?;
    all_results.push(result.clone());
    result.print();

    // Warm start
    let result = profile_single_inference(
        &executor,
        &input_ids,
        10,
        "Warm Start (cached kernels)",
        true,
    ).await?;
    all_results.push(result.clone());
    result.print();

    // ========================================================================
    // Test 2: Token count scaling
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 2: Output Token Count Scaling                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    for token_count in [5, 10, 20, 50] {
        let result = profile_single_inference(
            &executor,
            &input_ids,
            token_count,
            &format!("{} output tokens", token_count),
            false,
        ).await?;
        all_results.push(result.clone());
        result.print();
    }

    // ========================================================================
    // Test 3: Input length scaling
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 3: Input Sequence Length Scaling                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let prompts = vec![
        ("Short (4 tokens)", "The quick brown"),
        ("Medium (8 tokens)", "The quick brown fox jumps over the lazy"),
        ("Long (16 tokens)", "The quick brown fox jumps over the lazy dog and then runs through the forest"),
    ];

    for (label, prompt) in &prompts {
        let encoding = tokenizer.encode(*prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();

        let result = profile_single_inference(
            &executor,
            &ids,
            10,
            label,
            true,
        ).await?;
        all_results.push(result.clone());
        result.print();
    }

    // ========================================================================
    // Test 4: Sustained throughput (multiple runs)
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 4: Sustained Throughput (10 sequential runs)        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let mut throughputs = Vec::new();
    let sustained_start = Instant::now();
    let num_runs = 10;

    for i in 0..num_runs {
        let result = profile_single_inference(
            &executor,
            &input_ids,
            10,
            &format!("Run {}", i + 1),
            false,
        ).await?;
        throughputs.push(result.tokens_per_second);
    }

    let sustained_duration = sustained_start.elapsed();
    let total_tokens = num_runs * 10;
    let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let overall_throughput = (total_tokens as f64 / sustained_duration.as_secs_f64()).round();

    println!("Total runs: {}", num_runs);
    println!("Total tokens generated: {}", total_tokens);
    println!("Total time: {:.2}s", sustained_duration.as_secs_f64());
    println!("Average throughput per run: {:.2} tokens/sec", avg_throughput);
    println!("Overall sustained throughput: {:.1} tokens/sec\n", overall_throughput);

    // ========================================================================
    // Test 5: Batch processing
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 5: Batch Processing Performance                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let batch_prompts = vec![
        "The quick brown fox",
        "Once upon a time",
        "In a galaxy far",
    ];

    let mut batch_sequences = Vec::new();
    for prompt in &batch_prompts {
        let encoding = tokenizer.encode(*prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        batch_sequences.push(encoding.get_ids().to_vec());
    }

    let batch_start = Instant::now();
    let batch_outputs = executor.infer_batch(&batch_sequences, 10, 0.7, 0.95).await?;
    let batch_time = batch_start.elapsed();

    let total_output_tokens: usize = batch_outputs.iter().map(|o| o.len()).sum();
    let batch_throughput = (total_output_tokens as f64 / batch_time.as_secs_f64()).round();

    println!("Batch size: {}", batch_prompts.len());
    println!("Total output tokens: {}", total_output_tokens);
    println!("Total time: {:.2}s", batch_time.as_secs_f64());
    println!("Batch throughput: {:.1} tokens/sec", batch_throughput);
    println!("Avg time per sequence: {:.2}s\n", batch_time.as_secs_f64() / batch_prompts.len() as f64);

    // ========================================================================
    // Summary Report
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Performance Summary                                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    if all_results.len() >= 2 {
        let cold_start = &all_results[0];
        let warm_start = &all_results[1];

        if let (Some(cold_ttft), Some(warm_ttft)) = (cold_start.first_token_ms, warm_start.first_token_ms) {
            let warmup_benefit = cold_ttft as i64 - warm_ttft as i64;
            println!("Warmup Benefit:");
            println!("  Cold TTFT: {}ms", cold_ttft);
            println!("  Warm TTFT: {}ms", warm_ttft);
            println!("  Improvement: {}ms ({:.1}% faster)\n",
                warmup_benefit,
                (warmup_benefit as f64 / cold_ttft as f64) * 100.0
            );
        }
    }

    // Find best throughput
    let best_throughput = all_results.iter()
        .map(|r| r.tokens_per_second)
        .fold(0.0f64, f64::max);

    println!("Peak Throughput: {:.2} tokens/sec", best_throughput);
    println!("Sustained Throughput (10 runs): {:.1} tokens/sec", overall_throughput);
    println!("Batch Throughput: {:.1} tokens/sec", batch_throughput);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Profiling Complete                                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
