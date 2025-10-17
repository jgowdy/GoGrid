// Performance comparison: Homogeneous (single-device) vs Heterogeneous (multi-device) pipelines
//
// This benchmark compares:
// 1. Homogeneous: All layers on a single Metal device
// 2. Simulated Heterogeneous: Layers distributed across 2 Metal devices (same physical GPU)
//
// Measures:
// - Time To First Token (TTFT)
// - Tokens per second (throughput)
// - Per-token latency (P50, P95, P99)
// - Distribution overhead
//
// NOTE: On macOS with single physical GPU, this simulates multi-device by creating
// multiple logical device indices. On systems with multiple GPUs, this would test
// true heterogeneous distribution.

use std::sync::Arc;
use std::time::Instant;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[derive(Debug)]
struct BenchmarkResult {
    configuration: String,
    num_devices: usize,
    total_time_ms: u128,
    time_to_first_token_ms: u128,
    total_tokens_generated: usize,
    tokens_per_second: f64,
    avg_latency_ms: f64,
    p50_latency_ms: u128,
    p95_latency_ms: u128,
    p99_latency_ms: u128,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║ Configuration: {:<44}║", self.configuration);
        println!("╚═══════════════════════════════════════════════════════════╝");

        println!("\nDevices: {}", self.num_devices);
        println!("\nPerformance:");
        println!("  Time To First Token (TTFT): {}ms", self.time_to_first_token_ms);
        println!("  Total time: {}ms ({:.2}s)", self.total_time_ms, self.total_time_ms as f64 / 1000.0);
        println!("  Total tokens: {}", self.total_tokens_generated);
        println!("  Throughput: {:.2} tokens/sec", self.tokens_per_second);

        println!("\nPer-Token Latency:");
        println!("  Average: {:.2}ms", self.avg_latency_ms);
        println!("  P50 (median): {}ms", self.p50_latency_ms);
        println!("  P95: {}ms", self.p95_latency_ms);
        println!("  P99: {}ms", self.p99_latency_ms);
    }
}

async fn benchmark_configuration(
    devices: &[GpuDevice],
    model_path: &str,
    config_name: &str,
    num_runs: usize,
    tokens_per_run: usize,
) -> anyhow::Result<BenchmarkResult> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Benchmarking: {:<44}║", config_name);
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create pipeline
    println!("Creating pipeline with {} device(s)...", devices.len());
    let pipeline = corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline::new(
        devices,
        model_path,
    )?;

    let executor = Arc::new(
        corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor::new(
            Arc::new(pipeline),
            model_path,
        ).await?
    );

    // Warm up Metal kernels
    println!("Warming up Metal kernels...");
    executor.warmup_metal_kernels().await?;
    println!("Warmup complete\n");

    // Run benchmarks
    let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 8-token input
    let mut total_time_ms = 0u128;
    let mut total_tokens = 0usize;
    let mut ttft = 0u128;
    let mut all_token_latencies = Vec::new();

    println!("Running {} benchmark iterations...", num_runs);

    for run in 0..num_runs {
        // Clear caches between runs for independence
        executor.clear_kv_caches().await;

        let run_start = Instant::now();
        let mut first_token_time: Option<Instant> = None;
        let mut prev_time = run_start;

        // Generate tokens and measure per-token latency
        let mut generated_count = 0;
        for _ in 0..tokens_per_run {
            let token_start = Instant::now();

            match executor.infer(&input_ids, 1, 0.7, 0.95).await {
                Ok(tokens) => {
                    if tokens.is_empty() {
                        break; // EOS or max tokens
                    }

                    let token_time = token_start.elapsed();
                    all_token_latencies.push(token_time.as_millis());

                    // Record TTFT on first successful token
                    if first_token_time.is_none() {
                        first_token_time = Some(token_start);
                        let ttft_ms = (token_start - run_start).as_millis();
                        if run == 0 {
                            ttft = ttft_ms;
                        }
                    }

                    generated_count += tokens.len();
                    total_tokens += tokens.len();
                    prev_time = token_start;
                }
                Err(e) => {
                    eprintln!("  Run {} failed: {}", run + 1, e);
                    break;
                }
            }
        }

        let run_time = run_start.elapsed();
        total_time_ms += run_time.as_millis();

        if (run + 1) % 5 == 0 {
            println!("  Completed run {}/{}: generated {} tokens in {:.2}s",
                run + 1, num_runs, generated_count, run_time.as_secs_f64());
        }
    }

    println!("Benchmark complete\n");

    // Calculate statistics
    let avg_time_per_run = total_time_ms as f64 / num_runs as f64;
    let tokens_per_second = if total_time_ms > 0 {
        (total_tokens as f64 / total_time_ms as f64) * 1000.0
    } else {
        0.0
    };

    let avg_latency = if !all_token_latencies.is_empty() {
        all_token_latencies.iter().sum::<u128>() as f64 / all_token_latencies.len() as f64
    } else {
        0.0
    };

    // Calculate percentiles
    let mut sorted_latencies = all_token_latencies.clone();
    sorted_latencies.sort();

    let p50 = if !sorted_latencies.is_empty() {
        sorted_latencies[sorted_latencies.len() / 2]
    } else {
        0
    };

    let p95 = if !sorted_latencies.is_empty() {
        sorted_latencies[(sorted_latencies.len() as f64 * 0.95) as usize]
    } else {
        0
    };

    let p99 = if !sorted_latencies.is_empty() {
        sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize]
    } else {
        0
    };

    Ok(BenchmarkResult {
        configuration: config_name.to_string(),
        num_devices: devices.len(),
        total_time_ms,
        time_to_first_token_ms: ttft,
        total_tokens_generated: total_tokens,
        tokens_per_second,
        avg_latency_ms: avg_latency,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
    })
}

fn print_comparison(homogeneous: &BenchmarkResult, heterogeneous: &BenchmarkResult) {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Performance Comparison Summary                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Configuration Comparison:");
    println!("  Homogeneous:  {} device(s)", homogeneous.num_devices);
    println!("  Heterogeneous: {} device(s)", heterogeneous.num_devices);

    println!("\nTime To First Token (TTFT):");
    println!("  Homogeneous:  {}ms", homogeneous.time_to_first_token_ms);
    println!("  Heterogeneous: {}ms", heterogeneous.time_to_first_token_ms);
    let ttft_diff = heterogeneous.time_to_first_token_ms as i128 - homogeneous.time_to_first_token_ms as i128;
    if ttft_diff > 0 {
        println!("  Distribution overhead: +{}ms ({:.1}% slower)",
            ttft_diff,
            (ttft_diff as f64 / homogeneous.time_to_first_token_ms as f64) * 100.0
        );
    } else {
        println!("  Distribution benefit: {}ms ({:.1}% faster)",
            ttft_diff.abs(),
            (ttft_diff.abs() as f64 / homogeneous.time_to_first_token_ms as f64) * 100.0
        );
    }

    println!("\nThroughput (tokens/sec):");
    println!("  Homogeneous:  {:.2}", homogeneous.tokens_per_second);
    println!("  Heterogeneous: {:.2}", heterogeneous.tokens_per_second);
    let throughput_ratio = heterogeneous.tokens_per_second / homogeneous.tokens_per_second;
    if throughput_ratio < 1.0 {
        println!("  Distribution overhead: {:.1}% slower", (1.0 - throughput_ratio) * 100.0);
    } else {
        println!("  Distribution benefit: {:.1}% faster", (throughput_ratio - 1.0) * 100.0);
    }

    println!("\nPer-Token Latency (Average):");
    println!("  Homogeneous:  {:.2}ms", homogeneous.avg_latency_ms);
    println!("  Heterogeneous: {:.2}ms", heterogeneous.avg_latency_ms);
    let latency_diff = heterogeneous.avg_latency_ms - homogeneous.avg_latency_ms;
    if latency_diff > 0.0 {
        println!("  Distribution overhead: +{:.2}ms ({:.1}% slower)",
            latency_diff,
            (latency_diff / homogeneous.avg_latency_ms) * 100.0
        );
    } else {
        println!("  Distribution benefit: {:.2}ms ({:.1}% faster)",
            latency_diff.abs(),
            (latency_diff.abs() / homogeneous.avg_latency_ms) * 100.0
        );
    }

    println!("\nP95 Latency:");
    println!("  Homogeneous:  {}ms", homogeneous.p95_latency_ms);
    println!("  Heterogeneous: {}ms", heterogeneous.p95_latency_ms);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Key Insights                                              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    if throughput_ratio < 0.95 {
        println!("⚠ Significant overhead detected from pipeline distribution.");
        println!("  Consider using homogeneous configuration for this workload.");
        println!("  Distribution is beneficial when:");
        println!("    • Model is too large for a single device");
        println!("    • True heterogeneous hardware available (Metal + CUDA)");
        println!("    • Devices have different compute capabilities");
    } else if throughput_ratio < 1.0 {
        println!("✓ Minimal overhead from pipeline distribution.");
        println!("  Acceptable tradeoff for enabling larger model deployment.");
    } else {
        println!("✓ Pipeline distribution provides performance benefit!");
        println!("  This may indicate parallelization opportunities or");
        println!("  better memory locality in the distributed configuration.");
    }

    println!("\nNOTE: On macOS with single physical GPU, this simulates");
    println!("      multi-device via logical device indices. True heterogeneous");
    println!("      performance requires actual different GPU hardware.");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Pipeline Configuration Performance Comparison          ║");
    println!("║   Homogeneous vs Heterogeneous (Distributed)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";
    println!("Model: {}\n", model_path);

    // Benchmark parameters
    let num_runs = 10;
    let tokens_per_run = 20;

    println!("Benchmark Parameters:");
    println!("  Runs per configuration: {}", num_runs);
    println!("  Tokens per run: {}", tokens_per_run);
    println!("  Total tokens per config: {}\n", num_runs * tokens_per_run);

    // ========================================================================
    // Configuration 1: Homogeneous - Single Device (All layers on one GPU)
    // ========================================================================

    let homogeneous_devices = vec![
        GpuDevice {
            agent_id: "homogeneous-agent".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
    ];

    let homogeneous_result = benchmark_configuration(
        &homogeneous_devices,
        model_path,
        "Homogeneous (Single Device)",
        num_runs,
        tokens_per_run,
    ).await?;

    homogeneous_result.print();

    // ========================================================================
    // Configuration 2: Heterogeneous - Multi-Device (Layers distributed)
    // ========================================================================

    // NOTE: On macOS with single physical GPU, this simulates multi-device
    // by creating 2 logical devices pointing to same GPU. On systems with
    // multiple GPUs, use different device_index values for true heterogeneity.

    let heterogeneous_devices = vec![
        GpuDevice {
            agent_id: "hetero-agent-1".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB (simulated split)
            vram_free_bytes: 8 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max (Stage 1)".to_string(),
            is_allocated: false,
        },
        GpuDevice {
            agent_id: "hetero-agent-2".to_string(),
            device_index: 0, // Same physical device (simulation)
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 8 * 1024 * 1024 * 1024,
            vram_free_bytes: 8 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max (Stage 2)".to_string(),
            is_allocated: false,
        },
    ];

    let heterogeneous_result = benchmark_configuration(
        &heterogeneous_devices,
        model_path,
        "Heterogeneous (Distributed - 2 Devices)",
        num_runs,
        tokens_per_run,
    ).await?;

    heterogeneous_result.print();

    // ========================================================================
    // Comparison Summary
    // ========================================================================

    print_comparison(&homogeneous_result, &heterogeneous_result);

    Ok(())
}
