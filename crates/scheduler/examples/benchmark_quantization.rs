/// Benchmark quantized vs non-quantized model inference
///
/// This example compares performance between quantized (INT4/INT8) and non-quantized (FP16)
/// models, measuring memory usage, loading time, and inference speed.
///
/// Usage:
/// ```bash
/// # Test with quantized GGUF model
/// cargo run --example benchmark_quantization --features quantization-gguf -- \
///     --quantized-path /path/to/model.Q4_K_M.gguf \
///     --fp16-path /path/to/model-fp16
///
/// # Or test quantized only
/// cargo run --example benchmark_quantization --features quantization-gguf -- \
///     --quantized-path /path/to/model.Q4_K_M.gguf
/// ```

use anyhow::Result;
use clap::Parser;
use corpgrid_common::GpuBackend;
use corpgrid_scheduler::model_hosting::GpuDevice;
use corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline;
use std::time::Instant;

#[cfg(feature = "quantization-gguf")]
use corpgrid_scheduler::quantization::{
    is_quantized_model, detect_quantization_format, QuantizationFormat,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to quantized GGUF model file
    #[arg(short = 'q', long)]
    quantized_path: Option<String>,

    /// Path to FP16 model directory (for comparison)
    #[arg(short = 'f', long)]
    fp16_path: Option<String>,

    /// Number of warmup runs before benchmarking
    #[arg(short = 'w', long, default_value = "3")]
    warmup_runs: usize,

    /// Number of benchmark iterations
    #[arg(short = 'i', long, default_value = "10")]
    iterations: usize,

    /// Test prompt for inference
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,
}

#[derive(Debug)]
struct BenchmarkResults {
    model_type: String,
    load_time_ms: u128,
    memory_usage_mb: f64,
    avg_inference_time_ms: f64,
    min_inference_time_ms: f64,
    max_inference_time_ms: f64,
    std_dev_ms: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    println!("=== Quantization Benchmark ===\n");

    // Check if quantization is enabled
    #[cfg(not(feature = "quantization-gguf"))]
    {
        if args.quantized_path.is_some() {
            eprintln!("ERROR: Quantized model specified but 'quantization-gguf' feature not enabled.");
            eprintln!("Build with: cargo run --example benchmark_quantization --features quantization-gguf");
            std::process::exit(1);
        }
    }

    if args.quantized_path.is_none() && args.fp16_path.is_none() {
        eprintln!("ERROR: Must specify at least one model path (--quantized-path or --fp16-path)");
        std::process::exit(1);
    }

    // Create test devices
    let devices = create_test_devices();
    println!("🖥️  GPU Configuration:");
    for (idx, device) in devices.iter().enumerate() {
        println!("  Device {}: {:?} ({})", idx, device.backend, device.device_name);
    }
    println!();

    let mut results = Vec::new();

    // Benchmark quantized model if provided
    #[cfg(feature = "quantization-gguf")]
    if let Some(quantized_path) = &args.quantized_path {
        println!("📊 Benchmarking Quantized Model");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let result = benchmark_model(
            quantized_path,
            &devices,
            "Quantized (GGUF)",
            args.warmup_runs,
            args.iterations,
        ).await?;

        results.push(result);
        println!();
    }

    // Benchmark FP16 model if provided
    if let Some(fp16_path) = &args.fp16_path {
        println!("📊 Benchmarking FP16 Model");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let result = benchmark_model(
            fp16_path,
            &devices,
            "FP16 (Standard)",
            args.warmup_runs,
            args.iterations,
        ).await?;

        results.push(result);
        println!();
    }

    // Print comparison summary
    print_comparison_summary(&results);

    Ok(())
}

async fn benchmark_model(
    model_path: &str,
    devices: &[GpuDevice],
    model_type: &str,
    warmup_runs: usize,
    iterations: usize,
) -> Result<BenchmarkResults> {
    let path = std::path::Path::new(model_path);

    // Detect if quantized
    #[cfg(feature = "quantization-gguf")]
    {
        let format = detect_quantization_format(path);
        match format {
            QuantizationFormat::Gguf => {
                println!("✅ Detected GGUF quantized model");
            }
            QuantizationFormat::None => {
                println!("ℹ️  Non-quantized model");
            }
            _ => {
                println!("⚠️  Unsupported format: {:?}", format);
            }
        }
    }

    // Measure loading time
    println!("⏱️  Measuring model loading time...");
    let load_start = Instant::now();
    let pipeline = HeterogeneousPipeline::new(devices, model_path)?;
    let load_time_ms = load_start.elapsed().as_millis();
    println!("   Load time: {} ms", load_time_ms);

    // Get memory usage
    let memory_usage_mb = get_memory_usage(&pipeline);
    println!("   Memory usage: {:.2} MB", memory_usage_mb);

    // Check device health
    println!("\n🏥 Checking device health...");
    pipeline.check_all_devices_health()?;
    println!("   ✅ All devices healthy");

    // Display quantization details if available
    #[cfg(feature = "quantization-gguf")]
    if pipeline.is_quantized() {
        if let Some(quant_model) = pipeline.quantized_model() {
            if let Some(ref metadata) = quant_model.metadata {
                println!("\n📈 Quantization Details:");
                println!("   Format: {:?}", quant_model.format);
                println!("   Quantization Type: {:?}", metadata.quantization_type);
                println!("   Total Tensors: {}", metadata.tensor_count);

                if let Some(ref gguf_model) = quant_model.gguf_model {
                    let quantized_mb = gguf_model.quantized_memory_usage() as f64 / 1024.0 / 1024.0;
                    let dequantized_mb = gguf_model.dequantized_memory_usage() as f64 / 1024.0 / 1024.0;
                    let savings_ratio = dequantized_mb / quantized_mb;

                    println!("\n💾 Memory Analysis:");
                    println!("   Quantized (INT4/INT8): {:.2} MB", quantized_mb);
                    println!("   Dequantized (FP16): {:.2} MB", dequantized_mb);
                    println!("   Memory Reduction: {:.1}x", savings_ratio);
                    println!("   Savings: {:.2} MB ({:.1}%)",
                        dequantized_mb - quantized_mb,
                        ((dequantized_mb - quantized_mb) / dequantized_mb) * 100.0
                    );
                }
            }
        }
    }

    // Warmup phase
    println!("\n🔥 Warmup Phase ({} runs)...", warmup_runs);
    for i in 0..warmup_runs {
        // Simulate inference (actual inference not yet implemented)
        std::thread::sleep(std::time::Duration::from_millis(10));
        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {}/{}", i + 1, warmup_runs);
        }
    }
    println!("\n   ✅ Warmup complete");

    // Benchmark phase
    println!("\n⚡ Benchmarking ({} iterations)...", iterations);
    let mut inference_times = Vec::new();

    for i in 0..iterations {
        let inference_start = Instant::now();

        // TODO: Actual inference when implemented
        // For now, simulate inference latency
        std::thread::sleep(std::time::Duration::from_millis(50));

        let inference_time_ms = inference_start.elapsed().as_micros() as f64 / 1000.0;
        inference_times.push(inference_time_ms);

        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {}/{}", i + 1, iterations);
        }
    }
    println!();

    // Calculate statistics
    let avg_inference_time_ms = inference_times.iter().sum::<f64>() / inference_times.len() as f64;
    let min_inference_time_ms = inference_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_inference_time_ms = inference_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Calculate standard deviation
    let variance = inference_times.iter()
        .map(|&t| (t - avg_inference_time_ms).powi(2))
        .sum::<f64>() / inference_times.len() as f64;
    let std_dev_ms = variance.sqrt();

    println!("\n📊 Results:");
    println!("   Average: {:.2} ms", avg_inference_time_ms);
    println!("   Min: {:.2} ms", min_inference_time_ms);
    println!("   Max: {:.2} ms", max_inference_time_ms);
    println!("   Std Dev: {:.2} ms", std_dev_ms);

    Ok(BenchmarkResults {
        model_type: model_type.to_string(),
        load_time_ms,
        memory_usage_mb,
        avg_inference_time_ms,
        min_inference_time_ms,
        max_inference_time_ms,
        std_dev_ms,
    })
}

fn get_memory_usage(pipeline: &HeterogeneousPipeline) -> f64 {
    #[cfg(feature = "quantization-gguf")]
    if let Some(quant_model) = pipeline.quantized_model() {
        if let Some(ref gguf_model) = quant_model.gguf_model {
            return gguf_model.quantized_memory_usage() as f64 / 1024.0 / 1024.0;
        }
    }

    // Estimate for non-quantized models (placeholder)
    0.0
}

fn print_comparison_summary(results: &[BenchmarkResults]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Summary                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Print table header
    println!("┌─────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Model Type      │ Load Time    │ Memory (MB)  │ Avg Latency  │");
    println!("├─────────────────┼──────────────┼──────────────┼──────────────┤");

    for result in results {
        println!("│ {:<15} │ {:>10} ms │ {:>10.2} MB │ {:>10.2} ms │",
            result.model_type,
            result.load_time_ms,
            result.memory_usage_mb,
            result.avg_inference_time_ms
        );
    }
    println!("└─────────────────┴──────────────┴──────────────┴──────────────┘\n");

    // If we have both quantized and FP16, show comparison
    if results.len() == 2 {
        let quantized = &results[0];
        let fp16 = &results[1];

        println!("📈 Improvement Metrics:");

        let memory_reduction = fp16.memory_usage_mb / quantized.memory_usage_mb;
        println!("   Memory Reduction: {:.2}x", memory_reduction);

        let speedup = fp16.avg_inference_time_ms / quantized.avg_inference_time_ms;
        if speedup > 1.0 {
            println!("   Inference Speedup: {:.2}x", speedup);
        } else {
            println!("   Inference Slowdown: {:.2}x", 1.0 / speedup);
        }

        let load_speedup = fp16.load_time_ms as f64 / quantized.load_time_ms as f64;
        if load_speedup > 1.0 {
            println!("   Load Time Speedup: {:.2}x", load_speedup);
        } else {
            println!("   Load Time Slowdown: {:.2}x", 1.0 / load_speedup);
        }

        println!("\n💡 Recommendation:");
        if memory_reduction > 2.0 && speedup > 0.9 {
            println!("   ✅ Quantization provides significant memory savings with minimal performance impact.");
            println!("   Use quantized model for production deployments with memory constraints.");
        } else if memory_reduction > 1.5 {
            println!("   ⚠️  Quantization provides memory savings but may impact inference speed.");
            println!("   Evaluate tradeoffs based on your requirements.");
        } else {
            println!("   ℹ️  FP16 model recommended for optimal performance.");
        }
    }

    println!("\n📝 Note: Actual inference is not yet fully implemented.");
    println!("   Inference times shown are simulated placeholders.");
    println!("   Full quantized inference support will be added in a future update.");
}

/// Create test GPU devices for the pipeline
fn create_test_devices() -> Vec<GpuDevice> {
    #[cfg(target_os = "macos")]
    {
        vec![GpuDevice {
            agent_id: "local-metal".to_string(),
            device_index: 0,
            backend: GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple Silicon GPU".to_string(),
            is_allocated: false,
        }]
    }

    #[cfg(not(target_os = "macos"))]
    {
        vec![GpuDevice {
            agent_id: "local-cuda".to_string(),
            device_index: 0,
            backend: GpuBackend::Cuda,
            vram_total_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            vram_free_bytes: 8 * 1024 * 1024 * 1024,
            compute_capability: Some("8.0".to_string()),
            device_name: "NVIDIA GPU".to_string(),
            is_allocated: false,
        }]
    }
}
