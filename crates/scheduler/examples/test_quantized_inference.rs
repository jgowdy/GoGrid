/// Test quantized model inference with GGUF format
///
/// This example demonstrates loading and using quantized models (INT4/INT8) with the
/// heterogeneous pipeline. Quantization reduces memory usage by 2-4x while maintaining
/// good inference quality.
///
/// Usage:
/// ```bash
/// # Download a quantized GGUF model first, e.g.:
/// # huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
///
/// cargo run --example test_quantized_inference --features quantization-gguf -- \
///     --model-path /path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
/// ```

use anyhow::Result;
use clap::Parser;
use corpgrid_common::GpuBackend;
use corpgrid_scheduler::model_hosting::GpuDevice;
use corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline;

#[cfg(feature = "quantization-gguf")]
use corpgrid_scheduler::quantization::{
    is_quantized_model, detect_quantization_format, QuantizationFormat,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to quantized GGUF model file
    #[arg(short, long)]
    model_path: String,

    /// Test prompt for inference
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    println!("=== Quantized Model Inference Test ===\n");

    // Check if quantization is enabled
    #[cfg(not(feature = "quantization-gguf"))]
    {
        eprintln!("ERROR: This example requires the 'quantization-gguf' feature to be enabled.");
        eprintln!("Build with: cargo run --example test_quantized_inference --features quantization-gguf");
        std::process::exit(1);
    }

    #[cfg(feature = "quantization-gguf")]
    {
        let model_path = std::path::Path::new(&args.model_path);

        // 1. Detect quantization format
        println!("📊 Detecting quantization format...");
        let format = detect_quantization_format(model_path);

        match format {
            QuantizationFormat::Gguf => {
                println!("✅ Detected GGUF quantized model");
            }
            QuantizationFormat::None => {
                eprintln!("❌ ERROR: Model is not quantized (no .gguf file found)");
                eprintln!("Please provide a path to a GGUF quantized model file.");
                eprintln!("\nYou can download quantized models from HuggingFace, e.g.:");
                eprintln!("  huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
                std::process::exit(1);
            }
            _ => {
                eprintln!("❌ ERROR: Unsupported quantization format: {:?}", format);
                std::process::exit(1);
            }
        }

        // 2. Verify model exists and is quantized
        if !is_quantized_model(model_path) {
            eprintln!("❌ ERROR: Model is not quantized");
            std::process::exit(1);
        }

        // 3. Create simulated GPU devices
        println!("\n🖥️  Setting up GPU devices...");
        let devices = create_test_devices();

        println!("Devices configured:");
        for (idx, device) in devices.iter().enumerate() {
            println!("  Device {}: {:?} ({})", idx, device.backend, device.device_name);
        }

        // 4. Create heterogeneous pipeline with quantization
        println!("\n🔧 Creating heterogeneous pipeline with quantized model...");
        let pipeline = HeterogeneousPipeline::new(&devices, &args.model_path)?;

        // 5. Check if quantization was loaded
        if pipeline.is_quantized() {
            println!("✅ Quantized model successfully loaded!");

            if let Some(quant_model) = pipeline.quantized_model() {
                if let Some(ref metadata) = quant_model.metadata {
                    println!("\n📈 Quantization Details:");
                    println!("  Format: {:?}", quant_model.format);
                    println!("  Quantization Type: {:?}", metadata.quantization_type);
                    println!("  Total Tensors: {}", metadata.tensor_count);

                    // Calculate memory savings
                    if let Some(ref gguf_model) = quant_model.gguf_model {
                        let quantized_mb = gguf_model.quantized_memory_usage() as f64 / 1024.0 / 1024.0;
                        let dequantized_mb = gguf_model.dequantized_memory_usage() as f64 / 1024.0 / 1024.0;
                        let savings_ratio = dequantized_mb / quantized_mb;

                        println!("\n💾 Memory Usage:");
                        println!("  Quantized (INT4/INT8): {:.2} MB", quantized_mb);
                        println!("  Dequantized (FP16): {:.2} MB", dequantized_mb);
                        println!("  Memory Reduction: {:.1}x", savings_ratio);
                        println!("  Savings: {:.2} MB ({:.1}%)",
                            dequantized_mb - quantized_mb,
                            ((dequantized_mb - quantized_mb) / dequantized_mb) * 100.0
                        );

                        // List some tensor names
                        let tensor_names = gguf_model.tensor_names();
                        println!("\n📦 Sample Tensors (first 5):");
                        for (idx, name) in tensor_names.iter().take(5).enumerate() {
                            println!("  {}: {}", idx + 1, name);
                        }
                        if tensor_names.len() > 5 {
                            println!("  ... and {} more tensors", tensor_names.len() - 5);
                        }
                    }
                }
            }
        } else {
            println!("⚠️  Warning: Pipeline created but quantization not enabled");
        }

        // 6. Verify device health
        println!("\n🏥 Checking device health...");
        match pipeline.check_all_devices_health() {
            Ok(_) => println!("✅ All devices are healthy"),
            Err(e) => {
                eprintln!("❌ Device health check failed: {}", e);
                std::process::exit(1);
            }
        }

        println!("\n✅ Quantized model inference test completed successfully!");
        println!("\nNote: Actual inference with quantized models is not yet fully implemented.");
        println!("The pipeline currently loads and validates quantized models.");
        println!("Full inference support will be added in a future update.");
    }

    Ok(())
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
