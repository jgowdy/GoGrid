// Standalone test for heterogeneous pipeline implementation
// This tests the pipeline logic even with homogeneous devices (Metal)

use std::sync::Arc;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Testing Heterogeneous Pipeline Implementation ===\n");

    // Model path
    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";

    println!("Model: {}", model_path);
    println!("Expected: 22 layers, will be split across devices\n");

    // Create simulated mixed devices for testing
    // Note: For this test on macOS, we'll use Metal devices at different indices
    // to simulate the pipeline distribution logic
    let devices = vec![
        GpuDevice {
            agent_id: "test-agent-1".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
        GpuDevice {
            agent_id: "test-agent-2".to_string(),
            device_index: 0, // Same physical device, but testing logic
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024,
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
    ];

    println!("Creating heterogeneous pipeline...");
    println!("Devices: {} (simulating mixed for testing)", devices.len());

    // Create pipeline
    let pipeline = corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline::new(
        &devices,
        model_path,
    )?;

    println!("Pipeline created with {} stages", pipeline.num_stages());

    for i in 0..pipeline.num_stages() {
        if let Some((backend, start, end)) = pipeline.get_stage_info(i) {
            println!("  Stage {}: {:?} layers {}-{}", i, backend, start, end);
        }
    }

    println!("\nCreating pipeline executor (loading weights)...");
    let executor = corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor::new(
        Arc::new(pipeline),
        model_path,
    ).await?;

    println!("Executor created successfully!");

    // Test inference with a simple prompt
    println!("\n=== Running Test Inference ===");
    // Use a prompt that encourages generation rather than immediate EOS
    let test_prompt = "The quick brown fox";
    println!("Prompt: \"{}\"", test_prompt);

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Encode prompt
    let encoding = tokenizer.encode(test_prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("Input tokens: {} {:?}", input_ids.len(), &input_ids[..input_ids.len().min(10)]);

    // Run inference
    println!("\nStarting generation (max 10 tokens)...");
    let output_tokens = executor.infer(
        &input_ids,
        10,  // max_new_tokens
        0.7, // temperature
        0.95 // top_p
    ).await?;

    println!("\nGenerated {} tokens:", output_tokens.len());
    println!("Output token IDs: {:?}", output_tokens);

    // Decode output
    let output_text = tokenizer.decode(&output_tokens, false)
        .map_err(|e| anyhow::anyhow!("Failed to decode output: {}", e))?;

    println!("\nGenerated text:");
    println!("\"{}\"", output_text);

    println!("\n=== Test Complete ===");
    println!("✓ Pipeline created successfully");
    println!("✓ Weights loaded");
    println!("✓ Inference executed");
    println!("✓ Tokens generated");

    Ok(())
}
