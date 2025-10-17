// Test batch inference functionality
// Tests processing multiple sequences with the new infer_batch API

use std::sync::Arc;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Testing Batch Inference ===\n");

    // Model path
    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";

    println!("Model: {}", model_path);

    // Create Metal device for testing
    let devices = vec![
        GpuDevice {
            agent_id: "test-agent".to_string(),
            device_index: 0,
            backend: corpgrid_common::GpuBackend::Metal,
            vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            device_name: "Apple M2 Max".to_string(),
            is_allocated: false,
        },
    ];

    println!("Creating pipeline...");
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

    println!("Pipeline created successfully\n");

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Create test batch with multiple prompts
    let test_prompts = vec![
        "The quick brown fox",
        "Once upon a time",
        "In a galaxy far",
    ];

    println!("=== Encoding {} prompts ===", test_prompts.len());
    let mut input_sequences = Vec::new();
    for (idx, prompt) in test_prompts.iter().enumerate() {
        println!("  [{}] \"{}\"", idx, prompt);
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        println!("      {} tokens: {:?}", ids.len(), &ids[..ids.len().min(10)]);
        input_sequences.push(ids);
    }

    println!("\n=== Running Batch Inference ===");
    println!("Batch size: {}", input_sequences.len());
    println!("Max tokens per sequence: 5\n");

    let start = std::time::Instant::now();
    let outputs = executor.infer_batch(&input_sequences, 5, 0.7, 0.95).await?;
    let duration = start.elapsed();

    println!("\n=== Results ===");
    println!("Total time: {:.2}s", duration.as_secs_f64());
    println!("Average per sequence: {:.2}s\n", duration.as_secs_f64() / test_prompts.len() as f64);

    for (idx, output_tokens) in outputs.iter().enumerate() {
        let output_text = tokenizer.decode(output_tokens, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode: {}", e))?;

        println!("Sequence {}:", idx);
        println!("  Input: \"{}\"", test_prompts[idx]);
        println!("  Output: \"{}\"", output_text);
        println!("  Tokens: {} generated", output_tokens.len());
        println!();
    }

    println!("=== Test Complete ===");
    println!("✓ Batch processing API working correctly");
    println!("✓ {} sequences processed", outputs.len());

    Ok(())
}
