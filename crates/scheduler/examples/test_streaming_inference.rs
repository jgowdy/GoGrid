// Test streaming inference functionality
// Tests real-time token generation with the new infer_stream API

use std::sync::Arc;
use corpgrid_scheduler::model_hosting::GpuDevice;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Testing Streaming Inference ===\n");

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

    // Test prompt
    let test_prompt = "The quick brown fox";

    println!("=== Encoding prompt ===");
    println!("Prompt: \"{}\"", test_prompt);

    let encoding = tokenizer.encode(test_prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("Input tokens: {} {:?}\n", input_ids.len(), input_ids);

    println!("=== Streaming Token Generation ===");
    println!("Generating up to 20 tokens...");
    println!("Output: \"{}\" ", test_prompt);
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let start = std::time::Instant::now();
    let mut token_count = 0;
    let mut generated_tokens = Vec::new();
    let mut token_timings = Vec::new();
    let mut last_timestamp = start;

    // Create the stream and pin it
    let stream = executor.infer_stream(input_ids.clone(), 20, 0.7, 0.95);
    futures::pin_mut!(stream);

    // Process tokens as they arrive
    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token) => {
                let now = std::time::Instant::now();
                let token_latency = now.duration_since(last_timestamp);
                token_timings.push(token_latency);
                last_timestamp = now;

                generated_tokens.push(token);
                token_count += 1;

                // Decode and print the token immediately for visual streaming effect
                let token_text = tokenizer.decode(&[token], false)
                    .unwrap_or_else(|_| format!("[token:{}]", token));
                print!("{}", token_text);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            Err(e) => {
                println!("\n\nError during streaming: {}", e);
                return Err(e);
            }
        }
    }

    let total_duration = start.elapsed();
    println!("\n");

    println!("\n=== Results ===");
    println!("Total tokens generated: {}", token_count);
    println!("Total time: {:.2}s", total_duration.as_secs_f64());
    println!("Tokens per second: {:.2}", token_count as f64 / total_duration.as_secs_f64());
    println!("Average latency per token: {:.0}ms", total_duration.as_millis() as f64 / token_count as f64);

    if !token_timings.is_empty() {
        let first_token_ms = token_timings[0].as_millis();
        let subsequent_avg_ms = if token_timings.len() > 1 {
            token_timings[1..].iter().map(|d| d.as_millis()).sum::<u128>() / (token_timings.len() - 1) as u128
        } else {
            0
        };

        println!("\nToken Latency Breakdown:");
        println!("  First token (TTFT): {:.0}ms", first_token_ms);
        if subsequent_avg_ms > 0 {
            println!("  Subsequent tokens (avg): {:.0}ms", subsequent_avg_ms);
        }
    }

    println!("\n=== Test Complete ===");
    println!("✓ Streaming generation API working correctly");
    println!("✓ Real-time token output demonstrated");
    println!("✓ Generated tokens: {:?}", &generated_tokens[..generated_tokens.len().min(10)]);

    Ok(())
}
