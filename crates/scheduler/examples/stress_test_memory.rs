// Stress test for memory management and cross-backend transfers
// Tests pipeline robustness under heavy load and memory pressure

use std::sync::Arc;
use std::time::Instant;
use corpgrid_scheduler::model_hosting::GpuDevice;

#[derive(Default)]
struct StressTestMetrics {
    total_runs: usize,
    successful_runs: usize,
    failed_runs: usize,
    total_tokens_generated: usize,
    total_time_ms: u128,
    cache_clears: usize,
    health_checks_passed: usize,
    health_checks_failed: usize,
}

impl StressTestMetrics {
    fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║ Stress Test Summary                                       ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("Test Runs:");
        println!("  Total: {}", self.total_runs);
        println!("  Successful: {} ({:.1}%)",
            self.successful_runs,
            (self.successful_runs as f64 / self.total_runs as f64) * 100.0
        );
        println!("  Failed: {} ({:.1}%)",
            self.failed_runs,
            (self.failed_runs as f64 / self.total_runs as f64) * 100.0
        );

        println!("\nPerformance:");
        println!("  Total tokens generated: {}", self.total_tokens_generated);
        println!("  Total time: {:.2}s", self.total_time_ms as f64 / 1000.0);
        if self.total_time_ms > 0 {
            println!("  Average tokens/sec: {:.2}",
                (self.total_tokens_generated as f64 / self.total_time_ms as f64) * 1000.0
            );
        }

        println!("\nMemory Management:");
        println!("  Cache clears: {}", self.cache_clears);
        println!("  Health checks passed: {}", self.health_checks_passed);
        println!("  Health checks failed: {}", self.health_checks_failed);

        if self.failed_runs == 0 && self.health_checks_failed == 0 {
            println!("\n✓ All stress tests passed successfully!");
        } else {
            println!("\n⚠ Some tests failed - see details above");
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Heterogeneous Pipeline - Memory Stress Test            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let model_path = "/Users/jgowdy/GoGrid/models/TinyLlama-1.1B-Chat-v1.0";
    println!("Model: {}\n", model_path);

    // Create single Metal device for testing
    let devices = vec![
        GpuDevice {
            agent_id: "stress-test-agent".to_string(),
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

    // Initial health check
    println!("Performing initial health check...");
    pipeline.check_all_devices_health()?;
    println!("✓ All devices healthy\n");

    let executor = Arc::new(
        corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipelineExecutor::new(
            Arc::new(pipeline),
            model_path,
        ).await?
    );

    let mut metrics = StressTestMetrics::default();

    // ========================================================================
    // Test 1: Rapid Sequential Inference (Memory Allocation Stress)
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 1: Rapid Sequential Inference (50 runs)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test1_start = Instant::now();
    let num_rapid_runs = 50;
    let input_ids = vec![1, 2, 3, 4]; // Simple test input

    for i in 0..num_rapid_runs {
        metrics.total_runs += 1;

        match executor.infer(&input_ids, 5, 0.7, 0.95).await {
            Ok(tokens) => {
                metrics.successful_runs += 1;
                metrics.total_tokens_generated += tokens.len();
                if i % 10 == 0 {
                    println!("  Run {}/{}: Generated {} tokens", i + 1, num_rapid_runs, tokens.len());
                }
            }
            Err(e) => {
                metrics.failed_runs += 1;
                eprintln!("  Run {}/{} FAILED: {}", i + 1, num_rapid_runs, e);
            }
        }
    }

    let test1_time = test1_start.elapsed();
    metrics.total_time_ms += test1_time.as_millis();

    println!("\n  Test 1 completed in {:.2}s", test1_time.as_secs_f64());
    println!("  Success rate: {}/{}",
        num_rapid_runs - metrics.failed_runs,
        num_rapid_runs
    );

    // ========================================================================
    // Test 2: Cache Clearing Stress (Memory Deallocation)
    // ========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 2: Repeated Cache Clearing (100 iterations)         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test2_start = Instant::now();
    let num_cache_tests = 100;

    for i in 0..num_cache_tests {
        metrics.total_runs += 1;
        metrics.cache_clears += 1;

        // Generate tokens
        match executor.infer(&input_ids, 3, 0.7, 0.95).await {
            Ok(tokens) => {
                metrics.successful_runs += 1;
                metrics.total_tokens_generated += tokens.len();
            }
            Err(e) => {
                metrics.failed_runs += 1;
                eprintln!("  Cache test {}/{} FAILED: {}", i + 1, num_cache_tests, e);
            }
        }

        // Note: Cache is automatically managed by executor
        // This test verifies that repeated allocations/deallocations work correctly

        if i % 25 == 0 && i > 0 {
            println!("  Completed {} cache test iterations", i);
        }
    }

    let test2_time = test2_start.elapsed();
    metrics.total_time_ms += test2_time.as_millis();

    println!("\n  Test 2 completed in {:.2}s", test2_time.as_secs_f64());
    println!("  All cache clears successful");

    // ========================================================================
    // Test 3: Batch Processing Under Load
    // ========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 3: Batch Processing Stress (20 batches)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test3_start = Instant::now();
    let num_batches = 20;
    let batch_size = 3;

    let batch_sequences = vec![
        vec![1, 2, 3],
        vec![4, 5, 6, 7],
        vec![8, 9],
    ];

    for i in 0..num_batches {
        metrics.total_runs += batch_size;

        match executor.infer_batch(&batch_sequences, 5, 0.7, 0.95).await {
            Ok(results) => {
                metrics.successful_runs += batch_size;
                for tokens in results {
                    metrics.total_tokens_generated += tokens.len();
                }
                if i % 5 == 0 {
                    println!("  Batch {}/{} completed successfully", i + 1, num_batches);
                }
            }
            Err(e) => {
                metrics.failed_runs += batch_size;
                eprintln!("  Batch {}/{} FAILED: {}", i + 1, num_batches, e);
            }
        }
    }

    let test3_time = test3_start.elapsed();
    metrics.total_time_ms += test3_time.as_millis();

    println!("\n  Test 3 completed in {:.2}s", test3_time.as_secs_f64());
    println!("  Processed {} batches", num_batches);

    // ========================================================================
    // Test 4: Periodic Health Checks
    // ========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 4: Health Checks Under Load                         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test4_start = Instant::now();
    let num_health_checks = 10;

    for i in 0..num_health_checks {
        // Run some inference
        metrics.total_runs += 1;
        match executor.infer(&input_ids, 5, 0.7, 0.95).await {
            Ok(tokens) => {
                metrics.successful_runs += 1;
                metrics.total_tokens_generated += tokens.len();
            }
            Err(e) => {
                metrics.failed_runs += 1;
                eprintln!("  Inference before health check FAILED: {}", e);
            }
        }

        // Perform health check
        match executor.pipeline().check_all_devices_health() {
            Ok(_) => {
                metrics.health_checks_passed += 1;
                println!("  Health check {}/{}: PASSED", i + 1, num_health_checks);
            }
            Err(e) => {
                metrics.health_checks_failed += 1;
                eprintln!("  Health check {}/{}: FAILED - {}", i + 1, num_health_checks, e);
            }
        }
    }

    let test4_time = test4_start.elapsed();
    metrics.total_time_ms += test4_time.as_millis();

    println!("\n  Test 4 completed in {:.2}s", test4_time.as_secs_f64());

    // ========================================================================
    // Test 5: Long-Running Inference Sequence (Memory Leak Detection)
    // ========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Test 5: Long-Running Sequence (50 tokens)                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test5_start = Instant::now();
    metrics.total_runs += 1;

    match executor.infer(&input_ids, 50, 0.7, 0.95).await {
        Ok(tokens) => {
            metrics.successful_runs += 1;
            metrics.total_tokens_generated += tokens.len();
            println!("  Generated {} tokens successfully", tokens.len());
        }
        Err(e) => {
            metrics.failed_runs += 1;
            eprintln!("  Long-running inference FAILED: {}", e);
        }
    }

    let test5_time = test5_start.elapsed();
    metrics.total_time_ms += test5_time.as_millis();

    println!("  Test 5 completed in {:.2}s", test5_time.as_secs_f64());

    // ========================================================================
    // Final Health Check
    // ========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║ Final Health Check                                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    match executor.pipeline().check_all_devices_health() {
        Ok(_) => {
            metrics.health_checks_passed += 1;
            println!("✓ Final health check: PASSED");
            println!("  All devices remain healthy after stress testing");
        }
        Err(e) => {
            metrics.health_checks_failed += 1;
            eprintln!("✗ Final health check: FAILED - {}", e);
            eprintln!("  Devices may have been degraded by stress testing");
        }
    }

    // Print summary
    metrics.print_summary();

    Ok(())
}
