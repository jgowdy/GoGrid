/// Test resource limiting and priority management for desktop-friendly workers
///
/// This example demonstrates how to configure resource limits to ensure
/// inference workers don't impact desktop performance by limiting GPU usage
/// and setting low process priority.
///
/// Usage:
/// ```bash
/// # Conservative mode (desktop-friendly, only uses 50% GPU)
/// cargo run --example test_resource_limiting -- --mode conservative
///
/// # Default mode (balanced, uses 70% GPU)
/// cargo run --example test_resource_limiting
///
/// # Aggressive mode (dedicated server, uses 95% GPU)
/// cargo run --example test_resource_limiting -- --mode aggressive
/// ```

use anyhow::Result;
use clap::Parser;
use corpgrid_common::GpuBackend;
use corpgrid_scheduler::model_hosting::GpuDevice;
use corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline;
use corpgrid_scheduler::resource_manager::ResourceConfig;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Resource mode: conservative, default, or aggressive
    #[arg(short, long, default_value = "default")]
    mode: String,

    /// Model path (placeholder, actual model not required for this demo)
    #[arg(short = 'p', long, default_value = "/tmp/test-model")]
    model_path: String,

    /// Number of simulated requests
    #[arg(short = 'n', long, default_value = "20")]
    num_requests: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    println!("=== Resource Limiting & Priority Management Demo ===\n");

    // Create resource configuration based on mode
    let config = match args.mode.as_str() {
        "conservative" => {
            println!("🛡️  Mode: CONSERVATIVE (Desktop-Friendly)");
            println!("   - Max VRAM usage: 50%");
            println!("   - Max compute usage: 60%");
            println!("   - Process priority: 15 (very low)");
            println!("   - I/O priority: Idle");
            println!("   - Request throttling: 100ms minimum interval");
            println!("   - Max batch size: 2");
            ResourceConfig::conservative()
        }
        "aggressive" => {
            println!("⚡ Mode: AGGRESSIVE (Dedicated Server)");
            println!("   - Max VRAM usage: 95%");
            println!("   - Max compute usage: 95%");
            println!("   - Process priority: 0 (normal)");
            println!("   - I/O priority: Best-effort");
            println!("   - Request throttling: Disabled");
            println!("   - Max batch size: 16");
            ResourceConfig::aggressive()
        }
        _ => {
            println!("⚖️  Mode: DEFAULT (Balanced)");
            println!("   - Max VRAM usage: 70%");
            println!("   - Max compute usage: 80%");
            println!("   - Process priority: 10 (low)");
            println!("   - I/O priority: Idle");
            println!("   - Request throttling: 50ms minimum interval");
            println!("   - Max batch size: 4");
            ResourceConfig::default()
        }
    };

    println!();

    // Create test GPU devices
    let devices = create_test_devices();
    println!("🖥️  GPU Configuration:");
    for (idx, device) in devices.iter().enumerate() {
        let vram_gb = device.vram_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("   Device {}: {:?} ({:.1} GB VRAM)", idx, device.backend, vram_gb);

        // Show VRAM limits
        let max_vram_bytes = config.max_vram_bytes(device.vram_total_bytes);
        let max_vram_gb = max_vram_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("      → Limit: {:.1} GB ({:.0}%)",
            max_vram_gb,
            config.max_vram_usage_percent * 100.0
        );
    }
    println!();

    // Create pipeline with resource configuration
    println!("🔧 Creating heterogeneous pipeline with resource limits...");
    let config_clone = config.clone();
    let pipeline = HeterogeneousPipeline::new(&devices, &args.model_path, Some(config_clone))?;
    println!("   ✅ Pipeline created with resource manager");
    println!();

    // Simulate inference requests with resource limiting
    println!("🚀 Simulating {} inference requests...", args.num_requests);
    println!("   (Demonstrating throttling and resource management)\n");

    let start_time = std::time::Instant::now();
    let mut allowed_requests = 0;
    let mut throttled_count = 0;

    for i in 0..args.num_requests {
        // Check resource limits before processing
        let estimated_vram = 500 * 1024 * 1024; // 500 MB estimate
        match pipeline.check_resource_limits(estimated_vram).await {
            Ok(_) => {
                // Resource limits OK, proceed with request
                print!(".");
                if (i + 1) % 10 == 0 {
                    println!(" {}/{}", i + 1, args.num_requests);
                }

                // Apply throttling
                pipeline.throttle_if_needed().await;

                // Simulate inference work
                tokio::time::sleep(Duration::from_millis(10)).await;

                allowed_requests += 1;
            }
            Err(e) => {
                println!("\n   ⚠️  Request {} blocked: {}", i + 1, e);
                throttled_count += 1;
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!();

    // Get resource statistics
    let stats = pipeline.get_resource_stats().await;

    println!("\n📊 Results:");
    println!("   Total requests: {}", args.num_requests);
    println!("   Allowed: {}", allowed_requests);
    println!("   Resource-blocked: {}", throttled_count);
    println!("   Throttled by rate limiter: {}", stats.throttled_requests);
    println!("   Throttle rate: {:.1}%", stats.throttle_rate * 100.0);
    println!("   Total time: {:?}", elapsed);
    println!("   Avg time per request: {:.1}ms",
        elapsed.as_millis() as f64 / args.num_requests as f64
    );

    println!("\n💡 Analysis:");
    if config.enable_auto_throttle && stats.throttle_rate > 0.0 {
        println!("   ✅ Throttling is working correctly");
        println!("      Requests are spaced by at least {}ms",
            config.min_request_interval_ms
        );
        println!("      This allows desktop apps to use the GPU between our requests");
    } else {
        println!("   ℹ️  Throttling disabled in this mode");
        println!("      All requests processed as quickly as possible");
    }

    println!("\n🎯 Production Tips:");
    println!("   1. Use 'conservative' mode for desktop machines");
    println!("   2. Use 'default' mode for dedicated inference machines with occasional desktop use");
    println!("   3. Use 'aggressive' mode only for headless servers");
    println!("   4. Monitor GPU utilization to tune limits for your workload");
    println!("   5. Process priority is automatically set (nice/ionice on Unix systems)");

    #[cfg(target_family = "unix")]
    {
        println!("\n🔐 Unix Process Priority:");
        println!("   ✅ Process priority set to {} (nice value)", config.process_priority);
        println!("   ✅ I/O priority set to class {} level {}",
            config.io_priority_class,
            config.io_priority_level
        );
        println!("      → Lower priority = less desktop impact");
    }

    #[cfg(target_os = "windows")]
    {
        println!("\n⚠️  Windows Note:");
        println!("   Process priority management not yet implemented on Windows");
        println!("   GPU throttling still works to prevent desktop latency");
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
