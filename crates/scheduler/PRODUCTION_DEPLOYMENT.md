# Production Deployment Guide

## Overview

This guide covers production-ready features for deploying GoGrid workers, with special emphasis on **resource management** to prevent desktop latency when running inference on user machines.

## Key Features for Production

### 1. GPU Resource Limiting

GoGrid workers can be configured to use only a subset of available GPU resources, ensuring they don't impact desktop performance:

- **VRAM Usage Limits**: Restrict how much GPU memory the worker can use (e.g., 50-70%)
- **Compute Time Throttling**: Add delays between inference requests to give desktop apps GPU time
- **Batch Size Limiting**: Restrict batch sizes to limit memory spikes

### 2. Process Priority Management

On Unix systems (Linux/macOS), workers automatically set low process priority:

- **CPU Priority (nice)**: Default priority of 10-15 (low/very low)
- **I/O Priority (ionice)**: Default class 3 (idle) to minimize disk impact
- **Automatic Application**: Applied when pipeline is created

### 3. Three Resource Modes

#### Conservative Mode (Desktop-Friendly)
```rust
use corpgrid_scheduler::resource_manager::ResourceConfig;

let config = ResourceConfig::conservative();
// - Max VRAM: 50%
// - Max Compute: 60%
// - Process Priority: 15 (very low)
// - Request Throttling: 100ms minimum interval
// - Max Batch Size: 2
```

**Use for**: Desktop machines where user experience is critical

#### Default Mode (Balanced)
```rust
let config = ResourceConfig::default();
// - Max VRAM: 70%
// - Max Compute: 80%
// - Process Priority: 10 (low)
// - Request Throttling: 50ms minimum interval
// - Max Batch Size: 4
```

**Use for**: Dedicated inference machines with occasional desktop use

#### Aggressive Mode (Server)
```rust
let config = ResourceConfig::aggressive();
// - Max VRAM: 95%
// - Max Compute: 95%
// - Process Priority: 0 (normal)
// - Request Throttling: Disabled
// - Max Batch Size: 16
```

**Use for**: Headless servers and dedicated GPU nodes

## Usage Example

### Basic Setup with Resource Limits

```rust
use corpgrid_scheduler::heterogeneous_pipeline::HeterogeneousPipeline;
use corpgrid_scheduler::resource_manager::ResourceConfig;
use corpgrid_scheduler::model_hosting::GpuDevice;

// Choose appropriate mode for your deployment
let config = ResourceConfig::conservative(); // or default() or aggressive()

// Create pipeline with resource limits
let pipeline = HeterogeneousPipeline::new(
    &devices,
    "/path/to/model",
    Some(config)
)?;

// Resource limits are now enforced automatically
```

### Custom Resource Configuration

```rust
use corpgrid_scheduler::resource_manager::ResourceConfig;

let config = ResourceConfig {
    max_vram_usage_percent: 0.6,     // Use 60% of available VRAM
    max_compute_usage_percent: 0.7,  // Target 70% GPU utilization
    process_priority: 12,             // Low priority (Unix only)
    io_priority_class: 3,             // Idle I/O (Linux only)
    io_priority_level: 7,             // Lowest within idle class
    enable_auto_throttle: true,       // Enable request throttling
    min_request_interval_ms: 75,      // 75ms between requests
    max_batch_size: 3,                // Limit batch size
};

config.validate()?; // Ensure values are valid
```

### Integrating Resource Checks

```rust
// Before processing each request
let estimated_vram = calculate_model_vram_requirements();

match pipeline.check_resource_limits(estimated_vram).await {
    Ok(_) => {
        // Resource limits OK, apply throttling
        pipeline.throttle_if_needed().await;

        // Process inference request
        let result = pipeline_executor.infer(...).await?;
    }
    Err(e) => {
        // Resource limit exceeded, reject or queue request
        warn!("Request blocked due to resource limits: {}", e);
    }
}
```

### Monitoring Resource Usage

```rust
// Get current resource statistics
let stats = pipeline.get_resource_stats().await;

println!("Total requests: {}", stats.total_requests);
println!("Throttled: {} ({:.1}%)",
    stats.throttled_requests,
    stats.throttle_rate * 100.0
);

// Log for monitoring
info!(
    total_requests = stats.total_requests,
    throttled = stats.throttled_requests,
    throttle_rate = format!("{:.1}%", stats.throttle_rate * 100.0),
    "Resource usage stats"
);
```

## Production INT8/INT4 Quantization

Combine resource limiting with quantization for optimal memory efficiency:

```rust
// Use quantized model with conservative resource limits
let config = ResourceConfig::conservative();

// Load quantized GGUF model (2-4x memory reduction)
let pipeline = HeterogeneousPipeline::new(
    &devices,
    "/path/to/model.Q4_K_M.gguf",  // INT4 quantized
    Some(config)
)?;

// Check if quantization is active
if pipeline.is_quantized() {
    if let Some(quant_model) = pipeline.quantized_model() {
        println!("Using {} quantization", quant_model.format);

        if let Some(metadata) = &quant_model.metadata {
            println!("Quantization type: {:?}", metadata.quantization_type);
        }
    }
}
```

### Quantization + Resource Limiting Benefits

| Configuration | VRAM Usage | Desktop Impact | Throughput |
|--------------|------------|----------------|------------|
| FP16 + Aggressive | 100% | High | 100% |
| FP16 + Conservative | 50% | Minimal | ~60% |
| INT8 + Conservative | 25% | Minimal | ~70% |
| INT4 + Conservative | 12.5% | Minimal | ~75% |

## Platform-Specific Considerations

### macOS (Metal)

```rust
#[cfg(target_os = "macos")]
{
    // Metal backend with conservative limits
    let config = ResourceConfig::conservative();

    let devices = vec![GpuDevice {
        backend: GpuBackend::Metal,
        device_index: 0,
        vram_total_bytes: 16 * 1024 * 1024 * 1024, // 16 GB unified memory
        // ...
    }];

    let pipeline = HeterogeneousPipeline::new(&devices, model_path, Some(config))?;
}
```

**Notes**:
- Unified memory shared with CPU
- Conservative mode recommended for desktops
- Process priority works via `renice`
- No `ionice` support (macOS limitation)

### Linux (CUDA)

```rust
#[cfg(target_os = "linux")]
{
    // CUDA backend with full priority control
    let config = ResourceConfig::conservative();

    let devices = vec![GpuDevice {
        backend: GpuBackend::Cuda,
        device_index: 0,
        vram_total_bytes: 8 * 1024 * 1024 * 1024, // 8 GB VRAM
        // ...
    }];

    let pipeline = HeterogeneousPipeline::new(&devices, model_path, Some(config))?;
}
```

**Notes**:
- Dedicated VRAM (not shared with CPU)
- Full `renice` and `ionice` support
- Can query GPU utilization via nvidia-smi
- Consider X11/Wayland desktop impact

### Windows (CUDA)

```rust
#[cfg(target_os = "windows")]
{
    // CUDA backend (priority management not yet implemented)
    let config = ResourceConfig::conservative();

    let devices = vec![GpuDevice {
        backend: GpuBackend::Cuda,
        device_index: 0,
        vram_total_bytes: 8 * 1024 * 1024 * 1024,
        // ...
    }];

    let pipeline = HeterogeneousPipeline::new(&devices, model_path, Some(config))?;
}
```

**Notes**:
- Process priority management not yet implemented
- GPU throttling still works
- Consider Windows desktop compositor impact

## Testing Resource Limiting

Run the included example to test different resource modes:

```bash
# Test conservative mode (desktop-friendly)
cargo run --example test_resource_limiting -- --mode conservative

# Test default mode (balanced)
cargo run --example test_resource_limiting -- --mode default

# Test aggressive mode (server)
cargo run --example test_resource_limiting -- --mode aggressive
```

## Deployment Checklist

### For Desktop Deployments

- [ ] Use `ResourceConfig::conservative()`
- [ ] Enable auto-throttling (default: enabled)
- [ ] Set max VRAM to 50-60%
- [ ] Set process priority to 12-15
- [ ] Use INT4/INT8 quantization if possible
- [ ] Monitor user feedback on desktop responsiveness
- [ ] Test with typical desktop workloads (browser, IDE, etc.)

### For Server Deployments

- [ ] Use `ResourceConfig::aggressive()`
- [ ] Disable auto-throttling for maximum throughput
- [ ] Set max VRAM to 90-95%
- [ ] Set normal process priority (0)
- [ ] Monitor GPU utilization (target 80-95%)
- [ ] Load test with peak expected traffic
- [ ] Set up metrics/alerting for resource exhaustion

### For Hybrid Deployments

- [ ] Use `ResourceConfig::default()`
- [ ] Enable auto-throttling with 50-75ms interval
- [ ] Set max VRAM to 70%
- [ ] Set process priority to 8-10
- [ ] Monitor both inference performance and desktop impact
- [ ] Consider time-based scheduling (aggressive at night, conservative during day)

## Monitoring and Observability

### Key Metrics to Track

```rust
// Periodically log resource stats
tokio::spawn(async move {
    loop {
        let stats = pipeline.get_resource_stats().await;

        metrics::gauge!("inference.requests.total", stats.total_requests as f64);
        metrics::gauge!("inference.requests.throttled", stats.throttled_requests as f64);
        metrics::gauge!("inference.throttle_rate", stats.throttle_rate);

        tokio::time::sleep(Duration::from_secs(60)).await;
    }
});
```

### Recommended Alerts

- **High Throttle Rate** (>50%): May need to increase resource limits or add capacity
- **Low GPU Utilization** (<20% on server): Resource limits too conservative
- **VRAM Limit Errors**: Need more VRAM or smaller model/batch size
- **Desktop Lag Reports**: Resource limits not conservative enough

## Performance Tuning

### Finding the Right Balance

1. **Start Conservative**: Begin with `ResourceConfig::conservative()`
2. **Monitor Impact**: Track both inference performance and system responsiveness
3. **Gradually Increase**: Slowly raise VRAM/compute limits if desktop impact is minimal
4. **A/B Test**: Compare different configurations with real workloads
5. **User Feedback**: Collect feedback from users about desktop performance

### Optimization Tips

- **Quantization First**: Use INT8/INT4 before relaxing resource limits
- **Batch Wisely**: Smaller batches = less memory spikes = less desktop impact
- **Time-Based Limits**: Aggressive during off-hours, conservative during work hours
- **Per-App Limits**: Different limits for different applications (chat vs. batch processing)

## Troubleshooting

### Workers Still Impacting Desktop

- Reduce `max_vram_usage_percent` (try 40-50%)
- Increase `min_request_interval_ms` (try 150-200ms)
- Reduce `max_batch_size` to 1-2
- Verify process priority is being set (check with `ps` or `top`)

### Workers Too Slow

- Increase `max_vram_usage_percent` (try 80-90%)
- Decrease `min_request_interval_ms` (try 25-50ms)
- Increase `max_batch_size` to 4-8
- Consider using `aggressive()` mode on dedicated hardware
- Check if quantization is reducing accuracy too much

### VRAM Limit Errors

- Model too large for available VRAM
- Use more aggressive quantization (INT4 instead of INT8)
- Reduce batch size
- Distribute across multiple GPUs
- Consider smaller model variant

## Future Enhancements

Planned improvements to resource management:

- **Dynamic GPU Utilization Monitoring**: Automatically detect when GPU is busy with desktop apps
- **Adaptive Throttling**: Automatically adjust limits based on GPU load
- **Windows Priority Support**: Implement process priority on Windows
- **GPU Frequency Scaling**: Reduce GPU clock speeds to save power
- **Per-User Limits**: Different limits for different users on shared systems

## Example: Full Production Setup

```rust
use corpgrid_scheduler::{
    heterogeneous_pipeline::HeterogeneousPipeline,
    resource_manager::{ResourceConfig, ResourceStats},
    model_hosting::GpuDevice,
};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Determine resource mode based on environment
    let is_server = std::env::var("DEPLOYMENT_MODE")
        .map(|m| m == "server")
        .unwrap_or(false);

    let config = if is_server {
        info!("Starting in SERVER mode (aggressive resource usage)");
        ResourceConfig::aggressive()
    } else {
        info!("Starting in DESKTOP mode (conservative resource usage)");
        ResourceConfig::conservative()
    };

    // Create pipeline with resource limits
    let pipeline = HeterogeneousPipeline::new(
        &get_available_devices()?,
        &get_model_path()?,
        Some(config)
    )?;

    // Log resource configuration
    let rm = pipeline.resource_manager().lock().await;
    info!(
        vram_limit = format!("{:.0}%", rm.config().max_vram_usage_percent * 100.0),
        compute_limit = format!("{:.0}%", rm.config().max_compute_usage_percent * 100.0),
        priority = rm.config().process_priority,
        throttling = rm.config().enable_auto_throttle,
        "Resource limits configured"
    );
    drop(rm);

    // Start metrics reporting
    spawn_metrics_reporter(pipeline.clone());

    // Main inference loop
    loop {
        match receive_inference_request().await {
            Ok(request) => {
                // Check resource limits
                if let Err(e) = pipeline.check_resource_limits(request.estimated_vram).await {
                    warn!("Request rejected due to resource limits: {}", e);
                    send_error_response(&request, "Resource limits exceeded").await?;
                    continue;
                }

                // Apply throttling
                pipeline.throttle_if_needed().await;

                // Process request
                match process_inference(pipeline, request).await {
                    Ok(result) => send_result(&result).await?,
                    Err(e) => {
                        warn!("Inference failed: {}", e);
                        send_error_response(&request, "Inference failed").await?;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to receive request: {}", e);
                break;
            }
        }
    }

    Ok(())
}

fn spawn_metrics_reporter(pipeline: HeterogeneousPipeline) {
    tokio::spawn(async move {
        loop {
            let stats = pipeline.get_resource_stats().await;

            info!(
                total_requests = stats.total_requests,
                throttled = stats.throttled_requests,
                throttle_rate = format!("{:.1}%", stats.throttle_rate * 100.0),
                "Resource usage stats"
            );

            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    });
}
```

## Summary

The resource management system ensures GoGrid workers are production-ready for both desktop and server deployments:

- **Desktop-Friendly**: Conservative mode prevents GPU monopolization
- **Flexible**: Three preset modes + full customization
- **Automatic**: Priority management applied automatically (Unix)
- **Observable**: Built-in metrics for monitoring
- **Tested**: Includes test examples for validation

For questions or issues, see the [GitHub repository](https://github.com/yourusername/gogrid).
