#[cfg(target_os = "macos")]
use anyhow::Result;
use corpgrid_common::{GpuBackend, GpuInfo};
use metal::*;
use tracing::{debug, info};

pub fn detect_metal_gpus() -> Result<Vec<GpuInfo>> {
    let devices = Device::all();
    info!(count = devices.len(), "Metal devices detected");

    let mut gpus = Vec::new();

    for (i, device) in devices.iter().enumerate() {
        let name = device.name().to_string();

        // Get recommended working set size as a proxy for VRAM
        // On unified memory systems, this represents available GPU memory
        let vram_bytes = device.recommended_max_working_set_size();

        // Alternative: use max_buffer_length for M1/M2/M3
        // let vram_bytes = device.max_buffer_length();

        let driver_version = "Metal".to_string();

        debug!(
            index = i,
            name = %name,
            vram_gb = vram_bytes / (1024 * 1024 * 1024),
            "Found Metal GPU"
        );

        gpus.push(GpuInfo {
            name,
            backend: GpuBackend::Metal,
            vram_bytes,
            driver_version,
            compute_capability: None,
        });
    }

    Ok(gpus)
}
