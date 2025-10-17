#[cfg(target_os = "linux")]
use anyhow::Result;
use corpgrid_common::{GpuBackend, GpuInfo};
use nvml_wrapper::Nvml;
use tracing::{debug, info, warn};

pub fn detect_cuda_gpus() -> Result<Vec<GpuInfo>> {
    match Nvml::init() {
        Ok(nvml) => {
            let device_count = nvml.device_count()?;
            info!(count = device_count, "CUDA GPUs detected");

            let mut gpus = Vec::new();

            for i in 0..device_count {
                match nvml.device_by_index(i) {
                    Ok(device) => {
                        let name = device.name().unwrap_or_else(|_| format!("GPU {}", i));

                        let memory_info = device.memory_info()?;
                        let vram_bytes = memory_info.total;

                        let driver_version = nvml
                            .sys_driver_version()
                            .unwrap_or_else(|_| "Unknown".to_string());

                        let compute_capability = device
                            .cuda_compute_capability()
                            .map(|cc| format!("{}.{}", cc.major, cc.minor))
                            .ok();

                        debug!(
                            index = i,
                            name = %name,
                            vram_gb = vram_bytes / (1024 * 1024 * 1024),
                            driver = %driver_version,
                            compute_cap = ?compute_capability,
                            "Found CUDA GPU"
                        );

                        gpus.push(GpuInfo {
                            name,
                            backend: GpuBackend::Cuda,
                            vram_bytes,
                            driver_version,
                            compute_capability,
                        });
                    }
                    Err(e) => {
                        warn!(index = i, error = %e, "Failed to query GPU");
                    }
                }
            }

            Ok(gpus)
        }
        Err(e) => {
            debug!(error = %e, "NVML initialization failed - no CUDA GPUs available");
            Ok(vec![])
        }
    }
}
