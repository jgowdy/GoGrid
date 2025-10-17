use anyhow::Result;
use corpgrid_common::{DeviceInfo, GpuInfo};
use uuid::Uuid;

/// Collect device information for registration
pub fn collect_device_info() -> Result<DeviceInfo> {
    let device_id = get_or_create_device_id()?;
    let hostname = hostname::get()?
        .to_string_lossy()
        .to_string();

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let gpus = detect_gpus()?;

    // Get system memory
    let memory_bytes = get_system_memory();

    // Get CPU cores
    let cpu_cores = num_cpus::get() as u32;

    Ok(DeviceInfo {
        device_id,
        hostname,
        os,
        arch,
        gpus,
        memory_bytes,
        cpu_cores,
        site: std::env::var("CORPGRID_SITE").ok(),
    })
}

fn get_or_create_device_id() -> Result<String> {
    // Try to load from file, otherwise generate
    let id_path = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("corpgrid")
        .join("device_id");

    if let Ok(id) = std::fs::read_to_string(&id_path) {
        Ok(id.trim().to_string())
    } else {
        let id = Uuid::new_v4().to_string();

        // Try to save
        if let Some(parent) = id_path.parent() {
            std::fs::create_dir_all(parent).ok();
            std::fs::write(&id_path, &id).ok();
        }

        Ok(id)
    }
}

fn detect_gpus() -> Result<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // CUDA GPUs (Linux)
    #[cfg(target_os = "linux")]
    {
        if let Ok(cuda_gpus) = crate::gpu_cuda::detect_cuda_gpus() {
            gpus.extend(cuda_gpus);
        }
    }

    // Metal GPUs (macOS)
    #[cfg(target_os = "macos")]
    {
        if let Ok(metal_gpus) = crate::gpu_metal::detect_metal_gpus() {
            gpus.extend(metal_gpus);
        }
    }

    // Windows GPUs (DXGI)
    #[cfg(windows)]
    {
        if let Ok(windows_gpus) = crate::gpu_windows::detect_windows_gpus() {
            gpus.extend(windows_gpus);
        }
    }

    Ok(gpus)
}

fn get_system_memory() -> u64 {
    // Cross-platform memory detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = sys_info::mem_info() {
            return info.total * 1024; // Convert KB to bytes
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(info) = sys_info::mem_info() {
            return info.total * 1024;
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(info) = sys_info::mem_info() {
            return info.total * 1024;
        }
    }

    16 * 1024 * 1024 * 1024 // Default to 16GB
}
