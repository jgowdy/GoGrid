#[cfg(windows)]
use anyhow::Result;
use corpgrid_common::{GpuBackend, GpuInfo};
use tracing::{debug, info, warn};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub fn detect_windows_gpus() -> Result<Vec<GpuInfo>> {
    unsafe {
        let mut factory: Option<IDXGIFactory> = None;
        CreateDXGIFactory(&IDXGIFactory::IID, &mut factory as *mut _ as *mut _)?;

        let factory = factory.ok_or_else(|| anyhow::anyhow!("Failed to create DXGI factory"))?;

        let mut gpus = Vec::new();
        let mut adapter_index = 0;

        loop {
            match factory.EnumAdapters(adapter_index) {
                Ok(adapter) => {
                    let mut desc = DXGI_ADAPTER_DESC::default();
                    adapter.GetDesc(&mut desc)?;

                    // Convert wide string to Rust string
                    let name = String::from_utf16_lossy(&desc.Description)
                        .trim_end_matches('\0')
                        .to_string();

                    let vram_bytes = desc.DedicatedVideoMemory as u64;

                    // Determine backend based on vendor
                    // NVIDIA: VendorId = 0x10DE
                    let backend = if desc.VendorId == 0x10DE {
                        GpuBackend::Cuda
                    } else {
                        // AMD, Intel, etc. - would need DirectCompute or other backends
                        // For now, skip non-NVIDIA on Windows
                        adapter_index += 1;
                        continue;
                    };

                    debug!(
                        index = adapter_index,
                        name = %name,
                        vram_gb = vram_bytes / (1024 * 1024 * 1024),
                        vendor_id = desc.VendorId,
                        "Found Windows GPU"
                    );

                    gpus.push(GpuInfo {
                        name,
                        backend,
                        vram_bytes,
                        driver_version: "Windows".to_string(),
                        compute_capability: None,
                    });

                    adapter_index += 1;
                }
                Err(_) => break,
            }
        }

        info!(count = gpus.len(), "Windows GPUs detected");
        Ok(gpus)
    }
}
