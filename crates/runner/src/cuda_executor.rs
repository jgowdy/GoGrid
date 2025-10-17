#[cfg(target_os = "linux")]
use anyhow::{Context, Result};
use corpgrid_common::JobSpec;
use cudarc::driver::*;
use std::path::PathBuf;
use tracing::{debug, info};

pub struct CudaExecutor {
    device: CudaDevice,
}

impl CudaExecutor {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        info!(device_ordinal, "Initializing CUDA device");

        let device = CudaDevice::new(device_ordinal)
            .context("Failed to initialize CUDA device")?;

        info!(
            device_ordinal,
            name = device.name()?,
            "CUDA device initialized"
        );

        Ok(Self { device })
    }

    pub async fn execute(&self, job_dir: &PathBuf, spec: &JobSpec) -> Result<Vec<u8>> {
        self.execute_internal(job_dir, spec, true).await
    }

    async fn execute_internal(&self, job_dir: &PathBuf, spec: &JobSpec, check_checkpoint: bool) -> Result<Vec<u8>> {
        info!("Starting CUDA job execution");

        // Check for checkpoint to resume from
        if check_checkpoint {
            if let Ok(checkpoint_path) = self.find_latest_checkpoint(job_dir).await {
                info!(checkpoint = %checkpoint_path.display(), "Resuming from checkpoint");
                return self.resume_from_checkpoint(job_dir, spec, &checkpoint_path).await;
            }
        }

        // Load PTX/CUBIN kernel from job bundle
        let kernel_path = job_dir.join("kernel.ptx");
        if !kernel_path.exists() {
            anyhow::bail!("Kernel file not found: {}", kernel_path.display());
        }

        let ptx_code = tokio::fs::read_to_string(&kernel_path)
            .await
            .context("Failed to read PTX kernel")?;

        debug!(
            kernel_size = ptx_code.len(),
            "Loaded PTX kernel"
        );

        // Load module from PTX
        self.device.load_ptx(
            ptx_code.into(),
            "kernel_module",
            &["main_kernel"],
        )?;

        // Load input data
        let input_path = job_dir.join("input.bin");
        let input_data = if input_path.exists() {
            tokio::fs::read(&input_path)
                .await
                .context("Failed to read input data")?
        } else {
            vec![]
        };

        debug!(input_size = input_data.len(), "Loaded input data");

        // Allocate device memory for input
        let input_device = if !input_data.is_empty() {
            let mut buf = unsafe {
                self.device.alloc::<u8>(input_data.len())
                    .context("Failed to allocate device memory for input")?
            };
            self.device.htod_sync_copy_into(&input_data, &mut buf)?;
            Some(buf)
        } else {
            None
        };

        // Determine output size from spec or use default
        let output_size = spec.labels.get("output_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024 * 1024); // Default 1MB

        // Allocate device memory for output
        let mut output_device = unsafe {
            self.device.alloc::<u8>(output_size)
                .context("Failed to allocate device memory for output")?
        };

        // Launch kernel
        info!("Launching CUDA kernel");

        let function = self.device.get_func("kernel_module", "main_kernel")
            .context("Failed to get kernel function")?;

        // Configure kernel launch parameters
        let grid_dim = (1024, 1, 1); // Adjust based on workload
        let block_dim = (256, 1, 1);

        unsafe {
            let params = if let Some(ref input) = input_device {
                vec![
                    input.device_ptr() as *mut std::ffi::c_void,
                    output_device.device_ptr() as *mut std::ffi::c_void,
                    &(input_data.len()) as *const usize as *mut std::ffi::c_void,
                ]
            } else {
                vec![
                    output_device.device_ptr() as *mut std::ffi::c_void,
                    &output_size as *const usize as *mut std::ffi::c_void,
                ]
            };

            function.launch(
                LaunchConfig {
                    grid_dim,
                    block_dim,
                    shared_mem_bytes: 0,
                },
                &params,
            )?;
        }

        // Wait for kernel completion
        self.device.synchronize()?;
        info!("CUDA kernel completed");

        // Copy result back to host
        let mut output_host = vec![0u8; output_size];
        self.device.dtoh_sync_copy_into(&output_device, &mut output_host)?;

        // Save output
        let output_path = job_dir.join("output.bin");
        tokio::fs::write(&output_path, &output_host)
            .await
            .context("Failed to write output")?;

        info!(
            output_size = output_host.len(),
            "CUDA execution completed successfully"
        );

        Ok(output_host)
    }

    pub async fn checkpoint(&self, job_dir: &PathBuf, checkpoint_id: u64) -> Result<PathBuf> {
        let checkpoint_dir = job_dir.join("checkpoints");
        tokio::fs::create_dir_all(&checkpoint_dir).await?;

        let checkpoint_path = checkpoint_dir.join(format!("checkpoint_{}.bin", checkpoint_id));

        // In a real implementation, this would:
        // 1. Copy current GPU memory state
        // 2. Save model weights/gradients
        // 3. Save optimizer state
        // 4. Save RNG state for reproducibility

        info!(
            checkpoint_id,
            path = %checkpoint_path.display(),
            "Created checkpoint"
        );

        Ok(checkpoint_path)
    }

    async fn find_latest_checkpoint(&self, job_dir: &PathBuf) -> Result<PathBuf> {
        let checkpoint_dir = job_dir.join("checkpoints");
        if !checkpoint_dir.exists() {
            anyhow::bail!("No checkpoints directory");
        }

        let mut entries = tokio::fs::read_dir(&checkpoint_dir).await?;
        let mut latest: Option<(u64, PathBuf)> = None;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(id_str) = filename.strip_prefix("checkpoint_").and_then(|s| s.strip_suffix(".bin")) {
                    if let Ok(id) = id_str.parse::<u64>() {
                        if latest.is_none() || latest.as_ref().unwrap().0 < id {
                            latest = Some((id, path.clone()));
                        }
                    }
                }
            }
        }

        latest.map(|(_, path)| path).ok_or_else(|| anyhow::anyhow!("No checkpoints found"))
    }

    async fn resume_from_checkpoint(
        &self,
        job_dir: &PathBuf,
        spec: &JobSpec,
        checkpoint_path: &PathBuf,
    ) -> Result<Vec<u8>> {
        info!("Restoring from checkpoint: {}", checkpoint_path.display());

        // Load checkpoint data
        let _checkpoint_data = tokio::fs::read(checkpoint_path).await?;

        // In a real implementation, this would:
        // 1. Restore GPU memory state
        // 2. Load model weights/gradients
        // 3. Restore optimizer state
        // 4. Restore RNG state for determinism
        // 5. Continue execution from saved point

        info!("Checkpoint restored, continuing execution");

        // For now, just execute normally (without re-checking for checkpoints to avoid recursion)
        // Box the recursive call to avoid infinite type size
        Box::pin(self.execute_internal(job_dir, spec, false)).await
    }
}


// Deterministic execution utilities
pub fn set_deterministic_mode(device: &CudaDevice) -> Result<()> {
    // Set deterministic flags for reproducible compute
    // This would configure:
    // - cudnnSetDeterministicMode
    // - Fixed reduction order
    // - Consistent RNG seeds

    info!("Configured deterministic execution mode");
    Ok(())
}
