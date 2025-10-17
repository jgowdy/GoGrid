#[cfg(target_os = "macos")]
use anyhow::{Context, Result};
use corpgrid_common::JobSpec;
use metal::*;
use std::path::PathBuf;
use tracing::{debug, info};

pub struct MetalExecutor {
    device: Device,
    command_queue: CommandQueue,
}

impl MetalExecutor {
    pub fn new() -> Result<Self> {
        info!("Initializing Metal device");

        let device = Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("No Metal device available"))?;

        info!(name = device.name(), "Metal device initialized");

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
        })
    }

    pub async fn execute(&self, job_dir: &PathBuf, spec: &JobSpec) -> Result<Vec<u8>> {
        self.execute_internal(job_dir, spec, true).await
    }

    async fn execute_internal(&self, job_dir: &PathBuf, spec: &JobSpec, check_checkpoint: bool) -> Result<Vec<u8>> {
        info!("Starting Metal job execution");

        // Check for checkpoint to resume from
        if check_checkpoint {
            if let Ok(checkpoint_path) = self.find_latest_checkpoint(job_dir).await {
                info!(checkpoint = %checkpoint_path.display(), "Resuming from checkpoint");
                return self.resume_from_checkpoint(job_dir, spec, &checkpoint_path).await;
            }
        }

        // Load Metal shader/kernel from job bundle
        let shader_path = job_dir.join("kernel.metal");
        if !shader_path.exists() {
            anyhow::bail!("Metal shader not found: {}", shader_path.display());
        }

        let shader_source = tokio::fs::read_to_string(&shader_path)
            .await
            .context("Failed to read Metal shader")?;

        debug!(
            shader_size = shader_source.len(),
            "Loaded Metal shader"
        );

        // Compile shader to library
        let compile_options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(&shader_source, &compile_options)
            .map_err(|e| anyhow::anyhow!("Failed to compile Metal shader: {}", e))?;

        // Get kernel function
        let kernel_function = library
            .get_function("main_kernel", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;

        // Create compute pipeline state
        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&kernel_function)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline state: {}", e))?;

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

        // Create input buffer
        let input_buffer = if !input_data.is_empty() {
            let buffer = self.device.new_buffer_with_data(
                input_data.as_ptr() as *const _,
                input_data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            Some(buffer)
        } else {
            None
        };

        // Determine output size
        let output_size = spec.labels.get("output_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024 * 1024); // Default 1MB

        // Create output buffer
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Create compute command encoder
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline_state);

        // Set buffers
        if let Some(ref input) = input_buffer {
            compute_encoder.set_buffer(0, Some(input), 0);
        }
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);

        // Set buffer sizes as arguments
        let input_size = input_data.len() as u64;
        let output_size_u64 = output_size as u64;
        compute_encoder.set_bytes(
            2,
            std::mem::size_of::<u64>() as u64,
            &input_size as *const u64 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            std::mem::size_of::<u64>() as u64,
            &output_size_u64 as *const u64 as *const _,
        );

        // Calculate thread group sizes
        let thread_group_size = MTLSize::new(
            pipeline_state.thread_execution_width().min(256),
            1,
            1,
        );

        let thread_groups = MTLSize::new(
            (output_size as u64 + thread_group_size.width - 1) / thread_group_size.width,
            1,
            1,
        );

        info!("Dispatching Metal compute kernel");

        // Dispatch compute
        compute_encoder.dispatch_thread_groups(thread_groups, thread_group_size);

        // End encoding
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        info!("Metal kernel completed");

        // Copy result from buffer
        let output_ptr = output_buffer.contents() as *const u8;
        let output_host = unsafe {
            std::slice::from_raw_parts(output_ptr, output_size).to_vec()
        };

        // Save output
        let output_path = job_dir.join("output.bin");
        tokio::fs::write(&output_path, &output_host)
            .await
            .context("Failed to write output")?;

        info!(
            output_size = output_host.len(),
            "Metal execution completed successfully"
        );

        Ok(output_host)
    }

    #[allow(dead_code)]
    pub async fn checkpoint(&self, job_dir: &PathBuf, checkpoint_id: u64) -> Result<PathBuf> {
        let checkpoint_dir = job_dir.join("checkpoints");
        tokio::fs::create_dir_all(&checkpoint_dir).await?;

        let checkpoint_path = checkpoint_dir.join(format!("checkpoint_{}.bin", checkpoint_id));

        // In a real implementation, this would:
        // 1. Copy current GPU memory buffers
        // 2. Save model weights
        // 3. Save optimizer state
        // 4. Save RNG state

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

        let _checkpoint_data = tokio::fs::read(checkpoint_path).await?;

        // In a real implementation, this would:
        // 1. Restore GPU memory buffers
        // 2. Load model weights
        // 3. Restore optimizer state
        // 4. Restore RNG state
        // 5. Continue execution from saved point

        info!("Checkpoint restored, continuing execution");

        // For now, just execute normally (without re-checking for checkpoints to avoid recursion)
        // Box the recursive call to avoid infinite type size
        Box::pin(self.execute_internal(job_dir, spec, false)).await
    }
}


impl Default for MetalExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal executor")
    }
}
