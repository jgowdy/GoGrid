use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::model_hosting::{BackendType, GpuDevice, InferenceBackend, ModelMetadata, Precision};
use crate::heterogeneous_pipeline::{HeterogeneousPipeline, HeterogeneousPipelineExecutor};

/// Backend implementation enum
enum BackendImpl {
    MistralRs(Arc<mistralrs::Model>),
    HeterogeneousPipeline(Arc<HeterogeneousPipelineExecutor>),
}

/// Inference backend using mistral.rs or custom heterogeneous pipeline
pub struct MistralRsInferenceBackend {
    backend: BackendImpl,
    #[allow(dead_code)]
    tokenizer: Arc<Tokenizer>,
    backend_type: BackendType,
}

impl MistralRsInferenceBackend {
    pub async fn load(
        devices: &[GpuDevice],
        model_path: &str,
        _metadata: &ModelMetadata,
        _precision: Precision,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self> {
        info!(
            model_path = %model_path,
            num_devices = devices.len(),
            "Loading model with mistral.rs"
        );

        // Determine backend type from devices
        let backend_type = if devices.is_empty() {
            BackendType::HomogeneousCuda // Default
        } else if devices.iter().all(|d| matches!(d.backend, corpgrid_common::GpuBackend::Cuda)) {
            BackendType::HomogeneousCuda
        } else if devices.iter().all(|d| matches!(d.backend, corpgrid_common::GpuBackend::Metal)) {
            BackendType::HomogeneousMetal
        } else {
            BackendType::HeterogeneousPipeline
        };

        info!(backend_type = ?backend_type, "Determined backend type");

        // Create model builder - mistral.rs handles everything automatically
        use mistralrs::{TextModelBuilder, Device};

        // Build device configuration based on backend type
        let model = if devices.is_empty() {
            info!("No devices allocated, using CPU");
            TextModelBuilder::new(model_path)
                .with_device(Device::Cpu)
                .build()
                .await?
        } else {
            match backend_type {
                BackendType::HomogeneousCuda => {
                    // All CUDA devices - collect all device indices
                    let cuda_indices: Vec<usize> = devices.iter()
                        .map(|d| d.device_index)
                        .collect();

                    info!(
                        cuda_devices = ?cuda_indices,
                        num_devices = cuda_indices.len(),
                        "Configuring homogeneous CUDA setup with automatic tensor parallelism"
                    );

                    if cuda_indices.len() > 1 {
                        // Multi-GPU CUDA: mistral.rs automatically uses NCCL tensor parallelism
                        // when multiple CUDA devices are detected
                        info!(
                            "Multi-GPU tensor parallelism will be automatically enabled via NCCL for {} CUDA devices",
                            cuda_indices.len()
                        );

                        // Set CUDA_VISIBLE_DEVICES to expose all requested GPUs
                        let cuda_visible = cuda_indices.iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        std::env::set_var("CUDA_VISIBLE_DEVICES", &cuda_visible);

                        info!(cuda_visible_devices = %cuda_visible, "Set CUDA_VISIBLE_DEVICES for tensor parallelism");
                    }

                    // Use first device as primary; mistral.rs will automatically detect
                    // and use all CUDA devices via NCCL
                    let device = Device::cuda_if_available(0)?;

                    TextModelBuilder::new(model_path)
                        .with_device(device)
                        .build()
                        .await?
                },
                BackendType::HomogeneousMetal => {
                    // All Metal devices - collect all device indices
                    let metal_indices: Vec<usize> = devices.iter()
                        .map(|d| d.device_index)
                        .collect();

                    info!(
                        metal_devices = ?metal_indices,
                        num_devices = metal_indices.len(),
                        "Configuring homogeneous Metal setup with automatic tensor parallelism"
                    );

                    if metal_indices.len() > 1 {
                        // Multi-GPU Metal: mistral.rs automatically uses Ring backend
                        // tensor parallelism when multiple Metal devices are detected
                        info!(
                            "Multi-GPU tensor parallelism will be automatically enabled via Ring backend for {} Metal devices",
                            metal_indices.len()
                        );

                        // For Metal, mistral.rs will automatically detect all available GPUs
                        // No need to set environment variables like CUDA_VISIBLE_DEVICES
                    }

                    // Use first device as primary; mistral.rs will automatically detect
                    // and use all Metal devices via Ring backend
                    let device = Device::new_metal(metal_indices[0])?;

                    TextModelBuilder::new(model_path)
                        .with_device(device)
                        .build()
                        .await?
                },
                BackendType::HeterogeneousPipeline => {
                    // Mixed CUDA and Metal devices - use custom pipeline parallelism
                    info!(
                        num_devices = devices.len(),
                        "Configuring TRUE heterogeneous pipeline with mixed CUDA/Metal devices"
                    );

                    // Log device configuration
                    for (idx, device) in devices.iter().enumerate() {
                        info!(
                            device_idx = idx,
                            backend = ?device.backend,
                            device_index = device.device_index,
                            "Pipeline device"
                        );
                    }

                    // Create heterogeneous pipeline
                    info!("Creating custom heterogeneous pipeline executor");
                    let pipeline = Arc::new(HeterogeneousPipeline::new(devices, model_path, None)?);
                    let executor = Arc::new(HeterogeneousPipelineExecutor::new(pipeline, model_path).await?);

                    info!("Heterogeneous pipeline executor created successfully");

                    // Return early with heterogeneous backend
                    return Ok(Self {
                        backend: BackendImpl::HeterogeneousPipeline(executor),
                        tokenizer,
                        backend_type,
                    });
                },
            }
        };

        info!("Model loaded successfully with mistral.rs");

        Ok(Self {
            backend: BackendImpl::MistralRs(Arc::new(model)),
            tokenizer,
            backend_type,
        })
    }
}

#[async_trait::async_trait]
impl InferenceBackend for MistralRsInferenceBackend {
    async fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        debug!(
            input_len = input_ids.len(),
            max_new_tokens,
            backend_type = ?self.backend_type,
            "Starting generation"
        );

        match &self.backend {
            BackendImpl::MistralRs(model) => {
                // Use mistral.rs text-based API
                debug!("Using mistral.rs backend");

                // Decode tokens to text
                let input_text = self.tokenizer.decode(input_ids, false)
                    .map_err(|e| anyhow::anyhow!("Failed to decode input tokens: {}", e))?;

                debug!(input_text = %input_text, "Decoded input");

                // Create request with sampling parameters
                use mistralrs::{RequestBuilder, TextMessageRole};

                let request = RequestBuilder::new()
                    .set_sampler_temperature(temperature as f64)
                    .set_sampler_topp(top_p as f64)
                    .set_sampler_max_len(max_new_tokens)
                    .add_message(TextMessageRole::User, &input_text);

                // Send request and get response
                let response = model.send_chat_request(request).await?;

                // Extract generated text from response
                let output_text = response.choices[0]
                    .message
                    .content
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

                debug!(output_text = %output_text, "Generated text");

                // Encode the full output (input + generated) back to tokens
                let full_text = format!("{}{}", input_text, output_text);
                let output_encoding = self.tokenizer.encode(full_text, false)
                    .map_err(|e| anyhow::anyhow!("Failed to encode output text: {}", e))?;
                let output_tokens: Vec<u32> = output_encoding.get_ids().to_vec();

                info!(
                    input_tokens = input_ids.len(),
                    output_tokens = output_tokens.len(),
                    generated = output_tokens.len() - input_ids.len(),
                    "Generation complete with mistral.rs"
                );

                Ok(output_tokens)
            },
            BackendImpl::HeterogeneousPipeline(executor) => {
                // Use custom heterogeneous pipeline
                debug!("Using heterogeneous pipeline backend");

                info!("Running inference through heterogeneous pipeline");

                // Call the executor's infer method directly with token IDs
                let output_tokens = executor.infer(input_ids, max_new_tokens, temperature, top_p).await?;

                info!(
                    input_tokens = input_ids.len(),
                    output_tokens = output_tokens.len(),
                    "Generation complete with heterogeneous pipeline"
                );

                Ok(output_tokens)
            },
        }
    }

    fn backend_type(&self) -> BackendType {
        self.backend_type
    }
}
