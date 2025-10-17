use crate::model_hosting::{
    BackendType, GpuDevice, ModelHostingService as ModelHostingCore, Precision,
};
use crate::model_compatibility::{ModelCompatibilityChecker, DeviceCapabilities};
use corpgrid_proto::scheduler::{
    model_hosting_server::{ModelHosting, ModelHostingServer},
    ClusterResourceInfo, GetClusterStatusRequest, GetClusterStatusResponse,
    GetModelStatusRequest, GetModelStatusResponse, InferRequest, InferResponse,
    LoadModelRequest, LoadModelResponse, ModelAllocationInfo, ModelStatus, UnloadModelRequest,
    UnloadModelResponse,
};
use std::sync::Arc;
use std::path::Path;
use tonic::{Request, Response, Status};
use tracing::{info, error, warn};

/// gRPC service implementation for model hosting
pub struct ModelHostingGrpcService {
    hosting_service: Arc<ModelHostingCore>,
}

impl ModelHostingGrpcService {
    pub fn new(hosting_service: Arc<ModelHostingCore>) -> Self {
        Self { hosting_service }
    }

    pub fn into_grpc_server(self) -> ModelHostingServer<Self> {
        ModelHostingServer::new(self)
    }
}

#[tonic::async_trait]
impl ModelHosting for ModelHostingGrpcService {
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();

        info!(
            model_path = %req.model_path,
            precision = %req.precision,
            "Received LoadModel request"
        );

        // Parse precision
        let precision = match req.precision.as_str() {
            "fp32" => Precision::FP32,
            "fp16" => Precision::FP16,
            "bf16" => Precision::BF16,
            "int8" => Precision::INT8,
            "int4" => Precision::INT4,
            "" => Precision::FP16, // Default
            _ => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    model_id: String::new(),
                    error_message: format!("Invalid precision: {}", req.precision),
                    allocation: None,
                }));
            }
        };

        // TODO: Handle download_from_hub if requested
        // For now, assume model_path is local

        // Run compatibility check
        let compat_checker = ModelCompatibilityChecker::new();
        let model_path = Path::new(&req.model_path);

        // Infer model requirements
        let requirements = match compat_checker.infer_requirements(model_path) {
            Ok(reqs) => reqs,
            Err(e) => {
                warn!(error = %e, "Could not infer model requirements, proceeding without compatibility check");
                // Continue without compatibility check if we can't infer requirements
                match self
                    .hosting_service
                    .load_model(req.model_path.clone(), Some(precision))
                    .await
                {
                    Ok(model_id) => {
                        info!(model_id = %model_id, "Model loaded successfully (no compatibility check)");
                        return self.build_load_response(model_id).await;
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to load model");
                        return Ok(Response::new(LoadModelResponse {
                            success: false,
                            model_id: String::new(),
                            error_message: e.to_string(),
                            allocation: None,
                        }));
                    }
                }
            }
        };

        // Get available devices
        let pool = self.hosting_service.gpu_pool.read().await;
        let devices: Vec<DeviceCapabilities> = pool.devices.iter().map(|d| {
            DeviceCapabilities {
                backend: match d.backend {
                    corpgrid_common::GpuBackend::Cuda => "cuda".to_string(),
                    corpgrid_common::GpuBackend::Metal => "metal".to_string(),
                },
                vram_gb: d.vram_total_bytes / (1024 * 1024 * 1024),
                compute_capability: d.compute_capability,
                supports_fp16: true,  // Most modern GPUs support FP16
                supports_bf16: d.compute_capability.map_or(false, |(major, _)| major >= 8), // Ampere+
                supports_int8: true,
                supports_int4: true,
            }
        }).collect();
        drop(pool);

        // Check compatibility
        let compat_report = compat_checker.check_compatibility(&requirements, &devices);

        if !compat_report.compatible {
            error!(
                errors = ?compat_report.errors,
                "Model incompatible with available devices"
            );
            return Ok(Response::new(LoadModelResponse {
                success: false,
                model_id: String::new(),
                error_message: format!(
                    "Model incompatible with available devices:\n{}",
                    compat_report.errors.join("\n")
                ),
                allocation: None,
            }));
        }

        // Log warnings if any
        for warning in &compat_report.warnings {
            warn!(warning = %warning, "Model compatibility warning");
        }

        info!(
            recommended_precision = %compat_report.recommended_precision,
            estimated_tps = %compat_report.estimated_throughput_tps,
            "Model compatibility check passed"
        );

        match self
            .hosting_service
            .load_model(req.model_path.clone(), Some(precision))
            .await
        {
            Ok(model_id) => {
                info!(model_id = %model_id, "Model loaded successfully");
                self.build_load_response(model_id).await
            }
            Err(e) => {
                error!(error = %e, "Failed to load model");
                Ok(Response::new(LoadModelResponse {
                    success: false,
                    model_id: String::new(),
                    error_message: e.to_string(),
                    allocation: None,
                }))
            }
        }
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        let req = request.into_inner();

        info!(model_id = %req.model_id, "Received UnloadModel request");

        match self.hosting_service.unload_model(&req.model_id).await {
            Ok(_) => {
                info!(model_id = %req.model_id, "Model unloaded successfully");
                Ok(Response::new(UnloadModelResponse {
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!(model_id = %req.model_id, error = %e, "Failed to unload model");
                Ok(Response::new(UnloadModelResponse {
                    success: false,
                    error_message: e.to_string(),
                }))
            }
        }
    }

    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let req = request.into_inner();
        let start_time = std::time::Instant::now();

        info!(
            model_id = %req.model_id,
            input_len = req.input_tokens.len(),
            max_new_tokens = req.max_new_tokens,
            "Received Infer request"
        );

        match self
            .hosting_service
            .submit_inference(
                req.model_id.clone(),
                req.input_tokens,
                req.max_new_tokens as usize,
                req.temperature,
                req.top_p,
            )
            .await
        {
            Ok(output_tokens) => {
                let generation_time_ms = start_time.elapsed().as_millis() as u64;
                info!(
                    model_id = %req.model_id,
                    output_len = output_tokens.len(),
                    generation_time_ms,
                    "Inference completed"
                );

                Ok(Response::new(InferResponse {
                    success: true,
                    output_tokens,
                    error_message: String::new(),
                    generation_time_ms,
                }))
            }
            Err(e) => {
                error!(model_id = %req.model_id, error = %e, "Inference failed");
                Ok(Response::new(InferResponse {
                    success: false,
                    output_tokens: vec![],
                    error_message: e.to_string(),
                    generation_time_ms: 0,
                }))
            }
        }
    }

    async fn get_model_status(
        &self,
        _request: Request<GetModelStatusRequest>,
    ) -> Result<Response<GetModelStatusResponse>, Status> {
        let statuses = self.hosting_service.get_model_status().await;

        let models: Vec<ModelStatus> = statuses
            .into_iter()
            .map(|s| ModelStatus {
                model_id: s.model_id,
                model_name: s.model_name,
                num_parameters_b: s.num_parameters_b,
                devices: s.devices,
                backend_type: match s.backend_type {
                    BackendType::HomogeneousCuda => "homogeneous_cuda",
                    BackendType::HomogeneousMetal => "homogeneous_metal",
                    BackendType::HeterogeneousPipeline => "heterogeneous_pipeline",
                }
                .to_string(),
                requests_served: s.requests_served,
            })
            .collect();

        // Get cluster resource info
        let cluster_resources = self.get_cluster_resource_info().await;

        Ok(Response::new(GetModelStatusResponse {
            models,
            cluster_resources: Some(cluster_resources),
        }))
    }

    async fn get_cluster_status(
        &self,
        _request: Request<GetClusterStatusRequest>,
    ) -> Result<Response<GetClusterStatusResponse>, Status> {
        let resources = self.get_cluster_resource_info().await;

        Ok(Response::new(GetClusterStatusResponse {
            resources: Some(resources),
        }))
    }
}

impl ModelHostingGrpcService {
    async fn build_load_response(&self, model_id: String) -> Result<Response<LoadModelResponse>, Status> {
        // Get allocation info
        let statuses = self.hosting_service.get_model_status().await;
        let model_status = statuses.iter().find(|s| s.model_id == model_id);

        let allocation = model_status.map(|s| {
            // Calculate total VRAM from device names (parsed from status)
            // Device names are formatted as "DeviceName (Backend)"
            let device_backends: Vec<String> = s.devices.iter()
                .filter_map(|d| {
                    if d.contains("CUDA") {
                        Some("cuda".to_string())
                    } else if d.contains("Metal") {
                        Some("metal".to_string())
                    } else {
                        None
                    }
                })
                .collect();

            // Get total VRAM from hosting service
            let total_vram_gb = 0u64; // Will be populated when we track allocated VRAM

            let backend_type_str = match s.backend_type {
                BackendType::HomogeneousCuda => "homogeneous_cuda",
                BackendType::HomogeneousMetal => "homogeneous_metal",
                BackendType::HeterogeneousPipeline => "heterogeneous_pipeline",
            };

            ModelAllocationInfo {
                num_devices: s.devices.len() as u32,
                device_names: s.devices.clone(),
                device_backends,
                total_vram_gb,
                backend_type: backend_type_str.to_string(),
            }
        });

        Ok(Response::new(LoadModelResponse {
            success: true,
            model_id,
            error_message: String::new(),
            allocation,
        }))
    }

    async fn get_cluster_resource_info(&self) -> ClusterResourceInfo {
        // Get real cluster resource info from GPU pool
        // Access the hosting service's GPU pool
        let pool_arc = &self.hosting_service.gpu_pool;
        let pool = pool_arc.read().await;

        let total_gpus = pool.devices.len() as u32;
        let free_gpus = pool.devices.iter().filter(|d| !d.is_allocated).count() as u32;

        let total_vram_gb = pool.devices.iter()
            .map(|d| d.vram_total_bytes / (1024 * 1024 * 1024))
            .sum::<u64>();

        let free_vram_gb = pool.devices.iter()
            .filter(|d| !d.is_allocated)
            .map(|d| d.vram_free_bytes / (1024 * 1024 * 1024))
            .sum::<u64>();

        let cuda_gpus = pool.devices.iter()
            .filter(|d| d.backend == corpgrid_common::GpuBackend::Cuda)
            .count() as u32;

        let metal_gpus = pool.devices.iter()
            .filter(|d| d.backend == corpgrid_common::GpuBackend::Metal)
            .count() as u32;

        ClusterResourceInfo {
            total_gpus,
            free_gpus,
            total_vram_gb,
            free_vram_gb,
            cuda_gpus,
            metal_gpus,
        }
    }
}

/// Helper function to register agent GPUs with the model hosting service
pub async fn register_agent_gpus_with_hosting(
    hosting_service: &Arc<ModelHostingCore>,
    agent_id: String,
    gpus: Vec<corpgrid_common::GpuInfo>,
) {
    let gpu_devices: Vec<GpuDevice> = gpus
        .into_iter()
        .enumerate()
        .map(|(idx, gpu)| GpuDevice {
            agent_id: agent_id.clone(),
            device_index: idx,
            backend: gpu.backend,
            vram_total_bytes: gpu.vram_bytes,
            vram_free_bytes: gpu.vram_bytes, // Assume all free initially
            compute_capability: gpu.compute_capability.and_then(|cc| {
                let parts: Vec<&str> = cc.split('.').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse().unwrap_or(0),
                        parts[1].parse().unwrap_or(0),
                    ))
                } else {
                    None
                }
            }),
            device_name: gpu.name,
            is_allocated: false,
        })
        .collect();

    hosting_service
        .register_agent_gpus(agent_id, gpu_devices)
        .await;
}
