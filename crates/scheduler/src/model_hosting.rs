use anyhow::{Context, Result};
use corpgrid_common::GpuBackend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokenizers::Tokenizer;
use tracing::info;
use uuid::Uuid;

/// Server-side model hosting service that manages LLM models across heterogeneous GPU pools
/// Supports spanning models across CUDA and Metal devices transparently
pub struct ModelHostingService {
    /// Pool of all available GPUs across all registered agents
    pub(crate) gpu_pool: Arc<RwLock<GpuPool>>,
    /// Currently loaded models and their resource allocations
    loaded_models: Arc<RwLock<HashMap<String, LoadedModel>>>,
    /// Model metadata cache
    model_metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,
}

/// Represents a GPU device from an agent
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub agent_id: String,
    pub device_index: usize,
    pub backend: GpuBackend,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub compute_capability: Option<(u32, u32)>, // For CUDA
    pub device_name: String,
    pub is_allocated: bool,
}

/// Pool of all GPUs across the cluster
pub struct GpuPool {
    pub(crate) devices: Vec<GpuDevice>,
    /// Map agent_id -> list of device indices
    agent_devices: HashMap<String, Vec<usize>>,
}

impl Default for GpuPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPool {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            agent_devices: HashMap::new(),
        }
    }

    /// Register GPUs from an agent
    pub fn register_agent_gpus(&mut self, agent_id: String, gpus: Vec<GpuDevice>) {
        let mut device_indices = Vec::new();

        for gpu in gpus {
            device_indices.push(self.devices.len());
            self.devices.push(gpu);
        }

        let num_gpus = device_indices.len();
        self.agent_devices.insert(agent_id.clone(), device_indices);
        info!(agent_id = %agent_id, num_gpus = num_gpus, "Registered agent GPUs");
    }

    /// Remove agent's GPUs when agent disconnects
    pub fn unregister_agent(&mut self, agent_id: &str) {
        if let Some(indices) = self.agent_devices.remove(agent_id) {
            for idx in indices {
                if idx < self.devices.len() {
                    self.devices[idx].is_allocated = false;
                }
            }
        }
    }

    /// Find optimal GPU allocation for a model
    /// Can span across CUDA and Metal using pipeline parallelism
    pub fn find_allocation(&self, required_vram: u64, preferred_device_count: usize) -> Option<Vec<usize>> {
        // Strategy: Try homogeneous allocation first, then heterogeneous

        // Try single GPU first
        if let Some(device) = self.find_single_device(required_vram) {
            return Some(vec![device]);
        }

        // Try homogeneous multi-GPU (all CUDA or all Metal)
        if let Some(allocation) = self.find_homogeneous_allocation(required_vram, preferred_device_count) {
            return Some(allocation);
        }

        // Try heterogeneous allocation (CUDA + Metal)
        self.find_heterogeneous_allocation(required_vram, preferred_device_count)
    }

    fn find_single_device(&self, required_vram: u64) -> Option<usize> {
        for (idx, device) in self.devices.iter().enumerate() {
            if !device.is_allocated && device.vram_free_bytes >= required_vram {
                return Some(idx);
            }
        }
        None
    }

    fn find_homogeneous_allocation(&self, required_vram: u64, device_count: usize) -> Option<Vec<usize>> {
        // Try to find N devices of the same backend
        let vram_per_device = required_vram / device_count as u64;

        // Try CUDA-only allocation
        let cuda_devices: Vec<usize> = self.devices.iter()
            .enumerate()
            .filter(|(_, d)| !d.is_allocated && d.backend == GpuBackend::Cuda && d.vram_free_bytes >= vram_per_device)
            .map(|(idx, _)| idx)
            .take(device_count)
            .collect();

        if cuda_devices.len() == device_count {
            return Some(cuda_devices);
        }

        // Try Metal-only allocation
        let metal_devices: Vec<usize> = self.devices.iter()
            .enumerate()
            .filter(|(_, d)| !d.is_allocated && d.backend == GpuBackend::Metal && d.vram_free_bytes >= vram_per_device)
            .map(|(idx, _)| idx)
            .take(device_count)
            .collect();

        if metal_devices.len() == device_count {
            return Some(metal_devices);
        }

        None
    }

    fn find_heterogeneous_allocation(&self, required_vram: u64, device_count: usize) -> Option<Vec<usize>> {
        // Pipeline parallelism: allocate layers across mixed backends
        // Sort devices by available VRAM descending
        let mut available: Vec<(usize, u64)> = self.devices.iter()
            .enumerate()
            .filter(|(_, d)| !d.is_allocated)
            .map(|(idx, d)| (idx, d.vram_free_bytes))
            .collect();

        available.sort_by_key(|(_, vram)| std::cmp::Reverse(*vram));

        let mut allocation = Vec::new();
        let mut total_vram = 0u64;

        for (idx, vram) in available.iter().take(device_count) {
            allocation.push(*idx);
            total_vram += vram;

            if total_vram >= required_vram {
                return Some(allocation);
            }
        }

        None
    }

    /// Mark devices as allocated
    pub fn allocate_devices(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.devices.len() {
                self.devices[idx].is_allocated = true;
            }
        }
    }

    /// Mark devices as free
    pub fn free_devices(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.devices.len() {
                self.devices[idx].is_allocated = false;
            }
        }
    }

    /// Get total available VRAM across all free devices
    pub fn total_available_vram(&self) -> u64 {
        self.devices.iter()
            .filter(|d| !d.is_allocated)
            .map(|d| d.vram_free_bytes)
            .sum()
    }
}

/// A loaded model instance
pub struct LoadedModel {
    pub model_id: String,
    pub model_path: String,
    pub metadata: ModelMetadata,
    pub device_allocation: Vec<usize>, // Indices into GpuPool
    pub precision: Precision,
    pub inference_backend: Box<dyn InferenceBackend + Send + Sync>,
    pub tokenizer: Arc<Tokenizer>,
    pub requests_served: u64,
}

/// Model metadata extracted from config.json
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub context_length: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

impl Precision {
    pub fn bytes_per_param(&self) -> u64 {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 | Precision::BF16 => 2,
            Precision::INT8 => 1,
            Precision::INT4 => 0, // Special handling
        }
    }
}

/// Inference request
#[derive(Debug)]
pub struct InferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub input_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub response_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>>>,
}

/// Trait for inference backends (homogeneous or heterogeneous)
#[async_trait::async_trait]
pub trait InferenceBackend {
    async fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>>;

    fn backend_type(&self) -> BackendType;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackendType {
    HomogeneousCuda,
    HomogeneousMetal,
    HeterogeneousPipeline, // Mixed CUDA + Metal
}

impl Default for ModelHostingService {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelHostingService {
    pub fn new() -> Self {
        Self {
            gpu_pool: Arc::new(RwLock::new(GpuPool::new())),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            model_metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an agent's GPUs with the pool
    pub async fn register_agent_gpus(&self, agent_id: String, gpus: Vec<GpuDevice>) {
        let mut pool = self.gpu_pool.write().await;
        pool.register_agent_gpus(agent_id, gpus);
    }

    /// Unregister an agent's GPUs
    pub async fn unregister_agent(&self, agent_id: &str) {
        let mut pool = self.gpu_pool.write().await;
        pool.unregister_agent(agent_id);
    }

    /// Load a model with automatic resource allocation
    /// Returns model_id on success, error if insufficient resources
    pub async fn load_model(&self, model_path: String, precision: Option<Precision>) -> Result<String> {
        info!(model_path = %model_path, "Loading model with automatic resource allocation");

        // 1. Load model metadata to determine requirements
        let metadata = self.load_model_metadata(&model_path).await?;

        let precision = precision.unwrap_or(Precision::FP16);
        let required_vram = self.calculate_vram_requirement(&metadata, 1, precision);

        info!(
            model = %metadata.name,
            required_vram_gb = required_vram / (1024 * 1024 * 1024),
            parameters_b = metadata.num_parameters / 1_000_000_000,
            "Model metadata loaded"
        );

        // 2. Find optimal GPU allocation
        let pool = self.gpu_pool.read().await;
        let total_available = pool.total_available_vram();

        if total_available < required_vram {
            anyhow::bail!(
                "Insufficient VRAM across cluster: need {}GB, have {}GB available",
                required_vram / (1024 * 1024 * 1024),
                total_available / (1024 * 1024 * 1024)
            );
        }

        // Estimate optimal device count based on model size
        let preferred_device_count = self.estimate_device_count(&metadata, precision);

        let allocation = pool.find_allocation(required_vram, preferred_device_count)
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU allocation found"))?;

        drop(pool);

        // 3. Allocate devices
        let mut pool = self.gpu_pool.write().await;
        pool.allocate_devices(&allocation);
        let allocated_devices: Vec<GpuDevice> = allocation.iter()
            .map(|&idx| pool.devices[idx].clone())
            .collect();
        drop(pool);

        info!(
            num_devices = allocation.len(),
            devices = ?allocation,
            "Allocated devices for model"
        );

        // 4. Load tokenizer from model directory
        let tokenizer_path = std::path::PathBuf::from(&model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;

        info!("Tokenizer loaded successfully");

        let tokenizer_arc = Arc::new(tokenizer);

        // 5. Create inference backend based on allocation
        let backend = self.create_inference_backend(
            &allocated_devices,
            &model_path,
            &metadata,
            precision,
            tokenizer_arc.clone(),
        ).await?;

        // 6. Store loaded model
        let model_id = Uuid::new_v4().to_string();
        let loaded_model = LoadedModel {
            model_id: model_id.clone(),
            model_path,
            metadata,
            device_allocation: allocation,
            precision,
            inference_backend: backend,
            tokenizer: tokenizer_arc,
            requests_served: 0,
        };

        let mut models = self.loaded_models.write().await;
        models.insert(model_id.clone(), loaded_model);

        info!(model_id = %model_id, "Model loaded successfully");
        Ok(model_id)
    }

    /// Unload a model and free its resources
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.loaded_models.write().await;

        if let Some(model) = models.remove(model_id) {
            // Free GPU allocation
            let mut pool = self.gpu_pool.write().await;
            pool.free_devices(&model.device_allocation);

            info!(model_id = %model_id, "Model unloaded and resources freed");
            Ok(())
        } else {
            anyhow::bail!("Model not found: {}", model_id)
        }
    }

    /// Submit an inference request
    pub async fn submit_inference(
        &self,
        model_id: String,
        input_tokens: Vec<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let request = InferenceRequest {
            request_id: Uuid::new_v4().to_string(),
            model_id: model_id.clone(),
            input_tokens,
            max_new_tokens,
            temperature,
            top_p,
            response_tx: tx,
        };

        // Check if model is loaded
        {
            let models = self.loaded_models.read().await;
            if !models.contains_key(&model_id) {
                anyhow::bail!("Model not loaded: {}", model_id);
            }
        }

        // Process request immediately (could queue in production)
        self.process_request(request).await;

        // Wait for response
        rx.await?
    }

    async fn process_request(&self, request: InferenceRequest) {
        let model_id = request.model_id.clone();

        // Get model
        let models = self.loaded_models.read().await;
        let model = match models.get(&model_id) {
            Some(m) => m,
            None => {
                let _ = request.response_tx.send(Err(anyhow::anyhow!("Model not found")));
                return;
            }
        };

        // Run inference
        let result = model.inference_backend.generate(
            &request.input_tokens,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
        ).await;

        // Send response
        let _ = request.response_tx.send(result);
    }

    async fn load_model_metadata(&self, model_path: &str) -> Result<ModelMetadata> {
        // Check cache first
        {
            let cache = self.model_metadata_cache.read().await;
            if let Some(metadata) = cache.get(model_path) {
                return Ok(metadata.clone());
            }
        }

        // Load config.json
        let config_path = std::path::PathBuf::from(model_path).join("config.json");
        let config_str = tokio::fs::read_to_string(&config_path).await
            .context("Failed to read config.json")?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        let hidden_size = config["hidden_size"]
            .as_u64()
            .context("Missing hidden_size")? as usize;

        let num_layers = config["num_hidden_layers"]
            .as_u64()
            .context("Missing num_hidden_layers")? as usize;

        let num_heads = config["num_attention_heads"]
            .as_u64()
            .context("Missing num_attention_heads")? as usize;

        let vocab_size = config["vocab_size"]
            .as_u64()
            .context("Missing vocab_size")? as usize;

        let context_length = config
            .get("max_position_embeddings")
            .or_else(|| config.get("n_positions"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let num_parameters = Self::estimate_parameters(hidden_size, num_layers, vocab_size);

        let name = config["model_type"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        let metadata = ModelMetadata {
            name,
            num_parameters,
            num_layers,
            hidden_size,
            num_heads,
            vocab_size,
            context_length,
        };

        // Cache it
        let mut cache = self.model_metadata_cache.write().await;
        cache.insert(model_path.to_string(), metadata.clone());

        Ok(metadata)
    }

    fn estimate_parameters(hidden_size: usize, num_layers: usize, vocab_size: usize) -> u64 {
        let embedding_params = vocab_size * hidden_size;
        let attention_params = num_layers * (4 * hidden_size * hidden_size);
        let ffn_params = num_layers * (8 * hidden_size * hidden_size);
        let layer_norm_params = num_layers * (2 * hidden_size);

        (embedding_params + attention_params + ffn_params + layer_norm_params) as u64
    }

    fn calculate_vram_requirement(
        &self,
        metadata: &ModelMetadata,
        batch_size: usize,
        precision: Precision,
    ) -> u64 {
        let bytes_per_param = precision.bytes_per_param();
        let model_weights = metadata.num_parameters * bytes_per_param;

        // KV cache
        let kv_cache = (batch_size
            * metadata.num_layers
            * 2
            * metadata.hidden_size
            * metadata.context_length
            * bytes_per_param as usize) as u64;

        // Activation memory
        let activations = (batch_size * metadata.hidden_size * 4) as u64 * bytes_per_param;

        model_weights + kv_cache + activations
    }

    fn estimate_device_count(&self, metadata: &ModelMetadata, precision: Precision) -> usize {
        let bytes_per_param = precision.bytes_per_param();
        let model_size_gb = (metadata.num_parameters * bytes_per_param) / (1024 * 1024 * 1024);

        // Heuristic: Use 1 GPU per ~70GB of model
        let devices = model_size_gb.div_ceil(70) as usize;
        devices.max(1)
    }

    async fn create_inference_backend(
        &self,
        devices: &[GpuDevice],
        model_path: &str,
        metadata: &ModelMetadata,
        precision: Precision,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Box<dyn InferenceBackend + Send + Sync>> {
        use crate::mistralrs_backend::MistralRsInferenceBackend;

        info!("Creating inference backend using mistral.rs");

        let backend = MistralRsInferenceBackend::load(
            devices,
            model_path,
            metadata,
            precision,
            tokenizer,
        )
        .await?;

        Ok(Box::new(backend))
    }

    /// Get tokenizer for a loaded model
    pub async fn get_tokenizer(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        let models = self.loaded_models.read().await;
        models.get(model_id).map(|m| m.tokenizer.clone())
    }

    /// Get status of all loaded models
    pub async fn get_model_status(&self) -> Vec<ModelStatus> {
        let models = self.loaded_models.read().await;
        let pool = self.gpu_pool.read().await;

        models.values().map(|model| {
            let devices: Vec<String> = model.device_allocation.iter()
                .filter_map(|&idx| pool.devices.get(idx))
                .map(|d| format!("{} ({})", d.device_name, match d.backend {
                    GpuBackend::Cuda => "CUDA",
                    GpuBackend::Metal => "Metal",
                }))
                .collect();

            ModelStatus {
                model_id: model.model_id.clone(),
                model_name: model.metadata.name.clone(),
                num_parameters_b: model.metadata.num_parameters / 1_000_000_000,
                devices,
                backend_type: model.inference_backend.backend_type(),
                requests_served: model.requests_served,
            }
        }).collect()
    }
}

#[derive(Debug)]
pub struct ModelStatus {
    pub model_id: String,
    pub model_name: String,
    pub num_parameters_b: u64,
    pub devices: Vec<String>,
    pub backend_type: BackendType,
    pub requests_served: u64,
}

// All inference backends are now implemented in inference_backend.rs using Candle
