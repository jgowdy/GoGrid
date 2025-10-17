use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    VisionLanguage,
    ImageGeneration,
    AudioGeneration,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub model_type: ModelType,
    pub min_vram_gb: u64,
    pub recommended_vram_gb: u64,
    pub supported_backends: Vec<String>, // ["cuda", "metal", "cpu"]
    pub min_compute_capability: Option<(u32, u32)>, // For CUDA
    pub requires_fp16: bool,
    pub requires_bf16: bool,
    pub max_batch_size: Option<u32>,
    pub context_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub backend: String,
    pub vram_gb: u64,
    pub compute_capability: Option<(u32, u32)>,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub supports_int4: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub compatible: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommended_precision: String,
    pub estimated_throughput_tps: f32,
}

pub struct ModelCompatibilityChecker;

impl ModelCompatibilityChecker {
    pub fn new() -> Self {
        Self
    }

    /// Check if a model is compatible with available devices
    pub fn check_compatibility(
        &self,
        requirements: &ModelRequirements,
        devices: &[DeviceCapabilities],
    ) -> CompatibilityReport {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut compatible = true;

        // Check if any device supports the required backends
        let has_compatible_backend = devices.iter().any(|d| {
            requirements.supported_backends.contains(&d.backend)
        });

        if !has_compatible_backend {
            errors.push(format!(
                "No devices with compatible backend. Required: {:?}, Available: {:?}",
                requirements.supported_backends,
                devices.iter().map(|d| &d.backend).collect::<Vec<_>>()
            ));
            compatible = false;
        }

        // Check VRAM requirements
        let max_vram = devices.iter().map(|d| d.vram_gb).max().unwrap_or(0);

        if max_vram < requirements.min_vram_gb {
            errors.push(format!(
                "Insufficient VRAM. Required: {} GB, Available: {} GB",
                requirements.min_vram_gb,
                max_vram
            ));
            compatible = false;
        } else if max_vram < requirements.recommended_vram_gb {
            warnings.push(format!(
                "VRAM below recommended. Recommended: {} GB, Available: {} GB. Performance may be degraded.",
                requirements.recommended_vram_gb,
                max_vram
            ));
        }

        // Check compute capability for CUDA devices
        if let Some((req_major, req_minor)) = requirements.min_compute_capability {
            let cuda_devices: Vec<_> = devices.iter()
                .filter(|d| d.backend == "cuda")
                .collect();

            if !cuda_devices.is_empty() {
                let compatible_cuda = cuda_devices.iter().any(|d| {
                    if let Some((major, minor)) = d.compute_capability {
                        major > req_major || (major == req_major && minor >= req_minor)
                    } else {
                        false
                    }
                });

                if !compatible_cuda {
                    errors.push(format!(
                        "CUDA compute capability too low. Required: {}.{}, Available: {:?}",
                        req_major,
                        req_minor,
                        cuda_devices.iter()
                            .filter_map(|d| d.compute_capability)
                            .collect::<Vec<_>>()
                    ));
                    compatible = false;
                }
            }
        }

        // Check precision support
        if requirements.requires_fp16 {
            let has_fp16 = devices.iter().any(|d| d.supports_fp16);
            if !has_fp16 {
                errors.push("Model requires FP16 support, but no compatible devices found".to_string());
                compatible = false;
            }
        }

        if requirements.requires_bf16 {
            let has_bf16 = devices.iter().any(|d| d.supports_bf16);
            if !has_bf16 {
                warnings.push("Model recommends BF16 support. Will fall back to FP16/FP32.".to_string());
            }
        }

        // Determine recommended precision
        let recommended_precision = if devices.iter().any(|d| d.supports_int4) && max_vram < requirements.recommended_vram_gb {
            "int4".to_string()
        } else if devices.iter().any(|d| d.supports_int8) && max_vram < requirements.recommended_vram_gb {
            "int8".to_string()
        } else if devices.iter().any(|d| d.supports_bf16) {
            "bf16".to_string()
        } else if devices.iter().any(|d| d.supports_fp16) {
            "fp16".to_string()
        } else {
            "fp32".to_string()
        };

        // Estimate throughput based on model type and hardware
        let estimated_throughput_tps = self.estimate_throughput(requirements, devices, &recommended_precision);

        CompatibilityReport {
            compatible,
            warnings,
            errors,
            recommended_precision,
            estimated_throughput_tps,
        }
    }

    /// Infer model requirements from model files
    pub fn infer_requirements(&self, model_path: &Path) -> Result<ModelRequirements> {
        // Try to load config.json
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            bail!("config.json not found in model directory");
        }

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        // Extract model architecture
        let model_type = self.detect_model_type(&config)?;

        // Extract key parameters
        let hidden_size = config.get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096);

        let num_layers = config.get("num_hidden_layers")
            .or_else(|| config.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32);

        let vocab_size = config.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000);

        let context_length = config.get("max_position_embeddings")
            .or_else(|| config.get("max_sequence_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as u32;

        // Estimate VRAM requirements
        // Formula: (num_params * bytes_per_param) + (context_length * hidden_size * batch_size * 2)
        let num_params = self.estimate_parameter_count(hidden_size, num_layers, vocab_size);
        let params_gb = (num_params * 2) / (1024 * 1024 * 1024); // FP16 = 2 bytes
        let kv_cache_gb = ((context_length as u64 * hidden_size * 2 * 2) / (1024 * 1024 * 1024)).max(1);

        let min_vram_gb = params_gb + kv_cache_gb;
        let recommended_vram_gb = min_vram_gb + (min_vram_gb / 2); // 50% overhead for safety

        // Determine supported backends
        let supported_backends = vec!["cuda".to_string(), "metal".to_string()];

        // Check if model requires specific precision
        let torch_dtype = config.get("torch_dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float16");

        let requires_fp16 = torch_dtype == "float16";
        let requires_bf16 = torch_dtype == "bfloat16";

        Ok(ModelRequirements {
            model_type,
            min_vram_gb,
            recommended_vram_gb,
            supported_backends,
            min_compute_capability: Some((7, 0)), // Volta or newer
            requires_fp16,
            requires_bf16,
            max_batch_size: Some(8),
            context_length,
        })
    }

    fn detect_model_type(&self, config: &serde_json::Value) -> Result<ModelType> {
        let model_type_str = config.get("model_type")
            .or_else(|| config.get("architectures").and_then(|v| v.get(0)))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let model_type = if model_type_str.contains("vlm") || model_type_str.contains("Vision") || model_type_str.contains("llava") {
            ModelType::VisionLanguage
        } else if model_type_str.contains("stable-diffusion") || model_type_str.contains("flux") {
            ModelType::ImageGeneration
        } else if model_type_str.contains("whisper") || model_type_str.contains("audio") {
            ModelType::AudioGeneration
        } else if model_type_str.contains("embedding") || model_type_str.contains("bert") {
            ModelType::Embedding
        } else {
            ModelType::TextGeneration
        };

        Ok(model_type)
    }

    fn estimate_parameter_count(&self, hidden_size: u64, num_layers: u64, vocab_size: u64) -> u64 {
        // Rough estimation for transformer models
        // Each layer: 4 * hidden_size^2 (attention) + 8 * hidden_size^2 (FFN)
        let params_per_layer = 12 * hidden_size * hidden_size;
        let embedding_params = vocab_size * hidden_size;

        embedding_params + (params_per_layer * num_layers)
    }

    fn estimate_throughput(&self, requirements: &ModelRequirements, devices: &[DeviceCapabilities], precision: &str) -> f32 {
        // Base throughput estimates (tokens/sec) for different model types
        let base_tps = match requirements.model_type {
            ModelType::TextGeneration => {
                // Estimate based on model size
                let params_b = self.estimate_parameter_count(4096, 32, 32000) as f32 / 1e9;
                if params_b < 3.0 { 100.0 }
                else if params_b < 7.0 { 80.0 }
                else if params_b < 13.0 { 50.0 }
                else if params_b < 30.0 { 30.0 }
                else { 15.0 }
            },
            ModelType::VisionLanguage => 20.0,
            ModelType::ImageGeneration => 2.0,
            ModelType::AudioGeneration => 50.0,
            ModelType::Embedding => 200.0,
        };

        // Precision multiplier
        let precision_multiplier = match precision {
            "int4" => 2.0,
            "int8" => 1.5,
            "fp16" | "bf16" => 1.0,
            "fp32" => 0.5,
            _ => 1.0,
        };

        // Backend multiplier
        let backend_multiplier = if devices.iter().any(|d| d.backend == "cuda") {
            1.2 // CUDA is typically faster
        } else if devices.iter().any(|d| d.backend == "metal") {
            1.0
        } else {
            0.3 // CPU
        };

        base_tps * precision_multiplier * backend_multiplier
    }
}
