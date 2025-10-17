use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::{Path, PathBuf};
use tracing::{info, debug};

/// LLM model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub hidden_size: usize,
    #[allow(dead_code)]
    pub num_heads: usize,
    #[allow(dead_code)]
    pub vocab_size: usize,
    pub context_length: usize,
}

/// Model loader for LLM inference
pub struct LlmModelLoader {
    #[allow(dead_code)]
    cache_dir: PathBuf,
}

impl LlmModelLoader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Download model from HuggingFace Hub
    #[allow(dead_code)]
    pub async fn download_from_hub(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        info!(model_id = %model_id, "Downloading model from HuggingFace Hub");

        let api = Api::new()?;

        let repo = if let Some(rev) = revision {
            api.repo(Repo::with_revision(
                model_id.to_string(),
                RepoType::Model,
                rev.to_string(),
            ))
        } else {
            api.model(model_id.to_string())
        };

        // Download config
        let config_path = repo.get("config.json")
            .context("Failed to download config.json")?;

        info!(config_path = %config_path.display(), "Downloaded config");

        // Download model weights (safetensors preferred)
        let model_files = self.find_model_files(&repo).await?;

        info!(num_files = model_files.len(), "Downloaded model files");

        Ok(config_path.parent().unwrap().to_path_buf())
    }

    #[allow(dead_code)]
    async fn find_model_files(&self, repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        // Try safetensors first
        for i in 0..100 {
            let filename = if i == 0 {
                "model.safetensors".to_string()
            } else {
                format!("model-{:05}-of-*.safetensors", i + 1)
            };

            match repo.get(&filename) {
                Ok(path) => {
                    files.push(path);
                }
                Err(_) => break,
            }
        }

        // Fallback to PyTorch
        if files.is_empty() {
            for i in 0..100 {
                let filename = if i == 0 {
                    "pytorch_model.bin".to_string()
                } else {
                    format!("pytorch_model-{:05}-of-*.bin", i + 1)
                };

                match repo.get(&filename) {
                    Ok(path) => {
                        files.push(path);
                    }
                    Err(_) => break,
                }
            }
        }

        if files.is_empty() {
            anyhow::bail!("No model files found");
        }

        Ok(files)
    }

    /// Load model metadata from config
    pub fn load_metadata(&self, model_dir: &Path) -> Result<ModelMetadata> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
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

        // Estimate parameters
        let num_parameters = Self::estimate_parameters(hidden_size, num_layers, vocab_size);

        let name = config["model_type"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        debug!(
            name = %name,
            num_parameters_b = num_parameters / 1_000_000_000,
            num_layers,
            hidden_size,
            "Loaded model metadata"
        );

        Ok(ModelMetadata {
            name,
            num_parameters,
            num_layers,
            hidden_size,
            num_heads,
            vocab_size,
            context_length,
        })
    }

    fn estimate_parameters(hidden_size: usize, num_layers: usize, vocab_size: usize) -> u64 {
        // Rough estimate for transformer models
        let embedding_params = vocab_size * hidden_size;
        let attention_params = num_layers * (4 * hidden_size * hidden_size); // Q, K, V, O
        let ffn_params = num_layers * (8 * hidden_size * hidden_size); // Typical 4x expansion
        let layer_norm_params = num_layers * (2 * hidden_size);

        (embedding_params + attention_params + ffn_params + layer_norm_params) as u64
    }

    /// Load model weights from safetensors
    pub fn load_safetensors(&self, model_dir: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>> {
        let mut files = Vec::new();

        // Load all safetensors files
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let data = std::fs::read(&path)?;
                files.push((path.clone(), data));
                info!(file = %path.display(), "Loaded safetensors file");
            }
        }

        if files.is_empty() {
            anyhow::bail!("No safetensors files found in {}", model_dir.display());
        }

        Ok(files)
    }

    /// Calculate required VRAM for model
    pub fn calculate_vram_requirement(
        &self,
        metadata: &ModelMetadata,
        batch_size: usize,
        precision: Precision,
    ) -> u64 {
        let bytes_per_param = match precision {
            Precision::FP32 => 4,
            Precision::FP16 | Precision::BF16 => 2,
            Precision::INT8 => 1,
            Precision::INT4 => 0, // Handled specially
        };

        let model_weights = metadata.num_parameters * bytes_per_param;

        // KV cache: batch_size * num_layers * 2 (K+V) * hidden_size * context_length * bytes
        let kv_cache = (batch_size
            * metadata.num_layers
            * 2
            * metadata.hidden_size
            * metadata.context_length
            * bytes_per_param as usize) as u64;

        // Activation memory (rough estimate)
        let activations = (batch_size * metadata.hidden_size * 4) as u64 * bytes_per_param;

        let total = model_weights + kv_cache + activations;

        info!(
            model_gb = model_weights / (1024 * 1024 * 1024),
            kv_cache_gb = kv_cache / (1024 * 1024 * 1024),
            total_gb = total / (1024 * 1024 * 1024),
            "VRAM requirement calculated"
        );

        total
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}
