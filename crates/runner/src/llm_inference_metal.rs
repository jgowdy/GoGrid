#[cfg(target_os = "macos")]
use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::{info, debug};

use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config, Cache};
use tokenizers::Tokenizer;

use super::llm_loader::{LlmModelLoader, ModelMetadata, Precision};

/// Metal-based LLM inference engine for Apple Silicon using Candle
pub struct MetalLlmInference {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    metadata: ModelMetadata,
    cache: Cache,
}

impl MetalLlmInference {
    pub async fn load_model(
        model_path: &Path,
        precision: Precision,
    ) -> Result<Self> {
        info!(
            model_path = %model_path.display(),
            precision = ?precision,
            "Loading LLM model on Metal (Apple Silicon) with Candle"
        );

        // Initialize Metal device
        let device = Device::new_metal(0)?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model metadata
        let loader = LlmModelLoader::new(model_path.parent().unwrap().to_path_buf());
        let metadata = loader.load_metadata(model_path)?;

        info!(
            model = %metadata.name,
            parameters_b = metadata.num_parameters / 1_000_000_000,
            layers = metadata.num_layers,
            "Model metadata loaded"
        );

        // Load model config
        // Future: parse config.json to customize model architecture
        let config = Config::config_7b_v1(false); // Use default Llama 7B config

        // Determine dtype based on precision
        let dtype = match precision {
            Precision::FP32 => DType::F32,
            Precision::FP16 | Precision::BF16 => DType::F16,
            Precision::INT8 | Precision::INT4 => DType::F16,
        };

        // Check unified memory availability
        let vram_required = loader.calculate_vram_requirement(&metadata, 1, precision);
        info!(
            vram_required_gb = vram_required / (1024 * 1024 * 1024),
            "Unified memory requirement"
        );

        // Load safetensors weights
        let weights_files = loader.load_safetensors(model_path)?;
        let weight_paths: Vec<PathBuf> = weights_files.into_iter().map(|(p, _)| p).collect();

        // Build VarBuilder from safetensors
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)? };

        // Create model
        let model = Llama::load(vb, &config)?;

        // Initialize KV cache
        let cache = Cache::new(true, dtype, &config, &device)?;

        info!("Model loaded successfully on Metal");

        Ok(Self {
            model,
            tokenizer,
            device,
            metadata,
            cache,
        })
    }

    /// Run inference on input tokens
    pub async fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        info!(
            input_len = input_ids.len(),
            max_new_tokens,
            "Starting generation on Metal with Candle"
        );

        let mut output_ids = input_ids.to_vec();
        let mut pos = 0usize;

        for step in 0..max_new_tokens {
            debug!(step, sequence_len = output_ids.len(), "Generation step");

            // Prepare input tensor
            let input_tensor = Tensor::new(&output_ids[pos..], &self.device)?
                .unsqueeze(0)?;

            // Forward pass
            let logits = self.model.forward(&input_tensor, pos, &mut self.cache)?;

            // Get logits for last token
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = logits.get(logits.dim(0)? - 1)?;

            // Sample next token
            let next_token = self.sample_token(&logits, temperature, top_p)?;
            output_ids.push(next_token);
            pos = output_ids.len() - 1;

            // Check for EOS
            if next_token == 2 || next_token == self.tokenizer.get_vocab_size(false) as u32 {
                break;
            }
        }

        info!(
            generated_tokens = output_ids.len() - input_ids.len(),
            "Generation complete on Metal"
        );

        Ok(output_ids)
    }

    fn sample_token(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        let logits = if temperature <= 0.0 {
            logits.clone()
        } else {
            (logits / temperature as f64)?
        };

        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Top-p (nucleus) sampling
        let mut probs_idx: Vec<(usize, f32)> = probs_vec.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        probs_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let fallback = probs_idx[0].0 as u32;

        let mut cumsum = 0.0;
        let mut sampled_probs = Vec::new();
        for &(idx, prob) in &probs_idx {
            cumsum += prob;
            sampled_probs.push((idx, prob));
            if cumsum >= top_p {
                break;
            }
        }

        // Sample from remaining distribution
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in sampled_probs {
            cumsum += prob;
            if sample < cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to most likely token
        Ok(fallback)
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Estimate tokens per second throughput on Apple Silicon
    pub fn estimate_throughput(&self) -> f32 {
        let params_b = self.metadata.num_parameters as f32 / 1e9;
        // M3 Max can do ~80 tok/s for 7B models
        let base_throughput = 80.0;
        let scaling_factor = 7.0 / params_b;
        base_throughput * scaling_factor
    }
}
