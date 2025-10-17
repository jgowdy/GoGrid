use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor, DType};
use candle_transformers::models::llama as model_llama;
use candle_transformers::models::mistral as model_mistral;
use candle_transformers::models::mixtral as model_mixtral;
use candle_transformers::models::qwen2 as model_qwen2;
use candle_nn::Activation;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::model_hosting::{BackendType, GpuDevice, InferenceBackend, ModelMetadata, Precision};

/// Real LLM inference implementation using Candle
pub struct CandleInferenceBackend {
    model: Arc<Mutex<LlmModel>>,
    device: Device,
    #[allow(dead_code)]
    tokenizer: Arc<Tokenizer>,
    #[allow(dead_code)]
    metadata: ModelMetadata,
}

enum ModelVariant {
    Llama {
        model: model_llama::Llama,
        cache: model_llama::Cache,
    },
    Mistral(model_mistral::Model),
    Mixtral(model_mixtral::Model),
    Qwen2(model_qwen2::ModelForCausalLM),
}

struct LlmModel {
    variant: ModelVariant,
}

impl CandleInferenceBackend {
    pub async fn load(
        devices: &[GpuDevice],
        model_path: &str,
        metadata: &ModelMetadata,
        precision: Precision,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self> {
        info!(
            model_path = %model_path,
            num_devices = devices.len(),
            "Loading model with Candle"
        );

        // Determine device
        let device = if devices.is_empty() {
            Device::Cpu
        } else {
            match devices[0].backend {
                corpgrid_common::GpuBackend::Cuda => {
                    Device::new_cuda(devices[0].device_index)?
                }
                corpgrid_common::GpuBackend::Metal => {
                    Device::new_metal(devices[0].device_index)?
                }
            }
        };

        info!(device = ?device, "Initialized device");

        // Load model config
        let config_path = PathBuf::from(model_path).join("config.json");
        let config_str = tokio::fs::read_to_string(&config_path).await?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        let model_type = config["model_type"]
            .as_str()
            .context("Missing model_type in config")?;

        info!(model_type = %model_type, "Detected model type");

        // Determine dtype from precision
        let dtype = match precision {
            Precision::FP32 => DType::F32,
            Precision::FP16 => DType::F16,
            Precision::BF16 => DType::BF16,
            _ => DType::F16, // Default to FP16
        };

        let vb = Self::load_weights(model_path, &device, precision).await?;

        // Load model based on detected type
        let variant = if model_type == "llama" || model_type == "llama2" {
            // Parse Llama config
            let hidden_size = config["hidden_size"].as_u64().context("Missing hidden_size")? as usize;
            let intermediate_size = config["intermediate_size"].as_u64().context("Missing intermediate_size")? as usize;
            let vocab_size = config["vocab_size"].as_u64().context("Missing vocab_size")? as usize;
            let num_hidden_layers = config["num_hidden_layers"].as_u64().context("Missing num_hidden_layers")? as usize;
            let num_attention_heads = config["num_attention_heads"].as_u64().context("Missing num_attention_heads")? as usize;
            let num_key_value_heads = config.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads);
            let rms_norm_eps = config.get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6);
            let rope_theta = config.get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32;
            let bos_token_id = config.get("bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| Some(v as u32))
                .unwrap_or(Some(1));
            let eos_token_id = config.get("eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| Some(model_llama::LlamaEosToks::Single(v as u32)))
                .unwrap_or(Some(model_llama::LlamaEosToks::Single(2)));
            let max_position_embeddings = config.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(2048);

            let llama_config = model_llama::Config {
                hidden_size,
                intermediate_size,
                vocab_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                rms_norm_eps,
                rope_theta,
                bos_token_id,
                eos_token_id,
                max_position_embeddings,
                rope_scaling: None,
                tie_word_embeddings: false,
                use_flash_attn: false,
            };

            let model = model_llama::Llama::load(vb, &llama_config)?;
            let cache = model_llama::Cache::new(true, dtype, &llama_config, &device)?;

            ModelVariant::Llama { model, cache }
        } else if model_type == "mistral" {
            // Parse Mistral config
            let hidden_size = config["hidden_size"].as_u64().context("Missing hidden_size")? as usize;
            let intermediate_size = config["intermediate_size"].as_u64().context("Missing intermediate_size")? as usize;
            let vocab_size = config["vocab_size"].as_u64().context("Missing vocab_size")? as usize;
            let num_hidden_layers = config["num_hidden_layers"].as_u64().context("Missing num_hidden_layers")? as usize;
            let num_attention_heads = config["num_attention_heads"].as_u64().context("Missing num_attention_heads")? as usize;
            let num_key_value_heads = config.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads);
            let rms_norm_eps = config.get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5);
            let rope_theta = config.get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0);
            let max_position_embeddings = config.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(32768);
            let sliding_window = config.get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let head_dim = config.get("head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let hidden_act = Activation::Silu;  // Mistral uses SiLU activation

            let mistral_config = model_mistral::Config {
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                hidden_act,
                max_position_embeddings,
                rms_norm_eps,
                rope_theta,
                sliding_window,
                use_flash_attn: false,
            };

            let model = model_mistral::Model::new(&mistral_config, vb)?;

            ModelVariant::Mistral(model)
        } else if model_type == "mixtral" {
            // Use predefined Mixtral config (can't construct manually)
            let mixtral_config = model_mixtral::Config::v0_1_8x7b(false);

            let model = model_mixtral::Model::new(&mixtral_config, vb)?;

            ModelVariant::Mixtral(model)
        } else if model_type == "qwen2" {
            // Parse Qwen2 config
            let vocab_size = config["vocab_size"].as_u64().context("Missing vocab_size")? as usize;
            let hidden_size = config["hidden_size"].as_u64().context("Missing hidden_size")? as usize;
            let intermediate_size = config["intermediate_size"].as_u64().context("Missing intermediate_size")? as usize;
            let num_hidden_layers = config["num_hidden_layers"].as_u64().context("Missing num_hidden_layers")? as usize;
            let num_attention_heads = config["num_attention_heads"].as_u64().context("Missing num_attention_heads")? as usize;
            let num_key_value_heads = config.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads);
            let max_position_embeddings = config.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(32768);
            let sliding_window = config.get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(4096);
            let max_window_layers = config.get("max_window_layers")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_hidden_layers);
            let tie_word_embeddings = config.get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let rope_theta = config.get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0);
            let rms_norm_eps = config.get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6);
            let use_sliding_window = config.get("use_sliding_window")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let hidden_act = Activation::Silu;  // Qwen2 uses SiLU activation

            let qwen2_config = model_qwen2::Config {
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                max_position_embeddings,
                sliding_window,
                max_window_layers,
                tie_word_embeddings,
                rope_theta,
                rms_norm_eps,
                use_sliding_window,
                hidden_act,
            };

            let model = model_qwen2::ModelForCausalLM::new(&qwen2_config, vb)?;

            ModelVariant::Qwen2(model)
        } else {
            anyhow::bail!("Unsupported model type: {}. Supported types: llama, llama2, mistral, mixtral, qwen2", model_type);
        };

        let model = LlmModel { variant };

        info!("Model loaded successfully");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            device,
            tokenizer,
            metadata: metadata.clone(),
        })
    }

    async fn load_weights(
        model_path: &str,
        device: &Device,
        precision: Precision,
    ) -> Result<candle_nn::VarBuilder<'static>> {
        use candle_nn::VarBuilder;

        let model_dir = PathBuf::from(model_path);

        // Find safetensors files
        let mut safetensor_files = Vec::new();
        let mut entries = tokio::fs::read_dir(&model_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    safetensor_files.push(path);
                }
            }
        }

        if safetensor_files.is_empty() {
            anyhow::bail!("No safetensors files found in {}", model_dir.display());
        }

        info!(
            num_files = safetensor_files.len(),
            "Loading safetensors files"
        );

        // Load tensors from safetensors into HashMap
        let dtype = match precision {
            Precision::FP32 => candle_core::DType::F32,
            Precision::FP16 => candle_core::DType::F16,
            Precision::BF16 => candle_core::DType::BF16,
            _ => candle_core::DType::F16, // Default to FP16
        };

        let mut tensor_map: HashMap<String, Tensor> = HashMap::new();

        for path in &safetensor_files {
            let data = std::fs::read(path)?;
            let tensors = safetensors::SafeTensors::deserialize(&data)?;

            for (name, view) in tensors.tensors() {
                let shape: Vec<usize> = view.shape().to_vec();
                let tensor = Tensor::from_raw_buffer(
                    view.data(),
                    view.dtype().try_into()?,
                    &shape,
                    device,
                )?;
                tensor_map.insert(name.to_string(), tensor);
            }
        }

        info!(num_tensors = tensor_map.len(), "Loaded tensors");

        // Create VarBuilder from loaded tensors
        let vb = VarBuilder::from_tensors(tensor_map, dtype, device);

        Ok(vb)
    }

    fn sample_token(
        &self,
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?;

        // Apply temperature
        let logits = if temperature > 0.0 {
            (logits / temperature as f64)?
        } else {
            logits
        };

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
        let probs_vec = probs.to_vec1::<f32>()?;

        // Apply top-p (nucleus) sampling
        let token = if top_p < 1.0 {
            Self::sample_top_p(&probs_vec, top_p)?
        } else {
            // Sample from full distribution
            Self::sample_multinomial(&probs_vec)?
        };

        Ok(token as u32)
    }

    fn sample_top_p(probs: &[f32], top_p: f32) -> Result<usize> {
        // Sort probabilities in descending order
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted_indices.len();

        for (idx, &i) in sorted_indices.iter().enumerate() {
            cumsum += probs[i];
            if cumsum >= top_p {
                cutoff_idx = idx + 1;
                break;
            }
        }

        // Sample from top-p subset
        let top_indices = &sorted_indices[..cutoff_idx];
        let top_probs: Vec<f32> = top_indices.iter().map(|&i| probs[i]).collect();

        Self::sample_multinomial(&top_probs)
            .map(|idx| top_indices[idx])
    }

    fn sample_multinomial(probs: &[f32]) -> Result<usize> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample < cumsum {
                return Ok(idx);
            }
        }

        Ok(probs.len() - 1)
    }
}

#[async_trait::async_trait]
impl InferenceBackend for CandleInferenceBackend {
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
            "Starting generation"
        );

        let mut output_ids = input_ids.to_vec();

        // First forward pass - process all input tokens
        let input_tensor = Tensor::new(input_ids, &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        let mut current_pos = 0usize;

        // Process initial prompt
        let mut model_guard = self.model.lock().unwrap();
        let mut logits = match &mut model_guard.variant {
            ModelVariant::Llama { model, cache } => {
                model.forward(&input_tensor, current_pos, cache)?
            }
            ModelVariant::Mistral(model) => {
                model.forward(&input_tensor, current_pos)?
            }
            ModelVariant::Mixtral(model) => {
                model.forward(&input_tensor, current_pos)?
            }
            ModelVariant::Qwen2(model) => {
                model.forward(&input_tensor, current_pos)?
            }
        };
        drop(model_guard);

        current_pos += input_ids.len();

        // Get logits for last token of prompt
        // logits shape: [batch_size=1, vocab_size] (Candle returns only last position)
        // We want: [vocab_size]
        debug!("Logits shape after forward: {:?}", logits.shape());
        let last_logits = logits.i(0)?;  // Index batch=0 -> [vocab_size]
        debug!("Last logits shape: {:?}", last_logits.shape());
        let mut next_token = self.sample_token(&last_logits, temperature, top_p)?;

        debug!(token = next_token, "Sampled first token after prompt");

        output_ids.push(next_token);

        // Generation loop - generate remaining tokens one at a time
        for step in 1..max_new_tokens {
            debug!(step, pos = current_pos, "Generation step");

            // Check for EOS
            if next_token == 2 || next_token == 1 {
                // Common EOS tokens
                break;
            }

            // Process single new token
            let token_tensor = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;

            let mut model_guard = self.model.lock().unwrap();
            logits = match &mut model_guard.variant {
                ModelVariant::Llama { model, cache } => {
                    model.forward(&token_tensor, current_pos, cache)?
                }
                ModelVariant::Mistral(model) => {
                    model.forward(&token_tensor, current_pos)?
                }
                ModelVariant::Mixtral(model) => {
                    model.forward(&token_tensor, current_pos)?
                }
                ModelVariant::Qwen2(model) => {
                    model.forward(&token_tensor, current_pos)?
                }
            };
            drop(model_guard);

            current_pos += 1;

            // Sample next token
            // logits shape: [batch_size=1, vocab_size] (Candle returns only last position)
            // We want: [vocab_size]
            let last_logits = logits.i(0)?;  // Index batch=0 -> [vocab_size]
            next_token = self.sample_token(&last_logits, temperature, top_p)?;

            debug!(token = next_token, "Sampled token");

            output_ids.push(next_token);
        }

        info!(
            input_tokens = input_ids.len(),
            output_tokens = output_ids.len(),
            generated = output_ids.len() - input_ids.len(),
            "Generation complete"
        );

        Ok(output_ids)
    }

    fn backend_type(&self) -> BackendType {
        match self.device {
            Device::Cuda(_) => BackendType::HomogeneousCuda,
            Device::Metal(_) => BackendType::HomogeneousMetal,
            _ => BackendType::HomogeneousCuda, // Default
        }
    }
}
