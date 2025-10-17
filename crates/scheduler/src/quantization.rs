// Quantization support for memory-efficient LLM inference
//
// This module provides support for loading and running quantized models in various formats:
// - GGUF/llama.cpp quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K-Q6K)
// - GPTQ 4-bit quantization (future)
// - AWQ activation-aware quantization (future)
//
// Quantization reduces memory usage by 2-4x and can provide 1.5-3x speedup depending on
// hardware support for low-precision operations.

use anyhow::{anyhow, Result};
use candle_core::{Device as CandleDevice, DType, Tensor};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

#[cfg(feature = "quantization-gguf")]
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

#[cfg(feature = "quantization-gguf")]
use std::collections::HashMap;

/// Supported quantization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFormat {
    /// No quantization (FP16/FP32 model)
    None,
    /// GGUF/llama.cpp format (q4_0, q4_1, q5_0, q5_1, q8_0, etc.)
    Gguf,
    /// GPTQ 4-bit quantization (not yet implemented)
    Gptq,
    /// AWQ activation-aware 4-bit quantization (not yet implemented)
    Awq,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization format to use
    pub format: QuantizationFormat,
    /// Path to the quantized model file
    pub model_path: PathBuf,
    /// Optional: specific quantization level (e.g., "q4_0", "q8_0")
    pub quantization_level: Option<String>,
}

/// Detect the quantization format of a model based on file extension
///
/// # Arguments
/// * `model_path` - Path to the model directory or file
///
/// # Returns
/// * `QuantizationFormat` - Detected format (None if not quantized)
pub fn detect_quantization_format(model_path: &Path) -> QuantizationFormat {
    // Check if the path itself is a GGUF file
    if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        debug!("Detected GGUF file directly: {:?}", model_path);
        return QuantizationFormat::Gguf;
    }

    // Check if directory contains GGUF files
    if model_path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(model_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                    debug!("Detected GGUF file in directory: {:?}", path);
                    return QuantizationFormat::Gguf;
                }
            }
        }
    }

    // Check for GPTQ indicators (config.json with "quantization_config")
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            if let Ok(config_str) = std::fs::read_to_string(&config_path) {
                if config_str.contains("quantization_config") && config_str.contains("gptq") {
                    debug!("Detected GPTQ quantization in config.json");
                    return QuantizationFormat::Gptq;
                }
                if config_str.contains("quantization_config") && config_str.contains("awq") {
                    debug!("Detected AWQ quantization in config.json");
                    return QuantizationFormat::Awq;
                }
            }
        }
    }

    // Default: no quantization (safetensors FP16/FP32)
    QuantizationFormat::None
}

/// Find GGUF file in a directory or return the path if it's already a GGUF file
///
/// # Arguments
/// * `model_path` - Path to model directory or GGUF file
///
/// # Returns
/// * `Result<PathBuf>` - Path to the GGUF file
pub fn find_gguf_file(model_path: &Path) -> Result<PathBuf> {
    // If it's already a GGUF file, return it
    if model_path.is_file() && model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return Ok(model_path.to_path_buf());
    }

    // Search directory for GGUF files
    if model_path.is_dir() {
        let entries = std::fs::read_dir(model_path)
            .map_err(|e| anyhow!("Failed to read model directory: {}", e))?;

        let gguf_files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
            .collect();

        match gguf_files.len() {
            0 => Err(anyhow!("No GGUF files found in directory: {:?}", model_path)),
            1 => Ok(gguf_files[0].clone()),
            _ => {
                warn!(
                    "Multiple GGUF files found, using first: {:?}",
                    gguf_files[0]
                );
                Ok(gguf_files[0].clone())
            }
        }
    } else {
        Err(anyhow!(
            "Model path is neither a GGUF file nor a directory: {:?}",
            model_path
        ))
    }
}

/// Load GGUF quantization metadata
///
/// This reads the GGUF file header and metadata without loading the full model weights.
///
/// # Arguments
/// * `gguf_path` - Path to GGUF file
///
/// # Returns
/// * `Result<GgufMetadata>` - Metadata extracted from GGUF file
#[cfg(feature = "quantization-gguf")]
pub fn load_gguf_metadata(gguf_path: &Path) -> Result<GgufMetadata> {
    use std::fs::File;

    debug!("Loading GGUF metadata from: {:?}", gguf_path);

    let mut file = File::open(gguf_path)
        .map_err(|e| anyhow!("Failed to open GGUF file {:?}: {}", gguf_path, e))?;

    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("Failed to read GGUF file {:?}: {}", gguf_path, e))?;

    // Extract key metadata
    let mut metadata = GgufMetadata {
        path: gguf_path.to_path_buf(),
        tensor_count: content.tensor_infos.len(),
        metadata_kv: std::collections::HashMap::new(),
        quantization_type: None,
    };

    // Extract metadata key-value pairs
    for (key, value) in content.metadata.iter() {
        metadata.metadata_kv.insert(key.clone(), format!("{:?}", value));
    }

    // Try to determine quantization type from tensor data
    if let Some((_name, tensor_info)) = content.tensor_infos.iter().next() {
        metadata.quantization_type = Some(format!("{:?}", tensor_info.ggml_dtype));
    }

    info!(
        "GGUF metadata loaded: {} tensors, quantization: {:?}",
        metadata.tensor_count, metadata.quantization_type
    );

    Ok(metadata)
}

/// GGUF model metadata
#[cfg(feature = "quantization-gguf")]
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// Path to GGUF file
    pub path: PathBuf,
    /// Number of tensors in the model
    pub tensor_count: usize,
    /// Metadata key-value pairs
    pub metadata_kv: std::collections::HashMap<String, String>,
    /// Quantization type (e.g., "Q4_0", "Q8_0")
    pub quantization_type: Option<String>,
}

/// Load a quantized model (GGUF format)
///
/// This is the main entry point for loading quantized models. Currently supports GGUF format.
///
/// # Arguments
/// * `model_path` - Path to model directory or GGUF file
/// * `device` - Candle device to load model on
///
/// # Returns
/// * `Result<QuantizedModelHandle>` - Handle to the loaded quantized model
#[cfg(feature = "quantization-gguf")]
pub fn load_quantized_model(
    model_path: &Path,
    device: &CandleDevice,
) -> Result<QuantizedModelHandle> {
    let format = detect_quantization_format(model_path);

    match format {
        QuantizationFormat::Gguf => {
            info!("Loading GGUF quantized model from: {:?}", model_path);
            load_gguf_model(model_path, device)
        }
        QuantizationFormat::Gptq => {
            Err(anyhow!("GPTQ quantization not yet implemented"))
        }
        QuantizationFormat::Awq => {
            Err(anyhow!("AWQ quantization not yet implemented"))
        }
        QuantizationFormat::None => {
            Err(anyhow!("Model is not quantized (use standard loading)"))
        }
    }
}

/// Load a GGUF quantized model
///
/// # Arguments
/// * `model_path` - Path to model directory or GGUF file
/// * `device` - Candle device to load model on
///
/// # Returns
/// * `Result<QuantizedModelHandle>` - Handle to the loaded GGUF model
#[cfg(feature = "quantization-gguf")]
fn load_gguf_model(
    model_path: &Path,
    device: &CandleDevice,
) -> Result<QuantizedModelHandle> {
    let gguf_path = find_gguf_file(model_path)?;

    // Load full GGUF model with all weights
    let gguf_model = load_gguf_model_full(&gguf_path, device)?;

    let metadata = gguf_model.metadata.clone();

    info!(
        "GGUF model fully loaded: {} tensors, type: {:?}, quantized size: {:.2} MB, dequantized size: {:.2} MB",
        metadata.tensor_count,
        metadata.quantization_type,
        gguf_model.quantized_memory_usage() as f64 / 1024.0 / 1024.0,
        gguf_model.dequantized_memory_usage() as f64 / 1024.0 / 1024.0
    );

    Ok(QuantizedModelHandle {
        format: QuantizationFormat::Gguf,
        metadata: Some(metadata),
        gguf_model: Some(gguf_model),
    })
}

/// Loaded GGUF model with quantized tensors
#[cfg(feature = "quantization-gguf")]
pub struct GgufQuantizedModel {
    /// Model metadata
    pub metadata: GgufMetadata,
    /// Quantized tensors by name
    pub tensors: HashMap<String, QTensor>,
    /// Target device for dequantization
    pub device: CandleDevice,
}

#[cfg(feature = "quantization-gguf")]
impl std::fmt::Debug for GgufQuantizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufQuantizedModel")
            .field("metadata", &self.metadata)
            .field("tensor_count", &self.tensors.len())
            .field("device", &format!("{:?}", self.device))
            .finish()
    }
}

#[cfg(feature = "quantization-gguf")]
impl GgufQuantizedModel {
    /// Dequantize a specific tensor by name to FP16
    ///
    /// This converts a quantized tensor (INT4/INT8) to FP16 for computation
    pub fn dequantize_tensor(&self, tensor_name: &str) -> Result<Tensor> {
        let qtensor = self.tensors.get(tensor_name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", tensor_name))?;

        // Dequantize to FP16
        let dequantized = qtensor.dequantize(&self.device)
            .map_err(|e| anyhow!("Failed to dequantize tensor {}: {}", tensor_name, e))?;

        Ok(dequantized)
    }

    /// Get all tensor names in the model
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get memory usage estimate in bytes (quantized)
    pub fn quantized_memory_usage(&self) -> usize {
        self.tensors.values()
            .map(|t| {
                let elem_count: usize = t.shape().dims().iter().product();
                let bytes_per_elem = match t.dtype() {
                    GgmlDType::Q4_0 | GgmlDType::Q4_1 => 0.5,  // 4 bits per element
                    GgmlDType::Q5_0 | GgmlDType::Q5_1 => 0.625, // 5 bits per element
                    GgmlDType::Q8_0 | GgmlDType::Q8_1 => 1.0,   // 8 bits per element
                    GgmlDType::F32 => 4.0,
                    GgmlDType::F16 => 2.0,
                    _ => 1.0, // Approximate for K-quants
                };
                (elem_count as f64 * bytes_per_elem) as usize
            })
            .sum()
    }

    /// Get memory usage if dequantized to FP16 in bytes
    pub fn dequantized_memory_usage(&self) -> usize {
        self.tensors.values()
            .map(|t| {
                let elem_count: usize = t.shape().dims().iter().product();
                elem_count * 2 // FP16 is 2 bytes per element
            })
            .sum()
    }
}

/// Load full GGUF model with all weights
///
/// This loads the complete GGUF file including all quantized tensor weights
///
/// # Arguments
/// * `gguf_path` - Path to GGUF file
/// * `device` - Target device for dequantization
///
/// # Returns
/// * `Result<GgufQuantizedModel>` - Loaded quantized model with all weights
#[cfg(feature = "quantization-gguf")]
pub fn load_gguf_model_full(
    gguf_path: &Path,
    device: &CandleDevice,
) -> Result<GgufQuantizedModel> {
    use std::fs::File;

    info!("Loading full GGUF model from: {:?}", gguf_path);

    let mut file = File::open(gguf_path)
        .map_err(|e| anyhow!("Failed to open GGUF file {:?}: {}", gguf_path, e))?;

    // Read GGUF content
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("Failed to read GGUF file {:?}: {}", gguf_path, e))?;

    // Load metadata
    let mut metadata = GgufMetadata {
        path: gguf_path.to_path_buf(),
        tensor_count: content.tensor_infos.len(),
        metadata_kv: HashMap::new(),
        quantization_type: None,
    };

    for (key, value) in content.metadata.iter() {
        metadata.metadata_kv.insert(key.clone(), format!("{:?}", value));
    }

    if let Some((_name, tensor_info)) = content.tensor_infos.iter().next() {
        metadata.quantization_type = Some(format!("{:?}", tensor_info.ggml_dtype));
    }

    // Load all tensors
    let mut tensors = HashMap::new();

    for (tensor_name, _tensor_info) in content.tensor_infos.iter() {
        debug!("Loading quantized tensor: {}", tensor_name);

        // Load the quantized tensor
        // Note: QTensor is created from the GGUF content
        match content.tensor(&mut file, tensor_name, device) {
            Ok(qtensor) => {
                tensors.insert(tensor_name.clone(), qtensor);
            }
            Err(e) => {
                warn!("Failed to load tensor {}: {}", tensor_name, e);
                // Continue with other tensors
            }
        }
    }

    let quantized_mb = metadata.tensor_count as f64 * 0.5 / 1024.0 / 1024.0; // Rough estimate
    info!(
        "GGUF model loaded: {} tensors (~{:.1} MB quantized), type: {:?}",
        metadata.tensor_count, quantized_mb, metadata.quantization_type
    );

    Ok(GgufQuantizedModel {
        metadata,
        tensors,
        device: device.clone(),
    })
}

/// Handle to a loaded quantized model
///
/// This represents a quantized model that has been loaded into memory and is ready for inference.
#[cfg(feature = "quantization-gguf")]
#[derive(Debug)]
pub struct QuantizedModelHandle {
    /// Quantization format
    pub format: QuantizationFormat,
    /// Model metadata (GGUF only)
    pub metadata: Option<GgufMetadata>,
    /// Loaded GGUF model with weights (if loaded)
    pub gguf_model: Option<GgufQuantizedModel>,
}

/// Check if quantization features are enabled at compile time
///
/// Returns true if any quantization feature is enabled
pub fn is_quantization_enabled() -> bool {
    cfg!(feature = "quantization")
}

/// Check if GGUF quantization is enabled at compile time
pub fn is_gguf_enabled() -> bool {
    cfg!(feature = "quantization-gguf")
}

/// Check if a model path points to a quantized model
///
/// This is a convenience function that checks if a model uses quantization.
///
/// # Arguments
/// * `model_path` - Path to model directory or file
///
/// # Returns
/// * `bool` - True if the model is quantized
pub fn is_quantized_model(model_path: &Path) -> bool {
    detect_quantization_format(model_path) != QuantizationFormat::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gguf_extension() {
        let path = Path::new("/path/to/model.gguf");
        assert_eq!(detect_quantization_format(path), QuantizationFormat::Gguf);
    }

    #[test]
    fn test_detect_no_quantization() {
        let path = Path::new("/path/to/model.safetensors");
        assert_eq!(detect_quantization_format(path), QuantizationFormat::None);
    }

    #[test]
    fn test_is_quantized_model() {
        let gguf_path = Path::new("/path/to/model.gguf");
        assert!(is_quantized_model(gguf_path));

        let safetensors_path = Path::new("/path/to/model.safetensors");
        assert!(!is_quantized_model(safetensors_path));
    }
}
