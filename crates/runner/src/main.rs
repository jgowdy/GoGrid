use anyhow::Result;
use corpgrid_common::JobSpec;
use std::path::PathBuf;
use tracing::{info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(target_os = "linux")]
mod cuda_executor;

#[cfg(target_os = "macos")]
mod metal_executor;

mod llm_loader;

#[cfg(target_os = "linux")]
mod llm_inference_cuda;

#[cfg(target_os = "macos")]
mod llm_inference_metal;

/// Sandboxed job runner - executes actual compute workloads
/// This is the process that runs inside the sandbox and performs GPU/CPU work
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize minimal tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corpgrid_runner=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("CorpGrid Runner starting...");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: corpgrid-runner <job_dir>");
        std::process::exit(1);
    }

    let job_dir = PathBuf::from(&args[1]);

    // Load job specification
    let job_spec_path = job_dir.join("job.yaml");
    let job_spec = load_job_spec(&job_spec_path)?;

    info!(
        backend = ?job_spec.resources.as_ref().and_then(|r| r.backend.first()),
        vram_gb_min = job_spec.resources.as_ref().map(|r| r.vram_gb_min).unwrap_or(0),
        "Loaded job specification"
    );

    // Determine backend to use
    let backend = select_backend(&job_spec)?;
    info!(backend = ?backend, "Selected backend");

    // Execute job based on backend
    match backend {
        Backend::Cuda => {
            info!("Running CUDA workload");
            run_cuda_job(&job_dir, &job_spec).await?;
        }
        Backend::Metal => {
            info!("Running Metal workload");
            run_metal_job(&job_dir, &job_spec).await?;
        }
    }

    info!("Job completed successfully");
    Ok(())
}

#[derive(Debug)]
#[allow(dead_code)]
enum Backend {
    Cuda,
    Metal,
}

fn load_job_spec(path: &PathBuf) -> Result<JobSpec> {
    let contents = std::fs::read_to_string(path)?;
    let spec: JobSpec = serde_yaml::from_str(&contents)?;
    Ok(spec)
}

fn select_backend(spec: &JobSpec) -> Result<Backend> {
    let backends = spec.resources
        .as_ref()
        .map(|r| &r.backend)
        .ok_or_else(|| anyhow::anyhow!("No resources specified in job spec"))?;

    // Check available backends on this system
    #[cfg(target_os = "linux")]
    {
        if backends.contains(&corpgrid_common::GpuBackend::Cuda) {
            return Ok(Backend::Cuda);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if backends.contains(&corpgrid_common::GpuBackend::Metal) {
            return Ok(Backend::Metal);
        }
    }

    anyhow::bail!("No compatible backend found")
}

#[cfg(target_os = "linux")]
async fn run_cuda_job(job_dir: &PathBuf, spec: &JobSpec) -> Result<()> {
    use cuda_executor::CudaExecutor;
    use llm_inference_cuda::{CudaLlmInference, MultiGpuLlmInference};

    info!("Initializing CUDA executor");

    // Check if this is an LLM inference job
    if spec.labels.get("job_type") == Some(&"llm_inference".to_string()) {
        info!("Running LLM inference job");

        // Check for multi-GPU requirement
        let num_gpus = spec.resources.gpu as usize;

        if num_gpus > 1 {
            // Multi-GPU inference
            info!(num_gpus, "Using multi-GPU inference");

            let device_ordinals: Vec<usize> = (0..num_gpus).collect();
            let precision = parse_precision(spec.labels.get("precision"));

            let model_path = job_dir.join("model");
            let mut llm = MultiGpuLlmInference::load_model(&device_ordinals, &model_path, precision).await?;

            // Load input
            let input_ids = load_input_tokens(job_dir)?;
            let max_tokens = spec.labels.get("max_new_tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);

            // Generate
            let output_ids = llm.generate(&input_ids, max_tokens).await?;

            // Save output
            save_output_tokens(job_dir, &output_ids)?;
        } else {
            // Single GPU inference
            let device_ordinal = std::env::var("CUDA_VISIBLE_DEVICES")
                .ok()
                .and_then(|s| s.split(',').next().and_then(|d| d.parse().ok()))
                .unwrap_or(0);

            let precision = parse_precision(spec.labels.get("precision"));
            let model_path = job_dir.join("model");

            let mut llm = CudaLlmInference::load_model(device_ordinal, &model_path, precision).await?;

            info!(
                model = %llm.metadata().name,
                parameters_b = llm.metadata().num_parameters / 1_000_000_000,
                estimated_throughput = llm.estimate_throughput(),
                "LLM loaded"
            );

            // Load input
            let input_ids = load_input_tokens(job_dir)?;
            let max_tokens = spec.labels.get("max_new_tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);

            let temperature = spec.labels.get("temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7);

            let top_p = spec.labels.get("top_p")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9);

            // Generate
            let output_ids = llm.generate(&input_ids, max_tokens, temperature, top_p).await?;

            // Save output
            save_output_tokens(job_dir, &output_ids)?;
        }

        info!("LLM inference completed");
        return Ok(());
    }

    // Regular CUDA kernel execution
    let device_ordinal = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|s| s.split(',').next().and_then(|d| d.parse().ok()))
        .unwrap_or(0);

    let executor = CudaExecutor::new(device_ordinal)?;

    // Check for checkpoint to resume from
    let checkpoint_dir = job_dir.join("checkpoints");
    if checkpoint_dir.exists() {
        info!("Checkpoint directory found, resuming from checkpoint");
    }

    // Execute job
    let _result = executor.execute(job_dir, spec).await?;

    info!("CUDA job execution completed successfully");
    Ok(())
}

#[cfg(target_os = "macos")]
async fn run_metal_job(job_dir: &PathBuf, spec: &JobSpec) -> Result<()> {
    use metal_executor::MetalExecutor;
    use llm_inference_metal::MetalLlmInference;

    info!("Initializing Metal executor");

    // Check if this is an LLM inference job
    if spec.labels.get("job_type") == Some(&"llm_inference".to_string()) {
        info!("Running LLM inference job on Metal");

        let precision = parse_precision(spec.labels.get("precision"));
        let model_path = job_dir.join("model");

        let mut llm = MetalLlmInference::load_model(&model_path, precision).await?;

        info!(
            model = %llm.metadata().name,
            parameters_b = llm.metadata().num_parameters / 1_000_000_000,
            estimated_throughput = llm.estimate_throughput(),
            device = "Apple Silicon",
            "LLM loaded on Metal"
        );

        // Load input
        let input_ids = load_input_tokens(job_dir)?;
        let max_tokens = spec.labels.get("max_new_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let temperature = spec.labels.get("temperature")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);

        let top_p = spec.labels.get("top_p")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.9);

        // Generate
        let output_ids = llm.generate(&input_ids, max_tokens, temperature, top_p).await?;

        // Save output
        save_output_tokens(job_dir, &output_ids)?;

        info!("LLM inference completed on Metal");
        return Ok(());
    }

    // Regular Metal shader execution
    let executor = MetalExecutor::new()?;

    // Check for checkpoint to resume from
    let checkpoint_dir = job_dir.join("checkpoints");
    if checkpoint_dir.exists() {
        info!("Checkpoint directory found, resuming from checkpoint");
    }

    // Execute job
    let _result = executor.execute(job_dir, spec).await?;

    info!("Metal job execution completed successfully");
    Ok(())
}

// Helper functions
fn parse_precision(precision_str: Option<&String>) -> llm_loader::Precision {
    match precision_str.map(|s| s.as_str()) {
        Some("fp32") => llm_loader::Precision::FP32,
        Some("fp16") => llm_loader::Precision::FP16,
        Some("bf16") => llm_loader::Precision::BF16,
        Some("int8") => llm_loader::Precision::INT8,
        Some("int4") => llm_loader::Precision::INT4,
        _ => llm_loader::Precision::FP16, // Default
    }
}

fn load_input_tokens(job_dir: &PathBuf) -> Result<Vec<u32>> {
    let input_path = job_dir.join("input_tokens.bin");
    if !input_path.exists() {
        // Default prompt tokens if no input provided
        return Ok(vec![1, 15043, 338, 263, 1243]); // Example: "This is a test"
    }

    let data = std::fs::read(&input_path)?;
    let tokens: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(tokens)
}

fn save_output_tokens(job_dir: &PathBuf, tokens: &[u32]) -> Result<()> {
    let output_path = job_dir.join("output_tokens.bin");
    let mut data = Vec::with_capacity(tokens.len() * 4);

    for &token in tokens {
        data.extend_from_slice(&token.to_le_bytes());
    }

    std::fs::write(&output_path, &data)?;
    Ok(())
}

// Stub for unsupported platforms
#[cfg(not(target_os = "linux"))]
async fn run_cuda_job(_job_dir: &PathBuf, _spec: &JobSpec) -> Result<()> {
    anyhow::bail!("CUDA not supported on this platform")
}

#[cfg(not(target_os = "macos"))]
async fn run_metal_job(_job_dir: &PathBuf, _spec: &JobSpec) -> Result<()> {
    anyhow::bail!("Metal not supported on this platform")
}
