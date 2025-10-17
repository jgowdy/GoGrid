use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Job type - determines execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    /// Regular compute job (CUDA kernel, Metal shader, etc.)
    Compute,
    /// LLM inference job - server-side model hosting
    LlmInference,
}

impl Default for JobType {
    fn default() -> Self {
        Self::Compute
    }
}

/// Job specification from job.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    /// Resource requirements - optional for LLM inference (auto-determined)
    #[serde(default)]
    pub resources: Option<ResourceRequirements>,
    #[serde(default)]
    pub redundancy: RedundancyConfig,
    #[serde(default)]
    pub timeouts: TimeoutConfig,
    #[serde(default)]
    pub checkpointing: CheckpointConfig,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// Job type - determines execution strategy
    #[serde(default)]
    pub job_type: JobType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub gpu: u32,
    pub backend: Vec<GpuBackend>,
    pub vram_gb_min: u64,
    #[serde(default)]
    pub cpu_cores: Option<u32>,
    #[serde(default)]
    pub memory_gb: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    Metal,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "cuda"),
            GpuBackend::Metal => write!(f, "metal"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyConfig {
    pub replication_factor: u32,
    pub quorum: u32,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            replication_factor: 2,
            quorum: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub heartbeat_period_ms: u64,
    pub heartbeat_grace_missed: u32,
    #[serde(default = "default_execution_timeout")]
    pub execution_timeout_s: Option<u64>,
}

fn default_execution_timeout() -> Option<u64> {
    Some(3600) // 1 hour default
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            heartbeat_period_ms: 5000,
            heartbeat_grace_missed: 3,
            execution_timeout_s: default_execution_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub enabled: bool,
    #[serde(default = "default_checkpoint_interval")]
    pub interval_s: u64,
}

fn default_checkpoint_interval() -> u64 {
    60
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_s: default_checkpoint_interval(),
        }
    }
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub gpus: Vec<GpuInfo>,
    pub memory_bytes: u64,
    pub cpu_cores: u32,
    pub site: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub backend: GpuBackend,
    pub vram_bytes: u64,
    pub driver_version: String,
    pub compute_capability: Option<String>, // For CUDA
}

/// Job attempt status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttemptStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
    Timeout,
    Preempted,
}

/// Job shard - unit of work assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobShard {
    pub job_id: String,
    pub shard_id: String,
    pub bundle_hash: Vec<u8>,
    pub spec: JobSpec,
}
