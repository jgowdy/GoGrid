use corpgrid_common::{DeviceInfo, DeviceReputation, GpuBackend, JobSpec};
use std::collections::HashSet;
use tracing::{debug, info};

/// Candidate device for job placement
#[derive(Debug, Clone)]
pub struct DeviceCandidate {
    pub device_id: String,
    pub info: DeviceInfo,
    pub reputation: DeviceReputation,
    pub on_ac_power: bool,
    pub available_vram: u64,
    pub thermal_headroom: f64, // 0.0 to 1.0
    pub current_utilization: f64, // 0.0 to 1.0
}

/// Scoring weights for placement (must sum to 1.0)
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub reputation: f64,
    pub fit: f64,
    pub thermal_headroom: f64,
    pub fairness: f64,
    pub correlation_penalty: f64,
    pub utilization: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            reputation: 0.4,
            fit: 0.2,
            thermal_headroom: 0.1,
            fairness: 0.1,
            correlation_penalty: 0.15,
            utilization: 0.05,
        }
    }
}

/// Placement engine for scheduling jobs to devices
pub struct PlacementEngine {
    weights: ScoringWeights,
}

impl PlacementEngine {
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
        }
    }

    pub fn with_weights(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    /// Filter candidates by requirements
    pub fn filter_candidates(
        &self,
        candidates: &[DeviceCandidate],
        spec: &JobSpec,
    ) -> Vec<DeviceCandidate> {
        candidates
            .iter()
            .filter(|c| self.meets_requirements(c, spec))
            .cloned()
            .collect()
    }

    fn meets_requirements(&self, candidate: &DeviceCandidate, spec: &JobSpec) -> bool {
        // Must be on AC power
        if !candidate.on_ac_power {
            debug!(
                device_id = %candidate.device_id,
                "Device rejected: not on AC power"
            );
            return false;
        }

        // For LLM inference jobs, resources are determined server-side
        // So we skip resource checks here
        let Some(ref resources) = spec.resources else {
            return true;
        };

        // Check GPU backend compatibility
        let device_backends: HashSet<GpuBackend> = candidate
            .info
            .gpus
            .iter()
            .map(|g| g.backend)
            .collect();

        let required_backends: HashSet<GpuBackend> = resources
            .backend
            .iter()
            .copied()
            .collect();

        if device_backends.is_disjoint(&required_backends) {
            debug!(
                device_id = %candidate.device_id,
                "Device rejected: no compatible GPU backend"
            );
            return false;
        }

        // Check VRAM requirement
        let vram_gb = candidate.available_vram / (1024 * 1024 * 1024);
        if vram_gb < resources.vram_gb_min {
            debug!(
                device_id = %candidate.device_id,
                available_vram_gb = vram_gb,
                required_vram_gb = resources.vram_gb_min,
                "Device rejected: insufficient VRAM"
            );
            return false;
        }

        true
    }

    /// Score candidates for placement
    pub fn score_candidates(
        &self,
        candidates: &[DeviceCandidate],
        spec: &JobSpec,
        already_assigned: &[String], // Device IDs already assigned for this shard
    ) -> Vec<ScoredCandidate> {
        let mut scored: Vec<ScoredCandidate> = candidates
            .iter()
            .map(|c| {
                let score = self.compute_score(c, spec, already_assigned);
                ScoredCandidate {
                    device_id: c.device_id.clone(),
                    device: c.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored
    }

    fn compute_score(
        &self,
        candidate: &DeviceCandidate,
        spec: &JobSpec,
        already_assigned: &[String],
    ) -> f64 {
        let mut score = 0.0;

        // 1. Reputation score (0.4 weight)
        let reputation_score = candidate.reputation.lower_bound();
        score += reputation_score * self.weights.reputation;

        // 2. Fit score (0.2 weight) - prefer devices that fit well
        if let Some(ref resources) = spec.resources {
            let vram_gb = candidate.available_vram / (1024 * 1024 * 1024);
            let vram_ratio = resources.vram_gb_min as f64 / vram_gb as f64;
            let fit_score = (vram_ratio * 0.8 + 0.2).min(1.0); // Prefer tighter fit
            score += fit_score * self.weights.fit;
        } else {
            // For LLM jobs, just give a neutral fit score
            score += 0.5 * self.weights.fit;
        }

        // 3. Thermal headroom (0.1 weight)
        score += candidate.thermal_headroom * self.weights.thermal_headroom;

        // 4. Fairness - penalize devices already heavily utilized (0.1 weight)
        let fairness_score = 1.0 - candidate.current_utilization;
        score += fairness_score * self.weights.fairness;

        // 5. Correlation penalty - penalize devices already assigned (0.15 weight)
        let correlation_penalty = if already_assigned.contains(&candidate.device_id) {
            0.0 // Heavy penalty for same device
        } else {
            1.0
        };
        score += correlation_penalty * self.weights.correlation_penalty;

        // 6. Utilization - prefer underutilized devices (0.05 weight)
        let utilization_score = 1.0 - candidate.current_utilization;
        score += utilization_score * self.weights.utilization;

        score
    }

    /// Select N devices for replication with diversity
    pub fn select_diverse_replicas(
        &self,
        scored: &[ScoredCandidate],
        count: usize,
    ) -> Vec<ScoredCandidate> {
        if scored.len() <= count {
            return scored.to_vec();
        }

        let mut selected = Vec::with_capacity(count);
        let mut used_sites = HashSet::new();
        let mut used_backends = HashSet::new();
        let mut used_driver_versions = HashSet::new();

        // First pass: prefer diversity
        for candidate in scored {
            if selected.len() >= count {
                break;
            }

            let site = candidate.device.info.site.clone();
            let backends: HashSet<GpuBackend> = candidate
                .device
                .info
                .gpus
                .iter()
                .map(|g| g.backend)
                .collect();
            let driver_versions: HashSet<String> = candidate
                .device
                .info
                .gpus
                .iter()
                .map(|g| g.driver_version.clone())
                .collect();

            // Prefer different site, backend, and driver version
            let is_diverse = site.as_ref().map_or_else(|| true, |s| !used_sites.contains(s))
                && backends.iter().any(|b| !used_backends.contains(b))
                && driver_versions.iter().any(|v| !used_driver_versions.contains(v));

            if is_diverse || selected.is_empty() {
                if let Some(ref s) = site {
                    used_sites.insert(s.clone());
                }
                used_backends.extend(backends);
                used_driver_versions.extend(driver_versions);
                selected.push(candidate.clone());
            }
        }

        // Second pass: fill remaining slots with best scores
        for candidate in scored {
            if selected.len() >= count {
                break;
            }

            if !selected.iter().any(|s| s.device_id == candidate.device_id) {
                selected.push(candidate.clone());
            }
        }

        info!(
            selected_count = selected.len(),
            requested_count = count,
            "Selected diverse replicas"
        );

        selected
    }
}

impl Default for PlacementEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub device_id: String,
    pub device: DeviceCandidate,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use corpgrid_common::{GpuInfo, ResourceRequirements, RedundancyConfig, TimeoutConfig, CheckpointConfig};

    fn create_test_device(
        id: &str,
        backend: GpuBackend,
        vram_gb: u64,
        on_ac: bool,
        reputation_alpha: f64,
        reputation_beta: f64,
    ) -> DeviceCandidate {
        DeviceCandidate {
            device_id: id.to_string(),
            info: DeviceInfo {
                device_id: id.to_string(),
                hostname: format!("host-{}", id),
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                gpus: vec![GpuInfo {
                    name: "Test GPU".to_string(),
                    backend,
                    vram_bytes: vram_gb * 1024 * 1024 * 1024,
                    driver_version: "1.0".to_string(),
                    compute_capability: Some("8.0".to_string()),
                }],
                memory_bytes: 64 * 1024 * 1024 * 1024,
                cpu_cores: 16,
                site: Some("site1".to_string()),
            },
            reputation: DeviceReputation::with_prior(reputation_alpha, reputation_beta),
            on_ac_power: on_ac,
            available_vram: vram_gb * 1024 * 1024 * 1024,
            thermal_headroom: 0.8,
            current_utilization: 0.2,
        }
    }

    fn create_test_spec(backend: GpuBackend, vram_gb: u64) -> JobSpec {
        JobSpec {
            resources: Some(ResourceRequirements {
                gpu: 1,
                backend: vec![backend],
                vram_gb_min: vram_gb,
                cpu_cores: None,
                memory_gb: None,
            }),
            redundancy: RedundancyConfig::default(),
            timeouts: TimeoutConfig::default(),
            checkpointing: CheckpointConfig::default(),
            labels: Default::default(),
            job_type: corpgrid_common::JobType::Compute,
        }
    }

    #[test]
    fn test_filter_ac_power() {
        let engine = PlacementEngine::new();
        let spec = create_test_spec(GpuBackend::Cuda, 8);

        let candidates = vec![
            create_test_device("dev1", GpuBackend::Cuda, 16, true, 10.0, 1.0),
            create_test_device("dev2", GpuBackend::Cuda, 16, false, 10.0, 1.0), // On battery
        ];

        let filtered = engine.filter_candidates(&candidates, &spec);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].device_id, "dev1");
    }

    #[test]
    fn test_filter_vram() {
        let engine = PlacementEngine::new();
        let spec = create_test_spec(GpuBackend::Cuda, 16);

        let candidates = vec![
            create_test_device("dev1", GpuBackend::Cuda, 24, true, 10.0, 1.0), // Sufficient
            create_test_device("dev2", GpuBackend::Cuda, 8, true, 10.0, 1.0),  // Insufficient
        ];

        let filtered = engine.filter_candidates(&candidates, &spec);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].device_id, "dev1");
    }

    #[test]
    fn test_filter_backend() {
        let engine = PlacementEngine::new();
        let spec = create_test_spec(GpuBackend::Cuda, 8);

        let candidates = vec![
            create_test_device("dev1", GpuBackend::Cuda, 16, true, 10.0, 1.0),
            create_test_device("dev2", GpuBackend::Metal, 16, true, 10.0, 1.0), // Wrong backend
        ];

        let filtered = engine.filter_candidates(&candidates, &spec);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].device_id, "dev1");
    }

    #[test]
    fn test_scoring_reputation() {
        let engine = PlacementEngine::new();
        let spec = create_test_spec(GpuBackend::Cuda, 8);

        let candidates = vec![
            create_test_device("dev1", GpuBackend::Cuda, 16, true, 20.0, 1.0), // Excellent
            create_test_device("dev2", GpuBackend::Cuda, 16, true, 5.0, 10.0), // Poor
        ];

        let filtered = engine.filter_candidates(&candidates, &spec);
        let scored = engine.score_candidates(&filtered, &spec, &[]);

        assert_eq!(scored[0].device_id, "dev1"); // Higher reputation should score higher
    }
}
