use anyhow::Result;
use corpgrid_proto::scheduler_server::{Scheduler, SchedulerServer};
use corpgrid_proto::*;
use sqlx::Row;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::db::DbPool;
use crate::heartbeat::HeartbeatManager;
use crate::storage::CasStorage;
use crate::model_hosting::ModelHostingService;
use crate::model_hosting_service::register_agent_gpus_with_hosting;

pub struct SchedulerService {
    db: DbPool,
    heartbeat_manager: Arc<HeartbeatManager>,
    model_hosting: Arc<ModelHostingService>,
    // In-memory state (production would use Redis or similar)
    device_registry: Arc<RwLock<std::collections::HashMap<String, DeviceRegistration>>>,
}

#[derive(Clone)]
struct DeviceRegistration {
    device_info: DeviceInfo,
    power_status: PowerStatus,
    last_seen: chrono::DateTime<chrono::Utc>,
}

impl SchedulerService {
    pub fn new(
        db: DbPool,
        _storage: Arc<CasStorage>,
        heartbeat_manager: Arc<HeartbeatManager>,
        model_hosting: Arc<ModelHostingService>,
    ) -> Self {
        Self {
            db,
            heartbeat_manager,
            model_hosting,
            device_registry: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub fn into_grpc_server(self) -> SchedulerServer<Self> {
        SchedulerServer::new(self)
    }
}

#[tonic::async_trait]
impl Scheduler for SchedulerService {
    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
        let req = request.into_inner();

        info!(
            device_id = %req.device_id,
            "Agent registration request"
        );

        let device_info = req
            .device_info
            .ok_or_else(|| Status::invalid_argument("device_info is required"))?;

        let power_status = req
            .power_status
            .ok_or_else(|| Status::invalid_argument("power_status is required"))?;

        // Check AC power requirement
        if !power_status.on_ac_power {
            warn!(
                device_id = %req.device_id,
                "Device rejected: not on AC power"
            );
            return Ok(Response::new(RegisterAgentResponse {
                accepted: false,
                message: "Device must be on AC power to register".to_string(),
            }));
        }

        // Store device registration
        let mut registry = self.device_registry.write().await;
        registry.insert(
            req.device_id.clone(),
            DeviceRegistration {
                device_info: device_info.clone(),
                power_status,
                last_seen: chrono::Utc::now(),
            },
        );
        drop(registry);

        // Register GPUs with model hosting service
        let gpus: Vec<corpgrid_common::GpuInfo> = device_info.gpus.into_iter()
            .map(|g| corpgrid_common::GpuInfo {
                name: g.name,
                backend: if g.backend == "cuda" {
                    corpgrid_common::GpuBackend::Cuda
                } else {
                    corpgrid_common::GpuBackend::Metal
                },
                vram_bytes: g.vram_bytes,
                driver_version: g.driver_version,
                compute_capability: if g.compute_capability.is_empty() {
                    None
                } else {
                    Some(g.compute_capability)
                },
            })
            .collect();

        register_agent_gpus_with_hosting(&self.model_hosting, req.device_id.clone(), gpus).await;

        // Store agent's public key in database
        if let Err(e) = self.store_agent_public_key(&req.device_id, &req.public_key).await {
            warn!(error = %e, "Failed to store agent public key");
        }

        // Initialize reputation for new agents (alpha=1, beta=1 = Unproven tier)
        // Unproven agents will require higher redundancy until they prove themselves
        if let Err(e) = self.initialize_agent_reputation(&req.device_id).await {
            warn!(error = %e, "Failed to initialize agent reputation");
        }

        info!(
            device_id = %req.device_id,
            "Agent registered successfully"
        );

        Ok(Response::new(RegisterAgentResponse {
            accepted: true,
            message: "Registration successful".to_string(),
        }))
    }

    async fn poll_jobs(
        &self,
        request: Request<PollJobsRequest>,
    ) -> Result<Response<PollJobsResponse>, Status> {
        let req = request.into_inner();

        let power_status = req
            .power_status
            .ok_or_else(|| Status::invalid_argument("power_status is required"))?;

        // Check AC power - critical safety check
        if !power_status.on_ac_power {
            warn!(
                device_id = %req.device_id,
                "Device polling while on battery - no jobs assigned"
            );
            return Ok(Response::new(PollJobsResponse {
                assignments: vec![],
            }));
        }

        // Update last seen
        {
            let mut registry_lock = self.device_registry.write().await;
            if let Some(registration) = registry_lock.get_mut(&req.device_id) {
                registration.last_seen = chrono::Utc::now();
                registration.power_status = power_status;
            }
        }

        // Get pending jobs from database
        let pending_jobs = self.get_pending_jobs().await?;

        if pending_jobs.is_empty() {
            return Ok(Response::new(PollJobsResponse {
                assignments: vec![],
            }));
        }

        // Get device reputation from database
        let reputation = self.get_device_reputation(&req.device_id).await.unwrap_or_default();

        // Get device info for placement
        let device_candidate = {
            let registry = self.device_registry.read().await;
            registry.get(&req.device_id).map(|reg| {
                // Convert proto DeviceInfo to common DeviceInfo
                let common_device_info = corpgrid_common::DeviceInfo {
                    device_id: req.device_id.clone(),
                    hostname: reg.device_info.hostname.clone(),
                    os: reg.device_info.os.clone(),
                    arch: reg.device_info.arch.clone(),
                    gpus: reg.device_info.gpus.iter().map(|g| corpgrid_common::GpuInfo {
                        name: g.name.clone(),
                        backend: if g.backend == "cuda" {
                            corpgrid_common::GpuBackend::Cuda
                        } else {
                            corpgrid_common::GpuBackend::Metal
                        },
                        vram_bytes: g.vram_bytes,
                        driver_version: g.driver_version.clone(),
                        compute_capability: if g.compute_capability.is_empty() {
                            None
                        } else {
                            Some(g.compute_capability.clone())
                        },
                    }).collect(),
                    memory_bytes: reg.device_info.memory_bytes,
                    cpu_cores: reg.device_info.cpu_cores,
                    site: None, // TODO: Extract from labels or config
                };

                crate::placement::DeviceCandidate {
                    device_id: req.device_id.clone(),
                    info: common_device_info.clone(),
                    reputation, // Use actual reputation from database
                    on_ac_power: reg.power_status.on_ac_power,
                    available_vram: common_device_info.gpus.iter()
                        .map(|g| g.vram_bytes)
                        .sum(),
                    thermal_headroom: 0.8, // TODO: track thermal info
                    current_utilization: 0.3, // TODO: track utilization
                }
            })
        };

        let Some(candidate) = device_candidate else {
            return Ok(Response::new(PollJobsResponse {
                assignments: vec![],
            }));
        };

        // Try to assign a job using PlacementEngine
        let placement_engine = crate::placement::PlacementEngine::new();

        for job in pending_jobs {
            // Parse job spec
            let job_spec = match job.spec() {
                Ok(spec) => spec,
                Err(_) => continue, // Skip jobs with invalid specs
            };

            // Check if this device is suitable
            let filtered = placement_engine.filter_candidates(&[candidate.clone()], &job_spec);

            if !filtered.is_empty() {
                // Create lease with heartbeat manager
                let lease = self.heartbeat_manager.create_lease(
                    req.device_id.clone(),
                    job.shard_id.clone(),
                    job_spec.timeouts.heartbeat_period_ms,
                    job_spec.timeouts.heartbeat_grace_missed,
                ).await;

                // Assign this job
                let assignment = JobAssignment {
                    job_id: job.job_id.clone(),
                    shard_id: job.shard_id.clone(),
                    bundle_s3_key: job.bundle_s3_key.clone(),
                    bundle_hash: self.calculate_bundle_hash(&job.bundle_s3_key).await.unwrap_or_default(),
                    bundle_signature: job.bundle_signature.clone(),
                    spec: Some(corpgrid_proto::JobSpec {
                        backend: job_spec.resources
                            .as_ref()
                            .and_then(|r| r.backend.first())
                            .map(|b| b.to_string())
                            .unwrap_or_default(),
                        vram_gb_min: job_spec.resources
                            .as_ref()
                            .map(|r| r.vram_gb_min)
                            .unwrap_or(0),
                        heartbeat_period_ms: job_spec.timeouts.heartbeat_period_ms as u32,
                        heartbeat_grace_missed: job_spec.timeouts.heartbeat_grace_missed,
                        checkpointing_enabled: job_spec.checkpointing.enabled,
                        checkpoint_interval_s: job_spec.checkpointing.interval_s as u32,
                    }),
                    checkpoint_s3_key: self.get_latest_checkpoint(&job.job_id, &job.shard_id).await.unwrap_or_default(),
                    lease_expires_ms: lease.expires_at.timestamp_millis() as u64,
                };

                // Mark as assigned in database
                self.mark_job_assigned(&job.job_id, &job.shard_id, &lease.attempt_id, &req.device_id).await?;

                info!(
                    job_id = %assignment.job_id,
                    shard_id = %assignment.shard_id,
                    attempt_id = %lease.attempt_id,
                    device_id = %req.device_id,
                    "Assigned job to device"
                );

                return Ok(Response::new(PollJobsResponse {
                    assignments: vec![assignment],
                }));
            }
        }

        // No suitable job found
        Ok(Response::new(PollJobsResponse {
            assignments: vec![],
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();

        // Look up attempt_id from database
        let attempt_id_result = sqlx::query_scalar::<_, String>(
            r#"
            SELECT attempt_id
            FROM job_shards
            WHERE job_id = $1 AND shard_id = $2 AND status IN ('assigned', 'running')
            "#
        )
        .bind(&req.job_id)
        .bind(&req.shard_id)
        .fetch_optional(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        let Some(attempt_id) = attempt_id_result else {
            return Ok(Response::new(HeartbeatResponse {
                ack: false,
                should_continue: false,
                message: "No active assignment found for this job".to_string(),
            }));
        };

        // Critical: Check power status
        let power_status = req
            .power_status
            .ok_or_else(|| Status::invalid_argument("power_status is required"))?;

        if !power_status.on_ac_power {
            warn!(
                device_id = %req.device_id,
                job_id = %req.job_id,
                "Device on battery during heartbeat - must stop work"
            );

            // Mark attempt as preempted in database
            if let Err(e) = self.mark_attempt_preempted(&attempt_id).await {
                warn!(error = %e, "Failed to mark attempt as preempted");
            }

            return Ok(Response::new(HeartbeatResponse {
                ack: true,
                should_continue: false,
                message: "Battery detected - stop work immediately".to_string(),
            }));
        }

        // Record heartbeat

        match self.heartbeat_manager.heartbeat(&attempt_id).await {
            Ok(true) => Ok(Response::new(HeartbeatResponse {
                ack: true,
                should_continue: true,
                message: "Heartbeat acknowledged".to_string(),
            })),
            Ok(false) => Ok(Response::new(HeartbeatResponse {
                ack: false,
                should_continue: false,
                message: "Lease expired or unknown attempt".to_string(),
            })),
            Err(e) => {
                warn!(error = %e, "Failed to process heartbeat");
                Err(Status::internal("Failed to process heartbeat"))
            }
        }
    }

    async fn submit_result(
        &self,
        request: Request<SubmitResultRequest>,
    ) -> Result<Response<SubmitResultResponse>, Status> {
        let req = request.into_inner();

        info!(
            device_id = %req.device_id,
            job_id = %req.job_id,
            shard_id = %req.shard_id,
            result_s3_key = %req.result_s3_key,
            "Result submitted"
        );

        // 1. Download and verify result from S3
        let result_data = self.download_result_from_s3(&req.result_s3_key).await?;

        // 2. Verify result hash matches what agent claimed
        let result_hash = self.hash_result(&result_data);
        if result_hash != req.result_hash {
            return Err(Status::invalid_argument(format!(
                "Result hash mismatch: expected {:?}, got {:?}",
                req.result_hash, result_hash
            )));
        }

        // Verify result signature
        self.verify_result_signature(&req.device_id, &result_data, &req.signature).await?;

        // 3. Store result in database
        self.store_result(
            &req.job_id,
            &req.shard_id,
            &req.device_id,
            &req.result_s3_key,
            &result_hash,
        ).await?;

        // 4. Get all results for this shard
        let shard_results = self.get_shard_results(&req.job_id, &req.shard_id).await?;

        // 5. Check for quorum with reputation-based verification
        // Get quorum from job spec or use default of 2
        let job_spec: Result<corpgrid_common::JobSpec, _> = sqlx::query_scalar::<_, String>(
            r#"SELECT spec_json FROM job_shards WHERE job_id = $1 AND shard_id = $2 LIMIT 1"#
        )
        .bind(&req.job_id)
        .bind(&req.shard_id)
        .fetch_one(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Database error: {}", e)))
        .and_then(|json| serde_json::from_str(&json).map_err(|e| Status::internal(format!("JSON error: {}", e))));

        let base_quorum = job_spec
            .ok()
            .map(|spec| spec.redundancy.quorum as usize)
            .unwrap_or(2);

        // Fetch reputations for all devices that submitted results
        let device_reputations = self.get_shard_device_reputations(&shard_results).await?;

        // Adjust quorum based on reputation mix
        let required_quorum = self.calculate_required_quorum(base_quorum, &device_reputations);

        let (quorum_reached, consensus_hash) = self.check_quorum_with_reputation(&shard_results, required_quorum, &device_reputations);

        if quorum_reached {
            info!(
                job_id = %req.job_id,
                shard_id = %req.shard_id,
                num_results = shard_results.len(),
                "Quorum reached for shard"
            );

            // 6. Update reputations based on results
            if let Some(ref consensus) = consensus_hash {
                self.update_reputations(&shard_results, consensus).await?;
            } else {
                // This shouldn't happen if quorum_reached is true, but handle defensively
                return Err(Status::internal("Quorum reached but no consensus hash found"));
            }

            // 7. Mark shard as complete
            self.mark_shard_complete(&req.job_id, &req.shard_id).await?;

            Ok(Response::new(SubmitResultResponse {
                accepted: true,
                quorum_reached: true,
                message: "Result accepted, quorum reached".to_string(),
            }))
        } else {
            Ok(Response::new(SubmitResultResponse {
                accepted: true,
                quorum_reached: false,
                message: format!("Result accepted, awaiting quorum ({}/{})", shard_results.len(), required_quorum),
            }))
        }
    }

    async fn report_checkpoint(
        &self,
        request: Request<ReportCheckpointRequest>,
    ) -> Result<Response<ReportCheckpointResponse>, Status> {
        let req = request.into_inner();

        info!(
            device_id = %req.device_id,
            job_id = %req.job_id,
            shard_id = %req.shard_id,
            checkpoint_s3_key = %req.checkpoint_s3_key,
            "Checkpoint reported"
        );

        // Store checkpoint metadata in database
        self.store_checkpoint(
            &req.device_id,
            &req.job_id,
            &req.shard_id,
            &req.checkpoint_s3_key,
        ).await?;

        info!("Checkpoint stored successfully");

        Ok(Response::new(ReportCheckpointResponse { ack: true }))
    }
}

// Database helper methods implementation
impl SchedulerService {
    async fn get_pending_jobs(&self) -> Result<Vec<PendingJob>, Status> {
        let result = sqlx::query_as::<_, PendingJob>(
            r#"
            SELECT job_id, shard_id, bundle_s3_key, bundle_signature, spec_json
            FROM job_shards
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 10
            "#
        )
        .fetch_all(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        Ok(result)
    }

    async fn mark_job_assigned(
        &self,
        job_id: &str,
        shard_id: &str,
        attempt_id: &str,
        device_id: &str,
    ) -> Result<(), Status> {
        sqlx::query(
            r#"
            UPDATE job_shards
            SET status = 'assigned', assigned_device_id = $1, attempt_id = $2, assigned_at = datetime('now')
            WHERE job_id = $3 AND shard_id = $4
            "#
        )
        .bind(device_id)
        .bind(attempt_id)
        .bind(job_id)
        .bind(shard_id)
        .execute(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Failed to mark job assigned: {}", e)))?;

        Ok(())
    }

    async fn mark_attempt_preempted(&self, attempt_id: &str) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            UPDATE job_shards
            SET status = 'preempted'
            WHERE attempt_id = $1
            "#
        )
        .bind(attempt_id)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    async fn download_result_from_s3(&self, s3_key: &str) -> Result<Vec<u8>, Status> {
        use aws_config::BehaviorVersion;
        use aws_sdk_s3::Client;

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        let bucket = std::env::var("S3_BUCKET")
            .unwrap_or_else(|_| "corpgrid".to_string());

        let resp = client
            .get_object()
            .bucket(bucket)
            .key(s3_key)
            .send()
            .await
            .map_err(|e| Status::internal(format!("S3 error: {}", e)))?;

        let data = resp.body.collect().await
            .map_err(|e| Status::internal(format!("Failed to read S3 body: {}", e)))?
            .into_bytes()
            .to_vec();

        Ok(data)
    }

    fn hash_result(&self, data: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    async fn store_result(
        &self,
        job_id: &str,
        shard_id: &str,
        device_id: &str,
        result_s3_key: &str,
        result_hash: &[u8],
    ) -> Result<(), Status> {
        sqlx::query(
            r#"
            INSERT INTO shard_results (job_id, shard_id, device_id, result_s3_key, result_hash, submitted_at)
            VALUES ($1, $2, $3, $4, $5, datetime('now'))
            "#
        )
        .bind(job_id)
        .bind(shard_id)
        .bind(device_id)
        .bind(result_s3_key)
        .bind(result_hash)
        .execute(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Failed to store result: {}", e)))?;

        Ok(())
    }

    async fn get_shard_results(&self, job_id: &str, shard_id: &str) -> Result<Vec<ShardResult>, Status> {
        let results = sqlx::query_as::<_, ShardResult>(
            r#"
            SELECT device_id, result_hash, submitted_at
            FROM shard_results
            WHERE job_id = $1 AND shard_id = $2
            "#
        )
        .bind(job_id)
        .bind(shard_id)
        .fetch_all(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        Ok(results)
    }

    async fn get_shard_device_reputations(&self, results: &[ShardResult]) -> Result<std::collections::HashMap<String, corpgrid_common::DeviceReputation>, Status> {
        use std::collections::HashMap;
        let mut reputations = HashMap::new();

        for result in results {
            if !reputations.contains_key(&result.device_id) {
                let rep = self.get_device_reputation(&result.device_id).await?;
                reputations.insert(result.device_id.clone(), rep);
            }
        }

        Ok(reputations)
    }

    fn calculate_required_quorum(&self, base_quorum: usize, reputations: &std::collections::HashMap<String, corpgrid_common::DeviceReputation>) -> usize {
        use corpgrid_common::ReputationTier;

        // Count devices by reputation tier
        let mut unproven_count = 0;
        let mut bad_poor_count = 0;
        let mut trusted_count = 0; // Good or Excellent

        for rep in reputations.values() {
            match rep.tier() {
                ReputationTier::Unproven => unproven_count += 1,
                ReputationTier::Bad | ReputationTier::Poor => bad_poor_count += 1,
                ReputationTier::Fair | ReputationTier::Good | ReputationTier::Excellent => trusted_count += 1,
            }
        }

        // If all devices are trusted, use base quorum
        if unproven_count == 0 && bad_poor_count == 0 {
            return base_quorum;
        }

        // If we have untrusted devices, require higher quorum
        // Strategy: Require at least one trusted device to agree, OR require more untrusted devices
        if trusted_count >= 1 {
            // We have at least one trusted device, so base quorum is sufficient
            base_quorum
        } else {
            // All devices are untrusted, require higher quorum
            base_quorum + 1
        }
    }

    fn check_quorum_with_reputation(
        &self,
        results: &[ShardResult],
        required: usize,
        reputations: &std::collections::HashMap<String, corpgrid_common::DeviceReputation>
    ) -> (bool, Option<Vec<u8>>) {
        use std::collections::HashMap;
        use corpgrid_common::ReputationTier;

        // Count results by hash, weighted by reputation
        let mut hash_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut hash_has_trusted: HashMap<Vec<u8>, bool> = HashMap::new();

        for result in results {
            let count = hash_counts.entry(result.result_hash.clone()).or_insert(0);
            *count += 1;

            // Track if this hash has been submitted by a trusted device
            if let Some(rep) = reputations.get(&result.device_id) {
                if matches!(rep.tier(), ReputationTier::Good | ReputationTier::Excellent | ReputationTier::Fair) {
                    *hash_has_trusted.entry(result.result_hash.clone()).or_insert(false) = true;
                }
            }
        }

        // Find hash with most votes
        let max_count = hash_counts.values().max().copied().unwrap_or(0);

        // Collect all hashes with max_count votes
        let mut candidates: Vec<Vec<u8>> = hash_counts
            .iter()
            .filter(|(_, &count)| count == max_count)
            .map(|(hash, _)| hash.clone())
            .collect();

        // Prefer hashes verified by trusted devices
        candidates.sort_by(|a, b| {
            let a_trusted = hash_has_trusted.get(a).copied().unwrap_or(false);
            let b_trusted = hash_has_trusted.get(b).copied().unwrap_or(false);
            b_trusted.cmp(&a_trusted).then_with(|| a.cmp(b)) // Trusted first, then lexicographic
        });

        let consensus_hash = candidates.into_iter().next();

        (max_count >= required, consensus_hash)
    }

    async fn update_reputations(&self, results: &[ShardResult], consensus_hash: &[u8]) -> Result<(), Status> {
        // Update reputation for each device based on whether they matched consensus
        for result in results {
            let is_correct = result.result_hash == consensus_hash;

            sqlx::query(
                r#"
                UPDATE device_reputation
                SET
                    alpha = alpha + $1,
                    beta = beta + $2,
                    last_updated = datetime('now')
                WHERE device_id = $3
                "#
            )
            .bind(if is_correct { 1.0 } else { 0.0 })
            .bind(if is_correct { 0.0 } else { 1.0 })
            .bind(&result.device_id)
            .execute(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update reputation: {}", e)))?;
        }

        Ok(())
    }

    async fn mark_shard_complete(&self, job_id: &str, shard_id: &str) -> Result<(), Status> {
        sqlx::query(
            r#"
            UPDATE job_shards
            SET status = 'complete', completed_at = datetime('now')
            WHERE job_id = $1 AND shard_id = $2
            "#
        )
        .bind(job_id)
        .bind(shard_id)
        .execute(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Failed to mark shard complete: {}", e)))?;

        Ok(())
    }

    async fn store_checkpoint(
        &self,
        device_id: &str,
        job_id: &str,
        shard_id: &str,
        checkpoint_s3_key: &str,
    ) -> Result<(), Status> {
        sqlx::query(
            r#"
            INSERT INTO checkpoints (device_id, job_id, shard_id, checkpoint_s3_key, created_at)
            VALUES ($1, $2, $3, $4, datetime('now'))
            "#
        )
        .bind(device_id)
        .bind(job_id)
        .bind(shard_id)
        .bind(checkpoint_s3_key)
        .execute(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Failed to store checkpoint: {}", e)))?;

        Ok(())
    }

    async fn get_latest_checkpoint(&self, job_id: &str, shard_id: &str) -> Option<String> {
        sqlx::query_scalar::<_, String>(
            r#"
            SELECT checkpoint_s3_key
            FROM checkpoints
            WHERE job_id = $1 AND shard_id = $2
            ORDER BY created_at DESC
            LIMIT 1
            "#
        )
        .bind(job_id)
        .bind(shard_id)
        .fetch_optional(&self.db)
        .await
        .ok()
        .flatten()
    }

    async fn store_agent_public_key(&self, device_id: &str, public_key: &[u8]) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO devices (device_id, public_key, registered_at)
            VALUES ($1, $2, datetime('now'))
            ON CONFLICT (device_id) DO UPDATE
            SET public_key = $2, last_seen = datetime('now')
            "#
        )
        .bind(device_id)
        .bind(public_key)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    async fn initialize_agent_reputation(&self, device_id: &str) -> anyhow::Result<()> {
        // Initialize with uniform prior (alpha=1, beta=1)
        // This gives a score of 0.5 and tier of "Unproven" (< 10 samples)
        // Unproven agents will require higher redundancy checks
        sqlx::query(
            r#"
            INSERT INTO device_reputation (device_id, alpha, beta, last_updated)
            VALUES ($1, 1.0, 1.0, datetime('now'))
            ON CONFLICT (device_id) DO NOTHING
            "#
        )
        .bind(device_id)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    async fn get_device_reputation(&self, device_id: &str) -> Result<corpgrid_common::DeviceReputation, Status> {
        let result: Option<(f64, f64)> = sqlx::query_as(
            r#"
            SELECT alpha, beta
            FROM device_reputation
            WHERE device_id = $1
            "#
        )
        .bind(device_id)
        .fetch_optional(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Failed to fetch reputation: {}", e)))?;

        if let Some((alpha, beta)) = result {
            Ok(corpgrid_common::DeviceReputation::with_prior(alpha, beta))
        } else {
            // If no reputation exists, return default (Unproven)
            Ok(corpgrid_common::DeviceReputation::default())
        }
    }

    async fn calculate_bundle_hash(&self, s3_key: &str) -> Option<Vec<u8>> {
        use aws_config::BehaviorVersion;
        use aws_sdk_s3::Client;
        use sha2::{Sha256, Digest};

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        let bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "corpgrid".to_string());

        let resp = client
            .get_object()
            .bucket(bucket)
            .key(s3_key)
            .send()
            .await
            .ok()?;

        let data = resp.body.collect().await.ok()?.into_bytes().to_vec();

        let mut hasher = Sha256::new();
        hasher.update(&data);
        Some(hasher.finalize().to_vec())
    }

    async fn verify_result_signature(
        &self,
        device_id: &str,
        result_data: &[u8],
        signature: &[u8],
    ) -> Result<(), Status> {
        use ed25519_dalek::{Verifier, VerifyingKey, Signature};

        // Get device's public key from database
        let public_key_bytes = sqlx::query_scalar::<_, Vec<u8>>(
            r#"
            SELECT public_key
            FROM devices
            WHERE device_id = $1
            "#
        )
        .bind(device_id)
        .fetch_optional(&self.db)
        .await
        .map_err(|e| Status::internal(format!("Database error: {}", e)))?
        .ok_or_else(|| Status::not_found(format!("Device {} not found", device_id)))?;

        // Parse public key
        let public_key_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| Status::internal("Invalid public key length"))?;
        let public_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|e| Status::internal(format!("Invalid public key: {}", e)))?;

        // Parse signature
        let signature_array: [u8; 64] = signature
            .try_into()
            .map_err(|_| Status::invalid_argument("Invalid signature length"))?;
        let signature = Signature::from_bytes(&signature_array);

        // Verify signature
        public_key
            .verify(result_data, &signature)
            .map_err(|e| Status::permission_denied(format!("Signature verification failed: {}", e)))?;

        Ok(())
    }

    /// Reassign expired job attempts
    pub async fn reassign_expired_attempts(&self, expired_attempt_ids: Vec<String>) -> anyhow::Result<()> {
        for attempt_id in expired_attempt_ids {
            info!(attempt_id = %attempt_id, "Reassigning expired attempt");

            // Mark the attempt as expired and reset the shard to pending status
            let result = sqlx::query(
                r#"
                UPDATE job_shards
                SET status = 'pending', assigned_device_id = NULL, attempt_id = NULL
                WHERE attempt_id = $1 AND status IN ('assigned', 'running')
                RETURNING job_id, shard_id
                "#
            )
            .bind(&attempt_id)
            .fetch_optional(&self.db)
            .await?;

            if let Some(row) = result {
                let job_id: String = row.try_get("job_id")?;
                let shard_id: String = row.try_get("shard_id")?;

                info!(
                    job_id = %job_id,
                    shard_id = %shard_id,
                    attempt_id = %attempt_id,
                    "Expired attempt reassigned to pending"
                );

                // Record the expired attempt in history
                sqlx::query(
                    r#"
                    INSERT INTO attempt_history (job_id, shard_id, attempt_id, status, completed_at)
                    VALUES ($1, $2, $3, 'expired', datetime('now'))
                    "#
                )
                .bind(&job_id)
                .bind(&shard_id)
                .bind(&attempt_id)
                .execute(&self.db)
                .await
                .ok(); // Ignore errors for history table
            }
        }

        Ok(())
    }
}

// Helper structs for database queries
#[derive(sqlx::FromRow)]
struct PendingJob {
    job_id: String,
    shard_id: String,
    bundle_s3_key: String,
    bundle_signature: Vec<u8>,
    spec_json: String,
}

impl PendingJob {
    fn spec(&self) -> Result<corpgrid_common::JobSpec, serde_json::Error> {
        serde_json::from_str(&self.spec_json)
    }
}

#[allow(dead_code)]
#[derive(sqlx::FromRow)]
struct ShardResult {
    device_id: String,
    result_hash: Vec<u8>,
    submitted_at: String, // TEXT timestamp for SQLite compatibility
}
