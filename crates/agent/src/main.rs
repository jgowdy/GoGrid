use anyhow::{Context, Result};
use corpgrid_agent::{AgentClient, JobExecutor};
use corpgrid_common::PowerMonitor;
use corpgrid_proto::{JobAssignment, JobProgress};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Execute a single job assignment with heartbeats and result submission
async fn execute_job(
    executor: Arc<JobExecutor>,
    power_status: Arc<RwLock<corpgrid_common::PowerStatus>>,
    mut client: AgentClient,
    assignment: JobAssignment,
) -> Result<()> {
    let job_id = assignment.job_id.clone();
    let shard_id = assignment.shard_id.clone();

    info!(
        job_id = %job_id,
        shard_id = %shard_id,
        "Starting job execution"
    );

    // Shared state for heartbeat task
    let should_stop = Arc::new(RwLock::new(false));
    let should_stop_clone = should_stop.clone();
    let power_status_clone = power_status.clone();
    let job_id_clone = job_id.clone();
    let shard_id_clone = shard_id.clone();

    // Spawn heartbeat task
    let heartbeat_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        let mut client = match AgentClient::connect(
            std::env::var("SCHEDULER_URL")
                .unwrap_or_else(|_| "http://localhost:50051".to_string())
        ).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "Failed to create heartbeat client");
                return;
            }
        };

        loop {
            interval.tick().await;

            // Check if we should stop
            if *should_stop_clone.read().await {
                info!("Heartbeat task stopping");
                break;
            }

            let status = *power_status_clone.read().await;

            // Send heartbeat
            let progress = JobProgress {
                percent_complete: 0.5, // TODO: Track actual progress
                processed_items: 0,
                total_items: 0,
            };

            match client
                .send_heartbeat(
                    job_id_clone.clone(),
                    shard_id_clone.clone(),
                    status,
                    progress,
                )
                .await
            {
                Ok(response) => {
                    if !response.should_continue {
                        warn!(
                            message = %response.message,
                            "Scheduler requested job stop"
                        );
                        *should_stop_clone.write().await = true;
                        break;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Heartbeat failed");
                }
            }
        }
    });

    // Execute the job
    let result = match executor.execute(&assignment).await {
        Ok(r) => {
            info!(result_size = r.len(), "Job execution completed");
            r
        }
        Err(e) => {
            error!(error = %e, "Job execution failed");
            *should_stop.write().await = true;
            heartbeat_handle.abort();
            return Err(e);
        }
    };

    // Check if we were told to stop
    if *should_stop.read().await {
        warn!("Job execution cancelled by scheduler or battery detection");
        heartbeat_handle.abort();
        return Ok(());
    }

    // Upload result to S3
    info!("Uploading result to S3");
    let result_s3_key = format!("results/{}/{}/result.bin", job_id, shard_id);

    match upload_to_s3(&result_s3_key, &result).await {
        Ok(_) => info!(s3_key = %result_s3_key, "Result uploaded to S3"),
        Err(e) => {
            error!(error = %e, "Failed to upload result to S3");
            *should_stop.write().await = true;
            heartbeat_handle.abort();
            return Err(e);
        }
    }

    // Calculate result hash
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(&result);
    let result_hash = hasher.finalize().to_vec();

    // Sign the result with agent's private key
    let signature = sign_result(&result)?;

    // Submit result to scheduler
    info!("Submitting result to scheduler");
    match client
        .submit_result(
            job_id.clone(),
            shard_id.clone(),
            result_hash,
            result_s3_key,
            signature,
        )
        .await
    {
        Ok(response) => {
            info!(
                accepted = response.accepted,
                message = %response.message,
                "Result submitted"
            );
        }
        Err(e) => {
            error!(error = %e, "Failed to submit result");
            *should_stop.write().await = true;
            heartbeat_handle.abort();
            return Err(e);
        }
    }

    // Stop heartbeat task
    *should_stop.write().await = true;
    heartbeat_handle.abort();

    info!(
        job_id = %job_id,
        shard_id = %shard_id,
        "Job completed successfully"
    );

    Ok(())
}

/// Load or generate Ed25519 keypair for this agent
fn load_or_generate_keypair() -> Result<ed25519_dalek::SigningKey> {
    use ed25519_dalek::SigningKey;

    let key_path = std::env::var("AGENT_KEY_PATH")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/.corpgrid/agent_key", home)
        });

    // Try to load existing key
    if let Ok(key_bytes) = std::fs::read(&key_path) {
        if key_bytes.len() == 32 {
            let key_array: [u8; 32] = key_bytes.try_into().unwrap();
            info!("Loaded existing agent key from {}", key_path);
            return Ok(SigningKey::from_bytes(&key_array));
        }
    }

    // Generate new keypair
    use rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut OsRng);

    // Save private key
    if let Some(parent) = std::path::Path::new(&key_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&key_path, signing_key.to_bytes())?;

    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&key_path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&key_path, perms)?;
    }

    info!("Generated new agent key at {}", key_path);

    Ok(signing_key)
}

/// Sign result data with agent's private key
fn sign_result(data: &[u8]) -> Result<Vec<u8>> {
    use ed25519_dalek::Signer;

    let signing_key = load_or_generate_keypair()?;
    let signature = signing_key.sign(data);

    Ok(signature.to_bytes().to_vec())
}

/// Upload data to S3
async fn upload_to_s3(s3_key: &str, data: &[u8]) -> Result<()> {
    use aws_config::BehaviorVersion;
    use aws_sdk_s3::Client;

    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let client = Client::new(&config);

    let bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "corpgrid".to_string());

    client
        .put_object()
        .bucket(bucket)
        .key(s3_key)
        .body(data.to_vec().into())
        .send()
        .await
        .context("S3 PutObject failed")?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corpgrid_agent=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("CorpGrid Agent starting...");

    // Load configuration
    let scheduler_url = std::env::var("SCHEDULER_URL")
        .unwrap_or_else(|_| "http://localhost:50051".to_string());
    let work_dir = std::env::var("WORK_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("corpgrid"));

    // Create work directory
    tokio::fs::create_dir_all(&work_dir).await?;

    // Initialize power monitoring
    let power_status = Arc::new(RwLock::new(PowerMonitor::get_status()?));
    info!(
        on_ac_power = power_status.read().await.on_ac_power,
        battery_percent = power_status.read().await.battery_percent,
        "Initial power status"
    );

    // Start power monitoring
    let power_status_clone = power_status.clone();
    tokio::spawn(async move {
        if let Err(e) = PowerMonitor::monitor(move |status| {
            let power_status = power_status_clone.clone();
            tokio::spawn(async move {
                *power_status.write().await = status;

                if !status.on_ac_power {
                    error!("BATTERY DETECTED - All work must stop immediately!");
                    // TODO: Checkpoint and stop all running jobs
                }
            });
        })
        .await
        {
            error!(error = %e, "Power monitoring failed");
        }
    });

    // Connect to scheduler
    let mut client = AgentClient::connect(scheduler_url).await?;

    // Load keypair and get public key
    let signing_key = load_or_generate_keypair()?;
    let public_key = signing_key.verifying_key().to_bytes().to_vec();

    // Register with scheduler
    let status = *power_status.read().await;
    if !client.register(status, public_key).await? {
        error!("Agent registration rejected by scheduler");
        return Ok(());
    }

    info!("Agent registered successfully");

    // Initialize job executor
    let executor = Arc::new(JobExecutor::new(work_dir));

    // Main event loop
    let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

    loop {
        tokio::select! {
            _ = poll_interval.tick() => {
                // Check power status
                let status = *power_status.read().await;

                if !status.allows_execution() {
                    warn!("On battery power - skipping job polling");
                    continue;
                }

                // Poll for jobs
                match client.poll_jobs(status).await {
                    Ok(assignments) => {
                        if !assignments.is_empty() {
                            info!(count = assignments.len(), "Received job assignments");

                            for assignment in assignments {
                                info!(
                                    job_id = %assignment.job_id,
                                    shard_id = %assignment.shard_id,
                                    "Processing assignment"
                                );

                                // Spawn job execution task
                                let executor = executor.clone();
                                let power_status = power_status.clone();
                                let job_client = match AgentClient::connect(
                                    std::env::var("SCHEDULER_URL")
                                        .unwrap_or_else(|_| "http://localhost:50051".to_string())
                                ).await {
                                    Ok(c) => c,
                                    Err(e) => {
                                        error!(error = %e, "Failed to create job client");
                                        continue;
                                    }
                                };

                                tokio::spawn(async move {
                                    if let Err(e) = execute_job(
                                        executor,
                                        power_status,
                                        job_client,
                                        assignment,
                                    ).await {
                                        error!(error = %e, "Job execution failed");
                                    }
                                });
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to poll jobs");
                    }
                }
            }

            _ = signal::ctrl_c() => {
                info!("Shutdown signal received");
                break;
            }
        }
    }

    info!("Agent shutdown complete");
    Ok(())
}
