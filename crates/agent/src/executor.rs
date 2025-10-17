use anyhow::{Context, Result};
use corpgrid_proto::JobAssignment;
use std::process::Stdio;
use tracing::{debug, info, warn};

/// Job executor - runs jobs in sandboxed environment
pub struct JobExecutor {
    work_dir: std::path::PathBuf,
}

impl JobExecutor {
    pub fn new(work_dir: std::path::PathBuf) -> Self {
        Self { work_dir }
    }

    /// Execute a job assignment
    pub async fn execute(&self, assignment: &JobAssignment) -> Result<Vec<u8>> {
        info!(
            job_id = %assignment.job_id,
            shard_id = %assignment.shard_id,
            bundle_s3_key = %assignment.bundle_s3_key,
            "Starting job execution"
        );

        // Create job-specific work directory
        let job_dir = self.work_dir.join(&assignment.job_id).join(&assignment.shard_id);
        tokio::fs::create_dir_all(&job_dir).await?;

        // 1. Download bundle from S3
        let bundle_path = job_dir.join("bundle.tar.gz");
        info!(bundle_path = %bundle_path.display(), "Downloading bundle from S3");

        self.download_from_s3(&assignment.bundle_s3_key, &bundle_path).await
            .context("Failed to download bundle from S3")?;

        info!("Bundle downloaded successfully");

        // 2. Extract bundle
        let extract_dir = job_dir.join("extracted");
        tokio::fs::create_dir_all(&extract_dir).await?;

        info!(extract_dir = %extract_dir.display(), "Extracting bundle");
        self.extract_bundle(&bundle_path, &extract_dir).await
            .context("Failed to extract bundle")?;

        info!("Bundle extracted successfully");

        // 3. Verify bundle signature
        info!("Verifying bundle signature");
        self.verify_bundle_signature(&extract_dir, &assignment.bundle_signature)
            .await
            .context("Bundle signature verification failed")?;

        info!("Bundle signature verified");

        // 4. Find and run the job runner binary
        let runner_path = extract_dir.join("runner");
        if !runner_path.exists() {
            anyhow::bail!("Runner binary not found in bundle at {:?}", runner_path);
        }

        // Make runner executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&runner_path).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&runner_path, perms).await?;
        }

        info!(runner = %runner_path.display(), "Running job in sandbox");

        // 5. Run in sandbox with job parameters
        let args = vec![
            "--job-id".to_string(),
            assignment.job_id.clone(),
            "--shard-id".to_string(),
            assignment.shard_id.to_string(),
            "--work-dir".to_string(),
            job_dir.to_str().unwrap().to_string(),
        ];

        self.run_sandboxed(&runner_path, &job_dir, &args).await
            .context("Job execution failed")?;

        // 6. Read result file
        let result_path = job_dir.join("result.bin");
        if !result_path.exists() {
            anyhow::bail!("Result file not found at {:?}", result_path);
        }

        let result = tokio::fs::read(&result_path).await
            .context("Failed to read result file")?;

        info!(result_size = result.len(), "Job completed successfully");

        Ok(result)
    }

    async fn download_from_s3(&self, s3_key: &str, local_path: &std::path::Path) -> Result<()> {
        use aws_config::BehaviorVersion;
        use aws_sdk_s3::Client;

        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);

        let bucket = std::env::var("S3_BUCKET")
            .unwrap_or_else(|_| "corpgrid".to_string());

        debug!(bucket = %bucket, key = %s3_key, "Downloading from S3");

        let resp = client
            .get_object()
            .bucket(bucket)
            .key(s3_key)
            .send()
            .await
            .context("S3 GetObject failed")?;

        let data = resp.body.collect().await
            .context("Failed to read S3 object body")?
            .into_bytes();

        tokio::fs::write(local_path, data).await
            .context("Failed to write downloaded bundle")?;

        Ok(())
    }

    async fn extract_bundle(
        &self,
        bundle_path: &std::path::Path,
        extract_dir: &std::path::Path,
    ) -> Result<()> {
        use tokio::process::Command;

        let status = Command::new("tar")
            .arg("-xzf")
            .arg(bundle_path)
            .arg("-C")
            .arg(extract_dir)
            .arg("--no-absolute-names") // Prevent absolute path extraction
            .status()
            .await
            .context("Failed to run tar command")?;

        if !status.success() {
            anyhow::bail!("tar extraction failed with status: {:?}", status);
        }

        // Verify no path traversal occurred
        let mut entries = tokio::fs::read_dir(extract_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.starts_with(extract_dir) {
                anyhow::bail!("Path traversal detected: {:?}", path);
            }
        }

        Ok(())
    }

    async fn verify_bundle_signature(
        &self,
        extract_dir: &std::path::Path,
        expected_signature: &[u8],
    ) -> Result<()> {
        use sha2::{Sha256, Digest};

        // Calculate SHA256 of all files in bundle
        let mut hasher = Sha256::new();

        // Walk directory and hash all files
        let mut files = Vec::new();
        let mut entries = tokio::fs::read_dir(extract_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }

        // Sort for deterministic hashing
        files.sort();

        for file in files {
            let contents = tokio::fs::read(&file).await?;
            hasher.update(&contents);
        }

        let actual_signature = hasher.finalize();

        if &actual_signature[..] != expected_signature {
            anyhow::bail!(
                "Signature mismatch! Expected: {:?}, Got: {:?}",
                expected_signature,
                &actual_signature[..]
            );
        }

        Ok(())
    }

    /// Run the job runner binary in a sandbox
    async fn run_sandboxed(
        &self,
        runner_path: &std::path::Path,
        job_dir: &std::path::Path,
        args: &[String],
    ) -> Result<()> {
        use crate::sandbox::{create_sandboxed_command, SandboxConfig};

        info!(
            runner = %runner_path.display(),
            job_dir = %job_dir.display(),
            "Running job in sandbox"
        );

        let config = SandboxConfig::default();

        let mut cmd = create_sandboxed_command(runner_path, job_dir, &config)?;

        // Add job arguments after sandbox setup
        cmd.args(args);

        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!(
                exit_code = ?output.status.code(),
                stderr = %stderr,
                "Job runner failed"
            );
            anyhow::bail!("Job runner exited with error");
        }

        info!("Job completed successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        let temp_dir = std::env::temp_dir();
        let executor = JobExecutor::new(temp_dir);
        assert!(executor.work_dir.exists());
    }
}
