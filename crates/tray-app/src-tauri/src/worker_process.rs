use anyhow::{Context, Result};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

pub struct WorkerProcess {
    process: Arc<RwLock<Option<Child>>>,
    worker_binary_path: String,
}

impl WorkerProcess {
    pub fn new(worker_binary_path: String) -> Self {
        Self {
            process: Arc::new(RwLock::new(None)),
            worker_binary_path,
        }
    }

    pub async fn start(&self) -> Result<()> {
        let mut process_lock = self.process.write().await;

        // Check if already running
        if let Some(ref mut child) = *process_lock {
            // Check if process is still alive
            match child.try_wait() {
                Ok(None) => {
                    info!("Worker process already running");
                    return Ok(());
                }
                Ok(Some(status)) => {
                    warn!("Previous worker process exited with status: {}", status);
                }
                Err(e) => {
                    error!("Error checking worker process status: {}", e);
                }
            }
        }

        info!("Starting worker process: {}", self.worker_binary_path);

        // Spawn the worker process
        let child = Command::new(&self.worker_binary_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn worker process")?;

        info!("Worker process started with PID: {}", child.id());

        *process_lock = Some(child);
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        let mut process_lock = self.process.write().await;

        if let Some(mut child) = process_lock.take() {
            info!("Stopping worker process (PID: {})", child.id());

            // Try graceful shutdown first
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;

                let pid = Pid::from_raw(child.id() as i32);
                if let Err(e) = kill(pid, Signal::SIGTERM) {
                    warn!("Failed to send SIGTERM to worker process: {}", e);
                }

                // Wait for process to exit (with timeout)
                let wait_result = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    tokio::task::spawn_blocking(move || child.wait()),
                )
                .await;

                match wait_result {
                    Ok(Ok(Ok(status))) => {
                        info!("Worker process exited gracefully: {}", status);
                        return Ok(());
                    }
                    Ok(Ok(Err(e))) => {
                        error!("Error waiting for worker process: {}", e);
                    }
                    Ok(Err(e)) => {
                        error!("Join error waiting for worker process: {}", e);
                    }
                    Err(_) => {
                        warn!("Worker process did not exit gracefully, forcing kill");
                    }
                }

                // Force kill if still running
                if let Err(e) = kill(pid, Signal::SIGKILL) {
                    error!("Failed to send SIGKILL to worker process: {}", e);
                }
            }

            #[cfg(not(unix))]
            {
                // On Windows, just kill it
                if let Err(e) = child.kill() {
                    error!("Failed to kill worker process: {}", e);
                }
            }

            info!("Worker process stopped");
        } else {
            info!("Worker process not running");
        }

        Ok(())
    }

    pub async fn restart(&self) -> Result<()> {
        info!("Restarting worker process");
        self.stop().await?;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        self.start().await?;
        Ok(())
    }

    pub async fn is_running(&self) -> bool {
        let mut process_lock = self.process.write().await;

        if let Some(ref mut child) = *process_lock {
            match child.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => {
                    // Process has exited
                    *process_lock = None;
                    false
                }
                Err(_) => false,
            }
        } else {
            false
        }
    }

    pub async fn get_pid(&self) -> Option<u32> {
        let process_lock = self.process.read().await;
        process_lock.as_ref().map(|child| child.id())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        // Try to kill the process on drop (best effort)
        if let Some(mut child) = self.process.blocking_write().take() {
            let _ = child.kill();
        }
    }
}
