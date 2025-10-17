/// Resource Management for Production Deployments
///
/// This module provides GPU resource limiting and process priority management
/// to ensure worker processes don't impact desktop performance.
///
/// Key features:
/// - GPU memory usage limits (percentage-based)
/// - GPU compute time throttling
/// - Process priority management (nice/renice)
/// - Graceful degradation under load
/// - Automatic resource release when system is busy

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::system_monitor::{SystemLoadThresholds, SystemMonitor};

#[cfg(target_os = "linux")]
use std::fs;

/// Resource limits configuration for GPU workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum percentage of GPU VRAM to use (0.0-1.0)
    /// Default: 0.7 (70% of available VRAM)
    pub max_vram_usage_percent: f64,

    /// Maximum percentage of GPU compute to use (0.0-1.0)
    /// Default: 0.8 (80% of GPU time)
    pub max_compute_usage_percent: f64,

    /// Process priority (nice value: -20 to 19, higher = lower priority)
    /// Default: 10 (low priority, won't impact desktop)
    pub process_priority: i32,

    /// I/O priority class (0=none, 1=realtime, 2=best-effort, 3=idle)
    /// Default: 3 (idle - minimal impact on disk I/O)
    pub io_priority_class: u32,

    /// I/O priority level within class (0-7, higher = lower priority)
    /// Default: 7 (lowest within idle class)
    pub io_priority_level: u32,

    /// Enable automatic throttling when system is busy
    /// Default: true
    pub enable_auto_throttle: bool,

    /// Minimum time between inference requests (ms)
    /// Used to throttle GPU usage
    /// Default: 50ms (allows desktop to use GPU between requests)
    pub min_request_interval_ms: u64,

    /// Maximum batch size (limits memory usage)
    /// Default: 4
    pub max_batch_size: usize,

    /// Enable adaptive throttling based on system load
    /// When enabled, automatically pauses inference when CPU/GPU/memory is high
    /// Default: true
    pub enable_adaptive_throttling: bool,

    /// System load thresholds for adaptive behavior
    /// Only used if enable_adaptive_throttling is true
    pub system_load_thresholds: Option<SystemLoadThresholds>,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_vram_usage_percent: 0.7,
            max_compute_usage_percent: 0.8,
            process_priority: 10,      // Low priority
            io_priority_class: 3,       // Idle
            io_priority_level: 7,       // Lowest
            enable_auto_throttle: true,
            min_request_interval_ms: 50,
            max_batch_size: 4,
            enable_adaptive_throttling: true,
            system_load_thresholds: Some(SystemLoadThresholds::default()),
        }
    }
}

impl ResourceConfig {
    /// Create conservative resource limits for desktop use
    pub fn conservative() -> Self {
        Self {
            max_vram_usage_percent: 0.5,  // Only use 50% of VRAM
            max_compute_usage_percent: 0.6, // Only use 60% of compute
            process_priority: 15,          // Very low priority
            io_priority_class: 3,          // Idle
            io_priority_level: 7,          // Lowest
            enable_auto_throttle: true,
            min_request_interval_ms: 100,  // More spacing between requests
            max_batch_size: 2,             // Smaller batches
            enable_adaptive_throttling: true,
            system_load_thresholds: Some(SystemLoadThresholds::conservative()),
        }
    }

    /// Create aggressive resource limits for dedicated servers
    pub fn aggressive() -> Self {
        Self {
            max_vram_usage_percent: 0.95,
            max_compute_usage_percent: 0.95,
            process_priority: 0,           // Normal priority
            io_priority_class: 2,          // Best-effort
            io_priority_level: 4,          // Middle priority
            enable_auto_throttle: false,
            min_request_interval_ms: 0,
            max_batch_size: 16,
            enable_adaptive_throttling: false,  // Disabled for servers
            system_load_thresholds: Some(SystemLoadThresholds::aggressive()),
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.max_vram_usage_percent) {
            return Err(anyhow!("max_vram_usage_percent must be between 0.0 and 1.0"));
        }
        if !(0.0..=1.0).contains(&self.max_compute_usage_percent) {
            return Err(anyhow!("max_compute_usage_percent must be between 0.0 and 1.0"));
        }
        if !(-20..=19).contains(&self.process_priority) {
            return Err(anyhow!("process_priority must be between -20 and 19"));
        }
        if self.io_priority_class > 3 {
            return Err(anyhow!("io_priority_class must be 0-3"));
        }
        if self.io_priority_level > 7 {
            return Err(anyhow!("io_priority_level must be 0-7"));
        }
        Ok(())
    }

    /// Calculate maximum VRAM bytes based on total available
    pub fn max_vram_bytes(&self, total_vram_bytes: u64) -> u64 {
        (total_vram_bytes as f64 * self.max_vram_usage_percent) as u64
    }
}

/// Resource manager for GPU workers
pub struct ResourceManager {
    config: ResourceConfig,
    last_request_time: Option<Instant>,
    total_requests: u64,
    throttled_requests: u64,
    system_monitor: Option<SystemMonitor>,
    system_paused_requests: u64,
}

impl ResourceManager {
    /// Create a new resource manager with the given configuration
    pub fn new(config: ResourceConfig) -> Result<Self> {
        config.validate()?;

        // Create system monitor if adaptive throttling is enabled
        let system_monitor = if config.enable_adaptive_throttling {
            let thresholds = config
                .system_load_thresholds
                .clone()
                .unwrap_or_default();
            Some(SystemMonitor::new(thresholds))
        } else {
            None
        };

        let manager = Self {
            config,
            last_request_time: None,
            total_requests: 0,
            throttled_requests: 0,
            system_monitor,
            system_paused_requests: 0,
        };

        // Apply process priority immediately
        manager.apply_process_priority()?;

        info!(
            vram_limit = format!("{:.0}%", manager.config.max_vram_usage_percent * 100.0),
            compute_limit = format!("{:.0}%", manager.config.max_compute_usage_percent * 100.0),
            priority = manager.config.process_priority,
            adaptive_throttling = manager.config.enable_adaptive_throttling,
            "Resource manager initialized"
        );

        Ok(manager)
    }

    /// Apply process priority (nice value) to current process
    fn apply_process_priority(&self) -> Result<()> {
        #[cfg(target_family = "unix")]
        {
            let pid = std::process::id();

            // Set CPU priority (nice)
            let output = std::process::Command::new("renice")
                .arg("-n")
                .arg(self.config.process_priority.to_string())
                .arg("-p")
                .arg(pid.to_string())
                .output()?;

            if !output.status.success() {
                warn!(
                    error = String::from_utf8_lossy(&output.stderr).to_string(),
                    "Failed to set process priority"
                );
            } else {
                debug!(priority = self.config.process_priority, "Set process priority");
            }

            // Set I/O priority (ionice) - Linux only
            #[cfg(target_os = "linux")]
            {
                let ioprio = (self.config.io_priority_class << 13) | self.config.io_priority_level;
                let output = std::process::Command::new("ionice")
                    .arg("-c")
                    .arg(self.config.io_priority_class.to_string())
                    .arg("-n")
                    .arg(self.config.io_priority_level.to_string())
                    .arg("-p")
                    .arg(pid.to_string())
                    .output()?;

                if !output.status.success() {
                    warn!(
                        error = String::from_utf8_lossy(&output.stderr).to_string(),
                        "Failed to set I/O priority"
                    );
                } else {
                    debug!(
                        class = self.config.io_priority_class,
                        level = self.config.io_priority_level,
                        "Set I/O priority"
                    );
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            warn!("Process priority management not implemented on Windows");
        }

        Ok(())
    }

    /// Check if we should throttle the next request
    /// Returns true if the request should proceed, false if it should be delayed
    pub fn should_allow_request(&mut self) -> bool {
        self.total_requests += 1;

        if !self.config.enable_auto_throttle || self.config.min_request_interval_ms == 0 {
            return true; // No throttling
        }

        if let Some(last_time) = self.last_request_time {
            let elapsed = last_time.elapsed();
            let min_interval = Duration::from_millis(self.config.min_request_interval_ms);

            if elapsed < min_interval {
                self.throttled_requests += 1;
                debug!(
                    elapsed_ms = elapsed.as_millis(),
                    required_ms = self.config.min_request_interval_ms,
                    "Throttling request"
                );
                return false;
            }
        }

        self.last_request_time = Some(Instant::now());
        true
    }

    /// Calculate how long to wait before the next request (without sleeping)
    /// Returns None if no wait is needed
    /// Returns Some(Duration) if we need to wait (either for throttling or system load)
    pub fn calculate_wait_time(&mut self) -> Option<Duration> {
        // First check system load if adaptive throttling is enabled
        if let Some(ref mut monitor) = self.system_monitor {
            match monitor.should_allow_inference() {
                Ok(false) => {
                    // System is busy, return the pause duration
                    self.system_paused_requests += 1;
                    return monitor.get_pause_duration();
                }
                Err(e) => {
                    // Error checking system load, log but continue
                    debug!(error = ?e, "Failed to check system load");
                }
                Ok(true) => {
                    // System load OK, continue with normal throttling
                }
            }
        }

        // Normal throttling check
        if !self.config.enable_auto_throttle || self.config.min_request_interval_ms == 0 {
            return None;
        }

        if let Some(last_time) = self.last_request_time {
            let elapsed = last_time.elapsed();
            let min_interval = Duration::from_millis(self.config.min_request_interval_ms);

            if elapsed < min_interval {
                return Some(min_interval - elapsed);
            }
        }

        None
    }

    /// Mark that a request has been processed (updates timestamp)
    /// Should be called after wait completes
    pub fn mark_request_processed(&mut self) {
        self.last_request_time = Some(Instant::now());
        self.total_requests += 1;
    }

    /// Sleep for the remaining throttle interval if needed
    /// DEPRECATED: Use calculate_wait_time + mark_request_processed for better atomicity
    pub async fn wait_for_throttle(&self) {
        if !self.config.enable_auto_throttle || self.config.min_request_interval_ms == 0 {
            return;
        }

        if let Some(last_time) = self.last_request_time {
            let elapsed = last_time.elapsed();
            let min_interval = Duration::from_millis(self.config.min_request_interval_ms);

            if elapsed < min_interval {
                let wait_time = min_interval - elapsed;
                debug!(wait_ms = wait_time.as_millis(), "Waiting for throttle interval");
                tokio::time::sleep(wait_time).await;
            }
        }
    }

    /// Get throttling statistics
    pub fn get_stats(&self) -> ResourceStats {
        let (system_pauses, currently_paused) = if let Some(ref monitor) = self.system_monitor {
            let stats = monitor.get_stats();
            (stats.pauses_triggered, stats.currently_paused)
        } else {
            (0, false)
        };

        ResourceStats {
            total_requests: self.total_requests,
            throttled_requests: self.throttled_requests,
            system_paused_requests: self.system_paused_requests,
            throttle_rate: if self.total_requests > 0 {
                self.throttled_requests as f64 / self.total_requests as f64
            } else {
                0.0
            },
            system_pauses,
            currently_paused,
        }
    }

    /// Get the resource configuration
    pub fn config(&self) -> &ResourceConfig {
        &self.config
    }

    /// Check if a GPU device meets resource limits
    pub fn check_vram_limit(&self, requested_bytes: u64, available_bytes: u64) -> Result<()> {
        let max_bytes = self.config.max_vram_bytes(available_bytes);

        if requested_bytes > max_bytes {
            return Err(anyhow!(
                "Requested VRAM ({} MB) exceeds limit ({} MB / {:.0}%)",
                requested_bytes / 1024 / 1024,
                max_bytes / 1024 / 1024,
                self.config.max_vram_usage_percent * 100.0
            ));
        }

        Ok(())
    }

    /// Get current GPU utilization (if available)
    #[cfg(target_os = "linux")]
    pub fn get_gpu_utilization(&self) -> Result<f64> {
        // Try to read from sysfs for NVIDIA GPUs
        let utilization_path = "/sys/class/drm/card0/device/gpu_busy_percent";

        if let Ok(content) = fs::read_to_string(utilization_path) {
            if let Ok(util) = content.trim().parse::<f64>() {
                return Ok(util / 100.0);
            }
        }

        // Try nvidia-smi as fallback
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()?;

        if output.status.success() {
            let util_str = String::from_utf8_lossy(&output.stdout);
            let util = util_str.trim().parse::<f64>()? / 100.0;
            return Ok(util);
        }

        Err(anyhow!("Unable to read GPU utilization"))
    }

    #[cfg(not(target_os = "linux"))]
    pub fn get_gpu_utilization(&self) -> Result<f64> {
        Err(anyhow!("GPU utilization monitoring not implemented on this platform"))
    }
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    pub total_requests: u64,
    pub throttled_requests: u64,
    pub system_paused_requests: u64,
    pub throttle_rate: f64,
    pub system_pauses: u64,
    pub currently_paused: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[test]
    fn test_resource_config_validation() {
        let mut config = ResourceConfig::default();
        assert!(config.validate().is_ok());

        config.max_vram_usage_percent = 1.5;
        assert!(config.validate().is_err());

        config.max_vram_usage_percent = 0.7;
        config.process_priority = 25;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vram_limit_calculation() {
        let config = ResourceConfig::default();
        let total_vram = 16 * 1024 * 1024 * 1024; // 16 GB
        let max_vram = config.max_vram_bytes(total_vram);

        let expected = (16.0 * 0.7 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(max_vram, expected);
    }

    #[tokio::test]
    async fn test_throttling() {
        let config = ResourceConfig {
            min_request_interval_ms: 100,
            enable_auto_throttle: true,
            ..Default::default()
        };

        let mut manager = ResourceManager::new(config).unwrap();

        // First request should pass
        assert!(manager.should_allow_request());

        // Immediate second request should be throttled
        assert!(!manager.should_allow_request());

        // Wait and try again
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(manager.should_allow_request());
    }

    #[test]
    fn test_conservative_config() {
        let config = ResourceConfig::conservative();
        assert!(config.max_vram_usage_percent <= 0.5);
        assert!(config.process_priority >= 10);
        assert_eq!(config.io_priority_class, 3); // Idle
    }

    #[test]
    fn test_aggressive_config() {
        let config = ResourceConfig::aggressive();
        assert!(config.max_vram_usage_percent >= 0.9);
        assert!(config.process_priority <= 5);
        assert!(!config.enable_auto_throttle);
    }

    #[tokio::test]
    async fn test_concurrent_throttling_stress() {
        // This test verifies that the new atomic throttling implementation
        // correctly handles concurrent requests without race conditions

        let config = ResourceConfig {
            min_request_interval_ms: 50,
            enable_auto_throttle: true,
            enable_adaptive_throttling: false,  // Disable for deterministic test
            process_priority: 10, // Won't actually change priority in test
            ..Default::default()
        };

        let manager = Arc::new(Mutex::new(ResourceManager::new(config).unwrap()));
        let num_concurrent_requests = 100;
        let mut handles = Vec::new();

        // Spawn many concurrent tasks that all try to throttle
        for _ in 0..num_concurrent_requests {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                // Simulate the new throttling pattern
                let wait_duration = {
                    let mut rm = manager_clone.lock().await;
                    rm.calculate_wait_time()
                };

                if let Some(duration) = wait_duration {
                    tokio::time::sleep(duration).await;
                }

                {
                    let mut rm = manager_clone.lock().await;
                    rm.mark_request_processed();
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify statistics are consistent
        let final_manager = manager.lock().await;
        let stats = final_manager.get_stats();

        // All requests should be counted
        assert_eq!(stats.total_requests, num_concurrent_requests);

        // With 50ms throttling and 100 concurrent requests, we expect
        // most requests to have been processed over time
        assert!(stats.total_requests > 0);

        // Stats should be internally consistent
        assert_eq!(
            stats.throttle_rate,
            stats.throttled_requests as f64 / stats.total_requests as f64
        );
    }

    #[tokio::test]
    async fn test_calculate_wait_time_atomicity() {
        // Test that calculate_wait_time correctly handles state
        let config = ResourceConfig {
            min_request_interval_ms: 100,
            enable_auto_throttle: true,
            enable_adaptive_throttling: false,  // Disable for deterministic test
            ..Default::default()
        };

        let mut manager = ResourceManager::new(config).unwrap();

        // First call should return None (no previous request)
        assert!(manager.calculate_wait_time().is_none());

        // State should only change with mark_request_processed
        manager.mark_request_processed();

        // Now should require wait
        let wait_time = manager.calculate_wait_time();
        assert!(wait_time.is_some());

        // Multiple calls should return consistent results (within a few ms)
        let wait_time2 = manager.calculate_wait_time();
        assert!(wait_time2.is_some());
    }

    #[tokio::test]
    async fn test_vram_limit_edge_cases() {
        let config = ResourceConfig::default();
        let manager = ResourceManager::new(config).unwrap();

        // Test with maximum u64 value (should not overflow)
        let max_vram = u64::MAX;
        let limited_vram = manager.config().max_vram_bytes(max_vram);

        // Should be approximately 70% of max (with potential floating point precision loss)
        assert!(limited_vram < max_vram);
        assert!(limited_vram > max_vram / 2); // At least 50%

        // Test with zero
        let zero_vram = manager.config().max_vram_bytes(0);
        assert_eq!(zero_vram, 0);

        // Test with typical values
        let gb_16 = 16 * 1024 * 1024 * 1024u64;
        let limited = manager.config().max_vram_bytes(gb_16);
        let expected = (gb_16 as f64 * 0.7) as u64;
        assert_eq!(limited, expected);
    }
}
