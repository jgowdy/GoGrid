/// System Activity Monitoring for Adaptive Throttling
///
/// This module monitors system resources (CPU, GPU, memory) to automatically
/// back off when the system is busy, ensuring minimal impact on user activity.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

#[cfg(target_os = "linux")]
use std::fs;

/// System load thresholds for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoadThresholds {
    /// CPU load average threshold (per-core, 0.0-1.0)
    /// Above this, we pause inference
    /// Default: 0.7 (70% per core)
    pub cpu_load_threshold: f64,

    /// GPU utilization threshold (0.0-1.0)
    /// Above this, we pause inference
    /// Default: 0.8 (80% busy)
    pub gpu_utilization_threshold: f64,

    /// Memory usage threshold (0.0-1.0)
    /// Above this, we pause inference
    /// Default: 0.9 (90% used)
    pub memory_threshold: f64,

    /// How often to check system load (seconds)
    /// Default: 5 seconds
    pub check_interval_secs: u64,

    /// How long to pause when system is busy (seconds)
    /// Default: 30 seconds
    pub pause_duration_secs: u64,
}

impl Default for SystemLoadThresholds {
    fn default() -> Self {
        Self {
            cpu_load_threshold: 0.7,
            gpu_utilization_threshold: 0.8,
            memory_threshold: 0.9,
            check_interval_secs: 5,
            pause_duration_secs: 30,
        }
    }
}

impl SystemLoadThresholds {
    /// Create conservative thresholds (pause easily)
    pub fn conservative() -> Self {
        Self {
            cpu_load_threshold: 0.5,  // Pause at 50% CPU
            gpu_utilization_threshold: 0.6,  // Pause at 60% GPU
            memory_threshold: 0.8,  // Pause at 80% memory
            check_interval_secs: 3,  // Check more frequently
            pause_duration_secs: 60,  // Pause longer
        }
    }

    /// Create aggressive thresholds (rarely pause)
    pub fn aggressive() -> Self {
        Self {
            cpu_load_threshold: 0.9,  // Only pause at 90% CPU
            gpu_utilization_threshold: 0.95,  // Only pause at 95% GPU
            memory_threshold: 0.95,  // Only pause at 95% memory
            check_interval_secs: 10,  // Check less frequently
            pause_duration_secs: 15,  // Pause briefly
        }
    }
}

/// Current system load information
#[derive(Debug, Clone)]
pub struct SystemLoad {
    pub cpu_load: Option<f64>,  // Per-core load average (0.0-1.0+)
    pub gpu_utilization: Option<f64>,  // GPU busy percentage (0.0-1.0)
    pub memory_used: Option<f64>,  // Memory usage (0.0-1.0)
    pub timestamp: Instant,
}

/// System monitor for adaptive throttling
pub struct SystemMonitor {
    thresholds: SystemLoadThresholds,
    last_check: Option<Instant>,
    last_load: Option<SystemLoad>,
    paused_until: Option<Instant>,
    total_checks: u64,
    pauses_triggered: u64,
}

impl SystemMonitor {
    /// Create a new system monitor with the given thresholds
    pub fn new(thresholds: SystemLoadThresholds) -> Self {
        Self {
            thresholds,
            last_check: None,
            last_load: None,
            paused_until: None,
            total_checks: 0,
            pauses_triggered: 0,
        }
    }

    /// Check if we should allow inference based on current system load
    /// Returns true if we should proceed, false if we should wait
    pub fn should_allow_inference(&mut self) -> Result<bool> {
        // Check if we're currently in a pause period
        if let Some(paused_until) = self.paused_until {
            if Instant::now() < paused_until {
                let remaining = paused_until.duration_since(Instant::now());
                debug!(
                    remaining_secs = remaining.as_secs(),
                    "System monitor: paused due to high load"
                );
                return Ok(false);
            } else {
                // Pause period ended
                self.paused_until = None;
                debug!("System monitor: pause period ended");
            }
        }

        // Check if it's time to re-check system load
        let should_check = match self.last_check {
            None => true,
            Some(last) => {
                last.elapsed() >= Duration::from_secs(self.thresholds.check_interval_secs)
            }
        };

        if !should_check {
            // Use cached decision
            return Ok(true);
        }

        // Perform system load check
        self.total_checks += 1;
        let load = self.check_system_load()?;

        // Check each threshold
        let cpu_overloaded = load
            .cpu_load
            .map(|l| l > self.thresholds.cpu_load_threshold)
            .unwrap_or(false);

        let gpu_overloaded = load
            .gpu_utilization
            .map(|u| u > self.thresholds.gpu_utilization_threshold)
            .unwrap_or(false);

        let memory_overloaded = load
            .memory_used
            .map(|m| m > self.thresholds.memory_threshold)
            .unwrap_or(false);

        // If any resource is overloaded, trigger pause
        if cpu_overloaded || gpu_overloaded || memory_overloaded {
            self.pauses_triggered += 1;
            self.paused_until = Some(
                Instant::now() + Duration::from_secs(self.thresholds.pause_duration_secs),
            );

            warn!(
                cpu_load = ?load.cpu_load,
                gpu_util = ?load.gpu_utilization,
                memory_used = ?load.memory_used,
                pause_secs = self.thresholds.pause_duration_secs,
                "System monitor: High load detected, pausing inference"
            );

            return Ok(false);
        }

        self.last_check = Some(Instant::now());
        self.last_load = Some(load);

        Ok(true)
    }

    /// Get the current pause state
    pub fn get_pause_duration(&self) -> Option<Duration> {
        self.paused_until.map(|until| {
            let now = Instant::now();
            if now < until {
                until.duration_since(now)
            } else {
                Duration::from_secs(0)
            }
        })
    }

    /// Check system load (CPU, GPU, memory)
    fn check_system_load(&self) -> Result<SystemLoad> {
        let cpu_load = self.get_cpu_load().ok();
        let gpu_utilization = self.get_gpu_utilization().ok();
        let memory_used = self.get_memory_usage().ok();

        Ok(SystemLoad {
            cpu_load,
            gpu_utilization,
            memory_used,
            timestamp: Instant::now(),
        })
    }

    /// Get CPU load average (per-core)
    #[cfg(target_os = "linux")]
    fn get_cpu_load(&self) -> Result<f64> {
        // Read /proc/loadavg for 1-minute load average
        let loadavg = fs::read_to_string("/proc/loadavg")
            .map_err(|e| anyhow!("Failed to read /proc/loadavg: {}", e))?;

        let parts: Vec<&str> = loadavg.split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow!("Invalid /proc/loadavg format"));
        }

        let load_1min: f64 = parts[0]
            .parse()
            .map_err(|e| anyhow!("Failed to parse load average: {}", e))?;

        // Get number of CPU cores
        let num_cpus = num_cpus::get() as f64;

        // Return per-core load average
        Ok(load_1min / num_cpus)
    }

    #[cfg(target_os = "macos")]
    fn get_cpu_load(&self) -> Result<f64> {
        // Use sysctl to get load average
        let output = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("vm.loadavg")
            .output()
            .map_err(|e| anyhow!("Failed to execute sysctl: {}", e))?;

        if !output.status.success() {
            return Err(anyhow!("sysctl failed"));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        // Format: "{ 1.23 2.34 3.45 }"
        let parts: Vec<&str> = output_str
            .trim()
            .trim_matches(|c| c == '{' || c == '}')
            .split_whitespace()
            .collect();

        if parts.is_empty() {
            return Err(anyhow!("Invalid sysctl output"));
        }

        let load_1min: f64 = parts[0]
            .parse()
            .map_err(|e| anyhow!("Failed to parse load average: {}", e))?;

        let num_cpus = num_cpus::get() as f64;
        Ok(load_1min / num_cpus)
    }

    #[cfg(target_os = "windows")]
    fn get_cpu_load(&self) -> Result<f64> {
        // Windows doesn't have load average, could use performance counters
        // For now, return error
        Err(anyhow!(
            "CPU load monitoring not implemented on Windows"
        ))
    }

    /// Get GPU utilization (0.0-1.0)
    #[cfg(target_os = "linux")]
    fn get_gpu_utilization(&self) -> Result<f64> {
        // Try sysfs first (faster)
        let sysfs_path = "/sys/class/drm/card0/device/gpu_busy_percent";
        if let Ok(content) = fs::read_to_string(sysfs_path) {
            if let Ok(util) = content.trim().parse::<f64>() {
                return Ok(util / 100.0);
            }
        }

        // Fall back to nvidia-smi
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|e| anyhow!("Failed to execute nvidia-smi: {}", e))?;

        if !output.status.success() {
            return Err(anyhow!("nvidia-smi failed"));
        }

        let util_str = String::from_utf8_lossy(&output.stdout);
        let util = util_str
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("Failed to parse GPU utilization: {}", e))?;

        Ok(util / 100.0)
    }

    #[cfg(not(target_os = "linux"))]
    fn get_gpu_utilization(&self) -> Result<f64> {
        // Try nvidia-smi on all platforms
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|e| anyhow!("Failed to execute nvidia-smi: {}", e))?;

        if !output.status.success() {
            return Err(anyhow!("nvidia-smi not available"));
        }

        let util_str = String::from_utf8_lossy(&output.stdout);
        let util = util_str
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("Failed to parse GPU utilization: {}", e))?;

        Ok(util / 100.0)
    }

    /// Get memory usage (0.0-1.0)
    #[cfg(target_os = "linux")]
    fn get_memory_usage(&self) -> Result<f64> {
        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| anyhow!("Failed to read /proc/meminfo: {}", e))?;

        let mut total_kb = None;
        let mut available_kb = None;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok());
            } else if line.starts_with("MemAvailable:") {
                available_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok());
            }
        }

        match (total_kb, available_kb) {
            (Some(total), Some(available)) => {
                let used = total.saturating_sub(available);
                Ok(used as f64 / total as f64)
            }
            _ => Err(anyhow!("Failed to parse memory info")),
        }
    }

    #[cfg(target_os = "macos")]
    fn get_memory_usage(&self) -> Result<f64> {
        // Use vm_stat on macOS
        let output = std::process::Command::new("vm_stat")
            .output()
            .map_err(|e| anyhow!("Failed to execute vm_stat: {}", e))?;

        if !output.status.success() {
            return Err(anyhow!("vm_stat failed"));
        }

        // Parse vm_stat output
        // This is a simplified implementation
        let output_str = String::from_utf8_lossy(&output.stdout);

        let mut free_pages = 0u64;
        let mut active_pages = 0u64;
        let mut inactive_pages = 0u64;
        let mut wired_pages = 0u64;

        for line in output_str.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            let value_str = parts[1].trim().trim_end_matches('.');
            let value: u64 = value_str.parse().unwrap_or(0);

            if parts[0].contains("Pages free") {
                free_pages = value;
            } else if parts[0].contains("Pages active") {
                active_pages = value;
            } else if parts[0].contains("Pages inactive") {
                inactive_pages = value;
            } else if parts[0].contains("Pages wired down") {
                wired_pages = value;
            }
        }

        let total_pages = free_pages + active_pages + inactive_pages + wired_pages;
        if total_pages == 0 {
            return Err(anyhow!("Failed to calculate memory usage"));
        }

        let used_pages = active_pages + wired_pages;
        Ok(used_pages as f64 / total_pages as f64)
    }

    #[cfg(target_os = "windows")]
    fn get_memory_usage(&self) -> Result<f64> {
        // Windows: use GlobalMemoryStatusEx via winapi
        // For now, return error
        Err(anyhow!(
            "Memory usage monitoring not implemented on Windows"
        ))
    }

    /// Get monitoring statistics
    pub fn get_stats(&self) -> SystemMonitorStats {
        SystemMonitorStats {
            total_checks: self.total_checks,
            pauses_triggered: self.pauses_triggered,
            currently_paused: self.paused_until.is_some(),
            last_load: self.last_load.clone(),
        }
    }

    /// Get the configured thresholds
    pub fn thresholds(&self) -> &SystemLoadThresholds {
        &self.thresholds
    }
}

/// Statistics for system monitoring
#[derive(Debug, Clone)]
pub struct SystemMonitorStats {
    pub total_checks: u64,
    pub pauses_triggered: u64,
    pub currently_paused: bool,
    pub last_load: Option<SystemLoad>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_defaults() {
        let thresholds = SystemLoadThresholds::default();
        assert_eq!(thresholds.cpu_load_threshold, 0.7);
        assert_eq!(thresholds.gpu_utilization_threshold, 0.8);
        assert_eq!(thresholds.memory_threshold, 0.9);
    }

    #[test]
    fn test_conservative_thresholds() {
        let thresholds = SystemLoadThresholds::conservative();
        assert!(thresholds.cpu_load_threshold < 0.7);
        assert!(thresholds.pause_duration_secs > 30);
    }

    #[test]
    fn test_aggressive_thresholds() {
        let thresholds = SystemLoadThresholds::aggressive();
        assert!(thresholds.cpu_load_threshold > 0.7);
        assert!(thresholds.pause_duration_secs < 30);
    }

    #[tokio::test]
    async fn test_system_monitor_creation() {
        let thresholds = SystemLoadThresholds::default();
        let monitor = SystemMonitor::new(thresholds);

        let stats = monitor.get_stats();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.pauses_triggered, 0);
        assert!(!stats.currently_paused);
    }

    #[tokio::test]
    async fn test_pause_duration() {
        let mut monitor = SystemMonitor::new(SystemLoadThresholds::default());

        // Initially no pause
        assert!(monitor.get_pause_duration().is_none());

        // Set a pause
        monitor.paused_until = Some(Instant::now() + Duration::from_secs(10));

        // Should have pause duration
        let pause = monitor.get_pause_duration();
        assert!(pause.is_some());
        assert!(pause.unwrap().as_secs() <= 10);
    }
}
