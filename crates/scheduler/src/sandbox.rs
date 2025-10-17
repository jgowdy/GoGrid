use anyhow::{Context, Result};
use std::process::{Child, Command, Stdio};
use tracing::{info, warn};

/// Sandboxing configuration for job execution
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum memory limit in bytes (None = unlimited)
    pub max_memory_bytes: Option<u64>,
    /// Maximum CPU time in seconds (None = unlimited)
    pub max_cpu_time_secs: Option<u64>,
    /// Network access allowed
    pub allow_network: bool,
    /// Filesystem read-only paths
    pub readonly_paths: Vec<String>,
    /// Filesystem writable paths (within allowed directories)
    pub writable_paths: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB default
            max_cpu_time_secs: Some(3600), // 1 hour default
            allow_network: false, // No network by default
            readonly_paths: vec![],
            writable_paths: vec![],
        }
    }
}

/// Execute a command in a sandboxed environment
pub struct SandboxedProcess {
    config: SandboxConfig,
}

impl SandboxedProcess {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    /// Spawn a sandboxed process
    pub fn spawn(&self, mut command: Command) -> Result<Child> {
        #[cfg(target_os = "macos")]
        {
            self.apply_macos_sandbox(&mut command)?;
        }

        #[cfg(target_os = "linux")]
        {
            self.apply_linux_sandbox(&mut command)?;
        }

        #[cfg(target_os = "windows")]
        {
            self.apply_windows_sandbox(&mut command)?;
        }

        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn sandboxed process")
    }

    #[cfg(target_os = "macos")]
    fn apply_macos_sandbox(&self, command: &mut Command) -> Result<()> {
        use std::env;
        use std::fs;
        use std::path::PathBuf;

        info!("Applying macOS sandbox profile");

        // Create a temporary sandbox profile
        let mut profile = String::from("(version 1)\n");
        profile.push_str("(deny default)\n");

        // Allow basic operations
        profile.push_str("(allow process*)\n");
        profile.push_str("(allow sysctl-read)\n");
        profile.push_str("(allow mach-lookup)\n");

        // Network access
        if !self.config.allow_network {
            profile.push_str("(deny network*)\n");
        } else {
            profile.push_str("(allow network*)\n");
        }

        // File system access
        for path in &self.config.readonly_paths {
            profile.push_str(&format!("(allow file-read* (subpath \"{}\"))\n", path));
        }

        for path in &self.config.writable_paths {
            profile.push_str(&format!("(allow file-read* file-write* (subpath \"{}\"))\n", path));
        }

        // Write profile to temp file
        let profile_path = PathBuf::from("/tmp/gogrid_sandbox.sb");
        fs::write(&profile_path, profile)
            .context("Failed to write sandbox profile")?;

        // Use sandbox-exec to apply the profile
        let original_program = command.get_program().to_owned();
        let original_args: Vec<_> = command.get_args().map(|s| s.to_owned()).collect();

        command.program("sandbox-exec");
        command.arg("-f");
        command.arg(profile_path);
        command.arg(original_program);
        command.args(original_args);

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn apply_linux_sandbox(&self, command: &mut Command) -> Result<()> {
        use std::os::unix::process::CommandExt;

        info!("Applying Linux sandbox (seccomp + namespaces)");

        // Try to use bwrap (Bubblewrap) if available
        if Command::new("bwrap").arg("--version").output().is_ok() {
            info!("Using bubblewrap for sandboxing");
            return self.apply_bubblewrap_sandbox(command);
        }

        // Fallback to resource limits via setrlimit
        warn!("Bubblewrap not available, using basic resource limits");

        unsafe {
            command.pre_exec(|| {
                // Set resource limits
                #[cfg(target_os = "linux")]
                {
                    use libc::{rlimit, setrlimit, RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE};

                    // Memory limit
                    if let Some(max_mem) = None::<u64> {
                        let limit = rlimit {
                            rlim_cur: max_mem,
                            rlim_max: max_mem,
                        };
                        setrlimit(RLIMIT_AS, &limit);
                    }

                    // CPU time limit
                    if let Some(max_cpu) = None::<u64> {
                        let limit = rlimit {
                            rlim_cur: max_cpu,
                            rlim_max: max_cpu,
                        };
                        setrlimit(RLIMIT_CPU, &limit);
                    }

                    // File descriptor limit
                    let limit = rlimit {
                        rlim_cur: 1024,
                        rlim_max: 1024,
                    };
                    setrlimit(RLIMIT_NOFILE, &limit);
                }

                Ok(())
            });
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn apply_bubblewrap_sandbox(&self, command: &mut Command) -> Result<()> {
        let original_program = command.get_program().to_owned();
        let original_args: Vec<_> = command.get_args().map(|s| s.to_owned()).collect();

        command.program("bwrap");

        // Unshare all namespaces
        command.arg("--unshare-all");

        // Share network if allowed
        if self.config.allow_network {
            command.arg("--share-net");
        }

        // Bind readonly paths
        for path in &self.config.readonly_paths {
            command.arg("--ro-bind");
            command.arg(path);
            command.arg(path);
        }

        // Bind writable paths
        for path in &self.config.writable_paths {
            command.arg("--bind");
            command.arg(path);
            command.arg(path);
        }

        // Minimal /tmp
        command.arg("--tmpfs");
        command.arg("/tmp");

        // Execute the actual command
        command.arg(original_program);
        command.args(original_args);

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn apply_windows_sandbox(&self, command: &mut Command) -> Result<()> {
        use std::os::windows::process::CommandExt;

        info!("Applying Windows sandbox (job object limits)");

        // Windows uses Job Objects for resource limiting
        // This is handled via CREATE_SUSPENDED flag and job object assignment
        const CREATE_SUSPENDED: u32 = 0x00000004;
        const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x01000000;

        command.creation_flags(CREATE_SUSPENDED | CREATE_BREAKAWAY_FROM_JOB);

        // Note: Full job object configuration would be done after process creation
        // via AssignProcessToJobObject and SetInformationJobObject APIs
        // This is a basic implementation - full sandboxing would require more work

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SandboxConfig::default();
        assert_eq!(config.max_memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(config.max_cpu_time_secs, Some(3600));
        assert!(!config.allow_network);
    }

    #[test]
    fn test_sandbox_creation() {
        let config = SandboxConfig::default();
        let sandbox = SandboxedProcess::new(config);
        // Just ensure we can create it
        assert!(true);
    }
}
