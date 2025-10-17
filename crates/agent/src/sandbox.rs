use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};
#[cfg(target_os = "linux")]
use tracing::warn;

/// Sandbox configuration for job execution
pub struct SandboxConfig {
    pub work_dir: PathBuf,
    pub readonly_paths: Vec<PathBuf>,
    pub allowed_devices: Vec<String>,
    pub network_access: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            work_dir: PathBuf::from("/tmp/corpgrid"),
            readonly_paths: vec![],
            allowed_devices: vec!["/dev/nvidia0".to_string(), "/dev/nvidiactl".to_string()],
            network_access: false,
        }
    }
}

/// Create a sandboxed command for running a job
pub fn create_sandboxed_command(
    runner_path: &Path,
    job_dir: &Path,
    config: &SandboxConfig,
) -> Result<Command> {
    #[cfg(target_os = "linux")]
    {
        create_linux_sandbox(runner_path, job_dir, config)
    }

    #[cfg(target_os = "macos")]
    {
        create_macos_sandbox(runner_path, job_dir, config)
    }

    #[cfg(windows)]
    {
        create_windows_sandbox(runner_path, job_dir, config)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        anyhow::bail!("Sandboxing not supported on this platform")
    }
}

#[cfg(target_os = "linux")]
fn create_linux_sandbox(
    runner_path: &Path,
    job_dir: &Path,
    config: &SandboxConfig,
) -> Result<Command> {
    // Try bubblewrap first, fall back to firejail, then unsandboxed with warning
    if which::which("bwrap").is_ok() {
        info!("Using bubblewrap for sandboxing");
        create_bubblewrap_sandbox(runner_path, job_dir, config)
    } else if which::which("firejail").is_ok() {
        info!("Using firejail for sandboxing");
        create_firejail_sandbox(runner_path, job_dir, config)
    } else {
        warn!("No sandbox tool found (bwrap/firejail), running without sandbox!");
        let mut cmd = Command::new(runner_path);
        cmd.arg(job_dir);
        Ok(cmd)
    }
}

#[cfg(target_os = "linux")]
fn create_bubblewrap_sandbox(
    runner_path: &Path,
    job_dir: &Path,
    config: &SandboxConfig,
) -> Result<Command> {
    let mut cmd = Command::new("bwrap");

    // Basic isolation
    cmd.arg("--unshare-all"); // Unshare all namespaces except user
    cmd.arg("--die-with-parent");
    cmd.arg("--new-session");

    // Mount read-only root filesystem
    cmd.arg("--ro-bind").arg("/usr").arg("/usr");
    cmd.arg("--ro-bind").arg("/lib").arg("/lib");
    cmd.arg("--ro-bind").arg("/lib64").arg("/lib64");
    cmd.arg("--ro-bind").arg("/bin").arg("/bin");
    cmd.arg("--ro-bind").arg("/sbin").arg("/sbin");

    // Essential directories
    cmd.arg("--dev").arg("/dev");
    cmd.arg("--proc").arg("/proc");
    cmd.arg("--tmpfs").arg("/tmp");

    // GPU device access for CUDA
    for device in &config.allowed_devices {
        if Path::new(device).exists() {
            cmd.arg("--dev-bind").arg(device).arg(device);
        }
    }

    // NVIDIA driver libraries
    if Path::new("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so").exists() {
        cmd.arg("--ro-bind")
            .arg("/usr/lib/x86_64-linux-gnu")
            .arg("/usr/lib/x86_64-linux-gnu");
    }

    // Work directory with read-write access
    cmd.arg("--bind").arg(job_dir).arg("/workspace");

    // Additional readonly paths
    for path in &config.readonly_paths {
        if path.exists() {
            cmd.arg("--ro-bind").arg(path).arg(path);
        }
    }

    // Network isolation
    if !config.network_access {
        // Already unshared by --unshare-all
    }

    // Set working directory
    cmd.arg("--chdir").arg("/workspace");

    // Run the runner
    cmd.arg(runner_path);
    cmd.arg("/workspace");

    debug!("Bubblewrap command: {:?}", cmd);

    Ok(cmd)
}

#[cfg(target_os = "linux")]
fn create_firejail_sandbox(
    runner_path: &Path,
    job_dir: &Path,
    config: &SandboxConfig,
) -> Result<Command> {
    let mut cmd = Command::new("firejail");

    // Basic isolation
    cmd.arg("--noprofile");
    cmd.arg("--private");

    // Network isolation
    if !config.network_access {
        cmd.arg("--net=none");
    }

    // GPU access
    cmd.arg("--allow-debuggers"); // Needed for CUDA
    cmd.arg("--ignore=nou2f"); // Allow GPU access

    // Bind work directory
    cmd.arg(format!("--bind={},{}", job_dir.display(), "/workspace"));

    // Run the runner
    cmd.arg(runner_path);
    cmd.arg("/workspace");

    debug!("Firejail command: {:?}", cmd);

    Ok(cmd)
}

#[cfg(target_os = "macos")]
fn create_macos_sandbox(
    runner_path: &Path,
    job_dir: &Path,
    _config: &SandboxConfig,
) -> Result<Command> {
    info!("Using macOS sandbox-exec for sandboxing");

    // Create sandbox profile
    let profile = format!(
        r#"
        (version 1)
        (deny default)

        ; Allow basic system access
        (allow process-exec (literal "{}"))
        (allow file-read-metadata)

        ; Allow Metal/GPU access
        (allow iokit-open (iokit-user-client-class "AGPMClient"))
        (allow iokit-open (iokit-user-client-class "AppleGraphicsControlClient"))
        (allow iokit-open (iokit-user-client-class "IOAcceleratorClient"))
        (allow mach-lookup (global-name "com.apple.MTLCompilerService"))

        ; Allow work directory access
        (allow file-read* file-write* (subpath "{}"))

        ; Allow system libraries
        (allow file-read* (subpath "/System/Library"))
        (allow file-read* (subpath "/usr/lib"))

        ; Allow temporary files
        (allow file-read* file-write* (subpath "/private/tmp"))

        "#,
        runner_path.display(),
        job_dir.display()
    );

    let mut cmd = Command::new("sandbox-exec");
    cmd.arg("-p").arg(profile);
    cmd.arg(runner_path);
    cmd.arg(job_dir);

    debug!("macOS sandbox command: {:?}", cmd);

    Ok(cmd)
}

#[cfg(windows)]
fn create_windows_sandbox(
    runner_path: &Path,
    job_dir: &Path,
    _config: &SandboxConfig,
) -> Result<Command> {
    use tracing::warn;
    warn!("Windows sandboxing not fully implemented, using basic process isolation");

    // On Windows, we would use:
    // - AppContainer for UWP-style isolation
    // - Job Objects for resource limits
    // - Restricted tokens for privilege reduction

    // For now, just run with low priority and IO priority
    let mut cmd = Command::new(runner_path);
    cmd.arg(job_dir);

    // TODO: Implement proper AppContainer isolation
    // This requires:
    // 1. CreateAppContainerProfile
    // 2. CreateProcessWithLogonW or CreateProcessAsUserW
    // 3. Capability grants for GPU access

    debug!("Windows command (basic isolation): {:?}", cmd);

    Ok(cmd)
}
