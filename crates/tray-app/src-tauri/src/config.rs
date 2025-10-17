use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub coordinator: CoordinatorConfig,
    pub updates: UpdatesConfig,
    pub worker: WorkerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatesConfig {
    pub enabled: bool,
    pub endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub max_vram_gb: Option<f64>,
    pub pause_on_activity: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            coordinator: CoordinatorConfig {
                host: String::new(),
                port: 8443,
            },
            updates: UpdatesConfig {
                enabled: true,
                endpoints: vec![],
            },
            worker: WorkerConfig {
                max_vram_gb: None,
                pause_on_activity: true,
            },
        }
    }
}

impl Config {
    /// Get the configuration file path
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = if cfg!(target_os = "macos") {
            dirs::config_dir()
                .ok_or_else(|| anyhow::anyhow!("Failed to get config directory"))?
                .join("GoGrid Worker")
        } else if cfg!(target_os = "linux") {
            dirs::config_dir()
                .ok_or_else(|| anyhow::anyhow!("Failed to get config directory"))?
                .join("gogrid-worker")
        } else if cfg!(target_os = "windows") {
            dirs::config_dir()
                .ok_or_else(|| anyhow::anyhow!("Failed to get config directory"))?
                .join("GoGrid Worker")
        } else {
            return Err(anyhow::anyhow!("Unsupported platform"));
        };

        fs::create_dir_all(&config_dir)
            .context("Failed to create config directory")?;

        Ok(config_dir.join("config.toml"))
    }

    /// Load configuration from file and environment variables
    /// Environment variables take precedence over config file
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        let mut config = if config_path.exists() {
            info!("Loading configuration from {:?}", config_path);
            let contents = fs::read_to_string(&config_path)
                .context("Failed to read config file")?;
            toml::from_str(&contents)
                .context("Failed to parse config file")?
        } else {
            info!("Config file not found, using defaults");
            Self::default()
        };

        // Override with environment variables if present
        if let Ok(host) = std::env::var("GOGRID_COORDINATOR_HOST") {
            info!("Using GOGRID_COORDINATOR_HOST from environment: {}", host);
            config.coordinator.host = host;
        }

        if let Ok(port_str) = std::env::var("GOGRID_COORDINATOR_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                info!("Using GOGRID_COORDINATOR_PORT from environment: {}", port);
                config.coordinator.port = port;
            }
        }

        if let Ok(endpoints_str) = std::env::var("GOGRID_UPDATE_ENDPOINTS") {
            let endpoints: Vec<String> = endpoints_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !endpoints.is_empty() {
                info!("Using GOGRID_UPDATE_ENDPOINTS from environment");
                config.updates.endpoints = endpoints;
            }
        }

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        let toml_string = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        fs::write(&config_path, toml_string)
            .context("Failed to write config file")?;

        info!("Configuration saved to {:?}", config_path);
        Ok(())
    }

    /// Check if configuration is complete
    pub fn is_configured(&self) -> bool {
        !self.coordinator.host.is_empty()
    }

    /// Prompt user for configuration via dialog
    pub async fn prompt_for_config() -> Result<Self> {
        info!("First run detected - configuration required");

        // Log welcome message
        info!("Welcome to GoGrid Worker!");
        info!("You need to configure your coordinator server address.");
        info!("This can be your own self-hosted coordinator or a coordinator URL provided to you.");

        // Check if environment variables are set
        if let (Ok(host), Ok(port_str)) = (
            std::env::var("GOGRID_COORDINATOR_HOST"),
            std::env::var("GOGRID_COORDINATOR_PORT"),
        ) {
            if let Ok(port) = port_str.parse::<u16>() {
                info!("Using coordinator from environment: {}:{}", host, port);

                let mut config = Config::default();
                config.coordinator.host = host.clone();
                config.coordinator.port = port;

                // Prompt for update endpoints
                if let Ok(endpoints_str) = std::env::var("GOGRID_UPDATE_ENDPOINTS") {
                    config.updates.endpoints = endpoints_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                } else {
                    // Suggest using same server for updates
                    config.updates.endpoints = vec![
                        format!("https://{}:{}/updates/{{{{target}}}}/{{{{current_version}}}}", host, port)
                    ];
                }

                config.save()?;
                return Ok(config);
            }
        }

        // Show configuration instructions
        let config_path = Self::config_path()?;
        warn!("Configuration required!");
        warn!("Please set environment variables:");
        warn!("  GOGRID_COORDINATOR_HOST=your-server.com");
        warn!("  GOGRID_COORDINATOR_PORT=8443");
        warn!("Or edit the configuration file:");
        warn!("  {}", config_path.display());
        warn!("After configuring, restart GoGrid Worker.");

        // Create a template config file
        let mut config = Config::default();
        config.coordinator.host = "your-coordinator-server.com".to_string();
        config.coordinator.port = 8443;
        config.updates.endpoints = vec![
            "https://your-coordinator-server.com:8443/updates/{{target}}/{{current_version}}".to_string()
        ];
        config.save()?;

        Err(anyhow::anyhow!(
            "Configuration required. Please set GOGRID_COORDINATOR_HOST and GOGRID_COORDINATOR_PORT \
             environment variables or edit {}",
            config_path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.coordinator.port, 8443);
        assert!(config.updates.enabled);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("coordinator"));
        assert!(toml_str.contains("updates"));
    }
}
