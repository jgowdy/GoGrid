#[cfg(target_os = "windows")]
use anyhow::Context;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerStatus {
    pub on_ac_power: bool,
    pub battery_percent: u8,
}

impl PowerStatus {
    /// Check if execution is allowed based on power status
    /// Battery operation is strictly prohibited per design
    pub fn allows_execution(&self) -> bool {
        self.on_ac_power
    }
}

/// Cross-platform power monitor
pub struct PowerMonitor;

impl PowerMonitor {
    /// Get current power status
    pub fn get_status() -> Result<PowerStatus> {
        #[cfg(target_os = "macos")]
        {
            Self::get_status_macos()
        }

        #[cfg(target_os = "windows")]
        {
            Self::get_status_windows()
        }

        #[cfg(target_os = "linux")]
        {
            Self::get_status_linux()
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            anyhow::bail!("Unsupported platform for power monitoring")
        }
    }

    #[cfg(target_os = "macos")]
    fn get_status_macos() -> Result<PowerStatus> {
        crate::power_macos::get_macos_power_status()
    }

    #[cfg(target_os = "windows")]
    fn get_status_windows() -> Result<PowerStatus> {
        use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};

        let mut status = SYSTEM_POWER_STATUS::default();
        unsafe {
            GetSystemPowerStatus(&mut status)
                .context("Failed to get Windows power status")?;
        }

        let on_ac_power = status.ACLineStatus == 1;
        let battery_percent = if status.BatteryLifePercent == 255 {
            100 // Unknown, assume full
        } else {
            status.BatteryLifePercent
        };

        Ok(PowerStatus {
            on_ac_power,
            battery_percent,
        })
    }

    #[cfg(target_os = "linux")]
    fn get_status_linux() -> Result<PowerStatus> {
        use std::fs;
        use std::path::Path;

        // Try UPower via sysfs
        let power_supply_path = Path::new("/sys/class/power_supply");

        if !power_supply_path.exists() {
            warn!("Power supply sysfs not found, assuming AC power");
            return Ok(PowerStatus {
                on_ac_power: true,
                battery_percent: 100,
            });
        }

        let mut on_ac_power = false;
        let mut battery_percent = 100u8;

        // Check for AC adapter
        for entry in fs::read_dir(power_supply_path)? {
            let entry = entry?;
            let path = entry.path();
            let type_path = path.join("type");

            if let Ok(psu_type) = fs::read_to_string(&type_path) {
                if psu_type.trim() == "Mains" {
                    let online_path = path.join("online");
                    if let Ok(online) = fs::read_to_string(&online_path) {
                        on_ac_power = online.trim() == "1";
                    }
                } else if psu_type.trim() == "Battery" {
                    let capacity_path = path.join("capacity");
                    if let Ok(capacity) = fs::read_to_string(&capacity_path) {
                        if let Ok(cap) = capacity.trim().parse::<u8>() {
                            battery_percent = cap;
                        }
                    }
                }
            }
        }

        Ok(PowerStatus {
            on_ac_power,
            battery_percent,
        })
    }

    /// Monitor power status and call callback when it changes
    pub async fn monitor<F>(mut callback: F) -> Result<()>
    where
        F: FnMut(PowerStatus) + Send + 'static,
    {
        let mut last_status = Self::get_status()?;
        callback(last_status);

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            match Self::get_status() {
                Ok(status) => {
                    if status != last_status {
                        if !status.on_ac_power && last_status.on_ac_power {
                            warn!("POWER TRANSITION: AC -> Battery - execution must stop immediately");
                        } else if status.on_ac_power && !last_status.on_ac_power {
                            info!("POWER TRANSITION: Battery -> AC - execution can resume");
                        }
                        callback(status);
                        last_status = status;
                    }
                }
                Err(e) => {
                    warn!("Failed to get power status: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_status_allows_execution() {
        let ac_status = PowerStatus {
            on_ac_power: true,
            battery_percent: 100,
        };
        assert!(ac_status.allows_execution());

        let battery_status = PowerStatus {
            on_ac_power: false,
            battery_percent: 80,
        };
        assert!(!battery_status.allows_execution());
    }
}
