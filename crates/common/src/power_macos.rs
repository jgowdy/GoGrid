#[cfg(target_os = "macos")]
use anyhow::Result;
use core_foundation::array::CFArray;
use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use core_foundation_sys::base::{CFRelease, CFTypeRef};

use super::PowerStatus;

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> CFTypeRef;
    fn IOPSCopyPowerSourcesList(blob: CFTypeRef) -> CFTypeRef;
    fn IOPSGetPowerSourceDescription(blob: CFTypeRef, ps: CFTypeRef) -> CFTypeRef;
}

pub fn get_macos_power_status() -> Result<PowerStatus> {
    unsafe {
        // Get power sources info
        let ps_info = IOPSCopyPowerSourcesInfo();
        if ps_info.is_null() {
            anyhow::bail!("Failed to get power sources info");
        }

        // Get list of power sources
        let ps_list = IOPSCopyPowerSourcesList(ps_info);
        if ps_list.is_null() {
            CFRelease(ps_info);
            anyhow::bail!("Failed to get power sources list");
        }

        let list: CFArray<CFType> = CFArray::wrap_under_create_rule(ps_list as *const _);

        let mut on_ac_power = false;
        let mut battery_percent = 100u8;

        // Iterate through power sources
        for i in 0..list.len() {
            if let Some(ps) = list.get(i) {
                let ps_ref = ps.as_CFTypeRef();
                let desc = IOPSGetPowerSourceDescription(ps_info, ps_ref);

                if desc.is_null() {
                    continue;
                }

                let dict = CFDictionary::<CFString, CFTypeRef>::wrap_under_get_rule(desc as *const _);

                // Check power source state (AC Power, Battery Power, etc.)
                if let Some(state) = dict.find(CFString::from_static_string("Power Source State").as_concrete_TypeRef()) {
                    let state_str = CFString::wrap_under_get_rule(*state as *const _);
                    on_ac_power = state_str.to_string() == "AC Power";
                }

                // Get current capacity (battery percentage)
                if let Some(current_capacity) = dict.find(CFString::from_static_string("Current Capacity").as_concrete_TypeRef()) {
                    let capacity = CFNumber::wrap_under_get_rule(*current_capacity as *const _);
                    if let Some(val) = capacity.to_i32() {
                        battery_percent = val.clamp(0, 100) as u8;
                    }
                }

                // If we found battery info, we're done
                if !on_ac_power {
                    break;
                }
            }
        }

        CFRelease(ps_info);

        Ok(PowerStatus {
            on_ac_power,
            battery_percent,
        })
    }
}
