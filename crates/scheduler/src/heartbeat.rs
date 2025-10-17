use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, warn};
use uuid::Uuid;

/// Heartbeat and lease manager
pub struct HeartbeatManager {
    leases: RwLock<HashMap<String, Lease>>,
}

#[derive(Debug, Clone)]
pub struct Lease {
    pub attempt_id: String,
    pub device_id: String,
    pub shard_id: String,
    pub expires_at: DateTime<Utc>,
    pub heartbeat_period_ms: u64,
    pub grace_missed: u32,
    pub missed_count: u32,
    pub last_heartbeat: Option<DateTime<Utc>>,
}

impl HeartbeatManager {
    pub fn new() -> Self {
        Self {
            leases: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new lease for an attempt
    pub async fn create_lease(
        &self,
        device_id: String,
        shard_id: String,
        heartbeat_period_ms: u64,
        grace_missed: u32,
    ) -> Lease {
        let attempt_id = Uuid::new_v4().to_string();

        let lease_duration_ms = heartbeat_period_ms * (grace_missed as u64 + 1);
        let expires_at = Utc::now() + Duration::milliseconds(lease_duration_ms as i64);

        let lease = Lease {
            attempt_id: attempt_id.clone(),
            device_id,
            shard_id,
            expires_at,
            heartbeat_period_ms,
            grace_missed,
            missed_count: 0,
            last_heartbeat: None,
        };

        let mut leases = self.leases.write().await;
        leases.insert(attempt_id, lease.clone());

        debug!(
            attempt_id = %lease.attempt_id,
            expires_at = %lease.expires_at,
            "Created lease"
        );

        lease
    }

    /// Record a heartbeat
    pub async fn heartbeat(&self, attempt_id: &str) -> Result<bool> {
        let mut leases = self.leases.write().await;

        if let Some(lease) = leases.get_mut(attempt_id) {
            let now = Utc::now();

            if now > lease.expires_at {
                warn!(
                    attempt_id = %attempt_id,
                    "Heartbeat received after lease expiration"
                );
                return Ok(false);
            }

            // Reset missed count and extend lease
            lease.missed_count = 0;
            lease.last_heartbeat = Some(now);

            let lease_duration_ms = lease.heartbeat_period_ms * (lease.grace_missed as u64 + 1);
            lease.expires_at = now + Duration::milliseconds(lease_duration_ms as i64);

            debug!(
                attempt_id = %attempt_id,
                new_expires_at = %lease.expires_at,
                "Heartbeat received, lease extended"
            );

            Ok(true)
        } else {
            warn!(
                attempt_id = %attempt_id,
                "Heartbeat for unknown attempt"
            );
            Ok(false)
        }
    }

    /// Check for expired leases
    pub async fn check_expired(&self) -> Vec<String> {
        let mut leases = self.leases.write().await;
        let now = Utc::now();

        let expired: Vec<String> = leases
            .iter()
            .filter(|(_, lease)| now > lease.expires_at)
            .map(|(id, _)| id.clone())
            .collect();

        for attempt_id in &expired {
            if let Some(lease) = leases.remove(attempt_id) {
                warn!(
                    attempt_id = %attempt_id,
                    device_id = %lease.device_id,
                    shard_id = %lease.shard_id,
                    expired_at = %lease.expires_at,
                    "Lease expired"
                );
            }
        }

        expired
    }

    /// Release a lease (on completion or failure)
    pub async fn release(&self, attempt_id: &str) {
        let mut leases = self.leases.write().await;
        if leases.remove(attempt_id).is_some() {
            debug!(
                attempt_id = %attempt_id,
                "Lease released"
            );
        }
    }

    /// Get lease info
    pub async fn get_lease(&self, attempt_id: &str) -> Option<Lease> {
        let leases = self.leases.read().await;
        leases.get(attempt_id).cloned()
    }

    /// Get all active leases
    pub async fn get_all_leases(&self) -> Vec<Lease> {
        let leases = self.leases.read().await;
        leases.values().cloned().collect()
    }
}

impl Default for HeartbeatManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_create_lease() {
        let manager = HeartbeatManager::new();

        let lease = manager
            .create_lease("dev1".to_string(), "shard1".to_string(), 5000, 3)
            .await;

        assert_eq!(lease.device_id, "dev1");
        assert_eq!(lease.shard_id, "shard1");
        assert!(lease.expires_at > Utc::now());
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let manager = HeartbeatManager::new();

        let lease = manager
            .create_lease("dev1".to_string(), "shard1".to_string(), 5000, 3)
            .await;

        let result = manager.heartbeat(&lease.attempt_id).await.unwrap();
        assert!(result);

        let updated = manager.get_lease(&lease.attempt_id).await.unwrap();
        assert!(updated.last_heartbeat.is_some());
        assert_eq!(updated.missed_count, 0);
    }

    #[tokio::test]
    async fn test_lease_expiration() {
        let manager = HeartbeatManager::new();

        // Create lease with very short timeout
        let lease = manager
            .create_lease("dev1".to_string(), "shard1".to_string(), 100, 1)
            .await;

        // Wait for expiration
        sleep(TokioDuration::from_millis(300)).await;

        let expired = manager.check_expired().await;
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], lease.attempt_id);

        // Lease should be removed
        assert!(manager.get_lease(&lease.attempt_id).await.is_none());
    }

    #[tokio::test]
    async fn test_release_lease() {
        let manager = HeartbeatManager::new();

        let lease = manager
            .create_lease("dev1".to_string(), "shard1".to_string(), 5000, 3)
            .await;

        manager.release(&lease.attempt_id).await;

        assert!(manager.get_lease(&lease.attempt_id).await.is_none());
    }
}
