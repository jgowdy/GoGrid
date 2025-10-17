use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

use crate::{WorkerInfo, WorkerStatus};

#[derive(Debug, Clone)]
pub struct RegisteredWorker {
    pub info: WorkerInfo,
    pub last_heartbeat: Instant,
    pub jobs_completed: u64,
}

pub struct WorkerRegistry {
    workers: HashMap<Uuid, RegisteredWorker>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
        }
    }

    pub fn register_worker(&mut self, worker_id: Uuid, info: WorkerInfo) {
        let registered = RegisteredWorker {
            info,
            last_heartbeat: Instant::now(),
            jobs_completed: 0,
        };
        self.workers.insert(worker_id, registered);
    }

    pub fn update_worker_status(&mut self, worker_id: Uuid, status: WorkerStatus) {
        if let Some(worker) = self.workers.get_mut(&worker_id) {
            worker.info.status = status;
            worker.last_heartbeat = Instant::now();
        }
    }

    pub fn get_worker(&self, id: &Uuid) -> Option<&RegisteredWorker> {
        self.workers.get(id)
    }

    pub fn list_workers(&self) -> Vec<&RegisteredWorker> {
        self.workers.values().collect()
    }

    pub fn remove_stale_workers(&mut self, timeout_secs: u64) -> usize {
        let now = Instant::now();
        let before_count = self.workers.len();

        self.workers.retain(|_, worker| {
            now.duration_since(worker.last_heartbeat).as_secs() < timeout_secs
        });

        before_count - self.workers.len()
    }
}
