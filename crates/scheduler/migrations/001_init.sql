-- Devices table
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY,
    public_key BLOB NOT NULL,
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    hostname TEXT,
    os TEXT,
    arch TEXT,
    memory_bytes INTEGER,
    cpu_cores INTEGER,
    site TEXT,
    on_ac_power INTEGER NOT NULL DEFAULT 1,
    battery_percent INTEGER NOT NULL DEFAULT 100,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Device reputation table (Beta distribution)
CREATE TABLE device_reputation (
    device_id TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_devices_active ON devices(is_active, on_ac_power);
CREATE INDEX idx_devices_site ON devices(site);

-- GPU info table
CREATE TABLE device_gpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    backend TEXT NOT NULL, -- 'cuda' or 'metal'
    vram_bytes INTEGER NOT NULL,
    driver_version TEXT NOT NULL,
    compute_capability TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_device_gpus_device ON device_gpus(device_id);
CREATE INDEX idx_device_gpus_backend ON device_gpus(backend);

-- Jobs table
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    bundle_s3_key TEXT NOT NULL,
    bundle_hash BLOB NOT NULL,
    bundle_signature BLOB NOT NULL,
    -- Job spec (JSON)
    spec TEXT NOT NULL,
    -- Status
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX idx_jobs_status ON jobs(status);

-- Job shards (work units)
CREATE TABLE job_shards (
    job_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    bundle_s3_key TEXT NOT NULL,
    bundle_signature BLOB NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, assigned, running, completed, failed, preempted
    assigned_device_id TEXT,
    attempt_id TEXT,
    assigned_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (job_id, shard_id)
);

CREATE INDEX idx_job_shards_job ON job_shards(job_id);
CREATE INDEX idx_job_shards_status ON job_shards(status);
CREATE INDEX idx_job_shards_attempt ON job_shards(attempt_id) WHERE attempt_id IS NOT NULL;

-- Shard results table (for Byzantine fault tolerance / quorum)
CREATE TABLE shard_results (
    job_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    result_s3_key TEXT NOT NULL,
    result_hash BLOB NOT NULL,
    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (job_id, shard_id, device_id)
);

CREATE INDEX idx_shard_results_job_shard ON shard_results(job_id, shard_id);

-- Job attempts (replication)
CREATE TABLE job_attempts (
    attempt_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    replica_index INTEGER NOT NULL,
    -- Lease
    lease_expires_at TEXT NOT NULL,
    -- Status
    status TEXT NOT NULL DEFAULT 'assigned', -- assigned, running, completed, failed, timeout, preempted
    -- Heartbeat
    last_heartbeat TEXT,
    heartbeat_missed_count INTEGER NOT NULL DEFAULT 0,
    -- Progress
    progress_percent REAL DEFAULT 0.0,
    processed_items INTEGER DEFAULT 0,
    total_items INTEGER DEFAULT 0,
    -- Results
    result_hash BLOB,
    result_s3_key TEXT,
    result_signature BLOB,
    -- Timestamps
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(shard_id, replica_index)
);

CREATE INDEX idx_job_attempts_shard ON job_attempts(shard_id);
CREATE INDEX idx_job_attempts_device ON job_attempts(device_id);
CREATE INDEX idx_job_attempts_status ON job_attempts(status);
CREATE INDEX idx_job_attempts_lease ON job_attempts(lease_expires_at) WHERE status IN ('assigned', 'running');

-- Checkpoints
CREATE TABLE checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    checkpoint_s3_key TEXT NOT NULL,
    checkpoint_hash BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_checkpoints_job_shard ON checkpoints(job_id, shard_id);

-- Attempt history for tracking reassignments
CREATE TABLE attempt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_attempt_history_job_shard ON attempt_history(job_id, shard_id);

-- Audit log for all results
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    event_type TEXT NOT NULL, -- 'assigned', 'heartbeat', 'checkpoint', 'result', 'timeout', etc.
    event_data TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_audit_log_attempt ON audit_log(attempt_id);
CREATE INDEX idx_audit_log_device ON audit_log(device_id);
CREATE INDEX idx_audit_log_job ON audit_log(job_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);

-- Metrics aggregation table
CREATE TABLE metrics_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_metrics_snapshot_type ON metrics_snapshot(metric_type, metric_name);
CREATE INDEX idx_metrics_snapshot_timestamp ON metrics_snapshot(timestamp);
