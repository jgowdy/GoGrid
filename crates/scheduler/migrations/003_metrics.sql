-- Token and rate metrics tracking

CREATE TABLE IF NOT EXISTS inference_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    api_key_id TEXT REFERENCES api_keys(id) ON DELETE SET NULL,
    request_id TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    generation_time_ms INTEGER NOT NULL,
    tokens_per_second REAL NOT NULL,
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Job metrics tracking
CREATE TABLE IF NOT EXISTS job_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    device_id TEXT,
    backend TEXT,
    start_time TEXT,
    end_time TEXT,
    duration_seconds REAL,
    gpu_utilization_avg REAL,
    vram_used_gb REAL,
    power_consumption_watts REAL,
    temperature_celsius REAL,
    status TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Request rate tracking (aggregated per minute)
CREATE TABLE IF NOT EXISTS request_rate_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL, -- 'inference', 'job', 'api_call'
    model_id TEXT,
    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    request_count INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    avg_duration_ms REAL,
    total_tokens INTEGER,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for performance
CREATE INDEX idx_inference_metrics_model_id ON inference_metrics(model_id);
CREATE INDEX idx_inference_metrics_user_id ON inference_metrics(user_id);
CREATE INDEX idx_inference_metrics_created_at ON inference_metrics(created_at);
CREATE INDEX idx_job_metrics_job_id ON job_metrics(job_id);
CREATE INDEX idx_job_metrics_device_id ON job_metrics(device_id);
CREATE INDEX idx_job_metrics_created_at ON job_metrics(created_at);
CREATE INDEX idx_request_rate_metrics_metric_type ON request_rate_metrics(metric_type);
CREATE INDEX idx_request_rate_metrics_model_id ON request_rate_metrics(model_id);
CREATE INDEX idx_request_rate_metrics_window_start ON request_rate_metrics(window_start);
