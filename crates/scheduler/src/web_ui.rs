use crate::db::DbPool;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use std::sync::Arc;
use uuid::Uuid;

pub struct WebUiState {
    pub db: Arc<DbPool>,
}

pub fn create_web_ui_router(db: Arc<DbPool>) -> Router {
    let state = Arc::new(WebUiState { db });

    Router::new()
        .route("/", get(dashboard_handler))
        .route("/jobs", get(jobs_handler))
        .route("/devices", get(devices_handler))
        .route("/gpu-metrics", get(gpu_metrics_handler))
        .route("/token-metrics", get(token_metrics_handler))
        .route("/users", get(users_handler))
        .route("/api-keys", get(api_keys_handler))
        .route("/status", get(status_handler))
        .with_state(state)
}

async fn dashboard_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query active devices (seen in last 5 minutes)
    let active_devices = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM devices WHERE last_seen > NOW() - INTERVAL '5 minutes'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query running jobs
    let running_jobs = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM job_shards WHERE status = 'running'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query pending jobs
    let pending_jobs = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM job_shards WHERE status = 'pending'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query completed jobs (last 24 hours)
    let completed_jobs = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM job_shards WHERE status = 'completed' AND completed_at > NOW() - INTERVAL '24 hours'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query failed jobs (last 24 hours)
    let failed_jobs = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM job_shards WHERE status = 'failed' AND completed_at > NOW() - INTERVAL '24 hours'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query total GPU count
    let total_gpus = sqlx::query_scalar::<_, i64>(
        "SELECT COALESCE(SUM(gpu_count), 0) FROM devices WHERE last_seen > NOW() - INTERVAL '5 minutes'"
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or(0);

    // Query recent activity (last 10 events)
    let recent_activity: Vec<(String, String, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
        r#"
        SELECT
            CONCAT('Job ', job_id, ' shard ', shard_index) as label,
            status,
            COALESCE(assigned_at, created_at) as timestamp
        FROM job_shards
        ORDER BY COALESCE(assigned_at, created_at) DESC
        LIMIT 10
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let activity_html = if recent_activity.is_empty() {
        "<p>No recent activity</p>".to_string()
    } else {
        let mut html = String::from("<ul style='list-style: none; padding: 0;'>");
        for (label, status, timestamp) in recent_activity {
            let time_ago = format_time_ago(timestamp);
            html.push_str(&format!(
                "<li style='padding: 8px 0; border-bottom: 1px solid #eee;'><strong>{}</strong> - {} <span style='color: #999; font-size: 0.9em;'>({})</span></li>",
                label, status, time_ago
            ));
        }
        html.push_str("</ul>");
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-value {{
            font-weight: bold;
            color: #27ae60;
        }}
        .metric-value.warning {{
            color: #f39c12;
        }}
        .metric-value.danger {{
            color: #e74c3c;
        }}
        nav {{
            margin-top: 20px;
        }}
        nav a {{
            color: #3498db;
            text-decoration: none;
            margin-right: 20px;
        }}
        nav a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CorpGrid Control Plane</h1>
            <p>Distributed GPU Compute Platform</p>
            <nav>
                <a href="/">Dashboard</a>
                <a href="/jobs">Jobs</a>
                <a href="/devices">Devices</a>
                <a href="/gpu-metrics">GPU Metrics</a>
                <a href="/token-metrics">Token Metrics</a>
                <a href="/users">Users</a>
                <a href="/api-keys">API Keys</a>
                <a href="/metrics">Prometheus</a>
            </nav>
        </header>

        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <span>Status</span>
                <span class="metric-value">OPERATIONAL</span>
            </div>
            <div class="metric">
                <span>Active Devices</span>
                <span class="metric-value">{}</span>
            </div>
            <div class="metric">
                <span>Total GPUs</span>
                <span class="metric-value">{}</span>
            </div>
            <div class="metric">
                <span>Running Jobs</span>
                <span class="metric-value {}">{}</span>
            </div>
            <div class="metric">
                <span>Pending Jobs</span>
                <span class="metric-value {}">{}</span>
            </div>
        </div>

        <div class="card">
            <h2>Last 24 Hours</h2>
            <div class="metric">
                <span>Completed Jobs</span>
                <span class="metric-value">{}</span>
            </div>
            <div class="metric">
                <span>Failed Jobs</span>
                <span class="metric-value {}">{}</span>
            </div>
        </div>

        <div class="card">
            <h2>Recent Activity</h2>
            {}
        </div>

        <div class="card">
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="https://github.com/yourorg/corpgrid">Documentation</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
    "#,
        active_devices,
        total_gpus,
        if running_jobs > 0 { "" } else { "warning" },
        running_jobs,
        if pending_jobs > 0 { "warning" } else { "" },
        pending_jobs,
        completed_jobs,
        if failed_jobs > 0 { "danger" } else { "" },
        failed_jobs,
        activity_html
    ))
}

fn format_time_ago(timestamp: chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(timestamp);

    if duration.num_seconds() < 60 {
        format!("{}s ago", duration.num_seconds())
    } else if duration.num_minutes() < 60 {
        format!("{}m ago", duration.num_minutes())
    } else if duration.num_hours() < 24 {
        format!("{}h ago", duration.num_hours())
    } else {
        format!("{}d ago", duration.num_days())
    }
}

async fn jobs_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query jobs from database
    let jobs: Vec<(String, i32, String, Option<String>, Option<String>, Option<f32>)> = sqlx::query_as(
        r#"
        SELECT
            job_id,
            shard_index,
            status,
            assigned_device_id,
            backend,
            percent_complete
        FROM job_shards
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let rows_html = if jobs.is_empty() {
        "<tr><td colspan=\"6\" style=\"text-align: center; padding: 40px;\">No jobs found</td></tr>".to_string()
    } else {
        let mut html = String::new();
        for (job_id, shard_index, status, device_id, backend, progress) in jobs {
            let status_class = match status.as_str() {
                "pending" => "status-pending",
                "running" => "status-running",
                "completed" => "status-completed",
                "failed" => "status-failed",
                _ => "",
            };
            let progress_pct = progress.unwrap_or(0.0) * 100.0;
            let backend_display = backend.unwrap_or_else(|| "-".to_string());
            let device_display = device_id.unwrap_or_else(|| "-".to_string());

            html.push_str(&format!(
                r#"<tr>
                    <td>{}-{}</td>
                    <td class="{}">{}</td>
                    <td>{}</td>
                    <td>{:.1}%</td>
                    <td>{}</td>
                </tr>"#,
                job_id, shard_index, status_class, status, backend_display, progress_pct, device_display
            ));
        }
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - Jobs</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c3e50;
            color: white;
        }}
        .status-pending {{ color: #f39c12; font-weight: bold; }}
        .status-running {{ color: #3498db; font-weight: bold; }}
        .status-completed {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Jobs</h1>
    <p><a href="/">← Back to Dashboard</a></p>
    <table>
        <thead>
            <tr>
                <th>Job ID</th>
                <th>Status</th>
                <th>Backend</th>
                <th>Progress</th>
                <th>Device</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</body>
</html>
    "#, rows_html))
}

async fn devices_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query devices with their reputation scores
    let devices: Vec<(String, String, i32, bool, chrono::DateTime<chrono::Utc>, Option<f64>, Option<f64>)> = sqlx::query_as(
        r#"
        SELECT
            d.device_id,
            d.hostname,
            d.gpu_count,
            d.on_ac_power,
            d.last_seen,
            r.alpha,
            r.beta
        FROM devices d
        LEFT JOIN device_reputation r ON d.device_id = r.device_id
        ORDER BY d.last_seen DESC
        LIMIT 100
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let rows_html = if devices.is_empty() {
        "<tr><td colspan=\"6\" style=\"text-align: center; padding: 40px;\">No devices registered</td></tr>".to_string()
    } else {
        let mut html = String::new();
        for (device_id, hostname, gpu_count, on_ac_power, last_seen, alpha, beta) in devices {
            let power_class = if on_ac_power { "power-ac" } else { "power-battery" };
            let power_text = if on_ac_power { "AC Power" } else { "Battery" };

            // Calculate reputation score from Beta distribution
            let reputation = if let (Some(a), Some(b)) = (alpha, beta) {
                let score = a / (a + b);
                format!("{:.2}%", score * 100.0)
            } else {
                "N/A".to_string()
            };

            let time_ago = format_time_ago(last_seen);
            let is_active = chrono::Utc::now().signed_duration_since(last_seen).num_minutes() < 5;
            let status_color = if is_active { "#27ae60" } else { "#999" };

            html.push_str(&format!(
                r#"<tr>
                    <td><code>{}</code></td>
                    <td>{}</td>
                    <td>{} GPU{}</td>
                    <td class="{}">{}</td>
                    <td style="color: {}">{}</td>
                    <td>{}</td>
                </tr>"#,
                device_id,
                hostname,
                gpu_count,
                if gpu_count != 1 { "s" } else { "" },
                power_class,
                power_text,
                status_color,
                time_ago,
                reputation
            ));
        }
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - Devices</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c3e50;
            color: white;
        }}
        .power-ac {{ color: #27ae60; font-weight: bold; }}
        .power-battery {{ color: #e74c3c; font-weight: bold; }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Devices</h1>
    <p><a href="/">← Back to Dashboard</a></p>
    <table>
        <thead>
            <tr>
                <th>Device ID</th>
                <th>Hostname</th>
                <th>GPUs</th>
                <th>Power</th>
                <th>Last Seen</th>
                <th>Reputation</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</body>
</html>
    "#, rows_html))
}

async fn gpu_metrics_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query GPU information from devices
    #[derive(Debug)]
    struct GpuMetric {
        device_id: String,
        hostname: String,
        gpu_count: i32,
        on_ac_power: bool,
        last_seen: chrono::DateTime<chrono::Utc>,
        backend: Option<String>,
        vram_total_gb: Option<i32>,
    }

    let gpu_metrics: Vec<GpuMetric> = sqlx::query_as::<_, (String, String, i32, bool, chrono::DateTime<chrono::Utc>, Option<String>, Option<i32>)>(
        r#"
        SELECT
            device_id,
            hostname,
            gpu_count,
            on_ac_power,
            last_seen,
            backend,
            vram_total_gb
        FROM devices
        ORDER BY last_seen DESC
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|(device_id, hostname, gpu_count, on_ac_power, last_seen, backend, vram_total_gb)| {
        GpuMetric {
            device_id,
            hostname,
            gpu_count,
            on_ac_power,
            last_seen,
            backend,
            vram_total_gb,
        }
    })
    .collect();

    // Query active job assignments per device
    let job_assignments: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT assigned_device_id, COUNT(*)
        FROM job_shards
        WHERE status IN ('assigned', 'running') AND assigned_device_id IS NOT NULL
        GROUP BY assigned_device_id
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let job_map: std::collections::HashMap<String, i64> = job_assignments.into_iter().collect();

    // Generate GPU cards HTML
    let gpu_cards_html = if gpu_metrics.is_empty() {
        "<div class=\"empty-state\"><p>No GPUs registered in the cluster</p></div>".to_string()
    } else {
        let mut html = String::from("<div class=\"gpu-grid\">");
        for metric in &gpu_metrics {
            let is_active = chrono::Utc::now().signed_duration_since(metric.last_seen).num_minutes() < 5;
            let status_class = if is_active { "status-online" } else { "status-offline" };
            let status_text = if is_active { "Online" } else { "Offline" };
            let power_class = if metric.on_ac_power { "power-ac" } else { "power-battery" };
            let power_text = if metric.on_ac_power { "AC Power" } else { "Battery" };
            let backend_text = metric.backend.as_deref().unwrap_or("Unknown");
            let vram_text = metric.vram_total_gb.map(|v| format!("{} GB", v)).unwrap_or_else(|| "N/A".to_string());
            let active_jobs = job_map.get(&metric.device_id).copied().unwrap_or(0);
            let time_ago = format_time_ago(metric.last_seen);

            html.push_str(&format!(
                r#"
                <div class="gpu-card">
                    <div class="gpu-card-header">
                        <h3>{}</h3>
                        <span class="badge {}">{}</span>
                    </div>
                    <div class="gpu-card-body">
                        <div class="metric-row">
                            <span class="metric-label">Device ID:</span>
                            <span><code>{}</code></span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">GPUs:</span>
                            <span>{}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Backend:</span>
                            <span>{}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">VRAM:</span>
                            <span>{}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Power:</span>
                            <span class="{}">{}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Active Jobs:</span>
                            <span class="job-count">{}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Last Seen:</span>
                            <span>{}</span>
                        </div>
                    </div>
                </div>
                "#,
                metric.hostname,
                status_class,
                status_text,
                metric.device_id,
                metric.gpu_count,
                backend_text,
                vram_text,
                power_class,
                power_text,
                active_jobs,
                time_ago
            ));
        }
        html.push_str("</div>");
        html
    };

    // Calculate cluster-wide statistics
    let total_gpus: i32 = gpu_metrics.iter().map(|m| m.gpu_count).sum();
    let online_devices = gpu_metrics.iter().filter(|m| {
        chrono::Utc::now().signed_duration_since(m.last_seen).num_minutes() < 5
    }).count();
    let total_vram_gb: i32 = gpu_metrics.iter().filter_map(|m| m.vram_total_gb).sum();
    let cuda_gpus = gpu_metrics.iter().filter(|m| m.backend.as_deref() == Some("cuda")).map(|m| m.gpu_count).sum::<i32>();
    let metal_gpus = gpu_metrics.iter().filter(|m| m.backend.as_deref() == Some("metal")).map(|m| m.gpu_count).sum::<i32>();

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - GPU Metrics</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        nav {{
            margin-top: 20px;
        }}
        nav a {{
            color: #3498db;
            text-decoration: none;
            margin-right: 20px;
        }}
        nav a:hover {{
            text-decoration: underline;
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .gpu-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .gpu-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .gpu-card-header {{
            background: #34495e;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .gpu-card-header h3 {{
            margin: 0;
            font-size: 18px;
        }}
        .gpu-card-body {{
            padding: 20px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric-row:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #666;
            font-weight: 500;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-online {{
            background: #27ae60;
            color: white;
        }}
        .status-offline {{
            background: #95a5a6;
            color: white;
        }}
        .power-ac {{
            color: #27ae60;
            font-weight: bold;
        }}
        .power-battery {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .job-count {{
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: bold;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .empty-state {{
            background: white;
            padding: 60px;
            text-align: center;
            border-radius: 8px;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GPU Metrics Dashboard</h1>
            <p>Real-time GPU cluster monitoring</p>
            <nav>
                <a href="/">Dashboard</a>
                <a href="/jobs">Jobs</a>
                <a href="/devices">Devices</a>
                <a href="/gpu-metrics">GPU Metrics</a>
                <a href="/token-metrics">Token Metrics</a>
                <a href="/users">Users</a>
                <a href="/api-keys">API Keys</a>
                <a href="/metrics">Prometheus</a>
            </nav>
        </header>

        <div class="stats-bar">
            <div class="stat-card">
                <h3>Total GPUs</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>Online Devices</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>Total VRAM</h3>
                <div class="value">{} GB</div>
            </div>
            <div class="stat-card">
                <h3>CUDA GPUs</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>Metal GPUs</h3>
                <div class="value">{}</div>
            </div>
        </div>

        {}
    </div>
</body>
</html>
    "#,
        total_gpus,
        online_devices,
        total_vram_gb,
        cuda_gpus,
        metal_gpus,
        gpu_cards_html
    ))
}

async fn users_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query users from database
    let users: Vec<(Uuid, String, String, Option<String>, String, bool, chrono::DateTime<chrono::Utc>, Option<chrono::DateTime<chrono::Utc>>)> = sqlx::query_as(
        r#"
        SELECT id, username, email, full_name, role, is_active, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let rows_html = if users.is_empty() {
        "<tr><td colspan=\"7\" style=\"text-align: center; padding: 40px;\">No users found</td></tr>".to_string()
    } else {
        let mut html = String::new();
        for (id, username, email, full_name, role, is_active, created_at, last_login) in users {
            let status_class = if is_active { "status-active" } else { "status-inactive" };
            let status_text = if is_active { "Active" } else { "Inactive" };
            let role_class = match role.as_str() {
                "admin" => "role-admin",
                "user" => "role-user",
                _ => "role-readonly",
            };
            let full_name_display = full_name.unwrap_or_else(|| "-".to_string());
            let last_login_display = last_login.map(format_time_ago).unwrap_or_else(|| "Never".to_string());
            let created_display = format_time_ago(created_at);

            html.push_str(&format!(
                r#"<tr>
                    <td><code>{}</code></td>
                    <td><strong>{}</strong></td>
                    <td>{}</td>
                    <td>{}</td>
                    <td><span class="badge {}">{}</span></td>
                    <td><span class="badge {}">{}</span></td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>"#,
                id,
                username,
                email,
                full_name_display,
                role_class,
                role,
                status_class,
                status_text,
                last_login_display,
                created_display
            ));
        }
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - User Management</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c3e50;
            color: white;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-active {{
            background: #27ae60;
            color: white;
        }}
        .status-inactive {{
            background: #95a5a6;
            color: white;
        }}
        .role-admin {{
            background: #e74c3c;
            color: white;
        }}
        .role-user {{
            background: #3498db;
            color: white;
        }}
        .role-readonly {{
            background: #95a5a6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>User Management</h1>
        <a href="/">← Back to Dashboard</a>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Email</th>
                <th>Full Name</th>
                <th>Role</th>
                <th>Status</th>
                <th>Last Login</th>
                <th>Created</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</body>
</html>
    "#, rows_html))
}

async fn token_metrics_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query token usage metrics
    let total_stats: (i64, i64, i64, i64) = sqlx::query_as(
        r#"
        SELECT
            COUNT(*) as total_requests,
            COALESCE(SUM(input_tokens), 0) as total_input,
            COALESCE(SUM(output_tokens), 0) as total_output,
            COALESCE(SUM(total_tokens), 0) as total_tokens
        FROM inference_metrics
        WHERE created_at > NOW() - INTERVAL '24 hours'
        "#
    )
    .fetch_one(&*state.db)
    .await
    .unwrap_or((0, 0, 0, 0));

    // Query per-model metrics
    let model_metrics: Vec<(String, i64, i64, i64, i64, f64, f64)> = sqlx::query_as(
        r#"
        SELECT
            model_id,
            COUNT(*) as request_count,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(total_tokens) as total_tokens,
            AVG(tokens_per_second) as avg_tps,
            AVG(generation_time_ms) as avg_time_ms
        FROM inference_metrics
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY model_id
        ORDER BY total_tokens DESC
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    // Query hourly request rate
    let hourly_rates: Vec<(chrono::DateTime<chrono::Utc>, i64)> = sqlx::query_as(
        r#"
        SELECT
            DATE_TRUNC('hour', created_at) as hour,
            COUNT(*) as request_count
        FROM inference_metrics
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour DESC
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let model_rows_html = if model_metrics.is_empty() {
        "<tr><td colspan=\"7\" style=\"text-align: center; padding: 40px;\">No metrics data available</td></tr>".to_string()
    } else {
        let mut html = String::new();
        for (model_id, requests, input_tokens, output_tokens, total_tokens, avg_tps, avg_time_ms) in model_metrics {
            html.push_str(&format!(
                r#"<tr>
                    <td><strong>{}</strong></td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{:.1}</td>
                    <td>{:.0} ms</td>
                </tr>"#,
                model_id,
                requests,
                input_tokens,
                output_tokens,
                total_tokens,
                avg_tps,
                avg_time_ms
            ));
        }
        html
    };

    let rate_chart_html = if hourly_rates.is_empty() {
        "<p>No request rate data available</p>".to_string()
    } else {
        let max_count = hourly_rates.iter().map(|(_, count)| count).max().unwrap_or(&1);
        let mut html = String::from("<div class=\"rate-chart\">");
        for (hour, count) in hourly_rates.iter().rev().take(24).rev() {
            let bar_height = ((*count as f64 / *max_count as f64) * 100.0) as i32;
            let hour_str = hour.format("%H:00").to_string();
            html.push_str(&format!(
                r#"
                <div class="rate-bar">
                    <div class="bar" style="height: {}%"></div>
                    <div class="label">{}</div>
                    <div class="count">{}</div>
                </div>
                "#,
                bar_height,
                hour_str,
                count
            ));
        }
        html.push_str("</div>");
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - Token Metrics</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c3e50;
            color: white;
        }}
        .rate-chart {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: flex-end;
            height: 200px;
            gap: 5px;
        }}
        .rate-bar {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            position: relative;
        }}
        .rate-bar .bar {{
            width: 100%;
            background: #3498db;
            border-radius: 4px 4px 0 0;
            min-height: 2px;
        }}
        .rate-bar .label {{
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }}
        .rate-bar .count {{
            position: absolute;
            top: -20px;
            font-size: 11px;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Token & Rate Metrics (Last 24 Hours)</h1>
        <a href="/">← Back to Dashboard</a>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <h3>Total Requests</h3>
            <div class="value">{}</div>
        </div>
        <div class="stat-card">
            <h3>Input Tokens</h3>
            <div class="value">{}</div>
        </div>
        <div class="stat-card">
            <h3>Output Tokens</h3>
            <div class="value">{}</div>
        </div>
        <div class="stat-card">
            <h3>Total Tokens</h3>
            <div class="value">{}</div>
        </div>
    </div>

    <h2>Request Rate (Hourly)</h2>
    {}

    <h2 style="margin-top: 30px;">Per-Model Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Requests</th>
                <th>Input Tokens</th>
                <th>Output Tokens</th>
                <th>Total Tokens</th>
                <th>Avg TPS</th>
                <th>Avg Time</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</body>
</html>
    "#,
        total_stats.0,
        total_stats.1,
        total_stats.2,
        total_stats.3,
        rate_chart_html,
        model_rows_html
    ))
}

async fn api_keys_handler(State(state): State<Arc<WebUiState>>) -> impl IntoResponse {
    // Query API keys from database
    let keys: Vec<(Uuid, String, String, String, Vec<String>, bool, Option<chrono::DateTime<chrono::Utc>>, Option<chrono::DateTime<chrono::Utc>>, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
        r#"
        SELECT
            k.id,
            u.username,
            k.key_prefix,
            k.name,
            k.scopes,
            k.is_active,
            k.last_used,
            k.expires_at,
            k.created_at
        FROM api_keys k
        JOIN users u ON k.user_id = u.id
        ORDER BY k.created_at DESC
        "#
    )
    .fetch_all(&*state.db)
    .await
    .unwrap_or_default();

    let rows_html = if keys.is_empty() {
        "<tr><td colspan=\"7\" style=\"text-align: center; padding: 40px;\">No API keys found</td></tr>".to_string()
    } else {
        let mut html = String::new();
        for (id, username, key_prefix, name, scopes, is_active, last_used, expires_at, created_at) in keys {
            let status_class = if is_active { "status-active" } else { "status-inactive" };
            let status_text = if is_active { "Active" } else { "Inactive" };
            let scopes_display = scopes.join(", ");
            let last_used_display = last_used.map(format_time_ago).unwrap_or_else(|| "Never".to_string());
            let expires_display = expires_at.map(|dt| {
                if dt < chrono::Utc::now() {
                    "Expired".to_string()
                } else {
                    format!("in {}", format_time_ago(dt))
                }
            }).unwrap_or_else(|| "Never".to_string());
            let created_display = format_time_ago(created_at);

            html.push_str(&format!(
                r#"<tr>
                    <td><code>{}</code></td>
                    <td><strong>{}</strong></td>
                    <td>{}</td>
                    <td><code>{}</code></td>
                    <td>{}</td>
                    <td><span class="badge {}">{}</span></td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>"#,
                id,
                username,
                name,
                key_prefix,
                scopes_display,
                status_class,
                status_text,
                last_used_display,
                expires_display,
                created_display
            ));
        }
        html
    };

    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CorpGrid - API Keys</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c3e50;
            color: white;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-active {{
            background: #27ae60;
            color: white;
        }}
        .status-inactive {{
            background: #95a5a6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>API Key Management</h1>
        <a href="/">← Back to Dashboard</a>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>User</th>
                <th>Name</th>
                <th>Key Prefix</th>
                <th>Scopes</th>
                <th>Status</th>
                <th>Last Used</th>
                <th>Expires</th>
                <th>Created</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</body>
</html>
    "#, rows_html))
}

async fn status_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "application/json")],
        r#"{"status":"ok","service":"corpgrid-scheduler","version":"0.1.0"}"#,
    )
}
