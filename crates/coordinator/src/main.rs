/// GoGrid Coordinator Service - Central coordination server
use anyhow::{Context, Result};
use clap::Parser;
use quinn::{Endpoint, ServerConfig};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;
use axum::{
    Router,
    routing::get,
    extract::Path as AxumPath,
    response::{Json, IntoResponse, Html},
    http::StatusCode,
};
use tower_http::services::ServeDir;

mod worker_registry;
use worker_registry::WorkerRegistry;

#[derive(Parser, Debug)]
#[command(name = "gogrid-coordinator")]
struct Args {
    #[arg(short, long, default_value = "/opt/gogrid/config/coordinator.toml")]
    config: PathBuf,

    #[arg(long, default_value = "0.0.0.0")]
    bind_addr: String,

    #[arg(long, default_value = "8443")]
    port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: Uuid,
    pub hostname: String,
    pub gpu_info: Option<GpuInfo>,
    pub available_vram_gb: f64,
    pub status: WorkerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_gb: f64,
    pub compute_capability: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerStatus {
    Idle,
    Running,
    Paused,
    Error(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    Register(WorkerInfo),
    Heartbeat { worker_id: Uuid, status: WorkerStatus },
    JobComplete { job_id: Uuid, result: JobResult },
    JobFailed { job_id: Uuid, error: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerMessage {
    Registered { worker_id: Uuid },
    JobAssignment { job_id: Uuid, job_data: Vec<u8> },
    Pause,
    Resume,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobResult {
    pub output: Vec<u8>,
    pub execution_time_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .json()
        .init();

    let args = Args::parse();

    info!("GoGrid Coordinator starting on {}:{}...", args.bind_addr, args.port);

    // Create worker registry
    let registry = Arc::new(RwLock::new(WorkerRegistry::new()));

    // Generate self-signed certificate for development
    let cert = rcgen::generate_simple_self_signed(vec!["bx.ee".to_string(), "localhost".to_string()])
        .context("Failed to generate certificate")?;
    let cert_der = cert.serialize_der()
        .context("Failed to serialize certificate")?;
    let key_der = cert.serialize_private_key_der();

    let cert_chain = vec![rustls::pki_types::CertificateDer::from(cert_der)];
    let private_key = rustls::pki_types::PrivateKeyDer::try_from(key_der)
        .map_err(|e| anyhow::anyhow!("Failed to parse private key: {}", e))?;

    let server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, private_key)
        .context("Failed to create server crypto config")?;

    let mut server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .context("Failed to create QUIC server config")?,
    ));

    // Configure transport
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(std::time::Duration::from_secs(5)));
    transport.max_idle_timeout(Some(std::time::Duration::from_secs(60).try_into().unwrap()));
    server_config.transport_config(Arc::new(transport));

    // Bind to address
    let bind_addr: IpAddr = args.bind_addr.parse()
        .context("Invalid bind address")?;
    let socket_addr = SocketAddr::new(bind_addr, args.port);

    let endpoint = Endpoint::server(server_config, socket_addr)
        .context("Failed to create QUIC endpoint")?;

    info!("Listening on {}", socket_addr);
    info!("Ready to accept worker connections");

    // Start HTTP server for updates on port 8443 (HTTPS)
    let http_app = create_update_server();
    let http_addr = SocketAddr::new(bind_addr, args.port);

    tokio::spawn(async move {
        info!("Starting HTTP server on {} for updates", http_addr);
        // Note: In production, this should use HTTPS with proper certificates
        if let Err(e) = axum::serve(
            tokio::net::TcpListener::bind(http_addr).await.unwrap(),
            http_app
        ).await {
            error!("HTTP server error: {}", e);
        }
    });

    // Accept QUIC connections
    while let Some(incoming) = endpoint.accept().await {
        let registry = registry.clone();
        tokio::spawn(async move {
            match incoming.await {
                Ok(connection) => {
                    info!("New connection from {}", connection.remote_address());
                    if let Err(e) = handle_connection(connection, registry).await {
                        error!("Connection error: {}", e);
                    }
                }
                Err(e) => {
                    error!("Incoming connection error: {}", e);
                }
            }
        });
    }

    info!("Shutting down");
    Ok(())
}

async fn handle_connection(
    connection: quinn::Connection,
    registry: Arc<RwLock<WorkerRegistry>>,
) -> Result<()> {
    let remote = connection.remote_address();
    info!("Handling connection from {}", remote);

    // Accept bidirectional streams
    loop {
        match connection.accept_bi().await {
            Ok((mut send, mut recv)) => {
                // Read message
                let message_bytes = match recv.read_to_end(1024 * 1024).await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        warn!("Failed to read message: {}", e);
                        continue;
                    }
                };

                // Deserialize message
                let message: ClientMessage = match bincode::deserialize(&message_bytes) {
                    Ok(msg) => msg,
                    Err(e) => {
                        warn!("Failed to deserialize message: {}", e);
                        continue;
                    }
                };

                // Handle message
                let response = handle_message(message, &registry).await;

                // Send response if needed
                if let Some(response) = response {
                    let response_bytes = bincode::serialize(&response)
                        .context("Failed to serialize response")?;
                    send.write_all(&response_bytes)
                        .await
                        .context("Failed to send response")?;
                    send.finish().context("Failed to finish stream")?;
                }
            }
            Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                info!("Connection closed by peer: {}", remote);
                break;
            }
            Err(e) => {
                error!("Stream accept error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn handle_message(
    message: ClientMessage,
    registry: &Arc<RwLock<WorkerRegistry>>,
) -> Option<ServerMessage> {
    match message {
        ClientMessage::Register(worker_info) => {
            info!("Worker registration: {} ({})", worker_info.worker_id, worker_info.hostname);
            let worker_id = worker_info.worker_id;
            let mut reg = registry.write().await;
            reg.register_worker(worker_id, worker_info);
            Some(ServerMessage::Registered { worker_id })
        }
        ClientMessage::Heartbeat { worker_id, status } => {
            info!("Heartbeat from worker {}: {:?}", worker_id, status);
            let mut reg = registry.write().await;
            reg.update_worker_status(worker_id, status);
            None
        }
        ClientMessage::JobComplete { job_id, result } => {
            info!("Job {} completed: {} bytes, {} ms",
                job_id, result.output.len(), result.execution_time_ms);
            None
        }
        ClientMessage::JobFailed { job_id, error } => {
            warn!("Job {} failed: {}", job_id, error);
            None
        }
    }
}

fn create_update_server() -> Router {
    Router::new()
        .route("/", get(downloads_page))
        .route("/downloads", get(downloads_page))
        .route("/updates/:target/:version", get(get_update_manifest))
        .nest_service("/files", ServeDir::new("/opt/gogrid/updates"))
}

async fn downloads_page() -> Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GoGrid Worker Downloads</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 3rem;
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #8b9dc3;
            font-size: 1.2rem;
        }
        .version-badge {
            display: inline-block;
            background: rgba(0, 217, 255, 0.2);
            border: 1px solid #00d9ff;
            color: #00d9ff;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin-top: 1rem;
            font-weight: 600;
        }
        .downloads {
            display: grid;
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .download-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        .download-card:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: #00d9ff;
            transform: translateY(-2px);
        }
        .platform-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .platform-icon {
            font-size: 2rem;
        }
        .platform-name {
            font-size: 1.5rem;
            font-weight: 600;
        }
        .platform-info {
            color: #8b9dc3;
            margin-bottom: 1rem;
        }
        .download-button {
            display: inline-block;
            background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
            color: #1a1a2e;
            text-decoration: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .download-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4);
        }
        .file-info {
            display: inline-block;
            margin-left: 1rem;
            color: #8b9dc3;
            font-size: 0.9rem;
        }
        .coming-soon {
            opacity: 0.5;
        }
        .coming-soon .download-button {
            background: rgba(255, 255, 255, 0.1);
            color: #8b9dc3;
            cursor: not-allowed;
        }
        .coming-soon .download-button:hover {
            transform: none;
            box-shadow: none;
        }
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #8b9dc3;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GoGrid Worker</h1>
            <p class="subtitle">Distributed GPU Inference Client</p>
            <div class="version-badge">
                <span class="status-indicator"></span>
                Latest Version: 0.1.0
            </div>
        </header>

        <div class="downloads">
            <div class="download-card">
                <div class="platform-header">
                    <div class="platform-icon"></div>
                    <div class="platform-name">macOS</div>
                </div>
                <div class="platform-info">
                    For Apple Silicon (M1/M2/M3) and Intel Macs<br>
                    Requires macOS 10.15 (Catalina) or later
                </div>
                <a href="/files/GoGrid Worker_0.1.0_aarch64.dmg" class="download-button" download>
                    Download for macOS
                </a>
                <span class="file-info">DMG • 27 MB</span>
            </div>

            <div class="download-card coming-soon">
                <div class="platform-header">
                    <div class="platform-icon">🪟</div>
                    <div class="platform-name">Windows</div>
                </div>
                <div class="platform-info">
                    For Windows 10/11 (64-bit)<br>
                    GPU support: NVIDIA CUDA 11.8+
                </div>
                <a class="download-button">
                    Coming Soon
                </a>
                <span class="file-info">EXE • ~25 MB</span>
            </div>

            <div class="download-card coming-soon">
                <div class="platform-header">
                    <div class="platform-icon">🐧</div>
                    <div class="platform-name">Linux</div>
                </div>
                <div class="platform-info">
                    Ubuntu 20.04+, Debian, Fedora, Arch<br>
                    GPU support: NVIDIA CUDA 11.8+
                </div>
                <a class="download-button">
                    Coming Soon
                </a>
                <span class="file-info">AppImage • ~25 MB</span>
            </div>
        </div>

        <footer>
            <p>
                <strong>What is GoGrid Worker?</strong><br>
                GoGrid Worker is a system tray application that allows you to contribute your idle GPU
                to the GoGrid distributed inference network. Earn rewards while your computer is idle.
            </p>
            <p style="margin-top: 1rem;">
                <a href="https://bx.ee/dashboard" style="color: #00d9ff; text-decoration: none;">
                    Dashboard
                </a> •
                <a href="https://github.com/gogrid" style="color: #00d9ff; text-decoration: none;">
                    GitHub
                </a> •
                <a href="https://bx.ee/docs" style="color: #00d9ff; text-decoration: none;">
                    Documentation
                </a>
            </p>
            <p style="margin-top: 1rem; font-size: 0.9rem;">
                © 2025 GoGrid • Open Source • MIT License
            </p>
        </footer>
    </div>
</body>
</html>"#;

    Html(html.to_string())
}

#[derive(Serialize)]
struct UpdateManifest {
    version: String,
    notes: String,
    pub_date: String,
    platforms: std::collections::HashMap<String, PlatformUpdate>,
}

#[derive(Serialize)]
struct PlatformUpdate {
    signature: String,
    url: String,
}

async fn get_update_manifest(
    AxumPath((target, current_version)): AxumPath<(String, String)>,
) -> impl IntoResponse {
    info!("Update check from {} (current: {})", target, current_version);

    // For now, return no update available
    // In production, compare versions and return update info if newer exists
    let latest_version = "0.1.0";

    if current_version.as_str() >= latest_version {
        return (StatusCode::NO_CONTENT, "").into_response();
    }

    // Map Tauri target names to our platform names
    let platform_file = match target.as_str() {
        "darwin-aarch64" => "GoGrid_Worker_0.1.0_aarch64.app.tar.gz",
        "darwin-x86_64" => "GoGrid_Worker_0.1.0_x64.app.tar.gz",
        "linux-x86_64" => "GoGrid_Worker_0.1.0_amd64.AppImage.tar.gz",
        "windows-x86_64" => "GoGrid_Worker_0.1.0_x64-setup.nsis.zip",
        _ => {
            warn!("Unknown target: {}", target);
            return (StatusCode::NOT_FOUND, "Unknown platform").into_response();
        }
    };

    let mut platforms = std::collections::HashMap::new();
    platforms.insert(
        target.clone(),
        PlatformUpdate {
            signature: String::new(), // TODO: Add actual signature
            url: format!("https://bx.ee:8443/files/{}", platform_file),
        },
    );

    let manifest = UpdateManifest {
        version: latest_version.to_string(),
        notes: "Update to latest version with improvements".to_string(),
        pub_date: chrono::Utc::now().to_rfc3339(),
        platforms,
    };

    Json(manifest).into_response()
}
