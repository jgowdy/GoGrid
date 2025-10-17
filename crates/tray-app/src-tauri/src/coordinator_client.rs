use anyhow::{Context, Result};
use quinn::{ClientConfig, Endpoint};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

// Coordinator settings (must be configured via environment variables or config file)
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const RECONNECT_INTERVAL: Duration = Duration::from_secs(10);

fn get_coordinator_host() -> Option<String> {
    std::env::var("GOGRID_COORDINATOR_HOST").ok()
}

fn get_coordinator_port() -> Option<u16> {
    std::env::var("GOGRID_COORDINATOR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
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

pub struct CoordinatorClient {
    worker_id: Uuid,
    worker_info: Arc<RwLock<WorkerInfo>>,
    endpoint: Option<Endpoint>,
    connected: Arc<RwLock<bool>>,
}

impl CoordinatorClient {
    pub fn new(worker_id: Uuid) -> Self {
        let worker_info = WorkerInfo {
            worker_id,
            hostname: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            gpu_info: None, // Will be populated by detecting GPU
            available_vram_gb: 0.0,
            status: WorkerStatus::Idle,
        };

        Self {
            worker_id,
            worker_info: Arc::new(RwLock::new(worker_info)),
            endpoint: None,
            connected: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn connect_with_config(&mut self, host: &str, port: u16) -> Result<()> {
        info!("Connecting to coordinator at {}:{}...", host, port);

        // Create QUIC client configuration with relaxed certificate verification
        // In production, we should use proper certificate verification
        let crypto = rustls::ClientConfig::builder()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();

        let mut client_config = ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
                .context("Failed to create QUIC crypto config")?,
        ));

        // Configure transport
        let mut transport = quinn::TransportConfig::default();
        transport.keep_alive_interval(Some(Duration::from_secs(5)));
        transport.max_idle_timeout(Some(Duration::from_secs(60).try_into().unwrap()));
        client_config.transport_config(Arc::new(transport));

        // Create endpoint
        let mut endpoint = Endpoint::client(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            0,
        ))
        .context("Failed to create QUIC endpoint")?;
        endpoint.set_default_client_config(client_config);

        // Connect to coordinator
        let server_addr = match format!("{}:{}", host, port).parse::<SocketAddr>() {
            Ok(addr) => addr,
            Err(_) => {
                // If parsing fails, try DNS resolution
                tokio::net::lookup_host(format!("{}:{}", host, port))
                    .await
                    .context("Failed to resolve hostname")?
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Failed to resolve coordinator hostname"))?
            }
        };

        let connection = endpoint
            .connect(server_addr, host)
            .context("Failed to initiate connection")?
            .await
            .context("Failed to establish connection")?;

        info!("Connected to coordinator");

        self.endpoint = Some(endpoint);
        *self.connected.write().await = true;

        // Send registration message
        self.send_registration(&connection).await?;

        // Start heartbeat task
        self.start_heartbeat_task(connection.clone());

        Ok(())
    }

    async fn send_registration(&self, connection: &quinn::Connection) -> Result<()> {
        info!("Sending registration to coordinator...");

        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .context("Failed to open bidirectional stream")?;

        // Send registration message
        let worker_info = self.worker_info.read().await.clone();
        let message = ClientMessage::Register(worker_info);
        let message_bytes = bincode::serialize(&message)
            .context("Failed to serialize registration message")?;

        send.write_all(&message_bytes)
            .await
            .context("Failed to send registration")?;
        send.finish().context("Failed to finish send stream")?;

        // Wait for response
        let response_bytes = recv
            .read_to_end(1024)
            .await
            .context("Failed to read registration response")?;

        let response: ServerMessage = bincode::deserialize(&response_bytes)
            .context("Failed to deserialize registration response")?;

        match response {
            ServerMessage::Registered { worker_id } => {
                info!("Successfully registered with coordinator as worker {}", worker_id);
                Ok(())
            }
            _ => {
                warn!("Unexpected response during registration: {:?}", response);
                Ok(())
            }
        }
    }

    fn start_heartbeat_task(&self, connection: quinn::Connection) {
        let worker_id = self.worker_id;
        let worker_info = self.worker_info.clone();
        let connected = self.connected.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);

            loop {
                interval.tick().await;

                if !*connected.read().await {
                    info!("Not connected, stopping heartbeat");
                    break;
                }

                // Send heartbeat
                let status = worker_info.read().await.status.clone();
                let message = ClientMessage::Heartbeat { worker_id, status };

                match Self::send_message(&connection, &message).await {
                    Ok(_) => {
                        info!("Heartbeat sent successfully");
                    }
                    Err(e) => {
                        error!("Failed to send heartbeat: {}", e);
                        *connected.write().await = false;
                        break;
                    }
                }
            }

            info!("Heartbeat task stopped");
        });
    }

    async fn send_message(connection: &quinn::Connection, message: &ClientMessage) -> Result<()> {
        let (mut send, _recv) = connection
            .open_bi()
            .await
            .context("Failed to open stream")?;

        let message_bytes = bincode::serialize(message)
            .context("Failed to serialize message")?;

        send.write_all(&message_bytes)
            .await
            .context("Failed to send message")?;
        send.finish().context("Failed to finish stream")?;

        Ok(())
    }

    pub async fn update_status(&self, status: WorkerStatus) {
        let mut info = self.worker_info.write().await;
        info.status = status;
    }

    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    pub async fn disconnect(&mut self) {
        info!("Disconnecting from coordinator...");
        *self.connected.write().await = false;

        if let Some(endpoint) = self.endpoint.take() {
            endpoint.close(0u32.into(), b"client disconnect");
        }
    }
}

impl Drop for CoordinatorClient {
    fn drop(&mut self) {
        if let Some(endpoint) = self.endpoint.take() {
            endpoint.close(0u32.into(), b"client dropped");
        }
    }
}

