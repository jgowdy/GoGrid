use anyhow::Result;
use corpgrid_common::PowerStatus;
use corpgrid_proto::scheduler_client::SchedulerClient;
use corpgrid_proto::*;
use tonic::transport::Channel;
use tracing::info;

use crate::device_info::collect_device_info;

/// gRPC client for scheduler communication
pub struct AgentClient {
    client: SchedulerClient<Channel>,
    device_id: String,
}

impl AgentClient {
    pub async fn connect(scheduler_url: String) -> Result<Self> {
        let client = SchedulerClient::connect(scheduler_url).await?;

        let device_info = collect_device_info()?;
        let device_id = device_info.device_id.clone();

        info!(device_id = %device_id, "Connected to scheduler");

        Ok(Self { client, device_id })
    }

    pub async fn register(&mut self, power_status: PowerStatus, public_key: Vec<u8>) -> Result<bool> {
        let device_info = collect_device_info()?;

        let request = RegisterAgentRequest {
            device_id: self.device_id.clone(),
            device_info: Some(convert_device_info(&device_info)),
            power_status: Some(convert_power_status(&power_status)),
            public_key,
        };

        let response = self.client.register_agent(request).await?;
        let resp = response.into_inner();

        info!(
            device_id = %self.device_id,
            accepted = resp.accepted,
            message = %resp.message,
            "Registration response"
        );

        Ok(resp.accepted)
    }

    pub async fn poll_jobs(&mut self, power_status: PowerStatus) -> Result<Vec<JobAssignment>> {
        let device_info = collect_device_info()?;

        let capabilities = DeviceCapabilities {
            backends: device_info
                .gpus
                .iter()
                .map(|g| g.backend.to_string())
                .collect(),
            max_vram_bytes: device_info
                .gpus
                .iter()
                .map(|g| g.vram_bytes)
                .max()
                .unwrap_or(0),
            supported_features: vec![],
        };

        let request = PollJobsRequest {
            device_id: self.device_id.clone(),
            capabilities: Some(capabilities),
            power_status: Some(convert_power_status(&power_status)),
        };

        let response = self.client.poll_jobs(request).await?;
        let resp = response.into_inner();

        Ok(resp.assignments)
    }

    pub async fn send_heartbeat(
        &mut self,
        job_id: String,
        shard_id: String,
        power_status: PowerStatus,
        progress: JobProgress,
    ) -> Result<HeartbeatResponse> {
        let request = HeartbeatRequest {
            device_id: self.device_id.clone(),
            job_id,
            shard_id,
            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
            progress: Some(progress),
            power_status: Some(convert_power_status(&power_status)),
        };

        let response = self.client.heartbeat(request).await?;
        Ok(response.into_inner())
    }

    pub async fn submit_result(
        &mut self,
        job_id: String,
        shard_id: String,
        result_hash: Vec<u8>,
        result_s3_key: String,
        signature: Vec<u8>,
    ) -> Result<SubmitResultResponse> {
        let request = SubmitResultRequest {
            device_id: self.device_id.clone(),
            job_id,
            shard_id,
            result_hash,
            result_s3_key,
            signature,
            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
        };

        let response = self.client.submit_result(request).await?;
        Ok(response.into_inner())
    }

    pub async fn report_checkpoint(
        &mut self,
        job_id: String,
        shard_id: String,
        checkpoint_s3_key: String,
        checkpoint_hash: Vec<u8>,
    ) -> Result<()> {
        let request = ReportCheckpointRequest {
            device_id: self.device_id.clone(),
            job_id,
            shard_id,
            checkpoint_s3_key,
            checkpoint_hash,
            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
        };

        self.client.report_checkpoint(request).await?;
        Ok(())
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }
}

fn convert_device_info(info: &corpgrid_common::DeviceInfo) -> DeviceInfo {
    DeviceInfo {
        hostname: info.hostname.clone(),
        os: info.os.clone(),
        arch: info.arch.clone(),
        gpus: info.gpus.iter().map(convert_gpu_info).collect(),
        memory_bytes: info.memory_bytes,
        cpu_cores: info.cpu_cores,
    }
}

fn convert_gpu_info(info: &corpgrid_common::GpuInfo) -> GpuInfo {
    GpuInfo {
        name: info.name.clone(),
        backend: info.backend.to_string(),
        vram_bytes: info.vram_bytes,
        driver_version: info.driver_version.clone(),
        compute_capability: info.compute_capability.clone().unwrap_or_default(),
    }
}

fn convert_power_status(status: &PowerStatus) -> corpgrid_proto::PowerStatus {
    corpgrid_proto::PowerStatus {
        on_ac_power: status.on_ac_power,
        battery_percent: status.battery_percent as u32,
    }
}
