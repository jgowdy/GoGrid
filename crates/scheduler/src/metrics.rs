use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct JobLabels {
    pub status: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceLabels {
    pub backend: String,
    pub site: String,
}

pub struct Metrics {
    registry: Arc<RwLock<Registry>>,

    // Job metrics
    pub jobs_total: Family<JobLabels, Counter>,
    pub job_attempts_total: Family<JobLabels, Counter>,
    pub job_duration_seconds: Family<JobLabels, Histogram>,

    // Placement metrics
    pub placement_latency_seconds: Histogram,
    pub placement_score: Histogram,

    // Heartbeat metrics
    pub heartbeats_total: Counter,
    pub heartbeat_timeouts_total: Counter,
    pub heartbeat_loss_percent: Gauge,

    // Lease metrics
    pub lease_expirations_total: Counter,
    pub active_leases: Gauge,

    // Replication metrics
    pub speculative_executions_total: Counter,
    pub quorum_time_seconds: Histogram,

    // Device metrics
    pub device_count: Family<DeviceLabels, Gauge>,
    pub device_reputation_score: Family<DeviceLabels, Gauge>,

    // Retry metrics
    pub retry_count: Histogram,
}

impl Metrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        // Job metrics
        let jobs_total = Family::<JobLabels, Counter>::default();
        registry.register(
            "corpgrid_jobs_total",
            "Total number of jobs",
            jobs_total.clone(),
        );

        let job_attempts_total = Family::<JobLabels, Counter>::default();
        registry.register(
            "corpgrid_job_attempts_total",
            "Total number of job attempts",
            job_attempts_total.clone(),
        );

        let job_duration_seconds = Family::<JobLabels, Histogram>::new_with_constructor(|| {
            Histogram::new(exponential_buckets(0.1, 2.0, 10))
        });
        registry.register(
            "corpgrid_job_duration_seconds",
            "Job execution duration",
            job_duration_seconds.clone(),
        );

        // Placement metrics
        let placement_latency_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 10));
        registry.register(
            "corpgrid_placement_latency_seconds",
            "Time to place a job on a device",
            placement_latency_seconds.clone(),
        );

        let placement_score = Histogram::new(exponential_buckets(0.01, 1.1, 20));
        registry.register(
            "corpgrid_placement_score",
            "Device placement score distribution",
            placement_score.clone(),
        );

        // Heartbeat metrics
        let heartbeats_total = Counter::default();
        registry.register(
            "corpgrid_heartbeats_total",
            "Total heartbeats received",
            heartbeats_total.clone(),
        );

        let heartbeat_timeouts_total = Counter::default();
        registry.register(
            "corpgrid_heartbeat_timeouts_total",
            "Total heartbeat timeouts",
            heartbeat_timeouts_total.clone(),
        );

        let heartbeat_loss_percent = Gauge::default();
        registry.register(
            "corpgrid_heartbeat_loss_percent",
            "Percentage of heartbeats lost",
            heartbeat_loss_percent.clone(),
        );

        // Lease metrics
        let lease_expirations_total = Counter::default();
        registry.register(
            "corpgrid_lease_expirations_total",
            "Total lease expirations",
            lease_expirations_total.clone(),
        );

        let active_leases = Gauge::default();
        registry.register(
            "corpgrid_active_leases",
            "Number of active leases",
            active_leases.clone(),
        );

        // Replication metrics
        let speculative_executions_total = Counter::default();
        registry.register(
            "corpgrid_speculative_executions_total",
            "Total speculative executions launched",
            speculative_executions_total.clone(),
        );

        let quorum_time_seconds = Histogram::new(exponential_buckets(0.1, 2.0, 15));
        registry.register(
            "corpgrid_quorum_time_seconds",
            "Time to reach quorum",
            quorum_time_seconds.clone(),
        );

        // Device metrics
        let device_count = Family::<DeviceLabels, Gauge>::default();
        registry.register(
            "corpgrid_device_count",
            "Number of registered devices",
            device_count.clone(),
        );

        let device_reputation_score = Family::<DeviceLabels, Gauge>::default();
        registry.register(
            "corpgrid_device_reputation_score",
            "Device reputation score (0-1)",
            device_reputation_score.clone(),
        );

        // Retry metrics
        let retry_count = Histogram::new(exponential_buckets(1.0, 2.0, 5));
        registry.register(
            "corpgrid_retry_count",
            "Number of retries per job",
            retry_count.clone(),
        );

        Self {
            registry: Arc::new(RwLock::new(registry)),
            jobs_total,
            job_attempts_total,
            job_duration_seconds,
            placement_latency_seconds,
            placement_score,
            heartbeats_total,
            heartbeat_timeouts_total,
            heartbeat_loss_percent,
            lease_expirations_total,
            active_leases,
            speculative_executions_total,
            quorum_time_seconds,
            device_count,
            device_reputation_score,
            retry_count,
        }
    }

    pub async fn export(&self) -> String {
        let registry = self.registry.read().await;
        let mut buffer = String::new();
        encode(&mut buffer, &registry).unwrap();
        buffer
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
