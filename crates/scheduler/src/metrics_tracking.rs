use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetric {
    pub model_id: String,
    pub user_id: Option<Uuid>,
    pub api_key_id: Option<Uuid>,
    pub request_id: Option<String>,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
    pub generation_time_ms: i64,
    pub tokens_per_second: f32,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetric {
    pub job_id: String,
    pub shard_index: i32,
    pub device_id: Option<String>,
    pub backend: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_seconds: Option<f32>,
    pub gpu_utilization_avg: Option<f32>,
    pub vram_used_gb: Option<f32>,
    pub power_consumption_watts: Option<f32>,
    pub temperature_celsius: Option<f32>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub model_id: String,
    pub total_requests: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_tokens: i64,
    pub avg_tokens_per_second: f32,
    pub avg_generation_time_ms: f32,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub day: DateTime<Utc>,
}

pub struct MetricsTracker {
    db: PgPool,
}

impl MetricsTracker {
    pub fn new(db: PgPool) -> Self {
        Self { db }
    }

    pub async fn record_inference_metric(&self, metric: InferenceMetric) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO inference_metrics (
                model_id, user_id, api_key_id, request_id,
                input_tokens, output_tokens, total_tokens,
                generation_time_ms, tokens_per_second,
                success, error_message
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(&metric.model_id)
        .bind(metric.user_id)
        .bind(metric.api_key_id)
        .bind(&metric.request_id)
        .bind(metric.input_tokens)
        .bind(metric.output_tokens)
        .bind(metric.total_tokens)
        .bind(metric.generation_time_ms)
        .bind(metric.tokens_per_second)
        .bind(metric.success)
        .bind(&metric.error_message)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    pub async fn record_job_metric(&self, metric: JobMetric) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO job_metrics (
                job_id, shard_index, device_id, backend,
                start_time, end_time, duration_seconds,
                gpu_utilization_avg, vram_used_gb,
                power_consumption_watts, temperature_celsius, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            "#
        )
        .bind(&metric.job_id)
        .bind(metric.shard_index)
        .bind(&metric.device_id)
        .bind(&metric.backend)
        .bind(metric.start_time)
        .bind(metric.end_time)
        .bind(metric.duration_seconds)
        .bind(metric.gpu_utilization_avg)
        .bind(metric.vram_used_gb)
        .bind(metric.power_consumption_watts)
        .bind(metric.temperature_celsius)
        .bind(&metric.status)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    pub async fn get_metrics_summary(&self, model_id: Option<&str>, days: i32) -> Result<Vec<MetricsSummary>> {
        let summaries = if let Some(mid) = model_id {
            sqlx::query_as::<_, (String, i64, i64, i64, i64, f64, f64, i64, i64, DateTime<Utc>)>(
                r#"
                SELECT
                    model_id,
                    COUNT(*) as total_requests,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(tokens_per_second) as avg_tokens_per_second,
                    AVG(generation_time_ms) as avg_generation_time_ms,
                    COUNT(*) FILTER (WHERE success = true) as successful_requests,
                    COUNT(*) FILTER (WHERE success = false) as failed_requests,
                    DATE_TRUNC('day', created_at) as day
                FROM inference_metrics
                WHERE model_id = $1 AND created_at > NOW() - INTERVAL '1 day' * $2
                GROUP BY model_id, DATE_TRUNC('day', created_at)
                ORDER BY day DESC
                "#
            )
            .bind(mid)
            .bind(days)
            .fetch_all(&self.db)
            .await?
        } else {
            sqlx::query_as::<_, (String, i64, i64, i64, i64, f64, f64, i64, i64, DateTime<Utc>)>(
                r#"
                SELECT
                    model_id,
                    COUNT(*) as total_requests,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(tokens_per_second) as avg_tokens_per_second,
                    AVG(generation_time_ms) as avg_generation_time_ms,
                    COUNT(*) FILTER (WHERE success = true) as successful_requests,
                    COUNT(*) FILTER (WHERE success = false) as failed_requests,
                    DATE_TRUNC('day', created_at) as day
                FROM inference_metrics
                WHERE created_at > NOW() - INTERVAL '1 day' * $1
                GROUP BY model_id, DATE_TRUNC('day', created_at)
                ORDER BY day DESC
                "#
            )
            .bind(days)
            .fetch_all(&self.db)
            .await?
        };

        Ok(summaries.into_iter().map(|s| MetricsSummary {
            model_id: s.0,
            total_requests: s.1,
            total_input_tokens: s.2,
            total_output_tokens: s.3,
            total_tokens: s.4,
            avg_tokens_per_second: s.5 as f32,
            avg_generation_time_ms: s.6 as f32,
            successful_requests: s.7,
            failed_requests: s.8,
            day: s.9,
        }).collect())
    }

    pub async fn get_hourly_request_rate(&self, hours: i32) -> Result<Vec<(DateTime<Utc>, i64)>> {
        let rates = sqlx::query_as::<_, (DateTime<Utc>, i64)>(
            r#"
            SELECT
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as request_count
            FROM inference_metrics
            WHERE created_at > NOW() - INTERVAL '1 hour' * $1
            GROUP BY DATE_TRUNC('hour', created_at)
            ORDER BY hour DESC
            "#
        )
        .bind(hours)
        .fetch_all(&self.db)
        .await?;

        Ok(rates)
    }

    pub async fn get_token_usage_by_user(&self, days: i32) -> Result<Vec<(Uuid, String, i64, i64, i64)>> {
        let usage = sqlx::query_as::<_, (Uuid, String, i64, i64, i64)>(
            r#"
            SELECT
                u.id,
                u.username,
                COUNT(*) as request_count,
                SUM(m.input_tokens) as total_input_tokens,
                SUM(m.output_tokens) as total_output_tokens
            FROM inference_metrics m
            JOIN users u ON m.user_id = u.id
            WHERE m.created_at > NOW() - INTERVAL '1 day' * $1
            GROUP BY u.id, u.username
            ORDER BY total_input_tokens + total_output_tokens DESC
            "#
        )
        .bind(days)
        .fetch_all(&self.db)
        .await?;

        Ok(usage)
    }

    pub async fn refresh_metrics_summary(&self) -> Result<()> {
        sqlx::query("SELECT refresh_metrics_summary()")
            .execute(&self.db)
            .await?;
        Ok(())
    }
}
