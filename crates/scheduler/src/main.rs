use anyhow::Result;
use axum::{routing::get, Router};
use corpgrid_scheduler::{
    create_openai_api_router, db, CasStorage, HeartbeatManager, Metrics,
    ModelHostingGrpcService, ModelHostingService, SchedulerService,
};
use std::sync::Arc;
use tokio::signal;
use tonic::transport::Server;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corpgrid_scheduler=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("CorpGrid Scheduler starting...");

    // Load configuration from environment
    // Defaults to SQLite at ./corpgrid.db if DATABASE_URL not set
    let database_url = db::default_database_url();
    let s3_bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "corpgrid".to_string());
    let bind_addr: std::net::SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".to_string())
        .parse()?;
    let metrics_addr: std::net::SocketAddr = std::env::var("METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9090".to_string())
        .parse()?;
    let web_ui_addr: std::net::SocketAddr = std::env::var("WEB_UI_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()?;
    let openai_api_addr: std::net::SocketAddr = std::env::var("OPENAI_API_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8000".to_string())
        .parse()?;

    // Initialize database
    info!("Connecting to database...");
    let db_pool = db::create_pool(&database_url).await?;

    info!("Running database migrations...");
    db::run_migrations(&db_pool).await?;

    // Initialize S3 storage
    info!("Initializing S3 storage...");
    let storage = Arc::new(CasStorage::new(s3_bucket).await?);

    // Initialize heartbeat manager
    let heartbeat_manager = Arc::new(HeartbeatManager::new());

    // Initialize metrics
    let metrics = Arc::new(Metrics::new());

    // Initialize model hosting service
    info!("Initializing model hosting service...");
    let model_hosting = Arc::new(ModelHostingService::new());

    // Create scheduler service
    let scheduler_service = SchedulerService::new(db_pool.clone(), storage.clone(), heartbeat_manager.clone(), model_hosting.clone());

    // Clone resources needed for the heartbeat checker
    let heartbeat_manager_for_checker = heartbeat_manager.clone();
    let db_pool_for_checker = db_pool.clone();

    // Start heartbeat expiration checker
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

            let expired = heartbeat_manager_for_checker.check_expired().await;
            if !expired.is_empty() {
                warn!(count = expired.len(), "Detected expired leases");

                // Reassign expired attempts directly using DB pool
                for attempt_id in expired {
                    info!(attempt_id = %attempt_id, "Reassigning expired attempt");

                    let result = sqlx::query(
                        r#"
                        UPDATE job_shards
                        SET status = 'pending', assigned_device_id = NULL, attempt_id = NULL
                        WHERE attempt_id = $1 AND status IN ('assigned', 'running')
                        "#
                    )
                    .bind(&attempt_id)
                    .execute(&db_pool_for_checker)
                    .await;

                    if let Err(e) = result {
                        error!(error = %e, attempt_id = %attempt_id, "Failed to reassign expired attempt");
                    }
                }
            }
        }
    });

    // Create gRPC server with scheduler service
    let scheduler_grpc = scheduler_service.into_grpc_server();

    let model_hosting_service = ModelHostingGrpcService::new(model_hosting.clone());
    let model_hosting_grpc = model_hosting_service.into_grpc_server();

    info!("Starting gRPC server on {}", bind_addr);
    info!("Starting metrics server on {}", metrics_addr);
    info!("Starting Web UI on {}", web_ui_addr);
    info!("Starting OpenAI API on {}", openai_api_addr);

    // Create metrics HTTP server
    let metrics_clone = metrics.clone();
    let metrics_app = Router::new().route(
        "/metrics",
        get(|| async move { metrics_clone.export().await }),
    );

    // Spawn metrics server
    let metrics_server = tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(&metrics_addr).await {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, "Failed to bind metrics server");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, metrics_app).await {
            error!(error = %e, "Metrics server error");
        }
    });

    // Spawn Web UI server
    // TODO: Re-enable after fixing DateTime queries for SQLite compatibility
    // let db_pool_for_web_ui = Arc::new(db_pool.clone());
    let web_ui_server = tokio::spawn(async move {
        info!("Web UI temporarily disabled - will be re-enabled after query compatibility fixes");
        tokio::time::sleep(tokio::time::Duration::from_secs(u64::MAX)).await;
        /*
        let listener = match tokio::net::TcpListener::bind(&web_ui_addr).await {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, "Failed to bind Web UI server");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, create_web_ui_router(db_pool_for_web_ui)).await {
            error!(error = %e, "Web UI server error");
        }
        */
    });

    // Create and spawn OpenAI API server
    let model_hosting_for_api = model_hosting.clone();
    let db_pool_for_api = db_pool.clone();
    let openai_api = create_openai_api_router(model_hosting_for_api, db_pool_for_api);
    let openai_api_server = tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(&openai_api_addr).await {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, "Failed to bind OpenAI API server");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, openai_api).await {
            error!(error = %e, "OpenAI API server error");
        }
    });

    // Run gRPC server with graceful shutdown
    let grpc_server_task = tokio::spawn(async move {
        Server::builder()
            .add_service(scheduler_grpc)
            .add_service(model_hosting_grpc)
            .serve_with_shutdown(bind_addr, async {
                signal::ctrl_c().await.ok();
                info!("Shutdown signal received");
            })
            .await
    });

    // Wait for any to complete
    tokio::select! {
        _ = grpc_server_task => {
            info!("gRPC server stopped");
        }
        _ = metrics_server => {
            info!("Metrics server stopped");
        }
        _ = web_ui_server => {
            info!("Web UI server stopped");
        }
        _ = openai_api_server => {
            info!("OpenAI API server stopped");
        }
    }

    info!("Scheduler shutdown complete");
    Ok(())
}
