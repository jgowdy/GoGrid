// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use quinn::{Endpoint, ClientConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    Manager, State,
};
use tauri_plugin_shell::ShellExt;
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

mod config;
mod coordinator_client;
mod worker_process;

use config::Config;
use coordinator_client::CoordinatorClient;
use worker_process::WorkerProcess;

/// Application state
struct AppState {
    worker_status: Arc<RwLock<WorkerStatus>>,
    coordinator_client: Arc<RwLock<Option<CoordinatorClient>>>,
    worker_process: Arc<WorkerProcess>,
    config: Arc<RwLock<Config>>,
}

impl Default for AppState {
    fn default() -> Self {
        // Determine worker binary path
        let worker_binary = if cfg!(debug_assertions) {
            // In development, use the debug build
            "target/debug/corpgrid-scheduler"
        } else {
            // In production, look for the binary in the same directory as the tray app
            #[cfg(target_os = "macos")]
            {
                "../MacOS/corpgrid-scheduler"
            }
            #[cfg(target_os = "windows")]
            {
                "corpgrid-scheduler.exe"
            }
            #[cfg(target_os = "linux")]
            {
                "corpgrid-scheduler"
            }
        };

        Self {
            worker_status: Arc::new(RwLock::new(WorkerStatus::default())),
            coordinator_client: Arc::new(RwLock::new(None)),
            worker_process: Arc::new(WorkerProcess::new(worker_binary.to_string())),
            config: Arc::new(RwLock::new(Config::default())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerStatus {
    connected: bool,
    status: String,
    jobs_today: u64,
    earned_today: f64,
}

impl Default for WorkerStatus {
    fn default() -> Self {
        Self {
            connected: false,
            status: "Starting...".to_string(),
            jobs_today: 0,
            earned_today: 0.0,
        }
    }
}

#[tauri::command]
async fn get_status(state: State<'_, AppState>) -> Result<WorkerStatus, String> {
    let status = state.worker_status.read().await.clone();
    Ok(status)
}

#[tauri::command]
async fn pause_worker(state: State<'_, AppState>) -> Result<(), String> {
    info!("Pausing worker");

    // Stop the worker process
    state.worker_process
        .stop()
        .await
        .map_err(|e| format!("Failed to stop worker: {}", e))?;

    let mut status = state.worker_status.write().await;
    status.status = "Paused".to_string();
    Ok(())
}

#[tauri::command]
async fn resume_worker(state: State<'_, AppState>) -> Result<(), String> {
    info!("Resuming worker");

    // Start the worker process
    state.worker_process
        .start()
        .await
        .map_err(|e| format!("Failed to start worker: {}", e))?;

    let mut status = state.worker_status.write().await;
    status.status = "Running".to_string();
    Ok(())
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("GoGrid Worker starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState::default())
        .setup(|app| {
            // Load configuration
            let state: State<AppState> = app.state();
            let config = tauri::async_runtime::block_on(async {
                match Config::load() {
                    Ok(cfg) => {
                        if !cfg.is_configured() {
                            // First run - prompt for configuration
                            match Config::prompt_for_config().await {
                                Ok(cfg) => cfg,
                                Err(e) => {
                                    eprintln!("Configuration error: {}", e);
                                    return Err(e);
                                }
                            }
                        } else {
                            cfg
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to load configuration: {}", e);
                        return Err(e);
                    }
                }
            });

            let config = match config {
                Ok(cfg) => cfg,
                Err(e) => {
                    // Show error and exit
                    use tauri::api::dialog::blocking::MessageDialogBuilder;
                    MessageDialogBuilder::new("Configuration Error", &format!("Failed to load configuration: {}\n\nThe application will now exit.", e))
                        .show();
                    std::process::exit(1);
                }
            };

            // Store config in state
            tauri::async_runtime::block_on(async {
                *state.config.write().await = config;
            });

            // Create system tray
            let quit = MenuItemBuilder::with_id("quit", "Quit GoGrid").build(app)?;
            let pause = MenuItemBuilder::with_id("pause", "Pause Worker").build(app)?;
            let resume = MenuItemBuilder::with_id("resume", "Resume Worker").build(app)?;
            let settings = MenuItemBuilder::with_id("settings", "Settings...").build(app)?;
            let stats = MenuItemBuilder::with_id("stats", "Statistics...").build(app)?;
            let dashboard = MenuItemBuilder::with_id("dashboard", "Open Dashboard").build(app)?;

            let menu = MenuBuilder::new(app)
                .text("status", "Status: Starting...")
                .separator()
                .item(&pause)
                .item(&resume)
                .separator()
                .item(&settings)
                .item(&stats)
                .item(&dashboard)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("GoGrid Worker")
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "quit" => {
                        info!("Quit requested");
                        app.exit(0);
                    }
                    "pause" => {
                        info!("Pause requested");
                        let state: State<AppState> = app.state();
                        tauri::async_runtime::block_on(async {
                            if let Err(e) = state.worker_process.stop().await {
                                warn!("Failed to stop worker: {}", e);
                            }
                            let mut status = state.worker_status.write().await;
                            status.status = "Paused".to_string();
                        });
                    }
                    "resume" => {
                        info!("Resume requested");
                        let state: State<AppState> = app.state();
                        tauri::async_runtime::block_on(async {
                            if let Err(e) = state.worker_process.start().await {
                                warn!("Failed to start worker: {}", e);
                            }
                            let mut status = state.worker_status.write().await;
                            status.status = "Running".to_string();
                        });
                    }
                    "settings" => {
                        info!("Settings requested");
                        // TODO: Open settings window
                    }
                    "stats" => {
                        info!("Statistics requested");
                        // TODO: Open statistics window
                    }
                    "dashboard" => {
                        info!("Dashboard requested");
                        let _ = app.shell().open("https://bx.ee/dashboard", None);
                    }
                    _ => {}
                })
                .build(app)?;

            // Start background connection to coordinator
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                connect_to_coordinator(app_handle).await;
            });

            // Check for updates on startup
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                check_for_updates(app_handle).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            pause_worker,
            resume_worker
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn connect_to_coordinator(app: tauri::AppHandle) {
    let state: State<AppState> = app.state();

    // Get coordinator config
    let (host, port) = {
        let config = state.config.read().await;
        (config.coordinator.host.clone(), config.coordinator.port)
    };

    info!("Connecting to coordinator at {}:{}...", host, port);

    // Update status
    {
        let mut status = state.worker_status.write().await;
        status.status = "Connecting...".to_string();
    }

    // Create coordinator client
    let worker_id = Uuid::new_v4();
    let mut client = CoordinatorClient::new(worker_id);

    // Attempt to connect with retries
    loop {
        match client.connect_with_config(&host, port).await {
            Ok(_) => {
                info!("Successfully connected to coordinator");

                // Store client in state
                {
                    let mut client_lock = state.coordinator_client.write().await;
                    *client_lock = Some(client);
                }

                // Update status
                {
                    let mut status = state.worker_status.write().await;
                    status.connected = true;
                    status.status = "Connected - Idle".to_string();
                }

                break;
            }
            Err(e) => {
                warn!("Failed to connect to coordinator: {}. Retrying in 10 seconds...", e);

                // Update status
                {
                    let mut status = state.worker_status.write().await;
                    status.connected = false;
                    status.status = format!("Connection failed: {}. Retrying...", e);
                }

                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            }
        }
    }

    info!("Connected to coordinator");
}

async fn check_for_updates(app: tauri::AppHandle) {
    use tauri_plugin_updater::UpdaterExt;

    info!("Checking for updates...");

    // Wait a bit on startup to let coordinator connection establish first
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    loop {
        match app.updater() {
            Ok(updater) => {
                match updater.check().await {
                    Ok(Some(update)) => {
                        info!(
                            "Update available: {} -> {}",
                            update.current_version,
                            update.version
                        );

                        // Download and install the update
                        info!("Downloading update...");
                        match update.download_and_install(|_chunk_length, _content_length| {
                            // Progress callback - could update UI here
                        }, || {
                            // Download complete callback
                            info!("Update downloaded successfully");
                        }).await {
                            Ok(_) => {
                                info!("Update installed successfully, restart required");
                                // The app will restart automatically
                            }
                            Err(e) => {
                                warn!("Failed to download/install update: {}", e);
                            }
                        }
                    }
                    Ok(None) => {
                        info!("No updates available, running latest version");
                    }
                    Err(e) => {
                        warn!("Failed to check for updates: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to initialize updater: {}", e);
            }
        }

        // Check again in 24 hours
        tokio::time::sleep(tokio::time::Duration::from_secs(86400)).await;
    }
}
