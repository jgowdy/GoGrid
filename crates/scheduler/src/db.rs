use anyhow::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Pool, Sqlite};
use std::str::FromStr;

pub type DbPool = Pool<Sqlite>;

/// Create a database pool from a connection string
/// Currently supports SQLite (PostgreSQL support TODO)
///
/// Examples:
/// - SQLite: "sqlite://corpgrid.db" or "sqlite::memory:"
pub async fn create_pool(database_url: &str) -> Result<DbPool> {
    let options = SqliteConnectOptions::from_str(database_url)?
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(50)
        .connect_with(options)
        .await?;

    Ok(pool)
}

/// Get the default database URL
/// Defaults to SQLite if DATABASE_URL is not set
pub fn default_database_url() -> String {
    std::env::var("DATABASE_URL").unwrap_or_else(|_| {
        // Default to SQLite in the current directory
        let db_path = std::env::var("CORPGRID_DB_PATH")
            .unwrap_or_else(|_| "./corpgrid.db".to_string());
        format!("sqlite://{}", db_path)
    })
}

pub async fn run_migrations(pool: &DbPool) -> Result<()> {
    sqlx::migrate!("./migrations")
        .run(pool)
        .await?;
    Ok(())
}
