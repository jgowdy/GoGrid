use crate::db::DbPool;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone)]
pub struct ApiAuthState {
    pub db: DbPool,
}

/// Authenticated request context
#[derive(Clone, Debug)]
pub struct AuthContext {
    pub user_id: Uuid,
    pub api_key_id: Uuid,
    pub scopes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AuthErrorResponse {
    error: AuthErrorDetail,
}

#[derive(Debug, Serialize)]
struct AuthErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

/// API key authentication middleware
/// Extracts and validates API keys from Authorization header
pub async fn api_key_auth_middleware(
    State(state): State<Arc<ApiAuthState>>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Response {
    // Extract API key from Authorization header
    let api_key = match extract_api_key(&headers) {
        Some(key) => key,
        None => {
            return auth_error_response(
                StatusCode::UNAUTHORIZED,
                "Missing or invalid Authorization header. Expected: Authorization: Bearer <api_key>",
            );
        }
    };

    // Validate API key
    match validate_api_key(&state.db, api_key).await {
        Ok(Some(auth_context)) => {
            // Store auth context in request extensions for handlers to use
            request.extensions_mut().insert(auth_context);
            next.run(request).await
        }
        Ok(None) => {
            auth_error_response(
                StatusCode::UNAUTHORIZED,
                "Invalid API key",
            )
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to validate API key");
            auth_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
            )
        }
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            // Support both "Bearer <key>" and raw key
            if value.starts_with("Bearer ") {
                Some(value.trim_start_matches("Bearer "))
            } else if value.starts_with("cgk_") {
                Some(value)
            } else {
                None
            }
        })
}

async fn validate_api_key(db: &DbPool, api_key: &str) -> anyhow::Result<Option<AuthContext>> {
    use bcrypt::verify;
    use sqlx::Row;

    // Get all active API keys (using untyped query for Any database compatibility)
    let rows = sqlx::query(
        r#"
        SELECT k.id, k.user_id, k.key_hash, k.scopes, u.is_active
        FROM api_keys k
        JOIN users u ON k.user_id = u.id
        WHERE k.is_active = 1
        "#
    )
    .fetch_all(db)
    .await?;

    // Check each key hash
    for row in rows {
        let key_id: String = row.try_get("id")?;
        let user_id: String = row.try_get("user_id")?;
        let key_hash: String = row.try_get("key_hash")?;
        let scopes_json: String = row.try_get("scopes")?;
        let user_active: i32 = row.try_get("is_active")?;

        if user_active == 0 {
            continue;
        }

        // Parse scopes from JSON string
        let scopes: Vec<String> = serde_json::from_str(&scopes_json).unwrap_or_default();

        // Verify the API key against the stored hash
        if verify(api_key, &key_hash).unwrap_or(false) {
            // Update last_used timestamp
            let _ = sqlx::query("UPDATE api_keys SET last_used = datetime('now') WHERE id = ?")
                .bind(&key_id)
                .execute(db)
                .await;

            return Ok(Some(AuthContext {
                user_id: Uuid::parse_str(&user_id)?,
                api_key_id: Uuid::parse_str(&key_id)?,
                scopes,
            }));
        }
    }

    Ok(None)
}

fn auth_error_response(status: StatusCode, message: &str) -> Response {
    let error_response = AuthErrorResponse {
        error: AuthErrorDetail {
            message: message.to_string(),
            error_type: "authentication_error".to_string(),
        },
    };

    (status, Json(error_response)).into_response()
}

/// Extension trait for Request to get auth context
pub trait AuthContextExt {
    fn auth_context(&self) -> Option<&AuthContext>;
}

impl AuthContextExt for Request {
    fn auth_context(&self) -> Option<&AuthContext> {
        self.extensions().get::<AuthContext>()
    }
}
