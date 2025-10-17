use anyhow::Result;
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub full_name: Option<String>,
    pub role: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: Uuid,
    pub user_id: Uuid,
    pub key_prefix: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub is_active: bool,
    pub last_used: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: String,
    pub full_name: Option<String>,
    pub role: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub role: Option<String>,
    pub is_active: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub scopes: Vec<String>,
    pub expires_in_days: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CreateApiKeyResponse {
    pub api_key: String,
    pub key_info: ApiKey,
}

pub struct UserManager {
    db: PgPool,
}

impl UserManager {
    pub fn new(db: PgPool) -> Self {
        Self { db }
    }

    pub async fn create_user(&self, req: CreateUserRequest) -> Result<User> {
        let password_hash = hash(&req.password, DEFAULT_COST)?;

        let user = sqlx::query_as::<_, (Uuid, String, String, String, Option<String>, String, bool, DateTime<Utc>, DateTime<Utc>, Option<DateTime<Utc>>)>(
            r#"
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, username, email, password_hash, full_name, role, is_active, created_at, updated_at, last_login
            "#
        )
        .bind(&req.username)
        .bind(&req.email)
        .bind(&password_hash)
        .bind(&req.full_name)
        .bind(&req.role)
        .fetch_one(&self.db)
        .await?;

        Ok(User {
            id: user.0,
            username: user.1,
            email: user.2,
            password_hash: user.3,
            full_name: user.4,
            role: user.5,
            is_active: user.6,
            created_at: user.7,
            updated_at: user.8,
            last_login: user.9,
        })
    }

    pub async fn get_user_by_id(&self, user_id: Uuid) -> Result<Option<User>> {
        let user = sqlx::query_as::<_, (Uuid, String, String, String, Option<String>, String, bool, DateTime<Utc>, DateTime<Utc>, Option<DateTime<Utc>>)>(
            r#"
            SELECT id, username, email, password_hash, full_name, role, is_active, created_at, updated_at, last_login
            FROM users
            WHERE id = $1
            "#
        )
        .bind(user_id)
        .fetch_optional(&self.db)
        .await?;

        Ok(user.map(|u| User {
            id: u.0,
            username: u.1,
            email: u.2,
            password_hash: u.3,
            full_name: u.4,
            role: u.5,
            is_active: u.6,
            created_at: u.7,
            updated_at: u.8,
            last_login: u.9,
        }))
    }

    pub async fn get_user_by_username(&self, username: &str) -> Result<Option<User>> {
        let user = sqlx::query_as::<_, (Uuid, String, String, String, Option<String>, String, bool, DateTime<Utc>, DateTime<Utc>, Option<DateTime<Utc>>)>(
            r#"
            SELECT id, username, email, password_hash, full_name, role, is_active, created_at, updated_at, last_login
            FROM users
            WHERE username = $1
            "#
        )
        .bind(username)
        .fetch_optional(&self.db)
        .await?;

        Ok(user.map(|u| User {
            id: u.0,
            username: u.1,
            email: u.2,
            password_hash: u.3,
            full_name: u.4,
            role: u.5,
            is_active: u.6,
            created_at: u.7,
            updated_at: u.8,
            last_login: u.9,
        }))
    }

    pub async fn list_users(&self) -> Result<Vec<User>> {
        let users = sqlx::query_as::<_, (Uuid, String, String, String, Option<String>, String, bool, DateTime<Utc>, DateTime<Utc>, Option<DateTime<Utc>>)>(
            r#"
            SELECT id, username, email, password_hash, full_name, role, is_active, created_at, updated_at, last_login
            FROM users
            ORDER BY created_at DESC
            "#
        )
        .fetch_all(&self.db)
        .await?;

        Ok(users.into_iter().map(|u| User {
            id: u.0,
            username: u.1,
            email: u.2,
            password_hash: u.3,
            full_name: u.4,
            role: u.5,
            is_active: u.6,
            created_at: u.7,
            updated_at: u.8,
            last_login: u.9,
        }).collect())
    }

    pub async fn update_user(&self, user_id: Uuid, req: UpdateUserRequest) -> Result<User> {
        let user = sqlx::query_as::<_, (Uuid, String, String, String, Option<String>, String, bool, DateTime<Utc>, DateTime<Utc>, Option<DateTime<Utc>>)>(
            r#"
            UPDATE users
            SET
                email = COALESCE($2, email),
                full_name = COALESCE($3, full_name),
                role = COALESCE($4, role),
                is_active = COALESCE($5, is_active),
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, username, email, password_hash, full_name, role, is_active, created_at, updated_at, last_login
            "#
        )
        .bind(user_id)
        .bind(req.email)
        .bind(req.full_name)
        .bind(req.role)
        .bind(req.is_active)
        .fetch_one(&self.db)
        .await?;

        Ok(User {
            id: user.0,
            username: user.1,
            email: user.2,
            password_hash: user.3,
            full_name: user.4,
            role: user.5,
            is_active: user.6,
            created_at: user.7,
            updated_at: user.8,
            last_login: user.9,
        })
    }

    pub async fn delete_user(&self, user_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(user_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    pub async fn verify_password(&self, username: &str, password: &str) -> Result<Option<User>> {
        if let Some(user) = self.get_user_by_username(username).await? {
            if user.is_active && verify(password, &user.password_hash)? {
                // Update last_login
                sqlx::query("UPDATE users SET last_login = NOW() WHERE id = $1")
                    .bind(user.id)
                    .execute(&self.db)
                    .await?;
                return Ok(Some(user));
            }
        }
        Ok(None)
    }

    pub async fn create_api_key(&self, user_id: Uuid, req: CreateApiKeyRequest) -> Result<CreateApiKeyResponse> {
        use rand::Rng;

        // Generate API key: cgk_<random_32_chars>
        let random_part: String = rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(32)
            .map(char::from)
            .collect();
        let api_key = format!("cgk_{}", random_part);

        let key_prefix = api_key.chars().take(12).collect::<String>();
        let key_hash = hash(&api_key, DEFAULT_COST)?;

        let expires_at = req.expires_in_days.map(|days| {
            Utc::now() + chrono::Duration::days(days)
        });

        let key_info = sqlx::query_as::<_, (Uuid, Uuid, String, String, Vec<String>, bool, Option<DateTime<Utc>>, Option<DateTime<Utc>>, DateTime<Utc>)>(
            r#"
            INSERT INTO api_keys (user_id, key_hash, key_prefix, name, scopes, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, user_id, key_prefix, name, scopes, is_active, last_used, expires_at, created_at
            "#
        )
        .bind(user_id)
        .bind(&key_hash)
        .bind(&key_prefix)
        .bind(&req.name)
        .bind(&req.scopes)
        .bind(expires_at)
        .fetch_one(&self.db)
        .await?;

        Ok(CreateApiKeyResponse {
            api_key,
            key_info: ApiKey {
                id: key_info.0,
                user_id: key_info.1,
                key_prefix: key_info.2,
                name: key_info.3,
                scopes: key_info.4,
                is_active: key_info.5,
                last_used: key_info.6,
                expires_at: key_info.7,
                created_at: key_info.8,
            },
        })
    }

    pub async fn list_api_keys(&self, user_id: Uuid) -> Result<Vec<ApiKey>> {
        let keys = sqlx::query_as::<_, (Uuid, Uuid, String, String, Vec<String>, bool, Option<DateTime<Utc>>, Option<DateTime<Utc>>, DateTime<Utc>)>(
            r#"
            SELECT id, user_id, key_prefix, name, scopes, is_active, last_used, expires_at, created_at
            FROM api_keys
            WHERE user_id = $1
            ORDER BY created_at DESC
            "#
        )
        .bind(user_id)
        .fetch_all(&self.db)
        .await?;

        Ok(keys.into_iter().map(|k| ApiKey {
            id: k.0,
            user_id: k.1,
            key_prefix: k.2,
            name: k.3,
            scopes: k.4,
            is_active: k.5,
            last_used: k.6,
            expires_at: k.7,
            created_at: k.8,
        }).collect())
    }

    pub async fn revoke_api_key(&self, key_id: Uuid, user_id: Uuid) -> Result<()> {
        sqlx::query("UPDATE api_keys SET is_active = false WHERE id = $1 AND user_id = $2")
            .bind(key_id)
            .bind(user_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    pub async fn verify_api_key(&self, api_key: &str) -> Result<Option<(User, ApiKey)>> {
        // Get all active API keys (we need to check hash)
        let keys = sqlx::query_as::<_, (Uuid, Uuid, String, String, String, Vec<String>, bool, Option<DateTime<Utc>>, Option<DateTime<Utc>>, DateTime<Utc>)>(
            r#"
            SELECT id, user_id, key_hash, key_prefix, name, scopes, is_active, last_used, expires_at, created_at
            FROM api_keys
            WHERE is_active = true
            AND (expires_at IS NULL OR expires_at > NOW())
            "#
        )
        .fetch_all(&self.db)
        .await?;

        for key_data in keys {
            if verify(api_key, &key_data.2).unwrap_or(false) {
                // Found matching key, get user
                if let Some(user) = self.get_user_by_id(key_data.1).await? {
                    if user.is_active {
                        // Update last_used
                        sqlx::query("UPDATE api_keys SET last_used = NOW() WHERE id = $1")
                            .bind(key_data.0)
                            .execute(&self.db)
                            .await?;

                        let api_key = ApiKey {
                            id: key_data.0,
                            user_id: key_data.1,
                            key_prefix: key_data.3,
                            name: key_data.4,
                            scopes: key_data.5,
                            is_active: key_data.6,
                            last_used: key_data.7,
                            expires_at: key_data.8,
                            created_at: key_data.9,
                        };
                        return Ok(Some((user, api_key)));
                    }
                }
            }
        }

        Ok(None)
    }

    pub async fn log_audit(&self, user_id: Option<Uuid>, api_key_id: Option<Uuid>, action: &str, resource_type: Option<&str>, resource_id: Option<&str>, details: Option<serde_json::Value>, ip_address: Option<&str>, user_agent: Option<&str>) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO audit_log (user_id, api_key_id, action, resource_type, resource_id, details, ip_address, user_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#
        )
        .bind(user_id)
        .bind(api_key_id)
        .bind(action)
        .bind(resource_type)
        .bind(resource_id)
        .bind(details)
        .bind(ip_address)
        .bind(user_agent)
        .execute(&self.db)
        .await?;
        Ok(())
    }
}
