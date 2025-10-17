use crate::db::DbPool;
use crate::model_hosting::ModelHostingService;
use crate::api_auth::{ApiAuthState, api_key_auth_middleware};
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    middleware,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{error, info};

/// OpenAI-compatible API for LLM inference
/// Implements /v1/chat/completions, /v1/completions, and /v1/models endpoints
///
/// IMPORTANT: This API requires API key authentication for consumers
/// This is different from agent authentication (which uses Ed25519 keys for BFT)
pub fn create_openai_api_router(model_hosting: Arc<ModelHostingService>, db: DbPool) -> Router {
    let auth_state = Arc::new(ApiAuthState { db });

    Router::new()
        // Public endpoints (no auth required)
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model))
        // Protected endpoints (require API key auth)
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(middleware::from_fn_with_state(auth_state, api_key_auth_middleware))
        .with_state(model_hosting)
}

// ============================================================================
// OpenAI API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: Option<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    logprobs: Option<serde_json::Value>,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

// ============================================================================
// API Handlers
// ============================================================================

async fn list_models(
    State(model_hosting): State<Arc<ModelHostingService>>,
) -> Result<Json<ModelsResponse>, ApiError> {
    info!("GET /v1/models");

    let statuses = model_hosting.get_model_status().await;

    let models: Vec<ModelInfo> = statuses
        .into_iter()
        .map(|status| ModelInfo {
            id: status.model_id.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "corpgrid".to_string(),
        })
        .collect();

    Ok(Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    }))
}

async fn get_model(
    State(model_hosting): State<Arc<ModelHostingService>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfo>, ApiError> {
    info!(model_id = %model_id, "GET /v1/models/{}", model_id);

    let statuses = model_hosting.get_model_status().await;

    let model = statuses
        .iter()
        .find(|s| s.model_id == model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model {} not found", model_id)))?;

    Ok(Json(ModelInfo {
        id: model.model_id.clone(),
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "corpgrid".to_string(),
    }))
}

async fn completions(
    State(model_hosting): State<Arc<ModelHostingService>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    info!(model = %request.model, "POST /v1/completions");

    let prompt = request.prompt.as_deref().unwrap_or("");
    let max_tokens = request.max_tokens.unwrap_or(100);

    // Get tokenizer
    let tokenizer = model_hosting
        .get_tokenizer(&request.model)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Model {} not loaded", request.model)))?;

    // Tokenize input
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| ApiError::InternalError(format!("Tokenization failed: {}", e)))?;
    let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

    if request.stream {
        // Streaming response
        let stream = create_completion_stream(
            model_hosting,
            request.model.clone(),
            input_tokens.clone(),
            max_tokens,
            request.temperature,
            request.top_p,
            tokenizer,
        );

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming response
        let output_tokens = model_hosting
            .submit_inference(
                request.model.clone(),
                input_tokens.clone(),
                max_tokens as usize,
                request.temperature,
                request.top_p,
            )
            .await
            .map_err(|e| ApiError::InferenceError(e.to_string()))?;

        // Detokenize
        let output_text = tokenizer
            .decode(&output_tokens[input_tokens.len()..], true)
            .map_err(|e| ApiError::InternalError(format!("Detokenization failed: {}", e)))?;

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model,
            choices: vec![CompletionChoice {
                text: output_text,
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: input_tokens.len() as u32,
                completion_tokens: (output_tokens.len() - input_tokens.len()) as u32,
                total_tokens: output_tokens.len() as u32,
            },
        };

        Ok(Json(response).into_response())
    }
}

async fn chat_completions(
    State(model_hosting): State<Arc<ModelHostingService>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    info!(model = %request.model, "POST /v1/chat/completions");

    // Format chat messages into a prompt
    let prompt = format_chat_prompt(&request.messages);
    let max_tokens = request.max_tokens.unwrap_or(100);

    // Get tokenizer
    let tokenizer = model_hosting
        .get_tokenizer(&request.model)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Model {} not loaded", request.model)))?;

    // Tokenize input
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| ApiError::InternalError(format!("Tokenization failed: {}", e)))?;
    let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

    if request.stream {
        // Streaming response
        let stream = create_chat_completion_stream(
            model_hosting,
            request.model.clone(),
            input_tokens.clone(),
            max_tokens,
            request.temperature,
            request.top_p,
            tokenizer,
        );

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming response
        let output_tokens = model_hosting
            .submit_inference(
                request.model.clone(),
                input_tokens.clone(),
                max_tokens as usize,
                request.temperature,
                request.top_p,
            )
            .await
            .map_err(|e| ApiError::InferenceError(e.to_string()))?;

        // Detokenize
        let output_text = tokenizer
            .decode(&output_tokens[input_tokens.len()..], true)
            .map_err(|e| ApiError::InternalError(format!("Detokenization failed: {}", e)))?;

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: input_tokens.len() as u32,
                completion_tokens: (output_tokens.len() - input_tokens.len()) as u32,
                total_tokens: output_tokens.len() as u32,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    // Simple chat template - in production use model-specific templates
    // (Llama2, ChatML, Alpaca, etc.)
    let mut prompt = String::new();

    for message in messages {
        match message.role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n\n", message.content)),
            "user" => prompt.push_str(&format!("User: {}\n\n", message.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", message.content)),
            _ => prompt.push_str(&format!("{}: {}\n\n", message.role, message.content)),
        }
    }

    prompt.push_str("Assistant: ");
    prompt
}

fn create_completion_stream(
    model_hosting: Arc<ModelHostingService>,
    model: String,
    input_tokens: Vec<u32>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    tokenizer: Arc<Tokenizer>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let model_for_id = model.clone();
    let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    async_stream::stream! {
        // Run inference to generate all tokens
        match model_hosting
            .submit_inference(
                model,
                input_tokens.clone(),
                max_tokens as usize,
                temperature,
                top_p,
            )
            .await
        {
            Ok(output_tokens) => {
                let generated_tokens = &output_tokens[input_tokens.len()..];

                // Stream each token
                for token in generated_tokens {
                    if let Ok(text) = tokenizer.decode(&[*token], false) {
                        let response = serde_json::json!({
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_for_id,
                            "choices": [{
                                "text": text,
                                "index": 0,
                                "finish_reason": null,
                            }],
                        });

                        if let Ok(event) = Event::default().json_data(response) {
                            yield Ok(event);
                        }
                    }
                }

                // Send final event with finish_reason
                let response = serde_json::json!({
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_for_id,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "finish_reason": "stop",
                    }],
                });

                if let Ok(event) = Event::default().json_data(response) {
                    yield Ok(event);
                }

                // Send [DONE]
                yield Ok(Event::default().data("[DONE]"));
            }
            Err(e) => {
                error!("Inference failed: {}", e);
                yield Ok(Event::default().data("[DONE]"));
            }
        }
    }
}

fn create_chat_completion_stream(
    model_hosting: Arc<ModelHostingService>,
    model: String,
    input_tokens: Vec<u32>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    tokenizer: Arc<Tokenizer>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let model_for_id = model.clone();
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    async_stream::stream! {
        // Run inference to generate all tokens
        match model_hosting
            .submit_inference(
                model,
                input_tokens.clone(),
                max_tokens as usize,
                temperature,
                top_p,
            )
            .await
        {
            Ok(output_tokens) => {
                let generated_tokens = &output_tokens[input_tokens.len()..];

                // Stream each token
                for token in generated_tokens {
                    if let Ok(text) = tokenizer.decode(&[*token], false) {
                        let response = serde_json::json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_for_id,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": text,
                                },
                                "finish_reason": null,
                            }],
                        });

                        if let Ok(event) = Event::default().json_data(response) {
                            yield Ok(event);
                        }
                    }
                }

                // Send final event with finish_reason
                let response = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_for_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                });

                if let Ok(event) = Event::default().json_data(response) {
                    yield Ok(event);
                }

                // Send [DONE]
                yield Ok(Event::default().data("[DONE]"));
            }
            Err(e) => {
                error!("Inference failed: {}", e);
                yield Ok(Event::default().data("[DONE]"));
            }
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
#[allow(dead_code)]
enum ApiError {
    BadRequest(String),
    NotFound(String),
    InferenceError(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "invalid_request_error", msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found_error", msg),
            ApiError::InferenceError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "inference_error", msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg),
        };

        let error_response = ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                code: None,
            },
        };

        (status, Json(error_response)).into_response()
    }
}
