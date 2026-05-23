use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::{env, process};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::{clip_server, commands};

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:19829";
const SERVICE_NAME: &str = "OmniMind-Wiki-Core";
const SERVER_MODE: &str = "server-only";
const DEFAULT_MAX_CONTEXT_SIZE: usize = 204_800;
const APP_STATE_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

pub struct OmnimindResponse {
    status: u16,
    body: Value,
}

#[derive(Debug, Deserialize)]
struct ChatHistoryMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatContextRequest {
    query: String,
    #[serde(default)]
    history: Vec<ChatHistoryMessage>,
    #[serde(default)]
    max_history_messages: Option<usize>,
    #[serde(default)]
    max_context_size: Option<usize>,
    #[serde(default)]
    output_language: Option<String>,
    #[serde(default)]
    include_debug: bool,
}

struct ContextBudget {
    max_ctx: usize,
    response_reserve: usize,
    index_budget: usize,
    page_budget: usize,
    max_page_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectEntry {
    id: String,
    name: String,
    path: String,
    current: bool,
}

#[derive(Clone)]
struct CachedAppState {
    loaded_at: std::time::Instant,
    value: Option<Value>,
}

static APP_STATE_CACHE: OnceLock<Mutex<Option<CachedAppState>>> = OnceLock::new();

/// 判断当前命令行是否请求进入 OmniMind 无头服务模式。
pub fn should_start_from_args<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    args.into_iter().any(|arg| arg.as_ref() == "--server-only")
}

/// 启动 OmniMind 的独立无头服务。
pub fn start() {
    let bind_addr = env::var("OMNIMIND_WIKI_CORE_BIND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_string());

    if let Err(err) = run_blocking(&bind_addr) {
        eprintln!("[OmniMind Wiki Core] server-only start failed: {err}");
        process::exit(1);
    }
}

fn run_blocking(bind_addr: &str) -> Result<(), String> {
    let server =
        Server::http(bind_addr).map_err(|err| format!("failed to bind {bind_addr}: {err}"))?;

    eprintln!("[OmniMind Wiki Core] listening on http://{bind_addr}");
    for request in server.incoming_requests() {
        handle_request(request);
    }

    Ok(())
}

fn handle_request(mut request: tiny_http::Request) {
    let method = request.method().clone();
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or(&url).to_string();

    let response = match method {
        Method::Get => handle_get(&url),
        Method::Post => handle_post(&path, &mut request),
        Method::Delete => handle_delete(&path, &mut request),
        Method::Options => OmnimindResponse {
            status: 204,
            body: json!(null),
        },
        _ => error_response(405, "METHOD_NOT_ALLOWED", "Method not allowed"),
    };

    respond_json(request, response);
}

#[derive(Deserialize)]
struct SaveFileRequest {
    #[serde(alias = "projectId")]
    project_id: Option<String>,
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct DeleteFileRequest {
    #[serde(alias = "projectId")]
    project_id: Option<String>,
    path: String,
}

fn get_target_project(project_id: Option<&str>) -> Result<ProjectEntry, String> {
    match project_id {
        Some(id) => resolve_project(id),
        None => load_projects()
            .into_iter()
            .find(|p| p.current)
            .or_else(|| load_projects().first().cloned())
            .ok_or_else(|| "No project found".to_string()),
    }
}

fn handle_save_file(body: &str) -> OmnimindResponse {
    let req: SaveFileRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    let project = match get_target_project(req.project_id.as_deref()) {
        Ok(p) => p,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    let full_path = match safe_join(&project.path, &req.path) {
        Ok(p) => p,
        Err(e) => return error_response(400, "INVALID_PATH", &e),
    };

    if let Some(parent) = full_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return error_response(
                500,
                "CREATE_DIR_FAILED",
                &format!("Failed to create directory: {e}"),
            );
        }
    }

    if let Err(e) = fs::write(&full_path, req.content) {
        return error_response(500, "WRITE_FAILED", &format!("Failed to write file: {e}"));
    }

    OmnimindResponse {
        status: 200,
        body: json!({ "ok": true }),
    }
}

fn handle_delete_file(body: &str) -> OmnimindResponse {
    let req: DeleteFileRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    let project = match get_target_project(req.project_id.as_deref()) {
        Ok(p) => p,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    let full_path = match safe_join(&project.path, &req.path) {
        Ok(p) => p,
        Err(e) => return error_response(400, "INVALID_PATH", &e),
    };

    if !full_path.exists() {
        return error_response(404, "FILE_NOT_FOUND", "File does not exist");
    }

    if full_path.is_dir() {
        if let Err(e) = fs::remove_dir_all(&full_path) {
            return error_response(
                500,
                "DELETE_FAILED",
                &format!("Failed to delete directory: {e}"),
            );
        }
    } else {
        if let Err(e) = fs::remove_file(&full_path) {
            return error_response(500, "DELETE_FAILED", &format!("Failed to delete file: {e}"));
        }
    }

    OmnimindResponse {
        status: 200,
        body: json!({ "ok": true }),
    }
}

fn handle_get_file_content(query: &str) -> OmnimindResponse {
    let params = parse_query(query);
    let project_id = params.get("projectId").map(String::as_str);
    let path_str = match params.get("path") {
        Some(p) => p,
        None => return error_response(400, "PATH_REQUIRED", "path parameter is required"),
    };

    let project = match get_target_project(project_id) {
        Ok(p) => p,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    let full_path = match safe_join(&project.path, path_str) {
        Ok(p) => p,
        Err(e) => return error_response(400, "INVALID_PATH", &e),
    };

    if !full_path.exists() {
        return error_response(404, "FILE_NOT_FOUND", "File does not exist");
    }

    if full_path.is_dir() {
        return error_response(400, "NOT_A_FILE", "Path is a directory");
    }

    match fs::read_to_string(&full_path) {
        Ok(content) => OmnimindResponse {
            status: 200,
            body: json!({
                "ok": true,
                "content": content,
            }),
        },
        Err(e) => error_response(500, "READ_FAILED", &format!("Failed to read file: {e}")),
    }
}

fn handle_delete(path: &str, request: &mut tiny_http::Request) -> OmnimindResponse {
    let parts: Vec<&str> = path
        .trim_start_matches('/')
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();

    match parts.as_slice() {
        ["api", "v1", "files", "delete"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_delete_file(&body)
        }
        _ => error_response(404, "NOT_FOUND", "Not found"),
    }
}

fn handle_get(path: &str) -> OmnimindResponse {
    let (clean_path, query) = split_url(path);
    match clean_path.as_str() {
        "/health" | "/api/v1/health" => OmnimindResponse {
            status: 200,
            body: json!({
                "ok": true,
                "service": SERVICE_NAME,
                "mode": SERVER_MODE,
                "version": env!("CARGO_PKG_VERSION"),
            }),
        },
        "/api/v1/projects" => handle_projects(),
        "/api/v1/files/content" => handle_get_file_content(query),
        _ => {
            let parts: Vec<&str> = clean_path
                .trim_start_matches('/')
                .split('/')
                .filter(|part| !part.is_empty())
                .collect();

            match parts.as_slice() {
                ["api", "v1", "projects", project_id, "files"] => handle_files(project_id, query),
                _ => error_response(404, "NOT_FOUND", "Not found"),
            }
        }
    }
}

fn handle_post(path: &str, request: &mut tiny_http::Request) -> OmnimindResponse {
    let parts: Vec<&str> = path
        .trim_start_matches('/')
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();

    match parts.as_slice() {
        ["api", "v1", "document", "upload"] => handle_upload(request),
        ["api", "v1", "files", "save"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_save_file(&body)
        }
        ["api", "v1", "workspace", project_id, "update-embeddings"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_update_embeddings(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "chat-context"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_chat_context(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "search"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_search(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "vector-upsert"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_vector_upsert_chunks(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "vector-delete"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_vector_delete_page(project_id, &body)
        }
        ["api", "v1", "projects", _project_id, "chat"] => error_response(
            501,
            "CHAT_NOT_IMPLEMENTED",
            "LLM Wiki server-only 当前只提供知识上下文构建，不直接生成最终聊天回复。",
        ),
        _ => error_response(404, "NOT_FOUND", "Not found"),
    }
}

fn handle_upload(request: &mut tiny_http::Request) -> OmnimindResponse {
    let boundary = match get_multipart_boundary(request) {
        Some(b) => b,
        None => {
            return error_response(
                400,
                "INVALID_CONTENT_TYPE",
                "Content-Type must be multipart/form-data",
            )
        }
    };

    let mut multipart = multipart_2021::server::Multipart::with_body(request.as_reader(), boundary);
    let mut uploaded_files = Vec::new();

    // Use current project or fallback to first project
    let project = match load_projects()
        .into_iter()
        .find(|p| p.current)
        .or_else(|| load_projects().first().cloned())
    {
        Some(p) => p,
        None => return error_response(500, "NO_PROJECT_FOUND", "No project found to upload to"),
    };

    let wiki_dir = match safe_join(&project.path, "wiki") {
        Ok(path) => path,
        Err(e) => return error_response(500, "INVALID_PROJECT_PATH", &e),
    };

    if let Err(e) = fs::create_dir_all(&wiki_dir) {
        return error_response(
            500,
            "CREATE_DIR_FAILED",
            &format!("Failed to create wiki dir: {e}"),
        );
    }

    while let Ok(Some(mut field)) = multipart.read_entry() {
        let filename = field
            .headers
            .filename
            .clone()
            .unwrap_or_else(|| "uploaded_file.txt".to_string());
        let target_path = wiki_dir.join(&filename);

        let mut file = match fs::File::create(&target_path) {
            Ok(f) => f,
            Err(e) => {
                return error_response(
                    500,
                    "FILE_CREATE_FAILED",
                    &format!("Failed to create file {filename}: {e}"),
                )
            }
        };

        if let Err(e) = std::io::copy(&mut field.data, &mut file) {
            return error_response(
                500,
                "FILE_WRITE_FAILED",
                &format!("Failed to write file {filename}: {e}"),
            );
        }

        let location = format!("wiki/{}", filename);
        uploaded_files.push(json!({ "location": location }));
    }

    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "documents": uploaded_files,
            "location": uploaded_files.first().and_then(|f| f.get("location")).unwrap_or(&json!(null)),
        }),
    }
}

fn get_multipart_boundary(request: &tiny_http::Request) -> Option<String> {
    for header in request.headers() {
        if header.field.as_str().to_ascii_lowercase() == "content-type" {
            let value = header.value.as_str();
            if value.to_lowercase().contains("multipart/form-data") {
                if let Some(pos) = value.find("boundary=") {
                    return Some(value[pos + 9..].to_string());
                }
            }
        }
    }
    None
}

#[derive(Deserialize)]
struct UpdateEmbeddingsRequest {
    #[allow(dead_code)]
    adds: Vec<String>,
    #[allow(dead_code)]
    deletes: Vec<String>,
}

fn handle_update_embeddings(project_id: &str, body: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    let _req: UpdateEmbeddingsRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    // Trigger rescan
    match tauri::async_runtime::block_on(async {
        commands::file_sync::rescan_project_files_inner(
            None, // No app handle
            project.id.clone(),
            project.path.clone(),
            None, // Default config
        )
    }) {
        Ok(_) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true }),
        },
        Err(e) => error_response(500, "RESCAN_FAILED", &e),
    }
}

fn handle_projects() -> OmnimindResponse {
    let projects = load_projects();
    let current_project = projects.iter().find(|project| project.current).cloned();
    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "projects": projects,
            "currentProject": current_project,
        }),
    }
}

fn handle_files(project_id: &str, query: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };
    let params = parse_query(query);
    let root = params.get("root").map(String::as_str).unwrap_or("wiki");
    let recursive = params
        .get("recursive")
        .map(|v| v != "false")
        .unwrap_or(true);
    let max_files = params
        .get("maxFiles")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2000)
        .clamp(1, 10000);

    let rel = match root {
        "wiki" => "wiki",
        "sources" | "raw" | "raw/sources" => "raw/sources",
        "all" | "" => "",
        _ => return error_response(400, "INVALID_ROOT", "root must be wiki, sources, or all"),
    };

    if rel.is_empty() {
        match list_public_roots(&project.path, recursive, max_files) {
            Ok(files) => OmnimindResponse {
                status: 200,
                body: json!({
                    "ok": true,
                    "projectId": project.id,
                    "root": "all",
                    "files": files,
                    "truncated": false,
                }),
            },
            Err(e) => error_response(500, "LIST_FILES_FAILED", &e),
        }
    } else {
        let dir = match safe_join(&project.path, rel) {
            Ok(path) => path,
            Err(e) => return error_response(400, "INVALID_PATH", &e),
        };
        let mut count = 0;
        match list_tree(&project.path, &dir, recursive, max_files, &mut count) {
            Ok(files) => OmnimindResponse {
                status: 200,
                body: json!({
                    "ok": true,
                    "projectId": project.id,
                    "root": rel,
                    "files": files,
                    "truncated": false,
                }),
            },
            Err(e) => error_response(500, "LIST_FILES_FAILED", &e),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
    include_content: Option<bool>,
    query_embedding: Option<Vec<f32>>,
}

fn handle_search(project_id: &str, body: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };
    let req: SearchRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };
    if req.query.trim().is_empty() {
        return error_response(400, "QUERY_REQUIRED", "query is required");
    }

    let top_k = req.top_k.unwrap_or(10).clamp(1, 50);
    let query = req.query;

    let embedding_config = load_embedding_config();
    let query_embedding = match tauri::async_runtime::block_on(
        commands::search::resolve_query_embedding(&query, req.query_embedding, embedding_config),
    ) {
        Ok(embedding) => embedding,
        Err(e) => return error_response(400, "EMBEDDING_ERROR", &e),
    };

    match tauri::async_runtime::block_on(commands::search::search_project_inner(
        project.path.clone(),
        query,
        top_k,
        req.include_content.unwrap_or(false),
        query_embedding,
    )) {
        Ok(search) => OmnimindResponse {
            status: 200,
            body: json!({
                "ok": true,
                "projectId": project.id,
                "mode": search.mode,
                "tokenHits": search.token_hits,
                "vectorHits": search.vector_hits,
                "results": search.results,
            }),
        },
        Err(e) => error_response(500, "SEARCH_FAILED", &e),
    }
}

#[derive(Deserialize)]
struct VectorUpsertRequest {
    page_id: String,
    chunks: Vec<commands::vectorstore::ChunkUpsertInput>,
}

fn handle_vector_upsert_chunks(project_id: &str, body: &str) -> OmnimindResponse {
    let req: VectorUpsertRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_upsert_chunks(
        project.path.clone(),
        req.page_id,
        req.chunks,
    )) {
        Ok(_) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "message": "Vector chunks successfully upserted" }),
        },
        Err(e) => error_response(500, "VECTOR_UPSERT_FAILED", &e),
    }
}

#[derive(Deserialize)]
struct VectorDeleteRequest {
    page_id: String,
}

fn handle_vector_delete_page(project_id: &str, body: &str) -> OmnimindResponse {
    let req: VectorDeleteRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_delete_page(
        project.path.clone(),
        req.page_id,
    )) {
        Ok(_) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "message": "Page vectors successfully deleted" }),
        },
        Err(e) => error_response(500, "VECTOR_DELETE_FAILED", &e),
    }
}

fn handle_chat_context(project_id: &str, body: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };
    let request = match serde_json::from_str::<ChatContextRequest>(body) {
        Ok(value) => value,
        Err(err) => {
            return error_response(
                400,
                "INVALID_JSON",
                &format!("Request body must be valid JSON: {err}"),
            );
        }
    };

    if request.query.trim().is_empty() {
        return error_response(400, "INVALID_QUERY", "query must not be empty");
    }

    let budget = compute_context_budget(request.max_context_size);

    // Perform real search
    let embedding_config = load_embedding_config();
    let query_embedding = match tauri::async_runtime::block_on(
        commands::search::resolve_query_embedding(&request.query, None, embedding_config),
    ) {
        Ok(embedding) => embedding,
        Err(_) => None,
    };

    let search_results =
        match tauri::async_runtime::block_on(commands::search::search_project_inner(
            project.path.clone(),
            request.query.clone(),
            10,   // top_k for context
            true, // include_content
            query_embedding,
        )) {
            Ok(res) => res.results,
            Err(_) => Vec::new(),
        };

    let mut context_blocks = Vec::new();
    let mut references = Vec::new();
    let mut current_size = 0;

    for (idx, res) in search_results.into_iter().enumerate() {
        let ref_id = idx + 1;
        let content = res.content.unwrap_or_default();
        let block = format!("[{}] {}\n{}", ref_id, res.title, content);
        let block_size = block.chars().count();

        if current_size + block_size > budget.page_budget {
            // Truncate if single block is too large or we exceeded budget
            if current_size == 0 {
                let truncated: String = block.chars().take(budget.max_page_size).collect();
                context_blocks.push(json!({
                    "id": ref_id,
                    "title": res.title,
                    "content": truncated,
                }));
                references.push(json!({
                    "id": ref_id,
                    "title": res.title,
                    "path": res.path,
                }));
            }
            break;
        }

        context_blocks.push(json!({
            "id": ref_id,
            "title": res.title,
            "content": content,
        }));
        references.push(json!({
            "id": ref_id,
            "title": res.title,
            "path": res.path,
        }));
        current_size += block_size;
    }

    let status = if context_blocks.is_empty() {
        "EMPTY_CONTEXT"
    } else {
        "SUCCESS"
    };

    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "status": status,
            "project_id": project_id,
            "query": request.query,
            "context_blocks": context_blocks,
            "references": references,
            "budget": {
                "max_ctx": budget.max_ctx,
                "response_reserve": budget.response_reserve,
                "index_budget": budget.index_budget,
                "page_budget": budget.page_budget,
                "max_page_size": budget.max_page_size,
            },
            "retrieval_debug": build_retrieval_debug(&request),
        }),
    }
}

fn compute_context_budget(max_context_size: Option<usize>) -> ContextBudget {
    let max_ctx = max_context_size
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_CONTEXT_SIZE);
    let response_reserve = max_ctx * 15 / 100;
    let index_budget = max_ctx * 5 / 100;
    let page_budget = max_ctx * 50 / 100;
    let max_page_size = page_budget.min(5_000.max(page_budget * 30 / 100));

    ContextBudget {
        max_ctx,
        response_reserve,
        index_budget,
        page_budget,
        max_page_size,
    }
}

fn build_retrieval_debug(request: &ChatContextRequest) -> Value {

    if !request.include_debug {
        return json!(null);
    }

    json!({
        "mode": "server-only",
        "history_message_count": request.history.len(),
        "history_content_chars": request.history.iter().map(|item| item.content.chars().count()).sum::<usize>(),
    })
}

fn error_response(status: u16, code: &str, message: &str) -> OmnimindResponse {
    OmnimindResponse {
        status,
        body: json!({
            "ok": false,
            "error": {
                "code": code,
                "message": message,
            },
        }),
    }
}

fn read_request_body(request: &mut tiny_http::Request) -> Option<String> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_ok() && !body.is_empty() {
        Some(body)
    } else {
        None
    }
}

fn respond_json(request: tiny_http::Request, response: OmnimindResponse) {
    let status = StatusCode(response.status);
    let body = if response.status == 204 {
        String::new()
    } else {
        response.body.to_string()
    };
    let mut http_response = Response::from_string(body).with_status_code(status);

    for header in cors_headers() {
        http_response.add_header(header);
    }

    let _ = request.respond(http_response);
}

fn cors_headers() -> Vec<Header> {
    vec![
        Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap(),
        Header::from_bytes("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS").unwrap(),
        Header::from_bytes(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )
        .unwrap(),
        Header::from_bytes("Content-Type", "application/json").unwrap(),
    ]
}

fn split_url(url: &str) -> (String, &str) {
    match url.split_once('?') {
        Some((path, query)) => (path.to_string(), query),
        None => (url.to_string(), ""),
    }
}

fn parse_query(query: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for pair in query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        out.insert(percent_decode(k), percent_decode(v));
    }
    out
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn get_app_data_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("OMNIMIND_WIKI_CORE_DATA_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Guess path based on OS
    let home = env::var("HOME")
        .ok()
        .or_else(|| env::var("USERPROFILE").ok())?;
    let path = if cfg!(target_os = "macos") {
        PathBuf::from(home).join("Library/Application Support/com.llmwiki.app")
    } else if cfg!(target_os = "windows") {
        PathBuf::from(env::var("APPDATA").ok()?).join("com.llmwiki.app")
    } else {
        PathBuf::from(home).join(".local/share/com.llmwiki.app")
    };

    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn load_app_state() -> Option<Value> {
    let now = std::time::Instant::now();
    let lock = APP_STATE_CACHE.get_or_init(|| Mutex::new(None));
    let mut previous = None;
    if let Ok(cache) = lock.lock() {
        if let Some(cached) = cache.as_ref() {
            if now.duration_since(cached.loaded_at) < APP_STATE_CACHE_TTL {
                return cached.value.clone();
            }
            previous = cached.value.clone();
        }
    }

    let dir = get_app_data_dir()?;
    let path = dir.join("app-state.json");
    let loaded = fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    let value = loaded.or(previous);

    if let Ok(mut cache) = lock.lock() {
        *cache = Some(CachedAppState {
            loaded_at: now,
            value: value.clone(),
        });
    }
    value
}

fn load_projects() -> Vec<ProjectEntry> {
    let current = normalize_path(&clip_server::current_project_path());
    let mut by_path: BTreeMap<String, ProjectEntry> = BTreeMap::new();

    if let Some(parsed) = load_app_state() {
        if let Some(registry) = parsed.get("projectRegistry").and_then(Value::as_object) {
            for (id, value) in registry {
                let path = value.get("path").and_then(Value::as_str).unwrap_or("");
                if path.is_empty() {
                    continue;
                }
                let path = normalize_path(path);
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| project_name_from_path(&path));
                by_path.insert(
                    path.clone(),
                    ProjectEntry {
                        id: id.clone(),
                        name,
                        current: path == current,
                        path,
                    },
                );
            }
        }
    }

    for (name, path) in clip_server::all_projects() {
        let path = normalize_path(&path);
        by_path.entry(path.clone()).or_insert_with(|| ProjectEntry {
            id: read_project_id(&path).unwrap_or_else(|| path.clone()),
            name: if name.is_empty() {
                project_name_from_path(&path)
            } else {
                name
            },
            current: path == current,
            path,
        });
    }

    if !current.is_empty() {
        by_path
            .entry(current.clone())
            .or_insert_with(|| ProjectEntry {
                id: read_project_id(&current).unwrap_or_else(|| current.clone()),
                name: project_name_from_path(&current),
                current: true,
                path: current.clone(),
            });
    }

    by_path.into_values().collect()
}

fn resolve_project(project_id: &str) -> Result<ProjectEntry, String> {
    let project_id = percent_decode(project_id);
    let wants_current = project_id.eq_ignore_ascii_case("current");
    load_projects()
        .into_iter()
        .find(|p| {
            p.id == project_id
                || project_path_matches(&p.path, &project_id)
                || (wants_current && p.current)
        })
        .ok_or_else(|| format!("Unknown project: {project_id}"))
}

fn project_path_matches(stored_path: &str, candidate: &str) -> bool {
    let stored = normalize_path(stored_path);
    let candidate = normalize_path(candidate);
    if cfg!(windows) {
        stored.eq_ignore_ascii_case(&candidate)
    } else {
        stored == candidate
    }
}

fn read_project_id(path: &str) -> Option<String> {
    let raw = fs::read_to_string(Path::new(path).join(".llm-wiki/project.json")).ok()?;
    let parsed: Value = serde_json::from_str(&raw).ok()?;
    parsed
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn project_name_from_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Project")
        .to_string()
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").trim_end_matches('/').to_string()
}

fn load_embedding_config() -> Option<commands::search::SearchEmbeddingConfig> {
    let parsed = load_app_state()?;
    let value = parsed.get("embeddingConfig")?.clone();
    serde_json::from_value::<commands::search::SearchEmbeddingConfig>(value).ok()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiFileNode {
    name: String,
    path: String,
    is_dir: bool,
    size: Option<u64>,
    children: Option<Vec<ApiFileNode>>,
}

fn list_public_roots(
    project_path: &str,
    recursive: bool,
    max_files: usize,
) -> Result<Vec<ApiFileNode>, String> {
    let mut count = 0;
    let mut roots = Vec::new();
    for rel in ["purpose.md", "schema.md", "wiki", "raw/sources"] {
        let path = safe_join(project_path, rel)?;
        if !path.exists() {
            continue;
        }
        push_file_node(
            project_path,
            &path,
            recursive,
            max_files,
            &mut count,
            &mut roots,
        )?;
    }
    Ok(roots)
}

fn list_tree(
    project_path: &str,
    path: &Path,
    recursive: bool,
    max_files: usize,
    count: &mut usize,
) -> Result<Vec<ApiFileNode>, String> {
    let mut out = Vec::new();
    let entries = fs::read_dir(path).map_err(|e| format!("Failed to list directory: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        push_file_node(
            project_path,
            &entry.path(),
            recursive,
            max_files,
            count,
            &mut out,
        )?;
    }
    out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));
    Ok(out)
}

fn push_file_node(
    project_path: &str,
    path: &Path,
    recursive: bool,
    max_files: usize,
    count: &mut usize,
    out: &mut Vec<ApiFileNode>,
) -> Result<(), String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    if name.starts_with('.') {
        return Ok(());
    }
    let meta = fs::symlink_metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Ok(());
    }
    *count += 1;
    if *count > max_files {
        return Err(format!("File listing exceeds maxFiles limit ({max_files})"));
    }
    let is_dir = file_type.is_dir();
    let children = if recursive && is_dir {
        Some(list_tree(project_path, path, true, max_files, count)?)
    } else {
        None
    };
    out.push(ApiFileNode {
        name,
        path: relative_to_project(project_path, path),
        is_dir,
        size: if is_dir { None } else { Some(meta.len()) },
        children,
    });
    Ok(())
}

fn relative_to_project(project_path: &str, path: &Path) -> String {
    let root = Path::new(project_path);
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn safe_join(project_path: &str, rel: &str) -> Result<PathBuf, String> {
    let root = PathBuf::from(project_path);
    let rel = rel.trim_start_matches('/');
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err("Absolute paths are not allowed".to_string());
    }
    for component in rel_path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::Prefix(_) | Component::RootDir
        ) {
            return Err("Path traversal is not allowed".to_string());
        }
    }
    let joined = root.join(rel_path);
    // In server-only mode, we might not have canonicalize working the same way if paths don't exist
    // but we should still try to prevent traversal.
    Ok(joined)
}
