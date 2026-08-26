use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
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
/// chat-context 接受的绝对字符预算上限（1 MiB 字符）。
/// 该值保留上游 204_800 默认预算并覆盖 Python 自动客服当前 40_000 请求，
/// 同时阻止恶意极值把百分比计算或候选装箱放大到不可控范围。
const MAX_CHAT_CONTEXT_SIZE: usize = 1_048_576;
/// chat-context 的历史默认候选数。保持既有固定值 10，避免未传新字段的调用方行为漂移。
const DEFAULT_CHAT_CONTEXT_TOP_K: usize = 10;
/// 与 Rust 原生搜索的公开上限一致，防止服务端上下文接口被异常大候选数拖垮。
const MAX_CHAT_CONTEXT_TOP_K: usize = 50;
/// 单次资源范围过滤最多接受的来源路径数量。
///
/// 这是 privileged server-only 边界上的硬上限：即使调用方被错误配置，也不能通过
/// 超大 allowlist 放大请求解析、规范化和逐候选匹配成本。
const MAX_ALLOWED_SOURCE_PATHS: usize = 256;
/// 单个项目相对来源路径的最大字符数。
///
/// 这里按 Unicode 字符计数而不是字节计数，既能稳定限制用户可见路径长度，也避免把
/// 合法中文文件名按 UTF-8 字节数过度惩罚。
const MAX_ALLOWED_SOURCE_PATH_CHARS: usize = 1_024;
const APP_STATE_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);
/// Hybrid retrieval keeps at least this many `raw/sources` hits when available.
const HYBRID_SOURCES_FLOOR: usize = 2;

pub struct OmnimindResponse {
    status: u16,
    body: Value,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ChatHistoryMessage {
    role: String,
    content: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ChatContextRequest {
    query: String,
    /// 可选的调用方显式查询向量。
    ///
    /// OmniMind 的租户级 Embedding 配置保存在 Python 服务中，不能写入 Wiki Core
    /// 的全局 app-state，也不能把租户密钥下发给 Core。因此 server-only 模式允许
    /// Python 先完成向量化，再把向量交给 Core 原生 LanceDB/BM25/Graph/RRF 检索。
    #[serde(default)]
    query_embedding: Option<Vec<f32>>,
    #[serde(default)]
    history: Vec<ChatHistoryMessage>,
    #[serde(default)]
    max_history_messages: Option<usize>,
    #[serde(default)]
    max_context_size: Option<usize>,
    /// 可选的 Rust 原生候选数量；缺省继续使用既有的 10。
    #[serde(default)]
    top_k: Option<usize>,
    /// 可选的单块正文字符上限。仅显式提供时启用“截取当前块后继续装箱”的新行为。
    #[serde(default)]
    max_block_chars: Option<usize>,
    #[serde(default)]
    output_language: Option<String>,
    #[serde(default)]
    include_debug: bool,
    /// Retrieval mode for chat-context (case-insensitive).
    /// `wiki` (default) | `sources_only` | `hybrid`
    /// Aliases: `faithful`/`sources` → sources_only; `all` → hybrid.
    #[serde(default)]
    retrieval_mode: Option<String>,
    /// 可选的项目相对来源路径 allowlist。
    ///
    /// 缺省时保持旧客户端的全库检索行为；一旦提供，候选必须先通过精确路径匹配，
    /// 才能进入排序和上下文装箱。校验与规范化在 privileged 边界统一完成，调用方
    /// 不能借绝对路径或目录穿越扩大 Wiki Core 的可见范围。
    #[serde(default)]
    allowed_source_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetrievalMode {
    Wiki,
    SourcesOnly,
    Hybrid,
}

fn parse_retrieval_mode(raw: Option<&str>) -> RetrievalMode {
    match raw
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("wiki") => RetrievalMode::Wiki,
        Some("sources_only") | Some("faithful") | Some("sources") => RetrievalMode::SourcesOnly,
        Some("hybrid") | Some("all") => RetrievalMode::Hybrid,
        // Unknown values fall back to wiki for backward compatibility.
        _ => RetrievalMode::Wiki,
    }
}

fn retrieval_mode_label(mode: RetrievalMode) -> &'static str {
    match mode {
        RetrievalMode::Wiki => "wiki",
        RetrievalMode::SourcesOnly => "sources_only",
        RetrievalMode::Hybrid => "hybrid",
    }
}

fn scan_roots_for_mode(mode: RetrievalMode) -> Vec<String> {
    match mode {
        RetrievalMode::Wiki => vec!["wiki".to_string()],
        RetrievalMode::SourcesOnly => vec!["raw/sources".to_string()],
        RetrievalMode::Hybrid => vec!["wiki".to_string(), "raw/sources".to_string()],
    }
}

/// 校验并规范化调用方提供的来源路径 allowlist。
///
/// 安全边界说明：这里只接受项目内的相对文件路径，不接受空列表、空项、绝对路径、
/// Windows 盘符、控制字符以及 `.` / `..` 组件。反斜杠统一成 `/`，使 Windows 风格
/// 调用方与 Core 返回的项目相对路径使用同一个比较空间。返回 `BTreeSet` 同时去重，
/// 避免重复项放大后续匹配成本。
fn normalize_allowed_source_paths(
    requested: Option<&[String]>,
) -> Result<Option<BTreeSet<String>>, &'static str> {
    let Some(paths) = requested else {
        // 字段省略是明确的向后兼容路径：保持现有全库候选行为。
        return Ok(None);
    };
    if paths.is_empty() {
        return Err("allowed_source_paths must not be empty when provided");
    }
    if paths.len() > MAX_ALLOWED_SOURCE_PATHS {
        return Err("allowed_source_paths contains too many entries");
    }

    let mut normalized = BTreeSet::new();
    for path in paths {
        normalized.insert(normalize_project_relative_source_path(path)?);
    }
    Ok(Some(normalized))
}

/// 把单个来源路径收敛为可用于精确 allowlist 比较的项目相对路径。
fn normalize_project_relative_source_path(raw: &str) -> Result<String, &'static str> {
    if raw.trim().is_empty() {
        return Err("allowed source path must not be empty");
    }
    if raw.chars().count() > MAX_ALLOWED_SOURCE_PATH_CHARS {
        return Err("allowed source path is too long");
    }
    if raw.chars().any(char::is_control) {
        // NUL 以及换行等控制字符都可能造成跨语言解析或日志边界歧义，统一拒绝。
        return Err("allowed source path contains control characters");
    }

    let slash_normalized = raw.replace('\\', "/");
    if slash_normalized.starts_with('/') {
        return Err("allowed source path must be project-relative");
    }
    let bytes = slash_normalized.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        // 在 Unix 上 `C:/...` 会被 Path 误判为相对路径，因此显式拦截 Windows 盘符。
        return Err("allowed source path must not contain a drive prefix");
    }

    let mut components = Vec::new();
    for component in slash_normalized.split('/') {
        if component.is_empty() {
            // 不接受重复或尾随分隔符，防止同一路径出现多个非规范表示。
            return Err("allowed source path contains an empty component");
        }
        if component == "." || component == ".." {
            return Err("allowed source path contains traversal components");
        }
        components.push(component);
    }
    Ok(components.join("/"))
}

/// 计算一次 chat-context 请求真正交给 Core 搜索器的扫描根。
///
/// 设计取舍：未提供 allowlist 时继续返回检索模式的历史根目录，完整保持旧行为；提供
/// allowlist 时则只保留同时属于当前检索模式的精确资源路径。搜索器会先扫描这些路径，
/// 再在这个受限候选集合上执行 BM25 / Vector / Graph / Hybrid 排序和 top-k，从根源上
/// 避免“先全库截断、后过滤”把合法资源错误排出候选。
///
/// 若二者没有交集，本函数返回空集合。调用方必须直接返回 EMPTY_CONTEXT，不能把空
/// roots 交给 `search_project_inner`；后者为了兼容旧客户端会把空 roots 回退成 `wiki`，
/// 在显式资源范围下那会造成意外的全库扫描。
fn resolve_chat_context_scan_roots(
    mode: RetrievalMode,
    allowed_paths: Option<&BTreeSet<String>>,
) -> Vec<String> {
    let mode_roots = scan_roots_for_mode(mode);
    let Some(allowed_paths) = allowed_paths else {
        return mode_roots;
    };

    allowed_paths
        .iter()
        .filter(|path| {
            mode_roots.iter().any(|root| {
                path.as_str() == root
                    || path
                        .strip_prefix(root)
                        .is_some_and(|suffix| suffix.starts_with('/'))
            })
        })
        .cloned()
        .collect()
}

/// 显式资源范围经过文件系统安全边界校验后的不可变投影。
///
/// allowlist 省略时保留调用方原始项目路径，避免改变旧项目通过路径别名打开的行为；
/// 显式范围则固定到 canonical 项目根和 canonical 项目相对普通文件，供后续扫描复用。
struct ValidatedSourceFileScope {
    project_root: String,
    allowed_paths: Option<BTreeSet<String>>,
}

/// 在调用搜索前，把显式 allowlist 收敛为 canonical 项目根内的普通文件集合。
///
/// 安全规则：项目根本身、任意中间组件和最终文件都不能是 symlink；最终目标必须是普通
/// 文件，且 canonical 结果仍位于 canonical 项目根之下。调用方保存的同步记录可能晚于
/// Core 文件清理，因此仅当 `symlink_metadata` 明确返回 `NotFound` 时，把该候选视为当前
/// 不可用并从结果中剔除；权限、I/O、软链接、非普通文件等情况仍全部失败关闭。搜索阶段
/// 收到的是 canonical 根与 canonical 相对路径，随后仍由 `WalkDir` 的不跟随 symlink 行为
/// 及后置精确过滤提供纵深防御。所有错误均返回稳定分类，不包含调用方路径或系统错误正文。
fn validate_allowed_source_files(
    project_root: &str,
    allowed_paths: Option<BTreeSet<String>>,
) -> Result<ValidatedSourceFileScope, &'static str> {
    let Some(allowed_paths) = allowed_paths else {
        return Ok(ValidatedSourceFileScope {
            project_root: project_root.to_string(),
            allowed_paths: None,
        });
    };

    let root_path = Path::new(project_root);
    let root_metadata = fs::symlink_metadata(root_path)
        .map_err(|_| "project root is unavailable for source scope validation")?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err("project root must be a real directory for explicit source scope");
    }
    let canonical_root = fs::canonicalize(root_path)
        .map_err(|_| "project root cannot be canonicalized for source scope validation")?;

    let mut validated = BTreeSet::new();
    'allowed_path: for allowed_path in allowed_paths {
        let components = allowed_path.split('/').collect::<Vec<_>>();
        let mut candidate = canonical_root.clone();
        for (index, component) in components.iter().enumerate() {
            candidate.push(component);
            let metadata = match fs::symlink_metadata(&candidate) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    // 路径已经通过纯语法规范化，且逐组件从 canonical 项目根向下构造；
                    // 因而明确的 NotFound 只表示这条业务记录已经陈旧，不会扩大扫描范围。
                    // 整条候选直接跳过，绝不能把缺失文件或其父目录回退为全库扫描根。
                    continue 'allowed_path;
                }
                Err(_) => {
                    // PermissionDenied、循环链接、损坏挂载等异常不能等同于“文件不存在”。
                    // 若继续搜索会掩盖安全边界状态，因此保持原有 fail-closed 语义。
                    return Err("allowed source path metadata is unavailable");
                }
            };
            if metadata.file_type().is_symlink() {
                return Err("allowed source path must not contain symlinks");
            }
            let is_last = index + 1 == components.len();
            if is_last {
                if !metadata.is_file() {
                    return Err("allowed source path must reference a regular file");
                }
            } else if !metadata.is_dir() {
                return Err("allowed source path parent must be a real directory");
            }
        }

        let canonical_target = fs::canonicalize(&candidate)
            .map_err(|_| "allowed source path cannot be canonicalized")?;
        let relative = canonical_target
            .strip_prefix(&canonical_root)
            .map_err(|_| "allowed source path must remain inside the project root")?;
        let relative = relative
            .to_str()
            .ok_or("allowed source path must be valid UTF-8")?
            .replace('\\', "/");
        validated.insert(normalize_project_relative_source_path(&relative)?);
    }

    Ok(ValidatedSourceFileScope {
        project_root: canonical_root
            .to_str()
            .ok_or("canonical project root must be valid UTF-8")?
            .to_string(),
        allowed_paths: Some(validated),
    })
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
        // Pass full URL so POST handlers can read query (e.g. upload projectId).
        Method::Post => handle_post(&url, &mut request),
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtractImagesRequest {
    source_path: String,
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

fn handle_extract_images(project_id: &str, body: &str) -> OmnimindResponse {
    let req: ExtractImagesRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    if req.source_path.trim().is_empty() {
        return error_response(400, "SOURCE_PATH_REQUIRED", "sourcePath must not be empty");
    }

    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    // 这里强制把 Python 传来的相对路径重新锚定到 project 根目录，
    // 防止 server-only 模式被外部调用方借道访问任意磁盘路径。
    let source_full_path = match safe_join(&project.path, &req.source_path) {
        Ok(path) => path,
        Err(e) => return error_response(400, "INVALID_PATH", &e),
    };

    if !source_full_path.exists() {
        return error_response(404, "SOURCE_NOT_FOUND", "Source file does not exist");
    }
    if source_full_path.is_dir() {
        return error_response(400, "SOURCE_NOT_FILE", "sourcePath must point to a file");
    }

    // 所有抽出的图片统一落到当前项目的 `wiki/media/<slug>/`，
    // 这样后续 raw markdown、source summary 与检索链路都能复用同一套相对路径语义。
    let wiki_root = match safe_join(&project.path, "wiki") {
        Ok(path) => path,
        Err(e) => return error_response(500, "INVALID_WIKI_ROOT", &e),
    };

    let file_name = source_full_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("source");
    let media_slug = safe_media_slug(file_name);
    let media_dir = wiki_root.join("media").join(&media_slug);
    let source_string = source_full_path.to_string_lossy().to_string();
    let ext = source_full_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // 提取算法本体完全复用现有 Rust 原生实现；
    // server-only 这里仅做项目解析、路径约束与协议封装，不复制算法。
    let extracted = match ext.as_str() {
        "pdf" => match tauri::async_runtime::block_on(
            commands::extract_images::extract_and_save_pdf_images_cmd(
                source_string.clone(),
                media_dir.to_string_lossy().to_string(),
                wiki_root.to_string_lossy().to_string(),
            ),
        ) {
            Ok(images) => images,
            Err(e) => {
                return error_response(500, "EXTRACT_IMAGES_FAILED", &e);
            }
        },
        "pptx" | "docx" | "xlsx" => match tauri::async_runtime::block_on(
            commands::extract_images::extract_and_save_office_images_cmd(
                source_string.clone(),
                media_dir.to_string_lossy().to_string(),
                wiki_root.to_string_lossy().to_string(),
            ),
        ) {
            Ok(images) => images,
            Err(e) => {
                return error_response(500, "EXTRACT_IMAGES_FAILED", &e);
            }
        },
        _ => {
            return error_response(
                400,
                "UNSUPPORTED_SOURCE_TYPE",
                "Only pdf, pptx, docx and xlsx support explicit image extraction",
            )
        }
    };

    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "projectId": project.id,
            "sourcePath": req.source_path,
            "mediaDir": format!("wiki/media/{media_slug}"),
            "images": extracted,
        }),
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
                ["api", "v1", "projects", project_id, "graph"] => handle_graph(project_id, query),
                ["api", "v1", "projects", project_id, "vector-index"] => {
                    handle_vector_index_status(project_id)
                }
                _ => error_response(404, "NOT_FOUND", "Not found"),
            }
        }
    }
}

fn handle_post(path: &str, request: &mut tiny_http::Request) -> OmnimindResponse {
    let (clean_path, query) = split_url(path);
    let parts: Vec<&str> = clean_path
        .trim_start_matches('/')
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();

    match parts.as_slice() {
        ["api", "v1", "document", "upload"] => handle_upload(request, query),
        ["api", "v1", "files", "save"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_save_file(&body)
        }
        ["api", "v1", "files", "extract-text"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_extract_text(&body)
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
        ["api", "v1", "projects", project_id, "vector-replace"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_vector_replace_all_chunks(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "vector-delete"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_vector_delete_page(project_id, &body)
        }
        ["api", "v1", "projects", project_id, "vector-clear"] => {
            handle_vector_clear_chunks(project_id)
        }
        ["api", "v1", "projects", project_id, "vector-optimize"] => {
            handle_vector_optimize_chunks(project_id)
        }
        ["api", "v1", "projects", project_id, "extract-images"] => {
            let body = read_request_body(request).unwrap_or_default();
            handle_extract_images(project_id, &body)
        }
        ["api", "v1", "projects", _project_id, "chat"] => error_response(
            501,
            "CHAT_NOT_IMPLEMENTED",
            "LLM Wiki server-only 当前只提供知识上下文构建，不直接生成最终聊天回复。",
        ),
        _ => error_response(404, "NOT_FOUND", "Not found"),
    }
}

fn handle_upload(request: &mut tiny_http::Request, query: &str) -> OmnimindResponse {
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

    // Prefer explicit projectId (query) so upload/get_file_content share the same project.
    // Fallback: current project → first project → auto-create default workspace.
    let params = parse_query(query);
    let project = match get_target_project(params.get("projectId").map(String::as_str)) {
        Ok(p) => p,
        Err(_) => {
            // Unknown projectId still falls back for light single-workspace installs;
            // auto-create only when no projects exist at all.
            match load_projects()
                .into_iter()
                .find(|p| p.current)
                .or_else(|| load_projects().first().cloned())
            {
                Some(p) => p,
                None => {
                    let app_dir = get_app_data_dir().unwrap_or_else(|| {
                        let home = env::var("HOME")
                            .ok()
                            .or_else(|| env::var("USERPROFILE").ok())
                            .unwrap_or_else(|| ".".to_string());
                        PathBuf::from(home).join("Library/Application Support/com.llmwiki.app")
                    });
                    let default_path = app_dir.join("default-workspace");
                    let _ = fs::create_dir_all(&default_path);
                    let path_str = normalize_path(&default_path.to_string_lossy());
                    ProjectEntry {
                        id: "default".to_string(),
                        name: "Default Workspace".to_string(),
                        current: true,
                        path: path_str,
                    }
                }
            }
        }
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
    #[serde(default)]
    mode: commands::search::SearchMode,
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
    let mode = req.mode;

    let query_embedding = if mode.uses_vector() {
        let embedding_config = load_embedding_config();
        match tauri::async_runtime::block_on(commands::search::resolve_query_embedding(
            &query,
            req.query_embedding,
            embedding_config,
        )) {
            Ok(embedding) => embedding,
            Err(e) => return error_response(400, "EMBEDDING_ERROR", &e),
        }
    } else {
        None
    };

    match tauri::async_runtime::block_on(commands::search::search_project_by_mode_inner(
        project.path.clone(),
        query,
        top_k,
        req.include_content.unwrap_or(false),
        query_embedding,
        None,
        mode,
    )) {
        Ok(search) => OmnimindResponse {
            status: 200,
            body: search_response_body(&project.id, search),
        },
        Err(e) => error_response(500, "SEARCH_FAILED", &e),
    }
}

fn search_response_body(
    project_id: &str,
    search: commands::search::ProjectSearchResponse,
) -> Value {
    json!({
        "ok": true,
        "projectId": project_id,
        "mode": search.mode,
        "requestedMode": search.requested_mode,
        "executedMode": search.executed_mode,
        "tokenHits": search.token_hits,
        "vectorHits": search.vector_hits,
        "graphHits": search.graph_hits,
        "results": search.results,
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiGraphNode {
    id: String,
    label: String,
    node_type: String,
    path: String,
    link_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiGraphEdge {
    source: String,
    target: String,
    weight: f64,
}

#[derive(Deserialize)]
struct VectorUpsertRequest {
    page_id: String,
    chunks: Vec<commands::vectorstore::ChunkUpsertInput>,
}

#[derive(Deserialize)]
struct VectorReplaceRequest {
    pages: Vec<commands::vectorstore::PageChunkReplaceInput>,
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

/// 把后台预计算完成的全部页面向量作为一个 LanceDB 版本原子替换。
/// 解析、项目解析或 Core 写入失败时都不回显向量、正文或内部文件路径。
fn handle_vector_replace_all_chunks(project_id: &str, body: &str) -> OmnimindResponse {
    let request: VectorReplaceRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return error_response(400, "INVALID_VECTOR_REPLACE", "Invalid vector payload"),
    };
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_replace_all_chunks(
        project.path.clone(),
        request.pages,
    )) {
        Ok(()) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "project_id": project_id }),
        },
        Err(_) => error_response(
            500,
            "VECTOR_INDEX_REPLACE_FAILED",
            "Unable to replace vector index",
        ),
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

/// 返回 Core 原生分块向量索引的当前行数。
///
/// 该接口只暴露索引统计，不返回任何向量或知识正文，供 Python 的持久重建状态机
/// 做完成校验；索引数据仍完全由 Wiki Core/LanceDB 管理。
fn handle_vector_index_status(project_id: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_count_chunks(
        project.path.clone(),
    )) {
        Ok(chunk_count) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "project_id": project_id, "chunk_count": chunk_count }),
        },
        Err(_) => error_response(
            500,
            "VECTOR_INDEX_STATUS_FAILED",
            "Unable to read vector index status",
        ),
    }
}

/// 清空 Core 原生分块索引。
///
/// Python 重建器必须先把全部页面切块和向量化成功，才会调用本接口；这样 Embedding
/// Provider 暂时失败时不会提前破坏仍可用的旧索引。
fn handle_vector_clear_chunks(project_id: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_clear_chunks(
        project.path.clone(),
    )) {
        Ok(()) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "project_id": project_id }),
        },
        Err(_) => error_response(
            500,
            "VECTOR_INDEX_CLEAR_FAILED",
            "Unable to clear vector index",
        ),
    }
}

/// 对 Core 原生 LanceDB 分块表执行优化。
fn handle_vector_optimize_chunks(project_id: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    match tauri::async_runtime::block_on(commands::vectorstore::vector_optimize_chunks(
        project.path.clone(),
    )) {
        Ok(()) => OmnimindResponse {
            status: 200,
            body: json!({ "ok": true, "project_id": project_id }),
        },
        Err(_) => error_response(
            500,
            "VECTOR_INDEX_OPTIMIZE_FAILED",
            "Unable to optimize vector index",
        ),
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
    if request.max_block_chars == Some(0) {
        return error_response(
            400,
            "INVALID_MAX_BLOCK_CHARS",
            "max_block_chars must be greater than zero",
        );
    }
    let normalized_allowed_source_paths =
        match normalize_allowed_source_paths(request.allowed_source_paths.as_deref()) {
            Ok(paths) => paths,
            Err(message) => {
                return error_response(400, "INVALID_ALLOWED_SOURCE_PATHS", message);
            }
        };
    let validated_source_scope =
        match validate_allowed_source_files(&project.path, normalized_allowed_source_paths) {
            Ok(scope) => scope,
            Err(message) => {
                return error_response(400, "INVALID_ALLOWED_SOURCE_PATHS", message);
            }
        };
    let allowed_source_paths = validated_source_scope.allowed_paths;

    let budget = compute_context_budget(request.max_context_size);
    let effective_max_block_chars =
        compute_effective_max_block_chars(request.max_block_chars, &budget);
    let top_k = request
        .top_k
        .unwrap_or(DEFAULT_CHAT_CONTEXT_TOP_K)
        .clamp(1, MAX_CHAT_CONTEXT_TOP_K);
    let retrieval_mode = parse_retrieval_mode(request.retrieval_mode.as_deref());
    let mode_scan_roots = scan_roots_for_mode(retrieval_mode);
    let scan_roots = resolve_chat_context_scan_roots(retrieval_mode, allowed_source_paths.as_ref());
    let mode_label = retrieval_mode_label(retrieval_mode);
    let response_context = ChatContextResponseContext {
        project_id,
        request: &request,
        budget: &budget,
        effective_max_block_chars,
        mode_label,
        mode_scan_roots: &mode_scan_roots,
    };

    if scan_roots.is_empty() {
        // 显式 allowlist 与 retrieval mode 没有交集时必须在搜索前失败关闭。不能把空
        // roots 传给 Core 搜索器，因为其旧兼容合同会把空值回退为默认 wiki 全库。
        return response_context.build(
            Vec::new(),
            Vec::new(),
            SourceFilterStats {
                active: true,
                allowed_count: allowed_source_paths.as_ref().map_or(0, BTreeSet::len),
                candidate_count: 0,
                matched_count: 0,
            },
            None,
        );
    }

    // Perform real search (context assembly only — no final reply generation).
    let embedding_config = load_embedding_config();
    // Vector embeddings only help when wiki is in the scan set (wiki index).
    let query_embedding = if matches!(retrieval_mode, RetrievalMode::SourcesOnly) {
        None
    } else {
        match tauri::async_runtime::block_on(commands::search::resolve_query_embedding_strict(
            &request.query,
            request.query_embedding.clone(),
            embedding_config,
        )) {
            Ok(embedding) => embedding,
            Err(error) => return retrieval_failed_response(&error),
        }
    };

    let search_result = if allowed_source_paths.is_some() {
        tauri::async_runtime::block_on(commands::search::search_project_scoped_inner(
            validated_source_scope.project_root,
            request.query.clone(),
            top_k,
            true, // include_content
            query_embedding,
            scan_roots.clone(),
        ))
    } else {
        // allowlist 省略时继续走历史搜索入口，向量查询保持全项目行为。
        tauri::async_runtime::block_on(commands::search::search_project_inner(
            validated_source_scope.project_root,
            request.query.clone(),
            top_k,
            true,
            query_embedding,
            Some(scan_roots.clone()),
        ))
    };
    let search = match search_result {
        Ok(res) => res,
        Err(error) => return retrieval_failed_response(&error),
    };

    // 资源范围过滤必须发生在混合模式重排与上下文装箱之前。这样无论候选来自
    // BM25、Vector、Graph 还是 RRF，都没有未授权候选借排序或预算分支进入 Prompt。
    // 当 allowlist 已提供但没有任何精确匹配时，`filtered_results` 保持为空；后续会
    // 返回合法 EMPTY_CONTEXT，绝不为了“有答案”而放宽回全库。
    let (filtered_results, source_filter_stats) =
        filter_results_by_allowed_source_paths(search.results, allowed_source_paths.as_ref());
    let ordered = order_results_for_context(retrieval_mode, filtered_results);
    let (context_blocks, references) =
        assemble_context_blocks(ordered, &budget, effective_max_block_chars);

    response_context.build(
        context_blocks,
        references,
        source_filter_stats,
        Some(RetrievalSearchStats {
            search_mode: search.mode.as_str(),
            vector_hits: search.vector_hits,
            token_hits: search.token_hits,
            graph_hits: search.graph_hits,
        }),
    )
}

/// 统一构造 chat-context 的成功或合法空上下文响应。
///
/// 正常搜索与“模式不兼容、搜索前失败关闭”共用这一窄函数，保证二者的状态、预算、
/// references 和诊断信封完全一致，避免早返回分支复制协议字段后逐渐漂移。
struct ChatContextResponseContext<'a> {
    project_id: &'a str,
    request: &'a ChatContextRequest,
    budget: &'a ContextBudget,
    effective_max_block_chars: Option<usize>,
    mode_label: &'a str,
    /// 仅保存模式级安全根用于诊断，不能放入具体 allowlist 文件路径。
    mode_scan_roots: &'a [String],
}

impl ChatContextResponseContext<'_> {
    fn build(
        &self,
        context_blocks: Vec<Value>,
        references: Vec<Value>,
        source_filter_stats: SourceFilterStats,
        search_stats: Option<RetrievalSearchStats<'_>>,
    ) -> OmnimindResponse {
        let status = if context_blocks.is_empty() {
            "EMPTY_CONTEXT"
        } else {
            "SUCCESS"
        };
        let result_count = context_blocks.len();

        OmnimindResponse {
            status: 200,
            body: json!({
                "ok": true,
                "status": status,
                "project_id": self.project_id,
                "query": self.request.query,
                "context_blocks": context_blocks,
                "references": references,
                "budget": {
                    "max_ctx": self.budget.max_ctx,
                    "response_reserve": self.budget.response_reserve,
                    "index_budget": self.budget.index_budget,
                    "page_budget": self.budget.page_budget,
                    "max_page_size": self.budget.max_page_size,
                    "max_block_chars": self.effective_max_block_chars,
                },
                "retrieval_debug": build_retrieval_debug(
                    self.request,
                    self.mode_label,
                    self.mode_scan_roots,
                    result_count,
                    self.effective_max_block_chars,
                    source_filter_stats,
                    search_stats,
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceFilterStats {
    active: bool,
    allowed_count: usize,
    candidate_count: usize,
    matched_count: usize,
}

/// 按规范化后的项目相对路径精确过滤检索候选。
///
/// `None` 表示旧客户端省略字段，因此原样保留全部候选；`Some` 表示调用方明确启用
/// 资源范围，即使没有匹配也返回空集合。候选路径若不符合相同的安全规范，启用过滤
/// 时会被 fail-closed 排除，而不是尝试猜测或前缀匹配。
fn filter_results_by_allowed_source_paths(
    results: Vec<commands::search::ProjectSearchResult>,
    allowed_paths: Option<&BTreeSet<String>>,
) -> (
    Vec<commands::search::ProjectSearchResult>,
    SourceFilterStats,
) {
    let candidate_count = results.len();
    let Some(allowed_paths) = allowed_paths else {
        return (
            results,
            SourceFilterStats {
                active: false,
                allowed_count: 0,
                candidate_count,
                matched_count: candidate_count,
            },
        );
    };

    let filtered = results
        .into_iter()
        .filter(|result| {
            normalize_project_relative_source_path(&result.path)
                .map(|path| allowed_paths.contains(&path))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let matched_count = filtered.len();
    (
        filtered,
        SourceFilterStats {
            active: true,
            allowed_count: allowed_paths.len(),
            candidate_count,
            matched_count,
        },
    )
}

/// True when a search hit path belongs to the raw sources tree.
fn is_raw_sources_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    normalized.contains("raw/sources")
}

/// Hybrid mode reserves a floor of raw/sources hits (when present), then fills
/// remaining slots by score. Wiki-only / sources_only keep original ranking.
fn order_results_for_context(
    mode: RetrievalMode,
    results: Vec<commands::search::ProjectSearchResult>,
) -> Vec<commands::search::ProjectSearchResult> {
    if !matches!(mode, RetrievalMode::Hybrid) || results.is_empty() {
        return results;
    }

    let mut sources: Vec<commands::search::ProjectSearchResult> = Vec::new();
    let mut others: Vec<commands::search::ProjectSearchResult> = Vec::new();
    for item in results {
        if is_raw_sources_path(&item.path) {
            sources.push(item);
        } else {
            others.push(item);
        }
    }

    if sources.is_empty() {
        // No sources hits — keep wiki ranking as-is (others already score-ordered).
        return others;
    }

    // Search results arrive score-desc; preserve that within each bucket.
    let floor_n = HYBRID_SOURCES_FLOOR.min(sources.len());
    let mut floor: Vec<commands::search::ProjectSearchResult> = sources.drain(..floor_n).collect();
    let mut remainder = others;
    remainder.append(&mut sources);
    remainder.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });

    floor.append(&mut remainder);
    floor
}

fn assemble_context_blocks(
    ordered: Vec<commands::search::ProjectSearchResult>,
    budget: &ContextBudget,
    max_block_chars: Option<usize>,
) -> (Vec<Value>, Vec<Value>) {
    let mut context_blocks = Vec::new();
    let mut references = Vec::new();
    let mut current_size = 0;

    for res in ordered {
        // id 必须按最终实际装入顺序连续生成；否则被预算跳过的候选会留下不可引用的空洞。
        let ref_id = context_blocks.len().saturating_add(1);
        let content = res.content.clone().unwrap_or_default();
        let header = format!("[{}] {}\n", ref_id, res.title);
        let header_size = header.chars().count();

        if let Some(per_block_cap) = max_block_chars {
            // 显式单块上限是 OmniMind 的模型预算扩展：每个宽泛页面最多消耗固定正文预算，
            // 截取后仍继续尝试后排候选，避免第一篇产品总览独占全部上下文。
            let remaining = budget.page_budget.saturating_sub(current_size);
            if remaining <= header_size {
                break;
            }
            let content_cap = per_block_cap.min(remaining.saturating_sub(header_size));
            let truncated_content: String = content.chars().take(content_cap).collect();
            let block_size = header_size.saturating_add(truncated_content.chars().count());
            push_context_result(
                &mut context_blocks,
                &mut references,
                ref_id,
                &res,
                truncated_content,
            );
            current_size = current_size.saturating_add(block_size);
            continue;
        }

        // 未提供新字段时保留上游既有装箱语义：完整页优先；首块过大时按原预算截断后结束。
        let block_size = header_size.saturating_add(content.chars().count());

        if current_size.saturating_add(block_size) > budget.page_budget {
            if current_size == 0 {
                // 避免先拼出可能很大的完整 block 再截断；直接按字符链收集受控前缀。
                let truncated: String = header
                    .chars()
                    .chain(content.chars())
                    .take(budget.max_page_size)
                    .collect();
                push_context_result(
                    &mut context_blocks,
                    &mut references,
                    ref_id,
                    &res,
                    truncated,
                );
            }
            break;
        }

        push_context_result(&mut context_blocks, &mut references, ref_id, &res, content);
        current_size = current_size.saturating_add(block_size);
    }

    (context_blocks, references)
}

/// 将 Rust 原生检索证据无损投影到 chat-context 契约。
/// Python 只能消费这些候选并按最终模型预算裁剪，不能重新读取图谱来伪造来源或分数。
fn push_context_result(
    context_blocks: &mut Vec<Value>,
    references: &mut Vec<Value>,
    ref_id: usize,
    result: &commands::search::ProjectSearchResult,
    content: String,
) {
    let score_fields = json!({
        "score": result.score,
        "bm25Score": result.bm25_score,
        "vectorScore": result.vector_score,
        "graphScore": result.graph_score,
        "rrfScore": result.rrf_score,
        "graphRelatedTo": result.graph_related_to,
    });

    let mut block = score_fields.clone();
    if let Some(object) = block.as_object_mut() {
        object.insert("id".into(), json!(ref_id));
        object.insert("title".into(), json!(result.title));
        object.insert("path".into(), json!(result.path));
        object.insert("content".into(), json!(content));
    }
    context_blocks.push(block);

    let mut reference = score_fields;
    if let Some(object) = reference.as_object_mut() {
        object.insert("id".into(), json!(ref_id));
        object.insert("title".into(), json!(result.title));
        object.insert("path".into(), json!(result.path));
    }
    references.push(reference);
}

struct RetrievalSearchStats<'a> {
    search_mode: &'a str,
    vector_hits: usize,
    token_hits: usize,
    graph_hits: usize,
}

fn compute_context_budget(max_context_size: Option<usize>) -> ContextBudget {
    let max_ctx = max_context_size
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_CONTEXT_SIZE)
        .min(MAX_CHAT_CONTEXT_SIZE);
    // 即使未来上限调整，也使用饱和乘法，避免百分比预算在 usize 边界回绕。
    let response_reserve = max_ctx.saturating_mul(15) / 100;
    let index_budget = max_ctx.saturating_mul(5) / 100;
    let page_budget = max_ctx.saturating_mul(50) / 100;
    let max_page_size = page_budget.min(5_000.max(page_budget.saturating_mul(30) / 100));

    ContextBudget {
        max_ctx,
        response_reserve,
        index_budget,
        page_budget,
        max_page_size,
    }
}

/// 单块预算永远不能超过已经夹紧的总页面预算。
/// 保留 `None` 用于区分旧默认装箱语义与调用方显式启用的逐块截取语义。
fn compute_effective_max_block_chars(
    requested: Option<usize>,
    budget: &ContextBudget,
) -> Option<usize> {
    requested.map(|value| value.min(budget.page_budget))
}

fn build_retrieval_debug(
    request: &ChatContextRequest,
    retrieval_mode: &str,
    scan_roots: &[String],
    result_count: usize,
    effective_max_block_chars: Option<usize>,
    source_filter_stats: SourceFilterStats,
    search_stats: Option<RetrievalSearchStats<'_>>,
) -> Value {
    // 即使调用方没有开启详细 debug，也公开过滤是否生效，便于上层区分“全库空召回”
    // 与“资源范围内空召回”。只公开计数和布尔值，不把 allowlist 路径复制进诊断响应。
    if !request.include_debug {
        return json!({
            "retrieval_mode": retrieval_mode,
            "source_filter_active": source_filter_stats.active,
            "source_filter_allowed_count": source_filter_stats.allowed_count,
            "source_filter_matched_count": source_filter_stats.matched_count,
        });
    }

    let mut debug = json!({
        "retrieval_mode": retrieval_mode,
        "scan_roots": scan_roots,
        "result_count": result_count,
        "include_debug": true,
        "mode": "server-only",
        "history_message_count": request.history.len(),
        "history_content_chars": request.history.iter().fold(0usize, |total, item| total.saturating_add(item.content.chars().count())),
        "requested_top_k": request.top_k.unwrap_or(DEFAULT_CHAT_CONTEXT_TOP_K).clamp(1, MAX_CHAT_CONTEXT_TOP_K),
        "requested_max_context_size": request.max_context_size,
        "effective_max_context_size": compute_context_budget(request.max_context_size).max_ctx,
        "requested_max_block_chars": request.max_block_chars,
        "effective_max_block_chars": effective_max_block_chars,
        "source_filter_active": source_filter_stats.active,
        "source_filter_allowed_count": source_filter_stats.allowed_count,
        "source_filter_candidate_count": source_filter_stats.candidate_count,
        "source_filter_matched_count": source_filter_stats.matched_count,
        "source_filter_excluded_count": source_filter_stats.candidate_count.saturating_sub(source_filter_stats.matched_count),
    });

    if let Some(stats) = search_stats {
        if let Some(obj) = debug.as_object_mut() {
            obj.insert("search_mode".into(), json!(stats.search_mode));
            obj.insert("vector_hits".into(), json!(stats.vector_hits));
            obj.insert("token_hits".into(), json!(stats.token_hits));
            obj.insert("graph_hits".into(), json!(stats.graph_hits));
        }
    }

    debug
}

/// 检索失败与真实空召回必须使用不同的稳定信封。
/// 响应与 stderr 都不能回显文件内容、绝对路径或 Provider 原始异常。
fn retrieval_failed_response(internal_error: &str) -> OmnimindResponse {
    eprintln!("{}", safe_retrieval_log_line(internal_error));
    error_response(502, "RETRIEVAL_FAILED", "Knowledge retrieval failed")
}

/// 将任意内部错误投影为有限、稳定且不含原文的安全日志行。
///
/// 原始错误只参与内存中的类别判断，绝不拼接进返回值。这样即使底层错误包含绝对路径、
/// 租户文件名、供应商正文或凭据片段，stderr 也只会出现稳定原因码和安全分类。
fn safe_retrieval_log_line(internal_error: &str) -> String {
    // 分类最多查看前 4096 个字符；供应商若返回异常大的正文，也不能让日志分类阶段
    // 复制整段响应。截断内容仅驻留在本函数内，仍不会进入最终日志。
    let lower = internal_error
        .chars()
        .take(4_096)
        .collect::<String>()
        .to_ascii_lowercase();
    let category = if lower.contains("embedding") || lower.contains("provider") {
        "provider_or_embedding"
    } else if lower.contains("vector") || lower.contains("lance") {
        "vector_store"
    } else if lower.contains("scan")
        || lower.contains("file")
        || lower.contains("path")
        || lower.contains("permission")
    {
        "project_io"
    } else {
        "internal"
    };
    format!(
        "[OmniMind Server] chat-context retrieval failed code=RETRIEVAL_FAILED category={category}"
    )
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

fn safe_media_slug(file_name: &str) -> String {
    // 这里故意采用保守 ASCII slug 规则，与 Python 侧 source fallback 路径习惯保持一致，
    // 目的是降低跨语言链路里“同一源文件生成不同媒体目录名”的概率。
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(file_name);
    let mut slug = String::with_capacity(stem.len());
    let mut prev_dash = false;

    for ch in stem.chars() {
        let keep = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        if keep {
            slug.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }

    let trimmed = slug.trim_matches(|ch| ch == '-' || ch == '.' || ch == '_');
    if trimmed.is_empty() {
        "source".to_string()
    } else {
        trimmed.to_string()
    }
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

/// Resolve a project-relative or absolute path, ensuring it stays inside the
/// project root (path-traversal safe).
fn resolve_project_file_path(project_path: &str, path: &str) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("path must not be empty".to_string());
    }

    let root = PathBuf::from(project_path);
    let root_canon = root
        .canonicalize()
        .map_err(|e| format!("Failed to resolve project root: {e}"))?;

    let candidate = if Path::new(trimmed).is_absolute() {
        PathBuf::from(trimmed)
    } else {
        safe_join(project_path, trimmed)?
    };

    if !candidate.exists() {
        // Still reject absolute paths that clearly fall outside the project
        // even before existence checks, using component-normalized compare.
        if Path::new(trimmed).is_absolute() {
            let normalized = normalize_path_components(&candidate);
            let root_norm = normalize_path_components(&root_canon);
            if !path_starts_with(&normalized, &root_norm) {
                return Err("Path is outside project directory".to_string());
            }
        }
        return Err(format!("File does not exist: {}", candidate.display()));
    }

    let cand_canon = candidate
        .canonicalize()
        .map_err(|e| format!("Failed to resolve path: {e}"))?;
    if !path_starts_with(&cand_canon, &root_canon) {
        return Err("Path is outside project directory".to_string());
    }
    Ok(cand_canon)
}

fn normalize_path_components(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

fn path_starts_with(path: &Path, root: &Path) -> bool {
    let path_comps: Vec<_> = path.components().collect();
    let root_comps: Vec<_> = root.components().collect();
    if root_comps.len() > path_comps.len() {
        return false;
    }
    path_comps
        .iter()
        .zip(root_comps.iter())
        .all(|(a, b)| a == b)
}

/// Align with `commands::fs::read_file` extractable formats (not image/media placeholders).
const EXTRACT_TEXT_ALLOWED_EXTS: &[&str] = &[
    "epub", "mobi", "pdf", "doc", "docx", "pptx", "ppt", "xls", "xlsx", "odt", "ods", "odp", "txt",
    "md", "markdown", "org", "csv", "tsv", "html", "htm", "json", "xml", "yaml", "yml", "toml",
    "log", "rst", "text",
];
/// Same 100MB ceiling as ebook extraction (`commands::ebook::MAX_EBOOK_BYTES`).
const EXTRACT_CONTENT_MAX_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Deserialize)]
struct ExtractTextRequest {
    /// Mode A: project-relative or absolute path inside the wiki project.
    #[serde(default)]
    path: Option<String>,
    #[serde(default, alias = "projectId")]
    project_id: Option<String>,
    /// Mode B: file extension (without dot), e.g. `epub`, `md`.
    #[serde(default)]
    extension: Option<String>,
    /// Mode B: raw file bytes as standard base64 (DMS absolute-path workaround).
    #[serde(default, alias = "contentBase64")]
    content_base64: Option<String>,
    /// Mode B optional display name used only for `memory:<filename>` response path.
    #[serde(default)]
    filename: Option<String>,
}

fn normalize_extract_extension(raw: &str) -> String {
    raw.trim().trim_start_matches('.').to_lowercase()
}

fn is_extract_text_allowed_ext(extension: &str) -> bool {
    EXTRACT_TEXT_ALLOWED_EXTS.contains(&extension)
}

fn estimated_base64_decoded_len(b64_len: usize) -> u64 {
    // Standard base64: 4 chars → ≤3 bytes (padding ignored conservatively).
    (b64_len as u64).saturating_mul(3) / 4
}

fn content_bytes_within_limit(decoded_len: usize) -> Result<(), String> {
    if decoded_len as u64 > EXTRACT_CONTENT_MAX_BYTES {
        Err(format!(
            "Decoded content exceeds the {} MB limit",
            EXTRACT_CONTENT_MAX_BYTES / 1024 / 1024
        ))
    } else {
        Ok(())
    }
}

fn sanitize_memory_filename(filename: Option<&str>, extension: &str) -> String {
    let fallback = format!("upload.{extension}");
    let raw = filename
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    if raw.is_empty() {
        return fallback;
    }
    let base = Path::new(raw)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(raw);
    let cleaned: String = base
        .chars()
        .map(|c| match c {
            '/' | '\\' | '\0' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        fallback
    } else {
        cleaned
    }
}

fn map_read_file_error(e: &str) -> OmnimindResponse {
    let lower = e.to_lowercase();
    if lower.contains("does not exist") {
        return error_response(404, "FILE_NOT_FOUND", e);
    }
    if lower.contains("not supported") || lower.contains("unsupported") {
        return error_response(422, "UNSUPPORTED_FORMAT", e);
    }
    error_response(422, "EXTRACT_FAILED", e)
}

fn handle_extract_text(body: &str) -> OmnimindResponse {
    let req: ExtractTextRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => return error_response(400, "INVALID_JSON", &format!("Invalid JSON: {e}")),
    };

    let has_content = req
        .content_base64
        .as_ref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    if has_content {
        return handle_extract_text_from_content(&req);
    }

    handle_extract_text_from_path(&req)
}

/// Mode B: decode base64 → temp file → reuse `commands::fs::read_file` → cleanup.
fn handle_extract_text_from_content(req: &ExtractTextRequest) -> OmnimindResponse {
    let mut extension = req
        .extension
        .as_deref()
        .map(normalize_extract_extension)
        .filter(|s| !s.is_empty())
        .unwrap_or_default();

    if extension.is_empty() {
        if let Some(name) = req.filename.as_deref() {
            if let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str()) {
                extension = normalize_extract_extension(ext);
            }
        }
    }

    if extension.is_empty() {
        return error_response(
            400,
            "EXTENSION_REQUIRED",
            "extension is required for contentBase64 mode",
        );
    }

    if !is_extract_text_allowed_ext(&extension) {
        return error_response(
            422,
            "UNSUPPORTED_FORMAT",
            &format!("Unsupported extension for extract-text: .{extension}"),
        );
    }

    let b64 = req.content_base64.as_deref().unwrap_or("").trim();
    // Reject clearly oversized payloads before allocating decode buffer.
    if estimated_base64_decoded_len(b64.len()) > EXTRACT_CONTENT_MAX_BYTES + 3 {
        return error_response(
            413,
            "PAYLOAD_TOO_LARGE",
            &format!(
                "Decoded content exceeds the {} MB limit",
                EXTRACT_CONTENT_MAX_BYTES / 1024 / 1024
            ),
        );
    }

    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    let bytes = match B64.decode(b64.as_bytes()) {
        Ok(bytes) => bytes,
        Err(e) => {
            return error_response(400, "INVALID_BASE64", &format!("Invalid base64: {e}"));
        }
    };

    if let Err(msg) = content_bytes_within_limit(bytes.len()) {
        return error_response(413, "PAYLOAD_TOO_LARGE", &msg);
    }

    let memory_name = sanitize_memory_filename(req.filename.as_deref(), &extension);
    let temp_path = env::temp_dir().join(format!(
        "omnimind-extract-{}-{}.{}",
        process::id(),
        uuid::Uuid::new_v4(),
        extension
    ));

    if let Err(e) = fs::write(&temp_path, &bytes) {
        return error_response(
            500,
            "TEMP_WRITE_FAILED",
            &format!("Failed to write temp file: {e}"),
        );
    }

    let content = match tauri::async_runtime::block_on(commands::fs::read_file(
        temp_path.to_string_lossy().to_string(),
        Some(false),
    )) {
        Ok(text) => {
            let _ = fs::remove_file(&temp_path);
            text
        }
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            return map_read_file_error(&e);
        }
    };

    let chars = content.chars().count();
    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "path": format!("memory:{memory_name}"),
            "extension": extension,
            "content": content,
            "chars": chars,
        }),
    }
}

/// Mode A: path must resolve inside the target wiki project (unchanged contract).
fn handle_extract_text_from_path(req: &ExtractTextRequest) -> OmnimindResponse {
    let path = req.path.as_deref().unwrap_or("").trim();
    if path.is_empty() {
        return error_response(
            400,
            "PATH_REQUIRED",
            "path is required (or provide contentBase64 + extension)",
        );
    }

    let project = match get_target_project(req.project_id.as_deref()) {
        Ok(p) => p,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };

    let full_path = match resolve_project_file_path(&project.path, path) {
        Ok(p) => p,
        Err(e) => {
            let lower = e.to_lowercase();
            if lower.contains("outside project") || lower.contains("traversal") {
                return error_response(400, "INVALID_PATH", &e);
            }
            if lower.contains("does not exist") {
                return error_response(404, "FILE_NOT_FOUND", &e);
            }
            if lower.contains("empty") {
                return error_response(400, "PATH_REQUIRED", &e);
            }
            return error_response(400, "INVALID_PATH", &e);
        }
    };

    if full_path.is_dir() {
        return error_response(422, "NOT_A_FILE", "Path is a directory");
    }

    let extension = full_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Reuse commands::fs text extraction (pdf/office/epub/mobi/org/txt/md …).
    let content = match tauri::async_runtime::block_on(commands::fs::read_file(
        full_path.to_string_lossy().to_string(),
        Some(false),
    )) {
        Ok(text) => text,
        Err(e) => return map_read_file_error(&e),
    };

    let rel_path = relative_to_project(&project.path, &full_path);
    let chars = content.chars().count();

    OmnimindResponse {
        status: 200,
        body: json!({
            "ok": true,
            "path": rel_path,
            "extension": extension,
            "content": content,
            "chars": chars,
        }),
    }
}

fn handle_graph(project_id: &str, query: &str) -> OmnimindResponse {
    let project = match resolve_project(project_id) {
        Ok(project) => project,
        Err(e) => return error_response(404, "PROJECT_NOT_FOUND", &e),
    };
    let params = parse_query(query);
    let q = params.get("q").map(|s| s.to_lowercase());
    let node_type = params.get("nodeType").map(|s| s.to_lowercase());
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200)
        .clamp(1, 1000);

    match build_graph(&project.path) {
        Ok((mut nodes, edges)) => {
            if let Some(ref q) = q {
                nodes.retain(|n| {
                    n.id.to_lowercase().contains(q) || n.label.to_lowercase().contains(q)
                });
            }
            if let Some(ref node_type) = node_type {
                nodes.retain(|n| n.node_type == *node_type);
            }
            // Prefer highly-connected pages before applying limit so small
            // limits do not collapse to isolated leaves.
            nodes.sort_by(|a, b| {
                b.link_count
                    .cmp(&a.link_count)
                    .then_with(|| a.id.cmp(&b.id))
            });
            nodes.truncate(limit);
            let ids: std::collections::BTreeSet<String> =
                nodes.iter().map(|n| n.id.clone()).collect();
            let edges: Vec<ApiGraphEdge> = edges
                .into_iter()
                .filter(|e| ids.contains(&e.source) && ids.contains(&e.target))
                .collect();
            OmnimindResponse {
                status: 200,
                body: json!({ "ok": true, "projectId": project.id, "nodes": nodes, "edges": edges }),
            }
        }
        Err(e) => error_response(500, "GRAPH_BUILD_FAILED", &e),
    }
}

fn build_graph(project_path: &str) -> Result<(Vec<ApiGraphNode>, Vec<ApiGraphEdge>), String> {
    let wiki_root = Path::new(project_path).join("wiki");
    let mut raw: BTreeMap<String, (String, String, String, Vec<String>)> = BTreeMap::new();
    for entry in walkdir::WalkDir::new(&wiki_root)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file()
            || entry.path().extension().and_then(|s| s.to_str()) != Some("md")
        {
            continue;
        }
        let content = match fs::read_to_string(entry.path()) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let id = entry
            .path()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if id.is_empty() {
            continue;
        }
        let title =
            commands::search::extract_title(&content, entry.file_name().to_string_lossy().as_ref());
        let node_type = extract_type(&content);
        let path = relative_to_project(project_path, entry.path());
        let links = extract_wikilinks(&content);
        raw.insert(id, (title, node_type, path, links));
    }
    let ids: std::collections::BTreeSet<String> = raw.keys().cloned().collect();
    let mut link_count: BTreeMap<String, usize> = raw.keys().map(|id| (id.clone(), 0)).collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut edges = Vec::new();
    for (source, (_, _, _, links)) in &raw {
        for link in links {
            let Some(target) = resolve_link(link, &ids) else {
                continue;
            };
            if &target == source {
                continue;
            }
            let key = if source < &target {
                format!("{source}::{target}")
            } else {
                format!("{target}::{source}")
            };
            if seen.insert(key) {
                *link_count.entry(source.clone()).or_default() += 1;
                *link_count.entry(target.clone()).or_default() += 1;
                edges.push(ApiGraphEdge {
                    source: source.clone(),
                    target,
                    weight: 1.0,
                });
            }
        }
    }
    let nodes = raw
        .into_iter()
        .filter(|(_, (_, node_type, _, _))| node_type != "query")
        .map(|(id, (label, node_type, path, _))| ApiGraphNode {
            link_count: *link_count.get(&id).unwrap_or(&0),
            id,
            label,
            node_type,
            path,
        })
        .collect();
    Ok((nodes, edges))
}

/// Extract `type` only from YAML frontmatter (`---` … `---` at file start).
/// Body text that happens to contain `type:` is ignored; default is `other`.
fn extract_type(content: &str) -> String {
    let Some(frontmatter) = yaml_frontmatter_block(content) else {
        return "other".to_string();
    };
    for line in frontmatter.lines() {
        if let Some(value) = line.trim().strip_prefix("type:") {
            let parsed = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_lowercase();
            if !parsed.is_empty() {
                return parsed;
            }
        }
    }
    "other".to_string()
}

/// Return the YAML frontmatter body (between opening/closing `---` fences)
/// when the document starts with a frontmatter block.
fn yaml_frontmatter_block(content: &str) -> Option<&str> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let after_open = if let Some(rest) = content.strip_prefix("---\r\n") {
        rest
    } else if let Some(rest) = content.strip_prefix("---\n") {
        rest
    } else {
        return None;
    };

    // Closing fence must be on its own line: `\n---` at EOL or followed by
    // newline / EOF.
    let bytes = after_open.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let rest = &after_open[i + 1..];
            if rest == "---"
                || rest.starts_with("---\n")
                || rest.starts_with("---\r\n")
                || rest.starts_with("---\r")
            {
                return Some(&after_open[..i]);
            }
        }
        i += 1;
    }
    None
}

fn extract_wikilinks(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find("[[") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("]]") else {
            break;
        };
        let inner = &rest[..end];
        let target = inner.split('|').next().unwrap_or("").trim();
        if !target.is_empty() {
            out.push(target.to_string());
        }
        rest = &rest[end + 2..];
    }
    out
}

fn resolve_link(raw: &str, ids: &std::collections::BTreeSet<String>) -> Option<String> {
    if ids.contains(raw) {
        return Some(raw.to_string());
    }
    let normalized = raw.to_lowercase().replace(' ', "-");
    ids.iter()
        .find(|id| id.to_lowercase() == normalized || id.to_lowercase() == raw.to_lowercase())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use commands::search::ProjectSearchResult;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(prefix: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("omnimind-server-{prefix}-{id}-{seq}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn sample_result(path: &str, title: &str, score: f64) -> ProjectSearchResult {
        ProjectSearchResult {
            path: path.to_string(),
            title: title.to_string(),
            snippet: String::new(),
            title_match: false,
            score,
            bm25_score: Some(score),
            vector_score: None,
            graph_score: None,
            rrf_score: None,
            images: vec![],
            content: Some(format!("content for {title}")),
            graph_related_to: vec![],
        }
    }

    #[test]
    fn parse_retrieval_mode_defaults_and_aliases() {
        assert_eq!(parse_retrieval_mode(None), RetrievalMode::Wiki);
        assert_eq!(parse_retrieval_mode(Some("")), RetrievalMode::Wiki);
        assert_eq!(parse_retrieval_mode(Some("wiki")), RetrievalMode::Wiki);
        assert_eq!(parse_retrieval_mode(Some("WiKi")), RetrievalMode::Wiki);
        assert_eq!(
            parse_retrieval_mode(Some("sources_only")),
            RetrievalMode::SourcesOnly
        );
        assert_eq!(
            parse_retrieval_mode(Some("faithful")),
            RetrievalMode::SourcesOnly
        );
        assert_eq!(
            parse_retrieval_mode(Some("sources")),
            RetrievalMode::SourcesOnly
        );
        assert_eq!(parse_retrieval_mode(Some("hybrid")), RetrievalMode::Hybrid);
        assert_eq!(parse_retrieval_mode(Some("all")), RetrievalMode::Hybrid);
        assert_eq!(parse_retrieval_mode(Some("unknown")), RetrievalMode::Wiki);
    }

    #[test]
    fn search_request_accepts_only_the_four_operator_modes() {
        for mode in ["bm25", "vector", "graph", "hybrid"] {
            let body = format!(r#"{{"query":"alpha","mode":"{mode}"}}"#);
            assert!(serde_json::from_str::<SearchRequest>(&body).is_ok());
        }
        assert!(
            serde_json::from_str::<SearchRequest>(r#"{"query":"alpha","mode":"semantic"}"#)
                .is_err()
        );
    }

    #[test]
    fn search_response_body_uses_camel_case_mode_and_nullable_scores() {
        let search = commands::search::ProjectSearchResponse {
            mode: "bm25".to_string(),
            requested_mode: commands::search::SearchMode::Bm25,
            executed_mode: commands::search::SearchMode::Bm25,
            results: vec![sample_result("wiki/result.md", "Result", 7.0)],
            token_hits: 1,
            vector_hits: 0,
            graph_hits: 0,
        };

        let body = search_response_body("project-1", search);

        assert_eq!(body["requestedMode"], "bm25");
        assert_eq!(body["executedMode"], "bm25");
        assert_eq!(body["results"][0]["bm25Score"], 7.0);
        assert!(body["results"][0]["vectorScore"].is_null());
        assert!(body["results"][0]["graphScore"].is_null());
        assert!(body["results"][0]["rrfScore"].is_null());
    }

    #[test]
    fn scan_roots_for_mode_maps_expected_paths() {
        assert_eq!(scan_roots_for_mode(RetrievalMode::Wiki), vec!["wiki"]);
        assert_eq!(
            scan_roots_for_mode(RetrievalMode::SourcesOnly),
            vec!["raw/sources"]
        );
        assert_eq!(
            scan_roots_for_mode(RetrievalMode::Hybrid),
            vec!["wiki", "raw/sources"]
        );
    }

    #[test]
    fn retrieval_debug_always_includes_mode() {
        let request = ChatContextRequest {
            query: "q".into(),
            query_embedding: None,
            history: vec![],
            max_history_messages: None,
            max_context_size: None,
            top_k: None,
            max_block_chars: None,
            output_language: None,
            include_debug: false,
            retrieval_mode: Some("sources_only".into()),
            allowed_source_paths: None,
        };
        let inactive_filter = SourceFilterStats {
            active: false,
            allowed_count: 0,
            candidate_count: 0,
            matched_count: 0,
        };
        let compact = build_retrieval_debug(
            &request,
            "sources_only",
            &["raw/sources".into()],
            0,
            None,
            inactive_filter,
            None,
        );
        assert_eq!(compact["retrieval_mode"], "sources_only");
        assert_eq!(compact["source_filter_active"], false);
        assert_eq!(compact["source_filter_allowed_count"], 0);
        assert_eq!(compact["source_filter_matched_count"], 0);
        assert!(compact.get("scan_roots").is_none());
        assert!(compact.get("search_mode").is_none());

        let request_debug = ChatContextRequest {
            include_debug: true,
            ..request
        };
        let full = build_retrieval_debug(
            &request_debug,
            "sources_only",
            &["raw/sources".into()],
            3,
            None,
            SourceFilterStats {
                active: true,
                allowed_count: 2,
                candidate_count: 5,
                matched_count: 3,
            },
            Some(RetrievalSearchStats {
                search_mode: "hybrid",
                vector_hits: 4,
                token_hits: 7,
                graph_hits: 1,
            }),
        );
        assert_eq!(full["retrieval_mode"], "sources_only");
        assert_eq!(full["result_count"], 3);
        assert_eq!(full["scan_roots"][0], "raw/sources");
        assert_eq!(full["include_debug"], true);
        assert_eq!(full["search_mode"], "hybrid");
        assert_eq!(full["vector_hits"], 4);
        assert_eq!(full["token_hits"], 7);
        assert_eq!(full["graph_hits"], 1);
        assert_eq!(full["source_filter_active"], true);
        assert_eq!(full["source_filter_allowed_count"], 2);
        assert_eq!(full["source_filter_candidate_count"], 5);
        assert_eq!(full["source_filter_matched_count"], 3);
        assert_eq!(full["source_filter_excluded_count"], 2);
        assert!(
            !full.to_string().contains("raw/sources/private.md"),
            "过滤诊断不得回显 allowlist 中的具体来源路径"
        );
    }

    #[test]
    fn extract_type_reads_only_yaml_frontmatter() {
        let with_fm = "---\ntitle: Demo\ntype: concept\n---\n# Body\ntype: ignored\n";
        assert_eq!(extract_type(with_fm), "concept");

        let quoted = "---\ntype: \"entity\"\n---\nbody type: concept\n";
        assert_eq!(extract_type(quoted), "entity");

        let no_fm = "# Title\ntype: concept\n";
        assert_eq!(extract_type(no_fm), "other");

        let fm_without_type = "---\ntitle: only\n---\ntype: concept\n";
        assert_eq!(extract_type(fm_without_type), "other");
    }

    #[test]
    fn graph_nodes_prioritize_high_link_count_before_limit() {
        let root = test_dir("graph");
        let wiki = root.join("wiki");
        fs::create_dir_all(&wiki).unwrap();

        // Hub links to A and B → high degree. Leaves have fewer links.
        fs::write(
            wiki.join("hub.md"),
            "---\ntype: concept\n---\n# Hub\n[[leaf-a]] [[leaf-b]]\n",
        )
        .unwrap();
        fs::write(
            wiki.join("leaf-a.md"),
            "---\ntype: concept\n---\n# Leaf A\n[[hub]]\n",
        )
        .unwrap();
        fs::write(
            wiki.join("leaf-b.md"),
            "---\ntype: concept\n---\n# Leaf B\n[[hub]]\n",
        )
        .unwrap();
        fs::write(
            wiki.join("orphan.md"),
            "---\ntype: concept\n---\n# Orphan\nNo links here.\n",
        )
        .unwrap();

        let (mut nodes, _edges) = build_graph(root.to_str().unwrap()).unwrap();
        nodes.sort_by(|a, b| {
            b.link_count
                .cmp(&a.link_count)
                .then_with(|| a.id.cmp(&b.id))
        });
        nodes.truncate(2);

        assert_eq!(nodes.len(), 2);
        assert!(
            nodes.iter().all(|n| n.link_count >= 1),
            "limit=2 must prefer connected pages over orphan: {:?}",
            nodes
        );
        assert!(
            nodes.iter().any(|n| n.id == "hub"),
            "hub should survive truncate: {:?}",
            nodes
        );
        assert!(
            nodes.iter().all(|n| n.id != "orphan"),
            "orphan must not beat connected nodes: {:?}",
            nodes
        );
        assert!(
            nodes.iter().all(|n| n.node_type == "concept"),
            "frontmatter type must apply: {:?}",
            nodes
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn hybrid_sources_floor_keeps_raw_sources_hits() {
        let results = vec![
            sample_result("wiki/alpha.md", "Alpha", 100.0),
            sample_result("wiki/beta.md", "Beta", 90.0),
            sample_result("raw/sources/doc1.md", "Doc1", 50.0),
            sample_result("raw/sources/doc2.md", "Doc2", 40.0),
            sample_result("raw/sources/doc3.md", "Doc3", 30.0),
            sample_result("wiki/gamma.md", "Gamma", 20.0),
        ];

        let ordered = order_results_for_context(RetrievalMode::Hybrid, results);
        assert!(ordered.len() >= 3);
        // Floor of 2 sources first.
        assert!(is_raw_sources_path(&ordered[0].path));
        assert!(is_raw_sources_path(&ordered[1].path));
        assert_eq!(ordered[0].path, "raw/sources/doc1.md");
        assert_eq!(ordered[1].path, "raw/sources/doc2.md");
        // Remaining ordered by score: wiki/alpha (100), wiki/beta (90), doc3 (30), gamma (20)
        assert_eq!(ordered[2].path, "wiki/alpha.md");
        assert_eq!(ordered[3].path, "wiki/beta.md");

        // wiki-only leaves ranking untouched.
        let wiki_only = order_results_for_context(
            RetrievalMode::Wiki,
            vec![
                sample_result("wiki/a.md", "A", 10.0),
                sample_result("raw/sources/s.md", "S", 99.0),
            ],
        );
        assert_eq!(wiki_only[0].path, "wiki/a.md");
        assert_eq!(wiki_only[1].path, "raw/sources/s.md");

        // sources_only unchanged.
        let sources_only = order_results_for_context(
            RetrievalMode::SourcesOnly,
            vec![
                sample_result("raw/sources/a.md", "A", 10.0),
                sample_result("raw/sources/b.md", "B", 5.0),
            ],
        );
        assert_eq!(sources_only[0].path, "raw/sources/a.md");
        assert_eq!(sources_only[1].path, "raw/sources/b.md");
    }

    #[test]
    fn hybrid_sources_floor_constant_is_two() {
        assert_eq!(HYBRID_SOURCES_FLOOR, 2);
    }

    #[test]
    fn chat_context_request_new_limits_are_optional_and_default_compatible() {
        let request: ChatContextRequest =
            serde_json::from_str(r#"{"query":"门店在哪里"}"#).unwrap();

        assert_eq!(request.query_embedding, None);
        assert_eq!(request.top_k, None);
        assert_eq!(request.max_block_chars, None);
        assert_eq!(request.allowed_source_paths, None);
        assert_eq!(DEFAULT_CHAT_CONTEXT_TOP_K, 10);

        // 不仅反序列化合同保持兼容，省略字段时也必须完整保留旧有候选集合。
        let results = vec![
            sample_result("wiki/a.md", "A", 0.9),
            sample_result("raw/sources/b.md", "B", 0.8),
        ];
        let allowed = normalize_allowed_source_paths(request.allowed_source_paths.as_deref())
            .expect("省略字段不应产生校验错误");
        let (unfiltered, stats) = filter_results_by_allowed_source_paths(results, allowed.as_ref());
        assert_eq!(unfiltered.len(), 2);
        assert!(!stats.active);
        assert_eq!(stats.matched_count, 2);
    }

    #[test]
    fn allowed_source_paths_normalize_separators_and_deduplicate() {
        let requested = vec![
            "raw\\sources\\门店资料.md".to_string(),
            "raw/sources/门店资料.md".to_string(),
        ];

        let normalized = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .expect("显式 allowlist 应保持启用状态");

        assert_eq!(normalized.len(), 1);
        assert!(normalized.contains("raw/sources/门店资料.md"));
    }

    #[test]
    fn allowed_source_paths_become_exact_mode_compatible_scan_roots() {
        let requested = vec![
            "raw/sources/产品资料.md".to_string(),
            "raw/sources/服务说明.md".to_string(),
            "wiki/entities/产品.md".to_string(),
        ];
        let allowed = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .unwrap();

        assert_eq!(
            resolve_chat_context_scan_roots(RetrievalMode::SourcesOnly, Some(&allowed)),
            vec!["raw/sources/产品资料.md", "raw/sources/服务说明.md"]
        );
        assert_eq!(
            resolve_chat_context_scan_roots(RetrievalMode::Wiki, Some(&allowed)),
            vec!["wiki/entities/产品.md"]
        );
        assert_eq!(
            resolve_chat_context_scan_roots(RetrievalMode::Hybrid, Some(&allowed)),
            vec![
                "raw/sources/产品资料.md",
                "raw/sources/服务说明.md",
                "wiki/entities/产品.md",
            ]
        );

        // 省略 allowlist 时必须继续使用模式历史根，而不是收窄或改变默认行为。
        assert_eq!(
            resolve_chat_context_scan_roots(RetrievalMode::Wiki, None),
            vec!["wiki"]
        );
    }

    #[test]
    fn incompatible_allowed_paths_fail_closed_before_search() {
        let requested = vec!["raw/sources/仅原始资料.md".to_string()];
        let allowed = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .unwrap();
        let roots = resolve_chat_context_scan_roots(RetrievalMode::Wiki, Some(&allowed));
        assert!(roots.is_empty(), "wiki 模式不得扫描 raw/sources allowlist");

        let request: ChatContextRequest = serde_json::from_str(
            r#"{"query":"q","include_debug":true,"retrieval_mode":"wiki","allowed_source_paths":["raw/sources/仅原始资料.md"]}"#,
        )
        .unwrap();
        let budget = compute_context_budget(None);
        let mode_roots = vec!["wiki".to_string()];
        let response_context = ChatContextResponseContext {
            project_id: "project-1",
            request: &request,
            budget: &budget,
            effective_max_block_chars: None,
            mode_label: "wiki",
            mode_scan_roots: &mode_roots,
        };
        let response = response_context.build(
            Vec::new(),
            Vec::new(),
            SourceFilterStats {
                active: true,
                allowed_count: 1,
                candidate_count: 0,
                matched_count: 0,
            },
            None,
        );

        assert_eq!(response.status, 200);
        assert_eq!(response.body["status"], "EMPTY_CONTEXT");
        assert_eq!(response.body["context_blocks"], json!([]));
        assert_eq!(response.body["references"], json!([]));
        assert_eq!(
            response.body["retrieval_debug"]["source_filter_active"],
            true
        );
        assert_eq!(
            response.body["retrieval_debug"]["source_filter_candidate_count"],
            0
        );
        assert_eq!(
            response.body["retrieval_debug"]["scan_roots"],
            json!(["wiki"])
        );
        assert!(
            !response.body.to_string().contains("仅原始资料.md"),
            "失败关闭诊断不得回显具体 allowlist 路径"
        );
    }

    #[test]
    fn scoped_scan_finds_full_omniflux_question_even_when_global_top_twenty_excludes_target() {
        let root = test_dir("allowed-source-top-k");
        let sources = root.join("raw/sources");
        fs::create_dir_all(&sources).unwrap();

        let query = "OmniFlux Pro 的自清洁功能适合哪些使用场景？使用时有哪些注意事项？请根据知识库回答并标注引用。";
        let target_relative = "raw/sources/omniflux-pro.md";
        fs::write(
            root.join(target_relative),
            "# OmniFlux Pro 自清洁功能\n\n适合连续运行和粉尘较多的使用场景。注意事项：维护前断电，并按周期检查集尘组件。\n",
        )
        .unwrap();

        // 构造 24 个全局分数更高的干扰文件，稳定复现“目标资源排在全库 top20 外”。
        // 修复前先全库 top-k 再过滤会得到 EMPTY_CONTEXT；修复后扫描根就是目标文件，
        // top-k 只在授权候选内执行，因此完整真实问题仍能命中。
        for index in 0..24 {
            fs::write(
                sources.join(format!("distractor-{index:02}.md")),
                format!(
                    "# {query}\n\n{query}\n{query}\n这是用于验证全局排名截断的干扰资料 {index}。\n"
                ),
            )
            .unwrap();
        }

        let project_path = root.to_string_lossy().to_string();
        let global = tauri::async_runtime::block_on(commands::search::search_project_inner(
            project_path.clone(),
            query.to_string(),
            20,
            true,
            None,
            Some(vec!["raw/sources".to_string()]),
        ))
        .unwrap();
        assert_eq!(global.results.len(), 20);
        assert!(
            global
                .results
                .iter()
                .all(|item| item.path != target_relative),
            "测试前置条件必须证明目标资源确实位于全库 top20 之外"
        );

        let requested = vec![target_relative.to_string()];
        let allowed = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .unwrap();
        let scoped_roots =
            resolve_chat_context_scan_roots(RetrievalMode::SourcesOnly, Some(&allowed));
        assert_eq!(scoped_roots, vec![target_relative]);

        let scoped = tauri::async_runtime::block_on(commands::search::search_project_scoped_inner(
            project_path,
            query.to_string(),
            20,
            true,
            None,
            scoped_roots,
        ))
        .unwrap();
        let (filtered, stats) =
            filter_results_by_allowed_source_paths(scoped.results, Some(&allowed));
        let budget = compute_context_budget(None);
        let (blocks, references) = assemble_context_blocks(filtered, &budget, None);

        assert_eq!(stats.candidate_count, 1);
        assert_eq!(stats.matched_count, 1);
        assert_eq!(blocks[0]["path"], target_relative);
        assert_eq!(references[0]["path"], target_relative);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn allowed_source_paths_reject_empty_absolute_traversal_and_control_values() {
        let invalid_cases = vec![
            Vec::<String>::new(),
            vec!["".to_string()],
            vec!["   ".to_string()],
            vec!["/private/secret.md".to_string()],
            vec!["C:\\private\\secret.md".to_string()],
            vec!["raw/./sources.md".to_string()],
            vec!["raw/../secret.md".to_string()],
            vec!["raw//sources.md".to_string()],
            vec!["raw/sources/secret\0.md".to_string()],
        ];

        for invalid in invalid_cases {
            assert!(
                normalize_allowed_source_paths(Some(&invalid)).is_err(),
                "危险或非规范来源路径必须在 privileged 边界被拒绝: {invalid:?}"
            );
        }

        let too_many = (0..=MAX_ALLOWED_SOURCE_PATHS)
            .map(|index| format!("raw/sources/{index}.md"))
            .collect::<Vec<_>>();
        assert!(normalize_allowed_source_paths(Some(&too_many)).is_err());

        let too_long = vec![format!(
            "raw/sources/{}.md",
            "长".repeat(MAX_ALLOWED_SOURCE_PATH_CHARS)
        )];
        assert!(normalize_allowed_source_paths(Some(&too_long)).is_err());
    }

    #[test]
    fn explicit_source_scope_keeps_existing_files_and_skips_missing_candidates() {
        let root = test_dir("source-scope-regular-file");
        let sources = root.join("raw/sources");
        fs::create_dir_all(&sources).unwrap();
        fs::write(sources.join("产品资料.md"), "# 产品资料\n").unwrap();

        let allowed = BTreeSet::from([
            "raw/sources/产品资料.md".to_string(),
            "raw/sources/已清理资料.md".to_string(),
        ]);
        let validated =
            validate_allowed_source_files(root.to_str().unwrap(), Some(allowed)).unwrap();

        assert_eq!(
            validated.project_root,
            fs::canonicalize(&root).unwrap().to_string_lossy()
        );
        assert_eq!(
            validated.allowed_paths.unwrap(),
            BTreeSet::from(["raw/sources/产品资料.md".to_string()])
        );

        let directory = BTreeSet::from(["raw/sources".to_string()]);
        assert!(validate_allowed_source_files(root.to_str().unwrap(), Some(directory)).is_err());

        // 全部候选都已缺失时仍必须保留显式 `Some(empty)`，不能降级为 `None`；后续扫描根
        // 会因此保持空集合并返回 EMPTY_CONTEXT，而不是触发旧客户端的全库兼容分支。
        let missing = BTreeSet::from([
            "raw/sources/不存在.md".to_string(),
            "wiki/父目录也不存在/资料.md".to_string(),
        ]);
        let validated_missing =
            validate_allowed_source_files(root.to_str().unwrap(), Some(missing)).unwrap();
        let validated_missing_paths = validated_missing
            .allowed_paths
            .expect("显式白名单不能被缺失候选转换为未提供白名单");
        assert!(validated_missing_paths.is_empty());
        assert!(
            resolve_chat_context_scan_roots(RetrievalMode::Hybrid, Some(&validated_missing_paths))
                .is_empty(),
            "全缺失候选必须保持空扫描根，禁止回退全库"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn explicit_source_scope_rejects_file_directory_root_and_chained_symlinks() {
        use std::os::unix::fs::symlink;

        let root = test_dir("source-scope-symlinks");
        let sources = root.join("raw/sources");
        fs::create_dir_all(&sources).unwrap();
        fs::write(sources.join("inside.md"), "inside").unwrap();

        let outside = test_dir("source-scope-outside");
        fs::write(outside.join("outside.md"), "outside").unwrap();

        symlink("inside.md", sources.join("inside-link.md")).unwrap();
        symlink(outside.join("outside.md"), sources.join("outside-link.md")).unwrap();
        symlink(&outside, sources.join("outside-dir-link")).unwrap();
        symlink("inside-link.md", sources.join("chain-link.md")).unwrap();

        for path in [
            "raw/sources/inside-link.md",
            "raw/sources/outside-link.md",
            "raw/sources/outside-dir-link/outside.md",
            "raw/sources/chain-link.md",
        ] {
            let allowed = BTreeSet::from([path.to_string()]);
            assert!(
                validate_allowed_source_files(root.to_str().unwrap(), Some(allowed)).is_err(),
                "文件、目录或链式 symlink 均不得进入显式扫描范围: {path}"
            );
        }

        let root_link = root.with_extension("root-link");
        let _ = fs::remove_file(&root_link);
        symlink(&root, &root_link).unwrap();
        let allowed = BTreeSet::from(["raw/sources/inside.md".to_string()]);
        assert!(
            validate_allowed_source_files(root_link.to_str().unwrap(), Some(allowed)).is_err(),
            "项目根 symlink 必须在 canonical 范围建立前被拒绝"
        );

        let _ = fs::remove_file(root_link);
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(outside);
    }

    #[test]
    fn allowed_source_paths_filter_candidates_by_exact_normalized_path() {
        let requested = vec!["raw\\sources\\allowed.md".to_string()];
        let allowed = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .unwrap();
        let mut allowed_result = sample_result("raw/sources/allowed.md", "允许来源", 0.9);
        allowed_result.vector_score = Some(0.95);
        allowed_result.graph_score = Some(0.85);
        allowed_result.rrf_score = Some(0.75);
        let mut unauthorized_hybrid =
            sample_result("raw/sources/blocked.md", "未授权混合候选", 99.0);
        unauthorized_hybrid.vector_score = Some(1.0);
        unauthorized_hybrid.graph_score = Some(1.0);
        unauthorized_hybrid.rrf_score = Some(1.0);
        let results = vec![
            allowed_result,
            sample_result("raw/sources/allowed.md.backup", "相似后缀", 0.8),
            sample_result("raw/sources/sub/allowed.md", "相似文件名", 0.7),
            sample_result("wiki/allowed.md", "不同目录", 0.6),
            unauthorized_hybrid,
        ];

        let (filtered, stats) = filter_results_by_allowed_source_paths(results, Some(&allowed));

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].path, "raw/sources/allowed.md");
        assert_eq!(filtered[0].vector_score, Some(0.95));
        assert_eq!(filtered[0].graph_score, Some(0.85));
        assert_eq!(filtered[0].rrf_score, Some(0.75));
        assert_eq!(stats.allowed_count, 1);
        assert_eq!(stats.candidate_count, 5);
        assert_eq!(stats.matched_count, 1);
    }

    #[test]
    fn unmatched_allowed_source_paths_return_empty_context_without_global_fallback() {
        let requested = vec!["raw/sources/not-found.md".to_string()];
        let allowed = normalize_allowed_source_paths(Some(&requested))
            .unwrap()
            .unwrap();
        let results = vec![sample_result(
            "raw/sources/other.md",
            "全库中存在但未授权的来源",
            0.9,
        )];

        let (filtered, stats) = filter_results_by_allowed_source_paths(results, Some(&allowed));
        let budget = compute_context_budget(None);
        let (blocks, references) = assemble_context_blocks(filtered, &budget, None);

        assert!(stats.active);
        assert_eq!(stats.matched_count, 0);
        assert!(blocks.is_empty(), "未命中时不得把全库候选装入上下文");
        assert!(references.is_empty(), "未命中时不得泄漏全库引用");
    }

    #[test]
    fn chat_context_accepts_tenant_scoped_explicit_query_embedding() {
        let request: ChatContextRequest =
            serde_json::from_str(r#"{"query":"门店在哪里","query_embedding":[0.25,-0.5,0.75]}"#)
                .unwrap();

        assert_eq!(request.query_embedding, Some(vec![0.25, -0.5, 0.75]));
    }

    #[test]
    fn vector_index_management_routes_fail_closed_for_unknown_project() {
        for response in [
            handle_vector_index_status("definitely-missing-project"),
            handle_vector_clear_chunks("definitely-missing-project"),
            handle_vector_optimize_chunks("definitely-missing-project"),
            handle_vector_replace_all_chunks("definitely-missing-project", r#"{"pages":[]}"#),
        ] {
            assert_eq!(response.status, 404);
            assert_eq!(response.body["error"]["code"], "PROJECT_NOT_FOUND");
        }
    }

    #[test]
    fn extreme_context_budget_is_clamped_without_overflow_or_unbounded_allocation() {
        let body = format!(
            r#"{{"query":"门店在哪里","max_context_size":{},"max_block_chars":{}}}"#,
            usize::MAX,
            usize::MAX
        );
        let request: ChatContextRequest = serde_json::from_str(&body).unwrap();
        let budget = compute_context_budget(request.max_context_size);
        let block_cap = compute_effective_max_block_chars(request.max_block_chars, &budget);

        assert_eq!(budget.max_ctx, MAX_CHAT_CONTEXT_SIZE);
        assert!(budget.response_reserve <= MAX_CHAT_CONTEXT_SIZE);
        assert!(budget.index_budget <= MAX_CHAT_CONTEXT_SIZE);
        assert!(budget.page_budget <= MAX_CHAT_CONTEXT_SIZE);
        assert!(budget.max_page_size <= budget.page_budget);
        assert_eq!(block_cap, Some(budget.page_budget));

        // 极值只影响已夹紧的预算数值；装箱不会按请求中的 usize::MAX 预分配内存。
        let result = sample_result("wiki/sources/stores.md", "门店清单", 0.8);
        let (blocks, references) = assemble_context_blocks(vec![result], &budget, block_cap);
        assert_eq!(blocks.len(), 1);
        assert_eq!(references.len(), 1);
        assert!(blocks[0]["content"].as_str().unwrap().chars().count() <= budget.page_budget);
    }

    #[test]
    fn json_number_larger_than_usize_returns_parse_error_without_panicking() {
        let parsed = std::panic::catch_unwind(|| {
            serde_json::from_str::<ChatContextRequest>(
                r#"{"query":"q","max_context_size":184467440737095516160}"#,
            )
        });

        assert!(parsed.is_ok(), "超大 JSON 数值不得触发 panic");
        assert!(parsed.unwrap().is_err(), "超出 usize 的数值必须稳定拒绝");
    }

    #[test]
    fn explicit_per_block_cap_keeps_later_precise_source_and_native_scores() {
        let budget = ContextBudget {
            max_ctx: 500,
            response_reserve: 75,
            index_budget: 25,
            page_budget: 240,
            max_page_size: 50,
        };
        let mut broad = sample_result("wiki/entities/product.md", "产品总览", 0.9);
        broad.content = Some("宽泛产品内容".repeat(30));
        broad.graph_score = Some(0.4);
        broad.rrf_score = Some(0.9);
        broad.graph_related_to = vec!["产品".to_string()];
        let mut distractors = (2..=5)
            .map(|index| {
                let mut result = sample_result(
                    &format!("wiki/concepts/noise-{index}.md"),
                    &format!("干扰候选{index}"),
                    0.9 - index as f64 / 100.0,
                );
                result.content = Some("与当前问题相关度较低的维护内容".repeat(10));
                result
            })
            .collect::<Vec<_>>();
        let mut precise = sample_result("wiki/sources/stores.md", "门店清单", 0.8);
        precise.content = Some("北京、上海、广州、成都四家门店完整地址".to_string());
        let mut ordered = vec![broad];
        ordered.append(&mut distractors);
        ordered.push(precise);

        let (blocks, references) = assemble_context_blocks(ordered, &budget, Some(20));

        assert_eq!(blocks.len(), 6, "宽泛首块截取后仍应继续装入第六个精确来源");
        assert_eq!(blocks[0]["content"].as_str().unwrap().chars().count(), 20);
        assert_eq!(blocks[0]["path"], "wiki/entities/product.md");
        assert_eq!(blocks[0]["score"], 0.9);
        assert_eq!(blocks[0]["graphScore"], 0.4);
        assert_eq!(blocks[0]["rrfScore"], 0.9);
        assert_eq!(blocks[0]["graphRelatedTo"][0], "产品");
        assert_eq!(blocks[5]["path"], "wiki/sources/stores.md");
        assert_eq!(blocks[5]["score"], 0.8);
        assert_eq!(references[5]["path"], "wiki/sources/stores.md");
        assert_eq!(references[5]["score"], 0.8);
        assert!(references[5].get("content").is_none());
    }

    #[test]
    fn missing_per_block_cap_preserves_legacy_first_oversized_page_behavior() {
        let budget = ContextBudget {
            max_ctx: 40,
            response_reserve: 6,
            index_budget: 2,
            page_budget: 20,
            max_page_size: 10,
        };
        let mut broad = sample_result("wiki/entities/product.md", "产品总览", 0.9);
        broad.content = Some("超长内容".repeat(20));
        let precise = sample_result("wiki/sources/stores.md", "门店清单", 0.8);

        let (blocks, references) = assemble_context_blocks(vec![broad, precise], &budget, None);

        assert_eq!(blocks.len(), 1);
        assert_eq!(references.len(), 1);
        assert_eq!(blocks[0]["path"], "wiki/entities/product.md");
        assert_eq!(blocks[0]["content"].as_str().unwrap().chars().count(), 10);
    }

    #[test]
    fn retrieval_failure_uses_non_2xx_stable_envelope_without_internal_details() {
        let raw_error = "/private/workspace/secret.md: permission denied; provider body=sk-secret";
        let safe_log = safe_retrieval_log_line(raw_error);
        let response = retrieval_failed_response(raw_error);

        assert_eq!(response.status, 502);
        assert_eq!(response.body["ok"], false);
        assert_eq!(response.body["error"]["code"], "RETRIEVAL_FAILED");
        assert_eq!(
            response.body["error"]["message"],
            "Knowledge retrieval failed"
        );
        assert!(!response.body.to_string().contains("secret.md"));
        assert!(!response.body.to_string().contains("permission denied"));
        assert_eq!(
            safe_log,
            "[OmniMind Server] chat-context retrieval failed code=RETRIEVAL_FAILED category=provider_or_embedding"
        );
        for forbidden in [
            "/private/workspace",
            "secret.md",
            "permission denied",
            "provider body",
            "sk-secret",
        ] {
            assert!(
                !safe_log.contains(forbidden),
                "安全 stderr 投影不得包含原始错误片段: {forbidden}"
            );
        }
    }

    #[test]
    fn extract_text_path_resolution_rejects_traversal() {
        let root = test_dir("extract");
        fs::write(root.join("inside.md"), "hello extract").unwrap();
        let root_str = root.to_string_lossy().to_string();

        let ok = resolve_project_file_path(&root_str, "inside.md").unwrap();
        assert!(ok.ends_with("inside.md"));

        let abs_ok =
            resolve_project_file_path(&root_str, root.join("inside.md").to_str().unwrap()).unwrap();
        assert!(abs_ok.ends_with("inside.md"));

        let outside = std::env::temp_dir().join(format!(
            "omnimind-outside-{}.txt",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&outside, "secret").unwrap();
        let err = resolve_project_file_path(&root_str, outside.to_str().unwrap()).unwrap_err();
        assert!(
            err.to_lowercase().contains("outside"),
            "expected outside-project error, got {err}"
        );
        let trav = resolve_project_file_path(&root_str, "../secret.md").unwrap_err();
        assert!(
            trav.to_lowercase().contains("traversal")
                || trav.to_lowercase().contains("outside")
                || trav.to_lowercase().contains("not allowed"),
            "expected traversal rejection, got {trav}"
        );

        let _ = fs::remove_file(outside);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn extract_text_content_base64_md_extracts_text() {
        use base64::{engine::general_purpose::STANDARD as B64, Engine as _};

        let markdown = "# Hello DMS\n\ncontent from base64 mode\n";
        let body = json!({
            "extension": "md",
            "contentBase64": B64.encode(markdown.as_bytes()),
            "filename": "note.md",
        })
        .to_string();

        let resp = handle_extract_text(&body);
        assert_eq!(resp.status, 200, "body={}", resp.body);
        assert_eq!(resp.body["ok"], true);
        assert_eq!(resp.body["extension"], "md");
        assert_eq!(resp.body["path"], "memory:note.md");
        assert_eq!(resp.body["content"], markdown);
        assert_eq!(
            resp.body["chars"].as_u64().unwrap(),
            markdown.chars().count() as u64
        );
    }

    #[test]
    fn extract_text_content_mode_rejects_illegal_extension() {
        use base64::{engine::general_purpose::STANDARD as B64, Engine as _};

        let body = json!({
            "extension": "exe",
            "contentBase64": B64.encode(b"MZ fake binary"),
            "filename": "malware.exe",
        })
        .to_string();

        let resp = handle_extract_text(&body);
        assert_eq!(resp.status, 422, "body={}", resp.body);
        assert_eq!(resp.body["ok"], false);
        assert_eq!(resp.body["error"]["code"], "UNSUPPORTED_FORMAT");
    }

    #[test]
    fn extract_text_content_mode_rejects_oversized_payload() {
        // Limit aligned with ebook MAX_EBOOK_BYTES.
        assert_eq!(EXTRACT_CONTENT_MAX_BYTES, 100 * 1024 * 1024);

        // Exact post-decode gate.
        assert!(content_bytes_within_limit(EXTRACT_CONTENT_MAX_BYTES as usize).is_ok());
        let over = content_bytes_within_limit(EXTRACT_CONTENT_MAX_BYTES as usize + 1);
        assert!(over.is_err(), "expected size rejection");
        let msg = over.unwrap_err();
        assert!(msg.to_lowercase().contains("exceeds") || msg.contains("100"));

        // Pre-decode estimate: base64 length that implies >100MB must trip the gate
        // used by handle_extract_text_from_content (no multi-100MB allocation in tests).
        let over_decoded = EXTRACT_CONTENT_MAX_BYTES + 4;
        let b64_len = ((over_decoded + 2) / 3 * 4) as usize;
        let estimated = estimated_base64_decoded_len(b64_len);
        assert!(
            estimated > EXTRACT_CONTENT_MAX_BYTES + 3,
            "estimate {estimated} should exceed limit+3"
        );

        // Handler-level 413 using the same error path the estimate gate returns.
        let resp = error_response(
            413,
            "PAYLOAD_TOO_LARGE",
            &format!(
                "Decoded content exceeds the {} MB limit",
                EXTRACT_CONTENT_MAX_BYTES / 1024 / 1024
            ),
        );
        assert_eq!(resp.status, 413);
        assert_eq!(resp.body["error"]["code"], "PAYLOAD_TOO_LARGE");
    }

    #[test]
    fn extract_text_content_mode_requires_extension() {
        use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
        let body = json!({
            "contentBase64": B64.encode(b"hello"),
        })
        .to_string();
        let resp = handle_extract_text(&body);
        assert_eq!(resp.status, 400, "body={}", resp.body);
        assert_eq!(resp.body["error"]["code"], "EXTENSION_REQUIRED");
    }
}
