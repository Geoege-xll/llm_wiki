use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use walkdir::WalkDir;

use crate::commands::vectorstore;
use crate::panic_guard::run_guarded_async;

const DEFAULT_RESULTS: usize = 20;
const MAX_RESULTS: usize = 50;
const RRF_K: f64 = 60.0;
const FILENAME_EXACT_BONUS: f64 = 200.0;
const PHRASE_IN_TITLE_BONUS: f64 = 50.0;
const PHRASE_IN_CONTENT_PER_OCC: f64 = 20.0;
const MAX_PHRASE_OCC_COUNTED: usize = 10;
const TITLE_TOKEN_WEIGHT: f64 = 5.0;
const CONTENT_TOKEN_WEIGHT: f64 = 1.0;
const SNIPPET_CONTEXT: usize = 80;
const SEARCH_EMBEDDING_TIMEOUT_SECS: u64 = 8;
const MAX_SEARCH_FILES: usize = 10_000;
const MIN_GRAPH_RESULT_RATIO: f64 = 0.15;
const MAX_GRAPH_RESULT_RATIO: f64 = 0.30;
const MAX_GRAPH_SEEDS: usize = 20;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Bm25,
    Vector,
    Graph,
    #[default]
    Hybrid,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Vector => "vector",
            Self::Graph => "graph",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn uses_vector(self) -> bool {
        matches!(self, Self::Vector | Self::Hybrid)
    }

    fn uses_graph(self) -> bool {
        matches!(self, Self::Graph | Self::Hybrid)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchImageRef {
    pub url: String,
    pub alt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSearchResult {
    pub path: String,
    pub title: String,
    pub snippet: String,
    pub title_match: bool,
    pub score: f64,
    pub bm25_score: Option<f64>,
    pub vector_score: Option<f32>,
    pub graph_score: Option<f64>,
    pub rrf_score: Option<f64>,
    pub images: Vec<SearchImageRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Always present on the wire (including `[]`) for cross-language contract stability.
    /// `default` keeps older payloads without the field deserializable.
    #[serde(default)]
    pub graph_related_to: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSearchResponse {
    pub mode: String,
    pub requested_mode: SearchMode,
    pub executed_mode: SearchMode,
    pub results: Vec<ProjectSearchResult>,
    pub token_hits: usize,
    pub vector_hits: usize,
    pub graph_hits: usize,
}

#[derive(Debug, Clone)]
struct GraphPage {
    path: String,
    title: String,
    content: String,
    links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageLinkEntry {
    pub title: String,
    pub path: Option<String>,
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageLinksResponse {
    pub outgoing: Vec<PageLinkEntry>,
    pub backlinks: Vec<PageLinkEntry>,
    pub missing: Vec<PageLinkEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEmbeddingConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub output_dimensionality: Option<f64>,
    /// Extra HTTP headers to send with every embedding request, e.g.
    /// `X-Model-Provider-Id: siliconflow` for the mify gateway.
    /// Reserved names (Authorization, Content-Type, Host,
    /// Content-Length, x-goog-api-key) are skipped — they're managed
    /// by the client.
    #[serde(default)]
    pub extra_headers: Option<BTreeMap<String, String>>,
}

#[tauri::command]
pub async fn search_project(
    project_path: String,
    query: String,
    top_k: Option<usize>,
    include_content: Option<bool>,
    query_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
    mode: Option<SearchMode>,
) -> Result<ProjectSearchResponse, String> {
    run_guarded_async("search_project", async move {
        let mode = mode.unwrap_or_default();
        let query_embedding = if mode.uses_vector() {
            resolve_query_embedding(&query, query_embedding, embedding_config).await?
        } else {
            None
        };
        search_project_by_mode_inner(
            project_path,
            query,
            top_k.unwrap_or(DEFAULT_RESULTS),
            include_content.unwrap_or(false),
            query_embedding,
            None,
            mode,
        )
        .await
    })
    .await
}

#[tauri::command]
pub async fn embedding_fetch(
    text: String,
    cfg: SearchEmbeddingConfig,
    max_retries: Option<usize>,
) -> Result<Vec<f32>, String> {
    run_guarded_async("embedding_fetch", async move {
        fetch_embedding_with_retry(&text, &cfg, max_retries.unwrap_or(3)).await
    })
    .await
}

#[tauri::command]
pub async fn embedding_fetch_batch(
    texts: Vec<String>,
    cfg: SearchEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, String> {
    run_guarded_async("embedding_fetch_batch", async move {
        fetch_embedding_batch(&texts, &cfg).await
    })
    .await
}

#[tauri::command]
pub async fn get_page_links(
    project_path: String,
    file_path: String,
) -> Result<PageLinksResponse, String> {
    tokio::task::spawn_blocking(move || get_page_links_inner(&project_path, &file_path))
        .await
        .map_err(|err| format!("page links worker failed: {err}"))?
}

/// Build link relationships from Markdown source rather than the UI's lazy
/// file tree. Canonical paths enforce the project/wiki boundary, while the
/// returned paths stay project-relative so all desktop platforms share one
/// wire format and Windows separators never leak into frontend routing.
fn get_page_links_inner(project_path: &str, file_path: &str) -> Result<PageLinksResponse, String> {
    let project = fs::canonicalize(project_path)
        .map_err(|err| format!("Failed to resolve project path: {err}"))?;
    let file =
        fs::canonicalize(file_path).map_err(|err| format!("Failed to resolve page path: {err}"))?;
    let wiki_root = project.join("wiki");
    if !file.starts_with(&wiki_root)
        || !file.is_file()
        || file.extension().and_then(|value| value.to_str()) != Some("md")
    {
        return Err("Page links target must be an existing Markdown file under wiki/".to_string());
    }

    let mut pages = BTreeMap::<String, GraphPage>::new();
    let canonical_project = project.to_string_lossy();
    for entry in WalkDir::new(&wiki_root).into_iter().filter_map(Result::ok) {
        if pages.len() >= MAX_SEARCH_FILES
            || !entry.file_type().is_file()
            || entry.path().extension().and_then(|value| value.to_str()) != Some("md")
        {
            continue;
        }
        let Ok(content) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let path = relative_to_project(&canonical_project, entry.path());
        let title = extract_title(
            &content,
            entry
                .path()
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        );
        pages.insert(
            normalize_path(&path),
            GraphPage {
                path,
                title,
                links: extract_wikilinks(&content),
                content,
            },
        );
    }

    let current_path = normalize_path(&relative_to_project(&canonical_project, &file));
    let current = pages
        .get(&current_path)
        .ok_or_else(|| "Page is not available in the current wiki index".to_string())?;
    let mut outgoing = Vec::new();
    let mut missing = Vec::new();
    for link in &current.links {
        if let Some(target_path) = resolve_reader_wikilink(&pages, link) {
            if target_path.as_str() == current_path {
                continue;
            }
            if let Some(target) = pages.get(target_path) {
                outgoing.push(PageLinkEntry {
                    title: target.title.clone(),
                    path: Some(target.path.clone()),
                    snippet: None,
                });
            }
        } else {
            missing.push(PageLinkEntry {
                title: link.clone(),
                path: None,
                snippet: None,
            });
        }
    }

    let mut backlinks = Vec::new();
    for (path, page) in &pages {
        if path == &current_path {
            continue;
        }
        let links_here = page.links.iter().any(|link| {
            resolve_reader_wikilink(&pages, link)
                .is_some_and(|target| target.as_str() == current_path)
        });
        if links_here {
            backlinks.push(PageLinkEntry {
                title: page.title.clone(),
                path: Some(page.path.clone()),
                snippet: Some(build_snippet(&page.content, &current.title)),
            });
        }
    }

    outgoing.sort_by(|a, b| a.title.cmp(&b.title));
    outgoing.dedup_by(|a, b| a.path == b.path);
    backlinks.sort_by(|a, b| a.title.cmp(&b.title));
    missing.sort_by(|a, b| a.title.cmp(&b.title));
    missing.dedup_by(|a, b| a.title == b.title);
    Ok(PageLinksResponse {
        outgoing,
        backlinks,
        missing,
    })
}

/// Mirror `resolveRelatedSlug` in the frontend reader. Path-shaped links must
/// match their project-relative path exactly; bare links resolve by exact
/// filename after adding `.md`. Deliberately avoid title and fuzzy aliases so
/// the links panel never claims a target that the rendered page cannot open.
fn resolve_reader_wikilink<'a>(
    pages: &'a BTreeMap<String, GraphPage>,
    link: &str,
) -> Option<&'a String> {
    let link = link.trim().replace('\\', "/");
    if link.contains('/') {
        return pages.get_key_value(&link).map(|(path, _)| path);
    }
    let filename = if link.ends_with(".md") {
        link
    } else {
        format!("{link}.md")
    };
    pages.keys().find(|path| {
        std::path::Path::new(path.as_str())
            .file_name()
            .is_some_and(|name| name == filename.as_str())
    })
}

pub async fn resolve_query_embedding(
    query: &str,
    explicit_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
) -> Result<Option<Vec<f32>>, String> {
    resolve_query_embedding_with_policy(query, explicit_embedding, embedding_config, true).await
}

/// 为 server-only chat-context 提供严格的查询向量解析。
/// 普通桌面搜索继续允许 Provider 暂时失败后退化为关键词；自动客服上下文必须区分
/// “确实无召回”与“Embedding 检索失败”，因此这里会把 Provider 异常向上返回。
pub async fn resolve_query_embedding_strict(
    query: &str,
    explicit_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
) -> Result<Option<Vec<f32>>, String> {
    resolve_query_embedding_with_policy(query, explicit_embedding, embedding_config, false).await
}

async fn resolve_query_embedding_with_policy(
    query: &str,
    explicit_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
    allow_provider_failure_fallback: bool,
) -> Result<Option<Vec<f32>>, String> {
    if let Some(embedding) = explicit_embedding {
        return validate_query_embedding(embedding).map(Some);
    }
    let Some(cfg) = embedding_config else {
        return Ok(None);
    };
    if !cfg.enabled || cfg.endpoint.trim().is_empty() || cfg.model.trim().is_empty() {
        return Ok(None);
    }
    finish_query_embedding_fetch(
        fetch_embedding_with_retry(query, &cfg, 0).await,
        allow_provider_failure_fallback,
    )
}

fn finish_query_embedding_fetch(
    fetched: Result<Vec<f32>, String>,
    allow_provider_failure_fallback: bool,
) -> Result<Option<Vec<f32>>, String> {
    match fetched {
        Ok(embedding) => validate_query_embedding(embedding).map(Some),
        Err(err) if allow_provider_failure_fallback => {
            eprintln!("[Search] embedding disabled for this request: {err}");
            Ok(None)
        }
        Err(err) => Err(format!("query embedding failed: {err}")),
    }
}

fn validate_query_embedding(embedding: Vec<f32>) -> Result<Vec<f32>, String> {
    if embedding.is_empty() {
        return Err("queryEmbedding must not be empty".to_string());
    }
    if embedding.iter().any(|v| !v.is_finite()) {
        return Err("queryEmbedding must contain only finite numbers".to_string());
    }
    Ok(embedding)
}

/// Normalize scan roots for project-relative retrieval.
/// Defaults to `["wiki"]`. Unknown/unsafe roots are dropped; empty falls back to wiki.
pub fn normalize_scan_roots(scan_roots: Option<Vec<String>>) -> Vec<String> {
    let candidates = match scan_roots {
        Some(roots) if !roots.is_empty() => roots,
        _ => vec!["wiki".to_string()],
    };
    let mut out = Vec::new();
    for root in candidates {
        let cleaned = root.replace('\\', "/");
        let cleaned = cleaned.trim().trim_matches('/').to_string();
        if cleaned.is_empty() || cleaned.contains("..") || cleaned.starts_with('/') {
            continue;
        }
        // P0 allowlist: wiki + raw sources (and their subpaths).
        let allowed = cleaned == "wiki"
            || cleaned == "raw/sources"
            || cleaned.starts_with("wiki/")
            || cleaned.starts_with("raw/sources/");
        if !allowed {
            continue;
        }
        if !out.iter().any(|existing: &String| existing == &cleaned) {
            out.push(cleaned);
        }
    }
    if out.is_empty() {
        out.push("wiki".to_string());
    }
    out
}

fn is_wiki_scan_root(root: &str) -> bool {
    root == "wiki" || root.starts_with("wiki/")
}

pub async fn search_project_inner(
    project_path: String,
    query: String,
    top_k: usize,
    include_content: bool,
    query_embedding: Option<Vec<f32>>,
    scan_roots: Option<Vec<String>>,
) -> Result<ProjectSearchResponse, String> {
    let mut response = search_project_by_mode_inner(
        project_path,
        query,
        top_k,
        include_content,
        query_embedding,
        scan_roots,
        SearchMode::Hybrid,
    )
    .await?;
    match search_mode(
        response.token_hits == 0,
        response.vector_hits,
        response.graph_hits,
    ) {
        "keyword" => {
            response.results = rank_results_for_mode(
                response.results,
                SearchMode::Bm25,
                top_k.clamp(1, MAX_RESULTS),
            );
            response.mode = "keyword".to_string();
            response.executed_mode = SearchMode::Bm25;
        }
        "vector" => {
            response.results = rank_results_for_mode(
                response.results,
                SearchMode::Vector,
                top_k.clamp(1, MAX_RESULTS),
            );
            response.mode = "vector".to_string();
            response.executed_mode = SearchMode::Vector;
        }
        _ => {}
    }
    Ok(response)
}

pub async fn search_project_by_mode_inner(
    project_path: String,
    query: String,
    top_k: usize,
    include_content: bool,
    query_embedding: Option<Vec<f32>>,
    scan_roots: Option<Vec<String>>,
    mode: SearchMode,
) -> Result<ProjectSearchResponse, String> {
    search_project_by_mode_with_vector_search(
        project_path,
        query,
        top_k,
        include_content,
        query_embedding,
        scan_roots,
        mode,
        search_by_embedding_boxed,
    )
    .await
}

type VectorSearchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<PageVectorResult>, String>> + Send + 'a>>;
type VectorSearchFn = for<'a> fn(&'a str, Vec<f32>, usize) -> VectorSearchFuture<'a>;

async fn search_project_by_mode_with_vector_search(
    project_path: String,
    query: String,
    top_k: usize,
    include_content: bool,
    query_embedding: Option<Vec<f32>>,
    scan_roots: Option<Vec<String>>,
    mode: SearchMode,
    vector_search: VectorSearchFn,
) -> Result<ProjectSearchResponse, String> {
    if query.trim().is_empty() {
        return Err("query is required".to_string());
    }
    let limit = top_k.clamp(1, MAX_RESULTS);
    let tokens = tokenize_query(&query);
    let effective_tokens = if tokens.is_empty() {
        vec![query.trim().to_lowercase()]
    } else {
        tokens
    };
    let query_phrase = trim_query_punctuation(&query.to_lowercase());
    let mut results = Vec::new();
    let mut page_paths_by_stem = BTreeMap::new();
    let mut graph_pages = BTreeMap::new();

    let roots = normalize_scan_roots(scan_roots);
    let include_wiki = roots.iter().any(|root| is_wiki_scan_root(root));
    let mut searched_files = 0usize;

    for root_rel in &roots {
        let scan_root = Path::new(&project_path).join(root_rel);
        if !scan_root.exists() {
            // Missing roots are skipped (no 500) so sources_only/hybrid stay resilient.
            continue;
        }
        let root_is_wiki = is_wiki_scan_root(root_rel);
        for entry in WalkDir::new(&scan_root) {
            // 正式检索不能把目录遍历异常静默当成“没有知识”；上层会把该错误转换成
            // 可区分的 RETRIEVAL_FAILED，而不会返回 200 + EMPTY_CONTEXT。
            let entry = entry.map_err(|err| format!("Failed to scan retrieval root: {err}"))?;
            let ext = entry.path().extension().and_then(|s| s.to_str());
            if !entry.file_type().is_file()
                || (ext != Some("md") && ext != Some("txt"))
                || is_internal_search_artifact(
                    entry
                        .path()
                        .strip_prefix(&scan_root)
                        .unwrap_or(entry.path()),
                )
            {
                continue;
            }
            searched_files = searched_files.saturating_add(1);
            if searched_files > MAX_SEARCH_FILES {
                // 返回部分结果会让调用方无法区分“真实无召回”和“只扫描了前一部分文件”。
                // 因此超过受控上限时明确失败，由 chat-context 输出 RETRIEVAL_FAILED。
                return Err(format!(
                    "retrieval candidate limit exceeded ({MAX_SEARCH_FILES})"
                ));
            }
            let content = fs::read_to_string(entry.path())
                .map_err(|err| format!("Failed to read retrieval candidate: {err}"))?;
            let relative_path = relative_to_project(&project_path, entry.path());
            // Vector store page_ids are wiki-stem based; only index wiki stems.
            if root_is_wiki {
                if let Some(stem) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    let previous =
                        page_paths_by_stem.insert(stem.to_string(), relative_path.clone());
                    if let Some(previous) = previous {
                        eprintln!(
                            "[Search] duplicate wiki page stem '{stem}': '{previous}' and '{relative_path}' share one vector page_id"
                        );
                    }
                }
            }
            let title = extract_title(
                &content,
                entry
                    .path()
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default(),
            );
            let hit = score_file(
                &project_path,
                entry.path(),
                &content,
                &effective_tokens,
                &query_phrase,
                &query,
                include_content,
            );
            graph_pages.insert(
                normalize_path(&relative_path),
                GraphPage {
                    path: relative_path,
                    title,
                    links: extract_wikilinks(&content),
                    content,
                },
            );
            if let Some(hit) = hit {
                results.push(hit);
            }
        }
    }

    let mut token_sorted = (0..results.len()).collect::<Vec<_>>();
    token_sorted.sort_by(|a, b| {
        results[*b]
            .score
            .partial_cmp(&results[*a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| results[*a].path.cmp(&results[*b].path))
    });
    let mut token_rank = BTreeMap::new();
    for (idx, result_idx) in token_sorted.iter().enumerate() {
        let result = &results[*result_idx];
        token_rank.insert(normalize_path(&result.path), idx + 1);
    }

    let mut vector_score: BTreeMap<String, f32> = BTreeMap::new();
    let mut vector_hits = 0;
    // Vector index is wiki-page oriented; sources_only stays keyword-only (P0).
    if mode.uses_vector() && include_wiki {
        if let Some(embedding) = query_embedding {
            if !embedding.is_empty() {
                let vector_results = vector_search(&project_path, embedding, limit.max(10))
                    .await
                    .map_err(|err| format!("{} vector search failed: {err}", mode.as_str()))?;
                vector_hits = vector_results.len();
                for vr in &vector_results {
                    vector_score.insert(vr.id.clone(), vr.score);
                }
                materialize_vector_only_results(
                    &vector_results,
                    &page_paths_by_stem,
                    &project_path,
                    &mut results,
                    include_content,
                )?;
            }
        }
    }

    for result in &mut results {
        if let Some(score) = vector_score.get(&file_stem(&result.path)).copied() {
            result.vector_score = Some(score);
        }
    }

    let graph_results = if mode.uses_graph() {
        let seed_results = rank_results_for_mode(results.clone(), SearchMode::Hybrid, limit);
        collect_graph_results(
            &seed_results,
            &graph_pages,
            if mode == SearchMode::Graph {
                limit
            } else {
                graph_result_quota(limit, vector_hits)
            },
            include_content,
        )
    } else {
        Vec::new()
    };
    let graph_hits = graph_results.len();
    merge_graph_results(&mut results, graph_results);
    let results = rank_results_for_mode(results, mode, limit);

    Ok(ProjectSearchResponse {
        mode: mode.as_str().to_string(),
        requested_mode: mode,
        executed_mode: mode,
        token_hits: token_rank.len(),
        vector_hits,
        graph_hits,
        results,
    })
}

/// 判断文件是否属于编辑器临时文件、维护副本或备份目录。
/// 这里只过滤具有明确维护语义的命名，不按 `wiki/sources`、entity、concept 等业务目录过滤，
/// 因而不会误伤 LLMWiki 的合法结构化知识页。
fn is_internal_search_artifact(path: &Path) -> bool {
    let has_internal_directory = path.components().any(|component| {
        let value = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        value.starts_with('.')
            || matches!(
                value.as_str(),
                "backup" | "backups" | "_backup" | "_backups" | "tmp" | "temp"
            )
    });
    if has_internal_directory {
        return true;
    }

    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    file_name.starts_with(".#")
        || file_name.ends_with('~')
        || file_name.ends_with(".bak")
        || file_name.ends_with(".backup")
        || file_name.ends_with(".tmp")
        || file_name.ends_with(".temp")
        || file_name.contains(".bak.")
        || file_name.contains(".backup.")
        || file_name.contains(".tmp.")
        || file_name.contains(".temp.")
}

fn rank_results_for_mode(
    mut results: Vec<ProjectSearchResult>,
    mode: SearchMode,
    limit: usize,
) -> Vec<ProjectSearchResult> {
    match mode {
        SearchMode::Bm25 => {
            results.retain(|result| result.bm25_score.is_some());
            for result in &mut results {
                result.score = result.bm25_score.unwrap_or_default();
                result.vector_score = None;
                result.graph_score = None;
                result.rrf_score = None;
            }
        }
        SearchMode::Vector => {
            results.retain(|result| result.vector_score.is_some());
            for result in &mut results {
                result.score = result.vector_score.unwrap_or_default() as f64;
                result.bm25_score = None;
                result.graph_score = None;
                result.rrf_score = None;
            }
        }
        SearchMode::Graph => {
            results.retain(|result| result.graph_score.is_some());
            for result in &mut results {
                result.score = result.graph_score.unwrap_or_default();
                result.bm25_score = None;
                result.vector_score = None;
                result.rrf_score = None;
            }
        }
        SearchMode::Hybrid => {
            let bm25_rank = score_rank(&results, |result| result.bm25_score);
            let vector_rank = score_rank(&results, |result| {
                result.vector_score.map(|score| score as f64)
            });
            let graph_rank = score_rank(&results, |result| result.graph_score);
            results.retain(|result| {
                result.bm25_score.is_some()
                    || result.vector_score.is_some()
                    || result.graph_score.is_some()
            });
            for result in &mut results {
                let path = normalize_path(&result.path);
                let mut rrf = 0.0;
                for rank in [
                    bm25_rank.get(&path),
                    vector_rank.get(&path),
                    graph_rank.get(&path),
                ]
                .into_iter()
                .flatten()
                {
                    rrf += 1.0 / (RRF_K + *rank as f64);
                }
                result.score = rrf;
                result.rrf_score = Some(rrf);
            }
        }
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    results.truncate(limit);
    results
}

fn score_rank(
    results: &[ProjectSearchResult],
    score: impl Fn(&ProjectSearchResult) -> Option<f64>,
) -> BTreeMap<String, usize> {
    let mut scored = results
        .iter()
        .filter_map(|result| score(result).map(|value| (result.path.clone(), value)))
        .collect::<Vec<_>>();
    scored.sort_by(|(path_a, score_a), (path_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| path_a.cmp(path_b))
    });
    scored
        .into_iter()
        .enumerate()
        .map(|(index, (path, _))| (normalize_path(&path), index + 1))
        .collect()
}

/// Reserve 15-30% of the final window for one-hop graph expansion. A full
/// vector window leaves the minimum graph share; sparse vector retrieval moves
/// progressively toward the maximum. Missing graph candidates automatically
/// return their slots to keyword/vector results.
fn graph_result_quota(limit: usize, vector_hits: usize) -> usize {
    if limit < 2 {
        return 0;
    }
    let vector_coverage = vector_hits.min(limit) as f64 / limit as f64;
    let ratio = MAX_GRAPH_RESULT_RATIO
        - (MAX_GRAPH_RESULT_RATIO - MIN_GRAPH_RESULT_RATIO) * vector_coverage;
    ((limit as f64 * ratio).ceil() as usize).clamp(1, limit - 1)
}

fn collect_graph_results(
    ranked_results: &[ProjectSearchResult],
    pages: &BTreeMap<String, GraphPage>,
    limit: usize,
    include_content: bool,
) -> Vec<ProjectSearchResult> {
    if ranked_results.is_empty() || pages.is_empty() {
        return Vec::new();
    }

    let mut aliases = BTreeMap::<String, String>::new();
    for (normalized_path, page) in pages {
        let wiki_relative = page.path.strip_prefix("wiki/").unwrap_or(&page.path);
        let stem = file_stem(&page.path);
        for alias in [
            page.path.as_str(),
            wiki_relative,
            stem.as_str(),
            page.title.as_str(),
        ] {
            aliases.insert(normalize_graph_alias(alias), normalized_path.clone());
        }
    }

    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for (source, page) in pages {
        for link in &page.links {
            let Some(target) = aliases.get(&normalize_graph_alias(link)) else {
                continue;
            };
            if source == target {
                continue;
            }
            adjacency
                .entry(source.clone())
                .or_default()
                .insert(target.clone());
            adjacency
                .entry(target.clone())
                .or_default()
                .insert(source.clone());
        }
    }

    let seed_paths: Vec<String> = ranked_results
        .iter()
        .take(limit.min(MAX_GRAPH_SEEDS))
        .map(|result| normalize_path(&result.path))
        .collect();
    let seed_set: BTreeSet<String> = seed_paths.iter().cloned().collect();
    let mut candidate_scores = BTreeMap::<String, f64>::new();
    let mut candidate_seeds = BTreeMap::<String, BTreeSet<String>>::new();
    for (rank, seed) in seed_paths.iter().enumerate() {
        let Some(neighbors) = adjacency.get(seed) else {
            continue;
        };
        for neighbor in neighbors {
            if seed_set.contains(neighbor) {
                continue;
            }
            *candidate_scores.entry(neighbor.clone()).or_default() += 1.0 / (rank + 1) as f64;
            if let Some(seed_page) = pages.get(seed) {
                candidate_seeds
                    .entry(neighbor.clone())
                    .or_default()
                    .insert(seed_page.title.clone());
            }
        }
    }

    let mut candidates: Vec<(String, f64)> = candidate_scores.into_iter().collect();
    candidates.sort_by(|(path_a, score_a), (path_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| path_a.cmp(path_b))
    });
    candidates.truncate(limit);
    if candidates.is_empty() {
        return Vec::new();
    }

    let existing = ranked_results
        .iter()
        .map(|result| (normalize_path(&result.path), result.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut graph_results = Vec::new();
    for (path, graph_score) in candidates {
        if let Some(result) = existing.get(&path) {
            let mut result = result.clone();
            result.graph_score = Some(graph_score);
            result.graph_related_to = candidate_seeds
                .remove(&path)
                .unwrap_or_default()
                .into_iter()
                .collect();
            graph_results.push(result);
            continue;
        }
        let Some(page) = pages.get(&path) else {
            continue;
        };
        let related_titles = candidate_seeds
            .remove(&path)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let related = related_titles.join(", ");
        graph_results.push(ProjectSearchResult {
            path: page.path.clone(),
            title: page.title.clone(),
            snippet: format!("Graph neighbor of {related}"),
            title_match: false,
            score: graph_score,
            bm25_score: None,
            vector_score: None,
            graph_score: Some(graph_score),
            rrf_score: None,
            images: extract_image_refs(&page.content),
            content: include_content.then(|| page.content.clone()),
            graph_related_to: related_titles,
        });
    }
    graph_results
}

fn merge_graph_results(
    results: &mut Vec<ProjectSearchResult>,
    graph_results: Vec<ProjectSearchResult>,
) {
    let mut indexes = results
        .iter()
        .enumerate()
        .map(|(index, result)| (normalize_path(&result.path), index))
        .collect::<BTreeMap<_, _>>();
    for graph_result in graph_results {
        let path = normalize_path(&graph_result.path);
        if let Some(index) = indexes.get(&path).copied() {
            results[index].graph_score = graph_result.graph_score;
            results[index].graph_related_to = graph_result.graph_related_to;
        } else {
            indexes.insert(path, results.len());
            results.push(graph_result);
        }
    }
}

fn normalize_graph_alias(value: &str) -> String {
    value
        .split('#')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches(".md")
        .replace('\\', "/")
        .replace(' ', "-")
        .to_lowercase()
}

fn extract_wikilinks(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find("[[") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("]]") else {
            break;
        };
        let target = rest[..end].split('|').next().unwrap_or_default().trim();
        if !target.is_empty() {
            links.push(target.to_string());
        }
        rest = &rest[end + 2..];
    }
    links
}

fn search_mode(token_rank_empty: bool, vector_hits: usize, graph_hits: usize) -> &'static str {
    if graph_hits > 0 {
        "hybrid"
    } else if vector_hits == 0 {
        "keyword"
    } else if token_rank_empty {
        "vector"
    } else {
        "hybrid"
    }
}

#[derive(Debug, Clone)]
struct PageVectorResult {
    id: String,
    score: f32,
    chunk_text: String,
    heading_path: String,
}

/// 把同一页面的多个分块分数聚合为页面分数。
///
/// 最高相关分块是页面入选的主要依据；其余分块仅以平均值贡献 10% 的稳定性信号。
/// 禁止按分块数量直接累加，否则长页面会因为分块多而轻易饱和到 1.0，反向挤掉
/// 真正相关但较短的来源页。
fn blend_page_chunk_scores(chunks: &[vectorstore::ChunkSearchResult]) -> f32 {
    let Some(first) = chunks.first() else {
        return 0.0;
    };
    if chunks.len() == 1 {
        return first.score;
    }
    let tail_average =
        chunks.iter().skip(1).map(|chunk| chunk.score).sum::<f32>() / (chunks.len() - 1) as f32;
    (first.score * 0.9) + (tail_average * 0.1)
}

async fn search_by_embedding(
    project_path: &str,
    query_embedding: Vec<f32>,
    top_k: usize,
) -> Result<Vec<PageVectorResult>, String> {
    let raw_chunks = vectorstore::vector_search_chunks(
        project_path.to_string(),
        query_embedding,
        (top_k * 3).max(30),
    )
    .await?;
    if raw_chunks.is_empty() {
        return Ok(vec![]);
    }

    let mut by_page: BTreeMap<String, Vec<vectorstore::ChunkSearchResult>> = BTreeMap::new();
    for chunk in raw_chunks {
        by_page
            .entry(chunk.page_id.clone())
            .or_default()
            .push(chunk);
    }

    let mut ranked = Vec::new();
    for (id, mut chunks) in by_page {
        chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });
        let top_chunk = chunks[0].clone();
        let blended = blend_page_chunk_scores(&chunks);
        ranked.push(PageVectorResult {
            id,
            score: blended,
            chunk_text: top_chunk.chunk_text,
            heading_path: top_chunk.heading_path,
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    ranked.truncate(top_k);
    Ok(ranked)
}

fn search_by_embedding_boxed<'a>(
    project_path: &'a str,
    query_embedding: Vec<f32>,
    top_k: usize,
) -> VectorSearchFuture<'a> {
    Box::pin(search_by_embedding(project_path, query_embedding, top_k))
}

fn materialize_vector_only_results(
    vector_results: &[PageVectorResult],
    page_paths_by_stem: &BTreeMap<String, String>,
    project_path: &str,
    results: &mut Vec<ProjectSearchResult>,
    include_content: bool,
) -> Result<(), String> {
    let mut known: BTreeSet<String> = results.iter().map(|r| file_stem(&r.path)).collect();
    for vr in vector_results {
        if known.contains(&vr.id) {
            continue;
        }
        if let Some(rel) = page_paths_by_stem.get(&vr.id) {
            let path = Path::new(project_path).join(rel);
            // 扫描完成到向量物化之间文件仍可能被删除或变为不可读；不能把这种竞态
            // 静默降级成少一个候选，否则最终可能被误判为 EMPTY_CONTEXT。
            let content = fs::read_to_string(&path)
                .map_err(|err| format!("Failed to materialize vector candidate: {err}"))?;
            let file_name = Path::new(&rel)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let title = extract_title(&content, file_name);
            let snippet = build_vector_snippet(vr);
            results.push(ProjectSearchResult {
                path: rel.clone(),
                title,
                snippet,
                title_match: false,
                score: 0.0,
                bm25_score: None,
                vector_score: Some(vr.score),
                graph_score: None,
                rrf_score: None,
                images: extract_image_refs(&content),
                content: include_content.then_some(content),
                graph_related_to: Vec::new(),
            });
            known.insert(vr.id.clone());
        }
    }
    Ok(())
}

fn build_vector_snippet(result: &PageVectorResult) -> String {
    let mut text = result.chunk_text.trim().replace('\n', " ");
    if text.is_empty() {
        return String::new();
    }
    if text.chars().count() > SNIPPET_CONTEXT * 2 {
        text = text.chars().take(SNIPPET_CONTEXT * 2).collect::<String>();
        text.push_str("...");
    }
    let heading = result.heading_path.trim();
    if heading.is_empty() {
        text
    } else {
        format!("{heading}: {text}")
    }
}

fn score_file(
    project_path: &str,
    path: &Path,
    content: &str,
    tokens: &[String],
    query_phrase: &str,
    query: &str,
    include_content: bool,
) -> Option<ProjectSearchResult> {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let title = extract_title(content, file_name);
    let title_text = format!("{title} {file_name}");
    let title_lower = title_text.to_lowercase();
    let content_lower = content.to_lowercase();
    let stem = file_name.trim_end_matches(".md").to_lowercase();

    let filename_exact = !query_phrase.is_empty() && stem == query_phrase;
    let title_has_phrase = !query_phrase.is_empty() && title_lower.contains(query_phrase);
    let content_phrase_occ =
        count_occurrences(&content_lower, query_phrase).min(MAX_PHRASE_OCC_COUNTED);
    let title_token_score = token_match_score(&title_text, tokens);
    let content_token_score = token_match_score(content, tokens);

    if !filename_exact
        && !title_has_phrase
        && content_phrase_occ == 0
        && title_token_score == 0
        && content_token_score == 0
    {
        return None;
    }

    let score = (if filename_exact {
        FILENAME_EXACT_BONUS
    } else {
        0.0
    }) + (if title_has_phrase {
        PHRASE_IN_TITLE_BONUS
    } else {
        0.0
    }) + content_phrase_occ as f64 * PHRASE_IN_CONTENT_PER_OCC
        + title_token_score as f64 * TITLE_TOKEN_WEIGHT
        + content_token_score as f64 * CONTENT_TOKEN_WEIGHT;

    let snippet_anchor = if content_phrase_occ > 0 {
        query_phrase.to_string()
    } else {
        tokens
            .iter()
            .find(|token| content_lower.contains(token.as_str()))
            .cloned()
            .unwrap_or_else(|| query.to_string())
    };

    Some(ProjectSearchResult {
        path: relative_to_project(project_path, path),
        title,
        snippet: build_snippet(content, &snippet_anchor),
        title_match: title_token_score > 0 || title_has_phrase,
        score,
        bm25_score: Some(score),
        vector_score: None,
        graph_score: None,
        rrf_score: None,
        images: extract_image_refs(content),
        content: include_content.then_some(content.to_string()),
        graph_related_to: Vec::new(),
    })
}

pub fn tokenize_query(query: &str) -> Vec<String> {
    let raw = query
        .to_lowercase()
        .split(is_query_separator)
        .filter(|token| token.chars().count() > 1)
        .filter(|token| !is_stop_word(token))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    for token in raw {
        let chars = token.chars().collect::<Vec<_>>();
        let has_cjk = chars.iter().any(|c| ('\u{3400}'..='\u{9fff}').contains(c));
        if has_cjk && chars.len() > 2 {
            for pair in chars.windows(2) {
                out.push(pair.iter().collect());
            }
            for ch in &chars {
                let s = ch.to_string();
                if !is_stop_word(&s) {
                    out.push(s);
                }
            }
            out.push(token);
        } else {
            out.push(token);
        }
    }
    out.into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn is_query_separator(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_punctuation()
        || matches!(
            c,
            '，' | '。'
                | '！'
                | '？'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '·'
                | '～'
                | '…'
        )
}

fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        "的" | "是"
            | "了"
            | "什么"
            | "在"
            | "有"
            | "和"
            | "与"
            | "对"
            | "从"
            | "the"
            | "is"
            | "a"
            | "an"
            | "what"
            | "how"
            | "are"
            | "was"
            | "were"
            | "do"
            | "does"
            | "did"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "it"
            | "its"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "with"
            | "by"
            | "this"
            | "that"
            | "these"
            | "those"
    )
}

fn trim_query_punctuation(value: &str) -> String {
    value.trim_matches(is_query_separator).to_string()
}

fn token_match_score(text: &str, tokens: &[String]) -> usize {
    let lower = text.to_lowercase();
    tokens
        .iter()
        .filter(|token| lower.contains(token.as_str()))
        .count()
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.match_indices(needle).count()
}

pub fn extract_title(content: &str, file_name: &str) -> String {
    let has_frontmatter = content.starts_with("---");
    let mut in_frontmatter = has_frontmatter;
    let mut frontmatter_closed = false;
    for line in content.lines().skip(if has_frontmatter { 1 } else { 0 }) {
        let trimmed = line.trim();
        if in_frontmatter && trimmed == "---" {
            in_frontmatter = false;
            frontmatter_closed = true;
            continue;
        }
        if in_frontmatter && trimmed.starts_with("title:") {
            return trimmed
                .trim_start_matches("title:")
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        }
        if has_frontmatter && !frontmatter_closed {
            continue;
        }
        if let Some(title) = trimmed.strip_prefix("# ") {
            return title.trim().to_string();
        }
    }
    file_name.trim_end_matches(".md").replace('-', " ")
}

pub fn extract_image_refs(content: &str) -> Vec<SearchImageRef> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut rest = content;
    while let Some(start) = rest.find("![") {
        rest = &rest[start + 2..];
        let Some(alt_end) = rest.find("](") else {
            break;
        };
        let alt = &rest[..alt_end];
        rest = &rest[alt_end + 2..];
        let Some(url_end) = rest.find(')') else {
            break;
        };
        let url = &rest[..url_end];
        if !url.trim().is_empty()
            && !url.contains(char::is_whitespace)
            && seen.insert(url.to_string())
        {
            out.push(SearchImageRef {
                url: url.to_string(),
                alt: alt.to_string(),
            });
        }
        rest = &rest[url_end + 1..];
    }
    out
}

async fn fetch_embedding_with_retry(
    text: &str,
    cfg: &SearchEmbeddingConfig,
    max_retries: usize,
) -> Result<Vec<f32>, String> {
    let mut current = text.to_string();
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        match fetch_embedding_once(&current, cfg).await {
            Ok(embedding) => return Ok(embedding),
            Err(EmbeddingFetchError::Oversize(message)) => {
                if attempts <= max_retries
                    && current.len() > 64
                    && halve_text_on_char_boundary(&mut current)
                {
                    eprintln!(
                        "[Embedding] auto-halving after oversize error at {} chars; retrying at {} chars ({attempts}/{})",
                        text.chars().count(),
                        current.chars().count(),
                        max_retries + 1
                    );
                    continue;
                }
                return Err(format!(
                    "Endpoint rejected input even at {} chars. Lower Settings -> Embedding -> Max Chunk Chars. {message}",
                    current.len()
                ));
            }
            Err(EmbeddingFetchError::Other(message)) => return Err(message),
        }
    }
}

async fn fetch_embedding_batch(
    texts: &[String],
    cfg: &SearchEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() || texts.len() > 64 {
        return Err("Embedding batch must contain between 1 and 64 inputs".to_string());
    }
    if is_google_embedding_config(cfg) || is_doubao_multimodal_embedding_config(cfg) {
        return Err(
            "This embedding provider does not use the OpenAI-compatible batch format".to_string(),
        );
    }

    let endpoint = volcengine_embedding_endpoint(cfg);
    let mut req = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            SEARCH_EMBEDDING_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| format!("Embedding HTTP client error: {e}"))?
        .post(&endpoint)
        .header("Content-Type", "application/json");
    if is_local_or_private_http_endpoint(&endpoint) {
        req = req.header("Origin", "http://localhost");
    }
    if !cfg.api_key.trim().is_empty() {
        req = req.bearer_auth(cfg.api_key.trim());
    }
    if let Some(extra) = cfg.extra_headers.as_ref() {
        for (name, value) in extra {
            let name = name.trim();
            let value = value.trim();
            if !name.is_empty()
                && !value.is_empty()
                && is_safe_extra_header_name(name)
                && !is_reserved_extra_header_name(name)
            {
                req = req.header(name, value);
            }
        }
    }
    let response = req
        .json(&json!({ "model": cfg.model, "input": texts }))
        .send()
        .await
        .map_err(|e| format!("Embedding batch request failed: {e}"))?;
    let status = response.status();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Embedding batch response read failed: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "Embedding batch API HTTP {status}: {}",
            response_text.chars().take(200).collect::<String>()
        ));
    }
    let data: Value = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Embedding batch response parse failed: {e}: {}",
            response_text.chars().take(200).collect::<String>()
        )
    })?;
    parse_embedding_batch_values(&data, texts.len())
}

fn parse_embedding_batch_values(data: &Value, expected: usize) -> Result<Vec<Vec<f32>>, String> {
    let entries = data
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "Embedding batch response missing data array".to_string())?;
    if entries.len() != expected {
        return Err(format!(
            "Embedding batch returned {} vectors for {expected} inputs",
            entries.len()
        ));
    }
    let mut indexed = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let index = entry
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(position);
        if index >= expected {
            return Err("Embedding batch response contains an out-of-range index".to_string());
        }
        let values = entry
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| "Embedding batch response missing vector".to_string())?;
        let mut vector = Vec::with_capacity(values.len());
        for value in values {
            let number = value
                .as_f64()
                .ok_or_else(|| "Embedding batch response contains non-number values".to_string())?;
            if !number.is_finite() {
                return Err("Embedding batch response contains non-finite values".to_string());
            }
            vector.push(number as f32);
        }
        if vector.is_empty() {
            return Err("Embedding batch response vector is empty".to_string());
        }
        indexed.push((index, vector));
    }
    indexed.sort_by_key(|(index, _)| *index);
    if indexed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("Embedding batch response contains duplicate indexes".to_string());
    }
    let dimension = indexed.first().map(|(_, vector)| vector.len()).unwrap_or(0);
    if indexed.iter().any(|(_, vector)| vector.len() != dimension) {
        return Err("Embedding batch response contains inconsistent vector dimensions".to_string());
    }
    Ok(indexed.into_iter().map(|(_, vector)| vector).collect())
}

fn halve_text_on_char_boundary(text: &mut String) -> bool {
    let char_count = text.chars().count();
    if char_count <= 1 {
        return false;
    }
    let keep = (char_count / 2).max(1);
    *text = text.chars().take(keep).collect();
    true
}

enum EmbeddingFetchError {
    Oversize(String),
    Other(String),
}

async fn fetch_embedding_once(
    text: &str,
    cfg: &SearchEmbeddingConfig,
) -> Result<Vec<f32>, EmbeddingFetchError> {
    let is_google = is_google_embedding_config(cfg);
    let is_doubao_multimodal = is_doubao_multimodal_embedding_config(cfg);
    let endpoint = if is_google {
        google_embedding_endpoint(cfg)
    } else {
        volcengine_embedding_endpoint(cfg)
    };
    let mut req = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            SEARCH_EMBEDDING_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding HTTP client error: {e}")))?
        .post(&endpoint)
        .header("Content-Type", "application/json");
    // Browser-based local model servers often require a browser-like
    // Origin even when the request is routed through Rust. Keep this
    // reserved so user-supplied extra headers cannot override it.
    if is_local_or_private_http_endpoint(&endpoint) {
        req = req.header("Origin", "http://localhost");
    }
    if !cfg.api_key.trim().is_empty() {
        if is_google {
            req = req.header("x-goog-api-key", cfg.api_key.trim());
        } else {
            req = req.bearer_auth(cfg.api_key.trim());
        }
    }
    if let Some(extra) = cfg.extra_headers.as_ref() {
        for (name, value) in extra {
            let trimmed = name.trim();
            let value = value.trim();
            if trimmed.is_empty() || value.is_empty() || !is_safe_extra_header_name(trimmed) {
                continue;
            }
            if is_reserved_extra_header_name(trimmed) {
                continue;
            }
            req = req.header(trimmed, value);
        }
    }
    let body = if is_google {
        google_embedding_body(&cfg.model, text, cfg.output_dimensionality)
    } else if is_doubao_multimodal {
        doubao_multimodal_embedding_body(&cfg.model, text)
    } else {
        json!({ "model": cfg.model, "input": text })
    };
    let resp = req
        .json(&body)
        .send()
        .await
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding request failed: {e}")))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding response read failed: {e}")))?;
    if !status.is_success() {
        let preview = text.chars().take(200).collect::<String>();
        if looks_like_oversize_error(status.as_u16(), &text) {
            return Err(EmbeddingFetchError::Oversize(format!(
                "Embedding API HTTP {status}: {preview}"
            )));
        }
        return Err(EmbeddingFetchError::Other(format!(
            "Embedding API HTTP {status}: {preview}"
        )));
    }
    let data: Value = serde_json::from_str(&text).map_err(|e| {
        EmbeddingFetchError::Other(format!(
            "Embedding response parse failed: {e}: {}",
            text.chars().take(200).collect::<String>()
        ))
    })?;
    parse_embedding_values(&data, is_google, is_doubao_multimodal)
        .map_err(EmbeddingFetchError::Other)
}

fn looks_like_oversize_error(status: u16, body: &str) -> bool {
    if status == 413 {
        return true;
    }
    let lower = body.to_lowercase();
    lower.contains("too long")
        || lower.contains("maximum context")
        || lower.contains("max_tokens")
        || lower.contains("max tokens")
        || lower.contains("context length")
        || lower.contains("token limit")
        || lower.contains("exceeds")
        || lower.contains("input length")
}

fn parse_embedding_values(
    data: &Value,
    is_google: bool,
    is_doubao_multimodal: bool,
) -> Result<Vec<f32>, String> {
    let values = if is_google {
        data.get("embedding")
            .and_then(|v| v.get("values"))
            .and_then(Value::as_array)
    } else if is_doubao_multimodal {
        data.get("data")
            .and_then(|v| v.get("embedding"))
            .and_then(Value::as_array)
    } else {
        data.get("data")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("embedding"))
            .and_then(Value::as_array)
    }
    .ok_or_else(|| "Embedding response missing vector".to_string())?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let n = value
            .as_f64()
            .ok_or_else(|| "Embedding response contains non-number values".to_string())?;
        if !n.is_finite() {
            return Err("Embedding response contains non-finite values".to_string());
        }
        out.push(n as f32);
    }
    if out.is_empty() {
        return Err("Embedding response vector is empty".to_string());
    }
    Ok(out)
}

fn is_safe_extra_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            matches!(
                b,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
                    | b'0'..=b'9'
                    | b'A'..=b'Z'
                    | b'a'..=b'z'
            )
        })
}

fn is_reserved_extra_header_name(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "content-type" | "host" | "content-length" | "origin" | "x-goog-api-key"
    )
}

fn is_google_embedding_config(cfg: &SearchEmbeddingConfig) -> bool {
    let endpoint = cfg.endpoint.to_lowercase();
    endpoint.contains("generativelanguage.googleapis.com") || endpoint.contains(":embedcontent")
}

fn is_local_or_private_http_endpoint(endpoint: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .trim_matches('[')
        .trim_matches(']')
        .to_ascii_lowercase();
    if matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return true;
    }
    let octets = host
        .split('.')
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>();
    let Ok(octets) = octets else {
        return false;
    };
    if octets.len() != 4 {
        return false;
    }
    octets[0] == 10
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 168)
        || octets[0] == 127
}

fn is_volcengine_embedding_endpoint(endpoint: &str) -> bool {
    let host = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_lowercase()))
        .unwrap_or_else(|| {
            let trimmed = endpoint.trim();
            trimmed
                .split_once("://")
                .map(|(_, rest)| rest)
                .unwrap_or(trimmed)
                .split(['/', '?', '#'])
                .next()
                .unwrap_or("")
                .to_lowercase()
        });
    host == "volces.com" || host.ends_with(".volces.com") || host.contains("volcengine")
}

fn is_doubao_multimodal_embedding_config(cfg: &SearchEmbeddingConfig) -> bool {
    cfg.model
        .trim()
        .to_lowercase()
        .contains("doubao-embedding-vision")
}

fn volcengine_embedding_endpoint(cfg: &SearchEmbeddingConfig) -> String {
    let raw = cfg.endpoint.trim();
    if !is_volcengine_embedding_endpoint(raw) {
        return raw.to_string();
    }
    let suffix = if is_doubao_multimodal_embedding_config(cfg) {
        "/embeddings/multimodal"
    } else {
        "/embeddings"
    };
    append_endpoint_path(raw, suffix)
}

fn append_endpoint_path(endpoint: &str, target_suffix: &str) -> String {
    let suffix = target_suffix.trim_start_matches('/');
    match reqwest::Url::parse(endpoint) {
        Ok(mut url) => {
            let path = url.path().trim_end_matches('/').to_string();
            let lower_path = path.to_lowercase();
            let lower_suffix = format!("/{}", suffix.to_lowercase());
            if lower_path.ends_with(&lower_suffix) {
                url.set_path(if path.is_empty() { "/" } else { &path });
                return url.to_string();
            }
            if lower_path.ends_with("/embeddings/multimodal") && lower_suffix == "/embeddings" {
                let base = path.trim_end_matches("/multimodal");
                url.set_path(if base.is_empty() { "/" } else { base });
                return url.to_string();
            }
            if lower_path.ends_with("/embeddings") && lower_suffix == "/embeddings/multimodal" {
                url.set_path(&format!("{path}/multimodal"));
                return url.to_string();
            }
            let next = format!("{path}/{suffix}").replace("//", "/");
            url.set_path(&next);
            url.to_string()
        }
        Err(_) => {
            let (base, query) = endpoint.split_once('?').unwrap_or((endpoint, ""));
            let trimmed = base.trim_end_matches('/');
            let lower = trimmed.to_lowercase();
            let lower_suffix = format!("/{}", suffix.to_lowercase());
            let next = if lower.ends_with(&lower_suffix) {
                trimmed.to_string()
            } else if lower.ends_with("/embeddings/multimodal") && lower_suffix == "/embeddings" {
                trimmed.trim_end_matches("/multimodal").to_string()
            } else if lower.ends_with("/embeddings") && lower_suffix == "/embeddings/multimodal" {
                format!("{trimmed}/multimodal")
            } else {
                format!("{trimmed}/{suffix}")
            };
            if query.is_empty() {
                next
            } else {
                format!("{next}?{query}")
            }
        }
    }
}

fn google_embedding_endpoint(cfg: &SearchEmbeddingConfig) -> String {
    let raw = strip_google_api_key_query(cfg.endpoint.trim())
        .trim_end_matches('/')
        .to_string();
    if raw.to_lowercase().contains(":batchembedcontents") {
        return raw
            .replace(":batchEmbedContents", ":embedContent")
            .replace(":batchembedcontents", ":embedContent");
    }
    if raw.to_lowercase().contains(":embedcontent") {
        return raw;
    }
    let model = cfg.model.trim().trim_start_matches("models/");
    if raw.to_lowercase().contains("/models/") {
        format!("{raw}:embedContent")
    } else {
        format!("{raw}/models/{model}:embedContent")
    }
}

fn strip_google_api_key_query(endpoint: &str) -> String {
    if !endpoint.contains('?') {
        return endpoint.to_string();
    }
    match reqwest::Url::parse(endpoint) {
        Ok(mut url) => {
            let kept = url
                .query_pairs()
                .filter(|(key, _)| !key.eq_ignore_ascii_case("key"))
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect::<Vec<_>>();
            url.query_pairs_mut().clear().extend_pairs(kept);
            url.to_string().trim_end_matches('?').to_string()
        }
        Err(_) => endpoint
            .split_once('?')
            .map(|(base, query)| {
                let kept = query
                    .split('&')
                    .filter(|pair| {
                        let key = pair.split_once('=').map(|(k, _)| k).unwrap_or(*pair);
                        !key.eq_ignore_ascii_case("key")
                    })
                    .collect::<Vec<_>>();
                if kept.is_empty() {
                    base.to_string()
                } else {
                    format!("{base}?{}", kept.join("&"))
                }
            })
            .unwrap_or_else(|| endpoint.to_string()),
    }
}

fn google_embedding_body(model: &str, text: &str, output_dimensionality: Option<f64>) -> Value {
    let model_path = if model.trim().starts_with("models/") {
        model.trim().to_string()
    } else {
        format!("models/{}", model.trim())
    };
    let mut body = json!({
        "model": model_path,
        "content": { "parts": [{ "text": text }] },
    });
    if let Some(dim) = output_dimensionality
        .filter(|dim| dim.is_finite() && *dim >= 1.0)
        .map(|dim| dim.floor() as u32)
    {
        body["output_dimensionality"] = json!(dim);
    }
    body
}

fn doubao_multimodal_embedding_body(model: &str, text: &str) -> Value {
    json!({
        "model": model,
        "encoding_format": "float",
        "input": [{ "type": "text", "text": text }],
    })
}

pub fn build_snippet(content: &str, query: &str) -> String {
    let lower = content.to_lowercase();
    let q = query.to_lowercase();
    let idx = lower.find(&q).unwrap_or(0);
    let char_positions: Vec<usize> = content.char_indices().map(|(idx, _)| idx).collect();
    if char_positions.is_empty() {
        return String::new();
    }
    let match_char = char_positions
        .iter()
        .position(|byte| *byte >= idx)
        .unwrap_or(char_positions.len().saturating_sub(1));
    let query_chars = query.chars().count().max(1);
    let start_char = match_char.saturating_sub(SNIPPET_CONTEXT);
    let end_char = (match_char + query_chars + SNIPPET_CONTEXT).min(char_positions.len());
    let start = char_positions[start_char];
    let end = if end_char < char_positions.len() {
        char_positions[end_char]
    } else {
        content.len()
    };
    let mut snippet = content[start..end].replace('\n', " ");
    if start > 0 {
        snippet = format!("...{snippet}");
    }
    if end < content.len() {
        snippet.push_str("...");
    }
    snippet
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn relative_to_project(project_path: &str, path: &Path) -> String {
    let root = Path::new(project_path);
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_project() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("llm-wiki-search-test-{id}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join("wiki/concepts")).unwrap();
        path
    }

    fn write_page(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    #[test]
    fn tokenizes_cjk_bigrams_and_chars() {
        let tokens = tokenize_query("默会知识");
        assert!(tokens.contains(&"默会".to_string()));
        assert!(tokens.contains(&"知识".to_string()));
        assert!(tokens.contains(&"默".to_string()));
    }

    #[test]
    fn extracts_image_refs_without_duplicates() {
        let refs = extract_image_refs("![a](wiki/media/x.png)\n![b](wiki/media/x.png)");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].alt, "a");
    }

    #[test]
    fn extract_title_uses_frontmatter_or_heading_not_body_title_lines() {
        let with_frontmatter = "---\ntitle: Real Title\n---\n\ntitle: Body Label\n# Heading";
        assert_eq!(
            extract_title(with_frontmatter, "fallback-name.md"),
            "Real Title"
        );

        let without_frontmatter = "intro\ntitle: Body Label\n# Real Heading";
        assert_eq!(
            extract_title(without_frontmatter, "fallback-name.md"),
            "Real Heading"
        );

        assert_eq!(
            extract_title("plain body", "vector-database.md"),
            "vector database"
        );
    }

    #[test]
    fn explicit_query_embedding_is_validated() {
        assert!(validate_query_embedding(vec![0.1, 0.2]).is_ok());
        assert!(validate_query_embedding(vec![]).is_err());
        assert!(validate_query_embedding(vec![f32::NAN]).is_err());
        assert!(validate_query_embedding(vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn chat_context_embedding_policy_propagates_provider_failure() {
        let strict =
            finish_query_embedding_fetch(Err("injected provider outage".to_string()), false)
                .unwrap_err();
        assert!(strict.contains("query embedding failed"));
        assert!(strict.contains("injected provider outage"));

        let desktop_fallback =
            finish_query_embedding_fetch(Err("injected provider outage".to_string()), true)
                .unwrap();
        assert_eq!(desktop_fallback, None);
    }

    #[test]
    fn google_embedding_endpoint_strips_key_and_normalizes_batch_endpoint() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=URL_KEY&alt=json".to_string(),
            api_key: "HEADER_KEY".to_string(),
            model: "gemini-embedding-001".to_string(),
            output_dimensionality: Some(768.0),
            extra_headers: None,
        };

        let endpoint = google_embedding_endpoint(&cfg);
        assert!(endpoint.contains(":embedContent"));
        assert!(!endpoint.contains(":batchEmbedContents"));
        assert!(!endpoint.contains("URL_KEY"));
        assert!(endpoint.contains("alt=json"));

        let body = google_embedding_body("gemini-embedding-001", "hello", Some(768.0));
        assert_eq!(body["model"], "models/gemini-embedding-001");
        assert_eq!(body["output_dimensionality"], 768);
    }

    #[test]
    fn volcengine_embedding_endpoint_only_appends_for_volcengine_hosts() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-text-240715".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );

        let custom = SearchEmbeddingConfig {
            endpoint: "https://gateway.example.com/proxy/volcengine?upstream=volces.com"
                .to_string(),
            ..cfg.clone()
        };
        assert_eq!(
            volcengine_embedding_endpoint(&custom),
            "https://gateway.example.com/proxy/volcengine?upstream=volces.com"
        );
    }

    #[test]
    fn doubao_multimodal_endpoint_and_body_use_vision_shape() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ark.cn-beijing.volces.com/api/v3/embeddings?trace=1".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-vision".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };

        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal?trace=1"
        );

        let body = doubao_multimodal_embedding_body("doubao-embedding-vision", "hello");
        assert_eq!(body["model"], "doubao-embedding-vision");
        assert_eq!(body["encoding_format"], "float");
        assert_eq!(body["input"][0]["type"], "text");
        assert_eq!(body["input"][0]["text"], "hello");
    }

    #[test]
    fn volcengine_embedding_endpoint_does_not_duplicate_existing_suffixes() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ARK.cn-beijing.volces.com/api/v3/embeddings".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-text-240715".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );

        let vision = SearchEmbeddingConfig {
            endpoint: "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal".to_string(),
            model: "doubao-embedding-vision".to_string(),
            ..cfg.clone()
        };
        assert_eq!(
            volcengine_embedding_endpoint(&vision),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
        );

        let text_on_multimodal_endpoint = SearchEmbeddingConfig {
            model: "doubao-embedding-text-240715".to_string(),
            ..vision
        };
        assert_eq!(
            volcengine_embedding_endpoint(&text_on_multimodal_endpoint),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );
    }

    #[test]
    fn doubao_multimodal_detection_is_model_driven_for_custom_gateways() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://gateway.example.com/ark/embeddings/multimodal".to_string(),
            api_key: "KEY".to_string(),
            model: "doubao-embedding-vision".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };

        assert!(is_doubao_multimodal_embedding_config(&cfg));
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://gateway.example.com/ark/embeddings/multimodal"
        );

        let body = doubao_multimodal_embedding_body(&cfg.model, "hello");
        assert_eq!(body["input"][0]["type"], "text");
        assert_eq!(body["input"][0]["text"], "hello");
    }

    #[test]
    fn doubao_multimodal_response_shape_is_distinct_from_openai_shape() {
        let values =
            parse_embedding_values(&json!({ "data": { "embedding": [0.1, 0.2] } }), false, true)
                .expect("vision response should parse");
        assert_eq!(values, vec![0.1, 0.2]);

        assert!(
            parse_embedding_values(&json!({ "data": [{ "embedding": [0.1] }] }), false, true,)
                .is_err()
        );
    }

    #[test]
    fn extra_embedding_header_names_are_validated_and_reserved_names_are_skipped() {
        assert!(is_safe_extra_header_name("X-Model-Provider-Id"));
        assert!(is_safe_extra_header_name("x_trace.id"));
        assert!(!is_safe_extra_header_name(""));
        assert!(!is_safe_extra_header_name("Bad Header"));
        assert!(!is_safe_extra_header_name("中文"));

        assert!(is_reserved_extra_header_name("Authorization"));
        assert!(is_reserved_extra_header_name("content-type"));
        assert!(is_reserved_extra_header_name("Origin"));
        assert!(is_reserved_extra_header_name("X-Goog-Api-Key"));
        assert!(!is_reserved_extra_header_name("X-Model-Provider-Id"));
    }

    #[test]
    fn embedding_origin_header_is_limited_to_local_or_private_endpoints() {
        assert!(is_local_or_private_http_endpoint(
            "http://127.0.0.1:1234/v1/embeddings"
        ));
        assert!(is_local_or_private_http_endpoint(
            "http://192.168.1.20:11434/v1/embeddings"
        ));
        assert!(is_local_or_private_http_endpoint(
            "http://172.16.0.5/v1/embeddings"
        ));
        assert!(!is_local_or_private_http_endpoint(
            "https://api.openai.com/v1/embeddings"
        ));
    }

    #[test]
    fn embedding_halving_never_splits_cjk_codepoints() {
        let mut text = "默会知识库".to_string();
        assert!(halve_text_on_char_boundary(&mut text));
        assert_eq!(text, "默会");
    }

    #[test]
    fn google_embedding_body_floors_positive_dimensions_and_omits_invalid_values() {
        let body = google_embedding_body("gemini-embedding-001", "hello", Some(1.9));
        assert_eq!(body["output_dimensionality"], 1);

        let zero = google_embedding_body("gemini-embedding-001", "hello", Some(0.0));
        assert!(zero.get("output_dimensionality").is_none());

        let negative = google_embedding_body("gemini-embedding-001", "hello", Some(-4.0));
        assert!(negative.get("output_dimensionality").is_none());
    }

    #[test]
    fn search_mode_distinguishes_keyword_vector_and_hybrid() {
        assert_eq!(search_mode(false, 0, 0), "keyword");
        assert_eq!(search_mode(true, 3, 0), "vector");
        assert_eq!(search_mode(false, 3, 0), "hybrid");
        assert_eq!(search_mode(false, 0, 1), "hybrid");
    }

    #[test]
    fn explicit_search_modes_rank_by_their_native_score_only() {
        let candidates = vec![
            ProjectSearchResult {
                path: "wiki/bm25.md".to_string(),
                title: "BM25".to_string(),
                snippet: String::new(),
                title_match: true,
                score: 0.0,
                bm25_score: Some(9.0),
                vector_score: Some(0.2),
                graph_score: Some(0.4),
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: Vec::new(),
            },
            ProjectSearchResult {
                path: "wiki/vector.md".to_string(),
                title: "Vector".to_string(),
                snippet: String::new(),
                title_match: false,
                score: 0.0,
                bm25_score: Some(2.0),
                vector_score: Some(0.9),
                graph_score: Some(0.3),
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: Vec::new(),
            },
            ProjectSearchResult {
                path: "wiki/graph.md".to_string(),
                title: "Graph".to_string(),
                snippet: String::new(),
                title_match: false,
                score: 0.0,
                bm25_score: Some(1.0),
                vector_score: Some(0.1),
                graph_score: Some(0.8),
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: vec!["Seed".to_string()],
            },
        ];

        let bm25 = rank_results_for_mode(candidates.clone(), SearchMode::Bm25, 10);
        assert_eq!(bm25[0].path, "wiki/bm25.md");
        assert_eq!(bm25[0].score, 9.0);
        assert_eq!(bm25[0].vector_score, None);
        assert_eq!(bm25[0].graph_score, None);
        assert_eq!(bm25[0].rrf_score, None);

        let vector = rank_results_for_mode(candidates.clone(), SearchMode::Vector, 10);
        assert_eq!(vector[0].path, "wiki/vector.md");
        assert!((vector[0].score - 0.9).abs() < 0.000001);
        assert_eq!(vector[0].bm25_score, None);
        assert_eq!(vector[0].graph_score, None);
        assert_eq!(vector[0].rrf_score, None);

        let graph = rank_results_for_mode(candidates, SearchMode::Graph, 10);
        assert_eq!(graph[0].path, "wiki/graph.md");
        assert_eq!(graph[0].score, 0.8);
        assert_eq!(graph[0].bm25_score, None);
        assert_eq!(graph[0].vector_score, None);
        assert_eq!(graph[0].rrf_score, None);
    }

    #[test]
    fn hybrid_mode_exposes_only_real_operator_scores_and_rrf() {
        let candidates = vec![
            ProjectSearchResult {
                path: "wiki/both.md".to_string(),
                title: "Both".to_string(),
                snippet: String::new(),
                title_match: true,
                score: 0.0,
                bm25_score: Some(4.0),
                vector_score: Some(0.9),
                graph_score: None,
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: Vec::new(),
            },
            ProjectSearchResult {
                path: "wiki/bm25.md".to_string(),
                title: "BM25".to_string(),
                snippet: String::new(),
                title_match: true,
                score: 0.0,
                bm25_score: Some(8.0),
                vector_score: None,
                graph_score: None,
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: Vec::new(),
            },
        ];

        let hybrid = rank_results_for_mode(candidates, SearchMode::Hybrid, 10);
        assert_eq!(hybrid[0].path, "wiki/both.md");
        assert_eq!(hybrid[0].score, hybrid[0].rrf_score.unwrap());
        assert_eq!(hybrid[1].vector_score, None);
        assert_eq!(hybrid[1].graph_score, None);
    }

    #[test]
    fn search_contract_serializes_camel_case_nullable_scores() {
        let response = ProjectSearchResponse {
            mode: "bm25".to_string(),
            requested_mode: SearchMode::Bm25,
            executed_mode: SearchMode::Bm25,
            results: vec![ProjectSearchResult {
                path: "wiki/result.md".to_string(),
                title: "Result".to_string(),
                snippet: "body".to_string(),
                title_match: true,
                score: 3.5,
                bm25_score: Some(3.5),
                vector_score: None,
                graph_score: None,
                rrf_score: None,
                images: vec![],
                content: None,
                graph_related_to: Vec::new(),
            }],
            token_hits: 1,
            vector_hits: 0,
            graph_hits: 0,
        };

        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value["requestedMode"], "bm25");
        assert_eq!(value["executedMode"], "bm25");
        assert_eq!(value["results"][0]["bm25Score"], 3.5);
        assert!(value["results"][0]["vectorScore"].is_null());
        assert!(value["results"][0]["graphScore"].is_null());
        assert!(value["results"][0]["rrfScore"].is_null());
        // Cross-language wire contract: graphRelatedTo is always present, even empty.
        assert_eq!(value["results"][0]["graphRelatedTo"], serde_json::json!([]));
        assert!(value["results"][0].get("content").is_none());
    }

    #[test]
    fn empty_graph_related_to_serializes_as_empty_array() {
        let result = ProjectSearchResult {
            path: "wiki/result.md".to_string(),
            title: "Result".to_string(),
            snippet: "body".to_string(),
            title_match: false,
            score: 1.0,
            bm25_score: Some(1.0),
            vector_score: None,
            graph_score: None,
            rrf_score: None,
            images: vec![],
            content: None,
            graph_related_to: Vec::new(),
        };

        let value = serde_json::to_value(&result).unwrap();
        assert!(
            value.get("graphRelatedTo").is_some(),
            "graphRelatedTo must always be present on the wire, got keys: {:?}",
            value.as_object().map(|o| o.keys().collect::<Vec<_>>())
        );
        assert_eq!(value["graphRelatedTo"], serde_json::json!([]));

        let populated = ProjectSearchResult {
            graph_related_to: vec!["Seed A".to_string(), "Seed B".to_string()],
            ..result
        };
        let populated_value = serde_json::to_value(&populated).unwrap();
        assert_eq!(
            populated_value["graphRelatedTo"],
            serde_json::json!(["Seed A", "Seed B"])
        );
    }

    #[test]
    fn graph_quota_scales_from_thirty_to_fifteen_percent() {
        assert_eq!(graph_result_quota(1, 0), 0);
        assert_eq!(graph_result_quota(20, 0), 6);
        assert_eq!(graph_result_quota(20, 10), 5);
        assert_eq!(graph_result_quota(20, 20), 3);
        assert_eq!(graph_result_quota(10, 100), 2);
    }

    #[test]
    fn vector_only_materialization_uses_chunk_snippet_and_any_wiki_subdir() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/custom/deep-page.md",
            "---\ntitle: Deep Page\n---\n\n# Deep Page\n\nThe literal query is absent here.",
        );
        let vector_results = vec![PageVectorResult {
            id: "deep-page".to_string(),
            score: 0.91,
            chunk_text: "A semantic chunk explains the actual reason for retrieval.".to_string(),
            heading_path: "Section > Detail".to_string(),
        }];
        let mut results = Vec::new();
        let pages = BTreeMap::from([(
            "deep-page".to_string(),
            "wiki/custom/deep-page.md".to_string(),
        )]);

        materialize_vector_only_results(
            &vector_results,
            &pages,
            &root.to_string_lossy(),
            &mut results,
            false,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "wiki/custom/deep-page.md");
        assert_eq!(results[0].title, "Deep Page");
        assert_eq!(results[0].vector_score, Some(0.91));
        assert!(results[0].snippet.contains("Section > Detail"));
        assert!(results[0].snippet.contains("semantic chunk"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn vector_candidate_materialization_propagates_missing_file_error() {
        let root = tmp_project();
        let vector_results = vec![PageVectorResult {
            id: "missing-page".to_string(),
            score: 0.8,
            chunk_text: "stale vector chunk".to_string(),
            heading_path: String::new(),
        }];
        let pages = BTreeMap::from([(
            "missing-page".to_string(),
            "wiki/concepts/missing-page.md".to_string(),
        )]);

        let error = materialize_vector_only_results(
            &vector_results,
            &pages,
            &root.to_string_lossy(),
            &mut Vec::new(),
            true,
        )
        .unwrap_err();

        assert!(error.contains("Failed to materialize vector candidate"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn vector_snippet_empty_chunk_does_not_echo_query() {
        let vector = PageVectorResult {
            id: "empty".to_string(),
            score: 0.5,
            chunk_text: "  ".to_string(),
            heading_path: "Heading".to_string(),
        };

        assert_eq!(build_vector_snippet(&vector), "");
    }

    #[test]
    fn page_vector_score_does_not_reward_chunk_count_or_saturate() {
        let chunk = |index: u32, score: f32| vectorstore::ChunkSearchResult {
            chunk_id: format!("page#{index}"),
            page_id: "page".to_string(),
            chunk_index: index,
            chunk_text: String::new(),
            heading_path: String::new(),
            score,
        };

        let short_page = vec![chunk(0, 0.82)];
        let long_noise_page = vec![
            chunk(0, 0.70),
            chunk(1, 0.70),
            chunk(2, 0.70),
            chunk(3, 0.70),
            chunk(4, 0.70),
        ];

        assert!((blend_page_chunk_scores(&short_page) - 0.82).abs() < 0.000001);
        assert!((blend_page_chunk_scores(&long_noise_page) - 0.70).abs() < 0.000001);
        assert!(blend_page_chunk_scores(&short_page) > blend_page_chunk_scores(&long_noise_page));
    }

    #[tokio::test]
    async fn keyword_search_prefers_filename_exact_match() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/attention.md",
            "---\ntitle: Attention\n---\n\n# Attention\n\nbody about attention.",
        );
        write_page(
            &root,
            "wiki/concepts/random.md",
            "---\ntitle: Random\n---\n\n# Random\n\nattention is mentioned briefly.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "attention".into(),
            20,
            false,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "keyword");
        assert_eq!(out.results[0].title, "Attention");
        assert!(out.results[0].title_match);
        assert!(out.results[0].score > 100.0);
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn formal_search_excludes_backups_and_keeps_legitimate_wiki_structures() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/sources/store-source.md",
            "---\ntitle: 门店地址来源\n---\n\n门店地址证据。",
        );
        write_page(
            &root,
            "wiki/entities/store.md",
            "---\ntitle: 门店实体\n---\n\n门店地址实体。",
        );
        write_page(
            &root,
            "wiki/concepts/store.md",
            "---\ntitle: 门店概念\n---\n\n门店地址概念。",
        );
        write_page(
            &root,
            "wiki/entities/store.bak.md",
            "---\ntitle: 门店地址备份\n---\n\n门店地址备份。",
        );
        write_page(
            &root,
            "wiki/backups/store.md",
            "---\ntitle: 门店地址维护副本\n---\n\n门店地址维护副本。",
        );
        write_page(
            &root,
            "wiki/.maintenance/store.md",
            "---\ntitle: 门店地址临时副本\n---\n\n门店地址临时副本。",
        );

        let output = search_project_inner(
            root.to_string_lossy().to_string(),
            "门店地址".to_string(),
            50,
            true,
            None,
            None,
        )
        .await
        .unwrap();
        let paths = output
            .results
            .iter()
            .map(|result| result.path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"wiki/sources/store-source.md"));
        assert!(paths.contains(&"wiki/entities/store.md"));
        assert!(paths.contains(&"wiki/concepts/store.md"));
        assert!(!paths.iter().any(|path| path.contains(".bak.")));
        assert!(!paths.iter().any(|path| path.contains("/backups/")));
        assert!(!paths.iter().any(|path| path.contains("/.maintenance/")));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_always_blends_available_graph_neighbors() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/agent.md",
            "---\ntitle: Agent Runtime\n---\n\n# Agent Runtime\n\nagent runtime details. [[Tool Registry]]",
        );
        write_page(
            &root,
            "wiki/concepts/tool-registry.md",
            "---\ntitle: Tool Registry\n---\n\n# Tool Registry\n\nDefines callable tools.",
        );
        write_page(
            &root,
            "wiki/concepts/unrelated.md",
            "---\ntitle: Unrelated\n---\n\n# Unrelated\n\nNo graph connection.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "agent runtime".into(),
            10,
            false,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "hybrid");
        assert_eq!(out.graph_hits, 1);
        assert!(out
            .results
            .iter()
            .any(|result| result.title == "Agent Runtime"));
        assert!(out.results.iter().any(|result| {
            result.title == "Tool Registry"
                && result.snippet.contains("Graph neighbor")
                && result.graph_related_to == vec!["Agent Runtime"]
        }));
        assert!(!out.results.iter().any(|result| result.title == "Unrelated"));
        let _ = fs::remove_dir_all(root);
    }

    fn injected_vector_store_failure<'a>(
        _project_path: &'a str,
        _query_embedding: Vec<f32>,
        _top_k: usize,
    ) -> VectorSearchFuture<'a> {
        Box::pin(async { Err("injected vector store failure".to_string()) })
    }

    #[tokio::test]
    async fn explicit_vector_modes_propagate_vector_store_errors() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/agent.md",
            "---\ntitle: Agent\n---\n\n# Agent\n\nagent details.",
        );

        for mode in [SearchMode::Vector, SearchMode::Hybrid] {
            let error = search_project_by_mode_with_vector_search(
                root.to_string_lossy().to_string(),
                "agent".into(),
                10,
                false,
                Some(vec![0.1, 0.2]),
                None,
                mode,
                injected_vector_store_failure,
            )
            .await
            .unwrap_err();

            assert!(error.contains("injected vector store failure"));
            assert!(error.contains(mode.as_str()));
        }
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn non_vector_modes_do_not_call_vector_store() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/agent.md",
            "---\ntitle: Agent\n---\n\n# Agent\n\nagent details.",
        );

        for mode in [SearchMode::Bm25, SearchMode::Graph] {
            let response = search_project_by_mode_with_vector_search(
                root.to_string_lossy().to_string(),
                "agent".into(),
                10,
                false,
                Some(vec![0.1, 0.2]),
                None,
                mode,
                injected_vector_store_failure,
            )
            .await
            .unwrap();

            assert_eq!(response.executed_mode, mode);
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn page_links_resolve_outgoing_backlinks_and_missing_targets() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/Alpha.md",
            "---\ntitle: Alpha\n---\n# Alpha\n\n[[Beta|label]], [[Alpha]], [[Title Only]], and [[Missing Page]].",
        );
        write_page(
            &root,
            "wiki/concepts/Beta.md",
            "---\ntitle: Beta\n---\n# Beta\n\nBeta details.",
        );
        write_page(
            &root,
            "wiki/concepts/different-filename.md",
            "---\ntitle: Title Only\n---\n# Title Only\n",
        );
        write_page(
            &root,
            "wiki/concepts/gamma.md",
            "---\ntitle: Gamma\n---\n# Gamma\n\nGamma references [[Alpha]] here.",
        );

        let links = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/Alpha.md").to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(links.outgoing.len(), 1);
        assert_eq!(links.outgoing[0].title, "Beta");
        assert_eq!(links.backlinks.len(), 1);
        assert_eq!(links.backlinks[0].title, "Gamma");
        assert!(links.backlinks[0]
            .snippet
            .as_deref()
            .unwrap()
            .contains("Alpha"));
        assert_eq!(links.missing.len(), 2);
        assert!(links
            .missing
            .iter()
            .any(|entry| entry.title == "Missing Page"));
        assert!(links
            .missing
            .iter()
            .any(|entry| entry.title == "Title Only"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn page_links_require_markdown_input_and_exact_reader_paths() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/current.md",
            "[[wiki/concepts/target.md]] [[target#section]]",
        );
        write_page(&root, "wiki/concepts/target.md", "# Target");
        write_page(&root, "wiki/concepts/not-markdown.txt", "text");

        let links = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/current.md").to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(links.outgoing.len(), 1);
        assert_eq!(
            links.outgoing[0].path.as_deref(),
            Some("wiki/concepts/target.md")
        );
        assert_eq!(links.missing.len(), 1);
        assert_eq!(links.missing[0].title, "target#section");

        let error = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/not-markdown.txt")
                .to_str()
                .unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("Markdown"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_handles_cjk_bigram_queries() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/tacit.md",
            "---\ntitle: 默会知识\n---\n\n# 默会知识\n\n默会知识强调难以言明的实践经验。",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "默会知识".into(),
            20,
            false,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(out.results[0].title, "默会知识");
        assert!(out.results[0].snippet.contains("默会知识"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_phrase_in_content_beats_scattered_tokens() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/phrase.md",
            "---\ntitle: Phrase\n---\n\n# Phrase\n\nThe phrase vector database appears together.",
        );
        write_page(
            &root,
            "wiki/concepts/scattered.md",
            "---\ntitle: Scattered\n---\n\n# Scattered\n\nvector appears here. database appears later.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "vector database".into(),
            20,
            false,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(out.results[0].title, "Phrase");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn normalize_scan_roots_defaults_and_filters() {
        assert_eq!(normalize_scan_roots(None), vec!["wiki".to_string()]);
        assert_eq!(
            normalize_scan_roots(Some(vec![
                "raw/sources".into(),
                "../etc".into(),
                "wiki".into(),
                "wiki".into()
            ])),
            vec!["raw/sources".to_string(), "wiki".to_string()]
        );
        assert_eq!(
            normalize_scan_roots(Some(vec!["secret".into()])),
            vec!["wiki".to_string()]
        );
    }

    #[tokio::test]
    async fn sources_only_scans_raw_sources_not_wiki() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/wiki-only.md",
            "---\ntitle: Wiki Only\n---\n\n# Wiki Only\n\nZZWIKIONLYMARKER body.",
        );
        write_page(
            &root,
            "raw/sources/source-doc.md",
            "---\ntitle: Source Doc\n---\n\n# Source Doc\n\nZZSOURCEONLYMARKER body.",
        );

        let sources = search_project_inner(
            root.to_string_lossy().to_string(),
            "ZZSOURCEONLYMARKER".into(),
            10,
            true,
            None,
            Some(vec!["raw/sources".into()]),
        )
        .await
        .unwrap();
        assert_eq!(sources.results.len(), 1);
        assert!(sources.results[0].path.contains("raw/sources"));
        assert_eq!(sources.results[0].title, "Source Doc");

        let wiki_miss = search_project_inner(
            root.to_string_lossy().to_string(),
            "ZZWIKIONLYMARKER".into(),
            10,
            true,
            None,
            Some(vec!["raw/sources".into()]),
        )
        .await
        .unwrap();
        assert!(
            wiki_miss.results.is_empty(),
            "sources_only must not return wiki hits: {:?}",
            wiki_miss
                .results
                .iter()
                .map(|r| &r.path)
                .collect::<Vec<_>>()
        );

        let hybrid = search_project_inner(
            root.to_string_lossy().to_string(),
            "ZZWIKIONLYMARKER".into(),
            10,
            true,
            None,
            Some(vec!["wiki".into(), "raw/sources".into()]),
        )
        .await
        .unwrap();
        assert_eq!(hybrid.results.len(), 1);
        assert!(hybrid.results[0].path.starts_with("wiki/"));

        // Missing root should not error.
        let empty_root = tmp_project();
        write_page(
            &empty_root,
            "wiki/concepts/a.md",
            "---\ntitle: A\n---\n\n# A\n\nhello",
        );
        let skip_missing = search_project_inner(
            empty_root.to_string_lossy().to_string(),
            "hello".into(),
            10,
            false,
            None,
            Some(vec!["raw/sources".into()]),
        )
        .await
        .unwrap();
        assert!(skip_missing.results.is_empty());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(empty_root);
    }

    #[test]
    fn parses_openai_batch_vectors_in_input_order() {
        let response = json!({
            "data": [
                { "index": 1, "embedding": [2.0, 2.5] },
                { "index": 0, "embedding": [1.0, 1.5] }
            ]
        });
        let vectors = parse_embedding_batch_values(&response, 2).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 1.5], vec![2.0, 2.5]]);
    }

    #[test]
    fn rejects_duplicate_openai_batch_indexes() {
        let response = json!({
            "data": [
                { "index": 0, "embedding": [1.0] },
                { "index": 0, "embedding": [2.0] }
            ]
        });
        assert!(parse_embedding_batch_values(&response, 2)
            .unwrap_err()
            .contains("duplicate"));
    }
}
