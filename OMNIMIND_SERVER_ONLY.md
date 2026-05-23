# OmniMind Wiki Core Server-Only 运行说明

本文件定义 OmniMind 对 LLM Wiki fork 的无头服务化入口。目标是先把作者已有能力放进稳定服务边界，再验证和迁移作者已开放的本地 API。

## 边界原则

- 不物理删除上游 UI 代码；服务化模式只是不启动 WebView。
- 默认桌面入口继续执行上游 `llm_wiki_lib::run()`。
- OmniMind 服务入口只通过 `--server-only` 启动。
- Python 后端只允许对接 server-only 端口，不直接依赖 Tauri 桌面 API 端口。
- `chat` / `stream-chat` 在 Python Runtime SDK 编排完成前必须显式未实现，不能假装接管 SDR/模拟客户咨询主回复。

## 本地运行

```bash
npm run server:dev
```

等价命令：

```bash
cargo run --manifest-path src-tauri/Cargo.toml --bin llm-wiki -- --server-only
```

默认监听：

```text
127.0.0.1:19829
```

可通过环境变量覆盖：

```bash
OMNIMIND_WIKI_CORE_BIND=127.0.0.1:19829 npm run server:dev
```

## 构建服务二进制

```bash
npm run server:build
```

构建产物仍是 `llm-wiki` binary，但运行时必须传入 `--server-only`：

```bash
./src-tauri/target/release/llm-wiki --server-only
```

## 首轮验证命令

```bash
curl http://127.0.0.1:19829/api/v1/health

curl -X POST http://127.0.0.1:19829/api/v1/projects/default/chat-context \
  -H 'Content-Type: application/json' \
  -d '{"query":"客户问售后 SLA 是什么？","max_context_size":204800,"output_language":"Chinese","include_debug":true}'

curl -X POST http://127.0.0.1:19829/api/v1/projects/default/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello"}'
```

预期：

- `/api/v1/health` 返回 `ok=true`。
- `/chat-context` 返回 `status=EMPTY_CONTEXT`、`budget`、空 `context_blocks` 与空 `references`。
- `/chat` 返回 `501 CHAT_NOT_IMPLEMENTED`。

## 作者已有 API 验证顺序

作者当前 Tauri API 已有以下接口：

1. `GET /api/v1/projects`
2. `GET /api/v1/projects/{id}/files`
3. `GET /api/v1/projects/{id}/files/content`
4. `POST /api/v1/projects/{id}/search`
5. `GET /api/v1/projects/{id}/graph`
6. `POST /api/v1/projects/{id}/sources/rescan`

OmniMind 的接入顺序：

1. 先在 server-only 端口复刻只读接口：`projects`、`files`、`files/content`。
2. 再迁移检索与图谱接口：`search`、`graph`。
3. 再把真实结果接入 `chat-context`。
4. 最后评估带副作用的 `sources/rescan`。

不得跳过 server-only 端口，直接让 Python 长期依赖作者桌面 Tauri API。
