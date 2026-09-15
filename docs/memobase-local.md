# 本地 Memobase 部署

根目录 `docker-compose.yml` 复用仓库 `vendor/memobase/src/server` 的服务定义：

- `memobase-server-db`：`pgvector/pgvector:pg17`
- `memobase-server-redis`：`redis:7.4`
- `memobase-server-api`：由 `vendor/memobase/src/server/api` 构建

SQLite 仍是本项目的事实源。Memobase 不可用时，语音服务会继续使用本地 SQLite；这三个容器只提供长期记忆镜像与检索。

## 前置条件

- Docker Desktop 或 Docker Engine，且 `docker compose version` 可用。
- 可访问 Docker 镜像仓库和阿里云百炼。Memobase API 的随仓库 `config.yaml` 使用 `qwen-plus` 和 `text-embedding-v4`，因此需要有效的 `DASHSCOPE_API_KEY`。
- 项目根目录有 `.env`。首次部署可执行 `Copy-Item .env.example .env`，再编辑实际值。

## 配置

在 `.env` 中至少确认以下配置：

```text
DASHSCOPE_API_KEY=<阿里云百炼 Key>
MEMOBASE_PROJECT_URL=http://127.0.0.1:8019
MEMOBASE_API_KEY=<本地 API 访问令牌>
ACCESS_TOKEN=<与 MEMOBASE_API_KEY 相同的本地 API 访问令牌>
DATABASE_PASSWORD=<PostgreSQL 密码>
REDIS_PASSWORD=<Redis 密码>
```

其余 `DATABASE_*`、`REDIS_*`、`API_*` 和 `PROJECT_ID` 使用 `.env.example` 中与 vendor 配置一致的默认值即可。`data/memobase/postgres` 与 `data/memobase/redis` 保存容器数据，均位于已忽略的项目运行数据目录。

## 启动与验证

在项目根目录执行：

```powershell
docker compose up -d --build memobase-server-db memobase-server-redis memobase-server-api
docker compose ps
Invoke-RestMethod -Uri http://127.0.0.1:8019/api/v1/healthcheck
```

API 的启动流程会检查 LLM 与嵌入模型配置；健康检查可用表示 Memobase API 已完成启动。数据库、Redis 和 API 分别仅绑定到本机 `127.0.0.1:15432`、`127.0.0.1:16379` 和 `127.0.0.1:8019`。容器之间使用 Docker 网络中的服务名通信。

启动完整服务使用：

```powershell
docker compose up -d --build
```

直接在宿主机运行 `voice_server.py` 时，应用会使用 `.env` 中的 `http://127.0.0.1:8019`。

## 容器边界

宿主机运行 `voice_server.py` 时，应用使用 `.env` 中的 `http://127.0.0.1:8019`。完整 Compose 启动时，`voice-server` 会覆盖为 `http://memobase-server-api:8000`；应用仅额外允许这个固定的 Compose 服务名，不接受任意远端主机。

Memobase API 比语音服务晚就绪时，语音服务会继续写 SQLite，并按既有重试机制补连和补发 outbox。只启动 `voice-server` 而不启动 Memobase 服务时，同样会安全地保持 SQLite 回退。

## 语音检索延迟控制

语音检索可以继续保持关闭：将实际 `.env` 中的 `MEMOBASE_PROJECT_URL` 留空即可。
以下优化不自动启用 Memobase，也不更换 Embedding 模型或重建向量库。

- 应用缓存非空的远程事件检索结果，默认 TTL 为 60 秒、最多 128 条。缓存键包含患者、
  查询文本、本地记忆 revision、topk、相似度阈值及时间范围；用户之间不共享检索结果。
- 相同键的并发检索合并为一个 HTTP 查询链。不同查询默认最多同时执行 2 条，满额时
  立即降级而不是无限排队；失败和空结果不进入结果缓存，避免阻止后续恢复和新记忆可见。
- 命中缓存或收到慢请求结果后，仍重新执行 SQLite 来源、删除记录及当前有效记忆校验。
  本地修改导致 revision 改变后不复用旧键；绕过本应用的远端更新最多受 TTL 限制。
- 连接预热仅准备客户端和用户，不再发送“语义检索预热”文本调用一次 Embedding。
- ASR 文本稳定 350ms 后可以预取，两次预取至少间隔 500ms；同一轮未完成的预取不被
  partial 更新反复取消重发。最终文本规范化后不同仍重新检索，不按编辑距离直接复用旧答案。

可在实际 `.env` 中覆盖以下值（修改配置后需要在安全时段重启主服务）：

```dotenv
MEMORY_RETRIEVAL_TIMEOUT_S=0.25
MEMORY_RETRIEVAL_HTTP_TIMEOUT_S=1.5
MEMORY_RETRIEVAL_CACHE_TTL_S=60
MEMORY_RETRIEVAL_CACHE_MAX_ENTRIES=128
MEMORY_RETRIEVAL_MAX_INFLIGHT=2
MEMORY_PREFETCH_STABILITY_S=0.35
MEMORY_PREFETCH_MIN_INTERVAL_S=0.5
```

`MEMORY_RETRIEVAL_TIMEOUT_S` 是最终语音回复等待检索结果的预算，不是冷查询真实耗时。
默认 250ms 到期后跳过本轮远程长期事件，保留原有上下文策略；本地降级数据不会计作
语义检索命中。这会影响未及时完成的当轮召回，质量优先时可调大预算。

检索 HTTP 复用已有连接池，用户查找、创建和搜索使用剩余短预算，连接/连接池等待另有
小上限；后台镜像写入仍保留原来的 30 秒超时。取消 `asyncio` 任务不能强行终止线程中
已发出的 HTTP 请求，因此同时保留底层 I/O 超时和并发上限，不能把取消任务当成服务端
已经停止计算。迟到结果不会替换本轮已使用的新结果，但成功结果可以进入缓存。

日志 `[Latency] memobase_event_gist_search_ms=...` 新增 `cache=miss|hit|shared`，分别表示
实际加载、缓存命中及并发复用。离线回归可运行：

```bash
python -m pytest -q tests/test_memory_retrieval_cache.py tests/voice/test_realtime_companion.py
```

性能 A/B 对照时，`MEMORY_RETRIEVAL_CACHE_TTL_S=0` 可关闭结果缓存，但仍保留请求合并和
并发限制。分别记录缓存命中率、检索超时/降级率、真实语义命中率和语音首音时间。
新查询仍需要远程 Embedding；要降低冷查询耗时，应另行评测本地或更快的向量模型，
并在更换模型时重新生成全部文档向量、验证召回质量。

## 运维边界

查看本地 Memobase 日志：

```powershell
docker compose logs -f memobase-server-api
```

停止容器不会删除 `data/memobase/`。这些目录包含长期记忆的 PostgreSQL 与 Redis 数据，应纳入受控备份；不要将 `.env`、访问令牌或备份中的患者数据提交到版本库。
