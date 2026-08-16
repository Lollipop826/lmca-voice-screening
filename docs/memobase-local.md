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

## 运维边界

查看本地 Memobase 日志：

```powershell
docker compose logs -f memobase-server-api
```

停止容器不会删除 `data/memobase/`。这些目录包含长期记忆的 PostgreSQL 与 Redis 数据，应纳入受控备份；不要将 `.env`、访问令牌或备份中的患者数据提交到版本库。
