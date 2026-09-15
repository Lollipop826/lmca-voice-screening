# LMCA Share — 语音认知筛查服务

面向临床与照护场景的低压力语音认知筛查服务。患者通过语音完成引导式对话任务；服务负责实时语音交互、筛查会话状态、打断恢复、评分、历史记录和长期记忆。

> 这是一个**代码仓库**，不包含真实密钥、患者数据、录音、运行日志、私有证书或大模型权重。请先阅读本文的「安全与发布边界」再发布或部署。


## 功能概览

- FastAPI 服务，默认端口为 `8502`；提供浏览器界面、WebRTC 与兼容 WebSocket 接入。
- 云端 ASR/TTS 与 LLM 编排，支持火山引擎语音、阿里云百炼，以及 SiliconFlow 回退。
- 语音筛查会话与评分流程，包含声纹验证、记忆检索和会话持久化。
- 可选 SoulX-Duplug 全双工轮次判断，实现用户打断和恢复播放。
- 可选本地 Memobase、PostgreSQL 和 Redis，用于长期记忆；未启用时可使用本地回退路径。

## 代码结构

- `voice_server.py`：FastAPI 服务入口与依赖装配。
- `src/voice/`：实时连接、会话、媒体处理、播放控制与持久化。
- `src/voice/handlers/`：按消息和音频事件拆分的处理器。
- `src/agents/`：认知筛查流程、会话状态和领域策略。
- `src/context_management/`：长期记忆、检索与缓存。
- `src/tools/`：模型、语音和外部服务适配层。
- `src/web/`、`static/`：登录、管理、历史记录与语音对话页面。
- `tests/`：单元测试和集成边界测试。
- `docs/`：架构、部署、接口和实验文档。

## 环境要求

- Python 3.11
- 本地运行需安装 `ffmpeg` 与 `libsndfile1`
- Docker 部署需要 Docker Compose
- 生产模式需要相应云服务的账号与密钥；没有 SoulX 时可关闭该功能

## 快速启动

### Docker（推荐）

```bash
cp .env.example .env
# 编辑 .env，填写本机或部署环境自己的密钥与地址
docker compose up -d --build voice-server
curl http://127.0.0.1:8502/health
```

若需要本地 Memobase、PostgreSQL 和 Redis，运行完整 Compose：

```bash
docker compose up -d --build
```

详细的本地记忆部署与备份边界见 [docs/memobase-local.md](docs/memobase-local.md)。

### 本地 Python 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 后启动
bash start_voice_only.sh
```

`start_voice_only.sh` 在 `USE_SOULX_TURN_TAKING=true` 时会尝试连接或启动 SoulX。没有该外部服务时，请在 `.env` 中设置 `USE_SOULX_TURN_TAKING=false`。SoulX 的独立部署说明见 [docs/soulx_8502_integration.md](docs/soulx_8502_integration.md)。

## 配置与安全

从模板创建本地配置，绝不把真实配置加入 Git：

```bash
cp .env.example .env
```

以下内容只能保留在部署机器、密码管理器或受控对象存储中，**不得提交或上传到 GitHub**：

- `.env`、`.env.*`、历史 `.env` 备份，以及任何 API Key、Token、数据库密码。
- `certs/` 中的 TLS 私钥与证书。
- `data/` 中的患者资料、对话、录音、评分和导出数据。
- `tmp/`、日志、缓存、SQLite/向量索引与运行期生成的文件。
- 本地模型、训练产物、下载的权重和大型二进制文件。
- 临时调试脚本、编辑器备份及实验过程中生成的原始结果。

`.gitignore` 已覆盖这些常见本机文件；每次提交前仍应检查暂存区：

```bash
git status --short
git diff --cached --name-only
git check-ignore -v .env certs/voice_server.key
```

若密钥曾经提交到任何分支或远程仓库，请立即在服务商后台撤销并重新生成；仅删除文件不能使旧密钥失效。

## 大文件策略

代码仓库只保存源代码、配置模板、文档、测试以及运行必需且体积可控的静态依赖。模型权重、录音、患者数据、数据库转储和实验原始输出不上传。

需要共享大模型或数据时，使用受控对象存储、机构文件服务或发布页下载链接，并在文档中记录校验和与获取方式。只有确实需要由 Git 版本化、单个文件又超过常规代码体量的资源，才在取得仓库管理员同意后使用 Git LFS；不要把数据集或密钥放进 LFS。

当前已纳入 `static/vendor/` 的前端静态运行依赖会保留，因为页面会在运行时直接加载它们；其余可再生成、可下载或含隐私的数据不应添加。

## 测试

```bash
python -m pip install -r requirements.txt -r requirements-test.txt
python -m pytest -q
```

也可按领域运行：

```bash
python -m pytest -q tests/agents
python -m pytest -q tests/voice
python -m pytest -q tests/web
python -m pytest -q tests/integrations
```

## 相关文档

- [产品概览](docs/product-brief.md)
- [架构设计](docs/design-system.md)
- [公开 API](docs/public-api.md)
- [Memobase 本地部署](docs/memobase-local.md)
- [SoulX 全双工集成](docs/soulx_8502_integration.md)
- [Cloudflare 部署](docs/cloudflare-deployment.md)
