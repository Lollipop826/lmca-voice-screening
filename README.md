# 语音认知筛查服务（交付包）

> 交接入口：先阅读 [HANDOVER.md](HANDOVER.md)。其中记录当前磁盘状态、启动命令、验证结果、已知失败和未实施方案。本文档保留为项目说明，不保证完全反映当前工作区的本地数据状态。

面向临床/照护场景的**低压力语音认知筛查**服务。医护人员通过手机大小的界面引导患者完成 MMSE 式对话任务，系统负责语音交互、状态管理、打断恢复、评分与会话历史；患者全程只用语音参与，不需要看屏或操作。

本包是从完整开发目录裁剪出的**交付版**（约 3.9 GB，原目录 40 GB）。已去掉虚拟环境、训练产物、运行数据和真实密钥，只保留运行所需的代码与模型。详见文末「本包裁剪了什么」。

---

## 1. 这个项目是干什么的

- **服务入口**：`voice_server.py`，FastAPI 应用，默认监听端口 **8502**，同时支持 WebRTC 和兼容用的 WebSocket 接入。
- **对话流程**：每条连接有独立的会话与筛查 Agent，编排 ASR → Agent → TTS → 评分 → 持久化的完整回合。
- **语音识别 / 合成 / 大模型**：正式回复默认走阿里云百炼 `qwen3.7-flash`，会话长期整理走 `qwen-plus`；语音 ASR/TTS 使用火山引擎。SiliconFlow 保留为未配置百炼时的 LLM 回退，也可切换为本地 ASR（SenseVoice）/本地 TTS（ZipVoice）。
- **声纹验证**：本地 ONNX 模型（ERes2NetV2 中文 / WeSpeaker），确认是否为同一说话人。
- **全双工打断**：可选接入外部 **SoulX-Duplug** 服务做语义轮次判断（见第 5 节，属外部依赖，本包不含）。

### 目录职责

| 路径 | 职责 |
| --- | --- |
| `voice_server.py` | 服务组合入口，负责配置、依赖组装、FastAPI 生命周期和 WebRTC 接入 |
| `src/voice/` | 连接、会话、媒体、运行时、持久化和语音业务 |
| `src/voice/handlers/` | 按消息类型拆分的处理对象 |
| `src/web/` | 登录、管理页面、公开 API 和 HTTP 路由 |
| `src/agents/screening/` | 筛查状态、任务规划、回合阶段和领域策略 |
| `src/tools/` | Agent、检索、语音等外部能力适配器 |
| `src/context_management/` | 患者长期记忆和上下文策略 |
| `static/` | 浏览器页面和静态资源（含 `voice_chat.html`） |
| `kb/` | 检索知识库文本切片（RAG 用） |
| `models/` | 本地 ONNX / TTS 模型（见第 4 节） |
| `docs/` | 部署、公开 API、SoulX、声纹、WebRTC 文档 |
| `tests/` | 自动化测试 |

建议阅读顺序：`PRODUCT.md` → `DESIGN.md` → `DELIVERY.md` → `voice_server.py` → `src/voice/application.py`。

---

## 2. 环境要求

- Python **3.11**（Dockerfile 用的就是 3.11-slim）
- 系统依赖：`ffmpeg`、`libsndfile1`（Docker 已内置；本地部署需自行安装）
- 云端 API 账号：阿里云百炼（正式回复、长期整理和 Memobase）、火山引擎豆包语音；SiliconFlow 仅作 LLM 回退（见第 6 节密钥）
- 纯 API 模式**不需要 GPU**；仅在启用本地 TTS（ZipVoice）或本地 ASR 时才需要

---

## 3. 如何部署

### 方式 A：Docker（推荐，纯 API 模式）

```powershell
# 1. 准备密钥（本包只带了模板，没有真实密钥）
Copy-Item .env.example .env
# 编辑 .env，填入阿里云百炼和火山引擎豆包的实际 Key（见第 6 节）

# 2. 构建并启动
docker compose up -d --build

# 3. 健康检查
Invoke-WebRequest http://localhost:8502/health
```

访问 `http://<服务器IP>:8502/`。注意 `docker-compose.yml` 会挂载 `./data`、`./tmp`，首次启动会自动创建。Silero VAD 模型在构建镜像时自动下载，不依赖本包里的 `models/silero_vad.onnx`。

根目录 Compose 还包含本地 Memobase API、PostgreSQL 和 Redis；启动前需在 `.env` 中填写 `DASHSCOPE_API_KEY`，并让 `ACCESS_TOKEN` 与 `MEMOBASE_API_KEY` 保持相同。完整 Compose 启动时，语音服务通过固定的 Docker 服务名连接本地 Memobase；仅需语音服务及 SQLite 回退时，可执行 `docker compose up -d --build voice-server`。完整的部署、健康检查和数据备份边界见 [docs/memobase-local.md](docs/memobase-local.md)。

### 方式 B：本地脚本运行

```bash
# 1. 建虚拟环境并装依赖（本包不含虚拟环境）
python3.11 -m venv luyang
source luyang/bin/activate
pip install -r requirements.txt

# 2. 准备密钥
cp .env.example .env   # 然后编辑填入真实 Key

# 3. 启动
bash start_voice_only.sh
```

`start_voice_only.sh` 会自动定位 Python、设置 `PYTHONPATH`、清理旧进程，并在 `USE_SOULX_TURN_TAKING=true` 时尝试拉起 SoulX 轮次服务。**如果没有 SoulX 环境**，请在 `.env` 里设 `USE_SOULX_TURN_TAKING=false` 再启动，否则脚本会因找不到 SoulX 而退出。

> 脚本里 `SOULX_DIR`、`SOULX_PYTHON`、`CUDA_HOME=/usr/local/cuda-11.8` 等是原机器的绝对路径，换机器需按实际环境改，或直接用 Docker 方式绕开。

---

## 4. 本包包含的模型（`models/`，共约 3.7 GB）

| 模型 | 大小 | 用途 |
| --- | --- | --- |
| `bge-reranker-base/` | 2.1 G | RAG 精排（`RAG_FUSION_ENABLE_RERANKING=true` 时用） |
| `zipvoice_distill/` | 1.5 G | 本地 TTS（`USE_ARK_TTS=false` 时用） |
| `vocos-mel-24khz/` | 104 M | ZipVoice 声码器 |
| `eres2netv2-cn/` | 69 M | 声纹验证（默认，中文 20 万说话人） |
| `wespeaker-cnceleb/` | 26 M | 声纹验证（`resnet34` 快速回退） |
| `silero_vad.onnx` | 2.3 M | 语音活动检测（VAD） |

`silero_vad.onnx` 原开发目录 `models/` 里没有（代码从 torch 缓存加载），本包已从缓存补入，本地运行开箱即用。

---

## 5. 缺少什么 / 需要自行准备

裁剪时刻意去掉了这些，接收方需要自己补：

1. **Python 虚拟环境** — 本包不含，按第 3 节 `pip install -r requirements.txt` 重建。
2. **真实密钥 `.env`** — 只带了 `.env.example` / `.env.cloudflare.example` 模板，出于安全没带真实密钥。必须自己填（第 6 节）。
3. **SoulX-Duplug 服务** — 全双工打断/语义轮次判断依赖的外部服务（原机器在 `/home/luy/luyang/pause/SoulX-Duplug-main`，单独的 conda 环境 + GPU）。**本包不含**。不需要它时把 `.env` 里 `USE_SOULX_TURN_TAKING=false` 即可正常运行。详见 `docs/soulx_8502_integration.md`。
4. **运行数据 `data/`** — 历史语音录音、评分、对话记录等，属隐私数据未带。服务首次运行会自动创建空目录。
5. **知识库原始来源** — 本包带了 `kb/` 切片文本；若要重建向量库需自行处理。

### 本包裁剪了什么（相对原 40 GB 开发目录）

- `luyang/`（Python 虚拟环境，8.9 G）
- 代码中 0 引用的模型：`ad_resistance_detector_multiclass/`（21 G，抗性检测实际走云端 API `RESISTANCE_MODEL`）、`resistance_detector_4class/`、`resistance_detector_innovative/`、`spkrec-ecapa-voxceleb/`
- `data/`（运行数据，含语音录音）、`tmp/`（日志）、`delivery/`（旧打包）、`__pycache__/`
- 真实密钥文件 `.env`、`.env.cloudflare`、`.env.storage`

---

## 6. 必填密钥（`.env`）

编辑 `.env`，重点填以下几项（完整说明见 `.env.example` 注释）：

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 阿里云百炼 Key；正式回复、长期整理和本地 Memobase 都需要 |
| `DASHSCOPE_CHAT_MODEL` | 正式回复模型，默认 `qwen3.7-flash` |
| `DASHSCOPE_CONSOLIDATION_MODEL` | 会话结束后的长期整理模型，默认 `qwen-plus` |
| `SILICONFLOW_API_KEY` | 未配置百炼时的 LLM 回退 |
| `VOLC_APP_ID` / `VOLC_ACCESS_TOKEN` | 火山引擎豆包语音（云端 ASR/TTS） |
| `USE_ARK_ASR` / `USE_ARK_TTS` | 云端(true)还是本地(false) ASR/TTS |
| `USE_SOULX_TURN_TAKING` | 没有 SoulX 服务时设为 `false` |
| `PUBLIC_API_KEYS` | 对外 REST API（`/v1`）调用方 Key；不填会自动生成 bootstrap Key 到 `data/.public_api_key` |

抗性检测走 `RESISTANCE_MODEL` 指定的云端模型（默认 `doubao-seed-2-0-mini`），对应厂商的 Key 也需配置。

---

## 7. 相关文档

- `PRODUCT.md` — 产品定位与设计原则
- `DESIGN.md` — 架构设计
- `DELIVERY.md` — 交付说明与调用链
- `docs/public-api.md` — 对外 REST API
- `docs/speaker-verification.md` — 声纹验证
- `docs/webrtc_migration.md` — WebRTC 接入
- `docs/soulx_8502_integration.md` — SoulX 全双工集成
- `docs/cloudflare-deployment.md` — Cloudflare 隧道部署
