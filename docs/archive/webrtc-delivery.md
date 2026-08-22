# 语音筛查服务交付说明

> 历史快照，记录 2026-07-16 的交付状态；不代表当前分支、测试结果或部署状态。

更新时间：2026-07-16
开发分支：`feat/webrtc-migration`

## 交付结论

当前项目已经从单文件脚本整理为可阅读、可测试、可替换的对象化服务：

- `voice_server.py` 只负责配置、依赖组装、FastAPI 生命周期和 WebRTC 接入。
- WebRTC 与兼容 WebSocket 共用同一个 `VoiceEndpointApplication`。
- 每条连接拥有独立的 `VoiceSession` 和筛查 Agent，连接状态不会互相污染。
- 语音消息由独立 handler 处理，不再集中在一个巨型端点中。
- 筛查 Agent 已取消 11 个 mixin 拼接和动态 `self` 转发，改用显式协作者组合。
- 旧 Streamlit 页面、下载脚本、演示脚本、重复依赖文件、旧 Agent、旧 ASR/TTS
  实现、过期部署文件和本地打包缓存已经清理。

交付人员建议依次阅读：

1. `../product-brief.md`
2. `../design-system.md`
3. `voice_server.py`
4. `src/voice/application.py`
5. `src/agents/screening_agent_function_calling.py`

## 当前调用链

```text
浏览器 WebRTC
  └─ WebRTCPeerSession
      └─ WebRTCVoiceConnection
          └─ VoiceEndpointApplication

兼容 WebSocket
  └─ VoiceEndpointApplication

VoiceEndpointApplication
  ├─ VoiceSession
  ├─ VoiceConnectionIO
  ├─ VoiceMessageRouter
  ├─ 语音消息 handlers
  ├─ 持久化与患者记忆服务
  └─ ADScreeningAgentFunctionCalling
      └─ TurnPipeline
          ├─ ScreeningComfortPhase
          ├─ ScreeningAnswerPhase
          ├─ ScreeningRoutingPhase
          └─ ScreeningQuestionPhase
```

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `voice_server.py` | 服务组合入口，不承载核心业务流程 |
| `src/voice/` | 连接、会话、媒体、运行时、持久化和语音业务 |
| `src/voice/handlers/` | 按消息类型拆分的处理对象 |
| `src/web/` | 登录、管理页面、公开 API 和 HTTP 路由 |
| `src/agents/screening/` | 筛查状态、任务规划、回合阶段和领域策略 |
| `src/tools/` | Agent、检索、语音等外部能力适配器 |
| `src/context_management/` | 患者长期记忆和上下文策略 |
| `static/` | 当前服务实际使用的浏览器页面和静态资源 |
| `docs/` | 部署、公开 API、SoulX、声纹和 WebRTC 文档 |
| `tests/` | 开发阶段自动化测试；正式交付包可不包含 |

## 核心对象

- `VoiceEndpointApplication`：一条语音连接的组合根。
- `VoiceSession`：保存单连接状态，替代端点中的局部变量和 `nonlocal`。
- `WebRTCVoiceConnection`：把 WebRTC 适配为统一连接接口。
- `VoiceMessageRouter`：按消息类型分发到独立 handler。
- `SpeechTurnProcessor`：编排 ASR、Agent、TTS、评分和持久化。
- `LiveAudioInputHandler`：编排 SoulX、VAD、打断和实时音频提交。
- `ADScreeningAgentFunctionCalling`：130 行的筛查 Agent 对外门面和组合根。
- `ScreeningSessionState`：单个筛查会话的显式状态对象。
- `ScreeningToolGateway`：所有外部工具的访问边界。
- `ScreeningTaskPlanning`：任务候选、路由和过渡策略。
- `ScreeningConversationPolicy`：会话规则、无效回答和完成条件。
- `ScreeningAnswerEvaluation`：规则与模型回答评估。
- `ScreeningQuestionGeneration`：下一问题生成。
- `ScreeningBackgroundAnalysis`：按会话快照执行后台分析。
- `TurnPipeline`：组合四个明确回合阶段，不再依赖 mixin MRO。

## 必需文件

源码交付至少包含：

```text
voice_server.py
start_voice_only.sh
requirements.txt
Dockerfile
docker-compose.yml
config/
deploy/
docs/
src/
static/
docs/product-brief.md
docs/design-system.md
docs/archive/webrtc-delivery.md
.env.example
.env.cloudflare.example
```

以下目录属于运行环境或本地资产，不应直接提交到 Git；按部署环境单独准备：

```text
luyang/             Python 虚拟环境
models/             本地模型
data/               运行数据
kb/                 知识库
ZipVoice-master/    使用本地 ZipVoice 时需要
static/audio/       ZipVoice 参考音频
tmp/ logs/          运行时文件
```

## 验证方法

必须使用项目当前虚拟环境：

```bash
./luyang/bin/python -m pytest -q
./luyang/bin/python -m compileall -q voice_server.py src tests
git diff --check
```

本次源码验证结果：

```text
202 passed, 2 warnings
```

两个 warning 来自第三方 SWIG 类型的弃用提示，不是业务失败。

本次清理完成后，应同时确认：

```bash
docker compose config
curl http://127.0.0.1:8502/health
```

`/health` 只能证明当前进程健康。源码发生修改后，必须受控重启，才能验证新代码：

```bash
./start_voice_only.sh
```

## Git 回退

已有关键回退点：

- `b973a1d`：重构前、已运行验证的 WebRTC 稳定快照。
- `e37c50c`：语音端点组合根完成。
- `751a91f`：语音 handler 拆分完成。
- `2e4afa9`：两个 600 行语音流程完成分解。
- `968a996`：ToolGateway、TaskPlanning、BackgroundAnalysis 改为组合。
- `252eae1`：测试目录整理和本地杂项清理。
- `86ac5e8`：移除剩余 mixin，筛查 Agent 改为显式回合流水线。
- `fe7d176`：删除旧应用、下载脚本、演示代码和重复部署文件。
- `a271f00`：由 `TurnPipeline` 统一创建并持有筛查协作者。
- `bc88c20`：增加结构测试，锁定组合根和协作者所有权。

查看版本：

```bash
git log --oneline --decorate -20
```

回滚前先提交或备份当前工作区，然后切到已验证提交：

```bash
git switch --detach <commit>
```

## 后续代码债

当前架构边界已经清晰，剩余工作主要是继续缩短协作者内部实现：

- `screening/task_planning.py` 仍应拆为候选策略、LLM 路由和过渡策略。
- `screening/answer_evaluation.py` 与 `answer_flow.py` 仍可继续细分。
- `screening/tool_gateway.py` 可按评分、生成、存储和媒体工具分组。
- `src/voice/runtime.py` 与 `src/web/application.py` 仍偏大，但职责已经独立。

这些文件虽然仍长，但已经不再依赖隐式 MRO 或共享的上帝类命名空间，可以逐个安全演进。
