# Memobase 本地集成方案 - 心理健康对话系统

## 1. 目标与边界

本方案只做一件事：让系统在不把聊天记录无限塞进上下文的前提下，能按患者当前的话找回相关旧事。

必须满足：

1. 完整转写、音频路径、情绪轨迹和 MMSE 继续保存在现有 SQLite 中，作为唯一可追溯档案。
2. Memobase 完全在本机部署，不连接 `api.memobase.dev` 或其他云端项目。
3. 每轮回复都带入与患者当前话语相关的长期记忆，而不是只在会话开始时塞一张固定记忆卡。
4. 给模型的总上下文有固定上限；会话再长也不会无限增加。
5. Memobase 不可用时，语音对话和 SQLite 落库仍能继续。

本方案不包含：

- ChromaDB、FAISS 或另一套自建向量库。
- 情绪识别模型对比实验。
- 重做现有 ASR、情绪识别、MMSE 或音频存储。

它们与 Memobase 集成不是同一个任务，不能混在这份文档里。

## 2. 最终架构

```text
原始音频
  -> ASR 转写 + 情绪识别
  -> SQLite: 完整档案，永久保存
  -> Memobase: 用户画像、事件时间线、语义检索
  -> 有上限的上下文
  -> Agent 生成回复
```


| 组件                     | 职责                                                   | 不负责什么            |
| -------------------------- | -------------------------------------------------------- | ----------------------- |
| SQLite                   | 原始转写、助手回复、音频路径、情绪、MMSE、会话 ID      | 不负责语义召回        |
| Memobase                 | 患者画像、事件提炼、embedding 检索、生成长期记忆上下文 | 不作为病历原件        |
| `ConversationMemoryTool` | 本会话摘要、最近少量原话、当前任务状态                 | 不保存跨会话长期档案  |
| Agent                    | 根据当前话语和限定后的上下文回复                       | 不直接查询完整 SQLite |

Memobase 默认可以在处理后清理原始聊天 Blob。因此，无论 Memobase 是否成功，SQLite 都必须保留完整记录；不能反过来把 Memobase 当病历库。

## 3. 版本和本地部署

### 3.1 固定版本

使用 Memobase 官方最新稳定标签 `v0.0.42`，不再使用项目中现有的 `0.0.27`。

客户端和服务端必须来自**同一个标签**，不能用新版 Python 客户端连接一个旧服务端，也不能使用浮动的 `latest` 镜像。版本来源为官方仓库：[https://github.com/memodb-io/memobase](https://github.com/memodb-io/memobase)

项目依赖应改为：

```text
memobase @ git+https://github.com/memodb-io/memobase.git@v0.0.42
```

安装后必须验证：

```powershell
python -c "import memobase; print(memobase.__version__)"
```

预期输出为 `0.0.42`。若不是，停止后续接入，不允许混用版本。

### 3.2 本地服务

从同一标签构建官方服务，而不是使用未验证的第三方或 `latest` 镜像：

```powershell
New-Item -ItemType Directory -Force .\vendor | Out-Null
git clone --branch v0.0.42 --depth 1 https://github.com/memodb-io/memobase.git .\vendor\memobase
Set-Location .\vendor\memobase\src\server
docker compose build
docker compose up -d
```

官方服务依赖的 Postgres 和 Redis 同样运行在本机 Docker Compose 中。生产部署应固定该仓库标签和对应的 Compose 文件；不要自行改成单容器 SQLite 版。

应用侧仅允许指向本机：

```env
MEMOBASE_PROJECT_URL=http://127.0.0.1:8019
MEMOBASE_API_KEY=secret
```

不允许配置 `https://api.memobase.dev`。启动后先执行 `MemoBaseClient(...).ping()`；失败时不要启动 Memobase 记忆功能。

### 3.3 本地 embedding

Memobase 的语义检索需要 embedding 服务。服务端配置必须启用事件 embedding，并只连接本机的 embedding 服务，例如本机 Ollama；不能填入云端 OpenAI、Jina 或其他外部 URL。

项目现有的 BGE-M3 模型池可以继续给项目内知识库使用，但 Memobase 服务端不能直接读取 Python 进程中的模型对象。Memobase 需要通过其官方本地 embedding 配置调用一个本地 HTTP 服务。部署完成后必须在 Memobase 容器内验证该本地地址可达。

## 4. 上下文规则

每轮给 Agent 的上下文固定为以下五部分：

```text
患者基础信息 / 长期画像                 <= 400 字
本轮 Memobase 相关旧事                  <= 800 tokens
本会话摘要                              <= 600 字
最近 4 条消息（2 轮对话）                每条 <= 500 字
当前 MMSE、任务状态和情绪                <= 400 字
```

其中“本轮 Memobase 相关旧事”由当前患者刚说的话触发，例如：

```text
当前：我最近又不敢一个人洗澡了。
找回：2026-06-12，患者说曾在浴室滑倒，此后害怕独自洗澡。
```

embedding 数字只用于 Memobase 找记录，不会放进模型上下文；放进去的是找回的文字、时间和来源。

### 必须修改的现有行为

`ConversationMemoryTool.get_context()` 当前返回全部 `chat_history`。这必须改为只返回最后 `_RECENT_WINDOW` 条：

```python
recent = list(chat_history[-_RECENT_WINDOW:])
```

更早的内容只保留在会话摘要中。否则即使 Memobase 的返回有上限，聊天原文仍会无限撑大上下文。

## 5. 每轮完整数据流

### 5.1 会话开始

1. 按现有流程确定 `patient_id`，关联会话和患者。
2. 只注入基础身份信息，不再把一大段固定长期记忆卡长期挂在 `persistent_background`。
3. 清空上一会话的本会话摘要、最近消息和本轮临时记忆。

### 5.2 患者说一句话后、Agent 回复前

1. ASR 生成 `context.text`，情绪识别生成 `context.emotion_scores`。
2. 以 `patient_id` 和 `context.text` 调用 Memobase：

   ```python
   user.context(
       max_token_size=800,
       chats=[{"role": "user", "content": context.text}],
       require_event_summary=True,
   )
   ```
3. 返回内容作为**本轮临时背景**，而不是跨轮永久背景。
4. 将临时背景、本会话摘要、最近 4 条消息和当前状态交给现有 Agent。

不需要在正常回复链路里再额外调用 `search_event()`。`user.context(..., chats=[当前话语])` 已负责按当前话语组织画像和相关事件。`search_event()` 只留给医生端“查历史事件”的页面或诊断接口。

### 5.3 Agent 回复后

1. 先调用现有 `EmotionMemobase.capture_turn()`，将患者文本、助手回复、情绪和音频路径写入 SQLite。
2. SQLite 成功后，把本轮文本写成 `ChatBlob` 发送给本地 Memobase：

   ```python
   ChatBlob(
       messages=[
           {"role": "user", "content": user_message},
           {"role": "assistant", "content": assistant_message},
       ],
       fields={
           "session_id": session_id,
           "source_turn_id": local_turn_id,
           "dominant_emotion": dominant_emotion,
           "created_at": captured_at,
       },
   )
   ```

不传音频文件和绝对音频路径给 Memobase。音频只留在现有会话目录和 SQLite 路径中；长期记忆只需要文本和必要的结构化情绪信息。

3. Memobase 写入失败只记录同步失败，不撤销 SQLite 落库，也不影响本轮回复。

### 5.4 会话结束

1. 调用 `user.flush(sync=True)`，让本会话缓冲内容完成处理。
2. 记录本次成功同步到的本地轮次 ID。
3. 若失败，下次会话开始前从 SQLite 重放未同步轮次。

同步状态只需记录“最后成功同步的本地轮次 ID”。不新建第二套病历或向量库。

## 6. 真实代码接入点

只改已有模块，不创建 `LocalMemoryManager`、`LocalEmbeddingService` 或 `EmotionMemory`。


| 文件                                                | 修改内容                                                                                                             |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt`、`requirements_windows.txt`      | 把`memobase>=0.0.27,<0.1.0` 改为 `v0.0.42` 的固定 Git 依赖                                                           |
| `src/context_management/emotion_memobase.py`        | 保留 SQLite 写入；改为使用本地`MemoBaseClient`；提供“按当前话语获取本轮长期记忆”和“重放未同步轮次”               |
| `src/voice/services.py`                             | `PatientMemoryService` 增加获取本轮记忆和会话结束 flush 的职责；不再把本地 SQLite 的 7000 字上下文当作唯一长期记忆卡 |
| `src/tools/agent_tools/conversation_memory_tool.py` | 截断`recent` 为最近 4 条；增加仅本轮有效的 `turn_background`，拼入 `summary`                                         |
| `src/voice/handlers/speech_turn_processor.py`       | 在 ASR/情绪识别后、`context.prepare_agent()` 前，获取本轮 Memobase 背景并设置给 `tool_gateway.memory_tool`           |
| `src/voice/handlers/session_lifecycle_handler.py`   | 会话开始只设置基础患者信息；不再注入静态长记忆全文                                                                   |

关键注入点必须是实际被 Agent 使用的 `session.agent.tool_gateway.memory_tool`。只把文本写到 `SpeechTurnContext` 的新字段没有作用，因为 Agent 不会读取它。

## 7. 故障降级


| 情况                           | 行为                                                     |
| -------------------------------- | ---------------------------------------------------------- |
| Memobase 未启动或`ping()` 失败 | 不进行语义检索；使用 SQLite 的现有有限记忆卡和本会话摘要 |
| 本轮`context()` 超时或报错     | 这一轮不带长期相关旧事，但仍继续回复                     |
| 本轮写入 Memobase 失败         | SQLite 已保存；标记待重放，不影响患者对话                |
| SQLite 写入失败                | 视为本轮持久化失败并记录错误；不能伪称记忆已保存         |

## 8. 验收标准

以下项目全部通过才算完成：

1. `memobase.__version__` 为 `0.0.42`，本地 `ping()` 成功，日志中没有云端地址。
2. 完成 20 轮对话后，Agent 每轮收到的 `recent` 仍只有 4 条，不会随轮次增加。
3. 患者在一轮说“浴室滑倒”，结束会话；下次说“我又不敢一个人洗澡”，本轮上下文能包含该旧事及时间。
4. 关闭 Memobase 后，患者仍能完成对话，SQLite 中仍有完整文本、情绪和音频路径。
5. 恢复 Memobase 后，未同步的 SQLite 轮次会被补发一次，并能用于后续检索。
6. Memobase 容器、Postgres、Redis 和 embedding 服务均仅运行在本机网络；抓取应用日志时没有任何外部记忆或 embedding 请求。

## 9. 实施顺序

1. 用 `v0.0.42` 启动本地 Memobase、Postgres、Redis 和本地 embedding，并完成 `ping()` 与单句检索验证。
2. 升级项目 Python 客户端到同一标签，替换现有 `0.0.27`。
3. 先改 `ConversationMemoryTool` 的聊天窗口和本轮临时背景，验证上下文不再无限增长。
4. 再接入“回复前检索、回复后写入、会话结束 flush”。
5. 最后做跨会话检索、离线降级和未同步重放测试。

不要同时改情绪模型、MMSE、ASR 或前端。先把“能找回相关旧事且上下文不膨胀”做成可验证闭环。
