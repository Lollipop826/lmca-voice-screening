# 对外 REST API（v1）

基础地址：

```text
https://voice.luyang.icu
```

所有 `/v1` 接口都必须带 API Key。推荐使用 Bearer 认证：

```http
Authorization: Bearer <API_KEY>
```

也支持 `X-API-Key: <API_KEY>`。

## 获取和管理 API Key

推荐由管理员打开开发者工作台：

```text
https://voice.luyang.icu/developers#key-management
```

登录管理员账号后，可以为每个调用方创建独立 Key、查看最近调用时间并撤销 Key。完整 Key 只会在创建成功时显示一次；数据库只保存 SHA-256 哈希，之后无法恢复原文。

服务首次启动会生成一个 bootstrap Key，保存在部署机器的：

```text
data/.public_api_key
```

该文件权限为 `600`，不要提交、截图或发送到聊天中。生产环境建议在 `.env` 中配置：

```env
PUBLIC_API_KEYS=adsk_client_a,adsk_client_b
```

配置后重启服务；删除某个值即可撤销对应调用方的访问。环境变量和 bootstrap Key 不会显示在网页的 Key 列表中。默认每个 Key 每分钟最多 60 次请求，可通过 `PUBLIC_API_RATE_LIMIT_PER_MINUTE` 调整。

## 健康检查

```bash
curl https://voice.luyang.icu/v1/health \
  -H "Authorization: Bearer <API_KEY>"
```

## 文本初筛会话

先创建会话：

```bash
curl -X POST https://voice.luyang.icu/v1/screening/sessions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "context_variant": "baseline",
    "patient": {
      "name": "张三",
      "age": 70,
      "gender": "男",
      "education_years": 9
    }
  }'
```

响应中的 `session_id` 用于后续每一轮对话：

```bash
curl -X POST "https://voice.luyang.icu/v1/screening/sessions/<session_id>/turn" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"message":"我叫张三，今年七十岁。", "emotion":"neutral"}'
```

响应含 `response`（下一句回复）、`task_id`、`dimension_id`、`context_variant`、`context_diagnostics` 以及视觉任务指令。会话状态保存在服务内存中，默认 1 小时未调用即过期；服务重启后需要新建会话。

## 切换上下文管理方案

不传 `context_variant` 时使用 `baseline`，也就是项目当前内置的上下文逻辑。管理员可以在部署环境中配置一个或多个隔离的候选上下文服务；调用方先查询可用版本：

```bash
curl https://voice.luyang.icu/v1/context/variants \
  -H "Authorization: Bearer <API_KEY>"
```

创建会话时选定版本；一个会话创建后会始终固定使用该版本，不能在中途切换：

```bash
curl -X POST https://voice.luyang.icu/v1/screening/sessions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "context_variant": "candidate-v1",
    "patient": {"age": 70, "education_years": 9}
  }'
```

也可以使用请求头 `X-Context-Variant: candidate-v1`；请求体字段优先。未知版本会返回 `400 UNKNOWN_CONTEXT_VARIANT`，调用方不能提交任意服务地址。

`context_diagnostics` 会给出候选服务的耗时、输出消息数、版本和是否发生降级。候选服务超过配置的超时时间（默认 1.5 秒）或返回不合法数据时，本轮自动使用 `baseline`；连续失败 3 次后会短暂熔断，以免影响对话延迟。

### 候选服务协议

主项目向管理员预先配置的候选服务发送 `POST` JSON：

```json
{
  "schema_version": "1.0",
  "context_variant": "candidate-v1",
  "session_id": "api_xxx",
  "patient": {"age": 70},
  "messages": [{"role": "assistant", "content": "您好"}],
  "agent_state": {"task_done": [], "memory_words": null},
  "task_context": {"candidates": ["orientation_time_year"]},
  "limits": {"max_recent_messages": 12, "max_summary_characters": 4000}
}
```

候选服务返回：

```json
{
  "summary": "患者70岁，愿意配合筛查。",
  "recent_messages": [{"role": "assistant", "content": "您好"}],
  "facts": [{"kind": "age", "value": 70}],
  "discussed_topics": ["基本情况(70岁)"],
  "asked_questions": ["您今年多大年纪？"],
  "next_task_suggestion": {
    "task_id": "orientation_time_year",
    "bridge_hint": "从年龄自然过渡到今年年份"
  },
  "estimated_tokens": 120,
  "version": "candidate-2026-07-14"
}
```

其中 `summary` 和 `recent_messages` 是必需的上下文主体；其余字段可选。候选服务应保持无状态：每次只根据请求中的完整可见历史生成结果。接口约束以本节请求和响应示例为准。

## 视觉评估

调用方提交 base64 编码的图片、视频或抽帧。以下以图片为例：

```bash
curl -X POST https://voice.luyang.icu/v1/vision/evaluate \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "copy_pentagons",
    "image": "<base64_IMAGE>",
    "context": "患者临摹两个相交五边形"
  }'
```

支持的任务应与项目的视觉评估配置保持一致，例如 `copy_pentagons`、`language_writing_sentence` 或 `language_3step_action`。

## 注意事项

- 这是认知初筛辅助能力，不应作为诊断结论。
- API 会触发 ASR、视觉模型、LLM 或 TTS 等上游服务，需对调用方设置配额与计费规则。
- 不要把内部的 `ARK_API_KEY`、火山引擎 Token 或 Cloudflare Tunnel Token 提供给调用方。
