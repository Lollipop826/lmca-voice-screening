# 跨会话长期记忆消融协议

## 为什么不能只测一轮

一轮对话只能验证“检索开关是否生效”和检索耗时，不能验证长期记忆是否让回复更个性化。当前协议把实验拆成两个阶段：

1. **准备阶段**：通过真实 `EmotionMemobase` 写入一组明确、可核对的长期事实。
2. **评测阶段**：结束准备阶段后，使用同一 `patient_id` 开启全新的语音会话；A/B 两组使用相同音频、相同患者和相同实验顺序，唯一差异是长期记忆检索是否开启。

评测阶段两组都关闭长期记忆写入，避免 A 组先写入内容污染 B 组。

## 1. 准备隔离患者和记忆

fixture 示例在 `tests/fixtures/memory_ablation_cases.json`。它同时记录：

- 患者和要写入的事实、事件、偏好；
- 每个评测样本对应的人工参考文本；
- 该样本应该检验的记忆事实和场景说明。

先做 dry-run：

```bash
/data/luyang/envs/lmca/bin/python \
  scripts/prepare_memory_ablation_fixture.py \
  --fixture tests/fixtures/memory_ablation_cases.json \
  --db /path/to/the/running/service/voice_server.db \
  --output output/memory_ablation_fixture.json
```

确认 `--db` 与正在运行的语音服务使用的是同一个数据库后，再加 `--apply`。脚本只接受一个全新的 `patient_id`，不会覆盖已有患者：

```bash
/data/luyang/envs/lmca/bin/python \
  scripts/prepare_memory_ablation_fixture.py \
  --fixture tests/fixtures/memory_ablation_cases.json \
  --db /path/to/the/running/service/voice_server.db \
  --apply \
  --output output/memory_ablation_fixture.json
```

输出会记录写入的 `memory_item_ids`，这是后面核对“确实写入了目标记忆”的证据。

### 前提：Memobase 必须可用

准备脚本写入的是 SQLite `memory_items`，而评测阶段的检索走 Memobase 的
event gist 语义搜索。两者不是同一条路径，因此 Memobase 不可用时：

- `MEMORY_LOCAL_FALLBACK_ENABLED=true` 会让检索转向本地兜底，按 `updated_at`
  倒序倒出最近 3 条 active 记忆，**完全不看当前问题**；
- 这条兜底路径产出的上下文带 `[local_active_memory]` 标记，
  `retrieval_kind=local_fallback`，benchmark 会拒绝这类样本。

开跑前先确认语义检索真的在工作：

```bash
curl -ksS https://127.0.0.1:8427/health | python -m json.tool | grep -i memobase
```

如果 `MEMOBASE_PROJECT_URL` 为空或服务未启动，测到的只是“有没有在
prompt 里贴记忆卡”，不是“语义检索是否有用”。

## 2. 准备语音样本清单

为每个 `sample_id` 准备一段单声道 PCM16、16 kHz WAV。音频清单格式：

```json
{
  "samples": [
    {
      "sample_id": "memory-family-conflict-followup",
      "audio_path": "audio/family_conflict_followup.wav",
      "reference_text": "我这两天还是因为女儿不回消息觉得难受。"
    },
    {
      "sample_id": "memory-sleep-followup",
      "audio_path": "audio/sleep_followup.wav",
      "reference_text": "最近又睡不好，晚上总忍不住看工作消息。"
    }
  ]
}
```

`sample_id` 必须与 fixture 中的 `cases[].sample_id` 一致。建议每个相关场景至少 5 对有效样本；同一个场景的 A/B 必须使用同一段音频。

相关样本和“无关控制”建议分成两份 manifest。相关样本用于严格的记忆命中消融；无关控制用于检查 `memory_on` 是否会强行提及旧事，不能和“必须命中记忆”的样本混在同一份正式统计里。

## 3. 运行严格 A/B 消融

准备阶段结束后，退出准备会话；评测命令每一轮都会用 `force_new_session=true` 开启新会话，因此不会把上一轮聊天历史当作短期记忆：

```bash
/data/luyang/envs/lmca/bin/python \
  scripts/benchmark_voice_realtime.py \
  --server https://127.0.0.1:8427 \
  --username <user> --password <password> \
  --patient-id pt-memory-ablation-v1 \
  --audio-manifest output/memory_ablation_audio_manifest.json \
  --ablation memory \
  --runs 20 --warmup-runs 3 \
  --require-memory-hit \
  --insecure \
  --output output/ablation_memory_cross_session_20pairs.json
```

相关样本的脚本必须同时满足以下条件才把一对样本计入正式结果：

- A/B 都是 `asr_source=soulx_final` 且单轮；
- `memory_on` 的 `memory_retrieval_hit=true`；
- `memory_off` 的检索来源为 `disabled`；
- 两组的 `memory_writes_enabled=false`；
- 两组 `sample_id` 和音频哈希相同。

对无关控制 manifest，使用 `--no-require-memory-hit` 单独运行；这组结果只用于评估“无关时是否克制”，不应与相关样本合并计算记忆增益。

## 4. 进行回答质量盲评

把准备脚本的输出直接作为评审上下文，评审器能看到“本案例的参考记忆事实”，但看不到 A/B 哪一组开启了记忆：

```bash
/data/luyang/envs/lmca/bin/python \
  scripts/evaluate_response_quality.py prepare \
  --input output/ablation_memory_cross_session_20pairs.json \
  --context output/memory_ablation_fixture.json \
  --output output/memory_cross_session_quality_ratings.json \
  --key-output output/memory_cross_session_quality_key.json \
  --seed 20260911

/data/luyang/envs/lmca/bin/python \
  scripts/evaluate_response_quality.py judge \
  --ratings output/memory_cross_session_quality_ratings.json \
  --output output/memory_cross_session_quality_llm_ratings.json \
  --judge-raters llm_judge_1,llm_judge_2,llm_judge_3 \
  --api-key-env SILICONFLOW_API_KEY

/data/luyang/envs/lmca/bin/python \
  scripts/evaluate_response_quality.py analyze \
  --ratings output/memory_cross_session_quality_llm_ratings.json \
  --key output/memory_cross_session_quality_key.json \
  --output output/memory_cross_session_quality_result.json
```

质量指标包括相关性、个性化、连续性、事实准确性、克制性、帮助性、适切性和安全性。报告中同时给出：

- 记忆命中率和使用的 item 数；
- `memory_on - memory_off` 的各维度配对差值；
- 记忆无关控制样本中是否出现强行引用旧事；
- LLM judge 模型、温度和重复评审次数。

这样测到的才是“跨会话长期记忆对回答质量的贡献”，而不只是一次空检索的延迟差异。
