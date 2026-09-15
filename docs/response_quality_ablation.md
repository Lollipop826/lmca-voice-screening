# 回答质量消融评测

`benchmark_voice_realtime.py` 负责测延迟；
`evaluate_response_quality.py` 负责把已经完成的 A/B 回复变成盲评表，调用 LLM 盲评，再做质量统计。
两者分开，避免评审被延迟数字或 `memory_on/off` 标签影响。

## 1. 生成盲评表

先使用严格的、没有降级的 SoulX benchmark 结果：

```bash
/data/luyang/envs/lmca/bin/python scripts/evaluate_response_quality.py prepare \
  --input output/ablation_memory_strict_5pairs.json \
  --output output/memory_quality_ratings.json \
  --key-output output/memory_quality_key.json \
  --seed 20260913
```

`memory_quality_key.json` 只用于最终统计，LLM 评审阶段不会读取它。

可选的场景文件可以按 `case_id` 或 `sample_id` 提供参考事实，例如：

```json
{
  "cases": {
    "user_20260910_211156_428172_9e47e4": {
      "memory_facts": ["患者上周因和女儿争执而难过"],
      "scenario": "测试与既有事件相关的问题"
    }
  }
}
```

## 2. LLM 自动评分

下面命令使用 SiliconFlow 的 OpenAI-compatible 接口。API key 只从环境变量读取，不会写入结果文件；如果项目实际使用其他兼容接口，可替换 `--base-url`、`--model` 和 `--api-key-env`。

```bash
/data/luyang/envs/lmca/bin/python scripts/evaluate_response_quality.py judge \
  --ratings output/memory_quality_ratings.json \
  --output output/memory_quality_llm_ratings.json \
  --judge-raters llm_judge_1,llm_judge_2,llm_judge_3 \
  --api-key-env SILICONFLOW_API_KEY
```

每个 judge 都是一次独立的盲评请求；脚本会随机交换 A/B 展示位置，再映射回原盲评表，降低位置偏差。这里不需要、也不使用人工评分。

## 3. 评分维度

每个维度使用 1--5 分，A/B 的标签是随机打乱的。记忆消融默认评估：

- 相关性、个性化、连续性、事实准确性、克制性；
- 帮助性、适切性、安全性。

情绪消融会换成情绪匹配、共情、适度性、帮助性、适切性和安全性。

## 4. 分析

```bash
/data/luyang/envs/lmca/bin/python scripts/evaluate_response_quality.py analyze \
  --ratings output/memory_quality_llm_ratings.json \
  --key output/memory_quality_key.json \
  --output output/memory_quality_llm_result.json
```

结果包含每个维度的开启/关闭均值、中位数、配对差值、精确符号翻转检验，以及总体偏好。该结果是 LLM-only 评审，不混入人工分数；论文或报告中应注明模型、接口和 judge 次数，并把它视为模型评审结果而非人工金标准。
