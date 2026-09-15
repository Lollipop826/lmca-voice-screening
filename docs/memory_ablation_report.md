# 长期记忆模块消融实验报告

## 摘要

本实验比较了同一患者、同一语音输入在长期记忆开启和关闭两种条件下的表现。实验严格冻结评测阶段的记忆写入，只改变长期记忆检索开关；情绪模块关闭，以隔离记忆模块的影响。

本次获得 6 对有效配对样本，共 12 次语音调用。两组全部使用 SoulX 轮次判断和最终 ASR，12/12 次均为单轮，避免了多轮切分对延迟和回答质量的污染。记忆开启组 6/6 次命中预先写入的长期记忆，关闭组 0/6 次命中。

主要结论：

- 长期记忆几乎不影响 SoulX 首个增量文字和 ASR 阶段，SoulX 首字延迟中位数只增加约 8.8 ms。
- 记忆开启后，LLM 评审认为回答的个性化、相关性、连贯性和帮助性均有提升。
- 记忆开启组的首音频感知延迟中位数增加约 362.5 ms，主要出现在记忆检索后的 Agent/LLM 和 TTS 阶段，而不是 ASR。
- 该结果是小样本、合成语音条件下的初步验证，不能替代更大规模的人工语音实验。

## 1. 实验目的

验证长期记忆模块对以下两类指标的影响：

1. **响应延迟**：记忆检索是否拖慢 SoulX、ASR、LLM、TTS 和用户感知首音频。
2. **回答质量**：记忆是否使回答更相关、更个性化、更连贯和更有帮助，同时保持事实性、克制性和安全性。

本实验不评估情绪模块，也不把全双工打断行为混入记忆结论。

## 2. 实验配置

- 实验日期：2026-09-11
- 运行服务：`https://127.0.0.1:8427`
- 轮次判断和最终 ASR：SoulX Paraformer，记录字段为 `asr_source=soulx_final`
- 记忆数据库：运行服务使用的本地 SQLite 数据库
- 测试患者：`pt-memory-ablation-v1`
- 评测场景：家庭联系冲突、睡眠/工作消息、社区散步
- 语音输入：单声道 PCM16、16 kHz WAV；由项目 ArkTTS 生成的合成语音
- 情绪：关闭（`--emotion off`）
- 记忆写入：关闭；评测阶段不允许新记忆污染另一组
- 每个条件：6 次有效记录
- 配对方式：同一 `sample_id`、同一音频哈希、同一患者，比较 `memory_off` 与 `memory_on`

实验前通过 fixture 向隔离患者写入 4 条长期记忆：女儿联系冲突、周三与邻居散步、午夜后工作导致睡眠问题、睡前查看工作消息会影响入睡。

## 3. 有效性检查

<table>
  <thead>
    <tr><th>检查项</th><th>结果</th><th>判定</th></tr>
  </thead>
  <tbody>
    <tr><td>总记录</td><td>12</td><td>通过</td></tr>
    <tr><td>有效单轮记录</td><td>12/12</td><td>通过</td></tr>
    <tr><td>失败记录</td><td>0</td><td>通过</td></tr>
    <tr><td>SoulX 多轮切分</td><td>0 次</td><td>通过</td></tr>
    <tr><td>记忆开启检索命中</td><td>6/6</td><td>通过</td></tr>
    <tr><td>记忆关闭检索命中</td><td>0/6</td><td>符合消融预期</td></tr>
    <tr><td>评测阶段记忆写入</td><td>两组均关闭</td><td>通过</td></tr>
    <tr><td>ASR 归一化 CER</td><td>两组均为 0.0</td><td>通过</td></tr>
  </tbody>
</table>

这里的 CER=0.0 只表示本次合成测试语音与参考文本在脚本归一化口径下完全一致，不代表真实用户语音上的普遍 ASR 准确率。

## 4. 延迟结果

以下数值来自 6 对有效样本的中位数（p50）。所有差值均为“记忆开启 − 记忆关闭”。

<table>
  <thead>
    <tr><th>阶段</th><th>记忆关闭 p50</th><th>记忆开启 p50</th><th>开启−关闭</th></tr>
  </thead>
  <tbody>
    <tr><td>SoulX 首个增量文字</td><td>616.8 ms</td><td>625.4 ms</td><td>+8.8 ms</td></tr>
    <tr><td>轮次判断（末帧→vad_end）</td><td>878.0 ms</td><td>913.8 ms</td><td>+121.0 ms</td></tr>
    <tr><td>VAD 后 ASR/后处理</td><td>54.6 ms</td><td>51.8 ms</td><td>+1.5 ms</td></tr>
    <tr><td>Agent/LLM 流程</td><td>1549.9 ms</td><td>1656.8 ms</td><td>+104.8 ms</td></tr>
    <tr><td>TTS 首包</td><td>1536.2 ms</td><td>1751.1 ms</td><td>+205.0 ms</td></tr>
    <tr><td>用户感知首音频</td><td>4065.1 ms</td><td>4407.0 ms</td><td>+362.5 ms</td></tr>
    <tr><td>完整响应</td><td>7982.1 ms</td><td>7748.1 ms</td><td>−54.3 ms</td></tr>
  </tbody>
</table>

表中前两列是各组独立的 p50，第三列是逐对配对差值的 p50。两者口径不同，因此
第三列不等于前两列相减：`51.8 − 54.6 = −2.8 ms` 反映的是两组中位数之差，而
配对差值的 p50 是 `+1.5 ms`（均值 +0.6 ms）。消融应以配对差值为准，本报告
第三列统一取自 `paired_deltas`。VAD 后处理这一阶段的配对差值方向为轻微变慢，
但 1.5 ms 落在测量噪声内，不构成实质开销。

完整响应虽然在记忆开启组低 54.3 ms，但该指标受回答长度和 TTS 合成时长影响较大，不应解释为记忆模块降低了端到端延迟。

延迟结论是：记忆模块没有明显拖慢 SoulX 或 ASR；可观察到的开销主要位于记忆命中后的 Agent/LLM 处理和 TTS 首包阶段。由于样本量小且 TTS 首包存在波动，362.5 ms 应视为本次运行的估计值，而不是稳定的固定开销。

## 5. 回答质量评测

### 5.1 评测方法

回答质量采用 LLM-only 盲评，不进行人工评分。评审输入中隐藏 `memory_on`/`memory_off` 标签，并随机交换 A/B 展示位置。

- 评审器数量：3 个独立 LLM judge（`llm_judge_1`--`llm_judge_3`）
- 评测案例：6 个配对案例
- 完整评分：18 份（6 案例 × 3 评审器）
- 评分范围：1--5 分
- 评测维度：相关性、个性化、连贯性、事实准确性、克制性、帮助性、适切性、安全性
- judge 模型：`qwen-flash`，temperature 0.0，启用 A/B 位置随机交换
- judge 接口：OpenAI 兼容协议，base URL 为
  `https://dashscope.aliyuncs.com/compatible-mode/v1`（阿里云 DashScope）
- 盲评表随机种子 20260911，judge 调用种子 20260913

以上 judge 参数取自 `memory_cross_session_compact_quality_llm_ratings.json`
的 `judge_config` 字段，是本次运行的权威记录。

需要说明接口与 key 变量名的对应关系，否则复现时容易配错：
`evaluate_response_quality.py` 的 `--base-url` 默认值为
`os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"`，
而 `--api-key-env` 默认值固定为 `SILICONFLOW_API_KEY`。本次 `judge_config`
记录的 base URL 是 DashScope，说明运行时 `SILICONFLOW_BASE_URL` 被指向了
DashScope 兼容端点，因此 key 实际从 `SILICONFLOW_API_KEY` 读取，但请求发往
DashScope。复现时必须同时核对这两个变量，只看命令行中的 `--api-key-env`
会误判服务商。

### 5.2 记忆检索行为

以下字段来自 benchmark 记录，用于核对消融开关确实生效（每组 6 条）：

<table>
  <thead>
    <tr><th>字段</th><th>记忆关闭</th><th>记忆开启</th></tr>
  </thead>
  <tbody>
    <tr><td><code>memory_retrieval_source</code></td><td>全部 <code>disabled</code></td><td>全部 <code>realtime</code></td></tr>
    <tr><td><code>memory_retrieval_hit</code></td><td>0/6</td><td>6/6</td></tr>
    <tr><td>检索耗时 p50</td><td>0.3 ms</td><td>1.05 ms</td></tr>
    <tr><td>注入上下文字符数 p50</td><td>150</td><td>361</td></tr>
    <tr><td><code>memory_used_items</code></td><td>全部 0</td><td>全部 0</td></tr>
  </tbody>
</table>

`memory_retrieval_source` 全为 `disabled` / `realtime`，证明关闭组走的不是
本地兜底路径（`local_fallback`），满足协议对严格消融的要求。检索耗时本身在
1 ms 量级，说明第 4 节观察到的约 362.5 ms 感知开销不来自检索调用，而来自
更长 prompt 导致的下游生成与 TTS 变化：记忆开启组注入上下文从 150 字符增加
到 361 字符。

`memory_used_items` 在两组均为 0，与开启组 6/6 命中且上下文字符数翻倍相矛盾。
这是埋点缺陷，不是"未使用记忆"的证据：命中判定和上下文注入都已生效，但计数
字段没有被写入。协议第 4 节要求报告"使用的 item 数"，该指标目前无法从产物中
取得，需要先修脚本埋点。本报告不给出 item 数。

### 5.3 主要评分

<table>
  <thead>
    <tr><th>质量维度</th><th>记忆关闭</th><th>记忆开启</th><th>提升</th></tr>
  </thead>
  <tbody>
    <tr><td>相关性</td><td>4.28</td><td>4.89</td><td>+0.61</td></tr>
    <tr><td>个性化</td><td>3.56</td><td>4.61</td><td>+1.06</td></tr>
    <tr><td>连贯性</td><td>4.44</td><td>4.89</td><td>+0.44</td></tr>
    <tr><td>帮助性</td><td>4.22</td><td>4.78</td><td>+0.56</td></tr>
    <tr><td>事实准确性</td><td>4.67</td><td>5.00</td><td>+0.33</td></tr>
    <tr><td>克制性</td><td>4.89</td><td>5.00</td><td>+0.11</td></tr>
    <tr><td>适切性</td><td>4.72</td><td>4.94</td><td>+0.22</td></tr>
    <tr><td>安全性</td><td>5.00</td><td>5.00</td><td>0.00</td></tr>
  </tbody>
</table>

总体偏好统计中，评审器偏好记忆开启 12 次、关闭 2 次、平局 4 次。

### 5.4 评审器一致性

3 个评审器在同一案例上的两两一致率如下（18 个评审对）：

<table>
  <thead>
    <tr><th>质量维度</th><th>完全一致</th><th>误差 1 分内一致</th></tr>
  </thead>
  <tbody>
    <tr><td>相关性</td><td>0.611</td><td>0.944</td></tr>
    <tr><td>个性化</td><td>0.500</td><td>0.889</td></tr>
    <tr><td>连贯性</td><td>0.389</td><td>0.944</td></tr>
    <tr><td>事实准确性</td><td>0.778</td><td>0.778</td></tr>
    <tr><td>克制性</td><td>0.889</td><td>1.000</td></tr>
    <tr><td>帮助性</td><td>0.389</td><td>0.944</td></tr>
    <tr><td>适切性</td><td>0.667</td><td>1.000</td></tr>
    <tr><td>安全性</td><td>1.000</td><td>1.000</td></tr>
  </tbody>
</table>

连贯性和帮助性的完全一致率最低（0.389），说明这两个维度的绝对分数存在评审
分歧；但两者误差 1 分内的一致率都是 0.944，分歧幅度有限。安全性完全一致率
1.000 是因为全部评分都是 5 分，该维度在本批样本上没有区分度，不应解读为
评审器在安全性上判断力更强。

一致性数据取自 `memory_cross_session_compact_quality_result.json` 的
`inter_rater_agreement` 字段。

### 5.5 定性观察

记忆关闭时，回答通常只能复述当前情绪，例如“孩子不回消息，这种悬着的心确实让人着急”。

记忆开启后，回答能够引用与当前输入对应的历史事实，例如：

- “之前你也提过愿意这么做。”
- “特别是周三和邻居一起走。”
- “上次因为联系频率吵过，这次沉默可能跟那有关。”

这些例子说明记忆检索不仅改变了内部状态，而且确实影响了最终文字回答的上下文连续性和个性化程度。

## 6. 结果解释

本实验支持以下判断：

1. **记忆模块的主要收益在回答质量，而不是 ASR 延迟。** SoulX 首字和 ASR 后处理几乎不变，说明记忆不会把前端语音识别变慢。
2. **记忆模块的成本出现在生成链路。** 记忆开启后 Agent/LLM 和 TTS 首包变慢，用户感知首音频中位数增加约 0.36 秒。
3. **记忆对个性化的改善最明显。** 个性化平均分提升 1.06，是本次所有质量维度中最大的提升。
4. **没有观察到安全性或克制性下降。** 两组安全性均为 5.00，克制性略有提升；但这仍需要更大样本验证。

**上述提升在本轮样本量下均未达到统计显著。** LLM 评审脚本对每个维度计算精确符号翻转检验，检验单位是原始音频样本（`sample_id`）。本轮 6 对配对来自 3 个场景各重复 2 次，因此独立样本只有 3 个，检验结果为：相关性、个性化、连贯性、事实性、帮助性、适当性均为 `p = 0.5`，克制性 `p = 1.0`，安全性因两组评分完全相同无法检验。3 个样本在符号翻转检验下的最小可能双侧 p 值就是 0.25，所以这个设计本身无法产生显著结果，不能据此宣称记忆模块带来了可确认的质量提升。

需要特别说明两点，避免误引用：

1. 产物文件 `output/memory_cross_session_compact_quality_result.json` 由旧版分析脚本生成，其中 `exact_sign_flip_p` 记为个性化 0.0117、相关性 0.0322、帮助性 0.0508 等。这些数值把「同一案例被 3 个评审器重复评分」当成了独立观测，是伪重复导致的虚低 p 值，**不应引用**。当前脚本已改为先在案例内对评审器取均值、再按 `sample_id` 聚合。
2. 报告中的 `n_cases_tested = 6` 指盲评案例数，不是独立样本数。当前脚本同时输出 `distinct_samples`（本轮为 3）与 `n_samples_tested`（本轮为 3），并保留按案例聚合的 `exact_sign_flip_p_by_case` 作为诊断参考——后者更宽松，正是因为它仍把重复运行算作新增证据。

因此第 3 条关于个性化的判断应理解为效应量方向的观察（个性化配对差值均值 +1.06，但 3 个样本的逐样本差值中有一个为 −0.5，方向不一致），而不是经过检验的结论。

## 7. 局限性

- 样本量只有 6 对，统计功效有限。
- 语音由 ArkTTS 合成，不是人工录音；尚未覆盖口音、停顿、噪声和自然语速变化。
- 三个场景各自重复运行，场景覆盖仍然有限。
- 本轮情绪模块关闭，因此不能推断“记忆 + Emotion2Vec”共同开启时的最终表现。
- 无关控制样本没有纳入正式质量得分：该控制运行发生 SoulX 多轮切分，无法与严格单轮样本公平比较。
- TTS 首包和完整响应存在运行波动，不能仅凭本轮结果断言记忆固定增加 205 ms 的 TTS 延迟。
- LLM judge 不是人工金标准，评分可能受到模型偏好和提示词的影响。

## 8. 后续实验建议

1. 使用人工录音重新跑至少 20--30 对样本，并保持同一音频哈希的 A/B 配对。
2. 将家庭、睡眠、健康、日常活动和无关控制扩展到各 5--10 个独立案例。
3. 单独记录记忆检索耗时、LLM 首句耗时和 TTS 提供商首包耗时，定位本轮约 0.36 秒用户感知开销。
4. 在记忆开启条件下增加 Emotion2Vec，再做四组实验：记忆关/情绪关、记忆开/情绪关、记忆关/情绪开、记忆开/情绪开。
5. 保留 LLM-only 评分作为自动回归指标，同时抽取少量样本进行人工复核，检查 LLM judge 是否偏爱表面上的“引用记忆”。

## 9. 可复现实验产物

- 延迟与配对结果：`output/ablation_memory_cross_session_compact_6pairs.json`
- LLM 盲评输入：`output/memory_cross_session_compact_quality_ratings.json`
- LLM 盲评原始评分：`output/memory_cross_session_compact_quality_llm_ratings.json`
- 质量分析结果：`output/memory_cross_session_compact_quality_result.json`
- 记忆 fixture 与写入记录：`output/memory_ablation_fixture.json`
- 实验协议：`docs/memory_ablation_protocol.md`

本报告基于上述文件生成。相关测试于 2026-09-13 复核：
`tests/test_evaluate_response_quality.py` 9 passed、
`tests/test_prepare_memory_ablation_fixture.py` 4 passed、
`tests/voice/test_benchmark_voice_realtime.py` 8 passed，合计 21 passed。

注意 pytest 目前只安装在 `/home/student1/miniforge3/envs/lmca`，
`/data/luyang/envs/lmca` 可以运行本协议的三个脚本但不能跑测试。
