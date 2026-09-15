# 真实语音实验与消融结果核验报告（历史证据补充修订）

日期：2026-09-13（Asia/Shanghai）
修订范围：历史原始运行记录只读追溯 + 本次隔离服务预试 + 本地组件补测。**历史已跑通不等于本轮 48 轮正式实验已完成；各批次独立标注，不混合统计。**

## 1. 结论先行

补查确认：**项目此前已经有真实语音实验、历史四组消融记录，以及火山 ASR/TTS 成功证据；不能把本次预试失败解释成“项目从未跑通过”。** 本报告现将历史结果与本轮新试跑分开呈现。

- 历史火山真流式 ASR：5 批共 **50/50** 次成功，逐条有非空最终转写；历史 v2 四组各 10 条，共 **40 条**，两组情绪开启的 20/20 次均确认真实音频模型参与。
- 历史记忆专项另有 **20 对/40 条**和**6 对/12 条**有效单轮记录；前者情绪为文本回退，后者情绪关闭，不能与四组 v2 的真实音频情绪证据混淆。详见第 3 节。
- 今天上午仍有真实 ASR 与非空 TTS 音频成功证据；18:41 的独立探测返回资源未授权。**未发现足以认定“有人手动关掉火山”的证据，也不能确定连续故障的起止时间。**

**以下数字只属于本轮 9 月 13 日隔离预试和本地补测，不含上述历史记录：**

- 云端端到端预试执行 **3 个用户音频输入尝试，完整成功 0 个**；其中 1 个拿到 ASR 转写和一段 LLM 回复文本，0 个拿到该回复的真实 TTS 音频。连续 3 次失败后停止。
- 预先冻结的正式设计是 4 场景 × 4 组 × 3 次 = **48 个正式轮次，实际执行 0 个**。不要把计划样本数写成已完成样本数。
- 随后独立完成 **20 次真实 Emotion2Vec 多模态调用**，全部确认用了语音模型、无文本回退；另做 **40 次真实 SQLite 会话记忆加载检查**，开/关各 20 次。
- 发现影响对照有效性的真实问题：**长期记忆写入关闭不等于用户档案被冻结**。本次启动请求把已预置档案中的年龄/受教育年限改成默认值。

## 2. 实际环境与隔离边界

本节仅描述本轮隔离预试，不倒推历史实验的运行环境。使用 `/data/luyang/lmca-share` 在预试启动时的代码（指纹见 `preflight_plan.json`）、Python 3.11.16，测试服务仅绑定 `127.0.0.1:18427`。独立 SQLite 库，单一虚构用户，输入为已有的四条合成 WAV（单声道、16 kHz、16 bit，约 3.41–5.00 秒），没有使用真实患者录音。

实际模型：火山 BigASR；云 LLM `qwen3.7-flash`（温度 0.65，max_tokens 240）；ArkTTS（配置资源 `seed-tts-2.0`）；Emotion2Vec+ large，RTX 3090。

原服务端口 8427/8502、SoulX 8001、Memobase 18019 在本轮预试前检查时均未监听。隔离服务禁用了 SoulX、声纹和 Memobase 语义检索；长期记忆使用真实 SQLite 权威记忆卡读库路径。**记忆卡加载不是语义检索，不能写成“Memobase 检索命中”。**

未修改 `.env`，未重启生产服务，未成功安装任何依赖。模型缓存通过不包含 `requirements.txt` 的缓存符号链接视图加载，避免 FunASR 自动安装依赖。实验后已停止自己启动的隔离服务。[最初交付时保存的清理快照](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/cleanup_at_original_delivery.json)显示，当时 `.env` 与预记录的 7 个生产源码均匹配测试前哈希。**本次报告修订重新检查时**，`.env` 和其中 6 个源码仍匹配，`/data/luyang/lmca-share/src/context_management/emotion_memobase.py` 已不同于预试快照；本次修订没有编辑该业务源码，也不据此推断修改人或改动原因。当前比较与实验结束时的比较分别留存，不继续笼统声称“所有源码始终未变”。

## 3. 补充复核：历史真实运行与消融记录

本节是**2026-09-13 对已有产物的只读追溯**，没有重新调用云接口或重放历史录音。历史成功、本轮失败预试、本地组件微基准是三个不同证据层，不能合并样本数或替换本轮正式实验数据。

### 3.1 历史记录索引与证据强度

| 历史实验 | 已核验记录 | 能支持的结论与限制 | 原始来源 |
|---|---|---|---|
| 火山真流式 ASR 基准 | 5 批 × 10 次，共 **50/50 成功且最终转写非空** | 支持当时真实 ASR 调用可用；不是 ASR 准确率或 50 个独立语音样本 | [一批原始结果](/data/luyang/lmca-share/output/benchmark_ark_asr_streaming_10runs4.json)；全部 5 批列于 [追溯索引](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/summary.json) |
| 历史四组语音消融（v2） | 每组 10 条，共 **40 条**；各文件 `live=true`、`failures=[]` | 有逐轮 ASR/AI/TTS 事件计时；两组 E 开启共 20/20 确认 `audio_model_used=true`；不等于本轮协议重跑成功 | [Full](/data/luyang/lmca-share/output/e2e_v2_baseline.json) · [去记忆](/data/luyang/lmca-share/output/e2e_v2_no_memory.json) · [去情绪](/data/luyang/lmca-share/output/e2e_v2_no_emotion.json) · [双关闭](/data/luyang/lmca-share/output/e2e_v2_no_emotion_no_memory.json) |
| 9月11日记忆开关配对 | **20 对、40 条**有效单轮，失败 0；只有 1 条音频反复测量 | 记忆开启 20/20 标记命中、关闭 0/20；情绪 40/40 为 `text_fallback_model_unavailable`，不是音频情绪消融 | [20 对原始结果](/data/luyang/lmca-share/output/ablation_memory_formal_20pairs.json) |
| 9月11日跨会话记忆配对 | **6 对、12 条**有效单轮，失败 0；3 个场景各重复 2 次 | 开启 6/6 标记命中、关闭 0/6；情绪明确关闭，ASR 来源均为 `soulx_final`；不能归作火山 ASR 成功计数 | [跨会话原始结果](/data/luyang/lmca-share/output/ablation_memory_cross_session_compact_6pairs.json) · [对应报告](/data/luyang/lmca-share/docs/memory_ablation_report.md) |
| 9月11日22:38实时语音回归 | **3/3**有效单轮，失败 0；音频情绪 3/3 真正参与 | 支持实时链路与 Emotion2Vec 的历史运行；这不是配对消融，也不能仅凭链路通过宣称完整记忆效果 | [实时回归结果](/data/luyang/lmca-share/output/rt_continuous_20260911.json) |

前两类文件的保存时间为 9 月 10 日；这些 JSON 没有独立的实验创建时间字段，因此不把文件 mtime 当作精确运行时间。后三类日期由原始 `created_at` 转换到 Asia/Shanghai。五批 ASR 文件与旧项目目录现存副本逐一 SHA256 一致，只计一份，不因跨目录副本重复计数。以上不同实验**不汇总为一个总成功率**。

### 3.2 历史四组 v2 的可复核延迟

以下均为各组 10 条逐轮记录**重新计算的算术均值，单位 ms**，已逐项对齐文件内 `summary`，不混用旧版 `e2e_full.json` 或其他版本基线。

| 历史组别 | 记录数 | VAD_END→ASR结果 | ASR结果→首段AI文本 | 首段AI文本→首TTS音频事件 | 首段AI文本→TTS结束 |
|---|---:|---:|---:|---:|---:|
| 历史 Full（M1E1） | 10 | 4138.23 | 878.38 | 1052.00 | 5012.44 |
| 去记忆（M0E1） | 10 | 3910.15 | 1342.62 | 941.15 | 4370.49 |
| 去情绪（M1E0） | 10 | 3792.80 | 986.76 | 845.08 | 4047.46 |
| 双关闭（M0E0） | 10 | 3846.04 | 1410.97 | 1059.35 | 4674.36 |

口径说明与不能越界的结论：

1. 本组使用 `manual_audio` 一次提交整段 WAV，不是实时麦克风输入；不能将上表改名为“用户说完到听到回复”。ASR 列是 `VAD_END → ASR_RESULT`，**不是 AI 与 TTS 阶段相加**。
2. “ASR结果→首段AI文本”包含记忆与 agent 前处理，不是纯 LLM 推理；最后一列是首段文本到 TTS 结束的区间，包含后续文本/音频流的重叠，不能再与首 TTS 延迟相加。
3. 两个 E 开启组逐轮都为 `source=emotion2vec_audio+text`、`analysis_status=final`、`audio_model_used=true`；两个 E 关闭组均未使用音频模型。**这组确实有音频情绪开关证据，不能与 9 月 11 日那批文本回退记录混为一谈。**
4. 这 40 条均未在 `tts_audio` 字段保存音频载荷，保留的是事件计时。服务日志与持久化 WAV 证明项目历史上产生过真实音频，但不把别的会话 WAV 冒充为这 40 条逐轮录音的严格匹配证据。
5. 四组都是同一条 WAV 重复 10 次、没有暖机剔除（`warmup_runs=0`），且请求 `new_patient=true`。M 开启组同时开启长期记忆写入，M 关闭组同时关闭写入，读/写成本并未拆开；文件也不足以证明新用户均具有同等跨会话历史。因此它是**历史开关与延迟对照**，不是独立完成的记忆质量因果实验。
6. 上表只描述这批历史数据，未做可信的独立样本显著性推断，不据此宣称“记忆/情绪模块提高了回答质量”，也不计算跨批次提速比例。

### 3.3 火山可用性时间线与授权结论修正

- **9月12日晚至9月13日上午**：历史服务日志有火山 ASR 的 `status=success` 且 `text_chars>0`。最近一条非空结果位于 [日志第 20054 行](/data/luyang/lmca-share/tmp/voice_server.log#L20054)，请求标识以 `20260913110032` 开头，返回 32 个字符，资源为 `volc.seedasr.sauc.duration`。该时间来自请求标识，不把无独立时间戳的日志行当作精确完成时刻。
- **9月13日11:14左右**：TTS 日志记录“87块、6.6s、首块延迟1.24s”，见 [第 22870 行](/data/luyang/lmca-share/tmp/voice_server.log#L22870)；同一时段另有持久化 WAV，头部为 **157951 帧、24000 Hz、6.5813 秒**，文件创建元数据为 `2026-09-13T11:14:27.485301`。本次只读核查音频头部，未重放或复跑这段历史录音。日志与 WAV 是相互补充的可用性证据；日志没有给出该 WAV 文件名，因此不宣称已逐轮严格绑定。
- **本次傍晚预试**：首个用户输入仍获得真实 ASR 转写，随后 TTS 失败；另两个输入的 ASR 失败。不能由“上午成功、晚间失败”推断整个下午连续不可用。
- **9月13日18:41:34 / 18:41:36（Asia/Shanghai）**：独立 TTS / ASR 探测均收到 HTTP 403、`requested resource not granted`，见 [原始响应](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/cloud_connection_probe.json)。

据此修正为：**项目此前确有真实语音实验与成功调用，至少今天上午仍有成功证据；本次晚间探测所用资源访问被拒绝。现有证据不能确定故障的精确起点、持续性，也不能认定有人手动关闭服务。** 后者需要控制台资源状态、账务/授权信息与操作审计才能定位。

本次还只读比对了现项目与旧服务目录的现存 `.env`：火山 API key、access token、app ID 值相同，ASR 资源均为 `volc.seedasr.sauc.duration`，TTS 均为 `seed-tts-2.0`。报告只保存“是否一致”，不保存凭证值。这只能说明**当前两个配置文件一致**，不能还原历史进程的环境变量覆盖或排除云端授权变化。

### 3.4 历史结果怎样用于报告

可以引用第 3.2 节作为明确标注日期与口径的历史延迟结果；跨会话记忆专项可作为开关行为的补充证据。但质量收益应保留对应报告的限制：6 对只有 3 个独立场景，旧质量分析中把多评审重复打分当独立样本的虚低 p 值不可沿用；现有样本不能证明显著质量提升。没有可核验依据的“94.2%检索准确率”“35.3%倾听感提升”等旧数字不因发现历史日志而自动成立。

历史记录不能填入本轮的 48 个正式轮次，也不能用于掩盖本轮失败或第 5 节的档案漂移问题。新增 [脱敏追溯汇总](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/summary.json)、[源文件哈希索引](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/source_manifest.json)、[成功日志摘录](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/service_log_excerpts.json) 以便复核；证据包不附历史用户转写和录音。

## 4. 本轮云端预试的真实失败

### 4.1 已执行内容

第一轮 M1E1 的输入是“我这两天还是因为女儿不回消息觉得难受。”；真实 ASR 返回“我这两天还是因为女儿不回消息，觉得难受。”。服务随后确实完成 Emotion2Vec 音频推理（`audio_model_used=true`、`source=emotion2vec_audio+text`，本轮日志报告 187.7 ms），并收到 LLM 的一句“这种被悬在半空的感觉确实挺熬人的。”。

但 TTS 返回 `tts_end(reason=error, duration=0, chunks=0)`，没有输出音频。该轮是失败，不是完整回复成功。后两轮（M0E1、M1E0）在 ASR 阶段失败，M0E0 未运行。服务内部保留既有 ASR/TTS 重试逻辑；3 次是用户输入尝试数，不是底层请求总数。

### 4.2 直接探测到的错误体

9 月 13 日 18:41（Asia/Shanghai）另做一次 TTS、一次 ASR 连接探测，均被远端以 HTTP 403 拒绝：

- TTS：`[resource_id=volc.seedtts.default] requested resource not granted`
- ASR：`[resource_id=volc.seedasr.sauc.duration] requested resource not granted`

TTS 上述名称来自服务端错误体，并不表示本地配置成了该名称。可以确认的是**当时这两次请求的资源访问被拒绝**；不能仅凭这条报错断定是余额、套餐到期、凭证失效、具体哪项授权配置，更不能认定有人手动关闭。第 3 节的历史成功与此处晚间失败并不矛盾。启动健康检查曾成功，也不代表后续每个云请求仍然可用。

### 4.3 不输出虚假的延迟基线

失败轮次的接收时间戳均保留，但不据这些失败轮次计算或发布完整系统延迟均值、p95、模块收益百分比。第 3.2 节另列的是历史 v2 数据，不是本轮补成功样本。`manual_audio` 是一次性提交整段音频，绕过真实说话节奏/VAD/SoulX；即使成功，也只能叫“提交音频到收到首音频字节”，不能叫“用户说完到听到回复”。

另外，试跑发现当前服务对已有用户同样发送开场白，复用 helper 并未等待开场 TTS 结束。首轮 `input → vad_end` 已有约 3.24 秒排队，不能把它归因于 ASR。试跑均为暖机/排障记录，未纳入正式统计。

## 5. 新发现：关闭记忆写入仍会改档案

冻结的虚构档案是 **68 岁、受教育 9 年**。三个 `session_started` 原始事件都回显 **70 岁、6 年**，同时明确回显 `long_term_memory_writes_enabled=false`。测试前后权威记忆卡的文本差分也证实了这两个字段发生变化。

已定位的路径是：会话启动时先把缺省 profile 规范化成含默认值的非空字典；随后记忆服务不再按“空档案”恢复原档案，而是调用真实数据库档案更新。长期记忆写入门控并未覆盖这条档案更新路径。4 条历史记忆项本身未变化，但**完整提示上下文并非完全冻结**。

这也是原消融设计必须加强的地方：不能只核验开关或 memory_items 数量，还要校验 profile、完整记忆卡与会话上下文。该问题发生在本次合成测试库，没有改动真实用户档案。

本次只增强了新测试脚本：显式发送冻结档案、核验服务回显、等待非空开场音频正常结束后再计时、检测记忆/档案漂移并立即停止、保留失败记录。通过 **7 个脚本单元测试**。上述脚本增强没有编辑生产业务源码；截至最近一次 18:41 探测仍未通过授权，之后未重新探测；**增强后的云端跑法尚未完成真实重跑**，不能把软件单元测试当成云端验证成功。

## 6. 本地补测 A：真实 Emotion2Vec

正式组件计时为 4 条固定音频 × 5 次 = 20 次，另有 2 次暖机不计入统计。调用的是生产函数 `classify_multimodal_with_metadata`，文本输入使用冻结的参考转写，**不是新获得的 ASR 结果**。GPU 单进程顺序执行；模型加载和暖机均不计入热态耗时。

20/20 次都确认真实音频模型参与、无回退。语音模型调用耗时（含音频读取及模型调用）平均 **21.34 ms**；文本规则、音频模型、韵律提取及融合整个函数平均 **31.04 ms**。该热态微基准不含网络、ASR、LLM、TTS，不能与云端第一轮的 187.7 ms 直接换算为加速倍数。

- **女儿不回消息、觉得难受**：5 次均为 `calm`；语音模型调用平均 19.48 ms，多模态函数总耗时平均 29.01 ms。
- **睡不好、反复看工作消息**：5 次均为 `calm`；语音模型调用平均 22.96 ms，多模态函数总耗时平均 33.77 ms。
- **外出散步后轻松**：5 次均为 `joy`；语音模型调用平均 20.47 ms，多模态函数总耗时平均 28.88 ms。
- **买菜后忘带钥匙（无关控制）**：5 次均为 `calm`；语音模型调用平均 22.46 ms，多模态函数总耗时平均 32.52 ms。

**20/20 是调用可用性，不是情绪识别准确率。** 这四条合成音频没有独立的人工声学情绪标注，不能用句子里出现“难受”就判定音频标签一定应为悲伤。特别值得注意的是：四条参考文本在当前规则分类器中都得到七类均匀分布，文本侧没有区分出这些措辞，因此融合主要由声音/韵律侧决定。

这批素材适合检查链路是否实际运行，不足以评估情绪适配质量。后续需要有独立标注、语气变化的音频，以及文本/声学情绪冲突样本；不能据此写“加入情绪模块提高了共情效果”。

## 7. 本地补测 B：真实 SQLite 记忆开关

通过真实 `PatientMemoryService.resolve_for_session` 和真实数据库函数建立独立会话，开/关各 20 次。没有 mock 数据库，也没有把手写历史直接塞到 LLM 中；本项根本没有调用 LLM。

- 开启：**20/20** 次加载到全部 **4 条预置历史事实**，返回上下文 421 字符；调用平均 **9.06 ms**。
- 关闭：**20/20** 次上下文不含上述历史事实，仍保留基本档案，返回上下文 63 字符；调用平均 **8.13 ms**。
- 本地 40 次调用前后所保存的记忆项、快照、删除记录、历史轮次表状态相同；而整个云端预试前后的权威卡不相同，原因是第 5 节的档案覆盖，不能混淆这两个时间范围。
- Memobase 未配置，四个输入的语义证据均为空；不把本地卡片中含有 4 条事实计成语义检索正确率。

这只证明 SQLite 卡片开关与读库路径在这份固定状态上生效。没有产生 40 个回答，不能据此评价历史引用是否合适、无关记忆是否被硬塞、对话质量是否提高。

## 8. 下一次正式实验的准入条件

1. 先核查当前 ASR/TTS 资源状态、授权与操作审计，必要时恢复授权；再分别验证真实转写与非空合成音频。不能只看连接建立或 `/health` 为 ready，也不擅自开通付费资源。
2. 保留此次失败数据，使用新的隔离目录重新冻结 protocol；解决/规避默认档案覆盖，并检查完整上下文一致。
3. 四组暖机全部通过后，才顺序运行已确定的 4×4×3 设计，场景内随机组序。任何失败都计数，不自动挑成功样本补满。
4. 若要报告“完整系统”或“说完后的等待”，先恢复 SoulX/Memobase，再用实时分帧输入；本次 manual_audio 结果不能顶替。
5. 延迟、质量、检索准确率分开评价。四个场景的重复调用不等于 48 个独立样本；质量需独立标注与更多情景，不沿用没有证据的百分比。

## 9. 可核验文件

### 9.1 本次新增的历史追溯证据

- [历史记录脱敏汇总](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/summary.json) · [原始文件 SHA256 索引](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/source_manifest.json)
- [历史 ASR/TTS 成功日志摘录](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/service_log_excerpts.json) · [非空 TTS WAV 头部核验](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/historical_tts_audio_header.json)
- [当前两份配置的无凭证值对比](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/historical_review/current_config_comparison.json)
- 历史逐轮原始 JSON 的绝对路径见第 3 节及追溯索引。历史用户转写和录音不进入新版证据包，也未作为本轮输入重新运行。

### 9.2 本轮隔离预试和本地补测

- [冻结的云端协议](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/protocol.json)
- [3 次失败预试结果](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/warmup_results.jsonl)
- [逐轮原始 WebSocket 事件](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/trials)
- [接口 403 响应及请求日志标识](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/cloud_connection_probe.json)
- [档案变化证据](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/observed_profile_mutation.json) · [记忆卡差分](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/profile_card_diff.txt)
- [20 次真实情绪模型原始数据](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/local_components/emotion_trials.jsonl)
- [40 次真实 SQLite 服务调用原始数据](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/local_components/memory_trials.jsonl)
- [汇总与清理校验](/data/luyang/lmca-share/output/real_voice_ablation_20260913_183313/summary.json)

修订记录：补入历史真实运行、同版本四组延迟、音频情绪参与核验和火山可用性时间线；保留本轮 0/3 完整成功、0/48 正式轮次及档案漂移结论。历史 JSON 是追溯证据，不冒充本轮新结果；受控文本实验、计划值和跨版本基线没有混入本报告统计。
