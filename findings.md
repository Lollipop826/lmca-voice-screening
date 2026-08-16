# 实验发现

## 已知基线

- 旧测试仅有 1 条基线历史对话，开启组最终只有 3-4 个本地轮次和 1 条结构化 memory item。
- 旧测试只能代表少量记忆下的固定接入开销，不能代表规模增长。
- 旧烟雾测试中 Emotion2Vec 实际运行，两组情绪结果一致；Memobase 检索约 600 ms。

## 待核对

- 项目当前的真实 embedding 提供方、向量表和索引类型。
- Memobase 批量写入、flush、删除用户和验证检索的官方客户端接口。
- 最终采用独立实验 Memobase + 单患者逐档累积。每档测试时，全库与目标患者向量量均为0/10/100/500/1000/5000；六用户共存会让S0/S1也在6610条全局HNSW上查询，不能回答索引随规模增长的耗时。
- 当前模型官方同步 embedding 接口单次最多20条输入；event gist 每批硬限制20条。
- `append_user_event` 在 embedding 失败时可能仍插入 NULL 并返回成功，所以每批必须验收 event/gist embedding 非空；失败则删除整批事件后重试。

## 已核对实现

- SDK 公开写入 `ChatBlob -> flush` 会经过 LLM 抽取，gist 数量不可控，不适合精确档位。
- 服务端现有 `controllers.event.append_user_event()` 会生成真实 event 与 event-gist embedding，并写入 pgvector；按批调用可精确得到目标 gist 数量。
- 当前 embedding 为 `qwen3.7-text-embedding`、1024维，服务启动维度自检通过。
- `user_event_gists.embedding` 已建 HNSW cosine 索引 `idx_user_event_gists_embedding_hnsw_cosine`。
- 基准对话会在断连时写回记忆，已增加默认开启的 `ENABLE_LONG_TERM_MEMORY_WRITES`；实验设为 false 后只读检索仍运行，写入/整理/flush 均冻结。
- 只删除旧实验患者 `pt_89c6a351b75f4325` 映射出的 Memobase 用户；保留库中其他用户。

## S=1000 最终结果

- 1000条gist和1000个gist embedding全部存在，维度统一为1024；50个event embedding全部存在。
- HNSW cosine索引存在；命中查询返回3条，top similarity=0.606403；0.99阈值未命中查询返回0条。
- 开/关各30个有效E2E样本，Emotion2Vec有效样本为60/60；开启组失败尝试0，关闭组失败尝试5（4次BigASR连接重置、1次AI响应失败）。
- `asr_to_first_tts_byte_ms`：开启均值/p50/p95=`5334.09/5103.63/7824.85`，关闭=`4756.41/3228.25/11748.09`；均值增量577.68ms，中位数增量1875.38ms。
- 直接记忆检索30次：总耗时均值1802.24ms，query embedding均值1793.38ms，pgvector/HNSW均值6.00ms。主要瓶颈是远程query embedding，不是1000条向量搜索。
- 关闭组外部链路尾延迟更差，负的p95差值不表示开启记忆加速；这是非配对外部ASR/LLM/TTS抖动的证据。
