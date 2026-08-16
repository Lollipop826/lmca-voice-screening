# 实验进度

## 2026-08-12

- 用户确认规模：S0=0、S1=10、S2=100、S3=500、S4=1000、S5=5000。
- 已确认 embedding 必须包含在完整实验链路中。
- 已加载全局记忆、`karpathy-guidelines`、`ponytail`、`context-proxy` 和 `planning-with-files`。
- 当前进入现有实现与服务能力核对阶段。
- 完成真实调用链核对，确认可精确批量生成 event gist 及真实 embedding。
- 增加只读基准开关及健康检查/脚本校验，正式默认仍为开启写回。
- 相关测试 `58 passed`。
- 首次旧数据清理命令因组合了动态路径校验、批量删除和容器查询而被安全策略拦截；未删除任何文件，已改用显式绝对路径方案。
- 旧 `memory_latency*` 文件已删除；旧实验 Memobase 用户计数复核为 users/events/gists=`0/0/0`。
- 将实验隔离修正为独立 Memobase、单患者逐档累积，保证每档全局索引规模就是该档数量。
- 修复 `tests/test_memory_scale_fixture.py` 缺少 `argparse` 导入。
- `python -m py_compile` 通过；夹具、基准、语音服务和旧记忆切换相关回归共 `60 passed`。
- 开始建立独立 `lmca-memory-scale` Compose 环境与 S0 空基线，端口为 API `8020`、PostgreSQL `15433`、Redis `16380`。
- 隔离容器启动日志确认 `qwen3.7-text-embedding`、1024维和真实 document embedding sanity 成功。
- 发现本地已构建镜像代码旧于工作区，隔离库缺少当前源码定义的 HNSW；改为只读挂载当前 Memobase 包，并在空实验库补建该索引。
- S0冒烟通过，但正式组先后受 BigASR瞬时重置和Docker Desktop停止影响，未产生正式结果。
- 用户将范围调整为只测试 S=1000；停止其余规模，直接构造并验证1000条后运行开/关对照。
- S=1000本地基线完成：turns/memory items/emotion trajectory=`1000/1000/1000`。
- 真实embedding导入完成：events/event embeddings=`50/50`，gists/gist embeddings=`1000/1000`，维度1024，构建耗时152832.54ms。
- HNSW和真实查询验收通过：命中3条、top similarity=0.606403；0.99阈值未命中0条。
- E2E开启/关闭各完成30个有效样本，Emotion2Vec 60/60通过；原始结果保存为 `s1000-on.json` 和 `s1000-off.json`。
- 直接检索30次：总耗时均值1802.24ms，query embedding 1793.38ms，pgvector/HNSW 6.00ms。
- 实验后本地/远端计数仍为1000，关闭组记忆检索日志为0；实验语音服务已恢复长期记忆读取开启、写回关闭。
