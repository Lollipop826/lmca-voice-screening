# 林秀兰虚构演示测试数据

## 身份与用途

- 姓名：林秀兰
- `patient_id`：`pt_a71f682f92f74f7d`
- 数据性质：`extra_profile.fictional_demo=true`
- 原档案声明：人工拟定的虚构演示患者，不对应真实个人；仅用于心理健康陪伴系统的记忆链路演示，不构成诊断、治疗建议或风险结论。

## 包含范围

包内 `data/voice_server.db` 只保留该 patient id 关联的数据：

- 1 条患者档案
- 5 个 wellbeing 会话
- 13 条消息
- 13 条音频索引
- 27 条长期记忆轮次
- 19 个结构化记忆项
- 3 条结构化记忆停止使用记录
- 1 条长期记忆 snapshot 和 1 条同步状态
- 27 条情绪轨迹
- 6 条患者审计记录

包内 `data/voice_calls/` 只包含以下目录：

```text
call_20260806_175322_f4f30fc1
call_20260806_175337_13b3aedd
call_20260806_181101_1d2be478
call_20260806_181115_92d5049c
call_20260806_181322_be4f778d
```

## 脱敏与排除

- 未复制原始 `data/voice_server.db`；包内数据库是筛选后的独立副本。
- 删除所有其他患者、会话、消息、音频索引、记忆、情绪和审计记录。
- 清空 `users`、`public_api_keys` 和 `patient_assignments`，不携带本机用户名、密码哈希或 API key。
- 会话 owner 和审计操作者统一改为 `demo_admin`。
- memory deletion token 重建为无敏感意义的测试 token。
- 不包含 `.auth_secret`、`.public_api_key`、`.safety_hmac_secret`、`.env`、Memobase PostgreSQL 数据或其他本地运行数据。
- 数据库执行 `secure_delete` 和 `VACUUM`，避免已删除记录残留在空闲页。

## 使用说明

1. 将 `.env.example` 复制为 `.env` 并配置自己的服务凭据。
2. 首次登录请通过现有注册流程创建自己的账号；数据库不附带测试账号或默认密码。
3. 测试长期记忆时使用 `patient_id=pt_a71f682f92f74f7d`。
4. 音频文件路径保留为 `data/voice_calls/...`，与包内目录一致。
5. 该数据用于功能、迁移和回归测试，不用于医学结论或模型质量证明。
