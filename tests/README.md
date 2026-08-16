# 测试目录

自动化测试按业务边界分组：

- `agents/`：筛查 Agent、患者记忆和上下文。
- `voice/`：WebRTC、连接、会话、媒体、处理器和语音服务。
- `web/`：认证、公开 API 和 Web 应用。
- `integrations/`：Ark ASR、SoulX 等外部集成边界。

运行全部测试：

```bash
./luyang/bin/python -m pytest -q
```

只运行某个领域：

```bash
./luyang/bin/python -m pytest -q tests/agents
./luyang/bin/python -m pytest -q tests/voice
./luyang/bin/python -m pytest -q tests/web
./luyang/bin/python -m pytest -q tests/integrations
```
