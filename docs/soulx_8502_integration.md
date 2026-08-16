# SoulX-Duplug 作为 8502 全双工模块

## 最终架构

SoulX 只负责语义轮次判断，不启动自己的对话页面，也不替换 8502 现有业务：

```text
8502 浏览器麦克风（WebRTC 或 WebSocket）
  -> voice_server.py（16 kHz float32 PCM）
  -> SoulX ws://127.0.0.1:8000/turn
       blank   继续积累
       idle    静音、噪音或附和，不打断
       nonidle 用户有语义内容；AI 正在说话时立即停止播放
       speak   用户表达完整；把原始音频交给现有 ArkASR + MMSE Agent
  -> 现有 ArkTTS/WebRTC 下行、评分、历史记录和声纹验证
```

不再需要 `SoulX-Duplug-dialogue-system` 的 55556 页面，也不需要
`soulx_bridge` 的 6009 Agent/TTS 桥接。

## 启动

仍然只运行：

```bash
cd /path/to/voice-screening-service
./start_voice_only.sh
```

脚本会：

1. 在 GPU 1 启动 SoulX `127.0.0.1:8000`，或复用已经运行的实例；
2. 等待 SoulX 模型加载完成；
3. 启动现有 8502 服务；
4. 当前脚本退出时，只清理由它启动的 SoulX 进程。

日志：

- `tmp/soulx_turn.log`
- `tmp/voice_server.log`

## 配置

```dotenv
USE_SOULX_TURN_TAKING=true
SOULX_TURN_URL=ws://127.0.0.1:8000/turn
SOULX_TIMEOUT_S=3.0
SOULX_RETRY_INTERVAL_S=5.0
SOULX_MIN_UTTERANCE_RMS=0.008
ENABLE_FULL_DUPLEX=true
```

启动脚本还支持：

```bash
SOULX_DIR=/path/to/SoulX-Duplug-main
SOULX_PYTHON=/path/to/soulx/python
SOULX_CUDA_VISIBLE_DEVICES=1
```

SoulX 暂时不可用时，当前会话自动降级到原有 Silero VAD；到达重试时间后会自动重连。
本地 VAD 在 SoulX 健康时不会并行运行。`SOULX_MIN_UTTERANCE_RMS` 用于拒绝能量过低的
远场电视声、旁人声和噪音。

如果现场存在音量较大的电视、手机外放或旁人持续说话，仅靠能量阈值无法区分说话人，
应在 8502 设置页加载患者声纹并开启“声纹验证”。

## 验证

健康检查应显示：

```json
{
  "status": "ok",
  "webrtc": true,
  "turn_taking": "soulx",
  "soulx_turn_url": "ws://127.0.0.1:8000/turn"
}
```

协议测试：

```bash
./luyang/bin/python -m unittest discover \
  -s tests -p 'test_soulx_turn_taking.py' -v
```
