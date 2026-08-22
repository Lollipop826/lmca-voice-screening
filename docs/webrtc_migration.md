# WebRTC 迁移说明

## 1. 现状

- **默认传输**：WebRTC（`static/voice_chat.html` 中 `VOICE_TRANSPORT = 'webrtc'`）
- **回退路径**：WebSocket `/ws`（仅当 `?transport=websocket` 或浏览器不支持 `RTCPeerConnection` 时启用）
- **信令**：HTTP `POST /webrtc/offer`（SDP offer/answer 一次性交换）
- **ICE 配置下发**：HTTP `GET /webrtc/ice-config`
- **音频上行**：浏览器 `getUserMedia` → RTP (Opus) → `aiortc` → `WebRTCVoiceConnection.feed_audio_float` → PCM → ASR
- **音频下行**：TTS PCM → `WebRTCAssistantAudioTrack.enqueue_audio` → RTP (Opus) → 浏览器 `<audio>`
- **控制消息**：`RTCDataChannel('voice-events')` 双向 JSON（`ai_response_chunk` / 任务切换 / 图片指令等）

## 2. 为什么迁移

见 PPT 讲稿：延迟、AEC/ANS/AGC 原生、弱网自适应、自然打断、标准化协议栈。

## 3. 公网部署 TURN（对称型 NAT 必备）

### 3.1 快速起一个 coturn

```bash
sudo apt install -y coturn
sudo tee /etc/turnserver.conf <<'EOF'
listening-port=3478
fingerprint
lt-cred-mech
realm=voice.lyspeechlab.xyz
user=ad_screening:<强随机密码>
total-quota=100
bps-capacity=0
stale-nonce=600
no-loopback-peers
no-multicast-peers
EOF
sudo systemctl enable --now coturn
```

### 3.2 服务端环境变量

在 `.env` 追加：

```bash
# 自定义 STUN（可选，默认已有 Google/小米公共 STUN）
WEBRTC_STUN_URLS=stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302

# TURN（公网生产必填）
WEBRTC_TURN_URL=turn:voice.lyspeechlab.xyz:3478
WEBRTC_TURN_USERNAME=ad_screening
WEBRTC_TURN_CREDENTIAL=<强随机密码>
```

重启 `voice_server.py` 后，`GET /webrtc/ice-config` 会把该列表下发到浏览器。

## 4. 调试参数

| URL 参数 | 作用 |
|---|---|
| `?transport=webrtc`（默认） | 强制走 WebRTC |
| `?transport=websocket` | 强制走 WebSocket（debug 用） |
| `?transport=auto&fallback=websocket` | 先试 WebRTC，失败自动回退 WS |

## 5. 回滚

```bash
git log --oneline --decorate -20
git switch --detach <已验证的提交>
```

当前开发分支是 `feat/webrtc-migration`。回滚前先确认工作区没有未提交修改，并选择
`archive/webrtc-delivery.md` 中记录的已验证提交。

## 6. 已知限制

- TURN 未部署时，对称型 NAT（部分 4G / 企业防火墙）用户可能 ICE 失败并回退 WS。
- aiortc 的 Opus 编码 CPU 占用比 WS 直传 PCM 略高，但带宽占用降低约 5–10 倍。
