# SoulX-Duplug 全双工接入（当前部署：8427 / 8001）

## 最终架构

SoulX 负责语义轮次判断和流式转写，语音服务继续负责当前会话的对话、播报与持久化：

```text
8427 浏览器麦克风（WebRTC 或 WebSocket）
  -> voice_server.py（16 kHz float32 PCM）
  -> SoulX ws://127.0.0.1:8001/turn
       blank   继续积累
       idle    静音、噪音或附和，不打断
       nonidle 确认不是用户附和后，取消旧生成任务并停止正在播放的回复
       speak   用户表达完整；原始音频和最终转写交给当前会话 Agent
  -> 现有 ArkTTS/WebRTC 下行、评分、历史记录和声纹验证
```

不再需要 `SoulX-Duplug-dialogue-system` 的 55556 页面，也不需要
`soulx_bridge` 的 6009 Agent/TTS 桥接。

## 启动

本机当前使用：

```bash
cd /data/luyang/lmca-share
./start_luyang.sh
```

脚本会：

1. 根据 `.env` 在 GPU 0 启动 SoulX `127.0.0.1:8001`，或复用健康实例；
2. 等待 SoulX `/health` 确认模型已加载；
3. 启动现有 8427 服务，检查主服务和 SoulX 同时就绪；
4. 服务在脚本退出后继续运行，PID 分别记录到 `tmp/soulx_turn.pid` 和
   `tmp/voice_server_8427.pid`。端口占用但健康检查失败时，脚本报错。

修改 Python 代码或 `.env` 后需要重启对应服务。先确认通话结束，再根据 PID、
`/proc/<PID>/cwd` 和命令行核对进程，使用 `kill -TERM <PID>`；不要用宽泛的
`pkill -f` 杀进程。`start_voice_only.sh` 是另一套部署入口，不用于本机这次恢复。

日志：

- `tmp/soulx_turn.log`
- `tmp/voice_server.log`

## 配置

```dotenv
USE_SOULX_TURN_TAKING=true
SOULX_TURN_URL=ws://127.0.0.1:8001/turn
SOULX_TIMEOUT_S=10.0
SOULX_RETRY_INTERVAL_S=5.0
SOULX_MIN_UTTERANCE_RMS=0.008
ENABLE_FULL_DUPLEX=true
```

启动脚本还支持：

```bash
SOULX_DIR=/data/luyang/SoulX-Duplug
SOULX_PYTHON=/data/luyang/envs/soulx-duplug/bin/python
SOULX_HOST=127.0.0.1
SOULX_PORT=8001
SOULX_CUDA_VISIBLE_DEVICES=0
SOULX_DEVICE=cuda
```

SoulX 暂时不可用时，当前会话自动降级到 Silero VAD；到达重试时间后会自动重连。
本地 VAD 在 SoulX 健康时不会并行运行。`SOULX_MIN_UTTERANCE_RMS` 用于拒绝能量过低的
远场电视声、旁人声和噪音。

本地全双工打断复用连续 VAD 推理结果。AI 播放期间，满足概率、RMS 和连续帧门槛的
人声达到 `FULL_DUPLEX_STOP_DURATION`（默认 0.20 秒）就先停止播放，继续保存完整语音；
停播不等待静音结束、ASR 或语义分类。声纹验证启用时仍需验证目标说话人。
播放中的「嗯嗯」短回应保留原有豁免。后端记录检测人声和发送停播的时间，前端回报执行
停止的时间和处理耗时；跨端时钟可能有偏差，不能直接相减当作设备最终出声延迟。

如果现场存在音量较大的电视、手机外放或旁人持续说话，仅靠能量阈值无法区分说话人，
应在语音设置页加载患者声纹并开启“声纹验证”。

## SoulX 独立环境

当前源码来自 `https://github.com/Soul-AILab/SoulX-Duplug`，基准提交
`a0b9063843df69619b087b95b74597b2176910b8`。本地补丁保存在
`scripts/patches/soulx_lmca.patch`，包含模型健康检查、设备配置及轮次详情/RMS 字段。
重建源码时需在该提交上应用补丁；直接使用未打补丁的官方服务不会通过主服务健康检查。

主服务继续使用 `/home/student1/miniforge3/envs/lmca/bin/python`。SoulX 使用独立
venv，通过 `--system-site-packages` 复用现有 Torch 2.5.1 / CUDA 12.1，新增包仅装入 venv：

```bash
/home/student1/miniforge3/envs/lmca/bin/python -m venv --system-site-packages /data/luyang/envs/soulx-duplug
PIP_CACHE_DIR=/data/luyang/pip-cache /data/luyang/envs/soulx-duplug/bin/python -m pip install -r scripts/soulx_requirements.txt
```

以上是重建步骤，环境已存在时无需重复执行。不要修改共享主服务环境。官方权重
`Soul-AILab/SoulX-Duplug-0.6B` 放在 `/data/luyang/SoulX-Duplug/pretrained_models`；
Paraformer 权重由 ModelScope 缓存到 `/data/luyang/cache/modelscope`。首次启动前应完成
模型下载，避免启动超时。启动脚本将 HF、ModelScope 和临时缓存都指向 `/data/luyang`。
加载已有 Emotion2Vec 缓存时，脚本还会创建一个不含 `requirements.txt` 的符号链接目录，
避免 FunASR 在重启过程中自动安装依赖；权重和共享 Python 环境不变。
情绪预热还会用临时生成音频执行韵律特征提取，避免首次调用 Librosa/Numba 的初始化
延迟落到用户的第一轮回复上。

## 输入取代旧回复

SoulX `nonidle` 既能停止播放，也能取消尚未开始播报的旧生成任务；停播前先检查已识别
文本，尚无文本或只有“嗯”“对”“是的”等附和时保留原回复。判断按整段文本匹配，兼容
句号和重复附和；“嗯，我想换个话题”“不对”“等一下”仍可在 `nonidle` 阶段打断。

`speak` 也检查附和，避免句尾提交新轮次后，前端因 `asr_result` 再次停播。是否与原回复
重叠从输入阶段保留到最终转写，即使等待句尾时原回复已播完也不另起附和轮次。AI 已在
聆听时，用户单独回答“对”“是的”仍正常提交；AI 自己的“嗯嗯”或“我在听”不算正式回复。

真正的新输入在 `speak` 提交前再次检查旧任务，避免新句子排在旧回复后面。
情绪推理等待会传播所属语音任务的取消信号。
本地 VAD 回退路径复用已经确认的整句 ASR 结果；合并音频时重新识别整段，避免只使用
最后一个片段。最终转写为空时保留录音并向前端发送重说提示，不生成无依据的回复。

## 实时短回应

启用 `ENABLE_REALTIME_COMPANION` 后，普通短回应同时检查本轮音频时长、
SoulX 的语义状态和 `speech_detected`：

- 本轮超过 **5 秒**、语义未完成、患者仍在说话：只播报“嗯嗯”。
- 本轮超过 **7 秒**、语义未完成、患者已停顿：播报“我在听，您慢慢说。”。

时长从本轮首次检测到讲话后的音频帧累计，包含本轮内的停顿，不包含开口前的
静音、预录缓冲或网络等待。恰好 5 秒、7 秒时均不触发。
`incomplete_wait` 延续同一次 `incomplete` 判定；已提交完整轮次的 `speak`
及 `incomplete_timeout` 不触发普通短回应。缺少明确的说话状态时也不触发。

每帧都会重新检查条件，因此过早收到或内容没有变化的 `incomplete` 不会阻止
后续达到时长门槛时回应。普通回应仍共用至少 10 秒的间隔，每轮最多 2 次。
情绪分数和难受关键词不再单独触发普通安抚；显式安全风险提醒沿用原有优先路径。
“嗯嗯”允许患者继续讲话，较长的倾听提示会在患者恢复讲话时停止。

## 验证

健康检查应显示：

```json
{
  "status": "ok",
  "ready": true,
  "webrtc": true,
  "turn_taking": "soulx",
  "configured_turn_taking": "soulx",
  "turn_taking_health": {"available": true},
  "soulx_turn_url": "ws://127.0.0.1:8001/turn"
}
```

这是响应的部分字段。如果 SoulX 未就绪，应显示 `status: degraded`、`ready: false`、
`turn_taking: local`，即使 `startup.ready` 为 true 也不代表全双工依赖正常。

协议回归与真实 PCM 冒烟测试：

```bash
/home/student1/miniforge3/envs/lmca/bin/python -m pytest -q tests/integrations/test_soulx_turn_taking.py tests/voice/test_voice_health.py
/home/student1/miniforge3/envs/lmca/bin/python scripts/smoke_soulx_turn_taking.py --check-isolation
```

冒烟测试使用仓库测试音频，检查 `nonidle → speak`、非空最终转写，以及交错发送的另一条
静音会话没有串入文字，并报告每个 160ms 音频块的耗时。它只连接 SoulX，不创建患者数据。
