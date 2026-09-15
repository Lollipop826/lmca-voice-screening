"""
火山引擎豆包语音大模型 V3 WebSocket 双向流式 TTS（替换 ZipVoice 本地模型）

API 文档: wss://openspeech.bytedance.com/api/v3/tts/bidirection
鉴权文档: X-Api-App-Key (APP ID) + X-Api-Access-Key (Access Token)

环境变量：
  VOLC_APP_ID        - APP ID（控制台数字ID，非 API Key）
  VOLC_ACCESS_TOKEN  - Access Token（控制台获取的 UUID 格式）
  VOLC_TTS_VOICE     - 音色 (默认 zh_female_vv_uranus_bigtts)
  VOLC_TTS_RESOURCE  - 资源ID (默认 seed-tts-2.0)

接口与 ZipVoiceTTS 完全兼容：
  async def text_to_speech_streaming(text, emotion) -> AsyncGenerator[np.ndarray]
  每块 yield float32 numpy array, 24kHz
"""
import asyncio
import json
import os
import struct
import time
import uuid
import numpy as np
import websockets
from websockets.protocol import State
from typing import AsyncGenerator

from .ws_proxy import websocket_proxy_kwargs

_WSS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = int(_SAMPLE_RATE * 0.08)  # 80ms per yield chunk
_PING_TIMEOUT_S = 1.0

# 打断后取消并排空残留帧的时间上限。实测 CancelSession 到会话终态约
# 0.39s；预算留到 1s，优先保住热连接，仍明显短于重新握手的 1.3~1.8s。
# 超时则判定连接不可复用，摘除并转后台重建——重建有下一轮 LLM 生成（约 1.2s）
# 作掩护，不会落到用户可感延迟上。
_DRAIN_DEADLINE_S = 1.0

_TTS_EMOTION_PROFILES = {
    "neutral": ("neutral", 3, 0),
    "gentle": ("neutral", 3, -12),
    "calm": ("neutral", 3, -8),
    "happy": ("happy", 3, 5),
    "joy": ("happy", 3, 5),
    "sad": ("sad", 2, -5),
    "sadness": ("sad", 2, -5),
    "fear": ("neutral", 2, -10),
    "fearful": ("neutral", 2, -10),
    "anxiety": ("neutral", 2, -10),
    "anger": ("neutral", 2, -8),
    "angry": ("neutral", 2, -8),
    "confusion": ("neutral", 2, -5),
}

# ── Binary protocol constants ──────────────────────────────────────────────
_PROTO_HDR = bytes([0x11, 0x14, 0x10, 0x00])   # v1, full-client-req+event, JSON, no-compress
_EVT_START_CONN      = 1
_EVT_FINISH_CONN     = 2
_EVT_CONN_STARTED    = 50
_EVT_CONN_FAILED     = 51
_EVT_START_SESSION   = 100
_EVT_CANCEL_SESSION  = 101
_EVT_FINISH_SESSION  = 102
_EVT_SESSION_STARTED = 150
_EVT_SESSION_CANCELED = 151
_EVT_SESSION_FINISHED = 152
_EVT_SESSION_FAILED  = 153
_EVT_TASK_REQUEST    = 200
_EVT_TTS_RESPONSE    = 352   # Audio-only server frame


def _frame_no_id(event: int, payload: bytes) -> bytes:
    """Build a client frame WITHOUT session/connection id (StartConnection, FinishConnection)."""
    return (
        _PROTO_HDR
        + struct.pack(">i", event)
        + struct.pack(">I", len(payload))
        + payload
    )


def _frame_with_sid(event: int, session_id: str, payload: bytes) -> bytes:
    """Build a client frame WITH session id (StartSession, TaskRequest, FinishSession)."""
    sid = session_id.encode()
    return (
        _PROTO_HDR
        + struct.pack(">i", event)
        + struct.pack(">I", len(sid))
        + sid
        + struct.pack(">I", len(payload))
        + payload
    )


def _parse_server_frame(data: bytes) -> dict:
    """
    Parse a binary server frame.
    Returns dict with keys: event, audio (bytes|None), payload (dict|None), error (str|None)
    """
    if len(data) < 8:
        return {"event": None, "audio": None, "payload": None, "error": "frame too short"}

    msg_type = data[1] & 0xF0   # left 4 bits of byte 1
    serialization = data[2] & 0xF0
    compression = data[2] & 0x0F

    pos = 4
    event = struct.unpack(">i", data[pos:pos+4])[0]
    pos += 4

    result = {"event": event, "audio": None, "payload": None, "error": None}

    # All server frames carry an id field (connection_id or session_id)
    if len(data) < pos + 4:
        return result
    id_size = struct.unpack(">I", data[pos:pos+4])[0]
    pos += 4 + id_size   # skip the id bytes

    # Payload / audio
    if len(data) < pos + 4:
        return result
    payload_size = struct.unpack(">I", data[pos:pos+4])[0]
    pos += 4
    if len(data) < pos + payload_size:
        return result

    raw = data[pos:pos+payload_size]

    if msg_type == 0xB0:   # Audio-only response → raw PCM
        result["audio"] = raw
    elif msg_type == 0x90 and serialization == 0x10:   # Full-server response, JSON
        try:
            result["payload"] = json.loads(raw)
        except Exception:
            pass
    elif msg_type == 0xF0:   # Error frame
        try:
            result["payload"] = json.loads(raw)
        except Exception:
            pass
        result["error"] = str(result["payload"])

    return result


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


class ArkTTS:
    """
    火山引擎豆包语音大模型 V3 WebSocket 流式 TTS。
    接口与 ZipVoiceTTS 完全兼容。
    使用持久连接复用（只建连一次，每次合成只发 StartSession），降低首包延迟。
    """

    def __init__(self):
        self._app_id = os.getenv("VOLC_APP_ID", "")
        self._token  = os.getenv("VOLC_ACCESS_TOKEN", "")
        if not self._app_id or not self._token:
            raise ValueError("未配置 VOLC_APP_ID / VOLC_ACCESS_TOKEN，请在 .env 中添加")
        self._clone_speaker = os.getenv("VOLC_TTS_SPEAKER_ID", "").strip()
        self._voice = self._clone_speaker or os.getenv("VOLC_TTS_VOICE", "zh_female_vv_uranus_bigtts")
        configured_resource = os.getenv("VOLC_TTS_RESOURCE", "").strip()
        self._resource = configured_resource or ("seed-icl-2.0" if self._clone_speaker else "seed-tts-2.0")
        if self._clone_speaker and self._resource == "seed-tts-2.0":
            self._resource = "seed-icl-2.0"
        self._ws       = None   # 持久 WebSocket 连接
        self._ws_lock  = asyncio.Lock()
        self._recovery_task = None   # 后台重建任务，避免重连落在关键路径上
        print(f"[ArkTTS] 初始化完成，音色: {self._voice}, 资源: {self._resource}")

    @property
    def voice(self) -> str:
        return self._voice

    def set_voice(self, voice: str) -> None:
        value = str(voice or "").strip()
        if not value:
            raise ValueError("音色不能为空")
        self._voice = value

    async def _open_connection(self):
        """创建并初始化一条连接，但不抢占会话串行锁。"""
        conn_id = str(uuid.uuid4())
        ws_headers = {
            "X-Api-App-Key":     self._app_id,
            "X-Api-Access-Key":  self._token,
            "X-Api-Resource-Id": self._resource,
            "X-Api-Connect-Id":  conn_id,
        }
        t_conn = time.time()
        ws = await websockets.connect(
            _WSS_URL,
            additional_headers=ws_headers,
            open_timeout=10,
            **websocket_proxy_kwargs(),
        )
        try:
            await ws.send(_frame_no_id(_EVT_START_CONN, b"{}"))
            resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
            if resp["event"] != _EVT_CONN_STARTED:
                raise RuntimeError(f"建连失败 event={resp['event']}")
        except BaseException:
            await ws.close()
            raise
        print(f"[ArkTTS] 🔗 连接建立完成 ({time.time()-t_conn:.2f}s)，后续请求直接复用")
        return ws

    async def _ensure_connected(self) -> None:
        """确保 WebSocket 连接已建立并 StartConnection 完成（连接复用）。"""
        if self._ws is not None:
            websocket = self._ws
            try:
                # websockets.ping() 返回 Pong waiter；第一层 await 只把 Ping
                # 发出去，第二层才确认对端确实回了 Pong。
                pong_waiter = await websocket.ping()
                if pong_waiter is not None:
                    await asyncio.wait_for(pong_waiter, timeout=_PING_TIMEOUT_S)
                return
            except Exception:
                if self._ws is websocket:
                    self._ws = None
                self._schedule_reconnect(websocket)

        # 打断排空超时时会立即启动独立的后台建连。下一次请求复用该任务，
        # 避免再创建第二条连接，也避免完整握手从头落到关键路径上。
        recovery = self._recovery_task
        if recovery is not None:
            try:
                await asyncio.shield(recovery)
            finally:
                if self._recovery_task is recovery and recovery.done():
                    self._recovery_task = None
            if self._ws is not None:
                return

        self._ws = await self._open_connection()

    async def prewarm(self) -> None:
        async with self._ws_lock:
            await self._ensure_connected()

    async def _discard_session_tail(self, session_id: str) -> bool:
        """取消并读完被打断会话，让连接回到可复用状态。

        先发 CancelSession，让服务端停止继续合成；再读到会话终态。当前服务
        实测约 0.4~0.5s 收到 SessionCanceled / SessionFinished。
        若超过 _DRAIN_DEADLINE_S 仍未读完，说明服务端还在持续产出，此时放弃
        这条连接并在后台重建，避免让下一次合成等在排空上。
        """
        websocket = self._ws
        if websocket is None:
            return False
        started_at = time.monotonic()
        deadline = time.monotonic() + _DRAIN_DEADLINE_S
        try:
            await websocket.send(_frame_with_sid(
                _EVT_CANCEL_SESSION, session_id, b"{}"
            ))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                frame = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                resp = _parse_server_frame(frame)
                if resp["event"] in (
                    _EVT_SESSION_CANCELED,
                    _EVT_SESSION_FINISHED,
                    _EVT_SESSION_FAILED,
                ):
                    print(
                        "[ArkTTS] 🧹 打断会话已收尾，连接继续复用 "
                        f"({(time.monotonic()-started_at)*1000:.0f}ms)"
                    )
                    return True
                if resp["error"]:
                    raise RuntimeError("排空时收到错误帧")
        except Exception as exc:
            # 连接已不可复用：立刻摘除，重建放到后台，不占用本次请求。
            if self._ws is websocket:
                self._ws = None
                self._schedule_reconnect(websocket)
            print(
                "[ArkTTS] ♻️ 打断会话未及时收尾，连接转后台重建: "
                f"{type(exc).__name__}"
            )
            return False

    def _schedule_reconnect(self, stale_websocket):
        """后台关闭废弃连接并预建新连接，把重连开销移出关键路径。"""
        if self._recovery_task is not None and not self._recovery_task.done():
            return self._recovery_task

        async def _rebuild() -> None:
            if stale_websocket is not None and stale_websocket.state is State.OPEN:
                try:
                    await stale_websocket.close()
                except Exception:
                    pass
            try:
                websocket = await self._open_connection()
                if self._ws is None:
                    self._ws = websocket
                else:
                    await websocket.close()
            except Exception as exc:
                print(f"[ArkTTS] ⚠️ 后台重建连接失败: {type(exc).__name__}")

        self._recovery_task = asyncio.create_task(_rebuild())
        return self._recovery_task

    @staticmethod
    def _audio_params(emotion: str) -> dict:
        key = str(emotion or "neutral").strip().casefold()
        api_emotion, emotion_scale, speech_rate = _TTS_EMOTION_PROFILES.get(
            key, _TTS_EMOTION_PROFILES["neutral"]
        )
        return {
            "format": "pcm",
            "sample_rate": _SAMPLE_RATE,
            "speech_rate": speech_rate,
            "emotion": api_emotion,
            "emotion_scale": emotion_scale,
        }


    async def text_to_speech_streaming(
        self,
        text: str,
        emotion: str = "neutral",
        voice: str | None = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        WebSocket 双向流式 TTS，复用持久连接降低首包延迟。
        与 ZipVoiceTTS.text_to_speech_streaming 接口完全兼容。

        Yields:
            float32 numpy array, 24kHz
        """
        t0 = time.time()
        first_chunk = True
        selected_voice = str(voice or self._voice).strip() or self._voice

        async with self._ws_lock:   # 同一连接不支持并发 session
            for attempt in range(2):
                # 会话是否读到 SessionFinished。为 False 说明连接上仍有本次
                # 会话的残留帧，直接复用会让下一次 StartSession 读到错帧。
                session_drained = False
                session_id = None
                try:
                    await self._ensure_connected()
                    session_id = str(uuid.uuid4())
                    print(f"[ArkTTS] 🎵 流式合成 text_chars={len(text)}")

                    session_params = {
                        "speaker": selected_voice,
                        "audio_params": self._audio_params(emotion),
                    }
                    # StartSession
                    session_meta = {
                        "user": {"uid": "mmse_screening"},
                        "req_params": session_params,
                    }
                    await self._ws.send(_frame_with_sid(
                        _EVT_START_SESSION, session_id,
                        json.dumps(session_meta, ensure_ascii=False).encode()
                    ))
                    resp = _parse_server_frame(
                        await asyncio.wait_for(self._ws.recv(), timeout=10)
                    )
                    if resp["event"] != _EVT_SESSION_STARTED:
                        raise RuntimeError(f"Session 启动失败 event={resp['event']}")

                    # TaskRequest
                    task_payload = json.dumps(
                        {"req_params": {"text": text}}, ensure_ascii=False
                    ).encode()
                    await self._ws.send(_frame_with_sid(_EVT_TASK_REQUEST, session_id, task_payload))
                    # FinishSession
                    await self._ws.send(_frame_with_sid(_EVT_FINISH_SESSION, session_id, b"{}"))

                    # 接收音频直到 SessionFinished
                    while True:
                        data = await asyncio.wait_for(self._ws.recv(), timeout=30)
                        resp = _parse_server_frame(data)
                        evt  = resp["event"]

                        if resp["audio"]:
                            audio = _pcm_to_float32(resp["audio"])
                            if first_chunk:
                                print(f"[ArkTTS] 🎵 首包! 延迟={time.time()-t0:.2f}s")
                                first_chunk = False
                            for chunk in [audio[i:i+_CHUNK_SAMPLES]
                                          for i in range(0, len(audio), _CHUNK_SAMPLES)]:
                                yield chunk

                        elif evt == _EVT_SESSION_FINISHED:
                            print(f"[ArkTTS] ✅ 完成，总耗时 {time.time()-t0:.2f}s")
                            session_drained = True
                            break

                        elif evt == _EVT_SESSION_FAILED or resp["error"]:
                            raise RuntimeError("TTS Session 失败")

                    return  # 成功，退出重试循环

                except Exception as e:
                    stale_websocket = self._ws
                    self._ws = None
                    if stale_websocket is not None:
                        self._schedule_reconnect(stale_websocket)
                    if attempt == 0:
                        print(f"[ArkTTS] ⚠️ 连接异常，重试一次: {type(e).__name__}")
                    else:
                        print(f"[ArkTTS] ❌ 失败: {type(e).__name__}")
                        raise

                finally:
                    # 消费方 break 出 async for 时（打断场景），生成器停在 yield
                    # 处，本次会话剩余的音频帧与 SessionFinished 仍留在连接上。
                    # 不清理就复用会让下一次 StartSession 读到这些残留帧，从而
                    # 抛 RuntimeError 并触发一次约 1.6s 的重连。
                    if (
                        not session_drained
                        and session_id is not None
                        and self._ws is not None
                    ):
                        await self._discard_session_tail(session_id)

    async def close(self):
        """关闭持久连接。"""
        recovery = self._recovery_task
        self._recovery_task = None
        if recovery is not None and not recovery.done():
            recovery.cancel()
            try:
                await recovery
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.send(_frame_no_id(_EVT_FINISH_CONN, b"{}"))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None


# 兼容别名
VoiceTTS = ArkTTS
