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
from typing import AsyncGenerator

_WSS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = int(_SAMPLE_RATE * 0.08)  # 80ms per yield chunk

# ── Binary protocol constants ──────────────────────────────────────────────
_PROTO_HDR = bytes([0x11, 0x14, 0x10, 0x00])   # v1, full-client-req+event, JSON, no-compress
_EVT_START_CONN      = 1
_EVT_FINISH_CONN     = 2
_EVT_CONN_STARTED    = 50
_EVT_CONN_FAILED     = 51
_EVT_START_SESSION   = 100
_EVT_FINISH_SESSION  = 102
_EVT_SESSION_STARTED = 150
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
        self._voice    = os.getenv("VOLC_TTS_VOICE",    "zh_female_vv_uranus_bigtts")
        self._resource = os.getenv("VOLC_TTS_RESOURCE", "seed-tts-2.0")
        self._ws       = None   # 持久 WebSocket 连接
        self._ws_lock  = asyncio.Lock()
        print(f"[ArkTTS] 初始化完成，音色: {self._voice}, 资源: {self._resource}")

    async def _ensure_connected(self) -> None:
        """确保 WebSocket 连接已建立并 StartConnection 完成（连接复用）。"""
        if self._ws is not None:
            try:
                await self._ws.ping()  # 检测连接是否存活
                return
            except Exception:
                self._ws = None

        conn_id = str(uuid.uuid4())
        ws_headers = {
            "X-Api-App-Key":     self._app_id,
            "X-Api-Access-Key":  self._token,
            "X-Api-Resource-Id": self._resource,
            "X-Api-Connect-Id":  conn_id,
        }
        t_conn = time.time()
        ws = await websockets.connect(_WSS_URL, additional_headers=ws_headers, open_timeout=10)
        await ws.send(_frame_no_id(_EVT_START_CONN, b"{}"))
        resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
        if resp["event"] != _EVT_CONN_STARTED:
            await ws.close()
            raise RuntimeError(f"建连失败 event={resp['event']}")
        self._ws = ws
        print(f"[ArkTTS] 🔗 连接建立完成 ({time.time()-t_conn:.2f}s)，后续请求直接复用")

    async def prewarm(self) -> None:
        async with self._ws_lock:
            await self._ensure_connected()

    async def text_to_speech_streaming(
        self,
        text: str,
        emotion: str = "neutral",
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        WebSocket 双向流式 TTS，复用持久连接，首包延迟 ~200ms。
        与 ZipVoiceTTS.text_to_speech_streaming 接口完全兼容。

        Yields:
            float32 numpy array, 24kHz
        """
        t0 = time.time()
        first_chunk = True

        async with self._ws_lock:   # 同一连接不支持并发 session
            for attempt in range(2):
                try:
                    await self._ensure_connected()
                    session_id = str(uuid.uuid4())
                    print(f"[ArkTTS] 🎵 流式合成 text_chars={len(text)}")

                    # StartSession
                    session_meta = {
                        "user": {"uid": "mmse_screening"},
                        "req_params": {
                            "text": text,
                            "speaker": self._voice,
                            "audio_params": {
                                "format": "pcm",
                                "sample_rate": _SAMPLE_RATE,
                            },
                        },
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
                            break

                        elif evt == _EVT_SESSION_FAILED or resp["error"]:
                            raise RuntimeError("TTS Session 失败")

                    return  # 成功，退出重试循环

                except Exception as e:
                    self._ws = None   # 连接断了，下次重建
                    if attempt == 0:
                        print(f"[ArkTTS] ⚠️ 连接异常，重试一次: {type(e).__name__}")
                    else:
                        print(f"[ArkTTS] ❌ 失败: {type(e).__name__}")
                        raise

    async def close(self):
        """关闭持久连接。"""
        if self._ws:
            try:
                await self._ws.send(_frame_no_id(_EVT_FINISH_CONN, b"{}"))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None


# 兼容别名
VoiceTTS = ArkTTS
