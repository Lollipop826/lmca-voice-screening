"""
火山引擎 BigASR 大模型流式语音识别（替换 SenseVoice 本地模型）

协议文档: wss://openspeech.bytedance.com/api/v3/sauc/bigmodel
注意：BigASR 协议与 TTS 完全不同，使用 sequence number 而非 event number。

环境变量：
  VOLC_API_KEY       - 新版 API Key（优先使用）
  VOLC_APP_ID        - 旧版 APP ID
  VOLC_ACCESS_TOKEN  - 旧版 Access Token（与 TTS 共用同一套凭证）

接口与 SenseVoice 返回格式保持一致：
  {"text": str, "emotion": "neutral", "language": "zh", "event": "Speech"}
"""
import asyncio
import gzip
import json
import os
import struct
import uuid
import time
import inspect
from collections.abc import Callable
import numpy as np
import websockets
from websockets.exceptions import WebSocketException

_ARK_ASR_MODE_ALIASES = {
    "stream": "bigmodel",
    "bigmodel": "bigmodel",
    "nostream": "bigmodel_nostream",
    "bigmodel_nostream": "bigmodel_nostream",
    "async": "bigmodel_async",
    "bigmodel_async": "bigmodel_async",
}
_ARK_ASR_MODE = _ARK_ASR_MODE_ALIASES.get(os.getenv("ARK_ASR_MODE", "stream").strip().lower(), "bigmodel")
_ARK_ASR_ENDPOINTS = {
    "bigmodel": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
    "bigmodel_nostream": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream",
    "bigmodel_async": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
}
_WSS_URL     = _ARK_ASR_ENDPOINTS[_ARK_ASR_MODE]
_RESOURCE_ID = os.getenv("ARK_ASR_RESOURCE_ID", "volc.seedasr.sauc.duration")
_OPEN_TIMEOUT = float(os.getenv("ARK_ASR_OPEN_TIMEOUT", "10"))
_RECV_TIMEOUT = float(os.getenv("ARK_ASR_RECV_TIMEOUT", "15"))
_MAX_RETRIES = max(1, int(os.getenv("ARK_ASR_MAX_RETRIES", "2")))
_RETRY_DELAY = max(0.0, float(os.getenv("ARK_ASR_RETRY_DELAY", "0.6")))


class ArkASRError(RuntimeError):
    pass


class ArkASRTemporaryError(ArkASRError):
    pass


def ark_asr_streaming_supported() -> bool:
    return _ARK_ASR_MODE in {"bigmodel", "bigmodel_async"}


def _asr_headers() -> dict[str, str]:
    headers = {
        "X-Api-Resource-Id": _RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    api_key = os.getenv("VOLC_API_KEY", "").strip()
    if api_key:
        headers["X-Api-Key"] = api_key
        return headers
    app_id = os.getenv("VOLC_APP_ID", "").strip()
    token = os.getenv("VOLC_ACCESS_TOKEN", "").strip()
    if not app_id or not token:
        raise ValueError(
            "未配置 VOLC_API_KEY 或 VOLC_APP_ID / VOLC_ACCESS_TOKEN"
        )
    headers["X-Api-App-Key"] = app_id
    headers["X-Api-Access-Key"] = token
    return headers

# ── Binary protocol ───────────────────────────────────────────────────────────
# Byte 1: (msg_type << 4) | flags
#   msg_type: 0b0001=FullClientReq, 0b0010=AudioOnlyReq, 0b1001=FullServerResp, 0b1111=Error
#   flags:    0b0000=no-seq, 0b0001=pos-seq, 0b0010=last-no-seq, 0b0011=last-with-seq
# Byte 2: (serialization << 4) | compression
#   serialization: 0b0000=raw, 0b0001=JSON
#   compression:   0b0000=none, 0b0001=gzip


def _full_client_request(payload_json: bytes, sequence: int = 1) -> bytes:
    payload = gzip.compress(payload_json)
    hdr = bytes([0x11, 0x11, 0x11, 0x00])
    return (
        hdr
        + struct.pack(">i", abs(sequence))
        + struct.pack(">I", len(payload))
        + payload
    )


def _audio_request(audio: bytes, sequence: int, last: bool = False) -> bytes:
    payload = gzip.compress(audio)
    flags = 0x03 if last else 0x01
    hdr = bytes([0x11, 0x20 | flags, 0x01, 0x00])
    sequence = abs(sequence)
    if last:
        sequence = -sequence
    return (
        hdr
        + struct.pack(">i", sequence)
        + struct.pack(">I", len(payload))
        + payload
    )


def _parse_server_response(data: bytes) -> dict:
    """解析服务端响应帧"""
    msg_type = (data[1] >> 4) & 0x0F
    flags    = data[1] & 0x0F
    serializ = (data[2] >> 4) & 0x0F
    compress = data[2] & 0x0F

    # Error frame: [hdr][4B error_code][4B msg_len][msg]
    if msg_type == 0x0F:
        error_code = struct.unpack(">I", data[4:8])[0]
        msg_len    = struct.unpack(">I", data[8:12])[0]
        msg = data[12:12+msg_len].decode("utf-8", errors="replace")
        return {"is_last": True, "payload": None, "error": f"code={error_code}: {msg}"}

    pos = 4
    # flags bit 0: has sequence number
    seq = None
    if flags & 0x01:
        seq = struct.unpack(">i", data[pos:pos+4])[0]
        pos += 4

    # flags bit 1: is last packet
    is_last = bool(flags & 0x02) or (seq is not None and seq < 0)

    payload_size = struct.unpack(">I", data[pos:pos+4])[0]
    pos += 4
    raw = data[pos:pos+payload_size]

    if compress == 0x01 and raw:
        raw = gzip.decompress(raw)

    payload = None
    if serializ == 0x01 and raw:
        try:
            payload = json.loads(raw)
        except Exception:
            pass

    return {"is_last": is_last, "payload": payload, "error": None, "seq": seq}


def _to_pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _extract_text_from_payload(payload) -> str:
    """Extract one cumulative transcript without mixing it with utterance segments."""
    if not payload:
        return ""

    def _join_unique(values) -> str:
        unique = []
        for value in values:
            value = str(value or "").strip()
            if value and value not in unique:
                unique.append(value)
        return "".join(unique)

    def _extract(value) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return _join_unique(_extract(item) for item in value)
        if isinstance(value, dict):
            for key in ("text", "utterance", "sentence"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            for key in ("utterances", "segments", "results", "result", "additions"):
                if key in value:
                    text = _extract(value.get(key))
                    if text:
                        return text
        return ""

    if isinstance(payload, dict) and "result" in payload:
        result_text = _extract(payload.get("result"))
        if result_text:
            return result_text
    return _extract(payload)


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}ms"


def _header_get(headers, name: str) -> str:
    if headers is None:
        return ""
    return headers.get(name) or headers.get(name.lower()) or ""


def _extract_response_headers(ws):
    response_headers = getattr(ws, "response_headers", None)
    if response_headers is not None:
        return response_headers
    response = getattr(ws, "response", None)
    if response is not None:
        return getattr(response, "headers", None)
    return None


class ArkASRStreamingSession:
    """One BigASR stream for a VAD-delimited user utterance."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        on_result: Callable[[str, bool], object] | None = None,
        ws_connect=None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._on_result = on_result
        self._ws_connect = ws_connect or websockets.connect
        self._ws = None
        self._receiver = None
        self._finished = False
        self._text = ""
        self._next_sequence = 1

    @property
    def text(self) -> str:
        return self._text

    async def start(self) -> None:
        if self._ws is not None:
            return
        if not ark_asr_streaming_supported():
            raise ArkASRError(
                "实时识别要求 ARK_ASR_MODE=bigmodel 或 bigmodel_async"
            )

        self._ws = await self._ws_connect(
            _WSS_URL,
            additional_headers=_asr_headers(),
            open_timeout=_OPEN_TIMEOUT,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        )
        config = {
            "user": {"uid": "mmse_screening"},
            "audio": {
                "format": "pcm",
                "rate": self.sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
            },
        }
        await self._ws.send(
            _full_client_request(
                json.dumps(config, ensure_ascii=False).encode(),
                sequence=self._next_sequence,
            )
        )
        self._next_sequence += 1
        self._receiver = asyncio.create_task(self._receive())

    async def feed(self, audio: np.ndarray) -> None:
        if self._finished:
            return
        if self._ws is None:
            raise ArkASRError("BigASR 流式连接尚未启动")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            await self._ws.send(
                _audio_request(
                    _to_pcm_bytes(samples),
                    sequence=self._next_sequence,
                )
            )
            self._next_sequence += 1

    async def finish(self) -> str:
        if self._finished:
            return self._text
        self._finished = True
        if self._ws is None:
            return self._text
        try:
            await self._ws.send(
                _audio_request(
                    b"",
                    sequence=self._next_sequence,
                    last=True,
                )
            )
            if self._receiver is not None:
                await asyncio.wait_for(self._receiver, _RECV_TIMEOUT)
            return self._text
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        receiver, self._receiver = self._receiver, None
        if receiver is not None and not receiver.done():
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _receive(self) -> None:
        while True:
            data = await asyncio.wait_for(self._ws.recv(), _RECV_TIMEOUT)
            response = _parse_server_response(data)
            if response["error"]:
                raise ArkASRError("[ArkASR] 服务端识别失败")
            text = _extract_text_from_payload(response["payload"])
            if text:
                self._text = text
                if self._on_result is not None:
                    result = self._on_result(text, response["is_last"])
                    if inspect.isawaitable(result):
                        await result
            if response["is_last"]:
                return


def _print_timing_summary(status: str, attempt: int, max_retries: int, metrics: dict, text: str = "", error: str = ""):
    connect_ms = _elapsed_ms(metrics.get("connect_started_at"), metrics.get("connected_at"))
    send_ms = _elapsed_ms(metrics.get("send_started_at"), metrics.get("send_done_at"))
    first_packet_ms = _elapsed_ms(metrics.get("connected_at"), metrics.get("first_packet_at"))
    final_result_ms = _elapsed_ms(metrics.get("connected_at"), metrics.get("final_packet_at"))
    total_ms = _elapsed_ms(metrics.get("attempt_started_at"), metrics.get("attempt_finished_at"))
    text_length = len((text or "").strip())
    print(
        "[ArkASR][Timing] "
        f"status={status} | attempt={attempt}/{max_retries} | "
        f"mode={metrics.get('mode', _ARK_ASR_MODE)} | resource={metrics.get('resource_id', _RESOURCE_ID)} | "
        f"audio={metrics.get('audio_seconds', 0.0):.2f}s | chunks={metrics.get('chunk_count', 0)} | "
        f"connect={_fmt_ms(connect_ms)} | send={_fmt_ms(send_ms)} | "
        f"first_packet={_fmt_ms(first_packet_ms)} | final_result={_fmt_ms(final_result_ms)} | total={_fmt_ms(total_ms)}"
        + (f" | logid={metrics.get('logid')}" if metrics.get('logid') else "")
        + f" | text_chars={text_length}"
        + (f" | error_type={error}" if error else "")
    )


# ── 预热连接（可选，ARK_ASR_PREWARM=true 开启）──────────────────────────
# BigASR 每轮识别都要新建 WebSocket（TLS+握手+鉴权，实测 0.5-1s）。预热在上一轮
# 识别结束时后台建好下一条连接（只连、不发 config），下一轮直接复用，省掉握手耗时。
# 连接过期或被服务端关闭时无缝回退到当场新建，绝不降低成功率；复用/建连连续失败达
# 阈值后自动禁用，避免在不支持闲置连接的环境里空耗。
_ARK_ASR_PREWARM = os.getenv("ARK_ASR_PREWARM", "false").strip().lower() == "true"
_WARM_TTL = float(os.getenv("ARK_ASR_PREWARM_TTL", "10"))
_WARM_MAX_FAILS = 3

_warm_socket = None
_warm_lock = asyncio.Lock()
_warm_refill_task = None
_warm_fail_streak = 0
_warm_disabled = False


class _ArkASRWarmSocket:
    """已完成握手、尚未发送 config 的 BigASR 连接，一次性复用。"""
    __slots__ = ("_ws", "_created_at", "_taken")

    def __init__(self, ws):
        self._ws = ws
        self._created_at = time.perf_counter()
        self._taken = False

    def is_usable(self) -> bool:
        return (
            not self._taken
            and self._ws is not None
            and (time.perf_counter() - self._created_at) < _WARM_TTL
        )

    def take(self):
        self._taken = True
        ws, self._ws = self._ws, None
        return ws

    async def aclose(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


async def _open_warm_socket():
    try:
        headers = _asr_headers()
    except ValueError:
        return None
    ws = await websockets.connect(
        _WSS_URL,
        additional_headers=headers,
        open_timeout=_OPEN_TIMEOUT,
        close_timeout=5,
        ping_interval=20,
        ping_timeout=20,
        max_size=8 * 1024 * 1024,
    )
    return _ArkASRWarmSocket(ws)


async def _get_warm_socket():
    global _warm_socket
    async with _warm_lock:
        sock, _warm_socket = _warm_socket, None
    if sock is None:
        return None
    if sock.is_usable():
        return sock
    await sock.aclose()
    return None


async def _refill_warm_socket():
    global _warm_socket, _warm_fail_streak, _warm_disabled
    try:
        sock = await _open_warm_socket()
    except Exception:
        _warm_fail_streak += 1
        if _warm_fail_streak >= _WARM_MAX_FAILS:
            _warm_disabled = True
            print(f"[ArkASR] ⚠️ 预热连续失败{_warm_fail_streak}次，自动禁用预热（回退每轮新建）")
        return
    if sock is None:
        return
    async with _warm_lock:
        old, _warm_socket = _warm_socket, sock
    if old is not None:
        await old.aclose()


def _schedule_warm_refill():
    global _warm_refill_task
    if _warm_disabled:
        return
    if _warm_refill_task is not None and not _warm_refill_task.done():
        return
    try:
        _warm_refill_task = asyncio.create_task(_refill_warm_socket())
    except RuntimeError:
        pass  # 没有运行中的事件循环，跳过预热


def _new_attempt_metrics(audio, sample_rate, chunks):
    return {
        "attempt_started_at": time.perf_counter(),
        "connect_started_at": None,
        "connected_at": None,
        "send_started_at": None,
        "send_done_at": None,
        "first_packet_at": None,
        "final_packet_at": None,
        "attempt_finished_at": None,
        "audio_seconds": len(audio) / sample_rate,
        "chunk_count": len(chunks),
        "mode": _ARK_ASR_MODE,
        "resource_id": _RESOURCE_ID,
        "logid": "",
        "connect_id": str(uuid.uuid4()),
    }


async def _exchange_over_ws(ws, config, chunks, attempt_metrics):
    """在已连接的 ws 上完成一次 BigASR 识别，返回文字和未解析载荷标记。"""
    response_headers = _extract_response_headers(ws)
    attempt_metrics["logid"] = _header_get(response_headers, "X-Tt-Logid")
    text = ""
    has_unparsed_payload = False

    async def _send_all():
        attempt_metrics["send_started_at"] = time.perf_counter()
        await ws.send(_full_client_request(json.dumps(config, ensure_ascii=False).encode()))
        for i, chunk in enumerate(chunks):
            await ws.send(
                _audio_request(
                    chunk,
                    sequence=i + 2,
                    last=(i == len(chunks) - 1),
                )
            )
        if not chunks:
            await ws.send(_audio_request(b"", sequence=2, last=True))
        attempt_metrics["send_done_at"] = time.perf_counter()

    async def _recv_all():
        nonlocal text, has_unparsed_payload
        while True:
            data = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)
            if attempt_metrics["first_packet_at"] is None:
                attempt_metrics["first_packet_at"] = time.perf_counter()
            resp = _parse_server_response(data)
            if resp["error"]:
                raise ArkASRError(f"[ArkASR] 错误: {resp['error']}")
            if resp["payload"]:
                t = _extract_text_from_payload(resp["payload"])
                if t:
                    text = t
                else:
                    has_unparsed_payload = True
            attempt_metrics["final_packet_at"] = time.perf_counter()
            if resp["is_last"]:
                break

    await asyncio.gather(_send_all(), _recv_all())
    return text, has_unparsed_payload


async def _try_warm_recognize(warm, audio, sample_rate, config, chunks, t0):
    """走预热连接完成识别；成功返回结果 dict，失败返回 None 让调用方回退新建。"""
    global _warm_fail_streak, _warm_disabled
    wm = _new_attempt_metrics(audio, sample_rate, chunks)
    wm["connect_started_at"] = wm["connected_at"] = time.perf_counter()
    ws = warm.take()
    try:
        text, has_unparsed_payload = await _exchange_over_ws(ws, config, chunks, wm)
        wm["attempt_finished_at"] = time.perf_counter()
        _print_timing_summary("success_warm", 1, _MAX_RETRIES, wm, text=text)
        if not text and has_unparsed_payload:
            print("[ArkASR] ⚠️ 预热返回成功但未提取到文本，unparsed_payload=true")
        _warm_fail_streak = 0
        elapsed = time.perf_counter() - t0
        print(f"[ArkASR] ✅(预热) text_chars={len(text)} ({elapsed:.2f}s)")
        return {"text": text, "emotion": "neutral", "language": "zh", "event": "Speech"}
    except Exception as exc:
        _warm_fail_streak += 1
        note = ""
        if _warm_fail_streak >= _WARM_MAX_FAILS:
            _warm_disabled = True
            note = "（连续失败，自动禁用预热）"
        print(f"[ArkASR] ⚠️ 预热连接识别失败，回退新建{note}: {type(exc).__name__}")
        return None
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def ark_asr_recognize(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> dict:
    """
    调用火山引擎 BigASR (WebSocket) 识别一段完整音频。

    Returns:
        {"text": str, "emotion": "neutral", "language": "zh", "event": "Speech"}
    """
    ws_headers = _asr_headers()

    t0 = time.perf_counter()
    pcm = _to_pcm_bytes(audio)

    config = {
        "user": {"uid": "mmse_screening"},
        "audio": {
            "format": "pcm",
            "rate":   sample_rate,
            "bits":   16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
        },
    }

    chunk_bytes = sample_rate // 5 * 2   # 200ms per chunk (推荐值)
    chunks = [pcm[i:i+chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
    total_frames = 1 + len(chunks)   # 1 config frame + N audio frames

    if _ARK_ASR_PREWARM and not _warm_disabled:
        warm = await _get_warm_socket()
        if warm is not None:
            result = await _try_warm_recognize(warm, audio, sample_rate, config, chunks, t0)
            _schedule_warm_refill()
            if result is not None:
                return result

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        text = ""
        has_unparsed_payload = False
        attempt_metrics = {
            "attempt_started_at": time.perf_counter(),
            "connect_started_at": None,
            "connected_at": None,
            "send_started_at": None,
            "send_done_at": None,
            "first_packet_at": None,
            "final_packet_at": None,
            "attempt_finished_at": None,
            "audio_seconds": len(audio) / sample_rate,
            "chunk_count": len(chunks),
            "mode": _ARK_ASR_MODE,
            "resource_id": _RESOURCE_ID,
            "logid": "",
            "connect_id": str(uuid.uuid4()),
        }
        ws_headers["X-Api-Connect-Id"] = attempt_metrics["connect_id"]
        try:
            attempt_metrics["connect_started_at"] = time.perf_counter()
            async with websockets.connect(
                _WSS_URL,
                additional_headers=ws_headers,
                open_timeout=_OPEN_TIMEOUT,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            ) as ws:
                attempt_metrics["connected_at"] = time.perf_counter()
                text, has_unparsed_payload = await _exchange_over_ws(
                    ws,
                    config,
                    chunks,
                    attempt_metrics,
                )

            attempt_metrics["attempt_finished_at"] = time.perf_counter()
            _print_timing_summary("success", attempt, _MAX_RETRIES, attempt_metrics, text=text)
            if not text and has_unparsed_payload:
                print("[ArkASR] ⚠️ 返回成功但未提取到文本，unparsed_payload=true")
            elapsed = time.perf_counter() - t0
            print(
                f"[ArkASR] ✅ text_chars={len(text)} "
                f"({elapsed:.2f}s, attempt={attempt})"
            )
            if _ARK_ASR_PREWARM:
                _schedule_warm_refill()
            return {"text": text, "emotion": "neutral", "language": "zh", "event": "Speech"}
        except ArkASRError as e:
            attempt_metrics["attempt_finished_at"] = time.perf_counter()
            _print_timing_summary("server_error", attempt, _MAX_RETRIES, attempt_metrics, text=text, error=type(e).__name__)
            raise
        except (asyncio.TimeoutError, TimeoutError, OSError, WebSocketException) as e:
            last_error = e
            attempt_metrics["attempt_finished_at"] = time.perf_counter()
            attempt_elapsed = time.perf_counter() - attempt_metrics["attempt_started_at"]
            _print_timing_summary("retryable_error", attempt, _MAX_RETRIES, attempt_metrics, text=text, error=type(e).__name__)
            print(f"[ArkASR] ⚠️ attempt {attempt}/{_MAX_RETRIES} 失败: {type(e).__name__} (耗时 {attempt_elapsed:.2f}s)")
            if attempt >= _MAX_RETRIES:
                break
            await asyncio.sleep(_RETRY_DELAY * attempt)
        except Exception as e:
            attempt_metrics["attempt_finished_at"] = time.perf_counter()
            _print_timing_summary("unexpected_error", attempt, _MAX_RETRIES, attempt_metrics, text=text, error=type(e).__name__)
            raise ArkASRError(f"BigASR识别失败: {type(e).__name__}") from e

    raise ArkASRTemporaryError(
        f"BigASR连接失败（重试{_MAX_RETRIES}次，open_timeout={_OPEN_TIMEOUT}s，recv_timeout={_RECV_TIMEOUT}s）: {type(last_error).__name__ if last_error else 'Unknown'}"
    ) from last_error
