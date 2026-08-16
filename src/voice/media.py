from __future__ import annotations

import asyncio
import base64
import fractions
import json
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

try:
    import soxr
except Exception:
    soxr = None

try:
    from aiortc import (
        MediaStreamTrack,
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame

    AIORTC_IMPORT_ERROR = None
except Exception as exc:
    MediaStreamTrack = None
    MediaStreamError = Exception
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AudioFrame = None
    AIORTC_IMPORT_ERROR = exc


_OUTPUT_IDENTITY_FIELDS = (
    "session_id",
    "turn_id",
    "generation",
    "playback_id",
)


def _with_output_identity(source: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Keep output correlation fields when an RTP notification is reduced."""
    for key in _OUTPUT_IDENTITY_FIELDS:
        if key in source:
            payload[key] = source[key]
    return payload


def build_rtc_configuration(ice_server_entries: list[dict]):
    """Build an aiortc configuration from browser-style ICE server entries."""
    if RTCConfiguration is None or RTCIceServer is None:
        return None

    ice_servers = []
    for entry in ice_server_entries:
        kwargs = {"urls": entry.get("urls")}
        if entry.get("username"):
            kwargs["username"] = entry["username"]
        if entry.get("credential"):
            kwargs["credential"] = entry["credential"]
        ice_servers.append(RTCIceServer(**kwargs))
    return (
        RTCConfiguration(iceServers=ice_servers)
        if ice_servers
        else RTCConfiguration()
    )


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float audio, preferring anti-aliased soxr conversion."""
    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_array.size == 0 or src_sr == dst_sr:
        return audio_array

    if soxr is not None and audio_array.size > 1:
        try:
            return soxr.resample(
                audio_array,
                src_sr,
                dst_sr,
                quality="HQ",
            ).astype(np.float32, copy=False)
        except Exception:
            pass

    if audio_array.size == 1:
        target_len = max(1, int(round(dst_sr / max(src_sr, 1))))
        return np.repeat(audio_array, target_len).astype(np.float32)

    target_len = max(
        1,
        int(round(audio_array.size * float(dst_sr) / float(src_sr))),
    )
    src_positions = np.linspace(
        0,
        audio_array.size - 1,
        num=audio_array.size,
        dtype=np.float64,
    )
    dst_positions = np.linspace(
        0,
        audio_array.size - 1,
        num=target_len,
        dtype=np.float64,
    )
    return np.interp(
        dst_positions,
        src_positions,
        audio_array,
    ).astype(np.float32)


def audio_frame_to_mono_float(frame) -> tuple[np.ndarray, int]:
    """Convert a decoded WebRTC audio frame into clipped mono float32 PCM."""
    samples = np.asarray(frame.to_ndarray())
    original_dtype = samples.dtype

    layout = getattr(frame, "layout", None)
    layout_name = getattr(layout, "name", "") or ""
    layout_channels = getattr(layout, "channels", None)
    if layout_channels is not None:
        try:
            channel_count = len(layout_channels)
        except TypeError:
            channel_count = int(layout_channels)
    else:
        channel_count = 2 if "stereo" in layout_name else 1
    channel_count = max(1, int(channel_count or 1))

    if samples.ndim == 2:
        if samples.shape[0] == channel_count and samples.shape[1] > 1:
            audio = samples.mean(axis=0)
        elif (
            samples.shape[0] == 1
            and channel_count > 1
            and samples.shape[1] % channel_count == 0
        ):
            audio = samples.reshape(-1, channel_count).mean(axis=1)
        else:
            audio = samples.mean(axis=0)
    elif channel_count > 1 and samples.size % channel_count == 0:
        audio = samples.reshape(-1, channel_count).mean(axis=1)
    else:
        audio = samples
    audio = audio.reshape(-1)

    if np.issubdtype(original_dtype, np.floating):
        audio_float = audio.astype(np.float32, copy=False)
    elif original_dtype == np.uint8:
        audio_float = (audio.astype(np.float32) - 128.0) / 128.0
    elif original_dtype == np.int32:
        audio_float = audio.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio.astype(np.float32) / 32768.0

    sample_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
    return np.clip(audio_float, -1.0, 1.0), sample_rate


if MediaStreamTrack is not None:

    class WebRTCAssistantAudioTrack(MediaStreamTrack):
        """Server-side assistant audio track for WebRTC RTP downlink."""

        kind = "audio"

        def __init__(
            self,
            sample_rate: int = 48000,
            frame_samples: int = 960,
        ) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.frame_samples = frame_samples
            self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=240)
            self._buffer = np.zeros(0, dtype=np.float32)
            self._timestamp = 0
            self._start_time: float | None = None
            self._tts_resampler = None
            self._tts_resampler_src_sr = 0

        def enqueue_audio(
            self,
            audio: np.ndarray,
            source_sample_rate: int = 24000,
        ) -> None:
            audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio_array.size == 0:
                return

            source_rate = int(source_sample_rate or self.sample_rate)
            if source_rate != self.sample_rate:
                if soxr is not None:
                    if (
                        self._tts_resampler is None
                        or self._tts_resampler_src_sr != source_rate
                    ):
                        self._tts_resampler = soxr.ResampleStream(
                            source_rate,
                            self.sample_rate,
                            1,
                            dtype="float32",
                            quality="HQ",
                        )
                        self._tts_resampler_src_sr = source_rate
                    audio_array = self._tts_resampler.resample_chunk(
                        audio_array
                    ).reshape(-1)
                    if audio_array.size == 0:
                        return
                else:
                    audio_array = resample_audio(
                        audio_array,
                        source_rate,
                        self.sample_rate,
                    )

            audio_array = np.clip(
                audio_array,
                -1.0,
                1.0,
            ).astype(np.float32, copy=False)
            try:
                self._queue.put_nowait(audio_array)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(audio_array)
                except Exception:
                    pass

        def clear(self) -> None:
            self._buffer = np.zeros(0, dtype=np.float32)
            while True:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        async def recv(self):
            if self.readyState != "live":
                raise MediaStreamError

            if self._start_time is None:
                self._start_time = time.time()
                self._timestamp = 0
            else:
                self._timestamp += self.frame_samples
                wait = (
                    self._start_time
                    + (self._timestamp / self.sample_rate)
                    - time.time()
                )
                if wait > 0:
                    await asyncio.sleep(wait)

            while self._buffer.size < self.frame_samples:
                try:
                    chunk = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._buffer = np.concatenate(
                    [self._buffer, chunk]
                ).astype(np.float32, copy=False)

            if self._buffer.size >= self.frame_samples:
                frame_audio = self._buffer[: self.frame_samples]
                self._buffer = self._buffer[self.frame_samples :]
            elif self._buffer.size > 0:
                missing = self.frame_samples - self._buffer.size
                frame_audio = np.concatenate(
                    [self._buffer, np.zeros(missing, dtype=np.float32)]
                )
                self._buffer = np.zeros(0, dtype=np.float32)
            else:
                frame_audio = np.zeros(self.frame_samples, dtype=np.float32)

            pcm = (
                np.clip(frame_audio, -1.0, 1.0) * 32767.0
            ).astype(np.int16)
            frame = AudioFrame(
                format="s16",
                layout="mono",
                samples=self.frame_samples,
            )
            frame.planes[0].update(pcm.tobytes())
            frame.pts = self._timestamp
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame

else:
    WebRTCAssistantAudioTrack = None


class WebRTCVoiceConnection:
    """Adapt WebRTC RTP/DataChannel transport to the WebSocket-like contract."""

    def __init__(
        self,
        cookies: dict[str, str],
        peer_id: str,
        assistant_track=None,
        *,
        monotonic_factory: Callable[[], float] = time.monotonic,
        data_channel_stall_timeout_s: float = 4.0,
        data_channel_stall_min_bytes: int = 1024,
        logger=print,
    ) -> None:
        self.cookies = dict(cookies or {})
        self.peer_id = peer_id
        self.transport_name = "WebRTC"
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._control_channel = None
        self._audio_channel = None
        self.assistant_track = assistant_track
        self._log = logger
        self._closed = False
        self._incoming_audio_buffer = np.zeros(0, dtype=np.float32)
        self._mic_resampler = None
        self._mic_resampler_src_sr = 0
        self._rtp_tts_chunk_sent = False
        self._monotonic = monotonic_factory
        self._dc_stall_timeout_s = float(data_channel_stall_timeout_s)
        self._dc_stall_min_bytes = int(data_channel_stall_min_bytes)
        self._dc_nonzero_since: float | None = None
        self._dc_last_buffered_amount = 0
        self._dc_stall_reported = False
        self._dc_stall_handler: Callable[[str], Any] | None = None

    async def accept(self) -> None:
        return None

    async def close(self, code: int | None = None) -> None:
        self.disconnect()

    @property
    def closed(self) -> bool:
        return self._closed

    def attach_data_channel(self, channel) -> None:
        label = getattr(channel, "label", "") or ""
        if label == "voice-audio":
            self._audio_channel = channel
        else:
            self._control_channel = channel
            self._reset_data_channel_health()

        @channel.on("message")
        def on_message(message):
            self.feed_message(message)

        @channel.on("close")
        def on_close():
            if channel is self._control_channel:
                self.disconnect()

    def is_control_open(self) -> bool:
        return (
            self._control_channel is not None
            and getattr(self._control_channel, "readyState", "") == "open"
        )

    def set_data_channel_stall_handler(
        self,
        handler: Callable[[str], Any] | None,
    ) -> None:
        self._dc_stall_handler = handler

    def _reset_data_channel_health(self) -> None:
        self._dc_nonzero_since = None
        self._dc_last_buffered_amount = 0
        self._dc_stall_reported = False

    def _check_data_channel_health(self, channel) -> None:
        try:
            buffered = max(0, int(getattr(channel, "bufferedAmount", 0) or 0))
        except (TypeError, ValueError):
            return

        now = self._monotonic()
        if buffered <= 0:
            self._reset_data_channel_health()
            return

        if (
            self._dc_nonzero_since is None
            or buffered < self._dc_last_buffered_amount
        ):
            self._dc_nonzero_since = now

        self._dc_last_buffered_amount = buffered
        stalled_for = now - self._dc_nonzero_since
        if (
            self._dc_stall_reported
            or buffered < self._dc_stall_min_bytes
            or stalled_for < self._dc_stall_timeout_s
        ):
            return

        self._dc_stall_reported = True
        reason = (
            "DataChannel send buffer stalled: "
            f"{buffered} bytes for {stalled_for:.1f}s"
        )
        self._log(f"[WebRTC] ⚠️ {reason}")
        if self._dc_stall_handler is not None:
            self._dc_stall_handler(reason)
        else:
            self.disconnect()
        raise RuntimeError(reason)

    def _record_data_channel_buffered_amount(self, channel) -> None:
        try:
            self._dc_last_buffered_amount = max(
                0,
                int(getattr(channel, "bufferedAmount", 0) or 0),
            )
        except (TypeError, ValueError):
            pass

    def feed_message(
        self,
        message: str | bytes | bytearray | memoryview,
    ) -> None:
        if self._closed:
            return
        if isinstance(message, str):
            payload = {"type": "websocket.receive", "text": message}
        else:
            payload = {"type": "websocket.receive", "bytes": bytes(message)}
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def feed_audio_float(
        self,
        audio_float: np.ndarray,
        sample_rate: int = 16000,
    ) -> None:
        if self._closed:
            return
        audio = np.asarray(audio_float, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return

        source_rate = int(sample_rate or 16000)
        if source_rate != 16000:
            if soxr is not None:
                if (
                    self._mic_resampler is None
                    or self._mic_resampler_src_sr != source_rate
                ):
                    self._mic_resampler = soxr.ResampleStream(
                        source_rate,
                        16000,
                        1,
                        dtype="float32",
                        quality="HQ",
                    )
                    self._mic_resampler_src_sr = source_rate
                audio = self._mic_resampler.resample_chunk(audio).reshape(-1)
                if audio.size == 0:
                    return
            else:
                audio = resample_audio(audio, source_rate, 16000)

        self._incoming_audio_buffer = np.concatenate(
            [self._incoming_audio_buffer, audio]
        ).astype(np.float32, copy=False)
        # Match SoulX's native frontend cadence: 512 samples = 32 ms at 16 kHz.
        frame_samples = 512
        while self._incoming_audio_buffer.size >= frame_samples:
            chunk = self._incoming_audio_buffer[:frame_samples]
            self._incoming_audio_buffer = self._incoming_audio_buffer[
                frame_samples:
            ]
            pcm = (
                np.clip(chunk, -1.0, 1.0) * 32767.0
            ).astype(np.int16)
            self.feed_message(pcm.tobytes())

    def disconnect(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.assistant_track is not None:
            try:
                self.assistant_track.clear()
            except Exception:
                pass
        try:
            self._queue.put_nowait({"type": "websocket.disconnect"})
        except Exception:
            pass

    async def receive(self) -> dict[str, Any]:
        return await self._queue.get()

    async def send_json(self, data: dict) -> None:
        if self._closed:
            raise RuntimeError("WebRTC DataChannel is closed")
        channel = self._control_channel
        if channel is None or getattr(channel, "readyState", "") != "open":
            raise RuntimeError("WebRTC control DataChannel is not open")
        self._check_data_channel_health(channel)

        payload = dict(data or {})
        payload_type = payload.get("type")
        if self.assistant_track is not None:
            if payload_type == "tts_start":
                self._rtp_tts_chunk_sent = False
                self.assistant_track.clear()
                payload["media_transport"] = "webrtc_rtp"
            elif payload_type in {"tts_chunk", "tts_audio"}:
                self._enqueue_tts_audio(payload)
                if self._rtp_tts_chunk_sent:
                    return
                self._rtp_tts_chunk_sent = True
                payload = _with_output_identity(
                    payload,
                    {
                        "type": payload_type,
                        "sample_rate": payload.get("sample_rate") or 24000,
                        "media_transport": "webrtc_rtp",
                    },
                )
            elif payload_type in {
                "tts_end",
                "tts_stop",
                "stop_tts",
                "interrupt",
            }:
                if payload_type in {"tts_stop", "stop_tts", "interrupt"}:
                    self.assistant_track.clear()
                payload["media_transport"] = "webrtc_rtp"

        serialized = json.dumps(payload, ensure_ascii=False)
        channel.send(serialized)
        self._record_data_channel_buffered_amount(channel)
        self._log_data_channel_send(
            channel,
            payload_type,
            serialized,
        )

    def _enqueue_tts_audio(self, payload: dict) -> None:
        audio_base64 = payload.get("chunk") or payload.get("audio")
        if not audio_base64:
            return
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_float = np.frombuffer(audio_bytes, dtype=np.float32)
            self.assistant_track.enqueue_audio(
                audio_float,
                int(payload.get("sample_rate") or 24000),
            )
        except Exception as exc:
            self._log(f"[WebRTC] ⚠️ TTS RTP 入队失败: {type(exc).__name__}")

    def _log_data_channel_send(
        self,
        channel,
        payload_type: str | None,
        serialized: str,
    ) -> None:
        if payload_type not in {
            "ai_response",
            "ai_response_chunk",
            "asr_result",
            "tts_start",
            "tts_end",
            "vad_end",
        }:
            return
        try:
            buffered = getattr(channel, "bufferedAmount", None)
            ready = getattr(channel, "readyState", "?")
            message_bytes = len(serialized.encode("utf-8"))
            tag = (
                "⚠️ buffered高"
                if buffered is not None and buffered > 64 * 1024
                else "ok"
            )
            self._log(
                f"[WebRTC-DC SEND] type={payload_type} bytes={message_bytes} "
                f"buffered={buffered} ready={ready} {tag}"
            )
        except Exception as exc:
            self._log(
                f"[WebRTC-DC SEND] type={payload_type} diag_err={exc}"
            )


class HybridMediaRegistry:
    """Bind a WebSocket-owned voice session to an optional WebRTC media peer."""

    def __init__(self) -> None:
        self._transports: dict[str, HybridVoiceTransport] = {}

    @staticmethod
    def normalize_token(token: str | None) -> str:
        value = str(token or "").strip()
        if not 16 <= len(value) <= 128:
            return ""
        if not all(char.isalnum() or char in {"-", "_"} for char in value):
            return ""
        return value

    def register(self, token: str, transport: "HybridVoiceTransport") -> None:
        normalized = self.normalize_token(token)
        if not normalized:
            raise ValueError("invalid hybrid media token")
        self._transports[normalized] = transport

    def unregister(self, token: str, transport: "HybridVoiceTransport") -> None:
        normalized = self.normalize_token(token)
        if self._transports.get(normalized) is transport:
            self._transports.pop(normalized, None)

    def get(self, token: str | None) -> "HybridVoiceTransport | None":
        return self._transports.get(self.normalize_token(token))

    def transports(self) -> list["HybridVoiceTransport"]:
        return list(self._transports.values())


class HybridVoiceTransport:
    """Use WebSocket for control while attaching WebRTC RTP audio when ready."""

    transport_name = "WebSocket+WebRTC-RTP"

    def __init__(
        self,
        websocket,
        *,
        media_token: str,
        registry: HybridMediaRegistry,
        logger=print,
    ) -> None:
        self.websocket = websocket
        self.cookies = dict(getattr(websocket, "cookies", {}) or {})
        self.media_token = registry.normalize_token(media_token)
        if not self.media_token:
            raise ValueError("invalid hybrid media token")
        self.peer_id = f"hybrid_{self.media_token[:12]}"
        self._registry = registry
        self._log = logger
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None
        self._accepted = False
        self._closed = False
        self._disconnect_queued = False
        self._incoming_audio_buffer = np.zeros(0, dtype=np.float32)
        self._mic_resampler = None
        self._mic_resampler_src_sr = 0
        self._media_peer_id = ""
        self._media_ready = False
        self._media_session = None
        self.assistant_track = None
        self._rtp_tts_chunk_sent = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def media_session(self):
        return self._media_session

    @property
    def media_ready(self) -> bool:
        return bool(
            self._media_ready
            and self._media_session is not None
            and self.assistant_track is not None
        )

    async def accept(self) -> None:
        if self._accepted:
            return
        await self.websocket.accept()
        self._accepted = True
        self._registry.register(self.media_token, self)
        self._pump_task = asyncio.create_task(self._pump_websocket())
        self._log(
            f"[Hybrid] WebSocket 控制通道已注册: {self.media_token[:12]}"
        )

    async def receive(self) -> dict[str, Any]:
        return await self._queue.get()

    async def send_json(self, data: dict) -> None:
        if self._closed:
            raise RuntimeError("Hybrid WebSocket control channel is closed")
        payload = self._route_tts_to_rtp(dict(data or {}))
        if payload is not None:
            await self.websocket.send_json(payload)

    async def close(self, code: int | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry.unregister(self.media_token, self)
        await self._close_media("control websocket closed")
        current_task = asyncio.current_task()
        if self._pump_task is not None and self._pump_task is not current_task:
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)
        try:
            await self.websocket.close(code=code or 1000)
        except Exception:
            pass
        self._queue_disconnect()

    async def _pump_websocket(self) -> None:
        try:
            while not self._closed:
                event = await self.websocket.receive()
                await self._queue.put(event)
                if event.get("type") == "websocket.disconnect":
                    self._disconnect_queued = True
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"[Hybrid] WebSocket 接收异常: {type(exc).__name__}")
            self._queue_disconnect()
        finally:
            if not self._closed:
                self._closed = True
                self._registry.unregister(self.media_token, self)
                await self._close_media("control websocket disconnected")
                self._queue_disconnect()

    def _queue_disconnect(self) -> None:
        if self._disconnect_queued:
            return
        self._disconnect_queued = True
        try:
            self._queue.put_nowait({"type": "websocket.disconnect"})
        except Exception:
            pass

    def attach_media(self, peer_id: str, assistant_track, media_session) -> None:
        self._media_peer_id = str(peer_id or "")
        self.assistant_track = assistant_track
        self._media_session = media_session
        self._media_ready = False
        self._rtp_tts_chunk_sent = False
        self._log(
            f"[Hybrid] WebRTC RTP Peer 已附着: {self._media_peer_id}"
        )

    def set_media_ready(self, peer_id: str, ready: bool) -> None:
        if str(peer_id or "") != self._media_peer_id:
            return
        self._media_ready = bool(ready)
        self._log(
            f"[Hybrid] WebRTC RTP media_ready={self._media_ready}: "
            f"{self._media_peer_id}"
        )

    def detach_media(self, peer_id: str) -> None:
        if str(peer_id or "") != self._media_peer_id:
            return
        track = self.assistant_track
        self._media_ready = False
        self._media_peer_id = ""
        self._media_session = None
        self.assistant_track = None
        self._rtp_tts_chunk_sent = False
        if track is not None:
            try:
                track.clear()
            except Exception:
                pass
        self._log("[Hybrid] WebRTC RTP Peer 已分离，音频回退 WebSocket")

    async def _close_media(self, reason: str) -> None:
        media_session = self._media_session
        if media_session is None:
            return
        try:
            await media_session.close(reason)
        except Exception as exc:
            self._log(f"[Hybrid] 关闭 WebRTC RTP Peer 失败: {type(exc).__name__}")

    def feed_audio_float(
        self,
        audio_float: np.ndarray,
        sample_rate: int = 16000,
    ) -> None:
        if self._closed or not self.media_ready:
            return
        audio = np.asarray(audio_float, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return

        source_rate = int(sample_rate or 16000)
        if source_rate != 16000:
            if soxr is not None:
                if (
                    self._mic_resampler is None
                    or self._mic_resampler_src_sr != source_rate
                ):
                    self._mic_resampler = soxr.ResampleStream(
                        source_rate,
                        16000,
                        1,
                        dtype="float32",
                        quality="HQ",
                    )
                    self._mic_resampler_src_sr = source_rate
                audio = self._mic_resampler.resample_chunk(audio).reshape(-1)
                if audio.size == 0:
                    return
            else:
                audio = resample_audio(audio, source_rate, 16000)

        self._incoming_audio_buffer = np.concatenate(
            [self._incoming_audio_buffer, audio]
        ).astype(np.float32, copy=False)
        while self._incoming_audio_buffer.size >= 512:
            chunk = self._incoming_audio_buffer[:512]
            self._incoming_audio_buffer = self._incoming_audio_buffer[512:]
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
            try:
                self._queue.put_nowait(
                    {"type": "websocket.receive", "bytes": pcm.tobytes()}
                )
            except Exception:
                return

    def _route_tts_to_rtp(self, payload: dict) -> dict | None:
        if not self.media_ready:
            return payload

        payload_type = payload.get("type")
        track = self.assistant_track
        if payload_type == "tts_start":
            self._rtp_tts_chunk_sent = False
            track.clear()
            payload["media_transport"] = "webrtc_rtp"
        elif payload_type in {"tts_chunk", "tts_audio"}:
            audio_base64 = payload.get("chunk") or payload.get("audio")
            if audio_base64:
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_float = np.frombuffer(audio_bytes, dtype=np.float32)
                    track.enqueue_audio(
                        audio_float,
                        int(payload.get("sample_rate") or 24000),
                    )
                except Exception as exc:
                    self._log(f"[Hybrid] TTS RTP 入队失败: {type(exc).__name__}")
                    return payload
            if self._rtp_tts_chunk_sent:
                return None
            self._rtp_tts_chunk_sent = True
            payload = _with_output_identity(
                payload,
                {
                    "type": payload_type,
                    "sample_rate": payload.get("sample_rate") or 24000,
                    "media_transport": "webrtc_rtp",
                },
            )
        elif payload_type in {
            "tts_end",
            "tts_stop",
            "stop_tts",
            "interrupt",
        }:
            if payload_type in {"tts_stop", "stop_tts", "interrupt"}:
                track.clear()
            payload["media_transport"] = "webrtc_rtp"
        return payload


class WebRTCPeerSession:
    """Own one aiortc peer, its media tasks and the shared voice session task."""

    def __init__(
        self,
        peer_connection,
        connection: Any,
        *,
        peer_id: str,
        session_runner: Callable[[Any], Any] | None,
        peer_registry: set,
        media_only: bool = False,
        session_description_factory=RTCSessionDescription,
        audio_frame_converter=audio_frame_to_mono_float,
        media_stream_error_type=MediaStreamError,
        audio_writer: Callable[..., Any] | None = None,
        debug_dump_enabled: bool = False,
        debug_dump_seconds: float = 8.0,
        ice_gathering_timeout_s: float = 3.0,
        logger=print,
    ) -> None:
        self.peer_connection = peer_connection
        self.connection = connection
        self.peer_id = peer_id
        self._session_runner = session_runner
        self.peer_registry = peer_registry
        self.media_only = bool(media_only)
        self._session_description_factory = session_description_factory
        self._audio_frame_converter = audio_frame_converter
        self._media_stream_error_type = media_stream_error_type
        self._audio_writer = audio_writer
        self.debug_dump_enabled = bool(debug_dump_enabled)
        self.debug_dump_seconds = float(debug_dump_seconds)
        self.ice_gathering_timeout_s = float(ice_gathering_timeout_s)
        self._log = logger
        self.session_task: asyncio.Task | None = None
        self.media_tasks: set[asyncio.Task] = set()
        self.close_task: asyncio.Task | None = None
        self._closing = False
        self._registered = False
        self._ice_gathering_done = asyncio.Event()
        if not self.media_only:
            self.connection.set_data_channel_stall_handler(
                self._on_data_channel_stall
            )

    def _on_data_channel_stall(self, reason: str) -> None:
        if self._closing:
            return
        self._log(
            f"[WebRTC] ⚠️ 控制通道停滞，主动关闭 Peer: "
            f"{self.peer_id} ({reason})"
        )
        self.close_task = asyncio.create_task(self.close(reason))

    @classmethod
    def create(
        cls,
        *,
        cookies: dict[str, str],
        peer_id: str,
        ice_servers: list[dict],
        session_runner: Callable[[Any], Any],
        peer_registry: set,
        audio_writer: Callable[..., Any] | None = None,
        debug_dump_enabled: bool = False,
        logger=print,
    ) -> "WebRTCPeerSession":
        rtc_config = build_rtc_configuration(ice_servers)
        peer_connection = (
            RTCPeerConnection(configuration=rtc_config)
            if rtc_config
            else RTCPeerConnection()
        )
        assistant_track = WebRTCAssistantAudioTrack()
        peer_connection.addTrack(assistant_track)
        connection = WebRTCVoiceConnection(
            cookies,
            peer_id=f"webrtc_{peer_id}",
            assistant_track=assistant_track,
            logger=logger,
        )
        peer_session = cls(
            peer_connection,
            connection,
            peer_id=peer_id,
            session_runner=session_runner,
            peer_registry=peer_registry,
            audio_writer=audio_writer,
            debug_dump_enabled=debug_dump_enabled,
            logger=logger,
        )
        peer_session.register()
        return peer_session

    @classmethod
    def create_media_only(
        cls,
        *,
        peer_id: str,
        ice_servers: list[dict],
        media_transport: HybridVoiceTransport,
        peer_registry: set,
        audio_writer: Callable[..., Any] | None = None,
        debug_dump_enabled: bool = False,
        logger=print,
    ) -> "WebRTCPeerSession":
        rtc_config = build_rtc_configuration(ice_servers)
        peer_connection = (
            RTCPeerConnection(configuration=rtc_config)
            if rtc_config
            else RTCPeerConnection()
        )
        assistant_track = WebRTCAssistantAudioTrack()
        peer_connection.addTrack(assistant_track)
        peer_session = cls(
            peer_connection,
            media_transport,
            peer_id=peer_id,
            session_runner=None,
            peer_registry=peer_registry,
            media_only=True,
            audio_writer=audio_writer,
            debug_dump_enabled=debug_dump_enabled,
            logger=logger,
        )
        media_transport.attach_media(
            peer_id,
            assistant_track,
            peer_session,
        )
        peer_session.register()
        return peer_session

    def register(self) -> None:
        if self._registered:
            return
        self._registered = True
        self.peer_registry.add(self.peer_connection)
        self.peer_connection.on("track")(self._on_track)
        if not self.media_only:
            self.peer_connection.on("datachannel")(self._on_datachannel)
        self.peer_connection.on("connectionstatechange")(
            self._on_connection_state_change
        )
        self.peer_connection.on("icegatheringstatechange")(
            self._on_ice_gathering_state_change
        )

    async def negotiate(
        self,
        offer_sdp: str,
        offer_type: str = "offer",
    ) -> dict[str, str]:
        remote_description = self._session_description_factory(
            sdp=offer_sdp,
            type=offer_type,
        )
        await self.peer_connection.setRemoteDescription(remote_description)
        answer = await self.peer_connection.createAnswer()
        await self.peer_connection.setLocalDescription(answer)
        await self.wait_for_ice_gathering_complete()
        local_description = self.peer_connection.localDescription
        return {
            "sdp": local_description.sdp,
            "type": local_description.type,
        }

    async def close(self, reason: str = "") -> None:
        if self._closing:
            return
        self._closing = True
        if self.media_only:
            self.connection.detach_media(self.peer_id)
        else:
            self.connection.disconnect()
        media_tasks = list(self.media_tasks)
        for task in media_tasks:
            task.cancel()
        if media_tasks:
            await asyncio.gather(
                *media_tasks,
                return_exceptions=True,
            )
        try:
            await self.peer_connection.close()
        finally:
            self.peer_registry.discard(self.peer_connection)
            if reason:
                self._log(
                    f"[WebRTC] Peer {self.peer_id} closed: {reason}"
                )

    async def wait_for_ice_gathering_complete(self) -> None:
        if self.peer_connection.iceGatheringState == "complete":
            return
        try:
            await asyncio.wait_for(
                self._ice_gathering_done.wait(),
                timeout=self.ice_gathering_timeout_s,
            )
        except asyncio.TimeoutError:
            self._log(
                "[WebRTC] ICE gathering timeout "
                f"({self.peer_connection.iceGatheringState}), 返回当前 SDP"
            )

    def _on_track(self, track) -> None:
        self._log(
            f"[WebRTC] 收到媒体轨道: "
            f"kind={track.kind}, peer={self.peer_id}"
        )
        if track.kind != "audio":
            return
        task = asyncio.create_task(self._consume_audio_track(track))
        self.media_tasks.add(task)
        task.add_done_callback(self.media_tasks.discard)

    def _on_datachannel(self, channel) -> None:
        label = getattr(channel, "label", "") or ""
        self._log(
            f"[WebRTC] 收到 DataChannel: "
            f"{label or '(unnamed)'} ({self.peer_id})"
        )
        self.connection.attach_data_channel(channel)
        channel.on("open")(partial(self._on_channel_open, channel))
        if (
            getattr(channel, "readyState", "") == "open"
            and label != "voice-audio"
        ):
            self._start_session_if_ready()

    def _on_channel_open(self, channel) -> None:
        label = getattr(channel, "label", "") or ""
        self._log(
            f"[WebRTC] DataChannel open: "
            f"{label or '(unnamed)'} ({self.peer_id})"
        )
        if label != "voice-audio":
            self._start_session_if_ready()

    def _start_session_if_ready(self) -> None:
        if (
            self.media_only
            or self._session_runner is None
            or self.session_task is not None
            or not self.connection.is_control_open()
        ):
            return
        self._log(
            "[WebRTC] 控制 DataChannel 已打开，"
            f"启动语音会话: {self.peer_id}"
        )
        self.session_task = asyncio.create_task(
            self._session_runner(self.connection)
        )
        self.session_task.add_done_callback(self._on_session_done)

    def _on_session_done(self, task: asyncio.Task) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc:
            self._log(f"[WebRTC] 会话任务异常: {type(exc).__name__}")
        self.close_task = asyncio.create_task(
            self.close("session finished")
        )

    async def _on_connection_state_change(self) -> None:
        state = self.peer_connection.connectionState
        self._log(f"[WebRTC] Peer {self.peer_id} state={state}")
        if self.media_only:
            self.connection.set_media_ready(
                self.peer_id,
                state == "connected",
            )
        if state in {"failed", "closed"}:
            await self.close(state)

    def _on_ice_gathering_state_change(self) -> None:
        if self.peer_connection.iceGatheringState == "complete":
            self._ice_gathering_done.set()

    async def _consume_audio_track(self, track) -> None:
        frame_count = 0
        debug_dump: list[np.ndarray] = []
        debug_dump_samples = 0
        debug_dump_done = not self.debug_dump_enabled
        try:
            while True:
                frame = await track.recv()
                audio_float, sample_rate = self._audio_frame_converter(frame)
                self.connection.feed_audio_float(audio_float, sample_rate)
                frame_count += 1
                if not debug_dump_done:
                    debug_dump.append(audio_float.copy())
                    debug_dump_samples += audio_float.size
                    if (
                        debug_dump_samples
                        >= sample_rate * self.debug_dump_seconds
                    ):
                        debug_dump_done = True
                        self._write_debug_dump(
                            debug_dump,
                            sample_rate,
                        )
                        debug_dump = []
                if frame_count <= 3:
                    self._log_audio_frame(
                        frame,
                        frame_count,
                        audio_float,
                        sample_rate,
                    )
        except self._media_stream_error_type:
            self._log(
                f"[WebRTC] 麦克风 RTP 音频轨道结束: {self.peer_id}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"[WebRTC] 麦克风 RTP 音频消费异常: {type(exc).__name__}")

    def _write_debug_dump(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        if self._audio_writer is None:
            return
        try:
            dump_path = f"tmp/webrtc_mic_debug_{self.peer_id}.wav"
            self._audio_writer(
                dump_path,
                np.concatenate(chunks),
                sample_rate,
            )
            self._log(
                "[WebRTC] 🎧 麦克风原始音频已存盘用于诊断: "
                f"{dump_path} ({sample_rate}Hz)"
            )
        except Exception as exc:
            self._log(f"[WebRTC] 调试存盘失败: {type(exc).__name__}")

    def _log_audio_frame(
        self,
        frame,
        frame_count: int,
        audio_float: np.ndarray,
        sample_rate: int,
    ) -> None:
        layout = getattr(frame, "layout", None)
        frame_format = getattr(
            getattr(frame, "format", None),
            "name",
            "?",
        )
        raw_shape = np.asarray(frame.to_ndarray()).shape
        rms = (
            float(np.sqrt(np.mean(audio_float ** 2)))
            if audio_float.size
            else 0.0
        )
        peak = (
            float(np.max(np.abs(audio_float)))
            if audio_float.size
            else 0.0
        )
        self._log(
            f"[WebRTC] 麦克风 RTP 帧#{frame_count}: "
            f"sr={sample_rate}, fmt={frame_format}, "
            f"layout={getattr(layout, 'name', '?')}, "
            f"raw_shape={raw_shape}, "
            f"mono_samples={audio_float.size}, "
            f"rms={rms:.4f}, peak={peak:.4f}, "
            f"peer={self.peer_id}"
        )
