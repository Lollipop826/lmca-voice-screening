from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np


_OUTPUT_IDENTITY_EVENT_TYPES = {
    "ai_response",
    "ai_response_chunk",
    "tts_start",
    "tts_chunk",
    "tts_audio",
    "tts_end",
    "tts_stop",
    "stop_tts",
    "interrupt",
    "avatar_video",
    "avatar_video_error",
}
_OUTPUT_START_EVENT_TYPES = {
    "tts_start",
}


class VoiceConnectionController:
    """Receive messages while keeping interruption/audio responsive."""

    def __init__(self, connection, router) -> None:
        self.connection = connection
        self.router = router

    async def run(self) -> None:
        normal_lock = asyncio.Lock()
        audio_lock = asyncio.Lock()
        stop_event = asyncio.Event()
        active_tasks: set[asyncio.Task] = set()
        errors: list[Exception] = []

        async def dispatch_message(message: dict) -> None:
            try:
                message_type = message.get("type")
                if message_type == "interrupt":
                    route_result = await self.router.dispatch(message)
                elif message_type == "audio":
                    async with audio_lock:
                        route_result = await self.router.dispatch(message)
                else:
                    async with normal_lock:
                        route_result = await self.router.dispatch(message)
                if route_result.stop_connection:
                    stop_event.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
                stop_event.set()

        try:
            while True:
                receive_task = asyncio.create_task(
                    self.connection.receive_message()
                )
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    {receive_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and stop_event.is_set():
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)
                    break
                stop_task.cancel()
                await asyncio.gather(stop_task, return_exceptions=True)
                message = receive_task.result()
                if message is None:
                    break
                task = asyncio.create_task(dispatch_message(message))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)
                await asyncio.sleep(0)
                if stop_event.is_set():
                    break
        finally:
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        if errors:
            raise errors[0]


class VoiceConnectionIO:
    """Transport-neutral JSON/audio I/O for one WebSocket or WebRTC connection."""

    _LOGGED_PAYLOAD_TYPES = {
        "asr_result",
        "asr_error",
        "ai_response_chunk",
        "ai_response",
        "tts_start",
        "tts_end",
        "speaker_error",
        "processing_status",
        "avatar_video",
        "avatar_video_error",
        "update_score",
    }

    def __init__(self, transport: Any, *, session=None, logger=print) -> None:
        self.transport = transport
        self._session = session
        self._log = logger
        self.audio_frame_count = 0
        self.audio_sample_count = 0
        self._fallback_generation = 0
        self._fallback_turn_sequence = 0
        self._last_output_identity: dict[str, Any] | None = None
        self._active_output_identity: dict[str, Any] | None = None

    def bind_session(self, session) -> None:
        self._session = session

    def current_output_identity(self) -> dict[str, Any] | None:
        if self._active_output_identity is None:
            return None
        return dict(self._active_output_identity)

    def _normalize_output_identity(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = dict(data)
        payload_type = payload.get("type")
        if payload_type not in _OUTPUT_IDENTITY_EVENT_TYPES:
            return payload

        session = self._session
        processing = getattr(session, "processing", None)
        active_turn_id = str(getattr(processing, "active_turn_id", "") or "")
        active_playback_id = str(
            getattr(processing, "active_playback_id", "") or ""
        )
        processing_active = bool(getattr(processing, "is_active", False))
        processing_generation = int(
            getattr(processing, "generation", self._fallback_generation) or 0
        )
        session_id = str(
            payload.get("session_id")
            or getattr(session, "session_id", "")
            or ""
        )
        requested_turn_id = str(payload.get("turn_id") or "")
        uses_processing = bool(
            active_turn_id
            and (
                requested_turn_id == active_turn_id
                or (processing_active and not requested_turn_id)
            )
        )

        identity = None
        if uses_processing:
            identity = {
                "session_id": session_id,
                "turn_id": requested_turn_id or active_turn_id,
                "generation": payload.get("generation", processing_generation),
                "playback_id": payload.get("playback_id") or active_playback_id,
            }
        elif requested_turn_id and self._last_output_identity:
            previous = self._last_output_identity
            if (
                previous.get("session_id") == session_id
                and previous.get("turn_id") == requested_turn_id
            ):
                identity = dict(previous)
                if "generation" in payload:
                    identity["generation"] = payload["generation"]
                if "playback_id" in payload:
                    identity["playback_id"] = payload["playback_id"]

        if identity is None:
            if session_id != str((self._last_output_identity or {}).get("session_id") or ""):
                self._fallback_generation = processing_generation
                self._last_output_identity = None
                self._active_output_identity = None
            starts_new_output = payload_type in {
                "ai_response",
                "ai_response_chunk",
                "tts_start",
            } and not requested_turn_id and not self._active_output_identity
            if starts_new_output and self._last_output_identity:
                self._fallback_generation = max(
                    self._fallback_generation,
                    int(self._last_output_identity.get("generation") or 0),
                ) + 1
            if requested_turn_id:
                turn_id = requested_turn_id
            elif self._active_output_identity:
                turn_id = str(self._active_output_identity.get("turn_id") or "")
            else:
                self._fallback_turn_sequence += 1
                turn_id = f"turn_output_{self._fallback_turn_sequence:04d}"
            if requested_turn_id and self._last_output_identity:
                previous_turn = self._last_output_identity.get("turn_id")
                if previous_turn and previous_turn != requested_turn_id:
                    self._fallback_generation = max(
                        self._fallback_generation,
                        processing_generation,
                    ) + 1
            generation = payload.get("generation")
            if generation is None:
                generation = self._fallback_generation
            else:
                generation = int(generation)
                self._fallback_generation = max(self._fallback_generation, generation)
            identity = {
                "session_id": session_id,
                "turn_id": turn_id,
                "generation": generation,
                "playback_id": payload.get("playback_id")
                or f"playback-{turn_id}-{generation}",
            }

        identity["session_id"] = session_id
        identity["turn_id"] = str(identity.get("turn_id") or requested_turn_id)
        identity["generation"] = int(identity.get("generation") or 0)
        identity["playback_id"] = str(
            identity.get("playback_id")
            or f"playback-{identity['turn_id']}-{identity['generation']}"
        )
        payload.update(identity)
        self._last_output_identity = dict(identity)
        if payload_type not in {"tts_end", "tts_stop", "stop_tts", "interrupt"}:
            self._active_output_identity = dict(identity)
        elif payload_type in {"stop_tts", "interrupt", "tts_stop"}:
            self._active_output_identity = dict(identity)
        else:
            self._active_output_identity = None
        return payload

    def _accept_output_identity(
        self,
        payload: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> bool:
        """Reject late output before it reaches a WebSocket/WebRTC transport."""
        if not previous:
            return True
        generation = int(payload.get("generation") or 0)
        previous_generation = int(previous.get("generation") or 0)
        if generation < previous_generation:
            reason = "stale_generation"
        elif (
            generation == previous_generation
            and payload.get("playback_id") != previous.get("playback_id")
            and payload.get("type") not in _OUTPUT_START_EVENT_TYPES
        ):
            reason = "stale_playback"
        else:
            return True
        self._log(
            "[输出丢弃] "
            f"reason={reason} type={payload.get('type')} "
            f"session={payload.get('session_id') or '-'} "
            f"turn={payload.get('turn_id') or '-'} "
            f"generation={generation} playback={payload.get('playback_id') or '-'}"
        )
        return False

    @property
    def cookies(self) -> dict[str, str]:
        return dict(getattr(self.transport, "cookies", {}) or {})

    @property
    def closed(self) -> bool:
        return bool(getattr(self.transport, "closed", False))

    @property
    def transport_name(self) -> str:
        return str(getattr(self.transport, "transport_name", "WebSocket"))

    @property
    def client_id(self) -> str:
        return str(id(self.transport))

    async def accept(self) -> None:
        await self.transport.accept()

    async def close(self, code: int = 1000) -> None:
        await self.transport.close(code=code)

    async def receive_message(self) -> dict[str, Any] | None:
        """Return the next normalized control/audio message, or None on disconnect."""
        while True:
            try:
                raw_message = await self.transport.receive()
            except RuntimeError as exc:
                self._log(f"[断开] {self.transport_name} 已断开 ({type(exc).__name__})")
                return None

            if raw_message.get("type") == "websocket.disconnect":
                self._log("[断开] 收到 disconnect 消息")
                return None

            audio_bytes = raw_message.get("bytes")
            if audio_bytes:
                return await self._decode_binary_audio(audio_bytes)

            text = raw_message.get("text")
            if text:
                message = json.loads(text)
                if message.get("type") == "audio" and "data" in message:
                    audio_data = np.asarray(message["data"])
                    if np.issubdtype(audio_data.dtype, np.floating):
                        message["_audio_float"] = audio_data.astype(np.float32)
                    else:
                        message["_audio_float"] = (
                            audio_data.astype(np.float32) / 32768.0
                        )
                return message

    async def send_json(self, data: dict[str, Any]) -> bool:
        """Send JSON without letting a disconnected transport break the session loop."""
        if self.closed:
            return False
        state_snapshot = (
            self._fallback_generation,
            self._fallback_turn_sequence,
            self._last_output_identity,
            self._active_output_identity,
        )
        previous_identity = (
            dict(self._last_output_identity)
            if self._last_output_identity is not None
            else None
        )
        data = self._normalize_output_identity(data)
        if data.get("type") in _OUTPUT_IDENTITY_EVENT_TYPES and not self._accept_output_identity(
            data,
            previous_identity,
        ):
            (
                self._fallback_generation,
                self._fallback_turn_sequence,
                self._last_output_identity,
                self._active_output_identity,
            ) = state_snapshot
            return False
        try:
            await self.transport.send_json(data)
            self._log_outgoing_payload(data)
            return True
        except Exception as exc:
            self._log(
                f"[{self.transport_name}] 发送失败（连接可能已断开）: {exc}"
            )
            return False

    async def _decode_binary_audio(self, audio_bytes: bytes) -> dict[str, Any]:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        self.audio_frame_count += 1
        self.audio_sample_count += int(audio_data.size)

        if self.audio_frame_count == 1:
            rms = (
                float(np.sqrt(np.mean(np.square(audio_float))))
                if audio_float.size
                else 0.0
            )
            peak = (
                float(np.max(np.abs(audio_float)))
                if audio_float.size
                else 0.0
            )
            self._log(
                f"[麦克风] ✅ 收到首帧音频: samples={audio_data.size} "
                f"rms={rms:.4f} peak={peak:.4f}"
            )
            await self.send_json(
                {
                    "type": "audio_input_started",
                    "samples": int(audio_data.size),
                    "sample_rate": 16000,
                }
            )

        return {"type": "audio", "_audio_float": audio_float}

    def _log_outgoing_payload(self, data: dict[str, Any]) -> None:
        payload_type = data.get("type") if isinstance(data, dict) else None
        if payload_type not in self._LOGGED_PAYLOAD_TYPES:
            return

        turn_id = data.get("turn_id") or "-"
        text_value = data.get("text") or data.get("message") or ""
        text_len = len(str(text_value))
        extra = []
        if payload_type == "update_score":
            score_data = data.get("data") or {}
            extra.append(
                f"score={score_data.get('total_score', '-')}/"
                f"{score_data.get('total_max_score', '-')}"
            )
        if "sentence_index" in data:
            extra.append(f"idx={data.get('sentence_index')}")
        if "chunks" in data:
            extra.append(f"chunks={data.get('chunks')}")
        if "similarity" in data:
            try:
                extra.append(f"similarity={float(data.get('similarity')):.3f}")
            except Exception:
                extra.append(f"similarity={data.get('similarity')}")
        if payload_type == "ai_response":
            extra.append(f"finalize={bool(data.get('finalize_chunks'))}")
        extra_text = " | " + " | ".join(extra) if extra else ""
        self._log(
            f"[SEND][{self.transport_name}] type={payload_type} | "
            f"turn={turn_id} | text_len={text_len}{extra_text}"
        )
