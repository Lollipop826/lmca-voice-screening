from __future__ import annotations

from collections.abc import Callable
from typing import Any
import base64
import numpy as np
import time
import uuid

class ManualAudioInputHandler:
    """Assemble and normalize operator-controlled audio uploads."""

    MESSAGE_TYPES = {
        "manual_audio_blob_start",
        "manual_audio_blob_chunk",
        "manual_audio_blob_end",
        "manual_audio_blob",
        "manual_audio",
    }

    def __init__(
        self,
        connection,
        *,
        uploads: dict[str, dict],
        process_recorded_blob: Callable[..., Any],
        submit_speech: Callable[..., Any],
        normalize_doctor_markers: Callable[[list | None], list[dict]],
        resample_audio: Callable[[np.ndarray, int, int], np.ndarray],
        now_factory: Callable[[], float] = time.time,
        token_factory: Callable[[], str] | None = None,
        logger=print,
    ) -> None:
        self.connection = connection
        self.uploads = uploads
        self._process_recorded_blob = process_recorded_blob
        self._submit_speech = submit_speech
        self._normalize_doctor_markers = normalize_doctor_markers
        self._resample_audio = resample_audio
        self._now = now_factory
        self._token_factory = token_factory or (
            lambda: uuid.uuid4().hex[:8]
        )
        self._log = logger
        self._handlers = {
            "manual_audio_blob_start": self._handle_blob_start,
            "manual_audio_blob_chunk": self._handle_blob_chunk,
            "manual_audio_blob_end": self._handle_blob_end,
            "manual_audio_blob": self._handle_complete_blob,
            "manual_audio": self._handle_pcm_audio,
        }

    async def handle_message(self, message: dict) -> bool:
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            return False
        await handler(message)
        return True

    async def _handle_blob_start(self, message: dict) -> None:
        upload_id = str(message.get("upload_id") or self._token_factory())
        total_chunks = max(
            0,
            self._safe_int(message.get("total_chunks"), 0),
        )
        mime_type = str(message.get("mime_type") or "")
        self.uploads[upload_id] = {
            "mime_type": mime_type,
            "total_chunks": total_chunks,
            "chunks": [],
            "received_chunks": 0,
            "started_at": self._now(),
            "doctor_markers": (
                message.get("doctor_markers")
                if isinstance(message.get("doctor_markers"), list)
                else []
            ),
        }
        self._log(
            f"[手动语音] 开始接收分片录音: "
            f"upload={upload_id}, chunks={total_chunks}, mime={mime_type}"
        )

    async def _handle_blob_chunk(self, message: dict) -> None:
        upload_id = str(message.get("upload_id") or "")
        upload_state = self.uploads.get(upload_id)
        if not upload_state:
            self._log(f"[手动语音] ⚠️ 收到未知分片: upload={upload_id}")
            return

        chunk_data = str(message.get("data") or "")
        index = max(
            0,
            self._safe_int(
                message.get("index"),
                len(upload_state["chunks"]),
            ),
        )
        chunks = upload_state["chunks"]
        while len(chunks) <= index:
            chunks.append(None)
        if chunks[index] is None:
            upload_state["received_chunks"] = (
                int(upload_state.get("received_chunks") or 0) + 1
            )
        chunks[index] = chunk_data

    async def _handle_blob_end(self, message: dict) -> None:
        upload_id = str(message.get("upload_id") or "")
        upload_state = self.uploads.pop(upload_id, None)
        if not upload_state:
            self._log(
                f"[手动语音] ⚠️ 收到未知结束标记: upload={upload_id}"
            )
            return

        total_chunks = max(
            0,
            self._safe_int(upload_state.get("total_chunks"), 0),
        )
        chunks = list(upload_state.get("chunks") or [])
        expected_chunks = total_chunks or len(chunks)
        if len(chunks) < expected_chunks or any(
            part is None for part in chunks[:expected_chunks]
        ):
            self._log(
                f"[手动语音] ❌ 分片录音不完整: upload={upload_id}, "
                f"expected={expected_chunks}, "
                f"received={upload_state.get('received_chunks', 0)}"
            )
            await self._send_feedback(
                reason="generic",
                message="手动录音分片不完整，请再试一次",
                status_text="手动录音分片不完整，请再试一次",
                source="manual_audio_blob_chunked",
            )
            return

        audio_base64 = "".join(chunks[:expected_chunks])
        self._log(
            f"[手动语音] 分片录音接收完成: upload={upload_id}, "
            f"chunks={expected_chunks}, chars={len(audio_base64)}"
        )
        result = self._process_recorded_blob(
            audio_base64,
            str(upload_state.get("mime_type") or ""),
            source="manual_audio_blob_chunked",
            upload_id=upload_id,
            doctor_markers=(
                upload_state.get("doctor_markers")
                if isinstance(upload_state.get("doctor_markers"), list)
                else []
            ),
        )
        await self._await_if_needed(result)

    async def _handle_complete_blob(self, message: dict) -> None:
        result = self._process_recorded_blob(
            str(message.get("audio") or ""),
            str(message.get("mime_type") or ""),
            source="manual_audio_blob",
            doctor_markers=(
                message.get("doctor_markers")
                if isinstance(message.get("doctor_markers"), list)
                else []
            ),
        )
        await self._await_if_needed(result)

    async def _handle_pcm_audio(self, message: dict) -> None:
        audio_base64 = message.get("audio") or ""
        sample_rate = int(message.get("sample_rate") or 16000)
        if not audio_base64:
            await self._send_feedback(
                reason="asr_empty",
                message="未收到录音，请再试一次",
                status_text="未收到录音，请再试一次",
                source="manual_audio",
            )
            return

        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            if sample_rate != 16000:
                audio_float = self._resample_audio(
                    audio_float,
                    sample_rate,
                    16000,
                )
            if audio_float.size == 0:
                raise ValueError("empty manual audio")
            normalized_markers = self._normalize_doctor_markers(
                message.get("doctor_markers")
                if isinstance(message.get("doctor_markers"), list)
                else []
            )
            self._log(
                f"[手动语音] 收到手动录音，"
                f"时长 {audio_float.size / 16000:.2f}s"
            )
            await self.connection.send_json({"type": "vad_end"})
            result = self._submit_speech(
                audio_float,
                source="手动点击语音",
                extra_meta={
                    "doctor_markers": normalized_markers,
                    "manual_source": "manual_audio",
                    "sample_rate": sample_rate,
                },
            )
            await self._await_if_needed(result)
        except Exception as exc:
            self._log(f"[手动语音] ❌ 处理失败: {type(exc).__name__}")
            await self._send_feedback(
                reason="generic",
                message=f"手动录音处理失败：{exc}",
                status_text="手动录音处理失败，请再试一次",
                source="manual_audio",
            )

    async def _send_feedback(
        self,
        *,
        reason: str,
        message: str,
        status_text: str,
        source: str,
    ) -> None:
        await self.connection.send_json(
            {
                "type": "voice_input_feedback",
                "reason": reason,
                "message": message,
                "status_text": status_text,
                "source": source,
            }
        )

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result
