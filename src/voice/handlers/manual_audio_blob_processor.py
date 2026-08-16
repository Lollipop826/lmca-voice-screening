from __future__ import annotations

from .voice_feedback_presenter import VoiceFeedbackPresenter
from collections.abc import Callable
from typing import Any
import base64
import numpy as np

class ManualAudioBlobProcessor:
    """Decode and validate one browser MediaRecorder blob before submission."""

    def __init__(
        self,
        connection,
        *,
        processing_coordinator,
        feedback: VoiceFeedbackPresenter,
        decode_recorded_audio_blob: Callable[[bytes, str], Any],
        resample_audio: Callable[[np.ndarray, int, int], np.ndarray],
        audio_signal_stats: Callable[[np.ndarray, int], dict],
        audio_is_effectively_silent: Callable[[dict], bool],
        sample_rate: int = 16000,
        logger=print,
    ) -> None:
        self.connection = connection
        self.processing_coordinator = processing_coordinator
        self.feedback = feedback
        self._decode_recorded_audio_blob = decode_recorded_audio_blob
        self._resample_audio = resample_audio
        self._audio_signal_stats = audio_signal_stats
        self._audio_is_effectively_silent = audio_is_effectively_silent
        self.sample_rate = int(sample_rate)
        self._log = logger

    async def process(
        self,
        audio_base64: str,
        mime_type: str,
        source: str,
        upload_id: str = "",
        doctor_markers: list | None = None,
    ) -> None:
        if not audio_base64:
            await self.feedback.send(
                "asr_empty",
                "未收到录音，请再试一次",
                status_text="未收到录音，请再试一次",
                source=source,
            )
            return

        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_float, decoded_sample_rate = (
                self._decode_recorded_audio_blob(
                    audio_bytes,
                    mime_type,
                )
            )
            if decoded_sample_rate != self.sample_rate:
                audio_float = self._resample_audio(
                    audio_float,
                    decoded_sample_rate,
                    self.sample_rate,
                )
            if audio_float.size == 0:
                raise ValueError("empty decoded manual audio")

            normalized_markers = self.normalize_doctor_markers(
                doctor_markers
            )
            stats = self._audio_signal_stats(
                audio_float,
                self.sample_rate,
            )
            upload_suffix = f", upload={upload_id}" if upload_id else ""
            self._log(
                f"[手动语音] 收到 MediaRecorder 录音，"
                f"时长 {audio_float.size / self.sample_rate:.2f}s, "
                f"rms={stats['rms']:.5f}, peak={stats['peak']:.5f}, "
                f"dbfs={stats['dbfs']:.1f}, mime={mime_type}"
                f"{upload_suffix}, doctor_markers={len(normalized_markers)}"
            )
            if self._audio_is_effectively_silent(stats):
                await self.feedback.send(
                    "audio_too_quiet",
                    "这段录音音量太低，系统几乎没有收到声音。"
                    "请确认浏览器麦克风选对了，并靠近一点再说。",
                    status_text="录音音量太低，请再说一遍",
                    source=source,
                    duration_s=stats["duration_s"],
                )
                return

            await self.connection.send_json({"type": "vad_end"})
            await self.processing_coordinator.submit_or_queue(
                audio_float,
                source="手动点击语音",
                extra_meta={
                    "doctor_markers": normalized_markers,
                    "manual_source": source,
                    "mime_type": mime_type,
                },
            )
        except Exception as exc:
            self._log(f"[手动语音] ❌ MediaRecorder 处理失败: {type(exc).__name__}")
            await self.feedback.send(
                "generic",
                f"手动录音处理失败：{exc}",
                status_text="手动录音处理失败，请再试一次",
                source=source,
            )

    @staticmethod
    def normalize_doctor_markers(
        markers: list | None,
    ) -> list[dict]:
        normalized: list[dict] = []
        for item in markers or []:
            if not isinstance(item, dict):
                continue
            event = str(item.get("event") or "").strip()
            if event not in {"doctor_start", "doctor_end"}:
                continue
            try:
                at_ms = max(0.0, float(item.get("at_ms") or 0.0))
            except (TypeError, ValueError):
                at_ms = 0.0
            marker = {"event": event, "at_ms": at_ms}
            try:
                timestamp = (
                    float(item.get("ts"))
                    if item.get("ts") is not None
                    else None
                )
            except (TypeError, ValueError):
                timestamp = None
            if timestamp is not None:
                marker["ts"] = timestamp
            try:
                duration_ms = (
                    float(item.get("duration_ms"))
                    if item.get("duration_ms") is not None
                    else None
                )
            except (TypeError, ValueError):
                duration_ms = None
            if duration_ms is not None and duration_ms >= 0:
                marker["duration_ms"] = duration_ms
            if item.get("auto_closed") is True:
                marker["auto_closed"] = True
            normalized.append(marker)
        normalized.sort(
            key=lambda marker: (
                marker.get("at_ms", 0.0),
                0 if marker.get("event") == "doctor_start" else 1,
            )
        )
        return normalized
