from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
import time

class SoulXAudioHandler:
    """Run the SoulX semantic turn-taking path before local VAD fallback."""

    def __init__(
        self,
        connection,
        *,
        client,
        accumulator,
        turn_state,
        speaker,
        answer_completion,
        reset_local_vad: Callable[[], Any],
        is_ai_speaking: Callable[[], bool],
        stop_playback: Callable[[], Any],
        notify_answer_completion: Callable[..., Any],
        arm_answer_completion_window: Callable[[], Any],
        submit_speech: Callable[..., Any],
        audio_signal_stats: Callable[[np.ndarray, int], dict],
        unavailable_error_type: type[Exception],
        server_url: str,
        enabled_full_duplex: bool,
        minimum_utterance_rms: float,
        sample_rate: int = 16000,
        perf_counter_factory: Callable[[], float] = time.perf_counter,
        logger=print,
    ) -> None:
        self.connection = connection
        self.client = client
        self.accumulator = accumulator
        self.turn_state = turn_state
        self.speaker = speaker
        self.answer_completion = answer_completion
        self._reset_local_vad = reset_local_vad
        self._is_ai_speaking = is_ai_speaking
        self._stop_playback = stop_playback
        self._notify_answer_completion = notify_answer_completion
        self._arm_answer_completion_window = (
            arm_answer_completion_window
        )
        self._submit_speech = submit_speech
        self._audio_signal_stats = audio_signal_stats
        self._unavailable_error_type = unavailable_error_type
        self.server_url = server_url
        self.enabled_full_duplex = bool(enabled_full_duplex)
        self.minimum_utterance_rms = float(minimum_utterance_rms)
        self.sample_rate = int(sample_rate)
        self._perf_counter = perf_counter_factory
        self._log = logger
        # SoulX 服务端原生 chunk 为 2560 样本(160ms@16k)，正好由
        # 5 个 512 样本(32ms) WebRTC 上游帧组成。这里仍精确切块并保留
        # 余数，以兼容其他任意大小的音频输入帧，避免时间线漂移。
        self._batch_chunks: list[np.ndarray] = []
        self._batch_samples = 0
        self._batch_target_samples = 2560
        self._last_streamed_text = ""
        self._last_streamed_detail_signature = None

    def _reset_batch(self) -> None:
        self._batch_chunks = []
        self._batch_samples = 0

    def _take_native_batch(self) -> np.ndarray | None:
        """Pop exactly one native SoulX chunk and retain any remainder."""
        if self._batch_samples < self._batch_target_samples:
            return None

        remaining = self._batch_target_samples
        parts: list[np.ndarray] = []
        while remaining > 0:
            chunk = self._batch_chunks[0]
            take = min(int(chunk.size), remaining)
            parts.append(chunk[:take])
            if take == chunk.size:
                self._batch_chunks.pop(0)
            else:
                self._batch_chunks[0] = chunk[take:]
            self._batch_samples -= take
            remaining -= take

        return np.concatenate(parts).astype(np.float32, copy=False)

    async def handle_audio(self, audio_chunk: np.ndarray) -> bool:
        """Return True when SoulX consumed the chunk, else use local VAD."""
        if self.client is None or not self.client.retry_ready:
            self._reset_batch()
            return False

        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio.size:
            self._batch_chunks.append(audio.copy())
            self._batch_samples += int(audio.size)

        while self._batch_samples >= self._batch_target_samples:
            batch = self._take_native_batch()
            if batch is None or not await self._process_native_batch(batch):
                return False
        return True

    async def _process_native_batch(self, batch: np.ndarray) -> bool:
        try:
            soulx_state = self.client.process(batch)
            soulx_state = await self._await_if_needed(soulx_state)
        except self._unavailable_error_type as exc:
            error_text = str(exc)
            if self.turn_state.mark_soulx_unavailable(error_text):
                self._log(
                    "[SoulX] ⚠️ 轮次服务不可用，降级到本地 VAD: "
                    f"{error_text}"
                )
            self._reset_batch()
            self.accumulator.reset()
            return False

        if soulx_state is None:
            return False
        if self.turn_state.mark_soulx_connected():
            self._log(f"[SoulX] ✅ 轮次服务已连接: {self.server_url}")
        await self._observe_state_transition(soulx_state)
        complete_audio = self.accumulator.feed(
            batch,
            soulx_state.state,
        )

        if soulx_state.state == "nonidle":
            await self._handle_nonidle()
            return True
        if soulx_state.state == "speak":
            await self._handle_speak(soulx_state, complete_audio)
            return True

        if not self.accumulator.active:
            self.turn_state.soulx_barge_in_active = False
        return True

    async def _observe_state_transition(self, soulx_state) -> None:
        """Log state changes and stream changed partial transcripts.

        State transitions are always sent. During nonidle/speak, a changed
        Paraformer transcript is also sent even when the state is unchanged.
        Identical partials and blank states are suppressed, and send failures
        must never disrupt turn-taking.
        """
        if soulx_state.state == "blank":
            return
        previous_state, changed = self.turn_state.observe_soulx_state(
            soulx_state.state
        )
        preview = (
            soulx_state.text or soulx_state.asr_buffer or ""
        ).replace("\n", " ")[:80]
        raw_state = str(getattr(soulx_state, "raw_state", "") or "")
        detail_state = str(
            getattr(soulx_state, "detail_state", "") or ""
        )
        decision_source = str(
            getattr(soulx_state, "decision_source", "") or ""
        )
        wait_idle_count = int(
            getattr(soulx_state, "wait_idle_count", 0) or 0
        )
        max_wait_count = int(
            getattr(soulx_state, "max_wait_count", 0) or 0
        )
        monitoring_wait_silence = bool(
            getattr(soulx_state, "monitoring_wait_silence", False)
        )
        speech_detected = bool(
            getattr(soulx_state, "speech_detected", False)
        )
        chunk_rms = float(getattr(soulx_state, "chunk_rms", 0.0) or 0.0)
        has_detail = bool(raw_state or detail_state or decision_source)
        detail_signature = (
            raw_state,
            detail_state,
            decision_source,
            wait_idle_count,
            max_wait_count,
            monitoring_wait_silence,
            speech_detected,
        )
        detail_changed = (
            has_detail
            and detail_signature != self._last_streamed_detail_signature
        )
        partial_changed = (
            soulx_state.state in {"nonidle", "speak"}
            and bool(preview)
            and preview != self._last_streamed_text
        )
        if not changed and not partial_changed and not detail_changed:
            return
        if changed:
            self._log(
                f"[SoulX] 状态 {previous_state or '-'} → "
                f"{soulx_state.state}, detail={detail_state or '-'}, "
                f"source={decision_source or '-'}, text_chars={len(preview)}"
            )
        self._last_streamed_text = preview
        if has_detail:
            self._last_streamed_detail_signature = detail_signature
        payload = {
            "type": "soulx_state",
            "state": soulx_state.state,
            "text": preview,
        }
        if has_detail:
            payload.update(
                {
                    "raw_state": raw_state,
                    "detail_state": detail_state,
                    "decision_source": decision_source,
                    "wait_idle_count": wait_idle_count,
                    "max_wait_count": max_wait_count,
                    "monitoring_wait_silence": monitoring_wait_silence,
                    "speech_detected": speech_detected,
                    "chunk_rms": chunk_rms,
                }
            )
        try:
            await self.connection.send_json(payload)
        except Exception:
            pass

    async def _handle_nonidle(self) -> None:
        if (
            not self.enabled_full_duplex
            or self.turn_state.soulx_barge_in_active
            or not self._is_ai_speaking()
        ):
            return

        if self._should_gate_barge_in_by_speaker():
            barge_in_audio = self.accumulator.tail_audio(1.6)
            if barge_in_audio.size >= int(0.5 * self.sample_rate):
                verify_started = self._perf_counter()
                is_target, similarity = await self.speaker.verify(
                    barge_in_audio,
                    self.sample_rate,
                    source="soulx_barge_in",
                )
                verify_ms = (
                    self._perf_counter() - verify_started
                ) * 1000
                if not is_target:
                    self._log(
                        "[SoulX] ⏭️ nonidle 未通过声纹比对，不打断: "
                        f"similarity={similarity}, 耗时={verify_ms:.0f}ms"
                    )
                    return
                self._log(
                    "[SoulX] 🔓 nonidle 声纹比对通过: "
                    f"similarity={similarity}, 耗时={verify_ms:.0f}ms"
                )

        self.turn_state.soulx_barge_in_active = True
        await self._await_if_needed(self._stop_playback())
        self._log("[SoulX] 🛑 nonidle：已停止当前 AI 输出，继续聆听到 speak")

    async def _handle_speak(self, soulx_state, complete_audio) -> None:
        self.turn_state.soulx_barge_in_active = False
        await self._await_if_needed(self._reset_local_vad())
        minimum_samples = int(0.2 * self.sample_rate)
        if complete_audio is None or complete_audio.size < minimum_samples:
            self._log("[SoulX] ⚠️ speak 对应音频为空或过短，忽略")
            return

        speech_duration = complete_audio.size / float(self.sample_rate)
        stats = self._audio_signal_stats(
            complete_audio,
            self.sample_rate,
        )
        if stats["rms"] < self.minimum_utterance_rms:
            self._log(
                "[SoulX] ⏭️ speak 音频能量过低，忽略远场/背景语音: "
                f"rms={stats['rms']:.5f} < "
                f"{self.minimum_utterance_rms:.5f}, "
                f"text_chars={len(soulx_state.text or '')}"
            )
            return

        if self.speaker.enabled:
            is_target, similarity = await self.speaker.verify(
                complete_audio,
                self.sample_rate,
                source="soulx_speak",
            )
            if not is_target:
                self._log(
                    "[SoulX] ⏭️ speak 未通过声纹验证: "
                    f"similarity={similarity}"
                )
                return

        if self._is_ai_speaking():
            await self._await_if_needed(self._stop_playback())

        if (
            self.answer_completion.enabled
            and not self._is_ai_speaking()
        ):
            total_duration = self.answer_completion.append_segment(
                complete_audio
            )
            notification = self._notify_answer_completion(
                "captured",
                "SoulX 已捕获一段语义完整回答，系统继续按慢答模式观察。",
                segments=len(self.answer_completion.segments),
                duration_s=total_duration,
            )
            await self._await_if_needed(notification)
            await self._await_if_needed(
                self._arm_answer_completion_window()
            )
            return

        self._log(
            f"[SoulX] 🎯 speak：提交完整语音 {speech_duration:.2f}s"
        )
        await self.connection.send_json({"type": "vad_end"})
        extra_meta = {
            "turn_taking": "soulx",
            "soulx_text": soulx_state.text,
            "soulx_asr_buffer": soulx_state.asr_buffer,
        }
        detail_state = str(
            getattr(soulx_state, "detail_state", "") or ""
        )
        decision_source = str(
            getattr(soulx_state, "decision_source", "") or ""
        )
        raw_state = str(getattr(soulx_state, "raw_state", "") or "")
        if raw_state or detail_state or decision_source:
            extra_meta.update(
                {
                    "soulx_raw_state": raw_state,
                    "soulx_detail_state": detail_state,
                    "soulx_decision_source": decision_source,
                    "soulx_wait_idle_count": int(
                        getattr(soulx_state, "wait_idle_count", 0) or 0
                    ),
                    "soulx_max_wait_count": int(
                        getattr(soulx_state, "max_wait_count", 0) or 0
                    ),
                }
            )
        submission = self._submit_speech(
            complete_audio,
            source="SoulX语义轮次",
            extra_meta=extra_meta,
        )
        await self._await_if_needed(submission)

    def _should_gate_barge_in_by_speaker(self) -> bool:
        verifier = self.speaker.verifier
        return bool(
            self.speaker.enabled
            and self.speaker.enrolled
            and verifier is not None
            and verifier.is_enrolled
        )

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result
