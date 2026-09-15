from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np

from src.tools.emotion import classify_realtime_emotion
from src.tools.voice.ark_asr import ArkASRStreamingSession


def _within_edit_distance(left: str, right: str, limit: int) -> bool:
    left, right = str(left or ""), str(right or "")
    if abs(len(left) - len(right)) > limit:
        return False
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        row_low = current[0]
        for other_index, other in enumerate(right, 1):
            value = min(
                previous[other_index] + 1,
                current[other_index - 1] + 1,
                previous[other_index - 1] + (char != other),
            )
            current.append(value)
            row_low = min(row_low, value)
        if row_low > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _normalise_for_echo(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(text or "")).casefold()


@dataclass(frozen=True)
class RealtimeCompanionConfig:
    enabled: bool
    external_asr: bool = False
    emotion_interval_s: float = 0.8
    emotion_window_s: float = 3.0
    asr_chunk_s: float = 0.2
    asr_queue_size: int = 100
    memory_timeout_s: float = 0.25
    memory_prefetch_stability_s: float = 0.35
    memory_prefetch_min_interval_s: float = 0.5


class RealtimeCompanion:
    """Run rolling emotion checks and optional streaming ASR per connection."""

    _RISK_PHRASES = ("自杀", "想死", "不想活", "结束生命")
    _LISTENING_TEXT = "我在听，您慢慢说。"
    _BACKCHANNEL_TEXT = "嗯嗯"
    _SAFETY_TEXT = "听到您这样说，我很担心您现在的安全。请先不要独处，找一位您信任的人陪在身边。"

    def __init__(
        self,
        connection,
        *,
        session,
        patient_memory_service,
        stream_tts_audio: Callable[..., Any],
        config: RealtimeCompanionConfig,
        stream_factory=ArkASRStreamingSession,
        classify_window=classify_realtime_emotion,
        now_factory: Callable[[], float] = time.time,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.patient_memory_service = patient_memory_service
        self._stream_tts_audio = stream_tts_audio
        self.config = config
        self._stream_factory = stream_factory
        self._classify_window = classify_window
        self._now = now_factory
        self._log = logger
        self._external_asr = bool(config.external_asr)
        self._active: RealtimeTurn | None = None
        self._comfort_count = 0
        self._last_comfort_at = float("-inf")
        self._risk_announced = False
        self._comfort_task = None
        self._comfort_kind = ""

    async def observe_frame(
        self,
        audio: np.ndarray,
        *,
        was_speaking: bool,
        is_speaking: bool,
        complete_audio: np.ndarray | None,
    ) -> "RealtimeTurn | None":
        if not self.config.enabled:
            return None
        state = self._active
        if not was_speaking and is_speaking:
            await self._cancel_comfort()
            self._comfort_count = 0
            if state is not None:
                await state.aclose()
            state = RealtimeTurn(self)
            self._active = state
            state.start(use_streaming_asr=not self._external_asr)
        if state is None:
            return None
        state.observe_audio(audio)
        if complete_audio is not None:
            state.finish()
            self._active = None
            return state
        return None

    def set_external_asr(self, enabled: bool) -> None:
        self._external_asr = bool(enabled)

    async def fallback_to_local_asr(self) -> None:
        if not self._external_asr:
            return
        self._external_asr = False
        state, self._active = self._active, None
        if state is not None:
            await state.aclose()
        if self._comfort_kind in {"incomplete", "backchannel"}:
            await self._cancel_comfort()

    async def observe_soulx_frame(
        self,
        audio: np.ndarray,
        *,
        state: str,
        text: str = "",
        detail_state: str = "",
        speech_detected: bool | None = None,
        complete_audio: np.ndarray | None = None,
    ) -> "RealtimeTurn | None":
        """Consume one SoulX frame without starting a second ASR stream."""
        if not self.config.enabled:
            return None
        normalized_state = str(state or "").strip().lower()
        is_speaking = (
            bool(self._active)
            or speech_detected is True
            or normalized_state in {"nonidle", "speak"}
        )
        was_speaking = bool(self._active)
        if not was_speaking and not is_speaking:
            return None
        if not was_speaking:
            await self._cancel_comfort()
            self._comfort_count = 0
            self._active = RealtimeTurn(self)
            self._active.start(use_streaming_asr=False)
        turn = self._active
        if turn is None:
            return None
        turn.speech_detected = speech_detected
        # incomplete_wait continues the same unfinished expression. A timeout
        # that commits a formal turn must never also start an acknowledgement.
        turn.semantic_incomplete = str(detail_state or "").strip().lower() in {
            "incomplete",
            "incomplete_wait",
        }
        if speech_detected is True and self.listening_prompt_playing:
            await self._cancel_comfort()
        if complete_audio is not None and not was_speaking:
            turn.observe_audio(complete_audio)
        else:
            turn.observe_audio(audio)
        if text:
            await turn.observe_external_text(
                text,
                is_final=normalized_state == "speak",
            )
        if complete_audio is not None:
            turn.finish()
            self._active = None
            return turn
        if normalized_state != "speak":
            await self.observe_semantic_incomplete()
        return None

    def _incomplete_response(self) -> tuple[str, str] | None:
        turn = self._active
        if (
            not self.config.enabled
            or turn is None
            or turn._finished
            or turn.echo_suspected
            or not turn.semantic_incomplete
        ):
            return None
        if turn.speech_detected is True and turn.audio_duration_s > 5.0:
            return self._BACKCHANNEL_TEXT, "backchannel"
        if turn.speech_detected is False and turn.audio_duration_s > 7.0:
            return self._LISTENING_TEXT, "incomplete"
        return None

    async def observe_semantic_incomplete(self) -> None:
        """Recheck duration and current speech on every unfinished SoulX frame."""
        response = self._incomplete_response()
        if response is not None:
            text, kind = response
            await self._schedule_comfort(text, kind, False)

    @property
    def backchannel_playing(self) -> bool:
        return self._comfort_is_playing("backchannel")

    @property
    def listening_prompt_playing(self) -> bool:
        return self._comfort_is_playing("incomplete")

    def _comfort_is_playing(self, kind: str) -> bool:
        return (
            self._comfort_kind == kind
            and self._comfort_task is not None
            and not self._comfort_task.done()
        )

    async def reset(self) -> None:
        state, self._active = self._active, None
        if state is not None:
            await state.aclose()
        await self._cancel_comfort()
        self._comfort_count = 0
        self._last_comfort_at = float("-inf")
        self._risk_announced = False

    async def close(self) -> None:
        await self.reset()

    async def _cancel_comfort(self) -> None:
        task, self._comfort_task = self._comfort_task, None
        self._comfort_kind = ""
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def observe_partial_emotion(
        self,
        text: str,
        scores: dict[str, float],
    ) -> None:
        # Ordinary acknowledgements are gated by SoulX and utterance duration.
        # Keep the existing immediate safety path for explicit risk expressions.
        await self.observe_partial_text(text)

    async def observe_partial_text(self, text: str) -> None:
        """Keep explicit safety alerts independent of ordinary acknowledgements."""
        if not self._active or self._active.echo_suspected:
            return
        if self._is_risk_text(text):
            await self._schedule_comfort(self._SAFETY_TEXT, "safety", True)

    async def _schedule_comfort(
        self,
        text: str,
        kind: str,
        safety: bool,
    ) -> None:
        now = self._now()
        if safety:
            if self._risk_announced:
                return
            self._risk_announced = True
            self.session.runtime.high_risk_detected = True
            await self.connection.send_json(
                {"type": "safety_alert", "text": text, "severity": "high"}
            )
        elif (
            self._comfort_count >= 2
            or now - self._last_comfort_at < 10.0
            or (self._comfort_task is not None and not self._comfort_task.done())
        ):
            return
        else:
            self._comfort_count += 1
            self._last_comfort_at = now
        self._comfort_kind = kind
        self._comfort_task = asyncio.create_task(
            self._play_comfort(text, kind, safety, turn=self._active)
        )

    async def _play_comfort(
        self,
        text: str,
        kind: str,
        safety: bool,
        *,
        turn: "RealtimeTurn | None" = None,
    ) -> None:
        try:
            if kind in {"incomplete", "backchannel"} and (
                turn is not self._active
                or self._incomplete_response() != (text, kind)
            ):
                return
            await self.connection.send_json(
                {
                    "type": "companion_message",
                    "text": text,
                    "kind": kind,
                    "safety": safety,
                }
            )
            current_identity = getattr(
                self.connection,
                "current_output_identity",
                None,
            )
            identity = current_identity() if callable(current_identity) else None
            stream_kwargs = {
                "content_text": text,
                "emotion": "gentle",
                "label": f"-realtime-{kind}",
                "event_type": "tts_chunk",
                "allow_interrupt": True,
                "include_dtype": True,
                "persist": False,
                "companion": True,
                "cancel_as_result": True,
                "start_payload": {
                    "type": "tts_start",
                    "text": text,
                    "kind": kind,
                },
            }
            if identity:
                stream_kwargs.update(
                    {
                        "session_id": identity.get("session_id"),
                        "turn_id": identity.get("turn_id"),
                        "generation": identity.get("generation"),
                    }
                )
            result = await self._stream_tts_audio(
                text,
                **stream_kwargs,
            )
            duration = result["samples"] / 24000.0 if result["samples"] else 0.0
            end_payload = {
                "type": "tts_end",
                "duration": duration,
                "kind": kind,
            }
            output_identity = result.get("output_identity")
            if isinstance(output_identity, dict) and result.get("started"):
                end_payload.update(output_identity)
                if result.get("interrupted"):
                    end_payload["reason"] = result.get(
                        "interruption_reason"
                    ) or "cancelled"
                await self.connection.send_json(end_payload)
        except Exception as exc:
            self._log(f"[RealtimeCompanion] 安抚播报失败: {type(exc).__name__}")

    @classmethod
    def _is_risk_text(cls, text: str) -> bool:
        return any(phrase in str(text or "") for phrase in cls._RISK_PHRASES)


class RealtimeTurn:
    """State that survives from a detected speech start through final confirmation."""

    def __init__(self, companion: RealtimeCompanion) -> None:
        self.companion = companion
        self._queue = asyncio.Queue(maxsize=companion.config.asr_queue_size)
        self._window: deque[np.ndarray] = deque()
        self._window_samples = 0
        self._audio_samples = 0
        self.speech_detected: bool | None = None
        self.semantic_incomplete = False
        self._partial_history: deque[tuple[float, str]] = deque()
        self._stream_task = None
        self._emotion_task = None
        self._emotion_generation = 0
        self._next_emotion_at = 0.0
        self._effective_emotion_interval = companion.config.emotion_interval_s
        self._finished = False
        self._stream_error: Exception | None = None
        self._final_text = ""
        self._final_received = False
        self.partial_text = ""
        self.echo_suspected = False
        self._memory_query_id = 0
        self._memory_query_text = ""
        self._memory_task = None
        self._memory_result = ""
        self._memory_query_started_at = float("-inf")

    def start(self, *, use_streaming_asr: bool = True) -> None:
        if use_streaming_asr:
            self._stream_task = asyncio.create_task(self._run_stream())

    @property
    def audio_duration_s(self) -> float:
        """Audio since this utterance began, including pauses but not pre-roll."""
        return self._audio_samples / 16000.0

    def observe_audio(self, audio: np.ndarray) -> None:
        if self._finished:
            return
        samples = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if not samples.size:
            return
        self._audio_samples += len(samples)
        self._window.append(samples)
        self._window_samples += len(samples)
        keep_samples = int(self.companion.config.emotion_window_s * 16000)
        while self._window and self._window_samples - len(self._window[0]) >= keep_samples:
            self._window_samples -= len(self._window.popleft())
        if self._stream_task is not None:
            try:
                self._queue.put_nowait(samples)
            except asyncio.QueueFull:
                self._stream_error = RuntimeError("实时 ASR 队列积压")
        self._schedule_emotion()

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._stream_task is None:
            return
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            self._stream_error = RuntimeError("实时 ASR 队列积压")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except asyncio.QueueEmpty:
                pass

    async def aclose(self) -> None:
        self._finished = True
        self.cancel_memory_prefetch()
        task, self._stream_task = self._stream_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        emotion_task, self._emotion_task = self._emotion_task, None
        if emotion_task is not None and not emotion_task.done():
            emotion_task.cancel()
            await asyncio.gather(emotion_task, return_exceptions=True)

    async def final_text(self) -> str:
        if self._stream_task is not None:
            await asyncio.gather(self._stream_task, return_exceptions=True)
        return self._final_text

    @property
    def stream_error(self) -> Exception | None:
        return self._stream_error

    @property
    def has_final(self) -> bool:
        """Only an explicit final ASR packet is safe for the formal turn."""
        return self._final_received

    async def memory_for_final(self, text: str) -> str:
        if self.echo_suspected or not getattr(
            self.companion.session, "long_term_memory_enabled", True
        ):
            self.cancel_memory_prefetch()
            return ""
        final_text = str(text or "").strip()
        if not final_text:
            return ""
        if (
            self._memory_task is not None
            and not self._memory_task.cancelled()
            and _normalise_for_echo(final_text) == _normalise_for_echo(self._memory_query_text)
        ):
            task = self._memory_task
        else:
            task = self._start_memory_query(final_text)
        if task is None:
            return ""
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(0.01, float(self.companion.config.memory_timeout_s)),
            )
            return self._memory_result
        except asyncio.TimeoutError:
            self.companion._log("[RealtimeCompanion] 最终记忆检索超时，跳过长期事件")
            self.cancel_memory_prefetch()
            return ""
        return ""

    async def _run_stream(self) -> None:
        stream = None
        pending = np.empty(0, dtype=np.float32)
        chunk_samples = max(
            1,
            int(self.companion.config.asr_chunk_s * 16000),
        )
        try:
            stream = self.companion._stream_factory(on_result=self._on_result)
            await stream.start()
            while True:
                item = await self._queue.get()
                if item is None:
                    if pending.size:
                        await stream.feed(pending)
                    self._final_text = await stream.finish()
                    return
                pending = np.concatenate((pending, item))
                while pending.size >= chunk_samples:
                    await stream.feed(pending[:chunk_samples])
                    pending = pending[chunk_samples:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stream_error = exc
            self.companion._log(f"[RealtimeCompanion] 流式 ASR 失败: {exc}")
        finally:
            if stream is not None:
                await stream.aclose()

    async def _on_result(self, text: str, is_final: bool) -> None:
        await self._consume_text(text, is_final=is_final, emit=True)

    async def observe_external_text(self, text: str, *, is_final: bool) -> None:
        await self._consume_text(text, is_final=is_final, emit=False)

    async def _consume_text(
        self,
        text: str,
        *,
        is_final: bool,
        emit: bool,
    ) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self.partial_text = text
        if is_final:
            self._final_text = text
            self._final_received = True
        now = self.companion._now()
        self._partial_history.append((now, text))
        while self._partial_history and self._partial_history[0][0] < now - 3.0:
            self._partial_history.popleft()
        self.echo_suspected = self._is_suspected_echo(text)
        if self.echo_suspected:
            if emit:
                await self.companion.connection.send_json(
                    {"type": "asr_echo_suspected", "text": text}
                )
            return
        if emit:
            await self.companion.connection.send_json(
                {"type": "asr_partial", "text": text, "final": bool(is_final)}
            )
        if not is_final:
            await self.companion.observe_partial_text(text)
        self._maybe_prefetch(text, now)

    def _maybe_prefetch(self, text: str, now: float) -> None:
        if (
            len(text) < 5
            or self.echo_suspected
            or not getattr(self.companion.session, "long_term_memory_enabled", True)
        ):
            return
        if self._memory_task is not None and not self._memory_task.done():
            return
        if now - self._memory_query_started_at < max(
            0.05, self.companion.config.memory_prefetch_min_interval_s
        ):
            return
        prior = next(
            (
                value
                for timestamp, value in reversed(self._partial_history)
                if timestamp <= now - max(
                    0.05, self.companion.config.memory_prefetch_stability_s
                )
            ),
            "",
        )
        if not prior or not _within_edit_distance(text, prior, 2):
            return
        if (
            self._memory_task is not None
            and not self._memory_task.cancelled()
            and _normalise_for_echo(text) == _normalise_for_echo(self._memory_query_text)
        ):
            return
        self._start_memory_query(text)

    def _start_memory_query(self, text: str):
        patient_id = self.companion.session.lifecycle.current_patient_id
        if (
            not patient_id
            or self.companion.patient_memory_service is None
            or not getattr(self.companion.session, "long_term_memory_enabled", True)
        ):
            return None
        self.cancel_memory_prefetch()
        self._memory_query_id += 1
        query_id = self._memory_query_id
        self._memory_query_text = text
        self._memory_query_started_at = self.companion._now()
        self._memory_task = asyncio.create_task(
            self._retrieve_memory(query_id, text)
        )
        return self._memory_task

    def cancel_memory_prefetch(self) -> None:
        self._memory_query_id += 1
        self._memory_result = ""
        task, self._memory_task = self._memory_task, None
        if task is not None and not task.done():
            task.cancel()

    async def _retrieve_memory(self, query_id: int, text: str) -> str:
        try:
            result = await asyncio.to_thread(
                self.companion.patient_memory_service.get_turn_context,
                self.companion.session,
                text,
            )
        except Exception as exc:
            self.companion._log(f"[RealtimeCompanion] 记忆预取失败: {exc}")
            result = ""
        if query_id == self._memory_query_id:
            self._memory_result = str(result or "")
        return str(result or "")

    def _schedule_emotion(self) -> None:
        now = self.companion._now()
        if now < self._next_emotion_at or not self._window:
            return
        self._next_emotion_at = now + self._effective_emotion_interval
        self._emotion_generation += 1
        generation = self._emotion_generation
        if self._emotion_task is not None and not self._emotion_task.done():
            return
        self._start_emotion(generation)

    def _start_emotion(self, generation: int) -> None:
        window = np.concatenate(tuple(self._window))
        text = self.partial_text
        self._emotion_task = asyncio.create_task(
            self._infer_emotion(generation, text, window)
        )

    async def _infer_emotion(
        self,
        generation: int,
        text: str,
        window: np.ndarray,
    ) -> None:
        started = self.companion._now()
        scores = None
        try:
            scores = await asyncio.to_thread(
                self.companion._classify_window,
                text,
                window,
                16000,
            )
        except Exception as exc:
            self.companion._log(f"[RealtimeCompanion] 滚动情绪失败: {exc}")
        finally:
            elapsed = self.companion._now() - started
            self._effective_emotion_interval = max(
                self.companion.config.emotion_interval_s,
                elapsed,
            )
            self._next_emotion_at = max(
                self._next_emotion_at,
                self.companion._now() + self._effective_emotion_interval,
            )
        if self._finished:
            return
        if generation != self._emotion_generation:
            self._start_emotion(self._emotion_generation)
            return
        if scores is None:
            return
        await self.companion.observe_partial_emotion(
            text,
            {key: float(value) for key, value in dict(scores or {}).items()},
        )

    def _is_suspected_echo(self, text: str) -> bool:
        runtime = self.companion.session.runtime
        if not runtime.ai_streaming_tts:
            return False
        candidate = _normalise_for_echo(text)
        if len(candidate) < 4 or self._window_samples < 5600:
            return False
        recent = getattr(runtime, "recent_tts_texts", [])
        now = self.companion._now()
        runtime.recent_tts_texts = [
            item for item in recent if now - item.get("started_at", now) <= 10.0
        ]
        for item in runtime.recent_tts_texts:
            spoken = _normalise_for_echo(item.get("text", ""))
            if not spoken:
                continue
            if SequenceMatcher(None, candidate, spoken).ratio() >= 0.88:
                estimated = max(1.0, len(spoken) * 0.18)
                return self._window_samples / 16000.0 <= estimated + 1.2
        return False
