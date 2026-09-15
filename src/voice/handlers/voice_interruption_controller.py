from __future__ import annotations

from collections.abc import Callable
from typing import Any
import asyncio
import numpy as np
import time

class VoiceInterruptionController:
    """Own playback interruption, cached judgements and turn-taking resets."""

    _INTERRUPT_PHRASES = {
        "等一下",
        "等等",
        "停",
        "停一下",
        "等下",
        "慢着",
        "别说了",
        "停下",
        "打断一下",
    }

    def __init__(
        self,
        connection,
        *,
        session,
        vad_buffer,
        soulx_client,
        soulx_audio,
        quick_asr: Callable[..., Any],
        judge_answer_completion: Callable[..., Any],
        judge_interrupt_intent: Callable[..., Any],
        extract_latest_assistant_utterance: Callable[[list], str],
        normalize_interrupt_text: Callable[[str], str],
        soulx_session_id_factory: Callable[[], str],
        cancel_render_jobs: Callable[..., Any] | None = None,
        now_factory: Callable[[], float] = time.time,
        playback_stop_delay_s: float = 0.08,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.vad_buffer = vad_buffer
        self.soulx_client = soulx_client
        self.soulx_audio = soulx_audio
        self._quick_asr = quick_asr
        self._judge_answer_completion = judge_answer_completion
        self._judge_interrupt_intent = judge_interrupt_intent
        self._extract_latest_assistant_utterance = (
            extract_latest_assistant_utterance
        )
        self._normalize_interrupt_text = normalize_interrupt_text
        self._soulx_session_id_factory = soulx_session_id_factory
        self._cancel_render_jobs = cancel_render_jobs
        self._now = now_factory
        self.playback_stop_delay_s = float(playback_stop_delay_s)
        self._log = logger

    def is_ai_speaking(self, current_time=None) -> bool:
        now = self._now() if current_time is None else current_time
        return self.session.runtime.is_ai_speaking(now)

    def reset_capture(
        self,
        reset_waiting: bool = False,
        reset_vad: bool = False,
    ) -> None:
        self.session.runtime.reset_interrupt_capture(
            reset_waiting=reset_waiting
        )
        if reset_vad:
            self.vad_buffer.reset()

    def remember_judgement(
        self,
        audio_data: np.ndarray,
        text: str,
        intent: str,
    ) -> None:
        self.session.runtime.remember_interrupt_judgement(
            audio_samples=len(audio_data),
            text=text,
            intent=intent,
        )

    def reuse_judgement(
        self,
        audio_data: np.ndarray,
    ) -> tuple[str | None, str | None]:
        return self.session.runtime.reuse_interrupt_judgement(
            audio_samples=len(audio_data)
        )

    async def reset_turn_taking(self, reason: str = "") -> None:
        self.soulx_audio.reset()
        self.session.turn_taking.reset()
        if self.soulx_client is None:
            return
        result = self.soulx_client.reset(
            self._soulx_session_id_factory()
        )
        if hasattr(result, "__await__"):
            await result
        if reason:
            self._log(f"[SoulX] 🔄 已重置轮次会话: {reason}")

    async def judge_waiting_completion(
        self,
        audio_data: np.ndarray,
        cached_text: str | None = None,
    ) -> tuple[str, str]:
        text = str(cached_text or "").strip()
        if not text:
            result = self._quick_asr(audio_data)
            text = await result if hasattr(result, "__await__") else result
            text = str(text or "").strip()
        if not text or len(text) < 2:
            return text, "too_short"

        normalized_text = self._normalize_interrupt_text(text)
        if (
            normalized_text in self._INTERRUPT_PHRASES
            or any(
                phrase in normalized_text
                for phrase in self._INTERRUPT_PHRASES
            )
        ):
            return text, "complete"

        question_text = self._extract_latest_assistant_utterance(
            self.session.chat_history
        )
        if question_text:
            judgement = self._judge_answer_completion(
                question_text,
                text,
            )
            if hasattr(judgement, "__await__"):
                judgement = await judgement
            if judgement["label"] in {
                "likely_complete",
                "ask_repeat",
                "explicit_no_answer",
            }:
                self._log(
                    "[全双工] 🧠 等待补全按作答完整处理: "
                    f"label={judgement['label']}, text_chars={len(text)}"
                )
                return text, "complete"
            if judgement["label"] == "incomplete":
                self._log(
                    "[全双工] 🧠 等待补全按作答未完成处理: "
                    f"text_chars={len(text)}"
                )
                return text, "incomplete"

        intent = self._judge_interrupt_intent(text)
        if hasattr(intent, "__await__"):
            intent = await intent
        self._log(
            f"[全双工] 🧠 等待补全回退到打断判定: "
            f"{intent}, text_chars={len(text)}"
        )
        return text, intent

    async def stop_playback(self) -> None:
        stop_requested_at_ms = round(self._now() * 1000, 1)
        runtime = self.session.runtime
        processing = self.session.processing
        if self._cancel_render_jobs is not None:
            self._cancel_render_jobs(session=self.session)
        runtime.stop_generate = True
        runtime.ai_streaming_tts = False
        runtime.ai_speaking_until = 0.0
        processing.generation += 1
        turn_id = str(
            processing.active_turn_id
            or f"system-{self.session.session_id or 'session'}"
        )
        processing.active_playback_id = (
            f"playback-{turn_id}-{processing.generation}"
        )
        output_identity = {
            "session_id": self.session.session_id,
            "turn_id": turn_id,
            "generation": processing.generation,
            "playback_id": processing.active_playback_id,
            "server_stop_at_ms": stop_requested_at_ms,
        }
        if callable(getattr(self.connection, "current_output_identity", None)):
            await self.connection.send_json({"type": "stop_tts", **output_identity})
            await self.connection.send_json({"type": "interrupt", **output_identity})
        else:
            await self.connection.send_json({"type": "stop_tts"})
            await self.connection.send_json({"type": "interrupt"})
        self._log(
            f"[全双工停播] server_stop_at_ms={stop_requested_at_ms}, "
            f"server_sent_at_ms={self._now() * 1000:.1f}, "
            f"generation={processing.generation}"
        )
        await asyncio.sleep(self.playback_stop_delay_s)
