from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .live_audio_config import LiveAudioConfig
from .live_audio_frame_context import LiveAudioFrameContext
from .soulx_audio_handler import SoulXAudioHandler


class LiveAudioInputHandler:
    """Route live microphone frames through SoulX, VAD and interruption flows."""

    MESSAGE_TYPES = {"audio"}
    _INTERRUPT_PHRASES = (
        "等一下",
        "等等",
        "停",
        "停一下",
        "等下",
        "慢着",
        "别说了",
        "停下",
        "打断一下",
    )

    def __init__(
        self,
        connection,
        *,
        session,
        vad_buffer,
        soulx_audio_handler: SoulXAudioHandler,
        speaker,
        answer_completion,
        history_store,
        is_ai_speaking: Callable[..., bool],
        reset_interrupt_capture: Callable[..., Any],
        remember_interrupt_judgement: Callable[..., Any],
        reuse_interrupt_judgement: Callable[..., Any],
        judge_waiting_completion: Callable[..., Any],
        interrupt_active_processing: Callable[..., Any],
        stop_playback: Callable[..., Any],
        get_active_processing_audio: Callable[[], Any],
        notify_voice_input_feedback: Callable[..., Any],
        submit_speech: Callable[..., Any],
        notify_answer_completion: Callable[..., Any],
        arm_answer_completion_window: Callable[..., Any],
        quick_asr: Callable[..., Any],
        judge_interrupt_intent: Callable[..., Any],
        stream_tts_audio: Callable[..., Any],
        config: LiveAudioConfig,
        logger=print,
        realtime_companion=None,
        ensure_session_on_speech: Callable[..., Any] | None = None,
    ) -> None:
        self.connection = connection
        self.session = session
        self.vad_buffer = vad_buffer
        self.soulx_audio_handler = soulx_audio_handler
        self.speaker = speaker
        self.answer_completion = answer_completion
        self.history_store = history_store
        self._is_ai_speaking = is_ai_speaking
        self._reset_interrupt_capture = reset_interrupt_capture
        self._remember_interrupt_judgement = (
            remember_interrupt_judgement
        )
        self._reuse_interrupt_judgement = reuse_interrupt_judgement
        self._judge_waiting_completion = judge_waiting_completion
        self._interrupt_active_processing = interrupt_active_processing
        self._stop_playback = stop_playback
        self._get_active_processing_audio = get_active_processing_audio
        self._notify_voice_input_feedback = notify_voice_input_feedback
        self._submit_speech = submit_speech
        self._notify_answer_completion = notify_answer_completion
        self._arm_answer_completion_window = (
            arm_answer_completion_window
        )
        self._quick_asr = quick_asr
        self._judge_interrupt_intent = judge_interrupt_intent
        self._stream_tts_audio = stream_tts_audio
        self.config = config
        self._log = logger
        self._realtime_companion = realtime_companion
        self._ensure_session_on_speech = ensure_session_on_speech

    async def handle_message(self, message: dict) -> bool:
        """Orchestrate one frame without embedding the state machines inline."""
        if message.get("type") not in self.MESSAGE_TYPES:
            return False

        audio = message.get("_audio_float")
        if self._should_discard_audio(audio):
            return True
        if await self._route_soulx_audio(audio):
            return True

        frame = self._feed_local_vad(audio)
        if not await self._start_pending_session_on_speech(frame):
            return True
        await self._observe_realtime_frame(audio, frame)
        if await self._apply_local_vad_gates(frame):
            return True
        if await self._monitor_interruptions(frame):
            return True
        if frame.complete_audio is not None:
            await self._handle_complete_audio(frame)
        return True

    def _should_discard_audio(self, audio) -> bool:
        lifecycle = self.session.lifecycle
        return audio is None or (
            not lifecycle.started
            and not getattr(lifecycle, "awaiting_next_utterance", False)
        )

    async def _route_soulx_audio(self, audio) -> bool:
        realtime_enabled = bool(
            getattr(getattr(self._realtime_companion, "config", None), "enabled", False)
        )
        return not realtime_enabled and await self.soulx_audio_handler.handle_audio(audio)

    async def _start_pending_session_on_speech(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        lifecycle = self.session.lifecycle
        if not (
            not frame.was_speaking
            and self.vad_buffer.is_speaking
            and getattr(lifecycle, "awaiting_next_utterance", False)
            and self._ensure_session_on_speech is not None
        ):
            return True
        started = self._ensure_session_on_speech()
        if hasattr(started, "__await__"):
            started = await started
        if started:
            return True
        self.vad_buffer.reset()
        return False

    async def _observe_realtime_frame(
        self,
        audio,
        frame: LiveAudioFrameContext,
    ) -> None:
        if self._realtime_companion is None:
            return
        frame.realtime_turn = await self._realtime_companion.observe_frame(
            audio,
            was_speaking=frame.was_speaking,
            is_speaking=(
                frame.was_speaking
                or self.vad_buffer.is_speaking
                or frame.complete_audio is not None
            ),
            complete_audio=frame.complete_audio,
        )

    def _feed_local_vad(self, audio) -> LiveAudioFrameContext:
        frame = LiveAudioFrameContext(
            audio=audio,
            was_speaking=self.vad_buffer.is_speaking,
        )
        frame.complete_audio = self.vad_buffer.add_chunk(audio)
        (
            frame.drop_reason,
            frame.drop_duration_s,
        ) = self.vad_buffer.consume_drop_feedback()
        return frame

    async def _apply_local_vad_gates(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        runtime = self.session.runtime
        processing = self.session.processing
        frame.current_time = time.time()
        frame.in_ai_speaking = self._is_ai_speaking(frame.current_time)
        frame.early_listen_mode = (
            not self.config.enabled_full_duplex
            and frame.in_ai_speaking
            and not runtime.waiting_for_complete
            and (
                runtime.ai_speaking_until - frame.current_time
            )
            <= self.config.pre_end_arm_window_s
        )

        self._arm_early_listening(frame)
        if self._should_block_for_half_duplex(frame):
            return True
        if self._should_block_during_processing(frame):
            return True

        await self._report_vad_drop(frame)
        self._track_early_listening(frame)
        if self._reject_short_early_capture(frame):
            return True
        await self._finalize_early_capture(frame)

        if not self.config.enabled_full_duplex and frame.in_ai_speaking:
            return True
        if self.answer_completion.enabled and frame.in_ai_speaking:
            self._reset_interrupt_capture(reset_waiting=True)
            return True
        return False

    def _arm_early_listening(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        runtime = self.session.runtime
        if not frame.early_listen_mode or runtime.early_vad_armed:
            return
        runtime.early_vad_armed = True
        runtime.early_vad_started_during_ai = False
        runtime.early_vad_post_tts_chunks = 0
        self.vad_buffer.reset()
        remaining = max(
            runtime.ai_speaking_until - frame.current_time,
            0.0,
        )
        self._log(
            f"[VAD] 🔓 提前监听已开启，距离TTS结束约 {remaining:.2f}s"
        )

    def _should_block_for_half_duplex(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        runtime = self.session.runtime
        should_block = (
            not self.config.enabled_full_duplex
            and (
                frame.in_ai_speaking
                or runtime.waiting_for_complete
            )
            and not frame.early_listen_mode
        )
        if not should_block:
            return False
        if runtime.waiting_for_complete or runtime.interrupt_audio_buffer:
            self._reset_interrupt_capture(reset_waiting=True)
        if (
            frame.in_ai_speaking
            and runtime.early_vad_armed
            and not self.vad_buffer.is_speaking
        ):
            self._reset_early_listening()
        return True

    def _should_block_during_processing(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        processing = self.session.processing
        runtime = self.session.runtime
        if not processing.is_active or runtime.waiting_for_complete:
            return False
        return not (
            self.config.enabled_full_duplex
            and (
                frame.in_ai_speaking
                or processing.revision_enabled
            )
        )

    async def _report_vad_drop(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        if frame.drop_reason != "speech_too_short":
            return
        await self._notify_voice_input_feedback(
            "speech_too_short",
            "刚才这句太短或像噪音，系统没有处理。请靠近麦克风后再完整说一遍。",
            status_text="语音太短或不清晰，请再说一遍",
            duration_s=frame.drop_duration_s,
        )

    def _track_early_listening(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        runtime = self.session.runtime
        frame.current_time = time.time()
        frame.in_ai_speaking = self._is_ai_speaking(frame.current_time)
        if (
            runtime.early_vad_armed
            and frame.in_ai_speaking
            and not frame.was_speaking
            and self.vad_buffer.is_speaking
        ):
            runtime.early_vad_started_during_ai = True
            runtime.early_vad_post_tts_chunks = 0
            self._log("[VAD] 🎯 已在TTS尾段捕获到用户起始语音")
        elif (
            runtime.early_vad_started_during_ai
            and not frame.in_ai_speaking
            and self.vad_buffer.is_speaking
            and self.vad_buffer.has_speech(frame.audio)
            >= self.vad_buffer.WEAK_SPEECH_THRESHOLD
        ):
            runtime.early_vad_post_tts_chunks += 1

    def _reject_short_early_capture(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        runtime = self.session.runtime
        if (
            frame.complete_audio is None
            or not runtime.early_vad_started_during_ai
            or runtime.early_vad_post_tts_chunks
            >= self.config.minimum_post_tts_chunks
        ):
            return False
        self._log(
            "[VAD] ⏭️ 提前监听片段在TTS结束后持续不足 "
            f"{runtime.early_vad_post_tts_chunks} 块，丢弃"
        )
        self._reset_early_listening()
        return True

    async def _finalize_early_capture(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        runtime = self.session.runtime
        if (
            frame.complete_audio is not None
            and not frame.in_ai_speaking
            and not runtime.waiting_for_complete
        ):
            if not self.answer_completion.enabled:
                await self._send_vad_end(frame)
            if runtime.early_vad_started_during_ai:
                self._log(
                    "[VAD] 🕒 早听模式下捕获到完整语音，"
                    "等待 TTS 完全结束后再处理"
                )
            runtime.early_vad_started_during_ai = False
            runtime.early_vad_post_tts_chunks = 0
        elif (
            not frame.in_ai_speaking
            and runtime.early_vad_armed
            and not self.vad_buffer.is_speaking
        ):
            self._reset_early_listening()

    def _reset_early_listening(self) -> None:
        runtime = self.session.runtime
        runtime.early_vad_armed = False
        runtime.early_vad_started_during_ai = False
        runtime.early_vad_post_tts_chunks = 0

    async def _monitor_interruptions(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        runtime = self.session.runtime
        processing = self.session.processing
        should_monitor = (
            self._is_ai_speaking()
            or runtime.waiting_for_complete
            or (
                processing.is_active
                and processing.revision_enabled
            )
        )
        if not should_monitor:
            return False

        speech_probability = self.vad_buffer.has_speech(frame.audio)
        frame.current_time = time.time()
        chunk_rms = self.vad_buffer._chunk_rms(frame.audio[:512])
        has_interrupt_speech = (
            speech_probability
            >= self.config.interrupt_trigger_probability
            and chunk_rms >= self.config.interrupt_min_rms
        )
        if has_interrupt_speech:
            return await self._capture_interrupt_speech(
                frame,
                speech_probability,
                chunk_rms,
            )

        await self._handle_interrupt_silence(frame)
        return False

    async def _capture_interrupt_speech(
        self,
        frame: LiveAudioFrameContext,
        speech_probability: float,
        chunk_rms: float,
    ) -> bool:
        runtime = self.session.runtime
        runtime.interrupt_audio_buffer.append(frame.audio)
        runtime.interrupt_speech_run += 1
        runtime.last_speech_time = frame.current_time
        if (
            runtime.interrupt_speech_run
            < self.config.interrupt_min_consecutive_chunks
        ):
            return True

        duration = (
            sum(len(audio) for audio in runtime.interrupt_audio_buffer)
            / 16000
        )
        should_judge = (
            duration >= self.config.interrupt_min_duration
            and not runtime.waiting_for_complete
        )
        if not should_judge:
            return False

        self._log(
            "\n[全双工] 检测到用户声音，"
            f"累积 {duration:.2f}s，prob={speech_probability:.2f}, "
            f"rms={chunk_rms:.4f}，进行意图判断..."
        )
        full_audio = np.concatenate(runtime.interrupt_audio_buffer)
        if not await self._verify_interrupt_speaker(full_audio):
            self._reset_interrupt_capture(reset_waiting=True)
            return True

        text = await self._quick_asr(full_audio)
        if not text or len(text.strip()) < 2:
            self._log("[全双工] ⚠️ 识别失败或文本过短，可能是噪音，忽略")
            await self._notify_voice_input_feedback(
                "speech_too_short",
                "刚才这句太短或不够清晰，没有触发打断。您可以再完整说一遍。",
                status_text="打断语音太短，请再完整说一遍",
                duration_s=duration,
                source="interrupt_buffer",
            )
            self._reset_interrupt_capture(reset_waiting=True)
            return True

        self._log(f"[全双工] 识别到 text_chars={len(text)}")
        intent = await self._judge_interrupt_intent(text)
        self._remember_interrupt_judgement(full_audio, text, intent)
        await self._apply_interrupt_intent(text, intent)
        return False

    async def _verify_interrupt_speaker(self, audio) -> bool:
        if not self.speaker.enabled:
            return True
        is_target, similarity = await self.speaker.verify(
            audio,
            source="interrupt_buffer",
        )
        if is_target:
            return True
        if np.isfinite(similarity):
            self._log(
                "[全双工] ⏭️ 非目标说话人 "
                f"(相似度: {similarity:.2f})，忽略打断"
            )
        else:
            self._log(
                "[全双工] ⛔ 声纹验证未通过或不可用，忽略打断"
            )
        return False

    async def _apply_interrupt_intent(
        self,
        text: str,
        intent: str,
    ) -> None:
        runtime = self.session.runtime
        processing = self.session.processing
        if intent == "backchannel":
            self._log("[全双工] ❌ 应答词，不打断")
            self._reset_interrupt_capture(reset_waiting=True)
            return

        if intent == "incomplete":
            self._log(
                "\n[全双工打断] 🛑 检测到用户说话（不完整），"
                "先停止AI播放..."
            )
            if processing.is_active and processing.revision_enabled:
                await self._interrupt_active_processing(
                    "检测到用户继续补充（未说完）"
                )
            else:
                await self._stop_playback()
            runtime.waiting_for_complete = True
            runtime.interrupt_speech_run = 0
            self._log("[全双工] ⏳ AI已停止，等待用户说完整...")
            return

        if intent != "complete":
            return
        self._log("\n[全双工打断] 🛑 检测到有效打断（完整句子）！")
        if processing.is_active and processing.revision_enabled:
            await self._interrupt_active_processing(
                "检测到用户新的完整表达"
            )
        else:
            await self._stop_playback()
        if self._is_interrupt_request(text):
            await self._send_interrupt_confirmation(
                label="-interrupt-confirm"
            )
        self._reset_interrupt_capture(
            reset_waiting=True,
            reset_vad=True,
        )
        self._log("[全双工打断] ✅ 打断完成，等待用户新输入...")

    async def _handle_interrupt_silence(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        runtime = self.session.runtime
        runtime.interrupt_speech_run = 0
        if not runtime.interrupt_audio_buffer:
            return
        if not runtime.waiting_for_complete:
            self._reset_interrupt_capture()
            return

        silence_duration = frame.current_time - runtime.last_speech_time
        if (
            silence_duration
            <= self.config.interrupt_complete_silence_s
        ):
            return
        self._log(
            f"[全双工] 检测到停顿 {silence_duration:.2f}s，再次判断..."
        )
        full_audio = np.concatenate(runtime.interrupt_audio_buffer)
        cached_text, _ = self._reuse_interrupt_judgement(full_audio)
        text, intent = await self._judge_waiting_completion(
            full_audio,
            cached_text=cached_text,
        )
        if text:
            self._remember_interrupt_judgement(full_audio, text, intent)
        if not text or not intent:
            return
        if intent != "complete":
            self._log(
                "[全双工] ❌ 停顿后仍然不完整或是应答词，"
                f"放弃: text_chars={len(text)}"
            )
            self._reset_interrupt_capture(reset_waiting=True)
            return

        self._log(
            f"\n[全双工] ✅ 停顿后判定为完整句子，"
            f"text_chars={len(text)}"
        )
        runtime.waiting_for_complete = False
        if self._is_interrupt_request(text):
            await self._send_interrupt_confirmation(
                label="-interrupt-confirm-complete"
            )
            self._reset_interrupt_capture(
                reset_waiting=True,
                reset_vad=True,
            )
            frame.complete_audio = None
            return

        await self._send_vad_end(frame)
        self._log("[全双工] 🎯 开始处理补全后的完整语音")
        await self._submit_speech(
            full_audio,
            source="全双工补全语音",
        )
        self._reset_interrupt_capture(
            reset_waiting=True,
            reset_vad=True,
        )
        frame.complete_audio = None
        self._log("[全双工] 🧹 已同步清空 VAD buffer，防止重复处理")

    async def _send_interrupt_confirmation(self, *, label: str) -> None:
        response = "好的，请说。"
        self._log("[全双工打断] 📢 检测到打断请求，发送确认回复")
        await self.connection.send_json(
            {"type": "ai_response", "text": response}
        )
        await self.history_store.append("assistant", response)
        self.session.chat_history.append(
            {"role": "assistant", "content": response}
        )
        try:
            result = await self._stream_tts_audio(
                response,
                content_text=response,
                emotion="neutral",
                label=label,
                event_type="tts_chunk",
                allow_interrupt=False,
                include_dtype=True,
                start_payload={"type": "tts_start", "text": response},
            )
            duration = (
                result["samples"] / 24000.0
                if result["samples"] > 0
                else 1.5
            )
            if result.get("started"):
                end_payload = {"type": "tts_end", "duration": duration}
                end_payload.update(result.get("output_identity") or {})
                if result.get("interrupted"):
                    end_payload["reason"] = (
                        result.get("interruption_reason") or "cancelled"
                    )
                await self.connection.send_json(end_payload)
        except Exception as exc:
            self._log(
                f"[全双工打断] ⚠️ 发送确认语音失败: {exc}"
            )

    @classmethod
    def _is_interrupt_request(cls, text: str) -> bool:
        return any(phrase in text for phrase in cls._INTERRUPT_PHRASES)

    async def _handle_complete_audio(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        if await self._complete_waiting_utterance(frame):
            return
        if await self._handle_processing_revision(frame):
            return
        if await self._capture_answer_completion(frame):
            return
        if await self._handle_ai_speaking_utterance(frame):
            return

        await self._send_vad_end(frame)
        self._reset_interrupt_capture(reset_waiting=True)
        kwargs = {"source": "语音输入"}
        if frame.realtime_turn is not None:
            kwargs["extra_meta"] = {"realtime_turn": frame.realtime_turn}
        await self._submit_speech(frame.complete_audio, **kwargs)

    async def _complete_waiting_utterance(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        runtime = self.session.runtime
        if not runtime.waiting_for_complete:
            return False
        self._log("[VAD] ✅ 等待补全时检测到完整语音，用户说完了")
        if not runtime.interrupt_audio_buffer:
            self._log("[VAD] 🎯 无累积音频，直接处理VAD检测到的完整语音")
            runtime.waiting_for_complete = False
            return False

        combined_audio = np.concatenate(
            runtime.interrupt_audio_buffer + [frame.complete_audio]
        )
        self._log(
            "[VAD] 🎯 合并音频（累积 + VAD），"
            f"总时长: {len(combined_audio) / 16000:.2f}s"
        )
        cached_text, _ = self._reuse_interrupt_judgement(combined_audio)
        text, intent = await self._judge_waiting_completion(
            combined_audio,
            cached_text=cached_text,
        )
        if text:
            self._remember_interrupt_judgement(
                combined_audio,
                text,
                intent,
            )
        if not text or len(text.strip()) < 2:
            self._reset_interrupt_capture(reset_waiting=True)
            self._log("[VAD] ⚠️ 合并音频快速识别为空或过短，放弃处理")
            await self._notify_voice_input_feedback(
                "speech_too_short",
                "刚才这句太短或不够清晰，系统没有处理。您可以再完整说一遍。",
                status_text="语音太短或不清晰，请再说一遍",
                duration_s=len(combined_audio) / 16000.0,
                source="interrupt_complete",
            )
            return True
        if intent != "complete":
            self._reset_interrupt_capture(reset_waiting=True)
            self._log(
                f"[VAD] ❌ 合并音频判定为{intent}，放弃处理: "
                f"text_chars={len(text)}"
            )
            return True

        self._reset_interrupt_capture(reset_waiting=True)
        await self._send_vad_end(frame)
        await self._submit_speech(
            combined_audio,
            source="全双工合并语音",
        )
        return True

    async def _handle_processing_revision(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        processing = self.session.processing
        runtime = self.session.runtime
        if (
            not processing.is_active
            or not processing.revision_enabled
            or self._is_ai_speaking()
        ):
            return False

        text, intent = self._reuse_interrupt_judgement(
            frame.complete_audio
        )
        if intent is None:
            text = await self._quick_asr(frame.complete_audio)
            if text and len(text.strip()) >= 2:
                intent = await self._judge_interrupt_intent(text)
                self._remember_interrupt_judgement(
                    frame.complete_audio,
                    text,
                    intent,
                )

        if not text or len(text.strip()) < 2:
            self._log(
                "[全双工] ⚠️ 处理阶段语音过短或识别为空，忽略这次覆盖"
            )
            self._reset_interrupt_capture(reset_waiting=True)
            await self._notify_voice_input_feedback(
                "speech_too_short",
                "刚才这句太短或不够清晰，系统没有切换到新版回答。您可以再完整说一遍。",
                status_text="语音太短或不清晰，请再说一遍",
                duration_s=len(frame.complete_audio) / 16000.0,
                source="processing_complete",
            )
            return True
        if intent == "backchannel":
            self._log("[全双工] ⏭️ 处理阶段识别到应答词，保持当前处理")
            self._reset_interrupt_capture(reset_waiting=True)
            return True
        if intent != "complete":
            await self._wait_for_processing_revision(
                frame.complete_audio,
                text,
            )
            return True

        await self._replace_active_processing(frame)
        return True

    async def _wait_for_processing_revision(
        self,
        audio,
        text: str,
    ) -> None:
        runtime = self.session.runtime
        self._log(
            f"[全双工] ⏳ 处理阶段识别到未说完的补充："
            f"text_chars={len(text)}"
        )
        await self._interrupt_active_processing(
            "处理阶段检测到用户继续补充（未说完）"
        )
        base_audio = self._get_active_processing_audio()
        runtime.interrupt_audio_buffer = []
        if base_audio is not None and len(base_audio) > 0:
            runtime.interrupt_audio_buffer.append(base_audio)
        runtime.interrupt_audio_buffer.append(
            np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        )
        runtime.interrupt_speech_run = 0
        runtime.last_speech_time = time.time()
        runtime.waiting_for_complete = True

    async def _replace_active_processing(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        processing = self.session.processing
        self._log(
            "[全双工] 🎯 处理阶段捕获到新的完整语音，"
            "废弃当前处理并改用最新输入"
        )
        base_audio = self._get_active_processing_audio()
        revised_audio = np.asarray(
            frame.complete_audio,
            dtype=np.float32,
        ).reshape(-1).copy()
        source = "全双工处理阶段语音"
        if base_audio is not None and len(base_audio) > 0:
            revised_audio = np.concatenate(
                [base_audio, revised_audio]
            ).astype(np.float32, copy=False)
            source = f"{processing.active_source or '处理中语音'}+补充"
            self._log(
                "[全双工] 🔗 将处理中的原回答与补充语音合并后重算，"
                f"总时长: {len(revised_audio) / 16000:.2f}s"
            )
        await self._interrupt_active_processing(
            "处理阶段检测到新的完整语音"
        )
        await self._send_vad_end(frame)
        self._reset_interrupt_capture(reset_waiting=True)
        await self._submit_speech(revised_audio, source=source)

    async def _capture_answer_completion(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        if (
            not self.answer_completion.enabled
            or self._is_ai_speaking()
        ):
            return False
        duration = self.answer_completion.append_segment(
            frame.complete_audio
        )
        self._reset_interrupt_capture()
        await self._notify_answer_completion(
            "captured",
            "已捕获一段回答，系统会继续聆听，并在合适时自动观察是否提交；您也可以点击🧠手动判定。",
            segments=len(self.answer_completion.segments),
            duration_s=duration,
        )
        await self._arm_answer_completion_window()
        return True

    async def _handle_ai_speaking_utterance(
        self,
        frame: LiveAudioFrameContext,
    ) -> bool:
        if not self._is_ai_speaking():
            return False
        duration = len(frame.complete_audio) / 16000
        if duration < self.config.interrupt_min_complete_audio_s:
            self._log(
                "[VAD] ⏭️ AI说话期间检测到极短语音 "
                f"({duration:.1f}s < "
                f"{self.config.interrupt_min_complete_audio_s:.2f}s)，"
                "跳过（避免误打断）"
            )
            await self._notify_voice_input_feedback(
                "speech_too_short",
                "刚才这句太短，没有触发打断。您可以等提示结束后再完整说一遍。",
                status_text="打断语音太短，请再完整说一遍",
                duration_s=duration,
                source="interrupt_complete",
            )
            return True
        self._log(
            "[VAD] ️️ 检测到完整语音"
            f"（AI正在说话，{duration:.1f}s），用户主动打断或回答"
        )
        if not await self._verify_complete_speaker(frame.complete_audio):
            return True

        text = await self._quick_asr(frame.complete_audio)
        if not text or len(text.strip()) < 2:
            self._reset_interrupt_capture()
            self._log(
                "[VAD] ⚠️ AI说话期间完整语音快速识别为空或过短，"
                "继续播放"
            )
            await self._notify_voice_input_feedback(
                "speech_too_short",
                "刚才这句太短或不够清晰，没有触发打断。您可以再完整说一遍。",
                status_text="打断语音太短，请再完整说一遍",
                duration_s=duration,
                source="interrupt_complete",
            )
            return True

        intent = await self._judge_interrupt_intent(text)
        self._remember_interrupt_judgement(
            frame.complete_audio,
            text,
            intent,
        )
        if intent == "backchannel":
            self._reset_interrupt_capture()
            self._log(
                f"[VAD] ❌ AI说话期间识别为应答词，不打断: "
                f"text_chars={len(text)}"
            )
            return True
        if intent == "incomplete":
            await self._pause_for_incomplete_utterance(
                frame.complete_audio,
                text,
            )
            return True

        processing = self.session.processing
        if processing.is_active and processing.revision_enabled:
            await self._interrupt_active_processing(
                "AI回复前检测到用户新的完整表达"
            )
        else:
            await self._stop_playback()
        self._log("[VAD] ✅ AI已停止，准备处理用户输入")
        return False

    async def _verify_complete_speaker(self, audio) -> bool:
        if not self.speaker.enabled:
            return True
        is_target, similarity = await self.speaker.verify(
            audio,
            16000,
            source="interrupt_complete",
        )
        if is_target:
            return True
        if np.isfinite(similarity):
            self._log(
                "[VAD] ⏭️ 非目标说话人 "
                f"(相似度: {similarity:.3f})，跳过处理"
            )
        else:
            self._log("[VAD] ⛔ 声纹验证未通过或不可用，跳过处理")
        return False

    async def _pause_for_incomplete_utterance(
        self,
        audio,
        text: str,
    ) -> None:
        runtime = self.session.runtime
        processing = self.session.processing
        if processing.is_active and processing.revision_enabled:
            await self._interrupt_active_processing(
                "AI回复前检测到用户未说完的新表达"
            )
        else:
            await self._stop_playback()
        runtime.waiting_for_complete = True
        runtime.interrupt_audio_buffer = [audio]
        runtime.interrupt_speech_run = 0
        runtime.last_speech_time = time.time()
        self._log(
            "[VAD] ⏳ AI说话期间识别到未说完语句，"
            f"已停止播放并等待补全: text_chars={len(text)}"
        )

    async def _send_vad_end(
        self,
        frame: LiveAudioFrameContext,
    ) -> None:
        if frame.vad_end_sent:
            return
        self._log("[VAD] 🎯 发送 vad_end 到前端")
        await self.connection.send_json({"type": "vad_end"})
        frame.vad_end_sent = True
