from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from ..services import TTSPrewarmHandle
from src.tools.emotion import classify_multimodal_with_metadata
from .agent_output_presenter import AgentOutputPresenter
from .speech_turn_config import SpeechTurnConfig
from .speech_turn_context import SpeechTurnContext
from ..safety import FinalRiskGate
from ..modes import is_cognitive_screening, mode_for_session
from ..turn_insight import send_turn_insight


_DEFAULT_TTS_START = object()


class SpeechTurnProcessor:
    """Process one completed user utterance from ASR through Agent and TTS."""

    def __init__(
        self,
        connection,
        *,
        session,
        speaker,
        audio_store,
        history_store,
        presenter: AgentOutputPresenter,
        stream_tts_audio: Callable[..., Any],
        notify_voice_input_feedback: Callable[..., Any],
        normalize_profile: Callable[[dict | None], dict],
        parse_sensevoice_result: Callable[[Any], dict],
        audio_signal_stats: Callable[[np.ndarray, int], dict],
        audio_is_effectively_silent: Callable[[dict], bool],
        asr_model,
        tts,
        clean_for_tts: Callable[[str], str],
        config: SpeechTurnConfig,
        logger=print,
        capture_memory_turn: Callable[..., Any] | None = None,
        get_turn_memory: Callable[..., str] | None = None,
        get_cross_session_memory: Callable[..., str] | None = None,
        update_memory_emotion: Callable[..., Any] | None = None,
        update_memory_status: Callable[..., Any] | None = None,
        record_safety_event: Callable[..., Any] | None = None,
    ) -> None:
        self.connection = connection
        self.session = session
        self.speaker = speaker
        self.audio_store = audio_store
        self.history_store = history_store
        self.presenter = presenter
        self._stream_tts_audio = stream_tts_audio
        self._notify_voice_input_feedback = notify_voice_input_feedback
        self._normalize_profile = normalize_profile
        self._parse_sensevoice_result = parse_sensevoice_result
        self._audio_signal_stats = audio_signal_stats
        self._audio_is_effectively_silent = audio_is_effectively_silent
        self.asr_model = asr_model
        self.tts = tts
        self._clean_for_tts = clean_for_tts
        self.config = config
        self._log = logger
        self._capture_memory_turn = capture_memory_turn
        self._get_turn_memory = get_turn_memory
        self._get_cross_session_memory = get_cross_session_memory
        self._update_memory_emotion = update_memory_emotion
        self._update_memory_status = update_memory_status
        self._record_safety_event = record_safety_event

    def is_superseded(self, generation_token: int | None) -> bool:
        runtime = self.session.runtime
        processing = self.session.processing
        if generation_token is None:
            return runtime.stop_generate
        return (
            runtime.stop_generate
            or generation_token != processing.generation
        )

    def _agent_process_turn_compat(self, **kwargs):
        try:
            return self.session.agent.process_turn(**kwargs)
        except TypeError as exc:
            if "should_abort" not in str(exc):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("should_abort", None)
            self._log(
                "[Agent] ⚠️ 当前 Agent 不支持 should_abort，"
                "已降级为兼容调用"
            )
            return self.session.agent.process_turn(**fallback_kwargs)

    async def process(
        self,
        audio_data,
        process_generation_token: int | None = None,
        turn_id: str = "",
        extra_meta: dict | None = None,
    ) -> None:
        context = SpeechTurnContext(
            audio_data=audio_data,
            generation_token=process_generation_token,
            turn_id=turn_id,
            extra_meta=extra_meta,
            tts_prewarm=TTSPrewarmHandle.start(
                self.tts,
                enabled=self.config.use_ark_tts,
            ),
        )
        try:
            if not await self._verify_speaker(context):
                return

            recognition = await self._recognize(context)
            if not await self._apply_recognition(context, recognition):
                return
            if await self._skip_suspected_echo(context):
                return

            context.text, _ = self._merge_interrupted_text(context.text)
            context.user_history_entry["content"] = context.text

            context.risk_decision = FinalRiskGate.evaluate(
                context.text,
                source=context.asr_source
                or str((context.extra_meta or {}).get("asr_source") or "final"),
            )
            if not await self._accept_and_gate_turn(context):
                return

            if not await self._publish_recognition(context):
                return
            if await self._defer_for_pending_vision(context):
                return

            try:
                completed = await self._run_agent_turn(context)
            except Exception as exc:
                await self._send_processing_failure(context, exc)
            else:
                if not completed:
                    return

            await self._send_score_update()
            self._log("[FLOW] 🏁 处理完成\n")
        except Exception as exc:
            self._log(
                f"[错误] {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )

    async def _accept_and_gate_turn(self, context: SpeechTurnContext) -> bool:
        if not await self._accept_user_turn(context):
            return False
        await self._record_final_risk(context)
        self._mark_memory_status(context, turn_state="SAFETY_GATED")
        if context.risk_decision and context.risk_decision.level == "medium":
            self.session.runtime.high_risk_detected = False
        context.emotion_metadata = {
            "source": "text_asr",
            "analysis_status": "provisional",
            "audio_model_used": False,
            "dominant": context.dominant_emotion or context.emotion,
            "inference_ms": 0.0,
        }
        await self._send_turn_insight(context, "provisional")
        asyncio.create_task(self._enrich_emotion(context))
        return True

    async def _verify_speaker(self, context: SpeechTurnContext) -> bool:
        self.speaker.log_diagnostic()
        if not self.speaker.enabled:
            await self.speaker.warn_if_disabled()
            return True

        is_target, similarity = await self.speaker.verify(
            context.audio_data,
            16000,
            source="main",
            turn_id=context.turn_id,
        )
        if not is_target:
            if np.isfinite(similarity):
                self._log(
                    "[声纹] 🎯 验证结果: "
                    f"is_target={is_target}, similarity={similarity:.2f}"
                )
                self._log(
                    "[FLOW] ⏭️ 非目标说话人 "
                    f"(相似度: {similarity:.2f})，跳过处理"
                )
            else:
                self._log(
                    "[FLOW] ⛔ 声纹验证未通过或不可用，拒绝处理当前语音"
                )
            return False

        self._log(
            "[声纹] 🎯 验证结果: "
            f"is_target={is_target}, similarity={similarity:.2f}"
        )
        self._log(
            f"[声纹] ✅ 目标说话人确认 (相似度: {similarity:.2f})"
        )
        return True

    async def _recognize(self, context: SpeechTurnContext) -> dict | None:
        stats = self._audio_signal_stats(context.audio_data, 16000)
        self._log(
            f"[ASR][Audio] turn={context.turn_id or '-'} "
            f"source={self.session.processing.active_source or '-'} "
            f"duration={stats['duration_s']:.2f}s "
            f"rms={stats['rms']:.5f} peak={stats['peak']:.5f} "
            f"dbfs={stats['dbfs']:.1f}"
        )
        if self._audio_is_effectively_silent(stats):
            self._log("[ASR] ⚠️ 输入音频接近静音，跳过识别")
            await self._notify_voice_input_feedback(
                "audio_too_quiet",
                "这段录音音量太低，系统几乎没有收到声音。请确认麦克风输入后再说一遍。",
                status_text="录音音量太低，请再说一遍",
                source="main",
                turn_id=context.turn_id,
                duration_s=stats["duration_s"],
            )
            return None

        extra_meta = (
            context.extra_meta
            if isinstance(context.extra_meta, dict)
            else {}
        )
        if extra_meta.get("turn_taking") == "soulx":
            text = str(
                extra_meta.get("soulx_text")
                or extra_meta.get("soulx_asr_buffer")
                or ""
            ).strip()
            self._log(
                "\n[FLOW] 1️⃣ 直接使用 SoulX Paraformer 结果，"
                "跳过二次 ASR"
            )
            return {
                "text": text,
                "emotion": "neutral",
                "language": "zh",
                "event": "speech",
            }

        if self.config.use_ark_asr:
            return await self._recognize_with_ark(context)

        self._log(
            "\n[FLOW] 1️⃣ 语音转文字+情绪识别 "
            "(SenseVoice, 无文件I/O)..."
        )
        result = await asyncio.to_thread(
            self.asr_model.generate,
            input=context.audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        return self._parse_sensevoice_result(result)

    async def _apply_recognition(
        self,
        context: SpeechTurnContext,
        recognition: dict | None,
    ) -> bool:
        if recognition is None:
            if self._asr_failure(context):
                await self._handle_asr_failure(context)
            return False
        context.apply_recognition(recognition)
        if not context.text:
            await self._handle_asr_failure(context)
            return False
        return True

    async def _enrich_emotion(self, context: SpeechTurnContext) -> None:
        """Persist one audio file and run optional multimodal inference off-loop."""
        try:
            context.audio_meta = await self._persist_user_audio(context)
        except Exception as exc:
            self._log(f"[Emotion] ⚠️ 音频落盘失败，继续文本识别: {type(exc).__name__}")

        audio_path = None
        if isinstance(context.audio_meta, dict):
            audio_path = context.audio_meta.get("file_path")
        try:
            scores, metadata = await asyncio.to_thread(
                classify_multimodal_with_metadata,
                text=context.text,
                audio_path=audio_path,
            )
            context.apply_emotion_scores(scores)
            emotion_source = str(metadata.get("source") or "unavailable")
            if emotion_source == "text":
                emotion_source = "text_fallback_audio_missing"
            context.emotion_metadata = {
                "source": emotion_source,
                "analysis_status": "final",
                "audio_model_used": bool(metadata.get("audio_model_used")),
                "dominant": context.dominant_emotion or context.emotion,
                "inference_ms": metadata.get("inference_ms") or 0.0,
            }
            dominant = context.dominant_emotion or context.emotion or "calm"
            self._log(
                f"[Emotion] {dominant}="
                f"{context.emotion_scores.get(dominant, 0.0):.2f}"
                f" source={metadata['source']}"
                f" inference_ms={metadata['inference_ms']:.1f}"
            )
            if self._update_memory_emotion is not None:
                try:
                    self._update_memory_emotion(
                        self.session,
                        context.turn_id,
                        context.emotion_scores,
                        audio_path=audio_path,
                        emotion_source=emotion_source,
                        analysis_status="final",
                    )
                except TypeError:
                    self._update_memory_emotion(
                        self.session,
                        context.turn_id,
                        context.emotion_scores,
                        audio_path=audio_path,
                    )
                except Exception as exc:
                    self._log(f"[EmotionMemory] ⚠️ 最终情绪异步回写失败: {type(exc).__name__}")
        except Exception as exc:
            source = (
                "text_fallback_audio_missing"
                if not audio_path
                else "text_fallback_audio_error"
            )
            context.emotion_metadata = {
                "source": source,
                "analysis_status": "unavailable",
                "audio_model_used": False,
                "dominant": context.dominant_emotion or context.emotion,
                "inference_ms": 0.0,
            }
            self._log(f"[Emotion] ⚠️ 识别失败，保留 ASR 情绪: {type(exc).__name__}")
        finally:
            await self._send_turn_insight(context, "final")

    async def _send_turn_insight(self, context: SpeechTurnContext, state: str) -> None:
        if state in context.insight_sent:
            return
        try:
            sent = await send_turn_insight(
                self.connection,
                session=self.session,
                turn_id=context.turn_id,
                state=state,
                emotion=context.emotion,
                scores=context.emotion_scores,
                emotion_metadata=context.emotion_metadata,
                used_item_ids=context.memory_insight.get("used_item_ids", []),
                written_item_ids=context.memory_insight.get("written_item_ids", []),
                summary_changed=context.memory_insight.get("summary_changed", False),
                risk_decision=context.risk_decision,
                risk_handled=bool(
                    context.risk_decision
                    and getattr(context.risk_decision, "level", "") != "low"
                ),
            )
            if sent:
                context.insight_sent.add(state)
        except Exception as exc:
            self._log(f"[Insight] ⚠️ turn_insight发送失败: {type(exc).__name__}")

    async def _recognize_with_ark(
        self,
        context: SpeechTurnContext,
    ) -> dict | None:
        from src.tools.voice.ark_asr import (
            ArkASRError,
            ArkASRTemporaryError,
            ark_asr_recognize,
        )

        realtime_turn = (
            context.extra_meta.get("realtime_turn")
            if isinstance(context.extra_meta, dict)
            else None
        )
        if realtime_turn is not None:
            text = await realtime_turn.final_text()
            stream_error = getattr(realtime_turn, "stream_error", None)
            has_final = getattr(realtime_turn, "has_final", bool(text))
            if text and stream_error is None and has_final:
                self._log("\n[FLOW] 1️⃣ 使用 BigASR 流式最终文字")
                return {
                    "text": text,
                    "emotion": "neutral",
                    "language": "zh",
                    "event": "speech",
                }
            self._log(
                "[ASR] ⚠️ 流式结果未通过最终性校验，"
                "回退整段 BigASR 识别"
            )
            
        self._log("\n[FLOW] 1️⃣ 语音转文字 (火山引擎 BigASR)...")
        try:
            result = await ark_asr_recognize(context.audio_data)
            if result.get("text"):
                result["source"] = "fallback_final"
            return result
        except ArkASRTemporaryError as exc:
            self._log(f"[ASR] ⚠️ BigASR 临时失败: {type(exc).__name__}")
            self._set_asr_failure(context)
            await self.connection.send_json(
                {
                    "type": "asr_error",
                    "turn_id": context.turn_id,
                    "text": "刚刚网络有点慢，我这句没听清，您再说一遍好吗？",
                    "status_text": "语音识别超时，请再说一遍",
                    "detail": str(exc),
                }
            )
        except ArkASRError as exc:
            self._log(f"[ASR] ❌ BigASR 失败: {type(exc).__name__}")
            self._set_asr_failure(context)
            await self.connection.send_json(
                {
                    "type": "asr_error",
                    "turn_id": context.turn_id,
                    "text": "语音识别暂时出了点问题，您稍等一下再说一遍。",
                    "status_text": "语音识别暂时不可用",
                    "detail": str(exc),
                }
            )
        except Exception as exc:
            self._log(f"[ASR] ❌ BigASR 未知失败: {type(exc).__name__}")
            self._set_asr_failure(context)
            await self.connection.send_json(
                {
                    "type": "asr_error",
                    "turn_id": context.turn_id,
                    "text": "语音识别暂时不可用，请稍后再说一遍。",
                    "status_text": "语音识别暂时不可用",
                }
            )
        return None

    @staticmethod
    def _set_asr_failure(context: SpeechTurnContext) -> None:
        metadata = dict(context.extra_meta or {})
        metadata["asr_failure"] = True
        context.extra_meta = metadata

    @staticmethod
    def _asr_failure(context: SpeechTurnContext) -> bool:
        return bool(
            isinstance(context.extra_meta, dict)
            and context.extra_meta.get("asr_failure")
        )

    async def _handle_asr_failure(self, context: SpeechTurnContext) -> None:
        """Keep audio evidence and fail closed when no final text exists."""
        context.risk_decision = FinalRiskGate.evaluate(
            "", source="fallback_final"
        )
        await self._record_final_risk(context)
        try:
            context.audio_meta = await self._persist_user_audio(context)
        except Exception as exc:
            self._log(f"[ASR] ⚠️ 识别失败音频落盘失败: {type(exc).__name__}")
        self._log(
            f"[ASR] ⚠️ turn={context.turn_id} 无最终文字，已拒绝生成并保留音频"
        )

    async def _accept_user_turn(self, context: SpeechTurnContext) -> bool:
        """Persist accepted patient evidence before any cancellable output."""
        if not context.text:
            return False
        if context.turn_id in self.session.accepted_turn_ids:
            context.accepted = True
            return True
        if context.risk_decision and context.risk_decision.restricted:
            self.session.runtime.high_risk_detected = True
        try:
            context.audio_meta = await self._persist_user_audio(context)
        except Exception as exc:
            self._log(f"[TURN] ⚠️ 用户音频落盘失败: {type(exc).__name__}")
        history_meta = dict(context.audio_meta or {})
        try:
            await self.history_store.append(
                "user",
                context.text,
                context.emotion,
                context.language,
                history_meta,
                context.turn_id,
            )
            entry = dict(context.user_history_entry)
            entry["turn_id"] = context.turn_id
            context.user_history_entry["turn_id"] = context.turn_id
            if not any(
                item.get("turn_id") == context.turn_id
                for item in self.session.chat_history
                if isinstance(item, dict)
            ):
                self.session.chat_history.append(entry)
        except Exception as exc:
            self._log(f"[TURN] ⚠️ 用户历史落库失败: {type(exc).__name__}")
            return False
        if self._capture_memory_turn is not None:
            capture_started_at = time.perf_counter()
            try:
                self._capture_memory_turn(
                    self.session,
                    context.text,
                    "",
                    context.emotion_scores or context.emotion,
                    audio_path=(context.audio_meta or {}).get("file_path"),
                    turn_id=context.turn_id,
                    emotion_source="text_asr",
                    analysis_status="provisional",
                )
            except TypeError:
                self._capture_memory_turn(
                    self.session,
                    context.text,
                    "",
                    context.emotion_scores or context.emotion,
                    audio_path=(context.audio_meta or {}).get("file_path"),
                )
            except Exception as exc:
                self._log(f"[EmotionMemory] ⚠️ 用户证据捕获失败: {type(exc).__name__}")
            finally:
                context.memory_capture_elapsed_ms = (
                    time.perf_counter() - capture_started_at
                ) * 1000
                self._log(
                    f"[Latency] turn={context.turn_id or '-'} "
                    f"memory_capture_ms={context.memory_capture_elapsed_ms:.1f}"
                )
        self.session.accepted_turn_ids.add(context.turn_id)
        context.accepted = True
        return True

    async def _record_final_risk(self, context: SpeechTurnContext) -> None:
        decision = context.risk_decision
        if (
            decision is None
            or decision.level == "low"
            or context.risk_event_recorded
            or self._record_safety_event is None
        ):
            return
        status = (
            "risk_scan_failed"
            if decision.level == "unknown"
            else "review_required"
            if decision.level == "medium"
            else "handled"
        )
        try:
            await asyncio.to_thread(
                self._record_safety_event,
                patient_id=self.session.lifecycle.current_patient_id,
                session_id=self.session.session_id,
                turn_id=context.turn_id,
                source=decision.source,
                level=decision.level,
                rule_version=decision.rule_version,
                text_hmac=decision.text_hmac,
                status=status,
            )
            context.risk_event_recorded = True
        except Exception as exc:
            self._log(f"[Safety] ⚠️ 风险事件审计失败: {type(exc).__name__}")

    def _mark_memory_status(self, context: SpeechTurnContext, **kwargs) -> None:
        if self._update_memory_status is None:
            return
        try:
            self._update_memory_status(self.session, context.turn_id, **kwargs)
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 轮次状态更新失败: {type(exc).__name__}")

    async def _send_final_safety_response(self, context: SpeechTurnContext) -> bool:
        decision = context.risk_decision
        if decision is None or not decision.restricted:
            return False
        unknown = decision.level == "unknown"
        text = (
            FinalRiskGate.unknown_safety_text()
            if unknown
            else FinalRiskGate.safety_text()
        )
        await self.connection.send_json({
            "type": "safety_alert",
            "turn_id": context.turn_id,
            "severity": "unknown" if unknown else "high",
            "rule_version": decision.rule_version,
            "source": decision.source,
            "matched_rule_ids": list(decision.matched_rule_ids),
        })
        await self.connection.send_json({
            "type": "ai_response",
            "turn_id": context.turn_id,
            "text": text,
            "safety": True,
            "finalize_chunks": True,
        })
        tts_result = await self._stream_tts_audio(
            text,
            content_text=text,
            emotion="gentle",
            label="-safety",
            event_type="tts_chunk",
            allow_interrupt=True,
            include_dtype=True,
            persist=False,
            session_id=self.session.session_id,
            turn_id=context.turn_id,
            generation=context.generation_token,
            playback_id=self.session.processing.active_playback_id,
            start_payload={
                "type": "tts_start",
                "turn_id": context.turn_id,
                "text": self._clean_for_tts(text),
                "safety": True,
            },
        )
        if tts_result.get("started"):
            end_payload = {
                "type": "tts_end",
                "duration": (tts_result.get("samples") or 0) / 24000.0,
                "safety": True,
            }
            end_payload.update(tts_result.get("output_identity") or {})
            if tts_result.get("interrupted"):
                end_payload["reason"] = tts_result.get("interruption_reason") or "cancelled"
            await self.connection.send_json(end_payload)
        await self._persist_completed_turn(context, {}, text)
        return True

    async def _skip_suspected_echo(
        self,
        context: SpeechTurnContext,
    ) -> bool:
        realtime_turn = (
            context.extra_meta.get("realtime_turn")
            if isinstance(context.extra_meta, dict)
            else None
        )
        if realtime_turn is None or not realtime_turn.echo_suspected:
            return False
        self._log("[ASR] ⚠️ 可疑系统回声已存档，不进入正式回复")
        await self.connection.send_json(
            {
                "type": "voice_input_feedback",
                "reason": "suspected_tts_echo",
                "message": "已忽略可能由系统播报产生的回声。",
                "turn_id": context.turn_id,
            }
        )
        return True

    async def _publish_recognition(
        self,
        context: SpeechTurnContext,
    ) -> bool:
        self._log(
            f"[ASR] 语言: {context.language}, 情绪: {context.emotion}, "
            f"主导情绪: {context.dominant_emotion or '-'}, "
            f"事件: {context.event}"
        )
        if not context.text:
            self._log("[ASR] ⚠️ 识别结果为空，跳过处理")
            await self._notify_voice_input_feedback(
                "asr_empty",
                "刚才这句我没听清，您可以再说一遍。",
                status_text="没听清，请再说一遍",
                source="main",
                turn_id=context.turn_id,
            )
            return False

        turn_taking = self.session.turn_taking
        if self.is_superseded(context.generation_token):
            turn_taking.remember_interrupted_asr(
                context.text,
                current_time=time.time(),
                append=True,
            )
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            self._log(
                "[FLOW] ⚠️ ASR后检测到打断，暂存已识别文本待合并: "
                f"text_chars={len(context.text)}"
            )
            return False

        self._log(
            f"[ASR] ✅ 识别完成: text_chars={len(context.text)} "
            f"(情绪: {context.emotion})"
        )
        sent = await self.connection.send_json(
            {
                "type": "asr_result",
                "turn_id": context.turn_id,
                "text": context.text,
                "emotion": context.emotion,
                "emotion_label": context.dominant_emotion or context.emotion,
                "emotions": context.emotion_scores,
                "language": context.language,
                "high_risk": self.session.runtime.high_risk_detected,
            }
        )
        if not sent:
            return False
        context.asr_result_at = time.perf_counter()

        if self.is_superseded(context.generation_token):
            turn_taking.remember_interrupted_asr(
                context.text,
                current_time=time.time(),
            )
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            self._log(
                "[FLOW] ⚠️ ASR结果返回后检测到新的用户补充，"
                f"暂存文本待合并: text_chars={len(context.text)}"
            )
            return False
        return True

    def _merge_interrupted_text(self, text: str) -> tuple[str, str]:
        merged, prefix = self.session.turn_taking.merge_interrupted_asr(
            text,
            current_time=time.time(),
            ttl_seconds=60.0,
        )
        if prefix:
            self._log(
                "[FLOW] 🔗 合并被打断轮次的识别文本: "
                f"prefix_chars={len(prefix)} incoming_chars={len(text)}"
            )
        return merged, prefix

    async def _defer_for_pending_vision(
        self,
        context: SpeechTurnContext,
    ) -> bool:
        runtime = self.session.runtime
        if not runtime.pending_vision_task:
            return False

        elapsed = time.time() - runtime.vision_lock_time
        if elapsed > self.config.vision_lock_timeout:
            self._log(
                "[Vision] ⚠️ 视觉任务锁超时"
                f"({elapsed:.1f}s > {self.config.vision_lock_timeout}s)，"
                "自动释放"
            )
            runtime.pending_vision_task = None
            runtime.queued_user_text = None
            return False

        runtime.queued_user_text = context.text
        self._log(
            f"[Vision] 🔒 视觉任务 [{runtime.pending_vision_task}] "
            f"进行中({elapsed:.1f}s)，text_chars={len(context.text)}"
        )
        await self._send_vision_wait_ack(context)
        return True

    async def _send_vision_wait_ack(
        self,
        context: SpeechTurnContext,
    ) -> None:
        ack = "好的，我看到了，请稍等一下。"
        await self.connection.send_json(
            {
                "type": "ai_response",
                "turn_id": context.turn_id,
                "text": ack,
            }
        )
        await self.history_store.append(
            "assistant", ack, None, None, None, context.turn_id
        )
        self.session.chat_history.append(
            {"role": "assistant", "content": ack, "turn_id": context.turn_id}
        )
        tts_result = await self._stream_tts_audio(
            ack,
            content_text=ack,
            emotion="gentle",
            label="-vision-wait",
            event_type="tts_audio",
            allow_interrupt=False,
            include_dtype=False,
            session_id=self.session.session_id,
            turn_id=context.turn_id,
            generation=context.generation_token,
            playback_id=self.session.processing.active_playback_id,
            start_payload={
                "type": "tts_start",
                "turn_id": context.turn_id,
                "text": self._clean_for_tts(ack),
            },
        )
        total_samples = tts_result["samples"]
        duration_ms = (
            int(total_samples / 24000 * 1000)
            if total_samples > 0
            else 0
        )
        if tts_result.get("started"):
            end_payload = {
                "type": "tts_end",
                "duration_ms": duration_ms,
            }
            end_payload.update(tts_result.get("output_identity") or {})
            if tts_result.get("interrupted"):
                end_payload["reason"] = tts_result.get("interruption_reason") or "cancelled"
            await self.connection.send_json(end_payload)
        self._mark_memory_status(
            context,
            assistant_message=ack,
            response_status="responded",
            turn_state="RESPONDED",
        )
        self._log(
            "[Vision] 📢 已发送等待确认 "
            f"({tts_result['chunks']} chunks, {duration_ms}ms)"
        )

    async def _run_agent_turn(self, context: SpeechTurnContext) -> bool:
        if await self._send_final_safety_response(context):
            return True
        self._mark_memory_status(context, turn_state="GENERATING")
        await self._load_turn_memory(context)
        self._log(
            "\n[FLOW] 2️⃣ 🚀 AI流式生成 + 实时TTS - "
            f"用户情绪: {context.emotion}..."
        )
        context.prepare_agent(
            self._normalize_profile(self.session.patient_profile),
            self.session.chat_history,
            time.perf_counter(),
        )
        if self.config.use_llm_streaming:
            return await self._run_streaming_agent_turn(context)
        return await self._run_synchronous_agent_turn(context)

    async def _load_turn_memory(self, context: SpeechTurnContext) -> None:
        started_at = time.perf_counter()
        background = ""
        source = "none"
        realtime_turn = (
            context.extra_meta.get("realtime_turn")
            if isinstance(context.extra_meta, dict)
            else None
        )
        if realtime_turn is not None:
            if self.is_superseded(context.generation_token):
                cancel_prefetch = getattr(realtime_turn, "cancel_memory_prefetch", None)
                if callable(cancel_prefetch):
                    cancel_prefetch()
                source = "superseded"
                background = ""
            else:
                try:
                    background = await realtime_turn.memory_for_final(context.text)
                    source = "realtime"
                except Exception as exc:
                    self._log(f"[Memobase] ⚠️ 最终记忆确认失败: {type(exc).__name__}")
                    source = "realtime_error"
        elif self._get_turn_memory is not None:
            try:
                memory_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._get_turn_memory,
                        self.session,
                        context.text,
                    )
                )
                memory_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                background = await asyncio.wait_for(
                    asyncio.shield(memory_task),
                    timeout=max(0.01, float(self.config.memory_timeout_s)),
                )
                source = "primary"
            except asyncio.TimeoutError:
                self._log("[Memobase] ⚠️ 最终记忆检索超时，跳过长期事件")
                source = "timeout"
                memory_task.cancel()
                await asyncio.gather(memory_task, return_exceptions=True)
            except Exception as exc:
                self._log(f"[Memobase] ⚠️ 本轮旧事检索失败: {type(exc).__name__}")
                source = "primary_error"
        if self.is_superseded(context.generation_token):
            if realtime_turn is not None:
                cancel_prefetch = getattr(realtime_turn, "cancel_memory_prefetch", None)
                if callable(cancel_prefetch):
                    cancel_prefetch()
            background = ""
            source = "superseded"
        if context.risk_decision and context.risk_decision.restricted:
            background = (
                f"{background}\n" if background else ""
            ) + "【安全风险】患者刚表达了自伤或自杀相关内容。先确认其身边是否有人，并建议立即联系可信任的人和当地紧急援助；不要声称其已经安全。"
        elif context.risk_decision and context.risk_decision.level == "medium":
            background = (
                f"{background}\n" if background else ""
            ) + "【安全关注】患者表达明显痛苦。回复应保持简短、支持性和非诊断性，优先询问当下支持资源。"
        memory_tool = getattr(
            getattr(self.session.agent, "tool_gateway", None),
            "memory_tool",
            None,
        )
        setter = getattr(memory_tool, "set_turn_background", None)
        if callable(setter):
            setter(str(background or ""))
        context.memory_elapsed_ms = (time.perf_counter() - started_at) * 1000
        context.memory_chars = len(str(background or ""))
        context.memory_source = source
        self._log(
            f"[Latency] turn={context.turn_id or '-'} "
            f"memory_ms={context.memory_elapsed_ms:.1f} "
            f"memory_chars={context.memory_chars} source={source}"
        )

    def _start_streaming_agent(self, context: SpeechTurnContext) -> dict:
        loop = asyncio.get_running_loop()
        sentence_queue = asyncio.Queue()
        result_box = [None]
        error_box = [None]
        agent = self.session.agent

        def on_sentence(text):
            loop.call_soon_threadsafe(sentence_queue.put_nowait, text)

        def run_agent():
            agent.state._stream_sentence_cb = on_sentence
            try:
                result_box[0] = self._agent_process_turn_compat(
                    user_input=context.text,
                    session_id=self.session.session_id,
                    patient_profile=context.agent_profile,
                    chat_history=context.working_chat_history,
                    current_emotion=context.emotion,
                    should_abort=lambda: self.is_superseded(
                        context.generation_token
                    ),
                )
            except Exception as exc:
                error_box[0] = exc
            finally:
                agent.state._stream_sentence_cb = None
                loop.call_soon_threadsafe(sentence_queue.put_nowait, None)

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        return {
            "queue": sentence_queue,
            "result_box": result_box,
            "error_box": error_box,
            "thread": thread,
            "stream": None,
        }

    async def _consume_streaming_sentences(
        self,
        context: SpeechTurnContext,
        task: dict,
    ) -> dict:
        state = {
            "sentence_count": 0,
            "total_chunks": 0,
            "total_samples": 0,
            "first_latency": 0.0,
            "interrupted": False,
            "interruption_reason": "",
            "started": False,
            "output_identity": {},
            "audio_parts": [],
            "text_parts": [],
        }
        task["stream"] = state
        queue = task["queue"]
        thread = task["thread"]

        while not self.is_superseded(context.generation_token):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.02)
            except asyncio.TimeoutError:
                if not thread.is_alive() and queue.empty():
                    break
                continue
            if item is None:
                break

            state["sentence_count"] += 1
            sentence_index = state["sentence_count"]
            state["text_parts"].append(item)
            if sentence_index == 1:
                context.first_ai_sentence_at = time.perf_counter()
                self._log(
                    "[Stream] 🔥 首句到达! "
                    f"延迟={time.perf_counter() - context.agent_started_at:.2f}s: "
                    f"text_chars={len(item)}"
                )
                asr_to_agent_ms = (
                    (context.agent_started_at - context.asr_result_at) * 1000
                    if context.asr_result_at else 0.0
                )
                self._log(
                    f"[Latency] turn={context.turn_id or '-'} "
                    f"input_to_asr_result_ms="
                    f"{(context.asr_result_at - context.processing_started_at) * 1000:.1f} "
                    f"asr_result_to_agent_ms={asr_to_agent_ms:.1f} "
                    f"memory_capture_ms={context.memory_capture_elapsed_ms:.1f} "
                    f"memory_ms={context.memory_elapsed_ms:.1f} "
                    f"memory_chars={context.memory_chars} "
                    f"agent_to_first_sentence_ms="
                    f"{(context.first_ai_sentence_at - context.agent_started_at) * 1000:.1f} "
                    f"input_to_first_sentence_ms="
                    f"{(context.first_ai_sentence_at - context.processing_started_at) * 1000:.1f}"
                )
            else:
                self._log(
                    f"[Stream] 📝 分句#{sentence_index}: text_chars={len(item)}"
                )

            await self.connection.send_json(
                {
                    "type": "ai_response_chunk",
                    "turn_id": context.turn_id,
                    "text": item,
                    "accumulated_text": "".join(state["text_parts"]),
                    "sentence_index": sentence_index,
                    "is_first": sentence_index == 1,
                }
            )
            start_payload = (
                {
                    "type": "tts_start",
                    "turn_id": context.turn_id,
                    "text": self._clean_for_tts(item),
                }
                if sentence_index == 1
                else None
            )
            if start_payload is not None:
                state["output_identity"] = self._stream_output_identity(context)
            try:
                tts_result = await self._send_tts_chunks(
                    context,
                    item,
                    label=f"-S{sentence_index}",
                    persist=False,
                    content_text=item,
                    audio_parts=state["audio_parts"],
                    start_payload=start_payload,
                )
            except Exception:
                if start_payload is not None:
                    # VoiceTTSStreamer sends tts_start before entering the provider.
                    state["started"] = True
                raise
            state["total_chunks"] += tts_result["chunks"]
            state["total_samples"] += tts_result["samples"]
            if not state["started"] and tts_result.get("started"):
                state["started"] = True
            if tts_result.get("output_identity"):
                state["output_identity"] = tts_result["output_identity"]
            if sentence_index == 1:
                state["first_latency"] = tts_result["first_latency"]
            if tts_result["interrupted"]:
                state["interrupted"] = True
                state["interruption_reason"] = (
                    tts_result.get("interruption_reason") or "cancelled"
                )
                break
        return state

    async def _run_streaming_agent_turn(
        self,
        context: SpeechTurnContext,
    ) -> bool:
        task = self._start_streaming_agent(context)
        try:
            return await self._complete_streaming_agent_turn(context, task)
        except Exception as exc:
            stream = task.get("stream")
            if stream is not None:
                try:
                    await self._finalize_streaming_frontend(
                        context,
                        stream,
                        "error",
                    )
                except Exception as finalize_exc:
                    self._log(
                        "[Stream] ⚠️ 异常收尾失败: "
                        f"{type(finalize_exc).__name__}: {finalize_exc}"
                    )
            self._join_agent_thread(task)
            self._log(
                "[Stream] ❌ 流式回合失败: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            raise

    async def _complete_streaming_agent_turn(
        self,
        context: SpeechTurnContext,
        task: dict,
    ) -> bool:
        stream = await self._consume_streaming_sentences(context, task)

        if stream["interrupted"] or self.is_superseded(
            context.generation_token
        ):
            reason = (
                "response_interrupted"
                if stream["interrupted"]
                else "processing_superseded_in_loop"
            )
            self._log(
                "[Stream] ⏹️ 当前AI回复已被打断，"
                f"跳过剩余回复写回 reason={reason}"
            )
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            await self._finalize_streaming_frontend(
                context,
                stream,
                reason,
            )
            self._join_agent_thread(task)
            return False

        self._join_agent_thread(task)
        if task["error_box"][0]:
            raise task["error_box"][0]
        result = task["result_box"][0]
        if result is None:
            raise RuntimeError("Agent 超时无响应")

        self._log(
            f"[Agent] ✅ 完成: {stream['sentence_count']}个分句, "
            f"Agent耗时={time.perf_counter() - context.agent_started_at:.2f}s"
        )
        if result.get("superseded"):
            self._log("[Stream] ⏹️ Agent检测到处理已被覆盖，丢弃当前结果")
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            await self._finalize_streaming_frontend(
                context,
                stream,
                "agent_superseded",
                fallback_text=result.get("output") or "",
            )
            return False
        if self.is_superseded(context.generation_token):
            self._log(
                "[Stream] ⏹️ Agent结果已被新的用户表达覆盖，丢弃当前结果"
            )
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            await self._finalize_streaming_frontend(
                context,
                stream,
                "processing_superseded_after_agent",
                fallback_text=result.get("output") or "",
            )
            return False

        response_text = result.get("output", "请继续") or "请继续"
        streamed_text = "".join(stream["text_parts"]).strip()
        if streamed_text and len(streamed_text) > len(
            str(response_text).strip()
        ):
            response_text = streamed_text

        await self.connection.send_json(
            {
                "type": "ai_response",
                "turn_id": context.turn_id,
                "text": response_text,
                "finalize_chunks": True,
            }
        )
        if stream["audio_parts"]:
            self.audio_store.persist_assistant(
                np.concatenate(stream["audio_parts"]),
                content_text=response_text,
            )
        if not await self._finish_streaming_audio(
            context,
            stream,
            response_text,
        ):
            return False
        if self.is_superseded(context.generation_token):
            self._log(
                "[Stream] ⏹️ 回复播放完成前检测到新的用户补充，跳过写回"
            )
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            return False

        await self._persist_completed_turn(
            context,
            result,
            response_text,
        )
        return True

    def _join_agent_thread(self, task: dict) -> None:
        self.session.agent.state._stream_sentence_cb = None
        task["thread"].join(timeout=10)
        if task["thread"].is_alive():
            self._log(
                "[Stream] ⚠️ Agent线程仍未退出，"
                "等待后续 supersede 检查自行收敛"
            )

    def _stream_output_identity(self, context: SpeechTurnContext) -> dict:
        processing = self.session.processing
        return {
            "session_id": self.session.session_id,
            "turn_id": context.turn_id,
            "generation": context.generation_token,
            "playback_id": processing.active_playback_id,
        }

    async def _finalize_streaming_frontend(
        self,
        context: SpeechTurnContext,
        stream: dict,
        reason: str,
        fallback_text: str = "",
    ) -> None:
        if stream.get("started"):
            end_payload = {
                "type": "tts_end",
                "duration": stream["total_samples"] / 24000.0,
                "chunks": stream["total_chunks"],
                **(stream.get("output_identity") or {}),
                "reason": stream.get("interruption_reason") or reason,
            }
            await self.connection.send_json(end_payload)
            stream["started"] = False
        accumulated = "".join(stream["text_parts"]).strip()
        text_to_send = fallback_text.strip() or accumulated
        if not text_to_send:
            self._log(
                "[Stream] 🔚 silent-return 无文本可补 "
                f"turn={context.turn_id} reason={reason} "
                f"chunks={stream['sentence_count']}"
            )
            return
        try:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": context.turn_id,
                    "text": text_to_send,
                    "finalize_chunks": True,
                    "error": reason == "error",
                }
            )
            source = "result" if fallback_text else "accumulated"
            self._log(
                "[Stream] 🔚 silent-return 补发 ai_response "
                f"turn={context.turn_id} reason={reason} "
                f"chunks={stream['sentence_count']} "
                f"len={len(text_to_send)} src={source}"
            )
        except Exception as exc:
            self._log(
                "[Stream] ⚠️ 补发 ai_response 失败 "
                f"turn={context.turn_id} reason={reason}: {exc}"
            )

    async def _finish_streaming_audio(
        self,
        context: SpeechTurnContext,
        stream: dict,
        response_text: str,
    ) -> bool:
        if stream["total_samples"] > 0 and stream.get("started"):
            duration = stream["total_samples"] / 24000.0
            self.session.runtime.ai_speaking_until = time.time() + duration
            end_payload = {
                "type": "tts_end",
                "duration": duration,
                "chunks": stream["total_chunks"],
            }
            end_payload.update(stream.get("output_identity") or {})
            await self.connection.send_json(end_payload)
            self._log(
                f"[TTS] ✅ 流式完成: {stream['total_chunks']}块, "
                f"{duration:.1f}s, "
                f"首块延迟={stream['first_latency']:.2f}s"
            )
            return True

        if stream["sentence_count"] != 0:
            return True
        self._log("[Stream] ⚠️ 未收到流式句子，降级为普通TTS")
        if not response_text or self.is_superseded(context.generation_token):
            return True

        audio_parts = []
        result = await self._send_tts_chunks(
            context,
            response_text,
            persist=False,
            content_text=response_text,
            audio_parts=audio_parts,
        )
        if result["interrupted"]:
            if result.get("started"):
                end_payload = {
                    "type": "tts_end",
                    "duration": result["samples"] / 24000.0,
                    "chunks": result["chunks"],
                    **(result.get("output_identity") or {}),
                    "reason": result.get("interruption_reason") or "cancelled",
                }
                await self.connection.send_json(end_payload)
            self._log("[TTS] ⏹️ 降级回复播放被打断，跳过写回")
            return False
        if result["samples"] <= 0:
            return True

        duration = result["samples"] / 24000.0
        self.session.runtime.ai_speaking_until = time.time() + duration
        end_payload = {
            "type": "tts_end",
            "duration": duration,
            "chunks": result["chunks"],
        }
        end_payload.update(result.get("output_identity") or {})
        await self.connection.send_json(end_payload)
        self._log(
            f"[TTS] ✅ 降级完成: {result['chunks']}块, {duration:.1f}s"
        )
        if audio_parts:
            self.audio_store.persist_assistant(
                np.concatenate(audio_parts),
                content_text=response_text,
            )
        return True

    async def _run_synchronous_agent_turn(
        self,
        context: SpeechTurnContext,
    ) -> bool:
        result = self._agent_process_turn_compat(
            user_input=context.text,
            session_id=self.session.session_id,
            patient_profile=context.agent_profile,
            chat_history=context.working_chat_history,
            current_emotion=context.emotion,
            should_abort=lambda: self.is_superseded(
                context.generation_token
            ),
        )
        self._log(
            "[Agent] ✅ Agent 响应耗时: "
            f"{time.perf_counter() - context.agent_started_at:.2f}s"
        )
        if result.get("superseded"):
            self._log("[FLOW] ⚠️ Agent检测到处理已被覆盖，丢弃当前结果")
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            return False
        if self.is_superseded(context.generation_token):
            self._log("[FLOW] ⚠️ Agent返回后检测到新的用户补充，丢弃当前结果")
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            return False

        response_text = result.get("output", "请继续")
        if response_text:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": context.turn_id,
                    "text": response_text,
                }
            )
            if not await self._play_synchronous_response(
                context,
                response_text,
            ):
                return False

        if self.is_superseded(context.generation_token):
            self._log("[TTS] ⚠️ 回复完成前检测到新的用户补充，跳过写回")
            self._mark_memory_status(
                context,
                response_status="cancelled",
                turn_state="RESPONSE_CANCELLED",
            )
            return False
        await self._persist_completed_turn(
            context,
            result,
            response_text,
        )
        return True

    async def _play_synchronous_response(
        self,
        context: SpeechTurnContext,
        response_text: str,
    ) -> bool:
        if self.is_superseded(context.generation_token):
            self._log("[TTS] ⚠️ 生成前检测到打断，跳过")
            return True

        audio_parts = []
        result = await self._send_tts_chunks(
            context,
            response_text,
            persist=False,
            content_text=response_text,
            audio_parts=audio_parts,
        )
        if result["interrupted"]:
            if result.get("started"):
                end_payload = {
                    "type": "tts_end",
                    "duration": result["samples"] / 24000.0,
                    **(result.get("output_identity") or {}),
                    "reason": result.get("interruption_reason") or "cancelled",
                }
                await self.connection.send_json(end_payload)
            self._log("[TTS] ⏹️ 普通回复播放被打断")
            return False
        if result["samples"] <= 0:
            self._log("[TTS] ❌ 没有生成任何音频块")
            return True

        duration = result["samples"] / 24000.0
        self.session.runtime.ai_speaking_until = time.time() + duration
        end_payload = {
            "type": "tts_end",
            "duration": duration,
        }
        end_payload.update(result.get("output_identity") or {})
        await self.connection.send_json(end_payload)
        self._log(
            f"[TTS] ✅ 完成: {result['chunks']}块, {duration:.1f}s, "
            f"首块延迟={result['first_latency']:.2f}s"
        )
        if audio_parts:
            self.audio_store.persist_assistant(
                np.concatenate(audio_parts),
                content_text=response_text,
            )
        return True

    async def _send_tts_chunks(
        self,
        context: SpeechTurnContext,
        text: str,
        *,
        label: str = "",
        persist: bool = True,
        content_text: str | None = None,
        audio_parts: list | None = None,
        start_payload: dict | None | object = _DEFAULT_TTS_START,
    ) -> dict:
        await context.tts_prewarm.consume()
        result = await self._stream_tts_audio(
            text,
            content_text=content_text or text,
            emotion=context.tts_emotion,
            label=label,
            event_type="tts_chunk",
            allow_interrupt=True,
            include_dtype=True,
            persist=persist,
            session_id=self.session.session_id,
            turn_id=context.turn_id,
            generation=context.generation_token,
            playback_id=self.session.processing.active_playback_id,
            start_payload={
                "type": "tts_start",
                "turn_id": context.turn_id,
                "text": self._clean_for_tts(text),
            }
            if start_payload is _DEFAULT_TTS_START
            else start_payload,
        )
        if audio_parts is not None and result["audio_data"] is not None:
            audio_parts.append(result["audio_data"])
        return result

    async def _persist_user_audio(
        self,
        context: SpeechTurnContext,
    ) -> dict[str, Any] | None:
        if context.audio_meta is not None:
            return context.audio_meta
        return await asyncio.to_thread(
            self.audio_store.persist_user,
            context.audio_data,
            context.text,
            extra_meta=context.extra_meta,
        )

    async def _persist_completed_turn(
        self,
        context: SpeechTurnContext,
        result: dict,
        response_text: str,
    ) -> None:
        self.session.processing.revision_enabled = False
        await self.presenter.send_image_display(result, source="audio")
        await self.presenter.send_vision_command(result, source="audio")
        audio_meta = context.audio_meta or await self._persist_user_audio(context)
        if not any(
            item.get("role") == "user" and item.get("turn_id") == context.turn_id
            for item in self.session.chat_history
            if isinstance(item, dict)
        ):
            await self.history_store.append(
                "user", context.text, context.emotion, context.language, audio_meta,
                context.turn_id,
            )
            entry = dict(context.user_history_entry)
            entry["turn_id"] = context.turn_id
            self.session.chat_history.append(entry)
        if not any(
            item.get("role") == "assistant" and item.get("turn_id") == context.turn_id
            for item in self.session.chat_history
            if isinstance(item, dict)
        ):
            self.session.chat_history.append(
                {"role": "assistant", "content": response_text, "turn_id": context.turn_id}
            )
            await self.history_store.append(
                "assistant", response_text, None, None, None, context.turn_id
            )
        self._mark_memory_status(
            context,
            assistant_message=response_text,
            response_status="responded",
            turn_state="RESPONDED",
        )

        selected_topic = result.get("selected_topic")
        if selected_topic:
            asyncio.create_task(
                self.session.agent.background_analysis.analyze(
                    selected_topic,
                    self.session.chat_history,
                )
            )

    async def _send_processing_failure(
        self,
        context: SpeechTurnContext,
        exc: Exception,
    ) -> None:
        self._log(
            f"[FLOW] ❌ 处理失败: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        self._mark_memory_status(
            context,
            response_status="failed",
            turn_state="FAILED",
        )
        try:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": context.turn_id,
                    "text": "",
                    "finalize_chunks": True,
                    "error": True,
                }
            )
        except Exception as send_exc:
            self._log(f"[FLOW] ⚠️ 兜底 finalize 发送失败: {send_exc}")
        try:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": context.turn_id,
                    "text": "抱歉，刚才处理出了点问题，请您再说一次。",
                    "error": True,
                }
            )
        except Exception as send_exc:
            self._log(f"[FLOW] ⚠️ 兜底错误提示发送失败: {send_exc}")

    async def _send_score_update(self) -> None:
        if not is_cognitive_screening(getattr(self.session, "mode", "")):
            return
        try:
            agent = self.session.agent
            mmse_tool = agent.tool_gateway.mmse_tool
            if not mmse_tool:
                from src.tools.agent_tools import MMSEScoringTool

                mmse_tool = MMSEScoringTool()
            summary_json = mmse_tool._run(
                session_id=self.session.session_id,
                action="summary",
                dimension_id="orientation",
                score=0,
                education_years=(
                    self.session.patient_profile or {}
                ).get("education_years"),
            )
            summary_data = json.loads(summary_json)
            if summary_data.get("success"):
                await self.connection.send_json(
                    {
                        "type": "update_score",
                        "mode": mode_for_session(self.session),
                        "data": summary_data,
                    }
                )
        except Exception as exc:
            self._log(f"[评分] 发送评分失败: {type(exc).__name__}")
