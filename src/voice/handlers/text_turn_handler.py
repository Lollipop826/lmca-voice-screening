from __future__ import annotations

from .agent_output_presenter import AgentOutputPresenter
from collections.abc import Callable
from typing import Any
import asyncio
import functools
import json
import numpy as np
import threading
import time
from ..safety import FinalRiskGate
from ..modes import is_cognitive_screening, mode_for_session
from ..turn_insight import send_turn_insight
from src.tools.emotion import classify_emotion

class TextTurnHandler:
    """Own the complete text-input turn pipeline for one voice session."""

    MESSAGE_TYPES = {"text"}
    _DEFAULT_PROFILE = {
        "name": "测试",
        "age": 70,
        "gender": "女",
        "education_years": 6,
    }

    def __init__(
        self,
        connection,
        *,
        session,
        history_store,
        audio_store,
        presenter: AgentOutputPresenter,
        stream_tts_audio: Callable[..., Any],
        clean_for_tts: Callable[[str], str],
        normalize_profile: Callable[[dict | None], dict],
        use_llm_streaming: bool,
        use_ark_tts: bool = False,
        tts=None,
        mmse_tool_factory: Callable[[], Any] | None = None,
        now_factory: Callable[[], float] = time.time,
        logger=print,
        capture_memory_turn: Callable[..., Any] | None = None,
        record_safety_event: Callable[..., Any] | None = None,
        update_memory_status: Callable[..., Any] | None = None,
    ) -> None:
        self.connection = connection
        self.session = session
        self.history_store = history_store
        self.audio_store = audio_store
        self.presenter = presenter
        self._stream_tts_audio = stream_tts_audio
        self._clean_for_tts = clean_for_tts
        self._normalize_profile = normalize_profile
        self.use_llm_streaming = bool(use_llm_streaming)
        self.use_ark_tts = bool(use_ark_tts)
        self.tts = tts
        self._mmse_tool_factory = (
            mmse_tool_factory or self._default_mmse_tool_factory
        )
        self._now = now_factory
        self._log = logger
        self._capture_memory_turn = capture_memory_turn
        self._record_safety_event = record_safety_event
        self._update_memory_status = update_memory_status

    async def handle_message(self, message: dict) -> bool:
        if message.get("type") != "text":
            return False

        text = str(message.get("data") or "").strip()
        if not text:
            return True

        turn_id = self.session.next_turn_id()
        risk_decision = FinalRiskGate.evaluate(text, source="final_text")
        self._log(f"\n[文字输入] 用户 text_chars={len(text)}")
        self._log(
            f"[TURN] ▶️ start turn_id={turn_id} "
            "source=text generation=text audio_s=0.00"
        )
        await self._wait_until_idle()
        text_scores = classify_emotion(text)
        await self._send_turn_insight(
            turn_id,
            "provisional",
            text_scores,
            risk_decision,
        )

        if self.session.runtime.pending_vision_task:
            if risk_decision.level != "low":
                await self._record_risk(risk_decision, turn_id)
                if risk_decision.restricted:
                    await self._persist_user_evidence(text, turn_id)
                    self._update_status(turn_id, turn_state="SAFETY_GATED")
                    await self._send_safety_response(turn_id, risk_decision)
                    return True
            await self._queue_during_vision(text, turn_id)
            return True

        self.session.processing.is_active = True
        self.session.runtime.stop_generate = False
        processing = self.session.processing
        current_task = asyncio.current_task()
        processing.task = current_task
        processing.generation += 1
        processing.active_turn_id = turn_id
        processing.active_playback_id = (
            f"playback-{turn_id}-{processing.generation}"
        )
        await self.connection.send_json(
            {
                "type": "processing_status",
                "turn_id": turn_id,
                "stage": "agent",
                "text": "🧠 正在理解您的回答...",
            }
        )

        try:
            self.session.chat_history.append(
                {"role": "user", "content": text, "turn_id": turn_id}
            )
            await self.history_store.append(
                "user", text, None, None, None, turn_id
            )
            self._capture_user_evidence(text, turn_id)
            self._update_status(turn_id, turn_state="SAFETY_GATED")

            if risk_decision.level != "low":
                await self._record_risk(risk_decision, turn_id)
                if risk_decision.restricted:
                    await self._send_safety_response(
                        turn_id,
                        risk_decision,
                    )
                    return True

            started_at = self._now()
            self._update_status(turn_id, turn_state="GENERATING")
            profile = self._profile()
            prewarm_task = self._start_tts_prewarm()
            if self.use_llm_streaming:
                result, response, audio_duration = (
                    await self._run_streaming_turn(
                        text,
                        turn_id=turn_id,
                        profile=profile,
                        started_at=started_at,
                        prewarm_task=prewarm_task,
                    )
                )
            else:
                result, response, audio_duration = (
                    await self._run_synchronous_turn(
                        text,
                        turn_id=turn_id,
                        profile=profile,
                    )
                )

            await self.presenter.send_image_display(result, source="text")
            await self.presenter.send_vision_command(result, source="text")
            self._start_background_analysis(result)
            self._log(f"[文字输入] AI response_chars={len(response)}")

            if not self.use_llm_streaming:
                await self.connection.send_json(
                    {
                        "type": "ai_response",
                        "turn_id": turn_id,
                        "text": response,
                    }
                )
            await self.history_store.append(
                "assistant", response, None, None, None, turn_id
            )
            await self._send_score_update()
            self.session.chat_history.append(
                {"role": "assistant", "content": response, "turn_id": turn_id}
            )
            self._update_status(
                turn_id,
                assistant_message=response,
                response_status="responded",
                turn_state="RESPONDED",
            )
            await self._send_turn_insight(
                turn_id,
                "final",
                text_scores,
                risk_decision,
            )
            self.session.runtime.ai_speaking_until = (
                self._now() + audio_duration
            )
        except Exception as exc:
            self._log(f"[文字输入] ❌ 处理失败: {type(exc).__name__}")
            await self._send_failure_response(turn_id)
        finally:
            if processing.task is current_task:
                processing.task = None
                processing.is_active = False
                processing.active_audio = None
                processing.active_source = ""
                processing.active_turn_id = ""
                processing.active_playback_id = ""
        return True

    async def _record_risk(self, decision, turn_id: str) -> None:
        if self._record_safety_event is None:
            return
        try:
            await asyncio.to_thread(
                self._record_safety_event,
                patient_id=self.session.lifecycle.current_patient_id,
                session_id=self.session.session_id,
                turn_id=turn_id,
                source=decision.source,
                level=decision.level,
                rule_version=decision.rule_version,
                text_hmac=decision.text_hmac,
                status=("risk_scan_failed" if decision.level == "unknown" else "handled"),
            )
        except Exception as exc:
            self._log(f"[Safety] ⚠️ 文字风险审计失败: {type(exc).__name__}")

    async def _send_safety_response(self, turn_id: str, decision) -> None:
        text = (
            FinalRiskGate.unknown_safety_text()
            if decision.level == "unknown"
            else FinalRiskGate.safety_text()
        )
        await self.connection.send_json(
            {
                "type": "safety_alert",
                "turn_id": turn_id,
                "severity": "unknown" if decision.level == "unknown" else "high",
                "rule_version": decision.rule_version,
                "source": decision.source,
            }
        )
        await self.connection.send_json(
            {
                "type": "ai_response",
                "turn_id": turn_id,
                "text": text,
                "safety": True,
                "finalize_chunks": True,
            }
        )
        await self.history_store.append("assistant", text, None, None, None, turn_id)
        self.session.chat_history.append(
            {"role": "assistant", "content": text, "turn_id": turn_id}
        )
        self._update_status(
            turn_id,
            assistant_message=text,
            response_status="responded",
            turn_state="RESPONDED",
        )
        await self._send_turn_insight(
            turn_id,
            "final",
            classify_emotion(text),
            decision,
        )
        tts_result = self._stream_tts_audio(
            text,
            content_text=text,
            emotion="gentle",
            label="-text-safety",
            event_type="tts_chunk",
            allow_interrupt=True,
            include_dtype=True,
            persist=False,
            session_id=self.session.session_id,
            turn_id=turn_id,
            generation=self.session.processing.generation,
            playback_id=self.session.processing.active_playback_id,
            start_payload={
                "type": "tts_start",
                "turn_id": turn_id,
                "text": self._clean_for_tts(text),
            },
        )
        if hasattr(tts_result, "__await__"):
            tts_result = await tts_result
        if self._tts_started(tts_result):
            end_payload = {
                "type": "tts_end",
                "turn_id": turn_id,
                "duration": (
                    tts_result.get("samples", 0) / 24000.0
                    if tts_result
                    else 0.0
                ),
            }
            end_payload.update((tts_result or {}).get("output_identity") or {})
            if (tts_result or {}).get("interrupted"):
                end_payload["reason"] = (
                    (tts_result or {}).get("interruption_reason") or "cancelled"
                )
            await self.connection.send_json(end_payload)

    async def _queue_during_vision(
        self,
        text: str,
        turn_id: str,
    ) -> None:
        runtime = self.session.runtime
        runtime.queued_user_text = text
        self.session.chat_history.append(
            {"role": "user", "content": text, "turn_id": turn_id}
        )
        await self.history_store.append("user", text, None, None, None, turn_id)
        self._capture_user_evidence(text, turn_id)
        self._update_status(turn_id, turn_state="SAFETY_GATED")
        acknowledgement = "好的，我看到了，请稍等一下。"
        await self.connection.send_json(
            {
                "type": "ai_response",
                "turn_id": turn_id,
                "text": acknowledgement,
            }
        )
        await self.history_store.append(
            "assistant", acknowledgement, None, None, None, turn_id
        )
        self.session.chat_history.append(
            {"role": "assistant", "content": acknowledgement, "turn_id": turn_id}
        )
        self._update_status(
            turn_id,
            assistant_message=acknowledgement,
            response_status="responded",
            turn_state="RESPONDED",
        )
        self._log(
            f"[Vision] 🔒 视觉任务 [{runtime.pending_vision_task}] "
            f"进行中，text_chars={len(text)}"
        )
        await self._send_turn_insight(
            turn_id,
            "final",
            classify_emotion(text),
            FinalRiskGate.evaluate(text, source="final_text"),
        )

    async def _persist_user_evidence(self, text: str, turn_id: str) -> None:
        self.session.chat_history.append(
            {"role": "user", "content": text, "turn_id": turn_id}
        )
        await self.history_store.append("user", text, None, None, None, turn_id)
        self._capture_user_evidence(text, turn_id)

    def _capture_user_evidence(self, text: str, turn_id: str) -> None:
        if self._capture_memory_turn is None:
            return
        try:
            self._capture_memory_turn(
                self.session, text, "", None, turn_id=turn_id
            )
        except TypeError:
            try:
                self._capture_memory_turn(self.session, text, "", None)
            except Exception as exc:
                self._log(f"[EmotionMemory] ⚠️ 文字证据捕获失败: {type(exc).__name__}")
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 文字证据捕获失败: {type(exc).__name__}")

    async def _run_streaming_turn(
        self,
        text: str,
        *,
        turn_id: str,
        profile: dict,
        started_at: float,
        prewarm_task,
    ) -> tuple[dict, str, float]:
        loop = asyncio.get_running_loop()
        sentence_queue = asyncio.Queue()
        result_box = [None]
        error_box = [None]

        def on_sentence(sentence):
            loop.call_soon_threadsafe(
                sentence_queue.put_nowait,
                str(sentence),
            )

        def run_agent():
            self.session.agent.state._stream_sentence_cb = on_sentence
            try:
                result_box[0] = self.session.agent.process_turn(
                    user_input=text,
                    session_id=self.session.session_id,
                    patient_profile=profile,
                    chat_history=self.session.chat_history,
                )
            except Exception as exc:
                error_box[0] = exc
            finally:
                self.session.agent.state._stream_sentence_cb = None
                loop.call_soon_threadsafe(
                    sentence_queue.put_nowait,
                    None,
                )

        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()

        total_samples = 0
        chunk_count = 0
        sentence_index = 0
        tts_started = False
        stream_output_identity = {}
        stream_interrupted = False
        stream_interruption_reason = ""
        streamed_audio_parts = []
        streamed_text_parts = []

        while True:
            try:
                item = await asyncio.wait_for(
                    sentence_queue.get(),
                    timeout=0.02,
                )
            except asyncio.TimeoutError:
                if not agent_thread.is_alive() and sentence_queue.empty():
                    break
                continue
            if item is None:
                break

            sentence_index += 1
            streamed_text_parts.append(item)
            tts_text = self._clean_for_tts(item)
            if not tts_text:
                continue

            prewarm_task = await self._consume_prewarm(prewarm_task)
            first_sentence = not tts_started
            if first_sentence:
                self._log(
                    "[文字输入-Stream] 🔥 首句到达! "
                    f"延迟={self._now() - started_at:.2f}s: "
                    f"text_chars={len(item)}"
                )
            else:
                self._log(
                    f"[文字输入-Stream] 📝 分句#{sentence_index}: "
                    f"text_chars={len(item)}"
                )

            await self.connection.send_json(
                {
                    "type": "ai_response_chunk",
                    "turn_id": turn_id,
                    "text": item,
                    "accumulated_text": "".join(streamed_text_parts),
                    "sentence_index": sentence_index,
                    "is_first": sentence_index == 1,
                }
            )
            tts_result = self._stream_tts_audio(
                tts_text,
                content_text=item,
                emotion="neutral",
                label=f"-text-S{sentence_index}",
                event_type="tts_chunk",
                allow_interrupt=True,
                include_dtype=True,
                persist=False,
                start_payload=(
                    {
                        "type": "tts_start",
                        "turn_id": turn_id,
                        "text": tts_text,
                    }
                    if first_sentence
                    else None
                ),
            )
            tts_result = await self._await_if_needed(tts_result)
            if not tts_started and self._tts_started(tts_result):
                tts_started = True
            chunk_count += tts_result["chunks"]
            total_samples += tts_result["samples"]
            if tts_result.get("output_identity"):
                stream_output_identity = tts_result["output_identity"]
            if tts_result.get("interrupted"):
                stream_interrupted = True
                stream_interruption_reason = (
                    tts_result.get("interruption_reason") or "cancelled"
                )
                break
            if tts_result["audio_data"] is not None:
                streamed_audio_parts.append(tts_result["audio_data"])

        agent_thread.join(timeout=10)
        if error_box[0]:
            raise error_box[0]
        result = result_box[0]
        if result is None:
            raise RuntimeError("Agent 超时无响应")

        self._log(
            f"[文字输入-Stream] ✅ 完成: {sentence_index}个分句, "
            f"Agent耗时={self._now() - started_at:.2f}s"
        )
        response = result.get("output", "请继续") or "请继续"
        streamed_response = "".join(streamed_text_parts).strip()
        if (
            streamed_response
            and len(streamed_response) > len(str(response).strip())
        ):
            response = streamed_response

        await self.connection.send_json(
            {
                "type": "ai_response",
                "turn_id": turn_id,
                "text": response,
                "finalize_chunks": True,
            }
        )

        if streamed_audio_parts:
            self.audio_store.persist_assistant(
                np.concatenate(streamed_audio_parts),
                content_text=response,
            )

        audio_duration = 3.0
        if total_samples > 0:
            audio_duration = total_samples / 24000.0
        elif sentence_index == 0 and response:
            self._log(
                "[文字输入-Stream] ⚠️ 未收到流式句子，降级为普通TTS"
            )
            fallback = await self._stream_text_fallback(
                response,
                turn_id=turn_id,
                prewarm_task=prewarm_task,
            )
            tts_result = fallback["result"]
            tts_started = fallback["started"]
            stream_output_identity = fallback["output_identity"]
            stream_interrupted = fallback["interrupted"]
            stream_interruption_reason = fallback["interruption_reason"]
            chunk_count += tts_result["chunks"]
            total_samples += tts_result["samples"]
            if tts_result["audio_data"] is not None:
                self.audio_store.persist_assistant(
                    tts_result["audio_data"],
                    content_text=response,
                )
            if total_samples > 0:
                audio_duration = total_samples / 24000.0

        if tts_started:
            await self._send_text_tts_end(
                turn_id,
                audio_duration,
                chunk_count,
                stream_output_identity,
                stream_interruption_reason if stream_interrupted else "",
            )
        return result, str(response), audio_duration

    async def _send_text_tts_end(
        self,
        turn_id: str,
        duration: float,
        chunks: int,
        output_identity: dict,
        reason: str = "",
    ) -> None:
        payload = {
            "type": "tts_end",
            "turn_id": turn_id,
            "duration": duration,
            "chunks": chunks,
            **output_identity,
        }
        if reason:
            payload["reason"] = reason
        await self.connection.send_json(payload)

    async def _stream_text_fallback(
        self,
        response: str,
        *,
        turn_id: str,
        prewarm_task,
    ) -> dict:
        tts_text = self._clean_for_tts(response)
        if not tts_text:
            return {
                "result": {
                    "chunks": 0,
                    "samples": 0,
                    "audio_data": None,
                },
                "started": False,
                "output_identity": {},
                "interrupted": False,
                "interruption_reason": "",
            }
        await self._consume_prewarm(prewarm_task)
        result = await self._await_if_needed(
            self._stream_tts_audio(
                tts_text,
                content_text=response,
                emotion="neutral",
                label="-text-fallback",
                event_type="tts_chunk",
                allow_interrupt=True,
                include_dtype=True,
                persist=False,
                start_payload={
                    "type": "tts_start",
                    "turn_id": turn_id,
                    "text": tts_text,
                },
            )
        )
        return {
            "result": result,
            "started": self._tts_started(result),
            "output_identity": result.get("output_identity") or {},
            "interrupted": bool(result.get("interrupted")),
            "interruption_reason": result.get("interruption_reason") or "",
        }

    async def _run_synchronous_turn(
        self,
        text: str,
        *,
        turn_id: str,
        profile: dict,
    ) -> tuple[dict, str, float]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(
                self.session.agent.process_turn,
                user_input=text,
                session_id=self.session.session_id,
                patient_profile=profile,
                chat_history=self.session.chat_history,
            ),
        )
        response = result.get("output", "请继续")
        tts_text = self._clean_for_tts(response)
        tts_result = self._stream_tts_audio(
            tts_text,
            content_text=response,
            emotion="neutral",
            label="-text-sync",
            event_type="tts_chunk",
            allow_interrupt=True,
            include_dtype=True,
            persist=False,
            start_payload={
                "type": "tts_start",
                "turn_id": turn_id,
                "text": tts_text,
            },
        )
        tts_result = await self._await_if_needed(tts_result)
        total_samples = tts_result["samples"]
        chunk_count = tts_result["chunks"]
        audio_duration = (
            total_samples / 24000.0 if total_samples > 0 else 3.0
        )
        if self._tts_started(tts_result):
            end_payload = {
                "type": "tts_end",
                "turn_id": turn_id,
                "duration": audio_duration,
                "chunks": chunk_count,
            }
            end_payload.update(tts_result.get("output_identity") or {})
            if tts_result.get("interrupted"):
                end_payload["reason"] = (
                    tts_result.get("interruption_reason") or "cancelled"
                )
            await self.connection.send_json(end_payload)
        if tts_result["audio_data"] is not None:
            self.audio_store.persist_assistant(
                tts_result["audio_data"],
                content_text=response,
            )
        return result, str(response), audio_duration

    async def _send_score_update(self) -> None:
        if not is_cognitive_screening(getattr(self.session, "mode", "")):
            return
        try:
            mmse_tool = self.session.agent.tool_gateway.mmse_tool
            if not mmse_tool:
                mmse_tool = self._mmse_tool_factory()
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
                self._log(
                    "[评分] 已发送MMSE更新: "
                    f"{summary_data.get('total_score')}/"
                    f"{summary_data.get('total_max_score', 35)}"
                )
        except Exception as exc:
            self._log(f"[评分] 发送评分失败: {type(exc).__name__}")

    def _start_background_analysis(self, result: dict) -> None:
        selected_topic = result.get("selected_topic")
        if not selected_topic:
            return
        self._log(f"[文字输入] 🚀 触发后台映射任务: {selected_topic}")
        asyncio.create_task(
            self.session.agent.background_analysis.analyze(
                selected_topic,
                self.session.chat_history,
            )
        )

    def _start_tts_prewarm(self):
        if not self.use_ark_tts or self.tts is None:
            return None
        prewarm = getattr(self.tts, "prewarm", None)
        if not callable(prewarm):
            return None
        return asyncio.create_task(prewarm())

    async def _send_failure_response(self, turn_id: str) -> None:
        self._update_status(
            turn_id,
            assistant_message="抱歉，刚才处理出了点问题，请您再说一次。",
            response_status="failed",
            turn_state="FAILED",
        )
        try:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": turn_id,
                    "text": "",
                    "finalize_chunks": True,
                    "error": True,
                }
            )
        except Exception as exc:
            self._log(
                f"[文字输入] ⚠️ 兜底 finalize 发送失败: {exc}"
            )
        try:
            await self.connection.send_json(
                {
                    "type": "ai_response",
                    "turn_id": turn_id,
                    "text": "抱歉，刚才处理出了点问题，请您再说一次。",
                    "error": True,
                }
            )
        except Exception as exc:
            self._log(f"[文字输入] ⚠️ 兜底错误提示发送失败: {type(exc).__name__}")

        await self._send_turn_insight(
            turn_id,
            "final",
            {},
            None,
            unavailable=True,
        )

    async def _send_turn_insight(
        self,
        turn_id: str,
        state: str,
        scores: dict[str, float],
        risk_decision: Any,
        *,
        unavailable: bool = False,
    ) -> None:
        metadata = {
            "source": "unavailable" if unavailable else "text_rules",
            "analysis_status": "unavailable" if unavailable else state,
            "audio_model_used": False,
            "inference_ms": 0.0,
        }
        try:
            await send_turn_insight(
                self.connection,
                session=self.session,
                turn_id=turn_id,
                state=state,
                emotion="",
                scores=scores,
                emotion_metadata=metadata,
                risk_decision=risk_decision,
                risk_handled=bool(
                    risk_decision
                    and getattr(risk_decision, "level", "") != "low"
                ),
            )
        except Exception as exc:
            self._log(f"[Insight] ⚠️ 文字回合洞察发送失败: {type(exc).__name__}")

    def _update_status(self, turn_id: str, **kwargs: Any) -> None:
        if self._update_memory_status is None:
            return
        try:
            self._update_memory_status(self.session, turn_id, **kwargs)
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 文字轮次状态回写失败: {type(exc).__name__}")

    async def _wait_until_idle(self) -> None:
        while self.session.processing.is_active:
            await asyncio.sleep(0.1)

    def _profile(self) -> dict:
        profile = (
            self.session.patient_profile
            if self.session.patient_profile
            else self._DEFAULT_PROFILE
        )
        return self._normalize_profile(profile)

    @staticmethod
    async def _consume_prewarm(task):
        if task is None:
            return None
        try:
            await task
        except Exception:
            pass
        return None

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result

    @staticmethod
    def _tts_started(result) -> bool:
        return bool(result is not None and result.get("started"))

    @staticmethod
    def _default_mmse_tool_factory():
        from src.tools.agent_tools import MMSEScoringTool

        return MMSEScoringTool()
