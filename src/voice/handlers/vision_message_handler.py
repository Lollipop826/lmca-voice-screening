from __future__ import annotations

from .agent_output_presenter import AgentOutputPresenter
from ..modes import is_wellbeing_session
from collections.abc import Callable
from typing import Any
import asyncio
import functools
import time

class VisionMessageHandler:
    """Handle debug tasks, drawing evaluation and camera result messages."""

    MESSAGE_TYPES = {
        "debug_trigger_task",
        "drawing_submit",
        "vision_stop",
        "vision_eval_result",
    }

    def __init__(
        self,
        connection,
        *,
        session,
        history_store,
        presenter: AgentOutputPresenter,
        stream_tts_audio: Callable[..., Any],
        clean_for_tts: Callable[[str], str],
        normalize_profile: Callable[[dict | None], dict],
        evaluate_image: Callable[[str, str], dict] | None = None,
        now_factory: Callable[[], float] = time.time,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.history_store = history_store
        self.presenter = presenter
        self._stream_tts_audio = stream_tts_audio
        self._clean_for_tts = clean_for_tts
        self._normalize_profile = normalize_profile
        self._evaluate_image = evaluate_image or self._default_evaluate_image
        self._now = now_factory
        self._log = logger
        self._handlers = {
            "debug_trigger_task": self._handle_debug_task,
            "drawing_submit": self._handle_drawing_submit,
            "vision_stop": self._handle_vision_stop,
            "vision_eval_result": self._handle_vision_result,
        }

    async def handle_message(self, message: dict) -> bool:
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            return False
        if is_wellbeing_session(self.session):
            await self.connection.send_json(
                {
                    "type": "processing_status",
                    "stage": "blocked",
                    "text": "当前陪伴会话不使用认知视觉任务",
                }
            )
            return True
        await handler(message)
        return True

    async def _handle_debug_task(self, message: dict) -> None:
        task_id = str(message.get("task_id") or "").strip()
        self._log(f"[DebugTask] 收到测试任务触发: {task_id}")

        if task_id != "language_3step_action":
            self._log(f"[DebugTask] ⚠️ 不支持的测试任务: {task_id}")
            await self._send_processing_status("暂不支持这个测试任务")
            return
        if not self.session.lifecycle.started or not self.session.session_id:
            self._log(
                f"[DebugTask] ⚠️ 当前还没有有效会话，忽略测试任务: {task_id}"
            )
            await self._send_processing_status("请先开始会话，再使用测试按钮")
            return
        if self.session.processing.is_active:
            self._log(
                f"[DebugTask] ⚠️ 当前正在处理中，忽略测试任务: {task_id}"
            )
            await self._send_processing_status("当前正在处理中，请稍后再试")
            return
        if self.session.runtime.pending_vision_task:
            self._log(
                f"[DebugTask] ⚠️ 当前已有视觉任务在进行: "
                f"{self.session.runtime.pending_vision_task}"
            )
            await self._send_processing_status(
                "当前已有视频任务在进行，请先完成"
            )
            return

        try:
            result = self._build_debug_task_result(task_id)
            await self.presenter.send_image_display(result, source="debug")
            await self.presenter.send_vision_command(result, source="debug")
            response = result.get("output", "请按提示完成测试。")
            await self._send_agent_response(
                response,
                label="-debug-task",
            )
            self._log(f"[DebugTask] ✅ 已启动测试任务: {task_id}")
        except Exception as exc:
            self._log(f"[DebugTask] ❌ 启动测试任务失败: {type(exc).__name__}")
            await self._send_processing_status("测试任务启动失败，请稍后重试")

    async def _handle_drawing_submit(self, message: dict) -> None:
        image_base64 = message.get("image", "")
        drawing_mode = message.get("mode", "pentagons")
        if not image_base64:
            await self.connection.send_json(
                {"type": "drawing_result", "error": "未收到图片数据"}
            )
            return

        if drawing_mode == "writing_prompt":
            vision_task = "language_writing_sentence"
            evaluation_label = "手写句子"
        else:
            vision_task = "copy_pentagons"
            evaluation_label = "临摹五边形"
        self._log(
            f"[画图] 收到{evaluation_label}，开始 VLM 评估..."
        )
        await self.connection.send_json(
            {
                "type": "drawing_evaluating",
                "message": f"正在评估{evaluation_label}...",
            }
        )

        try:
            evaluation = await asyncio.to_thread(
                self._evaluate_image,
                image_base64,
                vision_task,
            )
            self._log(
                f"[画图] VLM 评估结果 ({vision_task}): {evaluation}"
            )
            await self.connection.send_json(
                {"type": "drawing_result", "result": evaluation}
            )
            synthetic_input = self._drawing_synthetic_input(
                drawing_mode,
                evaluation,
            )
            await self._run_synthetic_agent_turn(
                synthetic_input,
                source="drawing",
                label="-drawing",
            )
        except Exception as exc:
            self._log(f"[画图] ❌ 评估失败: {type(exc).__name__}")
            await self.connection.send_json(
                {"type": "drawing_result", "error": str(exc)}
            )

    async def _handle_vision_stop(self, _message: dict) -> None:
        runtime = self.session.runtime
        if not runtime.pending_vision_task:
            return
        self._log(
            f"[Vision] 🔓 收到 vision_stop，释放视觉任务锁: "
            f"{runtime.pending_vision_task}"
        )
        runtime.pending_vision_task = None
        runtime.queued_user_text = None

    async def _handle_vision_result(self, message: dict) -> None:
        task_id = message.get("task_id", "")
        evaluation = message.get("result", {})
        runtime = self.session.runtime
        self._log(
            f"[视觉评估] 收到前端评估结果: "
            f"task={task_id}, result={evaluation}"
        )

        if not runtime.pending_vision_task:
            self._log(
                f"[Vision] ⚠️ 忽略过期视觉结果: "
                f"task={task_id}, 当前无待处理视觉任务"
            )
            await self.connection.send_json(
                {"type": "vision_stop", "task_id": task_id}
            )
            return
        if task_id != runtime.pending_vision_task:
            self._log(
                f"[Vision] ⚠️ 忽略不匹配视觉结果: task={task_id}, "
                f"当前待处理={runtime.pending_vision_task}"
            )
            await self.connection.send_json(
                {"type": "vision_stop", "task_id": task_id}
            )
            return

        await self.connection.send_json(
            {"type": "vision_stop", "task_id": task_id}
        )
        saved_user_text = runtime.queued_user_text
        runtime.pending_vision_task = None
        runtime.queued_user_text = None
        if saved_user_text:
            self._log(
                "[Vision] 🔓 视觉任务完成，附带暂存语音: "
                f"text_chars={len(saved_user_text)}"
            )
        else:
            self._log("[Vision] 🔓 视觉任务完成，无暂存语音")

        synthetic_input = self._vision_synthetic_input(
            task_id,
            evaluation,
            saved_user_text=saved_user_text,
        )
        try:
            await self._run_synthetic_agent_turn(
                synthetic_input,
                source="vision",
                label="-vision",
            )
        except Exception as exc:
            self._log(f"[视觉评估] ❌ Agent 处理失败: {type(exc).__name__}")

    async def _run_synthetic_agent_turn(
        self,
        synthetic_input: str,
        *,
        source: str,
        label: str,
    ) -> None:
        await self._wait_until_idle()
        self.session.processing.is_active = True
        try:
            self.session.chat_history.append(
                {"role": "system", "content": synthetic_input}
            )
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    self.session.agent.process_turn,
                    user_input=synthetic_input,
                    session_id=self.session.session_id,
                    patient_profile=self._profile(),
                    chat_history=self.session.chat_history,
                ),
            )
            await self.presenter.send_image_display(result, source=source)
            await self.presenter.send_vision_command(result, source=source)
            response = result.get("output", "好的，我们继续。")
            await self._send_agent_response(response, label=label)
        finally:
            self.session.processing.is_active = False

    async def _send_agent_response(
        self,
        response: str,
        *,
        label: str,
    ) -> None:
        await self.connection.send_json(
            {"type": "ai_response", "text": response}
        )
        await self.history_store.append("assistant", response)
        self.session.chat_history.append(
            {"role": "assistant", "content": response}
        )

        tts_text = self._clean_for_tts(response)
        tts_result = self._stream_tts_audio(
            tts_text,
            content_text=response,
            emotion="neutral",
            label=label,
            event_type="tts_chunk",
            allow_interrupt=False,
            include_dtype=True,
            start_payload={"type": "tts_start", "text": tts_text},
        )
        tts_result = await self._await_if_needed(tts_result)
        total_samples = tts_result["samples"]
        chunk_count = tts_result["chunks"]
        audio_duration = (
            total_samples / 24000.0 if total_samples > 0 else 3.0
        )
        if tts_result.get("started"):
            end_payload = {
                "type": "tts_end",
                "duration": audio_duration,
                "chunks": chunk_count,
            }
            end_payload.update(tts_result.get("output_identity") or {})
            if tts_result.get("interrupted"):
                end_payload["reason"] = (
                    tts_result.get("interruption_reason") or "cancelled"
                )
            await self.connection.send_json(end_payload)
        self.session.runtime.ai_speaking_until = (
            self._now() + audio_duration
        )

    def _build_debug_task_result(self, task_id: str) -> dict:
        if task_id != "language_3step_action":
            raise ValueError(f"unsupported_debug_task:{task_id}")

        agent = self.session.agent
        profile = self._profile()
        full_name = agent.conversation_policy._get_full_name(profile)
        greeting = f"{full_name}，" if full_name else ""
        next_question = (
            f"{greeting}请您按我说的做三个动作，直接做就可以，不用回答。"
            "请您先举起右手，再把手握成拳头，最后把手放到胸前。"
            "如果右手不方便，用左手也可以。"
        )
        dimension_id = "language"
        dimension_name = agent.dimension_map.get(
            dimension_id,
            {},
        ).get("name", "语言")

        agent.task_planner._set_bridge_context(
            None,
            None,
            remember_topic=False,
        )
        state = agent.state
        state.session_id = self.session.session_id
        state.session_data["chat_history"] = self.session.chat_history
        state.current_dimension = agent.dimension_map[dimension_id]
        state.is_in_comfort_mode = False
        state.comfort_turn_count = 0
        state._last_task_id = task_id
        state._last_cognitive_task_id = task_id
        state._last_generated_question = next_question
        state._last_forced_task_id = task_id
        state._pending_consent_task_id = None
        state._consent_granted_task_id = None
        state._buffer_resume_task_id = None
        if next_question:
            state._asked_questions.append(next_question[:80])
            state._asked_questions = state._asked_questions[-20:]

        return {
            "output": next_question,
            "response": next_question,
            "dimension": dimension_name,
            "dimension_id": dimension_id,
            "vision_command": {
                "type": "vision_capture",
                "task_id": task_id,
                "delay": 5000,
                "mode": "manual",
            },
            "total_time": 0.0,
        }

    @staticmethod
    def _drawing_synthetic_input(
        drawing_mode: str,
        evaluation: dict,
    ) -> str:
        is_correct = evaluation.get("is_correct")
        quality = evaluation.get("quality_level", "unknown")
        if drawing_mode == "writing_prompt":
            if is_correct is True:
                return (
                    f"【患者手写完成】写了一个完整的句子，结果："
                    f"正确（{quality}）。"
                    f"{evaluation.get('evaluation_detail', '')}"
                )
            if is_correct is False:
                return (
                    f"【患者手写完成】手写句子，结果："
                    f"不正确（{quality}）。"
                    f"{evaluation.get('evaluation_detail', '')}"
                )
            return "【患者手写完成】手写完成，无法判断结果"

        if is_correct is True:
            return (
                f"【患者画图完成】临摹两个相交五边形，"
                f"结果：正确（{quality}）"
            )
        if is_correct is False:
            return (
                f"【患者画图完成】临摹两个相交五边形，"
                f"结果：不正确（{quality}）"
            )
        return "【患者画图完成】临摹完成，无法判断结果"

    @staticmethod
    def _vision_synthetic_input(
        task_id: str,
        evaluation: dict,
        *,
        saved_user_text: str | None,
    ) -> str:
        evaluation_success = evaluation.get("success", True)
        evaluation_error = evaluation.get("error") or ""
        is_correct = evaluation.get("is_correct")
        quality = evaluation.get("quality_level", "unknown")

        if task_id == "copy_pentagons":
            if evaluation_success is False:
                synthetic_input = (
                    "【患者画图完成】临摹两个相交五边形，"
                    "本次视觉评估失败或无法判断"
                )
            elif is_correct is True:
                synthetic_input = (
                    f"【患者画图完成】临摹两个相交五边形，"
                    f"结果：正确（{quality}）"
                )
            elif is_correct is False:
                synthetic_input = (
                    f"【患者画图完成】临摹两个相交五边形，"
                    f"结果：不正确（{quality}）"
                )
            else:
                synthetic_input = "【患者画图完成】临摹完成，无法判断结果"
        elif task_id == "language_reading_close_eyes":
            if evaluation_success is False:
                synthetic_input = (
                    "【视觉评估】患者的闭眼动作本次视觉评估失败"
                    "或暂时无法判断"
                )
            elif is_correct is True:
                synthetic_input = "【视觉评估】患者完成了闭眼动作"
            elif is_correct is False:
                synthetic_input = "【视觉评估】患者未做出闭眼动作"
            else:
                synthetic_input = "【视觉评估】患者的闭眼动作暂时无法判断"
        elif task_id == "language_3step_action":
            steps = evaluation.get("steps_completed")
            detail = (
                evaluation.get("evaluation_detail")
                or evaluation.get("detail")
                or ""
            )
            if evaluation_success is False:
                synthetic_input = (
                    "【视觉评估】患者的 MMSE 三步动作本次视觉评估失败"
                    "或暂时无法判断"
                )
            elif is_correct is True:
                synthetic_input = (
                    "【视觉评估】患者已完成 MMSE 三步动作，"
                    f"完成步骤数：{steps if steps is not None else 3}/3"
                )
            elif is_correct is False:
                synthetic_input = (
                    "【视觉评估】患者未完全完成 MMSE 三步动作，"
                    f"完成步骤数：{steps if steps is not None else 0}/3"
                )
            else:
                synthetic_input = (
                    "【视觉评估】患者的 MMSE 三步动作暂时无法判断"
                )
            if detail:
                synthetic_input += f"（{detail}）"
        else:
            synthetic_input = (
                f"【视觉评估结果】任务={task_id}, "
                f"正确={is_correct}, 质量={quality}"
            )

        if evaluation_error:
            synthetic_input += f"（错误信息：{evaluation_error}）"
        if saved_user_text:
            synthetic_input += f"（患者口头回复：{saved_user_text}）"
        return synthetic_input

    async def _wait_until_idle(self) -> None:
        while self.session.processing.is_active:
            await asyncio.sleep(0.1)

    async def _send_processing_status(self, text: str) -> None:
        await self.connection.send_json(
            {"type": "processing_status", "text": text}
        )

    def _profile(self) -> dict:
        if not self.session.patient_profile:
            return {}
        return self._normalize_profile(self.session.patient_profile)

    @staticmethod
    def _default_evaluate_image(image_base64: str, task: str) -> dict:
        from src.tools.agent_tools.vision_evaluation_tool import (
            evaluate_image_with_vlm,
        )

        return evaluate_image_with_vlm(image_base64, task)

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result
