from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any, Dict, List, Optional

from .answer_evaluation import ScreeningAnswerEvaluation
from .answer_flow import ScreeningAnswerPhase
from .background_analysis import ScreeningBackgroundAnalysis
from .catalog import ScreeningTaskCatalog
from .comfort_flow import ScreeningComfortPhase
from .conversation_policy import ScreeningConversationPolicy
from .question_classifier import ScreeningQuestionClassifier
from .question_flow import ScreeningQuestionPhase
from .question_generation import ScreeningQuestionGeneration
from .routing_flow import ScreeningRoutingPhase
from .session_lifecycle import ScreeningSessionLifecycle
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway
from .turn_models import (
    QuestionPhaseResult,
    RoutingPhaseResult,
    TASK_DESCRIPTIONS,
    TurnContext,
)


class TurnPipeline:
    """Run one screening turn through explicit, composed phases."""

    def __init__(
        self,
        *,
        use_local: bool,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        dimension_map: Dict[str, Dict[str, Any]],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
        verbose_logs: bool,
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.dimension_map = dimension_map
        self._log_summary = log_summary
        self._log_verbose_sink = log_verbose
        self.verbose_logs = verbose_logs

        self.tool_gateway = ScreeningToolGateway(
            use_local=use_local,
            state=state,
            catalog=catalog,
            target_policy_provider=lambda: self.task_planner,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )
        self.tool_gateway._init_tools()
        self.question_classifier = ScreeningQuestionClassifier(
            state=state,
            catalog=catalog,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )
        self.session_lifecycle = ScreeningSessionLifecycle(
            state=state,
            catalog=catalog,
            memory_tool_provider=lambda: self.tool_gateway.memory_tool,
            reset_classifier=self.question_classifier.reset,
        )
        self.task_planner = ScreeningTaskPlanning(
            use_local=use_local,
            state=state,
            catalog=catalog,
            memory_tool_provider=lambda: self.tool_gateway.memory_tool,
            classify_question=self.question_classifier.classify,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )
        self.policy = ScreeningConversationPolicy(
            use_local=use_local,
            state=state,
            catalog=catalog,
            tools=self.tool_gateway,
            task_planner=self.task_planner,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )
        self.answer_evaluator = ScreeningAnswerEvaluation(
            state=state,
            catalog=catalog,
            tools=self.tool_gateway,
            task_planner=self.task_planner,
            dimension_map=dimension_map,
            log_summary=log_summary,
        )
        self.question_generator = ScreeningQuestionGeneration(
            state=state,
            catalog=catalog,
            tools=self.tool_gateway,
            task_planner=self.task_planner,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )
        self.background_analysis = ScreeningBackgroundAnalysis(
            state=state,
            catalog=catalog,
            tools=self.tool_gateway,
            log_summary=log_summary,
            log_verbose=log_verbose,
        )

        phase_dependencies = {
            "state": state,
            "catalog": catalog,
            "task_planner": self.task_planner,
            "tools": self.tool_gateway,
            "log_summary": log_summary,
            "log_verbose": log_verbose,
        }
        self.comfort_phase = ScreeningComfortPhase(
            **phase_dependencies,
            policy=self.policy,
            answer_evaluator=self.answer_evaluator,
            question_generator=self.question_generator,
            dimension_map=dimension_map,
        )
        self.answer_phase = ScreeningAnswerPhase(
            **phase_dependencies,
            question_classifier=self.question_classifier,
            policy=self.policy,
            answer_evaluator=self.answer_evaluator,
        )
        self.routing_phase = ScreeningRoutingPhase(
            **phase_dependencies,
            policy=self.policy,
            answer_evaluator=self.answer_evaluator,
            dimension_map=dimension_map,
        )
        self.question_phase = ScreeningQuestionPhase(
            **phase_dependencies,
            policy=self.policy,
            question_generator=self.question_generator,
            dimension_map=dimension_map,
        )

    def _log_summary_card(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        self._log_summary(title, items)

    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)

    def process_turn(
        self,
        user_input: str,
        dimension: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        current_emotion: str = "neutral",
    ) -> Dict[str, Any]:
        """Run one screening turn through explicit evaluation, routing and question phases."""
        turn = self._prepare_turn(
            user_input=user_input,
            dimension=dimension,
            session_id=session_id,
            patient_profile=patient_profile,
            chat_history=chat_history,
            current_emotion=current_emotion,
        )
        next_task_id: Optional[str] = None

        try:
            last_task_id = self.state._last_task_id
            if (
                self.state.is_in_comfort_mode
                and last_task_id
                and last_task_id not in self.catalog.buffer_tasks
            ):
                self._log_verbose(
                    f"上轮执行了认知任务 [{last_task_id}]，自动退出闲聊模式以进行评分"
                )
                self.state.is_in_comfort_mode = False
                self.state.comfort_turn_count = 0
                self.state._comfort_entry_category = None

            if self.state.is_in_comfort_mode:
                return self.comfort_phase.handle(
                    user_answer=turn.user_answer,
                    session_id=turn.session_id,
                    patient_profile=turn.patient_profile,
                    chat_history=turn.chat_history,
                    current_emotion=turn.current_emotion,
                    dimension_id=turn.dimension_id,
                    dimension_name=turn.dimension_name,
                    start_time=turn.start_time,
                )

            evaluation = self.answer_phase.evaluate_previous_turn(
                user_answer=turn.user_answer,
                doctor_question=turn.doctor_question,
                patient_profile=turn.patient_profile,
                chat_history=turn.chat_history,
                dimension_id=turn.dimension_id,
                dimension_name=turn.dimension_name,
                start_time=turn.start_time,
            )
            if isinstance(evaluation, dict):
                return evaluation

            routing = self.routing_phase.record_score_and_route(
                last_task_id=evaluation.last_task_id,
                expected_answer=evaluation.expected_answer,
                eval_result=evaluation.evaluation,
                user_answer=turn.user_answer,
                doctor_question=turn.doctor_question,
                patient_profile=turn.patient_profile,
                chat_history=turn.chat_history,
                session_id=turn.session_id,
                dimension_id=turn.dimension_id,
                dimension_name=turn.dimension_name,
                start_time=turn.start_time,
            )
            if isinstance(routing, dict):
                return routing
            next_task_id = routing.next_task_id

            question = self.question_phase.generate_next_question(
                next_task_id=next_task_id,
                last_task_id=evaluation.last_task_id,
                eval_result=routing.evaluation,
                user_answer=turn.user_answer,
                patient_profile=turn.patient_profile,
                chat_history=turn.chat_history,
                current_emotion=turn.current_emotion,
                session_id=turn.session_id,
                dimension_id=routing.dimension_id,
                dimension_name=routing.dimension_name,
            )
            return self._build_turn_result(
                turn=turn,
                routing=routing,
                question=question,
                evaluation_elapsed=evaluation.elapsed,
            )
        except Exception as exc:
            return self._build_turn_error(exc)
        finally:
            self._schedule_memory_update(next_task_id)

    def _prepare_turn(
        self,
        *,
        user_input: str,
        dimension: Optional[Dict[str, Any]],
        session_id: Optional[str],
        patient_profile: Optional[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]],
        current_emotion: str,
    ) -> TurnContext:
        import uuid

        start_time = time.time()
        active_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.state.session_id = active_session_id
        self._log_verbose(
            "session_id检查: "
            f"active={self.state._active_session_id}, "
            f"new={active_session_id}"
        )
        self._log_verbose(
            "consent状态: "
            f"pending={self.state._pending_consent_task_id}, "
            f"granted={self.state._consent_granted_task_id}"
        )
        if self.state._active_session_id != active_session_id:
            self._log_summary_card(
                "Session Reset",
                {
                    "session_id": active_session_id,
                    "reason": "session_changed",
                },
            )
            self.session_lifecycle.reset(active_session_id)

        profile = patient_profile or {}
        history = chat_history or []
        self.state.session_data["chat_history"] = history
        if dimension is not None:
            self.state.current_dimension = dimension

        dimension_id = self.state.current_dimension.get("id", "orientation")
        dimension_name = self.state.current_dimension.get("name", "定向力")
        doctor_question = self.state._last_generated_question or "请开始评估"
        if doctor_question == "请开始评估":
            doctor_question = next(
                (
                    message.get("content", "请开始评估")
                    for message in reversed(history)
                    if message.get("role") == "assistant"
                ),
                doctor_question,
            )

        self.state._turn_counter += 1
        # 不变式：每轮开头无条件丢弃上一轮遗留、未被消费的检索预取，
        # 避免早返回路径（retry/comfort/repeat）留下的 future 被本轮
        # Phase 3 误消费为跨轮陈旧结果。
        self.tool_gateway.cancel_retrieval_prefetch()
        self._log_summary_card(
            "Agent Turn",
            {
                "Turn": self.state._turn_counter,
                "Doctor": doctor_question,
                "User": user_input,
                "Dimension": f"{dimension_name} ({dimension_id})",
                "Emotion": current_emotion,
                "PendingConsent": self.state._pending_consent_task_id or "-",
            },
        )
        return TurnContext(
            start_time=start_time,
            session_id=active_session_id,
            patient_profile=profile,
            chat_history=history,
            current_emotion=current_emotion,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            doctor_question=doctor_question,
            user_answer=user_input,
        )

    def _build_turn_result(
        self,
        *,
        turn: TurnContext,
        routing: RoutingPhaseResult,
        question: QuestionPhaseResult,
        evaluation_elapsed: float,
    ) -> Dict[str, Any]:
        self.tool_gateway._call_conversation_storage(
            turn.session_id,
            turn.user_answer,
            question.question,
        )
        total_time = time.time() - turn.start_time
        self._log_summary_card(
            "Agent Result",
            {
                "Task": question.effective_task_id or "-",
                "Output": question.question or "-",
                "TotalTime": f"{total_time:.2f}s",
            },
        )
        self.state._last_generated_question = question.question
        result = {
            "output": question.question,
            "response": question.question,
            "has_resistance": False,
            "resistance_category": "normal",
            "dimension": routing.dimension_name,
            "dimension_id": routing.dimension_id,
            "evaluation": routing.evaluation,
            "mmse_score": routing.mmse_score,
            "selected_topic": self.state._last_bridge_topic,
            "total_time": total_time,
            "step_times": {
                "detection_eval": evaluation_elapsed,
                "score_recording": routing.elapsed,
                "question_gen": question.elapsed,
            },
        }
        if question.image_display:
            result["image_display"] = question.image_display
        if question.vision_command:
            result["vision_command"] = question.vision_command
        return result

    def _build_turn_error(self, exc: Exception) -> Dict[str, Any]:
        self._log_summary_card("Process Turn Error", {"error": type(exc).__name__})
        message = "抱歉，我需要重新理解一下您的回答。能否再说一遍？"
        return {
            "output": message,
            "response": message,
            "error": type(exc).__name__,
        }

    def _schedule_memory_update(self, active_next_task: Optional[str]) -> None:
        history = self.state.session_data.get("chat_history", [])
        undone = [
            task
            for task in self.task_planner._get_valid_next_task_candidates()
            if task != active_next_task
        ]
        task_context = (
            {
                "candidates": undone,
                "task_descriptions": TASK_DESCRIPTIONS,
                "used_topics": list(self.state._used_bridge_topics),
                "last_topic": self.state._last_bridge_topic,
            }
            if undone
            else None
        )
        self._log_verbose(
            "触发异步摘要+任务预选"
            f"（chat_history={len(history)}条, 待选任务={len(undone)}个）"
        )
        self.tool_gateway.memory_tool.update_async(
            history,
            task_context=task_context,
        )
