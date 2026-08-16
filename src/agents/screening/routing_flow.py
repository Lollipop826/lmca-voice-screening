from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Dict, List, Optional

from .answer_evaluation import ScreeningAnswerEvaluation
from .catalog import ScreeningTaskCatalog
from .conversation_policy import ScreeningConversationPolicy
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway
from .turn_models import (
    RoutingPhaseResult,
    TaskChoice,
)


class ScreeningRoutingPhase:
    """Persist scoring progress and select the next screening task."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        task_planner: ScreeningTaskPlanning,
        tools: ScreeningToolGateway,
        policy: ScreeningConversationPolicy,
        answer_evaluator: ScreeningAnswerEvaluation,
        dimension_map: Dict[str, Dict[str, Any]],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.task_planner = task_planner
        self.tool_gateway = tools
        self.policy = policy
        self.answer_evaluator = answer_evaluator
        self.dimension_map = dimension_map
        self._log_summary = log_summary
        self._log_verbose_sink = log_verbose

    def _log_summary_card(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        self._log_summary(title, items)

    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)

    def record_score_and_route(
        self,
        *,
        last_task_id: Optional[str],
        expected_answer: Optional[str],
        eval_result: Dict[str, Any],
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        session_id: str,
        dimension_id: str,
        dimension_name: str,
        start_time: float,
    ) -> RoutingPhaseResult | Dict[str, Any]:
        """Persist the previous score, select and validate the next task."""
        self._log_verbose("阶段二: 评分与进度管理")
        started_at = time.time()
        mmse_result = self._record_previous_task_score(
            last_task_id=last_task_id,
            expected_answer=expected_answer,
            eval_result=eval_result,
            user_answer=user_answer,
            doctor_question=doctor_question,
            session_id=session_id,
            dimension_id=dimension_id,
        )
        choice = self._choose_next_task(
            last_task_id=last_task_id,
            eval_result=eval_result,
            mmse_result=mmse_result,
            user_answer=user_answer,
            patient_profile=patient_profile,
            session_id=session_id,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
        )
        choice = self._validate_task_choice(choice, chat_history)
        elapsed = time.time() - started_at
        self._log_summary_card(
            "Phase 2",
            {
                "NextTask": choice.next_task_id or "assessment_complete",
                "Dimension": (
                    choice.dimension_name
                    if choice.next_task_id
                    else "完成"
                ),
                "Time": f"{elapsed:.2f}s",
            },
        )
        if choice.next_task_id is None:
            return self._build_assessment_completion(
                session_id=session_id,
                patient_profile=patient_profile,
                start_time=start_time,
            )
        return RoutingPhaseResult(
            next_task_id=choice.next_task_id,
            dimension_id=choice.dimension_id,
            dimension_name=choice.dimension_name,
            evaluation=choice.evaluation,
            mmse_score=choice.mmse_score,
            elapsed=elapsed,
        )

    def _record_previous_task_score(
        self,
        *,
        last_task_id: Optional[str],
        expected_answer: Optional[str],
        eval_result: Dict[str, Any],
        user_answer: str,
        doctor_question: str,
        session_id: str,
        dimension_id: str,
    ) -> Dict[str, Any]:
        mmse_result = {"total_score": 0}
        if not last_task_id:
            return mmse_result

        current_turns = self.state._task_turns.get(last_task_id, 0) + 1
        self.state._task_turns[last_task_id] = current_turns
        task_config = self.catalog.task_config.get(last_task_id, {})
        min_turns = task_config.get("min_turns", 1)
        if last_task_id not in self.catalog.buffer_tasks:
            mmse_result = self.answer_evaluator._call_score_recording(
                session_id,
                task_config.get("dimension_id", dimension_id),
                eval_result,
                doctor_question,
                user_answer,
                task_config.get("max_points"),
                task_id=last_task_id,
                expected_answer=expected_answer,
                current_turn=current_turns,
            )
            if not eval_result.get("skipped"):
                self.policy._clear_invalid_attempts(last_task_id)

        if current_turns < min_turns:
            self._log_verbose(
                f"任务继续: {last_task_id} "
                f"(轮数: {current_turns}/{min_turns})"
            )
            return mmse_result

        self.state._task_done.add(last_task_id)
        self.policy._clear_invalid_attempts(last_task_id)
        cognitive_done = [
            task for task in self.state._task_done if task not in self.catalog.buffer_tasks
        ]
        self._log_summary_card(
            "Task Progress",
            {
                "task": last_task_id,
                "status": "completed",
                "turns": f"{current_turns}/{min_turns}",
                "buffer": last_task_id in self.catalog.buffer_tasks,
                "cognitive_done": len(cognitive_done),
            },
        )
        if last_task_id == "registration_3words":
            self.state._registration_ts = time.time()
            self._log_verbose("记录 registration 时间戳，recall 需等待2分钟")
        return mmse_result

    def _choose_next_task(
        self,
        *,
        last_task_id: Optional[str],
        eval_result: Dict[str, Any],
        mmse_result: Dict[str, Any],
        user_answer: str,
        patient_profile: Dict[str, Any],
        session_id: str,
        dimension_id: str,
        dimension_name: str,
    ) -> TaskChoice:
        user_is_asking = self.policy._is_user_asking_question(user_answer)
        if user_is_asking and last_task_id != "buffer_consent":
            self._log_verbose("检测到用户提问，先回答用户问题")
        if self.state._pending_consent_task_id:
            return self._route_pending_consent(
                eval_result=eval_result,
                mmse_result=mmse_result,
                user_answer=user_answer,
                patient_profile=patient_profile,
                session_id=session_id,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
            )
        if user_is_asking:
            self.state._precomputed_next_task = None
            self._log_verbose(
                f"用户提问检测: '{user_answer[:20]}...' → 是提问"
            )
            next_task_id = "buffer_answer_question"
        else:
            next_task_id = self._route_without_consent(last_task_id)
        return TaskChoice(
            next_task_id=next_task_id,
            evaluation=eval_result,
            mmse_score=mmse_result,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
        )

    def _route_pending_consent(
        self,
        *,
        eval_result: Dict[str, Any],
        mmse_result: Dict[str, Any],
        user_answer: str,
        patient_profile: Dict[str, Any],
        session_id: str,
        dimension_id: str,
        dimension_name: str,
    ) -> TaskChoice:
        pending = self.state._pending_consent_task_id
        direct = self.answer_evaluator._try_consume_pending_task_answer(
            pending,
            user_answer,
            patient_profile,
            session_id,
            dimension_id,
            dimension_name,
        )
        if direct:
            eval_result = direct["eval_result"]
            mmse_result = direct["mmse_result"]
            dimension_id = direct["dimension_id"]
            dimension_name = direct["dimension_name"]
            next_task_id = (
                "buffer_chat"
                if set(self.catalog.required_tasks) - self.state._task_done
                else None
            )
            decision = "direct_answer_consumed"
        elif self.policy._check_user_willing_to_continue(user_answer):
            self.state._consent_granted_task_id = pending
            consent_group = self.task_planner._get_consent_group(pending)
            if consent_group:
                self.state._consent_granted_groups.add(consent_group)
            self.state._pending_consent_task_id = None
            next_task_id = pending
            decision = "granted"
        else:
            self.state._task_cooldown_until[pending] = self.state._turn_counter + 3
            self.state._pending_consent_task_id = None
            next_task_id = "buffer_chat"
            decision = "declined"
        self._log_summary_card(
            "Consent Gate",
            {
                "pending": pending,
                "decision": decision,
                "next_task": next_task_id or "assessment_complete",
            },
        )
        return TaskChoice(
            next_task_id=next_task_id,
            evaluation=eval_result,
            mmse_score=mmse_result,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
        )

    def _route_without_consent(
        self,
        last_task_id: Optional[str],
    ) -> Optional[str]:
        remaining_tasks = set(self.catalog.required_tasks) - self.state._task_done
        resume_task_id = self.state._buffer_resume_task_id
        calc_config = self.catalog.task_config.get("attention_calc_life_math", {})
        calc_min_turns = calc_config.get("min_turns", 1)
        calc_turns = self.state._task_turns.get("attention_calc_life_math", 0)
        should_continue_calc = (
            last_task_id == "attention_calc_life_math"
            and "attention_calc_life_math" not in self.state._task_done
            and calc_turns < calc_min_turns
        )
        if should_continue_calc:
            self.state._buffer_resume_task_id = None
            self.task_planner._set_bridge_context(None, None, remember_topic=False)
            self.state._precomputed_next_task = None
            self._log_summary_card(
                "Task Routing",
                {
                    "reason": "continue_calculation",
                    "task": "attention_calc_life_math",
                    "turns": f"{calc_turns}/{calc_min_turns}",
                },
            )
            return "attention_calc_life_math"
        if last_task_id in self.catalog.buffer_tasks and resume_task_id:
            self.state._buffer_resume_task_id = None
            if resume_task_id not in self.state._task_done:
                self._log_summary_card(
                    "Task Routing",
                    {
                        "reason": "buffer_complete",
                        "action": "resume_task",
                        "task": resume_task_id,
                    },
                )
                return resume_task_id
        if (
            last_task_id
            and last_task_id not in self.catalog.buffer_tasks
            and remaining_tasks
        ):
            self.state._buffer_resume_task_id = (
                last_task_id if last_task_id not in self.state._task_done else None
            )
            self.task_planner._set_bridge_context(None, None, remember_topic=False)
            self._log_summary_card(
                "Task Routing",
                {
                    "reason": "post_task_buffer",
                    "from_task": last_task_id,
                    "next_task": "buffer_chat",
                },
            )
            return "buffer_chat"
        if self.state._precomputed_next_task is not None:
            return self._consume_precomputed_task()
        return self.task_planner._select_next_task()

    def _consume_precomputed_task(self) -> Optional[str]:
        valid_candidates = self.task_planner._get_valid_next_task_candidates()
        next_task_id = self.state._precomputed_next_task
        self.state._precomputed_next_task = None
        memory_tool = self.tool_gateway.memory_tool
        suggestion = (
            memory_tool.get_and_clear_suggestion()
            if memory_tool is not None
            else None
        )
        suggested_task = self._apply_memory_task_suggestion(
            suggestion,
            valid_candidates,
            precomputed_task=next_task_id,
        )
        if suggested_task:
            next_task_id = suggested_task

        if (
            next_task_id in self.state._task_done
            and next_task_id not in self.catalog.buffer_tasks
        ):
            self._log_summary_card(
                "Task Routing",
                {
                    "reason": "precomputed_done",
                    "task": next_task_id,
                    "action": "fast_fallback",
                },
            )
            fallback = next(
                (
                    task
                    for task in valid_candidates
                    if task != next_task_id
                ),
                None,
            )
            if fallback:
                self._log_summary_card(
                    "Task Routing",
                    {
                        "reason": "priority_fast_fallback",
                        "next_task": fallback,
                    },
                )
                return fallback
            self._log_verbose("无快速回退候选，走完整选择")
            return self.task_planner._select_next_task()

        self._log_summary_card(
            "Task Routing",
            {
                "reason": "use_precomputed",
                "next_task": next_task_id,
                "topic": self.state._last_bridge_hint,
            },
        )
        if next_task_id == "recall_3words" and self.state._registration_ts:
            elapsed = time.time() - self.state._registration_ts
            if elapsed < 120:
                self._log_summary_card(
                    "Task Routing",
                    {
                        "reason": "recall_wait_window",
                        "task": next_task_id,
                        "wait_left": f"{120 - elapsed:.0f}s",
                        "fallback": "buffer_chat",
                    },
                )
                return "buffer_chat"
        return next_task_id

    def _apply_memory_task_suggestion(
        self,
        suggestion: Optional[Dict[str, Any]],
        valid_candidates: List[str],
        *,
        precomputed_task: Optional[str],
    ) -> Optional[str]:
        if not suggestion:
            return None
        suggested_task = suggestion.get("task_id", "")
        if suggested_task not in valid_candidates:
            self._log_verbose(
                f"摘要预建议 '{suggested_task}' 不在当前合法候选中，"
                f"继续使用预计算任务 '{precomputed_task}'"
            )
            return None
        from_topic = suggestion.get("from_topic", "")
        to_topic = suggestion.get("to_topic", "")
        bridge = suggestion.get("bridge_hint", "")
        self.task_planner._set_bridge_context(
            bridge or (f"{from_topic}→{to_topic}" if from_topic else to_topic),
            to_topic,
            target_question=suggestion.get("target_question", ""),
            target_task_id=suggested_task,
        )
        self.state._current_turn_topic_set = True
        self.state._consecutive_buffer_count = 0
        self._log_summary_card(
            "Task Routing",
            {
                "reason": "summary_override_precomputed",
                "next_task": suggested_task,
                "precomputed": precomputed_task,
                "anchor": suggestion.get("anchor_fact", "") or "无",
            },
        )
        return suggested_task

    def _validate_task_choice(
        self,
        choice: TaskChoice,
        chat_history: List[Dict[str, str]],
    ) -> TaskChoice:
        next_task_id = choice.next_task_id
        if (
            next_task_id
            and next_task_id not in self.catalog.buffer_tasks
            and next_task_id in self.state._task_done
        ):
            self._log_summary_card(
                "Task Routing",
                {"reason": "defensive_reselect", "task": next_task_id},
            )
            next_task_id = self.task_planner._select_next_task()
        if (
            next_task_id
            and next_task_id not in self.catalog.buffer_tasks
            and self.task_planner._was_task_answered_recently(next_task_id, chat_history)
        ):
            self.state._task_done.add(next_task_id)
            self.state._buffer_resume_task_id = None
            self._log_summary_card(
                "Task Routing",
                {"reason": "recent_answer_guard", "task": next_task_id},
            )
            self.task_planner._set_bridge_context(None, None, remember_topic=False)
            fallback_candidates = [
                task
                for task in self.task_planner._get_valid_next_task_candidates()
                if task != next_task_id
                and not self.task_planner._was_task_answered_recently(
                    task,
                    chat_history,
                )
            ]
            if fallback_candidates:
                next_task_id = fallback_candidates[0]
            elif set(self.catalog.required_tasks) - self.state._task_done:
                next_task_id = "buffer_chat"
            else:
                next_task_id = None

        dimension_id = choice.dimension_id
        dimension_name = choice.dimension_name
        if next_task_id is None:
            self._log_summary_card("Assessment", {"status": "complete"})
        else:
            self._log_verbose(f"下一个任务: {next_task_id}")
            next_dimension = self.catalog.task_config.get(
                next_task_id,
                {},
            ).get("dimension_id")
            if next_dimension and next_dimension in self.dimension_map:
                self.state.current_dimension = self.dimension_map[next_dimension]
                dimension_id = next_dimension
                dimension_name = self.state.current_dimension.get("name", "未知")
        return TaskChoice(
            next_task_id=next_task_id,
            evaluation=choice.evaluation,
            mmse_score=choice.mmse_score,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
        )

    def _build_assessment_completion(
        self,
        *,
        session_id: str,
        patient_profile: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        summary = json.loads(
            self.tool_gateway.mmse_tool._run(
                session_id=session_id,
                dimension_id="",
                action="summary",
            )
        )
        total_score = summary.get("total_score", 0)
        scaled_score = summary.get("scaled_total_score", total_score)
        coverage = summary.get("coverage", 1.0)
        risk_assessment = self.answer_evaluator._calculate_alzheimers_risk(
            int(scaled_score)
        )
        self._log_summary_card(
            "MMSE Summary",
            {
                "RawScore": total_score,
                "ScaledScore": f"{scaled_score:.1f}/30",
                "Coverage": f"{coverage:.1%}",
                "Risk": risk_assessment,
            },
        )
        message = self.policy._generate_completion_message(
            int(scaled_score),
            risk_assessment,
            patient_profile,
        )
        return {
            "output": message,
            "response": message,
            "assessment_complete": True,
            "total_score": total_score,
            "scaled_score": scaled_score,
            "coverage": coverage,
            "max_score": summary.get("total_max_score", 35),
            "risk_assessment": risk_assessment,
            "dimension": "完成",
            "total_time": time.time() - start_time,
        }
