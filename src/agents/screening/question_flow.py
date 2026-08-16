from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Dict, List, Optional

from .catalog import ScreeningTaskCatalog
from .conversation_policy import ScreeningConversationPolicy
from .question_generation import ScreeningQuestionGeneration
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway
from .turn_models import (
    BRIDGED_SPECIAL_TASKS,
    QuestionDraft,
    QuestionPhaseResult,
    SPECIAL_QUESTION_TASKS,
    TASK_TO_SPECIAL_DIMENSION,
)


class ScreeningQuestionPhase:
    """Generate, register and decorate the next screening question."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        task_planner: ScreeningTaskPlanning,
        tools: ScreeningToolGateway,
        policy: ScreeningConversationPolicy,
        question_generator: ScreeningQuestionGeneration,
        dimension_map: Dict[str, Dict[str, Any]],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.task_planner = task_planner
        self.tool_gateway = tools
        self.policy = policy
        self.question_generator = question_generator
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

    def generate_next_question(
        self,
        *,
        next_task_id: str,
        last_task_id: Optional[str],
        eval_result: Dict[str, Any],
        user_answer: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        session_id: str,
        dimension_id: str,
        dimension_name: str,
    ) -> QuestionPhaseResult:
        """Generate, register and decorate the next question."""
        self._log_verbose(f"阶段三: 策略生成 (Next: {next_task_id})")
        started_at = time.time()
        draft = self._build_consent_draft(
            next_task_id,
            patient_profile,
            user_answer,
        )
        if draft.question is None:
            draft = self._compose_question(
                next_task_id=next_task_id,
                last_task_id=last_task_id,
                eval_result=eval_result,
                user_answer=user_answer,
                patient_profile=patient_profile,
                chat_history=chat_history,
                current_emotion=current_emotion,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                effective_task_id=draft.effective_task_id,
            )
        return self._finalize_question_draft(
            draft=draft,
            next_task_id=next_task_id,
            session_id=session_id,
            dimension_id=dimension_id,
            started_at=started_at,
        )

    def _build_consent_draft(
        self,
        next_task_id: str,
        patient_profile: Dict[str, Any],
        user_answer: str,
    ) -> QuestionDraft:
        if not self.task_planner._needs_consent_for_task(next_task_id):
            return QuestionDraft(None, next_task_id)
        task_status = (
            self.state.session_data.get("task_progress", {})
            .get(next_task_id, {})
            .get("status")
        )
        turns_so_far = self.state._task_turns.get(next_task_id, 0)
        consent_group = self.task_planner._get_consent_group(next_task_id)
        is_granted = (
            next_task_id == self.state._consent_granted_task_id
            or (
                consent_group in self.state._consent_granted_groups
                if consent_group
                else False
            )
            or task_status == "in_progress"
            or (turns_so_far > 0 and next_task_id not in self.state._task_done)
        )
        if not is_granted:
            self.state._pending_consent_task_id = next_task_id
            return QuestionDraft(
                self.policy._build_consent_prompt(
                    next_task_id,
                    patient_profile,
                    user_answer,
                ),
                "buffer_consent",
            )
        if next_task_id == self.state._consent_granted_task_id:
            self.state._consent_granted_task_id = None
        return QuestionDraft(None, next_task_id)

    def _compose_question(
        self,
        *,
        next_task_id: str,
        last_task_id: Optional[str],
        eval_result: Dict[str, Any],
        user_answer: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        dimension_id: str,
        dimension_name: str,
        effective_task_id: str,
    ) -> QuestionDraft:
        task_instruction = self.question_generator._get_task_instruction(next_task_id)
        if self.state._last_bridge_hint:
            task_instruction += (
                "\n\n【必须执行的过渡策略】\n"
                f"1. 先顺着话题「{self.state._last_bridge_hint}」回应一句。\n"
                "2. 然后必须自然落到本任务的核心问题。\n"
                "3. 不要一直停留在闲聊话题上，也不要说测评或测试。"
            )
        persona_hooks = self.question_generator._filter_persona_hooks_for_task(
            next_task_id,
            self.question_generator._extract_persona_hooks(patient_profile, chat_history),
        )
        must_include = self.question_generator._get_must_include_for_task(
            next_task_id
        )
        if next_task_id in SPECIAL_QUESTION_TASKS:
            visual_draft = self._generate_visual_special_question(
                next_task_id=next_task_id,
                last_task_id=last_task_id,
                patient_profile=patient_profile,
                chat_history=chat_history,
                current_emotion=current_emotion,
                dimension_name=dimension_name,
                task_instruction=task_instruction,
                persona_hooks=persona_hooks,
                must_include=must_include,
                effective_task_id=effective_task_id,
            )
            if visual_draft is not None:
                return visual_draft
            return self._generate_standard_special_question(
                next_task_id=next_task_id,
                last_task_id=last_task_id,
                user_answer=user_answer,
                patient_profile=patient_profile,
                chat_history=chat_history,
                current_emotion=current_emotion,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                task_instruction=task_instruction,
                persona_hooks=persona_hooks,
                must_include=must_include,
                effective_task_id=effective_task_id,
            )
        return self._generate_general_question(
            next_task_id=next_task_id,
            last_task_id=last_task_id,
            eval_result=eval_result,
            user_answer=user_answer,
            patient_profile=patient_profile,
            chat_history=chat_history,
            current_emotion=current_emotion,
            dimension_name=dimension_name,
            task_instruction=task_instruction,
            persona_hooks=persona_hooks,
            must_include=must_include,
            effective_task_id=effective_task_id,
        )

    def _generate_visual_special_question(
        self,
        *,
        next_task_id: str,
        last_task_id: Optional[str],
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        dimension_name: str,
        task_instruction: str,
        persona_hooks: List[str],
        must_include: List[str],
        effective_task_id: str,
    ) -> Optional[QuestionDraft]:
        if next_task_id in {"language_naming_watch", "language_naming_pencil"}:
            image_id = (
                "watch"
                if next_task_id == "language_naming_watch"
                else "pencil"
            )
            previous_task = self.policy._get_recent_cognitive_task_id(last_task_id)
            followup = previous_task in {
                "language_naming_watch",
                "language_naming_pencil",
            }
            title = (
                "我再给您看一张图片，请您说说这是什么？"
                if followup
                else "我给您看一张图片，请您说说这是什么？"
            )
            image_data = json.loads(
                self.tool_gateway.image_tool._run(
                    image_id=image_id,
                    title=title,
                    action="show",
                )
            )
            target = (
                self.question_generator._build_special_task_fixed_target_question(
                    next_task_id,
                    is_followup_naming=followup,
                )
                or title
            )
            question = self.question_generator._call_fixed_target_question_generation(
                dimension_name,
                patient_profile,
                chat_history,
                task_instruction,
                persona_hooks,
                must_include,
                current_emotion,
                next_task_id,
                target,
            )
            return QuestionDraft(
                question,
                effective_task_id,
                image_data.get("display_command")
                if image_data.get("success")
                else None,
            )
        if next_task_id == "language_reading_close_eyes":
            full_name = self.policy._get_full_name(patient_profile)
            greeting = f"{full_name}，" if full_name else ""
            image_data = json.loads(
                self.tool_gateway.image_tool._run(
                    image_id="close_eyes",
                    title="请看一下图片上的文字，照着做就行",
                    action="show",
                )
            )
            return QuestionDraft(
                (
                    f"{greeting}我给您看一句话，请您看一下，"
                    "按上面的内容做就可以，不用念出来。"
                ),
                effective_task_id,
                image_data.get("display_command")
                if image_data.get("success")
                else None,
                {
                    "type": "vision_capture",
                    "task_id": "language_reading_close_eyes",
                    "delay": 5000,
                    "mode": "auto",
                },
            )
        if next_task_id == "language_3step_action":
            full_name = self.policy._get_full_name(patient_profile)
            greeting = f"{full_name}，" if full_name else ""
            return QuestionDraft(
                (
                    f"{greeting}请您按我说的做三个动作，直接做就可以，"
                    "不用回答。请您先举起右手，再把手握成拳头，"
                    "最后把手放到胸前。如果右手不方便，用左手也可以。"
                ),
                effective_task_id,
                vision_command={
                    "type": "vision_capture",
                    "task_id": "language_3step_action",
                    "delay": 5000,
                    "mode": "manual",
                },
            )
        return None

    def _generate_standard_special_question(
        self,
        *,
        next_task_id: str,
        last_task_id: Optional[str],
        user_answer: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        dimension_id: str,
        dimension_name: str,
        task_instruction: str,
        persona_hooks: List[str],
        must_include: List[str],
        effective_task_id: str,
    ) -> QuestionDraft:
        last_dimension = (
            self.catalog.task_config.get(last_task_id, {}).get("dimension_id")
            if last_task_id
            else None
        )
        current_dimension = self.catalog.task_config.get(
            next_task_id,
            {},
        ).get("dimension_id")
        dimension_switch = (
            last_task_id is None
            or last_task_id in self.catalog.buffer_tasks
            or last_dimension != current_dimension
        )
        special_dimension = TASK_TO_SPECIAL_DIMENSION.get(
            next_task_id,
            dimension_id,
        )
        calculation_current = None
        calculation_step = 7
        if dimension_id == "attention_calculation":
            turns_so_far = self.state._task_turns.get(next_task_id, 0)
            resume = (
                turns_so_far > 0
                and next_task_id not in self.state._task_done
            )
            if not dimension_switch or resume:
                calculation_current = self.state._calculation_current_value
                calculation_step = self.state._calculation_step
        standard = self.tool_gateway._call_standard_question(
            special_dimension,
            is_dimension_switch=dimension_switch,
            memory_words=self.state.session_data.get("memory_words"),
            patient_name=self.policy._get_full_name(patient_profile),
            calculation_current_value=calculation_current,
            calculation_step=calculation_step,
            last_user_message=user_answer,
        )
        if not standard.get("has_standard_question"):
            question = self.tool_gateway._call_question_generation(
                dimension_name,
                "",
                patient_profile,
                chat_history,
                False,
                is_dimension_switch=True,
                needs_encouragement=False,
                resistance_info=None,
                task_instruction=task_instruction,
                persona_hooks=persona_hooks,
                must_include=must_include,
                patient_emotion=current_emotion,
                task_id=next_task_id,
            )
            return QuestionDraft(question, effective_task_id)

        if standard.get("memory_words"):
            self.state.session_data["memory_words"] = standard["memory_words"]
        if standard.get("calculation_config"):
            config = standard["calculation_config"]
            self.state.session_data["calculation_config"] = config
            expected_value = config.get("expected_answer")
            if isinstance(expected_value, int):
                self.state._calculation_current_value = expected_value
        image_display = None
        if standard.get("requires_image"):
            image_config = standard.get("image_config", {})
            image_data = json.loads(
                self.tool_gateway.image_tool._run(
                    image_id=image_config.get("image_id", "pentagons"),
                    title=image_config.get("title", "请看下面的图片"),
                    action="show",
                )
            )
            if image_data.get("success"):
                image_display = image_data.get("display_command")
        question = standard["question"]
        if next_task_id in BRIDGED_SPECIAL_TASKS:
            target = (
                self.question_generator._build_special_task_fixed_target_question(
                    next_task_id,
                    standard_result=standard,
                )
                or question
            )
            question = self.question_generator._call_fixed_target_question_generation(
                dimension_name,
                patient_profile,
                chat_history,
                task_instruction,
                persona_hooks,
                must_include,
                current_emotion,
                next_task_id,
                target,
            )
        return QuestionDraft(
            question,
            effective_task_id,
            image_display=image_display,
        )

    def _generate_general_question(
        self,
        *,
        next_task_id: str,
        last_task_id: Optional[str],
        eval_result: Dict[str, Any],
        user_answer: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        dimension_name: str,
        task_instruction: str,
        persona_hooks: List[str],
        must_include: List[str],
        effective_task_id: str,
    ) -> QuestionDraft:
        if next_task_id in self.catalog.buffer_tasks:
            question = (
                self.policy._generate_answer_to_user_question(
                    user_answer,
                    patient_profile,
                    chat_history,
                )
                if next_task_id == "buffer_answer_question"
                else self.question_generator._generate_buffer_question(
                    next_task_id,
                    patient_profile,
                    chat_history,
                )
            )
            return QuestionDraft(question, effective_task_id)

        retrieval = self.tool_gateway.consume_retrieval_prefetch(
            dimension_name=dimension_name,
            conversation_history=chat_history,
        )
        last_dimension = (
            self.catalog.task_config.get(last_task_id, {}).get("dimension_id")
            if last_task_id
            else None
        )
        current_dimension = self.catalog.task_config.get(
            next_task_id,
            {},
        ).get("dimension_id")
        dimension_switch = (
            last_task_id is None
            or last_task_id in self.catalog.buffer_tasks
            or last_dimension != current_dimension
        )
        question = self.tool_gateway._call_question_generation(
            dimension_name,
            retrieval.get("knowledge_context", ""),
            patient_profile,
            chat_history,
            eval_result.get("need_followup", False),
            is_dimension_switch=dimension_switch,
            needs_encouragement=eval_result.get(
                "needs_encouragement",
                False,
            ),
            resistance_info=eval_result.get("resistance_info"),
            task_instruction=task_instruction,
            persona_hooks=persona_hooks,
            must_include=must_include,
            patient_emotion=current_emotion,
            task_id=next_task_id,
        )
        return QuestionDraft(question, effective_task_id)

    def _finalize_question_draft(
        self,
        *,
        draft: QuestionDraft,
        next_task_id: str,
        session_id: str,
        dimension_id: str,
        started_at: float,
    ) -> QuestionPhaseResult:
        self.state._last_task_id = draft.effective_task_id
        if draft.effective_task_id not in self.catalog.buffer_tasks:
            self.state._last_cognitive_task_id = draft.effective_task_id
        if draft.question and len(draft.question) > 5:
            self.state._asked_questions.append(draft.question[:80])
            self.state._asked_questions = self.state._asked_questions[-20:]

        image_display = draft.image_display
        if not image_display and draft.effective_task_id != "buffer_consent":
            if next_task_id == "copy_pentagons":
                image_display = {
                    "type": "show_image",
                    "image_id": "pentagons",
                    "title": "请照着画出这个图形",
                    "open_drawing": True,
                }
            elif dimension_id == "language":
                image_result = self.tool_gateway._check_and_display_image(
                    draft.question,
                    session_id,
                )
                if image_result.get("should_display"):
                    image_display = image_result.get("display_command")
        elapsed = time.time() - started_at
        self._log_summary_card(
            "Phase 3",
            {
                "Task": next_task_id,
                "Question": draft.question or "-",
                "Time": f"{elapsed:.2f}s",
            },
        )
        return QuestionPhaseResult(
            question=draft.question,
            effective_task_id=draft.effective_task_id,
            image_display=image_display,
            vision_command=draft.vision_command,
            elapsed=elapsed,
        )
