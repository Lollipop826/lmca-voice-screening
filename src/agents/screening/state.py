from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.domain.dimensions import MMSE_DIMENSIONS


def _initial_session_data() -> dict[str, Any]:
    return {
        "memory_words": None,
        "calculation_config": None,
    }


@dataclass
class ScreeningSessionState:
    """All mutable data owned by one cognitive-screening conversation."""

    session_id: str | None = None
    current_dimension: dict[str, Any] = field(
        default_factory=lambda: MMSE_DIMENSIONS[0]
    )
    dimension_index: int = 0
    is_in_comfort_mode: bool = False
    comfort_turn_count: int = 0
    consecutive_failures: int = 0
    session_data: dict[str, Any] = field(default_factory=_initial_session_data)

    _active_session_id: str | None = None
    _comfort_entry_category: str | None = None
    _comfort_interrupted_task_id: str | None = None
    _last_task_id: str | None = None
    _last_cognitive_task_id: str | None = None
    _task_done: set[str] = field(default_factory=set)
    _task_attempts: dict[str, int] = field(default_factory=dict)
    _task_best: dict[str, dict[str, Any]] = field(default_factory=dict)
    _task_turns: dict[str, int] = field(default_factory=dict)
    _registration_ts: float | None = None
    _turn_counter: int = 0
    _session_start_ts: float = field(default_factory=time.time)
    _asked_to_continue: bool = False
    _asked_questions: list[str] = field(default_factory=list)
    _used_chat_topics: list[str] = field(default_factory=list)
    _used_bridge_topics: list[str] = field(default_factory=list)
    _last_bridge_hint: str | None = None
    _last_bridge_topic: str | None = None
    _last_target_question: str | None = None
    _last_target_task_id: str | None = None
    _consecutive_free_chat: int = 0
    _consecutive_buffer_count: int = 0
    _pending_consent_task_id: str | None = None
    _consent_granted_task_id: str | None = None
    _consent_granted_groups: set[str] = field(default_factory=set)
    _buffer_resume_task_id: str | None = None
    _task_cooldown_until: dict[str, int] = field(default_factory=dict)
    _last_forced_task_id: str | None = None
    _last_generated_question: str = "请开始评估"
    _pending_classification_task: Any = None
    _last_classification_result: str | None = None
    _available_candidates: list[str] = field(default_factory=list)
    _classification_executor: Any = None
    _stream_sentence_cb: Any = None
    _precomputed_next_task: str | None = None
    _current_turn_topic_set: bool = False
    _calculation_current_value: int = 100
    _calculation_step: int = 7

    def reset(
        self,
        session_id: str,
        *,
        visual_action_tasks: set[str] | frozenset[str],
    ) -> None:
        """Reset routing and scoring state for a newly bound session."""
        self._active_session_id = session_id
        self.session_id = session_id
        self._last_task_id = None
        self._last_generated_question = "请开始评估"
        self._task_done = set(visual_action_tasks)
        self._task_attempts = {}
        self._task_best = {}
        self._task_turns = {}
        self._registration_ts = None
        self._turn_counter = 0
        self._session_start_ts = time.time()
        self.is_in_comfort_mode = False
        self.comfort_turn_count = 0
        self._comfort_interrupted_task_id = None
        self._comfort_entry_category = None
        self._asked_to_continue = False
        self._asked_questions = []
        self._used_chat_topics = []
        self._used_bridge_topics = []
        self._last_bridge_hint = None
        self._last_bridge_topic = None
        self._last_target_question = None
        self._last_target_task_id = None
        self._consecutive_free_chat = 0
        self._consecutive_buffer_count = 0
        self._precomputed_next_task = None
        self._current_turn_topic_set = False
        self._pending_consent_task_id = None
        self._consent_granted_task_id = None
        self._consent_granted_groups = set()
        self._buffer_resume_task_id = None
        self._task_cooldown_until = {}
        self._last_forced_task_id = None
        self.session_data["memory_words"] = None
        self.session_data["calculation_config"] = None
        self.dimension_index = 0
        self.current_dimension = MMSE_DIMENSIONS[0]
        self.consecutive_failures = 0
