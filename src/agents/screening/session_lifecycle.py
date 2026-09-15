from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .catalog import ScreeningTaskCatalog
from .state import ScreeningSessionState


class ScreeningSessionLifecycle:
    """Reset one screening session and all session-bound collaborators."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        memory_tool_provider: Callable[[], Any],
        reset_classifier: Callable[[], None],
        mmse_memory_provider: Callable[[], Any] | Any = None,
    ) -> None:
        self.state = state
        self.catalog = catalog
        self._memory_tool_provider = memory_tool_provider
        self._reset_classifier = reset_classifier
        self._mmse_memory = mmse_memory_provider

    def set_memory_manager(self, memory_manager: Any) -> None:
        self._mmse_memory = memory_manager

    def sync_mmse_score(
        self,
        patient_id: str,
        score: int,
        weak_dimensions: list[str] | None = None,
        session_id: str | None = None,
    ) -> bool:
        memory = self._mmse_memory
        if callable(memory) and not hasattr(memory, "update_mmse_score"):
            memory = memory()
        updater = getattr(memory, "update_mmse_score", None)
        if not callable(updater) or not patient_id:
            return False
        if session_id:
            try:
                updater(patient_id, score, weak_dimensions or [], session_id=session_id)
            except TypeError:
                updater(patient_id, score, weak_dimensions or [])
        else:
            updater(patient_id, score, weak_dimensions or [])
        return True

    def reset(self, session_id: str) -> None:
        self._reset_classifier()
        self.state.reset(
            session_id,
            visual_action_tasks=self.catalog.visual_action_tasks,
        )
        memory_tool = self._memory_tool_provider()
        if memory_tool is not None:
            memory_tool.reset()
