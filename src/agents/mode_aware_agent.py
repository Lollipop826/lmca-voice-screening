from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.voice_modes import COGNITIVE_SCREENING, WELLBEING, normalize_session_mode


class ModeAwareAgent:
    """Keep transport handlers stable while isolating domain Agents by mode."""

    def __init__(
        self,
        *,
        cognitive_factory: Callable[..., Any],
        wellbeing_factory: Callable[..., Any],
        mode: str = WELLBEING,
    ) -> None:
        self._cognitive_factory = cognitive_factory
        self._wellbeing_factory = wellbeing_factory
        self._mode = normalize_session_mode(mode)
        self._agent = None

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = normalize_session_mode(mode)
        if mode == self._mode and self._agent is not None:
            return
        if self._agent is not None:
            raise RuntimeError("会话 Agent 模式不可中途切换")
        self._mode = mode

    def reset_for_session(self, mode: str) -> None:
        self._mode = normalize_session_mode(mode)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            factory = (
                self._cognitive_factory
                if self._mode == COGNITIVE_SCREENING
                else self._wellbeing_factory
            )
            self._agent = factory(use_local=False)
        return self._agent

    def process_turn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = self._get_agent().process_turn(*args, **kwargs)
        if isinstance(result, dict):
            result.setdefault("mode", self._mode)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get_agent(), name)
