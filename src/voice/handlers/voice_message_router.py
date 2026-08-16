from __future__ import annotations

from .message_handler_action import MessageHandlerAction
from .voice_route_result import VoiceRouteResult
from collections.abc import Callable
from typing import Any

class VoiceMessageRouter:
    """Route frontend message types to independently testable handler objects."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict], Any]] = {}

    def register(self, message_type: str, handler: Callable[[dict], Any]) -> None:
        normalized_type = str(message_type or "").strip()
        if not normalized_type:
            raise ValueError("message_type must not be empty")
        if normalized_type in self._handlers:
            raise ValueError(
                f"handler already registered for message type: {normalized_type}"
            )
        self._handlers[normalized_type] = handler

    def register_many(
        self,
        message_types,
        handler: Callable[[dict], Any],
    ) -> None:
        for message_type in message_types:
            self.register(message_type, handler)

    async def dispatch(self, message: dict) -> VoiceRouteResult:
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            return VoiceRouteResult(handled=False)
        result = handler(message)
        if hasattr(result, "__await__"):
            result = await result
        return VoiceRouteResult(
            handled=True,
            stop_connection=result is MessageHandlerAction.STOP_CONNECTION,
        )
