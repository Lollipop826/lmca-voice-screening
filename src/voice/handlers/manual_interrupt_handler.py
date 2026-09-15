from __future__ import annotations

from collections.abc import Callable
from typing import Any

class ManualInterruptHandler:
    """Handle the operator's explicit request to stop current assistant speech."""

    MESSAGE_TYPES = {"interrupt"}

    def __init__(
        self,
        *,
        session,
        stop_playback: Callable[[], Any],
        reset_interrupt_capture: Callable[..., Any],
        interrupt_active_processing: Callable[..., Any] | None = None,
        logger=print,
    ) -> None:
        self.session = session
        self._stop_playback = stop_playback
        self._interrupt_active_processing = interrupt_active_processing
        self._reset_interrupt_capture = reset_interrupt_capture
        self._log = logger

    async def handle_message(self, message: dict) -> bool:
        if message.get("type") != "interrupt":
            return False
        self._log("\n[打断] 用户手动请求打断")
        if self._interrupt_active_processing is not None:
            await self._await_if_needed(
                self._interrupt_active_processing("用户手动请求打断")
            )
        else:
            await self._await_if_needed(self._stop_playback())
        reset_result = self._reset_interrupt_capture(
            reset_waiting=True,
            reset_vad=True,
        )
        await self._await_if_needed(reset_result)
        self.session.runtime.reset_early_vad_capture()
        self._log("[打断] ✅ 手动打断完成")
        return True

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result
