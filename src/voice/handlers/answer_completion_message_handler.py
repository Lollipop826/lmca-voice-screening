from __future__ import annotations

from collections.abc import Callable
from typing import Any

class AnswerCompletionMessageHandler:
    """Handle operator controls for slow-answer completion mode."""

    MESSAGE_TYPES = {
        "set_answer_completion_mode",
        "request_answer_completion_check",
    }

    def __init__(
        self,
        state,
        *,
        notify_status: Callable[..., Any],
        request_check: Callable[..., Any],
    ) -> None:
        self.state = state
        self._notify_status = notify_status
        self._request_check = request_check

    async def handle_message(self, message: dict) -> bool:
        message_type = message.get("type")
        if message_type == "set_answer_completion_mode":
            self.state.enabled = bool(message.get("enabled"))
            self.state.reset_buffer()
            if self.state.enabled:
                await self._notify_status(
                    "enabled",
                    "慢答判定模式已开启。系统会先累积患者回答，"
                    "待您点击🧠按钮时再做“是否说完”的语义判定。",
                )
            else:
                await self._notify_status(
                    "disabled",
                    "慢答判定模式已关闭，已恢复普通语音处理流程。",
                )
            return True

        if message_type == "request_answer_completion_check":
            await self._request_check(trigger="manual")
            return True
        return False
