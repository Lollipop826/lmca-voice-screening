from __future__ import annotations

from collections.abc import Callable
import json
import time

from ..modes import is_cognitive_screening

class AgentOutputPresenter:
    """Present image and camera commands returned by the screening Agent."""

    def __init__(
        self,
        connection,
        *,
        session,
        now_factory: Callable[[], float] = time.time,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self._now = now_factory
        self._log = logger

    async def send_image_display(
        self,
        agent_result: dict,
        *,
        source: str = "",
    ) -> bool:
        if not isinstance(agent_result, dict):
            return False

        raw_command = agent_result.get("image_display")
        if not raw_command:
            return False
        if not is_cognitive_screening(getattr(self.session, "mode", "")):
            return False

        command = raw_command
        if isinstance(command, str):
            try:
                command = json.loads(command)
            except Exception as exc:
                self._log(
                    f"[ImageDisplay] ⚠️ 无法解析图片指令(JSON): "
                    f"{exc}, raw={raw_command}"
                )
                return False

        if not isinstance(command, dict):
            self._log(
                f"[ImageDisplay] ⚠️ 无效图片指令类型: {type(command)}"
            )
            return False

        payload = dict(command)
        command_type = payload.get("type")
        if command_type not in {"show_image", "hide_image"}:
            self._log(
                f"[ImageDisplay] ⚠️ 忽略未知图片指令: {payload}"
            )
            return False

        if command_type == "show_image":
            image_id = payload.get("image_id")
            if image_id and not payload.get("url"):
                payload["url"] = f"/api/mmse-image/{image_id}"

        if not await self.connection.send_json(payload):
            return False
        self._log(
            f"[ImageDisplay] 📤 已发送到前端"
            f"{f'({source})' if source else ''}: type={command_type}, "
            f"image_id={payload.get('image_id')}, url={payload.get('url')}"
        )
        return True

    async def send_vision_command(
        self,
        agent_result: dict,
        *,
        source: str = "",
    ) -> bool:
        if not isinstance(agent_result, dict):
            return False

        command = agent_result.get("vision_command")
        if not command or not isinstance(command, dict):
            return False
        if not is_cognitive_screening(getattr(self.session, "mode", "")):
            return False
        if not await self.connection.send_json(command):
            return False

        runtime = self.session.runtime
        runtime.pending_vision_task = command.get("task_id")
        runtime.queued_user_text = None
        runtime.vision_lock_time = self._now()
        self._log(
            f"[Vision] 📹 已发送视觉检测指令到前端"
            f"{f'({source})' if source else ''}: "
            f"task_id={command.get('task_id')}, "
            f"mode={command.get('mode')}, delay={command.get('delay')}"
        )
        self._log("[Vision] 🔒 已锁定语音输入，等待视觉评估结果")
        return True
