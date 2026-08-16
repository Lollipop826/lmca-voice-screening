from __future__ import annotations

class VoiceFeedbackPresenter:
    """Send consistent microphone/input feedback payloads."""

    def __init__(self, connection) -> None:
        self.connection = connection

    async def send(
        self,
        reason: str,
        message: str,
        status_text: str | None = None,
        **fields,
    ) -> None:
        payload = {
            "type": "voice_input_feedback",
            "reason": reason,
            "message": message,
        }
        if status_text:
            payload["status_text"] = status_text
        payload.update(
            {
                key: value
                for key, value in fields.items()
                if value is not None
            }
        )
        await self.connection.send_json(payload)
