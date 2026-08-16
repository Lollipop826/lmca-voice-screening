from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LiveAudioFrameContext:
    """State produced while routing one live microphone frame."""

    audio: Any
    was_speaking: bool = False
    complete_audio: Any = None
    drop_reason: str | None = None
    drop_duration_s: float = 0.0
    current_time: float = 0.0
    in_ai_speaking: bool = False
    early_listen_mode: bool = False
    vad_end_sent: bool = False
    realtime_turn: Any = None
