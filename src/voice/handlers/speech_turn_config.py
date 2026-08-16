from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SpeechTurnConfig:
    """Feature switches and timeouts for one complete speech turn."""

    use_ark_asr: bool
    use_ark_tts: bool
    use_llm_streaming: bool
    vision_lock_timeout: float = 30.0
    memory_timeout_s: float = 3.0
