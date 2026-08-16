from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LiveAudioConfig:
    """Thresholds used by the local VAD/full-duplex fallback pipeline."""

    enabled_full_duplex: bool
    pre_end_arm_window_s: float
    minimum_post_tts_chunks: int
    interrupt_min_duration: float
    interrupt_trigger_probability: float
    interrupt_min_rms: float
    interrupt_min_consecutive_chunks: int
    interrupt_complete_silence_s: float
    interrupt_min_complete_audio_s: float
