from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Optional

from src.voice_modes import normalize_session_mode


@dataclass
class VoiceTurnTakingState:
    """SoulX connection state and interrupted-ASR carry-over for one connection."""

    soulx_healthy: bool = False
    soulx_barge_in_active: bool = False
    soulx_last_state: str = ""
    soulx_last_error: str = ""
    interrupted_asr_prefix: str = ""
    interrupted_asr_time: float = 0.0

    def reset(self) -> None:
        self.soulx_healthy = False
        self.soulx_barge_in_active = False
        self.soulx_last_state = ""
        self.soulx_last_error = ""
        self.interrupted_asr_prefix = ""
        self.interrupted_asr_time = 0.0

    def mark_soulx_unavailable(self, error: str) -> bool:
        """Record an outage and report whether this error should be logged."""
        error_text = str(error or "")
        changed = error_text != self.soulx_last_error
        self.soulx_last_error = error_text
        self.soulx_healthy = False
        self.soulx_barge_in_active = False
        return changed

    def mark_soulx_connected(self) -> bool:
        """Record a healthy response and report whether this is a recovery."""
        recovered = not self.soulx_healthy
        self.soulx_healthy = True
        self.soulx_last_error = ""
        return recovered

    def observe_soulx_state(self, state: str) -> tuple[str, bool]:
        previous = self.soulx_last_state
        normalized = str(state or "")
        changed = normalized != previous
        if normalized and normalized != "blank":
            self.soulx_last_state = normalized
        return previous, changed

    def remember_interrupted_asr(
        self,
        text: str,
        *,
        current_time: float,
        append: bool = False,
    ) -> None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return
        if append and self.interrupted_asr_prefix:
            self.interrupted_asr_prefix = (
                f"{self.interrupted_asr_prefix}，{clean_text}"
            )
        else:
            self.interrupted_asr_prefix = clean_text
        self.interrupted_asr_time = float(current_time)

    def merge_interrupted_asr(
        self,
        text: str,
        *,
        current_time: float,
        ttl_seconds: float = 60.0,
    ) -> tuple[str, str]:
        """Merge a recent carried prefix, then always clear the carry-over."""
        current_text = str(text or "")
        prefix = self.interrupted_asr_prefix
        prefix_time = self.interrupted_asr_time
        self.interrupted_asr_prefix = ""
        self.interrupted_asr_time = 0.0
        if (
            prefix
            and float(current_time) - prefix_time <= float(ttl_seconds)
            and prefix not in current_text
        ):
            return f"{prefix}，{current_text}", prefix
        return current_text, ""


@dataclass
class VoiceRuntimeState:
    """Realtime media/vision state owned by one voice connection."""

    pending_vision_task: Optional[str] = None
    queued_user_text: Optional[str] = None
    vision_lock_time: float = 0.0
    stop_generate: bool = False
    ai_speaking_until: float = 0.0
    ai_streaming_tts: bool = False
    realtime_comfort_playing: bool = False
    high_risk_detected: bool = False
    recent_tts_texts: list[dict[str, Any]] = field(default_factory=list)
    interrupt_audio_buffer: list[Any] = field(default_factory=list)
    interrupt_speech_run: int = 0
    early_vad_armed: bool = False
    early_vad_started_during_ai: bool = False
    early_vad_post_tts_chunks: int = 0
    last_speech_time: float = 0.0
    waiting_for_complete: bool = False
    last_interrupt_text: str = ""
    last_interrupt_intent: Optional[str] = None
    last_interrupt_audio_samples: int = 0

    def is_ai_speaking(self, current_time: float) -> bool:
        return self.ai_streaming_tts or current_time < self.ai_speaking_until

    def reset_interrupt_capture(self, *, reset_waiting: bool = False) -> None:
        self.interrupt_audio_buffer.clear()
        self.interrupt_speech_run = 0
        self.last_speech_time = 0.0
        self.last_interrupt_text = ""
        self.last_interrupt_intent = None
        self.last_interrupt_audio_samples = 0
        if reset_waiting:
            self.waiting_for_complete = False

    def remember_interrupt_judgement(
        self,
        *,
        audio_samples: int,
        text: str,
        intent: str,
    ) -> None:
        self.last_interrupt_text = text or ""
        self.last_interrupt_intent = intent
        self.last_interrupt_audio_samples = int(audio_samples)

    def reuse_interrupt_judgement(
        self,
        *,
        audio_samples: int,
    ) -> tuple[str | None, str | None]:
        if (
            self.last_interrupt_intent
            and int(audio_samples) == self.last_interrupt_audio_samples
        ):
            return self.last_interrupt_text, self.last_interrupt_intent
        return None, None

    def reset_after_end_session(self) -> None:
        """Reset the same realtime fields cleared after an assessment ends."""
        self.pending_vision_task = None
        self.queued_user_text = None
        self.vision_lock_time = 0.0
        self.ai_speaking_until = 0.0
        self.ai_streaming_tts = False
        self.realtime_comfort_playing = False
        self.high_risk_detected = False
        self.recent_tts_texts.clear()
        self.interrupt_audio_buffer.clear()
        self.interrupt_speech_run = 0
        self.waiting_for_complete = False
        self.early_vad_armed = False
        self.early_vad_started_during_ai = False
        self.early_vad_post_tts_chunks = 0
        self.last_speech_time = 0.0

    def reset_for_new_session(self) -> None:
        """Reset realtime fields when the operator explicitly starts a new case."""
        self.pending_vision_task = None
        self.queued_user_text = None
        self.vision_lock_time = 0.0
        self.ai_speaking_until = 0.0
        self.ai_streaming_tts = False
        self.realtime_comfort_playing = False
        self.high_risk_detected = False
        self.recent_tts_texts.clear()
        self.reset_interrupt_capture(reset_waiting=True)

    def reset_early_vad_capture(self) -> None:
        self.early_vad_armed = False
        self.early_vad_started_during_ai = False
        self.early_vad_post_tts_chunks = 0


@dataclass
class VoiceProcessingState:
    """Lifecycle state for one connection's active/revision speech work."""

    task: Any = None
    is_active: bool = False
    revision_enabled: bool = False
    pending_audio: Any = None
    pending_source: str = ""
    pending_extra_meta: dict | None = None
    active_audio: Any = None
    active_source: str = ""
    generation: int = 0
    active_turn_id: str = ""
    active_playback_id: str = ""

    def reset_session_buffers(self) -> None:
        self.revision_enabled = False
        self.pending_audio = None
        self.pending_source = ""
        self.pending_extra_meta = None
        self.active_audio = None
        self.active_source = ""


@dataclass
class VoiceLifecycleState:
    """Conversation lifecycle flags owned by one transport connection."""

    current_patient_id: Optional[str] = None
    started: bool = False
    greeting_sent: bool = False
    awaiting_next_utterance: bool = False
    sealed: bool = False

    def reset(self) -> None:
        self.current_patient_id = None
        self.started = False
        self.greeting_sent = False
        self.awaiting_next_utterance = False
        self.sealed = False


@dataclass
class VoiceSession:
    """Mutable state owned by one WebSocket or WebRTC voice connection.

    The first migration step deliberately contains only transport-independent
    conversation data. Audio pipeline and task lifecycle flags can move here
    in later, independently verifiable groups.
    """

    connection: Any
    agent: Any
    owner_username: str
    session_id: str = ""
    mode: str = field(default_factory=lambda: normalize_session_mode(None))
    history_file: str = ""
    last_message_key: tuple[Optional[str], Optional[str]] = (None, None)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    patient_profile: dict[str, Any] = field(default_factory=dict)
    manual_audio_blob_uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    enroll_sample_uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime: VoiceRuntimeState = field(default_factory=VoiceRuntimeState)
    turn_taking: VoiceTurnTakingState = field(
        default_factory=VoiceTurnTakingState
    )
    processing: VoiceProcessingState = field(default_factory=VoiceProcessingState)
    lifecycle: VoiceLifecycleState = field(default_factory=VoiceLifecycleState)
    memory_engine: Any = None
    long_term_memory_enabled: bool = True
    turn_sequence: int = 0
    accepted_turn_ids: set[str] = field(default_factory=set)
    history_message_keys: set[tuple[str, str, str]] = field(default_factory=set)
    turn_generations: dict[str, int] = field(default_factory=dict)

    def bind_session(self, session_id: str) -> None:
        """Bind persisted identity without discarding in-memory conversation data."""
        self.session_id = str(session_id or "")
        self.history_file = (
            f"data/voice_calls/{self.session_id}/messages.json"
            if self.session_id
            else ""
        )
        self.last_message_key = (None, None)

    def create_fresh_session(
        self,
        create_session: Callable[[str], Any],
        id_factory: Callable[[], str],
        *,
        max_attempts: int = 5,
    ) -> str:
        """Create and bind a unique persisted session using injected storage."""
        last_error: Exception | None = None
        for _ in range(max(1, max_attempts)):
            candidate_session_id = str(id_factory() or "")
            if not candidate_session_id:
                last_error = ValueError("session id factory returned an empty value")
                continue
            try:
                create_session(candidate_session_id)
                self.bind_session(candidate_session_id)
                return self.session_id
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"创建唯一会话失败: {last_error}")

    def reset_conversation(self) -> None:
        """Clear transport-independent state before starting another assessment."""
        self.chat_history.clear()
        self.patient_profile.clear()
        self.lifecycle.reset()
        self.last_message_key = (None, None)
        self.accepted_turn_ids.clear()
        self.history_message_keys.clear()
        self.turn_generations.clear()

    def next_turn_id(self) -> str:
        self.turn_sequence += 1
        return f"turn_{self.turn_sequence:04d}"

    def observe_turn_id(self, turn_id: str) -> None:
        """Advance the local sequence when a persisted session is resumed."""
        suffix = str(turn_id or "").removeprefix("turn_")
        if suffix.isdigit():
            self.turn_sequence = max(self.turn_sequence, int(suffix))
