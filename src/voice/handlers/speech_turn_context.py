from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class SpeechTurnContext:
    """Mutable state carried through one completed speech turn."""

    audio_data: Any
    generation_token: int | None
    turn_id: str
    extra_meta: dict | None
    tts_prewarm: Any
    text: str = ""
    emotion: str = ""
    emotion_scores: dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = ""
    emotion_metadata: dict[str, Any] = field(default_factory=dict)
    memory_insight: dict[str, Any] = field(default_factory=dict)
    insight_sent: set[str] = field(default_factory=set)
    language: str = ""
    event: str = ""
    asr_source: str = "final"
    tts_emotion: str = "neutral"
    emotion_task: Any = None
    audio_meta: dict | None = None
    user_history_entry: dict = field(default_factory=dict)
    agent_profile: dict = field(default_factory=dict)
    working_chat_history: list = field(default_factory=list)
    processing_started_at: float = field(default_factory=perf_counter)
    asr_result_at: float = 0.0
    agent_started_at: float = 0.0
    memory_capture_elapsed_ms: float = 0.0
    memory_elapsed_ms: float = 0.0
    memory_chars: int = 0
    memory_source: str = "none"
    first_ai_sentence_at: float = 0.0
    risk_decision: Any = None
    risk_event_recorded: bool = False
    accepted: bool = False

    def apply_recognition(self, payload: dict) -> None:
        self.text = str(payload.get("text") or "")
        self.emotion = str(payload.get("emotion") or "")
        self.language = str(payload.get("language") or "")
        self.event = str(payload.get("event") or "")
        self.asr_source = str(payload.get("source") or "final")
        self.tts_emotion = {
            "sad": "gentle",
            "sadness": "gentle",
            "fear": "gentle",
            "fearful": "gentle",
            "anxiety": "gentle",
            "angry": "calm",
            "anger": "calm",
            "happy": "happy",
            "joy": "happy",
        }.get(self.emotion, "neutral")
        self.user_history_entry = {
            "role": "user",
            "content": self.text,
            "emotion": self.emotion,
        }

    def apply_emotion_scores(self, scores: dict[str, float] | None) -> None:
        if not isinstance(scores, dict) or not scores:
            return
        cleaned = {}
        for label, value in scores.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric >= 0:
                cleaned[str(label)] = numeric
        total = sum(cleaned.values())
        if total <= 0:
            return
        self.emotion_scores = {
            label: value / total for label, value in cleaned.items()
        }
        self.dominant_emotion = max(
            self.emotion_scores,
            key=self.emotion_scores.get,
        )
        self.user_history_entry["emotions"] = dict(self.emotion_scores)
        self.tts_emotion = {
            "sadness": "gentle",
            "fear": "gentle",
            "anxiety": "gentle",
            "anger": "calm",
            "joy": "happy",
            "confusion": "gentle",
        }.get(self.dominant_emotion, self.tts_emotion)

    def prepare_agent(self, profile: dict, chat_history: list, now: float) -> None:
        self.agent_profile = profile
        history = list(chat_history)
        turn_id = str(self.user_history_entry.get("turn_id") or "")
        already_present = any(
            isinstance(item, dict)
            and item.get("role") == "user"
            and (
                (turn_id and item.get("turn_id") == turn_id)
                or (not turn_id and item == self.user_history_entry)
            )
            for item in history
        )
        self.working_chat_history = history if already_present else history + [self.user_history_entry]
        self.agent_started_at = now
