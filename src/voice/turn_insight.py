from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from src.voice_modes import mode_for_session


_EMOTIONS = ("joy", "sadness", "anger", "fear", "anxiety", "calm", "confusion")
_SOURCES = {
    "text_asr",
    "text_rules",
    "emotion2vec_audio+text",
    "text_fallback_model_unavailable",
    "text_fallback_audio_missing",
    "text_fallback_audio_error",
    "unavailable",
}


def _enabled(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _scores(value: Any) -> dict[str, float]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, float] = {}
    for name in _EMOTIONS:
        try:
            number = float(source.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            number = 0.0
        result[name] = round(max(0.0, number), 6)
    total = sum(result.values())
    if total > 0:
        result = {name: round(value / total, 6) for name, value in result.items()}
    return result


def _emotion_payload(
    *,
    emotion: str = "",
    scores: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    state: str,
) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, Mapping) else {}
    raw_source = str(meta.get("source") or ("text_asr" if state == "provisional" else "unavailable"))
    source = raw_source if raw_source in _SOURCES else "unavailable"
    status = str(meta.get("analysis_status") or state).strip().lower()
    if status not in {"provisional", "final", "unavailable"}:
        status = "unavailable"
    audio_used = bool(meta.get("audio_model_used")) and source == "emotion2vec_audio+text"
    try:
        inference_ms = round(max(0.0, float(meta.get("inference_ms") or 0.0)), 1)
    except (TypeError, ValueError):
        inference_ms = 0.0
    clean_scores = _scores(scores)
    dominant = str(meta.get("dominant") or emotion or "").strip()
    if dominant not in _EMOTIONS:
        dominant = max(clean_scores, key=clean_scores.get) if any(clean_scores.values()) else "unknown"
    return {
        "source": source,
        "analysis_status": status,
        "audio_model_used": audio_used,
        "dominant": dominant,
        "scores": clean_scores,
        "inference_ms": inference_ms,
    }


def build_turn_insight(
    *,
    session: Any,
    turn_id: str,
    state: str,
    emotion: str = "",
    scores: Mapping[str, Any] | None = None,
    emotion_metadata: Mapping[str, Any] | None = None,
    used_item_ids: list[str] | None = None,
    written_item_ids: list[str] | None = None,
    summary_changed: bool = False,
    risk_decision: Any = None,
    risk_handled: bool = False,
) -> dict[str, Any]:
    risk_level = str(getattr(risk_decision, "level", "unknown") or "unknown").lower()
    if risk_level not in {"low", "medium", "high", "unknown"}:
        risk_level = "unknown"
    return {
        "type": "turn_insight",
        "turn_id": str(turn_id or ""),
        "session_id": str(getattr(session, "session_id", "") or ""),
        "state": "final" if state == "final" else "provisional",
        "mode": mode_for_session(session),
        "emotion": _emotion_payload(
            emotion=emotion,
            scores=scores,
            metadata=emotion_metadata,
            state=state,
        ),
        "memory": {
            "used_item_ids": [str(item) for item in (used_item_ids or []) if str(item)],
            "written_item_ids": [str(item) for item in (written_item_ids or []) if str(item)],
            "summary_changed": bool(summary_changed),
        },
        "risk": {
            "level": risk_level,
            "handled": bool(risk_handled),
        },
    }


async def send_turn_insight(connection, **kwargs: Any) -> bool:
    if not _enabled("ENABLE_TURN_INSIGHT"):
        return False
    return bool(await connection.send_json(build_turn_insight(**kwargs)))
