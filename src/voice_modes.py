from __future__ import annotations

import os
from typing import Any


WELLBEING = "wellbeing"
COGNITIVE_SCREENING = "cognitive_screening"
LEGACY = "legacy"
VALID_MODES = frozenset({WELLBEING, COGNITIVE_SCREENING, LEGACY})


def normalize_session_mode(
    mode: Any,
    *,
    default: str | None = None,
) -> str:
    value = str(mode or "").strip().lower()
    if value in VALID_MODES:
        return value
    configured = str(
        default
        if default is not None
        else os.getenv("DEFAULT_SESSION_MODE", WELLBEING)
    ).strip().lower()
    return configured if configured in VALID_MODES else WELLBEING


def is_cognitive_screening(mode: Any) -> bool:
    return normalize_session_mode(mode, default=LEGACY) == COGNITIVE_SCREENING


def is_wellbeing_session(session: Any) -> bool:
    """Only explicitly tagged production sessions are wellbeing sessions."""
    return str(getattr(session, "mode", "") or "").strip().lower() == WELLBEING


def mode_for_session(session: Any) -> str:
    return normalize_session_mode(getattr(session, "mode", None))
