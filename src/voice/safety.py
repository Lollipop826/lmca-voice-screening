"""Deterministic, local safety gate for final patient text."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RiskDecision:
    level: str
    rule_version: str
    source: str
    text: str
    matched_rule_ids: tuple[str, ...] = ()
    text_hmac: str = ""
    decided_at: str = ""

    @property
    def high(self) -> bool:
        return self.level == "high"

    @property
    def restricted(self) -> bool:
        return self.level in {"high", "unknown"}


class FinalRiskGate:
    """Keep final safety handling independent from Agent availability."""

    rule_version = "final-risk-v2"
    _HIGH_RULES = (
        ("self_harm_suicide", "自杀"),
        ("self_harm_want_to_die", "想死"),
        ("self_harm_not_want_live", "不想活"),
        ("self_harm_end_life", "结束生命"),
        ("self_harm_kill_self", "杀了自己"),
        ("self_harm_self_injury", "自残"),
        ("self_harm_hurt_self", "伤害自己"),
        ("self_harm_cannot_live", "活不下去"),
    )
    _MEDIUM_RULES = (
        ("distress_hopeless", "绝望"),
        ("distress_cannot_endure", "撑不住"),
        ("distress_no_meaning", "没有意义"),
        ("distress_want_disappear", "想消失"),
    )
    _audit_key: bytes | None = None

    @classmethod
    def evaluate(
        cls,
        text: str,
        *,
        source: str = "final",
        deadline_ms: int | None = None,
    ) -> RiskDecision:
        normalized = str(text or "").strip()
        started = time.monotonic()
        try:
            if not normalized:
                return cls._decision(
                    "unknown", source, normalized, ("final_text_empty",)
                )
            matches = tuple(
                rule_id
                for rule_id, phrase in cls._HIGH_RULES
                if phrase in normalized
            )
            level = "high" if matches else "low"
            if not matches:
                matches = tuple(
                    rule_id
                    for rule_id, phrase in cls._MEDIUM_RULES
                    if phrase in normalized
                )
                level = "medium" if matches else "low"
            limit = cls._deadline_ms(deadline_ms)
            if (time.monotonic() - started) * 1000 > limit:
                return cls._decision(
                    "unknown", source, normalized, ("risk_scan_timeout",)
                )
            return cls._decision(level, source, normalized, matches)
        except Exception:
            return cls._decision(
                "unknown", source, normalized, ("risk_scan_failed",)
            )

    @classmethod
    def _decision(
        cls,
        level: str,
        source: str,
        text: str,
        matched_rule_ids: tuple[str, ...],
    ) -> RiskDecision:
        return RiskDecision(
            level=level,
            rule_version=cls.rule_version,
            source=str(source or "final"),
            text=text,
            matched_rule_ids=matched_rule_ids,
            text_hmac=cls.text_hmac(text),
            decided_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _deadline_ms(value: int | None) -> int:
        if value is None:
            value = os.getenv("SAFETY_RISK_DEADLINE_MS", "100")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 100

    @classmethod
    def text_hmac(cls, text: str) -> str:
        return hmac.new(
            cls._get_audit_key(),
            str(text or "").encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _get_audit_key(cls) -> bytes:
        if cls._audit_key is not None:
            return cls._audit_key
        configured = (
            os.getenv("SAFETY_AUDIT_HMAC_KEY", "").strip()
            or os.getenv("AUTH_SECRET_KEY", "").strip()
        )
        if configured:
            cls._audit_key = configured.encode("utf-8")
            return cls._audit_key
        secret_path = Path(
            os.getenv("SAFETY_AUDIT_HMAC_FILE", "data/.safety_hmac_secret")
        )
        try:
            if secret_path.exists():
                cls._audit_key = secret_path.read_bytes()
            else:
                secret_path.parent.mkdir(parents=True, exist_ok=True)
                cls._audit_key = secrets.token_bytes(32)
                secret_path.write_bytes(cls._audit_key)
        except OSError:
            cls._audit_key = secrets.token_bytes(32)
        return cls._audit_key

    @staticmethod
    def safety_text() -> str:
        return "听到您这样说，我很担心您现在的安全。请先不要独处，找一位您信任的人陪在身边，并尽快联系当地紧急援助。"

    @staticmethod
    def unknown_safety_text() -> str:
        return "当前无法完成风险核验。请先不要独处，联系您信任的人陪伴；如有立即伤害自己的想法或危险，请联系当地紧急援助。"
