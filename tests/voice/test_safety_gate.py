from pathlib import Path
from tempfile import TemporaryDirectory

from src.db import database
from src.voice.safety import FinalRiskGate


def test_final_risk_gate_has_restricted_unknown_and_hmac_without_plaintext():
    high = FinalRiskGate.evaluate("我想自杀", source="fallback_final")
    medium = FinalRiskGate.evaluate("我很绝望", source="final")
    unknown = FinalRiskGate.evaluate("", source="final")

    assert high.level == "high"
    assert high.matched_rule_ids == ("self_harm_suicide",)
    assert high.restricted
    assert medium.level == "medium"
    assert not medium.restricted
    assert unknown.level == "unknown"
    assert unknown.restricted
    assert high.text_hmac
    assert "自杀" not in high.text_hmac


def test_message_and_safety_event_writes_are_idempotent():
    with TemporaryDirectory() as temp_dir:
        previous = database.DB_PATH
        database.DB_PATH = str(Path(temp_dir) / "voice.db")
        try:
            database.init_db()
            database.create_session("session-1")
            assert database.save_message(
                "session-1", "user", "我想自杀", turn_id="turn-1"
            )
            assert not database.save_message(
                "session-1", "user", "我想自杀", turn_id="turn-1"
            )
            first = database.record_safety_event(
                patient_id=None,
                session_id="session-1",
                turn_id="turn-1",
                source="fallback_final",
                level="high",
                rule_version="final-risk-v2",
                text_hmac="hmac-value",
                status="handled",
            )
            second = database.record_safety_event(
                patient_id=None,
                session_id="session-1",
                turn_id="turn-1",
                source="fallback_final",
                level="high",
                rule_version="final-risk-v2",
                text_hmac="hmac-value",
                status="handled",
            )
            assert first["id"] == second["id"]
            assert database.get_session_detail("session-1")["messages"][0]["turn_id"] == "turn-1"
        finally:
            database.DB_PATH = previous
