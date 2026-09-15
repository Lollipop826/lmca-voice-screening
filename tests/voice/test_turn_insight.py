import asyncio
from types import SimpleNamespace
from unittest import mock

from src.voice.turn_insight import send_turn_insight


class _Connection:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)
        return True


def test_turn_insight_release_flag_can_disable_events():
    connection = _Connection()
    session = SimpleNamespace(session_id="session-1", mode="wellbeing")

    with mock.patch.dict("os.environ", {"ENABLE_TURN_INSIGHT": "false"}):
        sent = asyncio.run(
            send_turn_insight(
                connection,
                session=session,
                turn_id="turn-1",
                state="final",
            )
        )

    assert sent is False
    assert connection.sent == []


def test_turn_insight_release_flag_defaults_to_enabled():
    connection = _Connection()
    session = SimpleNamespace(session_id="session-1", mode="wellbeing")

    with mock.patch.dict("os.environ", {}, clear=True):
        sent = asyncio.run(
            send_turn_insight(
                connection,
                session=session,
                turn_id="turn-1",
                state="provisional",
            )
        )

    assert sent is True
    assert connection.sent[0]["type"] == "turn_insight"


def test_turn_insight_reports_disabled_emotion_and_memory_ablation_metadata():
    connection = _Connection()
    session = SimpleNamespace(session_id="session-1", mode="wellbeing")

    with mock.patch.dict("os.environ", {}, clear=True):
        sent = asyncio.run(
            send_turn_insight(
                connection,
                session=session,
                turn_id="turn-1",
                state="final",
                emotion_metadata={
                    "source": "disabled",
                    "analysis_status": "disabled",
                },
                retrieval_source="disabled",
                retrieval_hit=False,
                retrieval_context_chars=42,
                retrieval_elapsed_ms=1.25,
                writes_enabled=False,
            )
        )

    assert sent is True
    payload = connection.sent[0]
    assert payload["emotion"]["source"] == "disabled"
    assert payload["emotion"]["analysis_status"] == "disabled"
    assert payload["memory"] == {
        "used_item_ids": [],
        "written_item_ids": [],
        "summary_changed": False,
        "retrieval_source": "disabled",
        "retrieval_kind": "none",
        "retrieval_hit": False,
        "retrieval_context_chars": 42,
        "retrieval_elapsed_ms": 1.2,
        "writes_enabled": False,
    }
