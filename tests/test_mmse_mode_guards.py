import json

from src.db import database
from src.tools.agent_tools.mmse_scoring_tool import MMSEScoringTool


def test_save_mmse_score_only_writes_cognitive_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "voice.db"))
    database.init_db()
    database.create_session("wellbeing-session", mode="wellbeing")
    database.create_session("cognitive-session", mode="cognitive_screening")

    assert not database.save_mmse_score(
        "wellbeing-session",
        "orientation",
        5,
        10,
    )
    assert not database.save_mmse_score("missing-session", "orientation", 5, 10)
    assert database.save_mmse_score(
        "cognitive-session",
        "orientation",
        5,
        10,
    )

    assert database.get_session_detail("wellbeing-session")["mmse_scores"] == []
    scores = database.get_session_detail("cognitive-session")["mmse_scores"]
    assert [(row["dimension_id"], row["score"]) for row in scores] == [
        ("orientation", 5)
    ]


def test_mmse_scoring_tool_blocks_wellbeing_before_file_write(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "voice.db"))
    database.init_db()
    database.create_session("wellbeing-session", mode="wellbeing")
    database.create_session("cognitive-session", mode="cognitive_screening")

    tool = MMSEScoringTool()

    blocked = json.loads(
        tool._run(
            "wellbeing-session",
            "orientation",
            score=5,
            max_score=10,
        )
    )
    assert blocked["success"] is False
    assert blocked["blocked_reason"] == "NON_COGNITIVE_SESSION"
    assert not (tmp_path / "data" / "mmse_scores" / "wellbeing-session_mmse.json").exists()
    assert database.get_session_detail("wellbeing-session")["mmse_scores"] == []

    saved = json.loads(
        tool._run(
            "cognitive-session",
            "orientation",
            score=5,
            max_score=10,
        )
    )
    assert saved["success"] is True
    assert (tmp_path / "data" / "mmse_scores" / "cognitive-session_mmse.json").exists()
    assert len(database.get_session_detail("cognitive-session")["mmse_scores"]) == 1
