from src.agents.wellbeing_companion_agent import WellbeingMemoryTool


def test_wellbeing_memory_snapshot_marks_session_summary_unavailable():
    snapshot = WellbeingMemoryTool().get_snapshot()

    assert snapshot["available"] is False
    assert snapshot["status"] == "idle"
    assert snapshot["history"] == []
