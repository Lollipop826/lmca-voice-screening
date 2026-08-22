from __future__ import annotations

from pathlib import Path

from src.context_management.emotion_memobase import EmotionMemobase


def _render_pair(tmp_path: Path, name: str, content: str | None, *, stopped: bool = False) -> tuple[str, str]:
    off = EmotionMemobase(str(tmp_path / f"{name}-off.db"), logger=lambda _message: None)
    on = EmotionMemobase(str(tmp_path / f"{name}-on.db"), logger=lambda _message: None)
    if content:
        on.update_memory_by_user("synthetic-patient", {"facts": [content]})
        if stopped:
            item = on.list_memory_items("synthetic-patient")[0]
            on.stop_memory_item("synthetic-patient", item["item_id"], item["version"])
    return (
        off.get_authoritative_card("synthetic-patient"),
        on.get_authoritative_card("synthetic-patient"),
    )


def _blind_contains(card: str, expected: str | None) -> bool:
    return bool(expected and expected in card)


def test_synthetic_memory_off_on_pairs_pass_recall_and_no_false_recall(tmp_path: Path):
    pairs = [
        ("explicit-preference", "喜欢清晨散步", "喜欢清晨散步"),
        ("explicit-boundary", "不希望夜间被提醒", "不希望夜间被提醒"),
        ("no-memory", None, None),
        ("stopped-memory", "已停止使用的旧兴趣", None),
    ]

    scores = []
    for name, content, expected in pairs:
        off, on = _render_pair(tmp_path, name, content, stopped=name == "stopped-memory")
        scores.append((_blind_contains(off, expected), _blind_contains(on, expected)))

    assert scores == [
        (False, True),
        (False, True),
        (False, False),
        (False, False),
    ]


def test_rollout_gate_remains_closed_without_explicit_enablement(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMORY_LOCAL_FALLBACK_ENABLED", raising=False)
    memory = EmotionMemobase(str(tmp_path / "gate.db"), logger=lambda _message: None)
    memory.update_memory_by_user("synthetic-patient", {"facts": ["仅用于门槛测试"]})

    assert memory.get_relevant_evidence("synthetic-patient", "请回忆") == ""
