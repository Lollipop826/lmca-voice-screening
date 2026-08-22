from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest

from src.context_management.emotion_memobase import (
    EmotionMemobase,
    MemoryRevisionConflict,
    _is_local_memobase_url,
    _token_estimate,
)
from src.tools.emotion.emotion_classifier import EMOTION_LABELS, EmotionClassifier


class _FakeMemobaseUser:
    def __init__(self, failures=0):
        self.failures = failures
        self.inserted = []
        self.context_calls = []
        self.search_event_gist_calls = []
        self.flush_calls = []
        self.deleted_events = []
        self.deleted_profiles = []

    def insert(self, blob):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("memobase unavailable")
        self.inserted.append(blob)
        return f"blob-{len(self.inserted)}"

    def context(self, **kwargs):
        self.context_calls.append(kwargs)
        return "2026-06-12，患者曾在浴室滑倒。"

    def search_event_gist(self, query, **kwargs):
        self.search_event_gist_calls.append({"query": query, **kwargs})
        return [
            {
                "id": "gist-1",
                "event_id": "event-1",
                "project_id": "default",
                "session_id": "old-session",
                "turn_id": "turn-old",
                "source_turn_id": "turn-old",
                "gist_data": {"content": "- 2026-06-12，患者曾在浴室滑倒。"},
                "created_at": "2026-06-12T08:00:00",
                "similarity": 0.82,
            }
        ]

    def flush(self, sync=False):
        self.flush_calls.append(sync)
        return True

    def profile(self, **_kwargs):
        return []

    def delete_event(self, event_id):
        self.deleted_events.append(event_id)
        return True

    def delete_profile(self, profile_id):
        self.deleted_profiles.append(profile_id)
        return True


class _FakeMemobaseClient:
    def __init__(self, user):
        self.user = user

    def get_or_create_user(self, patient_id):
        self.patient_id = patient_id
        return self.user


class _DeletionMemobaseUser(_FakeMemobaseUser):
    def profile(self, **_kwargs):
        return [{"id": "profile-delete", "content": "喜欢清晨散步"}]

    def search_event_gist(self, query, **kwargs):
        self.search_event_gist_calls.append({"query": query, **kwargs})
        return [
            {
                "id": "gist-delete",
                "event_id": "event-delete",
                "project_id": "default",
                "session_id": "old-session",
                "turn_id": "turn-delete",
                "source_turn_id": "",
                "gist_data": {"content": "患者喜欢清晨散步。"},
                "created_at": "2026-06-12T08:00:00",
                "similarity": 0.99,
            }
        ]


class _IdempotentMemobaseUser(_FakeMemobaseUser):
    def __init__(self):
        super().__init__()
        self._source_ids = {}
        self.flush_failures = 1

    def insert(self, blob):
        source_turn_id = blob.fields["source_turn_id"]
        if source_turn_id in self._source_ids:
            return self._source_ids[source_turn_id]
        blob_id = super().insert(blob)
        self._source_ids[source_turn_id] = blob_id
        return blob_id

    def flush(self, sync=False):
        if self.flush_failures:
            self.flush_failures -= 1
            raise RuntimeError("flush interrupted after insert")
        return super().flush(sync=sync)


def test_memobase_url_allows_only_loopback_or_compose_service():
    assert _is_local_memobase_url("http://127.0.0.1:8019")
    assert _is_local_memobase_url("http://memobase-server-api:8000")
    assert not _is_local_memobase_url("https://api.memobase.dev")
    assert not _is_local_memobase_url("http://memobase.example")


def test_emotion_classifier_returns_normalized_seven_dimensions():
    result = EmotionClassifier().classify("我最近很焦虑，但现在已经安心了")

    assert tuple(result) == EMOTION_LABELS
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert all(0.0 <= value <= 1.0 for value in result.values())
    assert result["anxiety"] > result["joy"]


def test_memory_survives_sessions_and_supports_controlled_updates(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))

    memory.capture_turn(
        "pt-1",
        "我叫张阿姨，最近喜欢钓鱼，孙子刚考上大学。",
        "记住了，我们可以慢慢聊。",
        session_id="session-a",
    )
    memory.capture_turn(
        "pt-1",
        "今天有点焦虑。",
        "先休息一下。",
        session_id="session-b",
    )
    memory.update_mmse_score("pt-1", 24, ["recall"])
    memory.add_comfort_strategy("pt-1", "聊孙子近况", True)

    context = memory.get_context_for_llm("pt-1")
    snapshot = memory.get_memory_for_user("pt-1")

    assert "孙子刚考上大学" in context
    assert "会话:session-a" in context
    assert "MMSE" not in context
    assert "recall" not in context
    assert snapshot["emotion"]["latest"]["dominant"] == "anxiety"
    assert snapshot["comfort_strategies"][0]["effective"] is True

    updated = memory.update_memory_by_user(
        "pt-1",
        {"narrative": "喜欢钓鱼，愿意聊孙子。", "facts": ["已由用户确认"]},
    )

    assert updated["narrative"] == "喜欢钓鱼，愿意聊孙子。"
    assert "已由用户确认" in memory.get_context_for_llm("pt-1")


def test_memory_items_list_update_delete_and_replay(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": [{"text": "喜欢散步"}],
            "preferences": [{"text": "睡前少看手机"}],
            "expected_revision": 0,
        },
    )

    items = memory.list_memory_items("pt-1")
    fact = next(item for item in items if item["category"] == "facts")
    updated = memory.update_memory_item(
        "pt-1",
        fact["item_id"],
        "喜欢清晨散步",
        fact["version"],
    )
    assert updated["content"] == "喜欢清晨散步"
    assert updated["version"] == fact["version"] + 1

    deleted = memory.delete_memory_item(
        "pt-1",
        updated["item_id"],
        updated["version"],
        "delete-token-001",
    )
    assert deleted["status"] == "deleted"
    assert deleted["content"] == ""
    assert deleted["idempotent_replay"] is False

    replay = memory.delete_memory_item(
        "pt-1",
        updated["item_id"],
        updated["version"],
        "delete-token-001",
    )
    assert replay["status"] == "deleted"
    assert replay["idempotent_replay"] is True

    active_ids = {item["item_id"] for item in memory.list_memory_items("pt-1")}
    assert updated["item_id"] not in active_ids
    deleted_items = memory.list_memory_items("pt-1", include_deleted=True)
    deleted_fact = next(item for item in deleted_items if item["item_id"] == updated["item_id"])
    assert deleted_fact["status"] == "deleted"
    old_fact = next(item for item in deleted_items if item["item_id"] == fact["item_id"])
    assert old_fact["status"] == "superseded"
    assert "喜欢清晨散步" not in memory.get_authoritative_card("pt-1")
    assert "喜欢清晨散步" not in memory.get_context_for_llm("pt-1")


def test_stale_edit_of_superseded_item_returns_current_revision_conflict(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    memory.update_memory_by_user("pt-1", {"facts": ["第一版"], "expected_revision": 0})
    original = memory.list_memory_items("pt-1")[0]
    current = memory.update_memory_item("pt-1", original["item_id"], "第二版", original["version"])

    with pytest.raises(MemoryRevisionConflict) as caught:
        memory.update_memory_item("pt-1", original["item_id"], "过期草稿", original["version"])

    assert caught.value.current_revision == current["version"]
    assert memory.get_memory_item("pt-1", original["item_id"])["item_id"] == current["item_id"]


def test_memory_item_delete_enqueues_memobase_cleanup(tmp_path: Path):
    user = _DeletionMemobaseUser()
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(user),
    )
    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": [{"text": "喜欢清晨散步"}],
            "expected_revision": 0,
        },
    )
    item = next(
        item for item in memory.list_memory_items("pt-1")
        if item["category"] == "facts"
    )

    deleted = memory.delete_memory_item(
        "pt-1",
        item["item_id"],
        item["version"],
        "delete-token-004",
    )
    executor = memory._memobase_executor
    if executor is not None:
        executor.shutdown(wait=True)
        memory._memobase_executor = None

    assert deleted["status"] == "deleted"
    assert deleted["memobase_delete_enqueued"] is True
    assert user.deleted_events == ["event-delete"]
    assert user.deleted_profiles == ["profile-delete"]


def test_empty_patient_context_is_bounded_and_capture_is_local(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))

    assert memory.get_context_for_llm("new-patient") == ""
    result = memory.capture_turn("new-patient", "你好", "你好。", session_id="s1")

    assert result["patient_id"] == "new-patient"
    assert set(result["emotion"]) == set(EMOTION_LABELS)
    assert len(memory.get_context_for_llm("new-patient")) <= 7000


def test_current_utterance_drives_bounded_memobase_context(tmp_path: Path):
    user = _FakeMemobaseUser()
    client = _FakeMemobaseClient(user)
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=client,
    )
    memory._enqueue_memobase = lambda _patient_id: None
    memory.capture_turn(
        "pt-1", "之前在浴室滑倒", "收到", session_id="old-session", turn_id="turn-old"
    )

    context = memory.get_relevant_context("pt-1", "我又不敢一个人洗澡了")

    assert "浴室滑倒" in context
    assert "gist-1" not in context
    assert user.context_calls == []
    assert user.search_event_gist_calls == [
        {
            "query": "我又不敢一个人洗澡了",
            "topk": 3,
            "similarity_threshold": 0.35,
        }
    ]
    assert client.patient_id != "pt-1"
    assert str(UUID(client.patient_id)) == client.patient_id


def test_memobase_user_creation_falls_back_when_sdk_rejects_missing_user():
    user = _FakeMemobaseUser()

    class Client:
        def get_or_create_user(self, _user_id):
            raise RuntimeError("not found")

        def add_user(self, data, id):
            self.data = data
            self.user_id = id

        def get_user(self, user_id):
            assert user_id == self.user_id
            return user

    client = Client()
    assert EmotionMemobase._get_or_create_memobase_user(client, "pt-1") is user
    assert client.data == {"patient_id": "pt-1"}
    assert str(UUID(client.user_id)) == client.user_id


def test_failed_sync_keeps_sqlite_and_replays_turns_in_order(tmp_path: Path):
    user = _FakeMemobaseUser(failures=1)
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(user),
        logger=lambda _message: None,
    )

    first = memory.capture_turn("pt-1", "第一句", "第一答", session_id="s1")
    second = memory.capture_turn("pt-1", "第二句", "第二答", session_id="s1")

    assert first["local_turn_id"] < second["local_turn_id"]
    assert len(memory.get_snapshot("pt-1")["recent_turns"]) == 2
    assert not memory.flush("pt-1")
    assert memory.flush("pt-1")
    assert [blob.fields["source_turn_id"] for blob in user.inserted] == [
        first["local_turn_id"],
        second["local_turn_id"],
    ]
    assert all("audio_path" not in blob.fields for blob in user.inserted)
    assert user.flush_calls == [True]


def test_permanent_memobase_failure_never_rolls_back_sqlite(tmp_path: Path):
    user = _FakeMemobaseUser(failures=100)
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(user),
        logger=lambda _message: None,
    )

    memory.capture_turn("pt-1", "仍要保存", "已收到", session_id="s1")

    assert not memory.replay_unsynced("pt-1")
    assert memory.get_snapshot("pt-1")["recent_turns"][0]["user_message"] == "仍要保存"


def test_sync_replay_needs_remote_source_turn_id_deduplication(tmp_path: Path):
    user = _IdempotentMemobaseUser()
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(user),
        logger=lambda _message: None,
    )
    memory._enqueue_memobase = lambda _patient_id: None
    memory.capture_turn("pt-1", "重放测试", "收到", session_id="s1")

    assert not memory.flush("pt-1")
    assert len(user.inserted) == 1
    assert memory.flush("pt-1")
    assert len(user.inserted) == 1
    assert memory.get_sync_state("pt-1")["last_synced_turn_id"] == 1


def test_turn_id_is_idempotent_and_final_emotion_updates_same_trajectory(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)

    first = memory.capture_turn(
        "pt-1", "我最近有点焦虑", "我听到了", session_id="s1", turn_id="turn-1",
        emotion_source="text_asr", analysis_status="provisional",
    )
    replay = memory.capture_turn(
        "pt-1", "重复投递的文本", "重复回复", session_id="s1", turn_id="turn-1"
    )

    assert replay["idempotent_replay"] is True
    assert replay["local_turn_id"] == first["local_turn_id"]
    assert len(memory.get_snapshot("pt-1")["recent_turns"]) == 1
    assert len(memory.get_emotion_trajectory("pt-1")) == 1

    updated = memory.update_emotion_analysis(
        "pt-1", "turn-1", {"sadness": 1.0}, session_id="s1",
        emotion_source="emotion2vec_audio+text", analysis_status="final",
    )
    trajectory = memory.get_emotion_trajectory("pt-1")
    assert updated["dominant"] == "sadness"
    assert len(trajectory) == 1
    assert trajectory[0]["turn_id"] == "turn-1"
    assert trajectory[0]["dominant_emotion"] == "sadness"
    assert trajectory[0]["source"] == "emotion2vec_audio+text"
    assert trajectory[0]["analysis_status"] == "final"


def test_snapshot_revision_conflict_and_incremental_consolidation_waterline(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "我叫小明", "记住了", session_id="s1", turn_id="turn-1")
    snapshot = memory.get_snapshot("pt-1")

    try:
        memory.update_memory_by_user(
            "pt-1", {"facts": ["错误版本"], "expected_revision": snapshot["revision"] - 1}
        )
    except MemoryRevisionConflict as exc:
        assert exc.current_revision == snapshot["revision"]
    else:
        raise AssertionError("stale revision must fail")

    result = memory.consolidate_pending_turns("pt-1", patch={"facts": ["已确认"]})
    assert result["turn_count"] == 1
    assert result["last_consolidated_turn_id"] == 1
    assert memory.get_snapshot("pt-1")["last_consolidated_turn_id"] == 1
    empty = memory.consolidate_pending_turns("pt-1", cutoff_turn_id=1)
    assert empty["turn_count"] == 0


def test_session_consolidation_uses_its_own_cutoff(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "旧会话", "收到", session_id="old", turn_id="old-1")
    memory.capture_turn("pt-1", "新会话", "收到", session_id="new", turn_id="new-1")

    old = memory.consolidate_pending_turns("pt-1", session_id="old")
    assert old["turn_count"] == 1
    assert old["last_consolidated_turn_id"] == 1

    new = memory.consolidate_pending_turns("pt-1", session_id="new")
    assert new["turn_count"] == 1
    assert new["last_consolidated_turn_id"] == 2


def test_out_of_order_session_consolidation_does_not_advance_past_pending_turn(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "旧会话事实", "收到", session_id="old", turn_id="old-1")
    memory.capture_turn("pt-1", "新会话事实", "收到", session_id="new", turn_id="new-1")

    new_first = memory.consolidate_pending_turns("pt-1", session_id="new")

    assert new_first["turn_count"] == 1
    assert new_first["last_consolidated_turn_id"] == 0
    assert memory.get_snapshot("pt-1")["last_consolidated_turn_id"] == 0

    old_second = memory.consolidate_pending_turns("pt-1", session_id="old")

    assert old_second["last_consolidated_turn_id"] == 1
    with memory._lock, memory._connection() as conn:
        statuses = conn.execute(
            "SELECT consolidation_status FROM emotion_memobase_turns "
            "WHERE patient_id=? ORDER BY id",
            ("pt-1",),
        ).fetchall()
    assert [row[0] for row in statuses] == ["consolidated", "consolidated"]


def test_consolidation_merges_contained_event_memories_and_turn_evidence(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn(
        "pt-1", "我最近心情不好，然后有点不开心。", "我听到了。",
        session_id="s1", turn_id="turn-1",
    )
    memory.capture_turn(
        "pt-1", "我最近心情不好。", "我在。",
        session_id="s1", turn_id="turn-2",
    )

    memory.consolidate_pending_turns("pt-1", session_id="s1")

    events = memory.get_snapshot("pt-1")["events"]
    assert len(events) == 1
    assert events[0]["text"] == "我最近心情不好，然后有点不开心。"
    assert set(events[0]["evidence_turn_ids"]) == {"turn-1", "turn-2"}
    assert len([item for item in memory.list_memory_items("pt-1") if item["category"] == "events"]) == 1


def test_authoritative_card_excludes_current_session_raw_turns(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _: None)
    memory.capture_turn(
        "pt-1",
        "这句原始对话不能出现在权威卡中",
        "收到。",
        session_id="current",
        turn_id="turn-current",
    )
    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": ["已确认的长期事实"],
            "expected_revision": memory.get_snapshot("pt-1")["revision"],
        },
    )

    card = memory.get_authoritative_card("pt-1")

    assert "已确认的长期事实" in card
    assert "这句原始对话不能出现在权威卡中" not in card


def test_cross_session_context_excludes_working_session(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _: None)
    memory.capture_turn(
        "pt-1", "旧会话证据", "旧回复", session_id="old", turn_id="turn-old"
    )
    memory.capture_turn(
        "pt-1", "当前会话证据", "当前回复", session_id="current", turn_id="turn-current"
    )

    context = memory.get_cross_session_context("pt-1", "current")

    assert "旧会话证据" in context
    assert "当前会话证据" not in context


def test_manual_revision_suppresses_replaced_raw_evidence(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _: None)
    memory.capture_turn(
        "pt-1",
        "已过时的历史说法",
        "收到。",
        session_id="old",
        turn_id="turn-old",
    )
    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": [
                {
                    "text": "医护确认的新事实",
                    "evidence_turn_ids": ["turn-old"],
                }
            ],
            "expected_revision": memory.get_snapshot("pt-1")["revision"],
        },
    )

    context = memory.get_cross_session_context("pt-1")

    assert "已过时的历史说法" not in context
    assert "医护确认的新事实" in memory.get_authoritative_card("pt-1")


def test_sync_state_records_recoverable_failure_fields(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "本地先保存", "收到", session_id="s1", turn_id="turn-1")
    state = memory.get_sync_state("pt-1")
    assert set(("last_synced_turn_id", "attempt_count", "next_retry_at", "last_error", "lease_until")) <= set(state)


def test_consolidation_does_not_skip_older_session_after_newer_session_finishes(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "旧会话里我叫小明", "收到", session_id="old", turn_id="old-1")
    memory.capture_turn("pt-1", "新会话里我喜欢散步", "收到", session_id="new", turn_id="new-1")

    newer = memory.consolidate_pending_turns("pt-1", session_id="new")
    older = memory.consolidate_pending_turns("pt-1", session_id="old")

    assert newer["turn_count"] == 1
    assert older["turn_count"] == 1


def test_snapshot_replacement_does_not_leave_old_item_active(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    first = memory.update_memory_by_user(
        "pt-1", {"facts": ["患者今年80岁"], "expected_revision": 0}
    )
    memory.update_memory_by_user(
        "pt-1",
        {"facts": ["患者今年81岁"], "expected_revision": first["revision"]},
    )

    active = memory.list_memory_items("pt-1")
    context = memory.get_context_for_llm("pt-1")
    assert [item["content"] for item in active] == ["患者今年81岁"]
    assert "患者今年80岁" not in context


def test_memory_item_edit_preserves_old_evidence_as_superseded(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.update_memory_by_user(
        "pt-1", {"facts": ["患者今年80岁"], "expected_revision": 0}
    )
    original = memory.list_memory_items("pt-1")[0]

    updated = memory.update_memory_item(
        "pt-1", original["item_id"], "患者今年81岁", original["version"]
    )

    assert updated["item_id"] != original["item_id"]
    assert updated["status"] == "active"
    history = memory.list_memory_items("pt-1", include_deleted=True)
    old = next(item for item in history if item["item_id"] == original["item_id"])
    assert old["status"] == "superseded"
    assert old["content"] == "患者今年80岁"


def test_ineffective_comfort_strategy_is_not_rendered_as_effective(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.add_comfort_strategy("pt-1", "连续追问细节", False)

    card = memory.get_authoritative_card("pt-1")

    assert "【有效安抚策略】连续追问细节" not in card


def test_missing_emotion_evidence_is_unavailable_instead_of_calm(tmp_path: Path):
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        emotion_classifier=lambda _text: {},
        logger=lambda _message: None,
    )

    result = memory.capture_turn("pt-1", "这是一句没有情绪证据的陈述", "收到", session_id="s1")

    assert result["emotion"] == {}
    assert result["dominant_emotion"] == "unknown"
    assert result["analysis_status"] == "unavailable"
    assert "calm" not in memory.get_context_for_llm("pt-1")


def test_memobase_hit_without_local_patient_provenance_is_not_used(tmp_path: Path):
    class ForeignHitUser(_FakeMemobaseUser):
        def search_event_gist(self, query, **kwargs):
            return [
                {
                    "id": "foreign-gist",
                    "event_id": "foreign-event",
                    "session_id": "foreign-session",
                    "turn_id": "foreign-turn",
                    "source_turn_id": "foreign-turn",
                    "patient_id": "other-patient",
                    "gist_data": {"content": "另一位患者的家庭冲突"},
                    "similarity": 0.99,
                }
            ]

    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(ForeignHitUser()),
        logger=lambda _message: None,
    )

    assert memory.get_relevant_evidence("pt-1", "家庭冲突") == ""


def test_mmse_history_is_not_written_as_a_memory_item(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)

    result = memory.update_snapshot(
        "pt-1",
        {"mmse_history": [{"score": 24, "recorded_at": "2026-08-16"}]},
    )

    assert result["mmse_history"] == [{"score": 24, "recorded_at": "2026-08-16"}]
    assert all(item["category"] != "mmse_history" for item in memory.list_memory_items("pt-1", include_deleted=True))


def test_cancelled_turn_is_skipped_and_failed_extraction_stays_pending(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "取消的回合", "", session_id="s1", turn_id="cancelled")
    memory.update_turn_status(
        "pt-1", "cancelled", session_id="s1", turn_state="FAILED"
    )

    skipped = memory.consolidate_pending_turns("pt-1", session_id="s1")

    assert skipped["turn_count"] == 0
    with memory._lock, memory._connection() as conn:
        status = conn.execute(
            "SELECT consolidation_status FROM emotion_memobase_turns WHERE patient_id=? AND turn_id=?",
            ("pt-1", "cancelled"),
        ).fetchone()[0]
    assert status == "skipped"

    memory.capture_turn("pt-1", "稍后重试", "收到", session_id="s1", turn_id="pending")
    try:
        memory.consolidate_pending_turns(
            "pt-1",
            session_id="s1",
            patch_builder=lambda _turns: (_ for _ in ()).throw(RuntimeError("extract failed")),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed consolidation must propagate and keep the turn pending")
    with memory._lock, memory._connection() as conn:
        status = conn.execute(
            "SELECT consolidation_status FROM emotion_memobase_turns WHERE patient_id=? AND turn_id=?",
            ("pt-1", "pending"),
        ).fetchone()[0]
    assert status == "pending"


def test_replay_skips_turn_whose_source_memory_was_invalidated(tmp_path: Path):
    user = _FakeMemobaseUser()
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(user),
        logger=lambda _message: None,
    )
    memory._enqueue_memobase = lambda _patient_id: None
    memory._schedule_mirror_ops = lambda _patient_id: False
    memory.capture_turn(
        "pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-source"
    )
    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": [{"text": "我叫小明", "evidence_turn_ids": ["turn-source"]}],
            "expected_revision": memory.get_snapshot("pt-1")["revision"],
        },
    )
    item = memory.list_memory_items("pt-1")[0]
    memory.delete_memory_item("pt-1", item["item_id"], item["version"], "delete-source")

    assert memory._replay_unsynced_impl("pt-1", True)
    assert user.inserted == []
    assert memory.get_sync_state("pt-1")["last_synced_turn_id"] == 1


def test_consolidation_is_idempotent_when_called_concurrently(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: memory.consolidate_pending_turns("pt-1", session_id="s1"),
                range(2),
            )
        )

    assert sorted(result["turn_count"] for result in results) == [0, 1]
    assert [item["content"] for item in memory.list_memory_items("pt-1")] == ["我叫小明"]
    with memory._lock, memory._connection() as conn:
        status = conn.execute(
            "SELECT consolidation_status FROM emotion_memobase_turns WHERE patient_id=? AND turn_id=?",
            ("pt-1", "turn-1"),
        ).fetchone()[0]
    assert status == "consolidated"


def test_legacy_snapshot_migration_is_idempotent_and_does_not_revive_tombstone(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    memory = EmotionMemobase(str(db_path), logger=lambda _message: None)
    memory.update_memory_by_user("pt-1", {"facts": ["已停止使用的旧事实"]})
    item = memory.list_memory_items("pt-1")[0]
    memory.delete_memory_item("pt-1", item["item_id"], item["version"], "stop-old-fact")
    with memory._lock, memory._connection() as conn:
        snapshot = memory._load_snapshot(conn, "pt-1")
        snapshot["facts"] = [{"item_id": item["item_id"], "text": "已停止使用的旧事实"}]
        memory._save_snapshot(conn, snapshot)
    memory.close()

    restarted = EmotionMemobase(str(db_path), logger=lambda _message: None)

    assert restarted.list_memory_items("pt-1") == []
    history = restarted.list_memory_items("pt-1", include_deleted=True)
    assert history[0]["status"] == "deleted"
    restarted.close()


def test_expired_items_are_excluded_from_prompt_card_and_local_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_FALLBACK_ENABLED", "true")
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.update_snapshot(
        "pt-1",
        {
            "facts": [
                {"text": "已过期的旧事实", "valid_until": "2000-01-01"},
                {"text": "仍然有效的事实", "valid_until": "2100-01-01"},
            ]
        },
    )

    card = memory.get_authoritative_card("pt-1")
    context = memory.get_context_for_llm("pt-1")
    fallback = memory.get_relevant_evidence("pt-1", "请回忆一下")

    assert "仍然有效的事实" in card
    assert "仍然有效的事实" in context
    assert "仍然有效的事实" in fallback
    assert "已过期的旧事实" not in card
    assert "已过期的旧事实" not in context
    assert "已过期的旧事实" not in fallback
    assert "来源：SQLite active memory_items" in fallback


def test_authoritative_card_is_bounded_by_its_token_budget(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.update_memory_by_user(
        "pt-1",
        {"facts": [f"事实{index}-" + "x" * 800 for index in range(10)]},
    )

    assert _token_estimate(memory.get_authoritative_card("pt-1")) <= 900


def test_remote_gist_requires_active_local_source_turn(tmp_path: Path):
    class SourceHitUser(_FakeMemobaseUser):
        def search_event_gist(self, query, **kwargs):
            return [
                {
                    "id": "source-gist",
                    "event_id": "source-event",
                    "session_id": "old",
                    "turn_id": "turn-source",
                    "source_turn_id": "turn-source",
                    "gist_data": {"content": "患者曾在浴室滑倒"},
                    "similarity": 0.9,
                }
            ]

    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(SourceHitUser()),
        logger=lambda _message: None,
    )
    memory._enqueue_memobase = lambda _patient_id: None
    memory._schedule_mirror_ops = lambda _patient_id: False
    memory.capture_turn(
        "pt-1", "我曾在浴室滑倒", "收到", session_id="old", turn_id="turn-source"
    )

    assert "浴室滑倒" in memory.get_relevant_evidence("pt-1", "我害怕洗澡")

    memory.update_memory_by_user(
        "pt-1",
        {
            "facts": [{"text": "停止使用这条旧经历", "evidence_turn_ids": ["turn-source"]}],
            "expected_revision": memory.get_snapshot("pt-1")["revision"],
        },
    )
    item = memory.list_memory_items("pt-1")[0]
    memory.delete_memory_item("pt-1", item["item_id"], item["version"], "stop-source")

    assert memory.get_relevant_evidence("pt-1", "我害怕洗澡") == ""


def test_context_includes_emotion_trend_not_just_latest_label(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn(
        "pt-1", "第一轮", "收到", session_id="s1", turn_id="turn-1", emotion={"sadness": 1.0}
    )
    memory.capture_turn(
        "pt-1", "第二轮", "收到", session_id="s1", turn_id="turn-2", emotion={"calm": 1.0}
    )

    assert "趋势：好转" in memory.get_context_for_llm("pt-1")


def test_remote_error_fallback_is_limited_to_three_active_items(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_FALLBACK_ENABLED", "true")
    class BrokenSearchUser(_FakeMemobaseUser):
        def search_event_gist(self, query, **kwargs):
            raise RuntimeError("remote unavailable")

    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=_FakeMemobaseClient(BrokenSearchUser()),
        logger=lambda _message: None,
    )
    memory._enqueue_memobase = lambda _patient_id: None
    memory.capture_turn("pt-1", "不能作为降级结果的原始对话", "收到", session_id="old")
    memory.update_memory_by_user(
        "pt-1", {"facts": [f"本地事实{index}" for index in range(4)]}
    )

    fallback = memory.get_relevant_evidence("pt-1", "请给我相关记忆")

    assert fallback.count("来源：SQLite active memory_items") == 3
    assert "不能作为降级结果的原始对话" not in fallback


def test_local_fallback_is_disabled_until_p5_gate_opens(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.update_memory_by_user("pt-1", {"facts": ["待启用的本地记忆"]})

    assert memory.get_relevant_evidence("pt-1", "请回忆") == ""
