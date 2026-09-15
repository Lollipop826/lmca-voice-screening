from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import httpx
import pytest

from src.context_management.emotion_memobase import EmotionMemobase
from src.context_management.retrieval_cache import (
    RetrievalBusyError,
    RetrievalCache,
    RetrievalHttpClient,
)


def test_cache_is_bounded_expires_and_returns_independent_values():
    now = [0.0]
    cache = RetrievalCache(ttl_s=10, max_entries=2, clock=lambda: now[0])
    first, source = cache.get("first", lambda: [{"value": "original"}])
    assert source == "miss"
    first[0]["value"] = "changed"
    assert cache.get("first", lambda: None) == ([{"value": "original"}], "hit")
    cache.get("second", lambda: ["second"])
    cache.get("first", lambda: None)
    cache.get("third", lambda: ["third"])
    assert cache.get("second", lambda: ["reloaded"])[1] == "miss"
    now[0] = 10.0
    assert cache.get("third", lambda: ["expired"])[1] == "miss"


def test_empty_results_and_failures_are_not_cached():
    cache = RetrievalCache()
    assert cache.get("empty", list) == ([], "miss")
    assert cache.get("empty", lambda: ["new memory"])[1] == "miss"

    def fail():
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        cache.get("failed", fail)
    assert cache.get("failed", lambda: ["recovered"])[1] == "miss"


def test_cache_can_be_disabled_without_disabling_query_execution():
    cache = RetrievalCache(ttl_s=0)
    assert cache.get("query", lambda: ["first"])[1] == "miss"
    assert cache.get("query", lambda: ["second"]) == (["second"], "miss")


def test_concurrent_identical_queries_share_one_load():
    started = threading.Event()
    joined = threading.Event()
    release = threading.Event()
    clock_calls = []
    loads = []

    def clock():
        clock_calls.append(True)
        if len(clock_calls) >= 2:
            joined.set()
        return 0.0

    def load():
        loads.append(True)
        started.set()
        assert release.wait(2)
        return [{"value": "shared"}]

    cache = RetrievalCache(clock=clock, wait_timeout_s=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(cache.get, "query", load)
        assert started.wait(1)
        follower = executor.submit(cache.get, "query", load)
        try:
            assert joined.wait(1)
        finally:
            release.set()
        primary_value, primary_source = owner.result(timeout=1)
        shared_value, shared_source = follower.result(timeout=1)
    assert primary_source == "miss"
    assert shared_source == "shared"
    assert len(loads) == 1
    shared_value[0]["value"] = "changed"
    assert primary_value == [{"value": "shared"}]
    assert cache.get("query", list)[0] == [{"value": "shared"}]


def test_waiter_timeout_does_not_release_the_inflight_slot():
    started = threading.Event()
    release = threading.Event()
    cache = RetrievalCache(max_inflight=1, wait_timeout_s=0.01)

    def load():
        started.set()
        assert release.wait(2)
        return ["late result"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(cache.get, "query", load)
        assert started.wait(1)
        try:
            with pytest.raises(FutureTimeoutError):
                cache.get("query", load)
            with pytest.raises(RetrievalBusyError):
                cache.get("other query", list)
        finally:
            release.set()
        assert owner.result(timeout=1)[0] == ["late result"]
    assert cache.get("query", list) == (["late result"], "hit")


def test_http_budget_reuses_the_pool_without_changing_write_timeout():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(
        base_url="http://127.0.0.1/api/v1",
        timeout=30,
        transport=httpx.MockTransport(handle),
    ) as client:
        bounded = RetrievalHttpClient(client, timeout_s=1.5)
        bounded.get("/users/patient")
        bounded.post("/users", json={"id": "patient"})
        assert client.timeout.read == 30
        for request in requests:
            assert 0 < request.extensions["timeout"]["read"] <= 1.5
            assert request.extensions["timeout"]["connect"] <= 0.25
            assert request.extensions["timeout"]["pool"] <= 0.1
        assert requests[1].extensions["timeout"]["read"] <= requests[0].extensions["timeout"]["read"]
        bounded._deadline = 0
        with pytest.raises(TimeoutError, match="budget exhausted"):
            bounded.get("/users/event_gist/search/patient")
        assert len(requests) == 2


class _User:
    def __init__(self):
        self.queries = []

    def search_event_gist(self, query, **params):
        self.queries.append((query, params))
        return [{
            "id": "gist-source",
            "event_id": "event-source",
            "source_turn_id": "turn-source",
            "turn_id": "turn-source",
            "session_id": "old-session",
            "gist_data": {"content": "患者曾在浴室滑倒"},
            "similarity": 0.9,
        }]


class _Client:
    def __init__(self, user):
        self.user = user
        self.user_lookups = []

    def get_or_create_user(self, user_id):
        self.user_lookups.append(user_id)
        return self.user


def _memory(tmp_path, user=None):
    user = user or _User()
    messages = []
    client = _Client(user)
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        memobase_client=client,
        emotion_classifier=lambda text: {"calm": 1.0},
        logger=messages.append,
    )
    memory._enqueue_memobase = lambda patient_id: None
    memory._schedule_mirror_ops = lambda patient_id: False
    memory.capture_turn(
        "patient-a", "我曾在浴室滑倒", "收到",
        session_id="old-session", turn_id="turn-source",
    )
    return memory, client, messages


def test_cached_retrieval_avoids_both_user_lookup_and_embedding(tmp_path):
    memory, client, messages = _memory(tmp_path)
    first = memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    second = memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    assert "浴室滑倒" in first
    assert first == second
    assert len(client.user.queries) == len(client.user_lookups) == 1
    assert any("cache=hit" in message for message in messages)


def test_sdk_retrieval_uses_short_http_budget_and_keeps_original_client(tmp_path):
    from memobase import MemoBaseClient

    requests = []
    gists = _User().search_event_gist("fixture")

    def handle(request):
        requests.append(request)
        if "/event_gist/search/" in request.url.path:
            data = {"gists": gists}
        else:
            data = {"id": request.url.path.rsplit("/", 1)[-1], "data": {}}
        return httpx.Response(200, json={"data": data, "errno": 0, "errmsg": ""})

    client = MemoBaseClient(project_url="http://127.0.0.1:8019", api_key="test-key")
    client.client.close()
    with httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handle),
        timeout=30,
    ) as original_http:
        client._client = original_http
        memory, fixture_client, messages = _memory(tmp_path)
        memory._memobase = client
        first = memory.get_relevant_evidence("patient-a", "我害怕洗澡")
        assert "浴室滑倒" in first
        assert memory.get_relevant_evidence("patient-a", "我害怕洗澡") == first
        assert len(requests) == 2
        assert client.client is original_http
        assert original_http.timeout.read == 30
        assert all(0 < request.extensions["timeout"]["read"] <= 1.5 for request in requests)


def test_cache_keys_isolate_patients_queries_and_search_parameters(tmp_path):
    memory, client, messages = _memory(tmp_path)
    memory.capture_turn(
        "patient-b", "我也曾滑倒", "收到",
        session_id="old-session", turn_id="turn-source",
    )
    memory.get_relevant_event_gists("patient-a", "我害怕洗澡", topk=3)
    memory.get_relevant_event_gists("patient-b", "我害怕洗澡", topk=3)
    memory.get_relevant_event_gists("patient-a", "我不害怕洗澡", topk=3)
    memory.get_relevant_event_gists("patient-a", "我害怕洗澡", topk=1)
    assert len(client.user.queries) == 4
    assert client.user_lookups[0] != client.user_lookups[1]


def test_memory_revision_changes_invalidate_cached_retrieval(tmp_path):
    memory, client, messages = _memory(tmp_path)
    memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    memory.update_memory_by_user(
        "patient-a",
        {"facts": ["家人可以陪同"], "expected_revision": memory.get_snapshot("patient-a")["revision"]},
    )
    memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    assert len(client.user.queries) == 2


def test_cache_hit_still_checks_live_local_provenance(tmp_path):
    memory, client, messages = _memory(tmp_path)
    assert memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    with memory._lock, memory._connection() as connection:
        connection.execute(
            "UPDATE emotion_memobase_turns SET turn_state='CANCELLED' WHERE patient_id=?",
            ("patient-a",),
        )
    assert memory.get_relevant_evidence("patient-a", "我害怕洗澡") == ""
    assert len(client.user.queries) == 1
    assert any("cache=hit" in message for message in messages)


def test_deleting_memory_prevents_old_cached_evidence_from_returning(tmp_path):
    memory, client, messages = _memory(tmp_path)
    memory.update_memory_by_user(
        "patient-a",
        {"facts": [{"text": "曾经滑倒", "evidence_turn_ids": ["turn-source"]}],
         "expected_revision": memory.get_snapshot("patient-a")["revision"]},
    )
    assert memory.get_relevant_evidence("patient-a", "我害怕洗澡")
    item = memory.list_memory_items("patient-a")[0]
    memory.delete_memory_item("patient-a", item["item_id"], item["version"], "delete-request")
    assert memory.get_relevant_evidence("patient-a", "我害怕洗澡") == ""
    assert len(client.user.queries) == 2


def test_inflight_result_is_checked_against_latest_local_state(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingUser(_User):
        def search_event_gist(self, query, **params):
            started.set()
            assert release.wait(2)
            return super().search_event_gist(query, **params)

    memory, client, messages = _memory(tmp_path, BlockingUser())
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(memory.get_relevant_evidence, "patient-a", "我害怕洗澡")
        assert started.wait(1)
        try:
            with memory._lock, memory._connection() as connection:
                connection.execute(
                    "UPDATE emotion_memobase_turns SET turn_state='CANCELLED' WHERE patient_id=?",
                    ("patient-a",),
                )
        finally:
            release.set()
        assert response.result(timeout=1) == ""
    assert memory.get_relevant_evidence("patient-a", "我害怕洗澡") == ""
    assert len(client.user.queries) == 1


def test_prewarming_only_initializes_connection_and_user(tmp_path):
    memory, client, messages = _memory(tmp_path)
    assert memory._prewarm_semantic_retrieval_impl("patient-a") is True
    assert len(client.user_lookups) == 1
    assert client.user.queries == []


def test_disabled_memobase_does_not_open_a_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMOBASE_PROJECT_URL", "")
    monkeypatch.setenv("MEMOBASE_API_KEY", "")
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda message: None)

    def unexpected_connection():
        raise AssertionError("disabled Memobase must stay disabled")

    monkeypatch.setattr(memory, "_ensure_memobase", unexpected_connection)
    assert memory.get_relevant_event_gists("patient-a", "查询", _include_degraded=True) == ([], True)
    assert memory.prewarm_semantic_retrieval("patient-a") is False
