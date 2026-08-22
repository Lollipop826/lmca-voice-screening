from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.context_management.emotion_memobase import EmotionMemobase
from scripts.validate_memory_reflection_qwen import (
    candidate_matches,
    evidence_is_bound,
    evaluate_case,
)


def _candidate(content: str, quote: str, turn_id: str) -> dict:
    return {
        "category": "facts",
        "content": content,
        "source_turn_id": turn_id,
        "evidence_turn_ids": [turn_id],
        "evidence_quote": quote,
        "observed_at": "2026-08-16",
        "valid_until": None,
        "sensitivity": "normal",
        "confidence": 0.95,
        "slot_key": "fact:test",
        "suggested_operation": "ADD",
    }


def test_session_reflection_stores_summary_and_shadow_candidate(tmp_path: Path):
    payload = {
        "session_summary": {
            "content": "患者明确说自己叫小明。",
            "evidence_turn_ids": ["turn-1"],
        },
        "memory_candidates": [_candidate("患者叫小明", "我叫小明", "turn-1")],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "认识你很高兴", session_id="s1", turn_id="turn-1")

    result = memory.reflect_session("pt-1", "s1")
    reflection = memory.get_session_reflection("pt-1", "s1")
    items = memory.list_memory_items("pt-1", include_deleted=True)

    assert result["status"] == "succeeded"
    assert result["candidate_count"] == 1
    assert reflection["session_summary"] == "患者明确说自己叫小明。"
    assert items[0]["status"] == "candidate"
    assert items[0]["source_turn_id"] == "turn-1"
    assert "患者叫小明" not in memory.get_context_for_llm("pt-1")


def test_latest_session_reflection_returns_previous_completed_summary(tmp_path: Path):
    payload = {
        "session_summary": {
            "content": "上一轮会话摘要。",
            "evidence_turn_ids": ["turn-1"],
        },
        "memory_candidates": [],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "昨天睡得还可以", "收到", session_id="s1", turn_id="turn-1")
    memory.reflect_session("pt-1", "s1")

    previous = memory.get_latest_session_reflection(
        "pt-1",
        exclude_session_id="s2",
    )

    assert previous["session_id"] == "s1"
    assert previous["session_summary"] == "上一轮会话摘要。"


def test_session_reflection_rejects_assistant_only_evidence(tmp_path: Path):
    payload = {
        "session_summary": {"content": "无", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [
            _candidate("患者患有抑郁症", "我担心你有抑郁症", "turn-1")
        ],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1", "我今天还好", "我担心你有抑郁症", session_id="s1", turn_id="turn-1"
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "succeeded"
    assert result["rejected_count"] == 1
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_rejects_denial_even_when_model_suggests_supersede(tmp_path: Path):
    payload = {
        "session_summary": {"content": "", "evidence_turn_ids": []},
        "memory_candidates": [
            {
                **_candidate("用户没有和女儿吵架", "我没有和女儿吵架", "turn-1"),
                "suggested_operation": "SUPERSEDE",
            }
        ],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1",
        "我没有和女儿吵架",
        "收到",
        session_id="s1",
        turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "succeeded"
    assert result["rejected_count"] == 1
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_does_not_treat_mei_xiang_as_symptom_denial(tmp_path: Path):
    payload = {
        "session_summary": {"content": "", "evidence_turn_ids": []},
        "memory_candidates": [
            _candidate("最近睡不着", "没想到自己最近睡不着", "turn-1")
        ],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1", "没想到自己最近睡不着", "收到",
        session_id="s1", turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["rejected_count"] == 0
    assert memory.list_memory_items("pt-1", include_deleted=True)[0]["content"] == "最近睡不着"


def test_session_reflection_rejects_short_weather_context_without_long_term_signal(tmp_path: Path):
    payload = {
        "session_summary": {"content": "", "evidence_turn_ids": []},
        "memory_candidates": [_candidate("天气不错", "天气不错", "turn-1")],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "天气不错", "收到", session_id="s1", turn_id="turn-1")

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "succeeded"
    assert result["rejected_count"] == 1
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_rejects_cross_session_evidence(tmp_path: Path):
    payload = {
        "session_summary": {"content": "无", "evidence_turn_ids": ["turn-other"]},
        "memory_candidates": [_candidate("患者叫小红", "我叫小红", "turn-other")],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "本会话内容", "收到", session_id="s1", turn_id="turn-1")
    memory.capture_turn("pt-1", "我叫小红", "收到", session_id="s2", turn_id="turn-other")

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "succeeded"
    assert result["rejected_count"] == 1
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_preserves_future_time_scope_from_user_turn(tmp_path: Path):
    payload = {
        "session_summary": {"content": "", "evidence_turn_ids": []},
        "memory_candidates": [
            {
                **_candidate("我这两天有点烦", "我这两天有点烦，过几天再说", "turn-1"),
                "category": "events",
                "suggested_operation": "CANDIDATE",
            }
        ],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1",
        "我这两天有点烦，过几天再说",
        "好的",
        session_id="s1",
        turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "succeeded"
    item = memory.list_memory_items("pt-1", include_deleted=True)[0]
    assert item["content"] == "我这两天有点烦，过几天再说"


def test_session_reflection_deduplicates_overlapping_same_turn_candidates(tmp_path: Path):
    payload = {
        "session_summary": {"content": "", "evidence_turn_ids": []},
        "memory_candidates": [
            _candidate("我最近不喜欢下雨天", "我最近不喜欢下雨天", "turn-1"),
            _candidate("我最近不喜欢下雨天，出门会烦", "我最近不喜欢下雨天，出门会烦", "turn-1"),
        ],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1",
        "我最近不喜欢下雨天，出门会烦",
        "收到",
        session_id="s1",
        turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["candidate_count"] == 1
    assert len(memory.list_memory_items("pt-1", include_deleted=True)) == 1
    assert memory.list_memory_items("pt-1", include_deleted=True)[0]["content"] == (
        "我最近不喜欢下雨天，出门会烦"
    )


def test_session_reflection_retries_without_duplicate_candidates(tmp_path: Path):
    calls = []
    payload = {
        "session_summary": {"content": "患者叫小明。", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [_candidate("患者叫小明", "我叫小明", "turn-1")],
    }

    def reflector(_prompt):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        return payload

    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=reflector,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    failed = memory.reflect_session("pt-1", "s1")
    succeeded = memory.reflect_session("pt-1", "s1", force=True)
    replay = memory.reflect_session("pt-1", "s1")

    assert failed["status"] == "failed"
    assert succeeded["status"] == "succeeded"
    assert replay["idempotent_replay"] is True
    assert len(calls) == 2
    assert len(memory.list_memory_items("pt-1", include_deleted=True)) == 1


def test_session_reflection_timeout_is_failed_and_retryable(tmp_path: Path):
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: (_ for _ in ()).throw(
            TimeoutError("qwen timeout")
        ),
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "failed"
    assert memory.get_session_reflection("pt-1", "s1")["status"] == "failed"
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_invalid_json_fails_closed(tmp_path: Path):
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: "{not valid json",
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    result = memory.reflect_session("pt-1", "s1")

    assert result["status"] == "failed"
    assert memory.list_memory_items("pt-1", include_deleted=True) == []


def test_session_reflection_migrates_legacy_table_and_adds_lease_token(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE emotion_memobase_session_reflections (
                patient_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (patient_id, session_id)
            )"""
        )

    memory = EmotionMemobase(str(db_path), logger=lambda _message: None)
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    claim = memory._claim_session_reflection("pt-1", "s1", force=False)
    reflection = memory.get_session_reflection("pt-1", "s1")

    assert claim["claimed"] is True
    assert reflection["lease_token"] == claim["lease_token"]
    assert "summary_evidence_json" in reflection


def test_expired_reflection_lease_fences_old_worker(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"), logger=lambda _message: None)
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")

    first = memory._claim_session_reflection("pt-1", "s1", force=False)
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        conn.execute(
            """UPDATE emotion_memobase_session_reflections
               SET lease_until=? WHERE patient_id=? AND session_id=?""",
            ("2000-01-01T00:00:00", "pt-1", "s1"),
        )
    second = memory._claim_session_reflection("pt-1", "s1", force=True)

    stale = memory._fail_session_reflection(
        "pt-1", "s1", RuntimeError("old worker"), first["lease_token"]
    )
    current = memory.get_session_reflection("pt-1", "s1")

    assert second["lease_token"] != first["lease_token"]
    assert stale["status"] == "stale"
    assert current["status"] == "running"
    assert current["lease_token"] == second["lease_token"]


def test_candidate_requires_explicit_confirmation_or_rejection(tmp_path: Path):
    payload = {
        "session_summary": {"content": "患者叫小明。", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [_candidate("患者叫小明", "我叫小明", "turn-1")],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")
    memory.reflect_session("pt-1", "s1")
    candidate = memory.list_memory_items("pt-1", include_deleted=True)[0]

    confirmed = memory.confirm_memory_item("pt-1", candidate["item_id"], candidate["version"])

    assert confirmed["status"] == "active"
    assert confirmed["version"] == candidate["version"] + 1
    assert "患者叫小明" in memory.get_context_for_llm("pt-1")

    rejected = memory.stop_memory_item("pt-1", confirmed["item_id"], confirmed["version"])

    assert rejected["status"] == "blocked"
    assert rejected["content"] == "患者叫小明"
    assert "患者叫小明" not in memory.get_context_for_llm("pt-1")
    assert memory.get_memory_item("pt-1", confirmed["item_id"])["status"] == "blocked"


def test_reject_candidate_blocks_without_making_it_active(tmp_path: Path):
    payload = {
        "session_summary": {"content": "患者叫小明。", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [_candidate("患者叫小明", "我叫小明", "turn-1")],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn("pt-1", "我叫小明", "收到", session_id="s1", turn_id="turn-1")
    memory.reflect_session("pt-1", "s1")
    candidate = memory.list_memory_items("pt-1", include_deleted=True)[0]

    rejected = memory.reject_memory_item("pt-1", candidate["item_id"], candidate["version"])

    assert rejected["status"] == "blocked"
    assert rejected["version"] == candidate["version"] + 1
    assert "患者叫小明" not in memory.get_context_for_llm("pt-1")


def test_reflection_prompt_preserves_negation_and_temporary_scope():
    prompt = EmotionMemobase._reflection_prompt([
        {"turn_id": "turn-1", "user_message": "我不喜欢钓鱼", "assistant_message": "收到"}
    ])

    assert "content 必须保留否定方向" in prompt
    assert "不得升级为永久边界" in prompt
    assert "明确纠正必须提取为一条 candidate" in prompt
    assert "evidence_turn_ids 中列出所有支持该事实的 user turn" in prompt


def test_qwen_oracle_candidate_match_checks_negation_polarity():
    case = {
        "category": "preferences",
        "allowed_categories": ["preferences"],
        "allowed_operations": ["ADD", "CANDIDATE"],
        "content_all_of": [["不喜欢", "不爱"], "钓鱼"],
    }

    assert candidate_matches(case, {
        "category": "preferences",
        "content": "用户不喜欢钓鱼",
        "suggested_operation": "ADD",
    })
    assert not candidate_matches(case, {
        "category": "preferences",
        "content": "用户喜欢钓鱼",
        "suggested_operation": "ADD",
    })


def test_qwen_oracle_allows_category_variant_but_checks_content():
    case = {
        "category": "events",
        "allowed_categories": ["events", "emotional_triggers"],
        "content_all_of": [["家人"], ["争吵"], ["难受"]],
    }

    assert candidate_matches(case, {
        "category": "emotional_triggers",
        "content": "与家人争吵后感到难受",
        "suggested_operation": "ADD",
    })
    assert not candidate_matches(case, {
        "category": "emotional_triggers",
        "content": "与女儿相处愉快",
        "suggested_operation": "ADD",
    })


def test_qwen_oracle_accepts_structured_expiry_as_time_scope():
    case = {
        "category": "events",
        "allowed_categories": ["events", "facts"],
        "content_all_of": [["这阵子", "近期", "暂时", "目前"], "住", "儿子"],
    }

    assert candidate_matches(case, {
        "category": "events",
        "content": "用户住在儿子家，计划下月返回自己的住所",
        "valid_until": "2026-09-30T23:59:59",
        "suggested_operation": "ADD",
    })


def test_qwen_oracle_requires_current_user_evidence_and_multi_turn_binding():
    case = {
        "id": "M-test",
        "category": "comfort_strategies",
        "allowed_categories": ["comfort_strategies", "preferences"],
        "content_all_of": [["晒太阳"], ["安静"]],
        "expect": "candidate",
        "turns": [["我喜欢晒太阳", "收到"], ["晒太阳让我安静", "明白"]],
        "evidence_indexes": [0, 1],
    }
    item = {
        "category": "preferences",
        "content": "晒太阳能让心里安静",
        "suggested_operation": "ADD",
        "source_turn_id": "M-test-turn-0",
        "evidence_turn_ids": ["M-test-turn-0"],
        "evidence_quote": "我喜欢晒太阳",
    }

    assert evidence_is_bound(case, item)
    verdict, failures = evaluate_case(case, {"status": "succeeded"}, [item])
    assert verdict == "fail"
    assert "multi_turn_evidence_incomplete" in failures

    item["evidence_turn_ids"] = ["M-test-turn-0", "M-test-turn-1"]
    verdict, failures = evaluate_case(case, {"status": "succeeded"}, [item])
    assert verdict == "pass"
    assert failures == []

    item["evidence_quote"] = "晒太阳让我安静"
    assert evidence_is_bound(case, item)

    item["evidence_quote"] = "你应该很孤独"
    assert not evidence_is_bound(case, item)


def test_temporary_boundary_scope_is_repaired_from_user_evidence(tmp_path: Path):
    payload = {
        "session_summary": {"content": "当前不讨论孙子话题。", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [{
            "category": "boundaries",
            "content": "用户不希望讨论孙子相关话题",
            "source_turn_id": "turn-1",
            "evidence_turn_ids": ["turn-1"],
            "evidence_quote": "我现在不想聊孙子的事",
            "observed_at": "2026-08-17T00:00:00Z",
            "valid_until": None,
            "sensitivity": "normal",
            "confidence": 0.9,
            "suggested_operation": "ADD",
        }],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1", "我现在不想聊孙子的事", "好的",
        session_id="s1", turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["rejected_count"] == 0
    items = memory.list_memory_items("pt-1", include_deleted=True)
    assert len(items) == 1
    assert "现在" in items[0]["content"]


def test_uncertain_user_statement_is_repaired_from_user_evidence(tmp_path: Path):
    payload = {
        "session_summary": {"content": "无", "evidence_turn_ids": ["turn-1"]},
        "memory_candidates": [{
            "category": "preferences",
            "content": "用户计划下周参加社区活动",
            "source_turn_id": "turn-1",
            "evidence_turn_ids": ["turn-1"],
            "evidence_quote": "我也许下周想去社区活动",
            "observed_at": "2026-08-17T00:00:00Z",
            "valid_until": None,
            "sensitivity": "normal",
            "confidence": 0.9,
            "suggested_operation": "ADD",
        }],
    }
    memory = EmotionMemobase(
        str(tmp_path / "memory.db"),
        session_reflector=lambda _prompt: payload,
        logger=lambda _message: None,
    )
    memory.capture_turn(
        "pt-1", "我也许下周想去社区活动", "收到",
        session_id="s1", turn_id="turn-1",
    )

    result = memory.reflect_session("pt-1", "s1")

    assert result["rejected_count"] == 0
    items = memory.list_memory_items("pt-1", include_deleted=True)
    assert len(items) == 1
    assert items[0]["content"] == "我也许下周想去社区活动"


def test_temporary_boundary_with_valid_until_is_acceptable_candidate():
    case = {
        "id": "N03",
        "category": "boundaries",
        "allowed_categories": ["boundaries"],
        "content_all_of": [["孙子"]],
        "expect": "candidate_or_none",
        "require_valid_until": True,
        "turns": [["我现在不想聊孙子的事", "好的"]],
    }
    item = {
        "category": "boundaries",
        "content": "不想聊孙子的事",
        "suggested_operation": "ADD",
        "source_turn_id": "N03-turn-0",
        "evidence_turn_ids": ["N03-turn-0"],
        "evidence_quote": "我现在不想聊孙子的事",
        "valid_until": "2026-08-18T11:00:00Z",
    }

    verdict, failures = evaluate_case(case, {"status": "succeeded"}, [item])

    assert verdict == "pass"
    assert failures == []


def test_shadow_reflection_oracle_has_fifty_simulated_patient_scenarios(tmp_path: Path):
    oracle_path = Path(__file__).parent / "fixtures" / "memory_reflection_oracle.json"
    cases = json.loads(oracle_path.read_text(encoding="utf-8"))

    assert len(cases) == 50
    assert {
        "negation", "past", "temporary", "correction", "conflict", "crisis",
        "assistant_induced", "multi_turn",
    } <= {case["kind"] for case in cases}

    for case in cases:
        turn_ids = [f"{case['id']}-turn-{index}" for index in range(len(case["turns"]))]
        evidence_indexes = case.get("evidence_indexes", [0])
        payload = {
            "session_summary": {
                "content": f"模拟会话 {case['id']} 的摘要",
                "evidence_turn_ids": [turn_ids[0]],
            },
            "memory_candidates": [
                {
                    "category": case["category"],
                    "content": case["content"],
                    "source_turn_id": turn_ids[0],
                    "evidence_turn_ids": [turn_ids[index] for index in evidence_indexes],
                    "evidence_quote": case["quote"],
                    "observed_at": "2026-08-16",
                    "valid_until": None,
                    "sensitivity": case["sensitivity"],
                    "confidence": 0.95,
                    "slot_key": f"oracle:{case['id']}",
                    "suggested_operation": case["operation"],
                }
            ],
        }
        memory = EmotionMemobase(
            str(tmp_path / f"{case['id']}.db"),
            session_reflector=lambda _prompt, result=payload: result,
            logger=lambda _message: None,
        )
        for index, (user_message, assistant_message) in enumerate(case["turns"]):
            memory.capture_turn(
                "pt-1",
                user_message,
                assistant_message,
                session_id="oracle-session",
                turn_id=turn_ids[index],
            )
        if case.get("existing"):
            memory.update_memory_by_user("pt-1", {"facts": [case["existing"]]})

        result = memory.reflect_session("pt-1", "oracle-session")
        items = memory.list_memory_items("pt-1", include_deleted=True)
        shadow_items = [item for item in items if item["source"] == "llm_shadow"]

        assert result["status"] == "succeeded", case["id"]
        assert not [item for item in shadow_items if item["status"] == "active"], case["id"]
        if case["expect"] in {"candidate", "candidate_or_none"}:
            assert len(shadow_items) <= 1, case["id"]
            if shadow_items:
                assert shadow_items[0]["status"] == "candidate", case["id"]
        else:
            assert shadow_items == [], case["id"]
        if case["expect"] == "rejected":
            assert result["rejected_count"] == 1, case["id"]
        if case.get("existing"):
            assert any(
                item["content"] == case["existing"] and item["status"] == "active"
                for item in items
            ), case["id"]
