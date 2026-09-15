import json

from scripts import prepare_memory_ablation_fixture as fixture
from src.db import database


def _raw_fixture(patient_id="pt-fixture-test"):
    return {
        "fixture_id": "fixture-test",
        "patient_id": patient_id,
        "profile": {"name": "测试患者", "age": 70},
        "memory": {
            "events": [{"text": "上周和女儿因为联系频率发生争执。"}],
            "facts": [],
            "preferences": [],
            "narrative": "仅用于测试",
        },
        "cases": [
            {
                "sample_id": "family-followup",
                "user_text": "我这两天还是因为女儿不回消息难受。",
                "memory_facts": ["上周和女儿因为联系频率发生争执。"],
                "scenario": "相关记忆",
            }
        ],
    }


def test_validate_fixture_requires_cross_session_case_metadata():
    config = fixture.validate_fixture(_raw_fixture())

    assert config["patient_id"] == "pt-fixture-test"
    assert config["cases"][0]["memory_facts"]
    assert config["memory"]["events"][0]["status"] == "active"


def test_prepare_fixture_dry_run_does_not_write(tmp_path):
    original_db_path = database.DB_PATH
    try:
        result = fixture.prepare_fixture(
            _raw_fixture(), db_path=str(tmp_path / "voice.db"), apply=False
        )
    finally:
        database.DB_PATH = original_db_path

    assert result["applied"] is False
    assert result["cross_session_required"] is True
    assert "memory_item_ids" not in result


def test_prepare_fixture_applies_real_memory_items(tmp_path):
    original_db_path = database.DB_PATH
    try:
        result = fixture.prepare_fixture(
            _raw_fixture("pt-fixture-apply"),
            db_path=str(tmp_path / "voice.db"),
            apply=True,
        )
        assert result["applied"] is True
        assert result["memory_item_count"] == 1
        assert result["memory_items"][0]["status"] == "active"
        assert database.get_patient("pt-fixture-apply") is not None
    finally:
        database.DB_PATH = original_db_path


def test_quality_context_accepts_fixture_case_list():
    from scripts import evaluate_response_quality as quality

    benchmark = {
        "configuration": {"ablation": "memory"},
        "records": [
            {
                "condition": "memory_on",
                "pair_index": 1,
                "sample_id": "family-followup",
                "valid_for_latency": True,
                "turn_count": 1,
                "asr_source": "soulx_final",
                "assistant_text": "记得你之前提过这件事。",
                "recognized_text": "我这两天还是因为女儿不回消息难受。",
                "memory_retrieval_hit": True,
                "memory_writes_enabled": False,
            },
            {
                "condition": "memory_off",
                "pair_index": 1,
                "sample_id": "family-followup",
                "valid_for_latency": True,
                "turn_count": 1,
                "asr_source": "soulx_final",
                "assistant_text": "听起来你现在有些难受。",
                "recognized_text": "我这两天还是因为女儿不回消息难受。",
                "memory_retrieval_hit": False,
                "memory_writes_enabled": False,
            },
        ],
    }
    context = {
        "cases": [
            {
                "sample_id": "family-followup",
                "memory_facts": ["上周和女儿因为联系频率发生争执。"],
                "scenario": "相关记忆",
            }
        ]
    }

    sheet, _ = quality.prepare_blind_sheet(
        benchmark, seed=1, raters=["r1"], context=context
    )

    assert sheet["cases"][0]["context"]["memory_facts"]
