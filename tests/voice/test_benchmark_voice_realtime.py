import json
import asyncio
from types import SimpleNamespace
import wave

from scripts import benchmark_voice_realtime as benchmark


def test_ablation_specs_are_paired_randomized_and_read_only():
    arguments = SimpleNamespace(
        warmup_runs=1,
        runs=3,
        ablation="memory",
        seed=7,
        emotion="on",
        disable_long_term_memory=False,
    )

    specs = benchmark._build_run_specs(arguments)

    assert len(specs) == 8
    assert all(not spec["memory_writes_enabled"] for spec in specs)
    measured = [spec for spec in specs if not spec["is_warmup"]]
    for pair_index in (1, 2, 3):
        pair = [spec for spec in measured if spec["pair_index"] == pair_index]
        assert {spec["condition"] for spec in pair} == {
            "memory_off",
            "memory_on",
        }
        assert {spec["memory_enabled"] for spec in pair} == {False, True}


def test_stage_metrics_split_tts_queue_from_provider_and_record_soulx_first_text():
    timeline = [
        (10.4, {"type": "soulx_state", "state": "nonidle", "text": "我今"}),
        (13.5, {"type": "vad_end"}),
        (
            13.7,
            {
                "type": "asr_result",
                "turn_id": "turn_0001",
                "text": "我今天很好",
                "source": "soulx_final",
            },
        ),
        (14.0, {"type": "ai_response_chunk", "turn_id": "turn_0001", "text": "好"}),
        (14.2, {"type": "tts_start", "turn_id": "turn_0001"}),
        (15.1, {"type": "tts_chunk", "turn_id": "turn_0001"}),
        (16.0, {"type": "tts_end", "turn_id": "turn_0001"}),
    ]

    selected = benchmark._select_measured_turn(timeline)
    record = benchmark._build_record(
        marks=selected["marks"],
        zero_at=13.0,
        stream_started_at=10.0,
        speech_frame_count=100,
        observed_turn_ids=selected["observed_turn_ids"],
        recognized_text=selected["recognized_text"],
        assistant_text=selected["assistant_text"],
        final_insight=None,
        patient_id="pt-1",
        patient_has_history=True,
        session_id="session-1",
        requested_flags={},
        effective_flags={},
        asr_source=selected["asr_source"],
    )

    assert record["valid_for_latency"] is True
    assert record["soulx_first_text_ms"] == 400.0
    assert record["post_vad_pipeline_ms"] == 200.0
    assert record["agent_pipeline_ms"] == 300.0
    assert record["tts_queue_ms"] == 200.0
    assert record["tts_provider_first_byte_ms"] == 900.0
    assert record["tts_first_byte_ms"] == 1100.0
    assert record["asr_source"] == "soulx_final"


def test_split_turn_records_are_marked_invalid_for_stage_statistics():
    record = benchmark._build_record(
        marks={"vad_end": 2.0, "asr_result": 3.0},
        zero_at=1.0,
        stream_started_at=0.0,
        speech_frame_count=1,
        observed_turn_ids=["turn_0001", "turn_0002"],
        recognized_text="文本",
        assistant_text="",
        final_insight=None,
        patient_id="pt-1",
        patient_has_history=True,
        session_id="session-1",
        requested_flags={},
        effective_flags={},
        asr_source="soulx_final",
    )

    assert record["valid_for_latency"] is False
    summary = benchmark._condition_summary([{**record, "condition": "default"}])
    assert summary["default"]["valid_single_turn_records"] == 0
    assert summary["default"]["metrics"]["post_vad_pipeline_ms"] == {"count": 0}


def _memory_insight(**memory_fields):
    return {
        "emotion": {"dominant": "sadness", "source": "text_asr"},
        "memory": {
            "retrieval_source": "realtime",
            "writes_enabled": False,
            **memory_fields,
        },
    }


def _record_with_insight(insight):
    return benchmark._build_record(
        marks={"vad_end": 2.0, "asr_result": 3.0},
        zero_at=1.0,
        stream_started_at=0.0,
        speech_frame_count=1,
        observed_turn_ids=["turn_0001"],
        recognized_text="我最近有点难过",
        assistant_text="我在听。",
        final_insight=insight,
        patient_id="pt-1",
        patient_has_history=True,
        session_id="session-1",
        requested_flags={},
        effective_flags={},
        asr_source="soulx_final",
    )


def test_record_captures_retrieval_kind_so_fallback_samples_can_be_rejected():
    """The ablation guard needs provenance, not just a boolean hit flag."""
    record = _record_with_insight(
        _memory_insight(
            retrieval_kind="local_fallback",
            retrieval_hit=False,
            used_item_ids=["event-walk", "fact-sleep"],
        )
    )

    assert record["memory_retrieval_kind"] == "local_fallback"
    assert record["memory_retrieval_hit"] is False
    assert record["memory_used_items"] == 2


def test_record_captures_semantic_retrieval_kind():
    record = _record_with_insight(
        _memory_insight(
            retrieval_kind="semantic",
            retrieval_hit=True,
            used_item_ids=[],
        )
    )

    assert record["memory_retrieval_kind"] == "semantic"
    assert record["memory_retrieval_hit"] is True


def test_character_error_rate_ignores_punctuation_and_whitespace():
    assert benchmark._character_error_rate("你好，世界！", "你好 世界") == 0.0
    assert benchmark._character_error_rate("你好世界", "你好世") == 0.25


def test_audio_manifest_resolves_relative_paths_and_reference_text(tmp_path):
    audio = tmp_path / "speech.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x01\x00" * 1600)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-1",
                        "audio_path": "speech.wav",
                        "reference_text": "今天很好",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    arguments = SimpleNamespace(
        audio_manifest=str(manifest),
        audio="unused.wav",
        reference_text=None,
    )

    cases = benchmark._load_audio_cases(arguments)

    assert cases == [
        {
            "sample_id": "sample-1",
            "audio_path": str(audio.resolve()),
            "reference_text": "今天很好",
        }
    ]


def test_session_start_requires_server_to_echo_effective_ablation_flags(monkeypatch):
    sleeps = []

    async def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(benchmark.asyncio, "sleep", fake_sleep)

    class WebSocket:
        def __init__(self):
            self.events = iter(
                [
                    {"type": "waiting_for_info"},
                    {"type": "patient_memory", "patient_id": "pt-1", "has_history": True},
                    {
                        "type": "session_started",
                        "session_id": "session-1",
                        "long_term_memory_enabled": True,
                        "long_term_memory_writes_enabled": False,
                        "emotion_enabled": False,
                    },
                    {"type": "tts_end", "duration": 2.5},
                ]
            )
            self.sent = []

        async def recv(self):
            return json.dumps(next(self.events))

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    websocket = WebSocket()
    handshake = asyncio.run(
        benchmark._start_session(
            websocket,
            patient_id="pt-1",
            create_patient=False,
            profile_name="unused",
            long_term_memory_enabled=True,
            long_term_memory_writes_enabled=False,
            emotion_enabled=False,
            verify_effective_flags=True,
            timeout_s=1.0,
        )
    )

    assert handshake["patient_has_history"] is True
    assert handshake["effective_flags"] == {
        "long_term_memory_enabled": True,
        "long_term_memory_writes_enabled": False,
        "emotion_enabled": False,
    }
    assert websocket.sent[0]["long_term_memory_writes_enabled"] is False
    assert sleeps == [2.5 + benchmark._GREETING_PLAYBACK_GUARD_S]
