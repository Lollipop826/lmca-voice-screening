from __future__ import annotations

import asyncio
import builtins
import importlib.util
import wave
from pathlib import Path

import pytest

import src.tools.emotion as emotion


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_emotion_benchmark_uses_audio_classifier_api(tmp_path, monkeypatch):
    module = _load_script("benchmark_emotion_models")

    class FakeClassifier:
        model_available = True
        last_error = None

        def __init__(self):
            self.paths = []

        def classify_audio(self, path):
            self.paths.append(path)
            label = "joy" if Path(path).stem == "joy" else "sadness"
            return {emotion_name: float(emotion_name == label) for emotion_name in module.EmotionModelBenchmark.EMOTION_LABELS}

    classifier = FakeClassifier()
    monkeypatch.setattr(emotion, "get_audio_emotion_classifier", lambda: classifier)
    joy = tmp_path / "joy.wav"
    sadness = tmp_path / "sadness.wav"
    joy.write_bytes(b"audio")
    sadness.write_bytes(b"audio")

    benchmark = module.EmotionModelBenchmark(str(tmp_path))
    benchmark.test_samples = [
        {"audio_path": str(joy), "true_emotion": "joy", "age_group": "young"},
        {"audio_path": str(sadness), "true_emotion": "sadness", "age_group": "elderly"},
        {"audio_path": str(tmp_path / "missing.wav"), "true_emotion": "calm"},
    ]

    result = benchmark.benchmark_emotion2vec()

    assert result.accuracy == 1.0
    assert classifier.paths[1:] == [str(joy), str(sadness)]


def test_streaming_latency_records_first_token_and_completion():
    module = _load_script("benchmark_latency")

    class Chunk:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def stream(self, _messages):
            yield Chunk("")
            yield Chunk("我在听")
            yield Chunk("。")

    result = module.measure_streaming_llm(
        FakeLLM(),
        prompt="测试",
        runs=2,
    )

    assert result["first_token"]["count"] == 2
    assert result["complete"]["count"] == 2


def test_audio_latency_uses_audio_classifier_api(tmp_path):
    module = _load_script("benchmark_latency")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    class FakeClassifier:
        model_available = True
        last_error = None

        def __init__(self):
            self.paths = []

        def classify_audio(self, path):
            self.paths.append(path)
            return {"calm": 1.0}

    classifier = FakeClassifier()
    result = module.measure_audio_emotion(audio, runs=2, classifier=classifier)

    assert result["count"] == 2
    assert classifier.paths == [str(audio)] * 3


def test_memobase_latency_uses_direct_context_retrieval():
    module = _load_script("benchmark_latency")

    class FakeMemory:
        def __init__(self):
            self.calls = []

        def is_memobase_available(self):
            return True

        def get_relevant_context(self, patient_id, text):
            self.calls.append((patient_id, text))
            return "相关旧事"

    memory = FakeMemory()
    result = module.measure_memobase_retrieval(
        memory,
        patient_id="patient-1",
        current_text="我又想起那件事了",
        runs=2,
    )

    assert result["count"] == 2
    assert result["context_chars"] == len("相关旧事")
    assert memory.calls == [("patient-1", "我又想起那件事了")] * 3


def test_memory_scale_uses_voice_context_limit_and_index(tmp_path):
    module = _load_script("benchmark_latency")

    result = module.measure_sqlite_memory_scale(
        [0, 12],
        runs=1,
        db_path=tmp_path / "memory-scale.db",
    )

    scenarios = result["scenarios"]
    assert [item["history_turns"] for item in scenarios] == [0, 12]
    assert all(item["voice_turn_context_chars"] <= 1800 for item in scenarios)
    assert all(
        "USE TEMP B-TREE" not in " ".join(item["query_plan"])
        for item in scenarios
    )


def test_voice_e2e_reads_pcm16_and_has_dry_run(tmp_path, capsys):
    module = _load_script("benchmark_voice_e2e")
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 160)

    assert len(module._read_pcm16(audio)) == 320
    assert module.main([
        "--audio", str(audio),
        "--username", "test",
        "--password", "test",
        "--patient-id", "patient-1",
    ]) == 0
    assert '"live": false' in capsys.readouterr().out


def test_voice_e2e_rejects_emotion_fallback_when_audio_is_required(tmp_path, monkeypatch):
    module = _load_script("benchmark_voice_e2e")
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 160)

    monkeypatch.setattr(module, "_server_flags", lambda _server: {
        "turn_insight": True,
        "long_term_memory": True,
    })
    monkeypatch.setattr(module, "_login", lambda *_args: "aa_session=test")

    async def benchmark(*_args, **_kwargs):
        return {
            "runs": [{
                "turn_id": "turn-1",
                "emotion": {"source": "text_fallback_model_unavailable", "audio_model_used": False},
            }],
            "summary": {},
        }

    monkeypatch.setattr(module, "_run_benchmark", benchmark)

    with pytest.raises(RuntimeError, match="audio emotion verification failed"):
        module.main([
            "--live",
            "--audio", str(audio),
            "--username", "test",
            "--password", "test",
            "--patient-id", "patient-1",
            "--require-audio-emotion",
        ])


def test_voice_e2e_rejects_unexpected_memory_write_state(tmp_path, monkeypatch):
    module = _load_script("benchmark_voice_e2e")
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 160)

    monkeypatch.setattr(module, "_server_flags", lambda _server: {
        "turn_insight": True,
        "long_term_memory": True,
        "long_term_memory_writes": True,
    })

    with pytest.raises(RuntimeError, match="write state does not match"):
        module.main([
            "--live",
            "--audio", str(audio),
            "--username", "test",
            "--password", "test",
            "--patient-id", "patient-1",
            "--expect-long-term-memory-writes", "off",
        ])


def test_voice_e2e_records_session_latency_and_final_emotion(monkeypatch):
    module = _load_script("benchmark_voice_e2e")

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, payload):
            self.sent.append(payload)

    websocket = FakeWebSocket()
    events = iter(
        [
            {"type": "asr_result", "turn_id": "turn-1"},
            {
                "type": "ai_response_chunk",
                "turn_id": "turn-1",
                "is_first": True,
            },
            {"type": "tts_start", "turn_id": "turn-1"},
            {"type": "tts_chunk", "turn_id": "turn-1"},
            {"type": "tts_end", "turn_id": "turn-1"},
            {
                "type": "turn_insight",
                "turn_id": "turn-1",
                "state": "final",
                "emotion": {
                    "source": "emotion2vec_audio+text",
                    "analysis_status": "final",
                    "audio_model_used": True,
                },
            },
        ]
    )

    async def prepare_session(*_args, **_kwargs):
        return "patient-1", 12.5

    async def receive_event(*_args):
        return next(events)

    monkeypatch.setattr(module, "_connect_ws", lambda *_args: websocket)
    monkeypatch.setattr(module, "_prepare_session", prepare_session)
    monkeypatch.setattr(module, "_recv_json", receive_event)

    result = asyncio.run(
        module._run_once(
            websocket_url="ws://test/ws",
            cookie="aa_session=test",
            pcm16=b"\x00\x00",
            timeout_s=1.0,
            patient_id="patient-1",
            new_patient=False,
            profile_name="test",
            long_term_memory_enabled=True,
            audio_output=None,
        )
    )

    assert result["turn_id"] == "turn-1"
    assert result["session_start_ms"] == 12.5
    assert "input_to_tts_end_ms" in result
    assert result["asr_to_first_ai_text_ms"] >= 0
    assert result["asr_to_first_tts_byte_ms"] >= 0
    assert result["emotion"]["audio_model_used"] is True
    assert len(websocket.sent) == 1


def test_voice_e2e_reports_server_error_without_waiting_for_tts_end(monkeypatch):
    module = _load_script("benchmark_voice_e2e")

    class FakeWebSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, _payload):
            return None

    async def prepare_session(*_args, **_kwargs):
        return "patient-1", 12.5

    events = iter(
        [
            {"type": "asr_result", "turn_id": "turn-1"},
            {
                "type": "ai_response",
                "turn_id": "turn-1",
                "error": True,
                "text": "服务端回合失败",
            },
        ]
    )

    async def receive_event(*_args):
        return next(events)

    monkeypatch.setattr(module, "_connect_ws", lambda *_args: FakeWebSocket())
    monkeypatch.setattr(module, "_prepare_session", prepare_session)
    monkeypatch.setattr(module, "_recv_json", receive_event)

    with pytest.raises(RuntimeError, match="服务端回合失败"):
        asyncio.run(
            module._run_once(
                websocket_url="ws://test/ws",
                cookie="aa_session=test",
                pcm16=b"\x00\x00",
                timeout_s=1.0,
                patient_id="patient-1",
                new_patient=False,
                profile_name="test",
                long_term_memory_enabled=True,
                audio_output=None,
            )
        )


def test_voice_e2e_retries_failed_logical_runs(monkeypatch):
    module = _load_script("benchmark_voice_e2e")
    attempts = 0

    async def run_once(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary ASR failure")
        return {
            "turn_id": "turn-1",
            "session_start_ms": 1.0,
            "emotion": {
                "source": "emotion2vec_audio+text",
                "audio_model_used": True,
            },
        }

    class Args:
        server = "http://test"
        warmup_runs = 0
        runs = 1
        max_attempts_per_run = 2
        retry_delay = 0
        timeout = 1
        patient_id = "patient-1"
        new_patient = False
        profile_name = "test"
        long_term_memory = "on"
        audio_output_dir = None
        require_audio_emotion = True

    monkeypatch.setattr(module, "_run_once", run_once)
    result = asyncio.run(module._run_benchmark(Args(), b"audio", "cookie"))

    assert attempts == 2
    assert len(result["runs"]) == 1
    assert result["failures"] == [{
        "phase": "measured",
        "logical_run": 1,
        "attempt": 1,
        "error_type": "RuntimeError",
        "detail": "temporary ASR failure",
    }]


def test_ark_asr_latency_records_partial_and_final_without_network():
    module = _load_script("benchmark_latency")
    streams = []

    class FakeStream:
        def __init__(self, *, sample_rate, on_result):
            self.sample_rate = sample_rate
            self.on_result = on_result
            self.chunks = []

        async def start(self):
            return None

        async def feed(self, chunk):
            self.chunks.append(list(chunk))
            if len(self.chunks) == 1:
                await self.on_result("暂时文字", False)

        async def finish(self):
            await self.on_result("最终文字", True)
            return "最终文字"

        async def aclose(self):
            return None

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    async def no_sleep(_seconds):
        return None

    result = asyncio.run(
        module.measure_ark_asr_stream(
            [0.0] * 3200,
            16000,
            stream_factory=factory,
            chunk_seconds=0.1,
            sleeper=no_sleep,
        )
    )

    assert result["chunks"] == 2
    assert result["first_partial_ms"] is not None
    assert result["final_text_ms"] >= result["first_partial_ms"]
    assert len(streams[0].chunks) == 2


def test_ark_asr_latency_does_not_report_final_as_partial():
    module = _load_script("benchmark_latency")

    class FakeStream:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            return None

        async def feed(self, _chunk):
            return None

        async def finish(self):
            return "只有最终文字"

        async def aclose(self):
            return None

    async def no_sleep(_seconds):
        return None

    result = asyncio.run(
        module.measure_ark_asr_stream(
            [0.0] * 1600,
            16000,
            stream_factory=FakeStream,
            sleeper=no_sleep,
        )
    )

    assert result["first_partial_ms"] is None
    assert result["final_text_ms"] is not None


def test_latency_cli_requires_live_before_measurement(monkeypatch, capsys):
    module = _load_script("benchmark_latency")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "soundfile" or name.startswith((
            "src.context_management.emotion_memobase",
            "src.tools.voice.ark_asr",
        )):
            raise AssertionError(f"非 --live 路径不应导入: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert module.main([
        "--audio", "sample.wav",
        "--memobase-patient-id", "patient-1",
        "--memory-text", "当前文本",
        "--ark-asr-audio", "sample.wav",
    ]) == 0
    output = capsys.readouterr().out
    assert '"live": false' in output
    assert '"memobase": true' in output
