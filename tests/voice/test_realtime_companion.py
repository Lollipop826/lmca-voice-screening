from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np

from src.voice.handlers.live_audio_input_handler import LiveAudioInputHandler
from src.voice.realtime_companion import (
    RealtimeCompanion,
    RealtimeCompanionConfig,
    RealtimeTurn,
)


class _Connection:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)
        return True


def _session():
    return SimpleNamespace(
        lifecycle=SimpleNamespace(current_patient_id="patient-1", started=True),
        runtime=SimpleNamespace(
            high_risk_detected=False,
            ai_streaming_tts=False,
            recent_tts_texts=[],
        ),
    )


def test_slow_emotion_inference_restarts_only_the_latest_window():
    async def scenario():
        release = threading.Event()
        started = threading.Event()
        calls = []
        now = [100.0]

        def classify(text, audio, sample_rate):
            calls.append((text, len(audio), sample_rate))
            if len(calls) == 1:
                started.set()
                release.wait(1)
            return {"sadness": 0.85}

        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0}

        companion = RealtimeCompanion(
            _Connection(),
            session=_session(),
            patient_memory_service=None,
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True),
            classify_window=classify,
            now_factory=lambda: now[0],
            logger=lambda _message: None,
        )
        turn = RealtimeTurn(companion)
        companion._active = turn
        turn._window.append(np.ones(1600, dtype=np.float32))
        turn._window_samples = 1600
        turn.partial_text = "我现在很难过"

        turn._schedule_emotion()
        assert await asyncio.to_thread(started.wait, 1)
        now[0] = 100.9
        turn._schedule_emotion()
        release.set()

        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(calls) == 2 and turn._emotion_task.done():
                break

        assert len(calls) == 2
        assert companion._comfort_count == 1
        await turn.aclose()

    asyncio.run(scenario())


def test_realtime_asr_does_not_feed_the_same_frame_to_soulx():
    async def scenario():
        calls = {"soulx": 0, "realtime": 0}

        class SoulX:
            async def handle_audio(self, _audio):
                calls["soulx"] += 1
                return True

        async def observe_frame(*_args, **_kwargs):
            calls["realtime"] += 1
            return None

        handler = object.__new__(LiveAudioInputHandler)
        handler.session = _session()
        handler.soulx_audio_handler = SoulX()
        handler.vad_buffer = SimpleNamespace(is_speaking=True)
        handler._realtime_companion = SimpleNamespace(
            config=SimpleNamespace(enabled=True),
            observe_frame=observe_frame,
        )
        handler._ensure_session_on_speech = None
        handler._feed_local_vad = lambda _audio: SimpleNamespace(
            was_speaking=False,
            complete_audio=None,
        )

        async def stop_after_realtime(_frame):
            return True

        handler._apply_local_vad_gates = stop_after_realtime
        assert await handler.handle_message(
            {"type": "audio", "_audio_float": np.ones(320)}
        )
        assert calls == {"soulx": 0, "realtime": 1}

    asyncio.run(scenario())


def test_new_speech_cancels_comfort_and_closes_its_playback():
    async def scenario():
        connection = _Connection()
        stream_started = asyncio.Event()

        class Stream:
            async def start(self):
                return None

            async def feed(self, _audio):
                return None

            async def finish(self):
                return ""

            async def aclose(self):
                return None

        async def stream_tts(_text, **kwargs):
            identity = {
                "session_id": "call-1",
                "turn_id": "turn-1",
                "generation": 1,
                "playback_id": "companion-1",
            }
            await connection.send_json(
                {**kwargs["start_payload"], **identity}
            )
            stream_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {
                    "samples": 0,
                    "interrupted": True,
                    "interruption_reason": "cancelled",
                    "output_identity": identity,
                    "started": True,
                }

        companion = RealtimeCompanion(
            connection,
            session=_session(),
            patient_memory_service=None,
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True),
            stream_factory=lambda **_kwargs: Stream(),
            logger=lambda _message: None,
        )
        await companion._schedule_comfort("我在听。", "sadness", False)
        await stream_started.wait()

        await companion.observe_frame(
            np.ones(320, dtype=np.float32),
            was_speaking=False,
            is_speaking=True,
            complete_audio=None,
        )

        assert companion._comfort_count == 0
        tts_end = next(
            payload
            for payload in connection.sent
            if payload["type"] == "tts_end"
        )
        assert tts_end["reason"] == "cancelled"
        assert tts_end["playback_id"] == "companion-1"
        await companion.reset()

    asyncio.run(scenario())


def test_explicit_distress_partial_triggers_comfort_without_emotion_inference():
    async def scenario():
        for text in (
            "我真的很难过，很伤心",
            "我感觉什么都没意思",
            "我感到很无望",
        ):
            connection = _Connection()
            session = _session()

            async def stream_tts(*_args, **_kwargs):
                return {"samples": 0}

            companion = RealtimeCompanion(
                connection,
                session=session,
                patient_memory_service=None,
                stream_tts_audio=stream_tts,
                config=RealtimeCompanionConfig(enabled=True),
                logger=lambda _message: None,
            )
            turn = RealtimeTurn(companion)
            companion._active = turn

            await turn._on_result(text, False)
            await asyncio.sleep(0)

            comfort = next(
                item for item in connection.sent
                if item["type"] == "companion_message"
            )
            assert comfort["kind"] == "distress"
            assert comfort["safety"] is False
            assert not session.runtime.high_risk_detected
            await companion.reset()

    asyncio.run(scenario())


def test_partial_text_keeps_safety_priority_and_ignores_nonclinical_boredom():
    async def scenario():
        connection = _Connection()
        session = _session()

        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0}

        companion = RealtimeCompanion(
            connection,
            session=session,
            patient_memory_service=None,
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True),
            logger=lambda _message: None,
        )
        turn = RealtimeTurn(companion)
        companion._active = turn

        await turn._on_result("这部电影没意思", False)
        await asyncio.sleep(0)
        assert not any(item["type"] == "companion_message" for item in connection.sent)

        await turn._on_result("我想自杀", False)
        await asyncio.sleep(0)
        assert session.runtime.high_risk_detected
        assert any(item["type"] == "safety_alert" for item in connection.sent)
        safety_message = next(
            item for item in connection.sent
            if item["type"] == "companion_message"
        )
        assert safety_message["kind"] == "safety"
        await companion.reset()

    asyncio.run(scenario())


def test_realtime_anxiety_score_can_trigger_comfort():
    async def scenario():
        connection = _Connection()

        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0}

        companion = RealtimeCompanion(
            connection,
            session=_session(),
            patient_memory_service=None,
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True),
            logger=lambda _message: None,
        )
        companion._active = RealtimeTurn(companion)

        await companion.observe_partial_emotion("", {"anxiety": 0.61})
        await companion.observe_partial_emotion("", {"anxiety": 0.61})
        await asyncio.sleep(0)

        comfort = next(
            item for item in connection.sent
            if item["type"] == "companion_message"
        )
        assert comfort["kind"] == "anxiety"
        await companion.reset()

    asyncio.run(scenario())


def test_realtime_turn_close_cancels_memory_prefetch():
    async def scenario():
        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0}

        companion = RealtimeCompanion(
            _Connection(),
            session=_session(),
            patient_memory_service=SimpleNamespace(
                get_turn_context=lambda *_args: "旧结果"
            ),
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True),
            logger=lambda _message: None,
        )
        turn = RealtimeTurn(companion)
        task = turn._start_memory_query("我又想起浴室滑倒")
        assert task is not None

        await turn.aclose()
        await asyncio.sleep(0)

        assert turn._memory_task is None
        assert turn._memory_result == ""
        assert task.cancelled() or task.done()

    asyncio.run(scenario())


def test_realtime_memory_timeout_does_not_fallback_to_sqlite():
    async def scenario():
        release = threading.Event()
        calls = {"turn": 0, "cross": 0}

        def get_turn_context(*_args):
            calls["turn"] += 1
            release.wait(1)
            return "过期结果"

        def get_cross_session_context(*_args):
            calls["cross"] += 1
            return "SQLite 旧事"

        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0}

        companion = RealtimeCompanion(
            _Connection(),
            session=_session(),
            patient_memory_service=SimpleNamespace(
                get_turn_context=get_turn_context,
                get_cross_session_context=get_cross_session_context,
            ),
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(enabled=True, memory_timeout_s=0.01),
            logger=lambda _message: None,
        )
        turn = RealtimeTurn(companion)

        result = await turn.memory_for_final("我又想起浴室滑倒")
        release.set()

        assert result == ""
        assert calls["turn"] == 1
        assert calls["cross"] == 0
        assert turn._memory_result == ""

    asyncio.run(scenario())
