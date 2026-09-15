from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

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
        assert companion._comfort_count == 0
        await turn.aclose()

    asyncio.run(scenario())


def test_soulx_route_is_primary_when_realtime_companion_is_enabled():
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
        assert calls == {"soulx": 1, "realtime": 0}

    asyncio.run(scenario())


def test_soulx_external_text_drives_companion_without_second_asr():
    async def scenario():
        async def stream_tts(*_args, **_kwargs):
            return {"samples": 0, "started": False}

        companion = RealtimeCompanion(
            _Connection(),
            session=_session(),
            patient_memory_service=None,
            stream_tts_audio=stream_tts,
            config=RealtimeCompanionConfig(
                enabled=True,
                external_asr=True,
            ),
            logger=lambda _message: None,
        )

        pending = await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="nonidle",
            text="我很难过",
        )
        assert pending is None
        assert companion._active is not None
        assert companion._active._stream_task is None

        turn = await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="speak",
            text="我很难过",
            complete_audio=np.ones(4000, dtype=np.float32),
        )
        assert turn is not None
        assert await turn.final_text() == "我很难过"
        assert turn.has_final
        assert companion._active is None
        await companion.reset()

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


def test_distress_text_alone_does_not_bypass_soulx_timing_gate():
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

            assert not any(
                item["type"] == "companion_message" for item in connection.sent
            )
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


def test_emotion_scores_alone_do_not_bypass_soulx_timing_gate():
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
        await companion.observe_partial_emotion("", {"sadness": 0.99})
        await companion.observe_partial_emotion("", {"joy": 0.99})
        await asyncio.sleep(0)

        assert not any(
            item["type"] == "companion_message" for item in connection.sent
        )
        await companion.reset()

    asyncio.run(scenario())


def _timed_companion(*, stream_tts=None):
    connection = _Connection()
    spoken = []
    now = [0.0]

    async def record_tts(text, **kwargs):
        spoken.append((text, kwargs))
        if stream_tts is not None:
            return await stream_tts(text, **kwargs)
        return {"samples": 0, "started": False}

    companion = RealtimeCompanion(
        connection,
        session=_session(),
        patient_memory_service=None,
        stream_tts_audio=record_tts,
        config=RealtimeCompanionConfig(enabled=True, external_asr=True),
        classify_window=lambda *_args: {},
        now_factory=lambda: now[0],
        logger=lambda _message: None,
    )
    return SimpleNamespace(
        companion=companion, connection=connection, spoken=spoken, now=now
    )


@pytest.mark.parametrize(
    "seconds,speaking,detail,expected",
    [
        (4.96, True, "incomplete", None),
        (5.0, True, "incomplete", None),
        (5.16, True, "incomplete", "嗯嗯"),
        (8.0, True, "incomplete", "嗯嗯"),
        (5.16, False, "incomplete", None),
        (7.0, False, "incomplete", None),
        (7.04, False, "incomplete", "我在听，您慢慢说。"),
        (8.0, True, "listening", None),
        (8.0, False, "listening", None),
        (8.0, None, "incomplete", None),
        (8.0, False, "incomplete_timeout", None),
        (8.0, True, "incomplete_wait", "嗯嗯"),
        (8.0, False, "incomplete_wait", "我在听，您慢慢说。"),
    ],
)
def test_soulx_acknowledgement_requires_duration_incomplete_and_current_speech(
    seconds, speaking, detail, expected
):
    async def scenario():
        context = _timed_companion()
        companion = context.companion
        # Only the last native frame carries the semantic decision. Earlier
        # audio establishes the utterance without an incomplete acknowledgement.
        await companion.observe_soulx_frame(
            np.ones(round(seconds * 16000) - 2560, dtype=np.float32),
            state="nonidle", detail_state="listening", speech_detected=True,
        )
        await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="nonidle" if speaking else "idle",
            text="我想说的是", detail_state=detail, speech_detected=speaking,
        )
        await asyncio.sleep(0)
        assert [text for text, _kwargs in context.spoken] == (
            [] if expected is None else [expected]
        )
        assert all(kwargs["companion"] for _text, kwargs in context.spoken)
        await companion.reset()

    asyncio.run(scenario())


def test_unchanged_incomplete_is_rechecked_after_threshold_and_keeps_rate_limits():
    async def scenario():
        context = _timed_companion()
        companion = context.companion

        async def feed(samples, speaking):
            await companion.observe_soulx_frame(
                np.ones(samples, dtype=np.float32),
                state="nonidle" if speaking else "idle",
                text="我想说的是", detail_state="incomplete",
                speech_detected=speaking,
            )
            await asyncio.sleep(0)

        await feed(80000, True)
        assert not context.spoken
        await feed(2560, True)
        assert [item[0] for item in context.spoken] == ["嗯嗯"]
        # Crossing seven seconds during the existing cooldown does not chatter.
        context.now[0] = 9.9
        await feed(32000, False)
        assert len(context.spoken) == 1
        context.now[0] = 10.0
        await feed(2560, False)
        assert [item[0] for item in context.spoken] == ["嗯嗯", "我在听，您慢慢说。"]
        context.now[0] = 20.0
        await feed(2560, False)
        assert len(context.spoken) == 2
        await companion.reset()

    asyncio.run(scenario())


def test_idle_audio_and_elapsed_wall_time_do_not_count_toward_speech_duration():
    async def scenario():
        context = _timed_companion()
        companion = context.companion
        await companion.observe_soulx_frame(
            np.zeros(16000 * 30, dtype=np.float32),
            state="idle", speech_detected=False,
        )
        assert companion._active is None
        context.now[0] = 100.0
        await companion.observe_soulx_frame(
            np.ones(16000, dtype=np.float32),
            state="nonidle", detail_state="incomplete", speech_detected=True,
        )
        context.now[0] = 1000.0
        await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="idle", detail_state="incomplete", speech_detected=False,
        )
        await asyncio.sleep(0)
        assert not context.spoken
        assert companion._active.audio_duration_s == pytest.approx(1.16)
        await companion.reset()

    asyncio.run(scenario())


def test_complete_turn_drops_queued_acknowledgement_and_resets_duration():
    async def scenario():
        context = _timed_companion()
        companion = context.companion
        await companion.observe_soulx_frame(
            np.ones(16000 * 8, dtype=np.float32),
            state="nonidle", detail_state="incomplete", speech_detected=True,
        )
        await companion.observe_soulx_frame(
            np.zeros(2560, dtype=np.float32),
            state="speak", detail_state="incomplete_timeout", speech_detected=False,
            complete_audio=np.ones(16000 * 8 + 2560, dtype=np.float32),
        )
        await asyncio.sleep(0)
        assert not context.spoken
        assert companion._active is None
        context.now[0] = 20.0
        await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="nonidle", detail_state="incomplete", speech_detected=True,
        )
        await asyncio.sleep(0)
        assert not context.spoken
        assert companion._active.audio_duration_s == pytest.approx(0.16)
        await companion.reset()

    asyncio.run(scenario())


def test_resumed_speech_cancels_listening_prompt_but_continued_speech_keeps_backchannel():
    async def scenario(initially_speaking):
        started = asyncio.Event()

        async def stream_tts(_text, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        context = _timed_companion(stream_tts=stream_tts)
        companion = context.companion
        await companion.observe_soulx_frame(
            np.ones(16000 * 8, dtype=np.float32),
            state="nonidle", detail_state="incomplete",
            speech_detected=initially_speaking,
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        playback = companion._comfort_task
        await companion.observe_soulx_frame(
            np.ones(2560, dtype=np.float32),
            state="nonidle", detail_state="incomplete", speech_detected=True,
        )
        assert playback.cancelled() is not initially_speaking
        assert companion.backchannel_playing is initially_speaking
        assert len(context.spoken) == 1
        await companion.reset()

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


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


def _memory_turn(get_turn_context, *, now_factory, timeout_s=0.25):
    async def stream_tts(*args, **kwargs):
        return {"samples": 0}

    companion = RealtimeCompanion(
        _Connection(),
        session=_session(),
        patient_memory_service=SimpleNamespace(get_turn_context=get_turn_context),
        stream_tts_audio=stream_tts,
        config=RealtimeCompanionConfig(enabled=True, memory_timeout_s=timeout_s),
        now_factory=now_factory,
        logger=lambda message: None,
    )
    return RealtimeTurn(companion)


def test_memory_prefetch_starts_after_350ms_of_stable_text():
    async def scenario():
        now = [0.0]
        queries = []

        def retrieve(session, text):
            queries.append(text)
            return "记忆"

        turn = _memory_turn(retrieve, now_factory=lambda: now[0])
        text = "我又想起浴室滑倒"
        turn._partial_history.append((0.0, text))
        now[0] = 0.2
        turn._maybe_prefetch(text, now[0])
        assert turn._memory_task is None
        now[0] = 0.36
        turn._maybe_prefetch(text, now[0])
        assert turn._memory_task is not None
        await turn._memory_task
        assert queries == [text]
        await turn.aclose()

    asyncio.run(scenario())


def test_partial_updates_do_not_replace_an_inflight_memory_prefetch():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        now = [1.0]
        queries = []

        def retrieve(session, text):
            queries.append(text)
            started.set()
            assert release.wait(2)
            return "记忆"

        turn = _memory_turn(retrieve, now_factory=lambda: now[0])
        turn._partial_history.append((0.0, "我喜欢一个人洗澡"))
        turn._maybe_prefetch("我喜欢一个人洗澡", now[0])
        task = turn._memory_task
        assert await asyncio.to_thread(started.wait, 1)
        try:
            now[0] = 1.8
            turn._maybe_prefetch("我不喜欢一个人洗澡", now[0])
            assert turn._memory_task is task
            assert not task.cancelled()
            assert queries == ["我喜欢一个人洗澡"]
        finally:
            release.set()
        await task
        await turn.aclose()

    asyncio.run(scenario())


def test_memory_prefetch_is_rate_limited_after_a_completed_query():
    async def scenario():
        now = [1.0]
        queries = []

        def retrieve(session, text):
            queries.append(text)
            return text

        turn = _memory_turn(retrieve, now_factory=lambda: now[0])
        turn._partial_history.append((0.0, "我喜欢一个人洗澡"))
        turn._maybe_prefetch("我喜欢一个人洗澡", now[0])
        await turn._memory_task
        now[0] = 1.1
        turn._maybe_prefetch("我不喜欢一个人洗澡", now[0])
        assert queries == ["我喜欢一个人洗澡"]
        now[0] = 1.6
        turn._maybe_prefetch("我不喜欢一个人洗澡", now[0])
        await turn._memory_task
        assert queries == ["我喜欢一个人洗澡", "我不喜欢一个人洗澡"]
        await turn.aclose()

    asyncio.run(scenario())


def test_final_query_reuses_matching_prefetch_but_not_a_negated_query():
    async def scenario():
        queries = []

        def retrieve(session, text):
            queries.append(text)
            return f"检索：{text}"

        turn = _memory_turn(retrieve, now_factory=lambda: 1.0)
        await turn._start_memory_query("我喜欢一个人洗澡")
        same = await turn.memory_for_final("我喜欢一个人洗澡。")
        assert same == "检索：我喜欢一个人洗澡"
        assert len(queries) == 1
        changed = await turn.memory_for_final("我不喜欢一个人洗澡")
        assert changed == "检索：我不喜欢一个人洗澡"
        assert queries == ["我喜欢一个人洗澡", "我不喜欢一个人洗澡"]
        await turn.aclose()

    asyncio.run(scenario())


def test_disabled_session_memory_never_starts_prefetch_or_final_lookup():
    async def scenario():
        def retrieve(session, text):
            raise AssertionError("memory is disabled")

        turn = _memory_turn(retrieve, now_factory=lambda: 1.0)
        turn.companion.session.long_term_memory_enabled = False
        turn._partial_history.append((0.0, "我又想起浴室滑倒"))
        turn._maybe_prefetch("我又想起浴室滑倒", 1.0)
        assert turn._memory_task is None
        assert turn._start_memory_query("我又想起浴室滑倒") is None
        assert await turn.memory_for_final("我又想起浴室滑倒") == ""
        await turn.aclose()

    asyncio.run(scenario())


def test_late_memory_result_cannot_replace_a_newer_final_query():
    async def scenario():
        release = threading.Event()
        completed = threading.Event()

        def retrieve(session, text):
            if text == "旧查询":
                assert release.wait(2)
                completed.set()
                return "旧结果"
            return "新结果"

        turn = _memory_turn(retrieve, now_factory=lambda: 1.0, timeout_s=0.05)
        try:
            assert await turn.memory_for_final("旧查询") == ""
            assert await turn.memory_for_final("新查询") == "新结果"
        finally:
            release.set()
        assert await asyncio.to_thread(completed.wait, 1)
        await asyncio.sleep(0)
        assert turn._memory_result == "新结果"
        assert turn._memory_query_text == "新查询"
        await turn.aclose()

    asyncio.run(scenario())
