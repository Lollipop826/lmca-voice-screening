import ast
import asyncio
from datetime import datetime
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from src.voice import (
    AgentResumeStateService,
    PatientMemoryService,
    PatientProfileService,
    SpeechProcessingCoordinator,
    TTSPrewarmHandle,
    VoiceConnectionCleanup,
    VoiceSession,
    VoiceSessionBootstrap,
    VoiceTTSStreamer,
)


VOICE_SERVER_PATH = Path(__file__).resolve().parents[2] / "voice_server.py"
VOICE_APPLICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "voice"
    / "application.py"
)


def _voice_application_class():
    tree = ast.parse(
        VOICE_APPLICATION_PATH.read_text(encoding="utf-8"),
        filename=str(VOICE_APPLICATION_PATH),
    )
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VoiceEndpointApplication"
    )


class TTSPrewarmHandleTests(unittest.TestCase):
    def test_prewarm_task_is_consumed_once(self):
        async def scenario():
            calls = []

            class TTS:
                async def prewarm(self):
                    calls.append("prewarm")

            handle = TTSPrewarmHandle.start(TTS(), enabled=True)
            self.assertTrue(handle.pending)

            await handle.consume()
            await handle.consume()

            self.assertEqual(calls, ["prewarm"])
            self.assertFalse(handle.pending)

        asyncio.run(scenario())

    def test_disabled_prewarm_has_no_pending_task(self):
        handle = TTSPrewarmHandle.start(object(), enabled=False)

        self.assertFalse(handle.pending)
        asyncio.run(handle.consume())


class VoiceTTSStreamerTests(unittest.TestCase):
    @staticmethod
    def _build_streamer(chunks):
        class Connection:
            def __init__(self):
                self.sent = []

            async def send_json(self, payload):
                self.sent.append(payload)
                return True

        class TTS:
            async def text_to_speech_streaming(self, _text, **_kwargs):
                for chunk in chunks:
                    yield chunk

        class AudioStore:
            def __init__(self):
                self.persisted = []

            def persist_assistant(self, audio, content_text=""):
                result = {
                    "samples": len(audio),
                    "content_text": content_text,
                }
                self.persisted.append(
                    (np.asarray(audio).copy(), content_text)
                )
                return result

        connection = Connection()
        session = VoiceSession(
            connection=connection,
            agent=object(),
            owner_username="doctor",
        )
        audio_store = AudioStore()
        times = iter([10.0, 10.25, 10.5])
        streamer = VoiceTTSStreamer(
            connection,
            session=session,
            audio_store=audio_store,
            tts=TTS(),
            clean_for_tts=lambda text: text.strip(),
            now_factory=lambda: next(times),
            logger=lambda _message: None,
        )
        return connection, session, audio_store, streamer

    def test_stream_sends_chunks_and_persists_combined_audio(self):
        async def scenario():
            first = np.array([0.1, 0.2], dtype=np.float32)
            second = np.array([0.3], dtype=np.float32)
            connection, session, audio_store, streamer = (
                self._build_streamer([first, second])
            )

            result = await streamer.stream(
                " 您好 ",
                content_text="您好",
                emotion="neutral",
            )

            self.assertEqual(result["chunks"], 2)
            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["first_latency"], 0.25)
            self.assertFalse(result["interrupted"])
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["tts_chunk", "tts_chunk"],
            )
            self.assertEqual(connection.sent[0]["dtype"], "float32")
            np.testing.assert_allclose(
                audio_store.persisted[0][0],
                [0.1, 0.2, 0.3],
            )
            self.assertEqual(audio_store.persisted[0][1], "您好")
            self.assertFalse(session.runtime.ai_streaming_tts)

        asyncio.run(scenario())

    def test_interrupt_stops_before_sending_and_resets_runtime_flag(self):
        async def scenario():
            connection, session, audio_store, streamer = (
                self._build_streamer(
                    [np.ones(2, dtype=np.float32)]
                )
            )
            session.runtime.stop_generate = True

            result = await streamer.stream(
                "回复",
                allow_interrupt=True,
            )

            self.assertTrue(result["interrupted"])
            self.assertEqual(result["chunks"], 0)
            self.assertEqual(connection.sent, [])
            self.assertEqual(audio_store.persisted, [])
            self.assertFalse(session.runtime.ai_streaming_tts)

        asyncio.run(scenario())

    def test_stream_forwards_explicit_output_identity(self):
        async def scenario():
            connection, session, _audio_store, streamer = (
                self._build_streamer([np.ones(2, dtype=np.float32)])
            )
            session.bind_session("call-tts")

            await streamer.stream(
                "回复",
                session_id="call-tts",
                turn_id="turn_0007",
                generation=7,
                playback_id="playback-turn_0007-7",
            )

            payload = connection.sent[0]
            self.assertEqual(payload["session_id"], "call-tts")
            self.assertEqual(payload["turn_id"], "turn_0007")
            self.assertEqual(payload["generation"], 7)
            self.assertEqual(
                payload["playback_id"],
                "playback-turn_0007-7",
            )

        asyncio.run(scenario())

    def test_empty_clean_text_skips_tts(self):
        async def scenario():
            connection, session, audio_store, streamer = (
                self._build_streamer([])
            )

            result = await streamer.stream("   ")

            self.assertEqual(result["samples"], 0)
            self.assertIsNone(result["audio_data"])
            self.assertEqual(connection.sent, [])
            self.assertEqual(audio_store.persisted, [])
            self.assertFalse(session.runtime.ai_streaming_tts)

        asyncio.run(scenario())

    def test_higher_priority_stream_preempts_companion_without_waiting(self):
        async def scenario():
            companion_started = asyncio.Event()
            release_companion = asyncio.Event()

            class Connection:
                def __init__(self):
                    self.sent = []

                async def send_json(self, payload):
                    self.sent.append(payload)
                    return True

            class TTS:
                async def text_to_speech_streaming(self, text, **_kwargs):
                    if text == "安抚":
                        companion_started.set()
                        yield np.ones(2, dtype=np.float32)
                        await release_companion.wait()
                        yield np.ones(2, dtype=np.float32)
                    else:
                        yield np.ones(3, dtype=np.float32)

            connection = Connection()
            session = VoiceSession(
                connection=connection,
                agent=object(),
                owner_username="doctor",
            )
            streamer = VoiceTTSStreamer(
                connection,
                session=session,
                audio_store=SimpleNamespace(persist_assistant=lambda *_args, **_kwargs: None),
                tts=TTS(),
                clean_for_tts=lambda text: text,
                logger=lambda _message: None,
            )

            companion_task = asyncio.create_task(
                streamer.stream("安抚", companion=True, allow_interrupt=True)
            )
            await companion_started.wait()
            formal_result = await asyncio.wait_for(
                streamer.stream("正式回复"),
                timeout=0.2,
            )
            release_companion.set()
            companion_result = await companion_task

            self.assertFalse(formal_result["interrupted"])
            self.assertEqual(formal_result["chunks"], 1)
            self.assertTrue(companion_result["interrupted"])
            self.assertEqual(companion_result["interruption_reason"], "preempted")

        asyncio.run(scenario())

    def test_cancelled_companion_returns_its_started_playback(self):
        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            class Connection:
                def __init__(self):
                    self.sent = []

                async def send_json(self, payload):
                    self.sent.append(payload)
                    return True

            class TTS:
                async def text_to_speech_streaming(self, _text, **_kwargs):
                    started.set()
                    yield np.ones(2, dtype=np.float32)
                    await release.wait()

            connection = Connection()
            session = VoiceSession(
                connection=connection,
                agent=object(),
                owner_username="doctor",
            )
            streamer = VoiceTTSStreamer(
                connection,
                session=session,
                audio_store=SimpleNamespace(
                    persist_assistant=lambda *_args, **_kwargs: None
                ),
                tts=TTS(),
                clean_for_tts=lambda text: text,
                logger=lambda _message: None,
            )
            task = asyncio.create_task(
                streamer.stream(
                    "安抚",
                    companion=True,
                    allow_interrupt=True,
                    cancel_as_result=True,
                    session_id="call-1",
                    turn_id="turn-1",
                    generation=1,
                    playback_id="companion-1",
                    start_payload={"type": "tts_start", "text": "安抚"},
                )
            )
            await started.wait()
            task.cancel()
            result = await task

            self.assertTrue(result["started"])
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["interruption_reason"], "cancelled")
            self.assertEqual(
                result["output_identity"]["playback_id"], "companion-1"
            )
            release.set()

        asyncio.run(scenario())

    def test_rejected_lower_priority_output_does_not_emit_tts_start(self):
        async def scenario():
            safety_started = asyncio.Event()
            release_safety = asyncio.Event()

            class Connection:
                def __init__(self):
                    self.sent = []

                async def send_json(self, payload):
                    self.sent.append(payload)
                    return True

            class TTS:
                async def text_to_speech_streaming(self, text, **_kwargs):
                    if text == "安全提示":
                        safety_started.set()
                        yield np.ones(2, dtype=np.float32)
                        await release_safety.wait()
                    else:
                        yield np.ones(2, dtype=np.float32)

            connection = Connection()
            session = VoiceSession(
                connection=connection,
                agent=object(),
                owner_username="doctor",
            )
            streamer = VoiceTTSStreamer(
                connection,
                session=session,
                audio_store=SimpleNamespace(
                    persist_assistant=lambda *_args, **_kwargs: None
                ),
                tts=TTS(),
                clean_for_tts=lambda text: text,
                logger=lambda _message: None,
            )
            safety_task = asyncio.create_task(
                streamer.stream(
                    "安全提示",
                    priority=3,
                    start_payload={"type": "tts_start", "text": "安全提示"},
                )
            )
            await safety_started.wait()
            rejected = await streamer.stream(
                "普通提示",
                companion=True,
                start_payload={"type": "tts_start", "text": "普通提示"},
            )
            release_safety.set()
            await safety_task

            self.assertFalse(rejected["started"])
            self.assertEqual(rejected["interruption_reason"], "preempted")
            self.assertEqual(
                [event["text"] for event in connection.sent if event["type"] == "tts_start"],
                ["安全提示"],
            )

        asyncio.run(scenario())


class PatientProfileServiceTests(unittest.TestCase):
    def test_normalize_applies_defaults_aliases_and_location_place(self):
        service = PatientProfileService(logger=lambda _message: None)

        profile = service.normalize(
            {
                "name": " 张阿姨 ",
                "age": "72",
                "sex": "女",
                "education_years": "9",
                "hospital_name": "市一医院",
                "department": "神经内科",
                "bed_number": "12床",
            }
        )

        self.assertEqual(
            profile,
            {
                "name": "张阿姨",
                "age": 72,
                "gender": "女",
                "education_years": 9,
                "hospital_name": "市一医院",
                "department": "神经内科",
                "bed_number": "12床",
                "hospital": "市一医院",
                "place": "市一医院 神经内科 12床",
            },
        )

    def test_invalid_numeric_fields_use_voice_defaults(self):
        service = PatientProfileService(logger=lambda _message: None)

        profile = service.normalize(
            {"name": "患者", "age": "unknown", "education_years": None}
        )

        self.assertEqual(profile["age"], 70)
        self.assertEqual(profile["education_years"], 6)

    def test_normalize_preserves_selected_patient_id_for_binding(self):
        service = PatientProfileService(logger=lambda _message: None)

        profile = service.normalize(
            {"patient_id": " pt-existing ", "name": "张阿姨"}
        )

        self.assertEqual(profile["patient_id"], "pt-existing")


class VoiceSessionBootstrapTests(unittest.TestCase):
    @staticmethod
    def _build_bootstrap():
        class Connection:
            def __init__(self):
                self.sent = []

            async def send_json(self, payload):
                self.sent.append(payload)

        connection = Connection()
        session = VoiceSession(
            connection=connection,
            agent=object(),
            owner_username="doctor",
        )
        created = []
        bootstrap = VoiceSessionBootstrap(
            connection,
            session=session,
            create_session=lambda session_id, **kwargs: created.append(
                (session_id, kwargs)
            ),
            now_factory=lambda: datetime(2026, 7, 16, 9, 8, 7),
            token_factory=lambda: "deadbeef",
            logger=lambda _message: None,
        )
        return connection, session, created, bootstrap

    def test_create_fresh_persists_and_binds_unique_session(self):
        _connection, session, created, bootstrap = self._build_bootstrap()

        session_id = bootstrap.create_fresh()

        self.assertEqual(session_id, "call_20260716_090807_deadbeef")
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(
            created,
            [(session_id, {"owner_username": "doctor"})],
        )

    def test_send_waiting_for_info_uses_bound_session(self):
        async def scenario():
            connection, _session, _created, bootstrap = (
                self._build_bootstrap()
            )
            bootstrap.create_fresh()

            await bootstrap.send_waiting_for_info()

            self.assertEqual(
                connection.sent,
                [
                    {
                        "type": "waiting_for_info",
                        "session_id": "call_20260716_090807_deadbeef",
                        "message": (
                            "请先填写患者基本信息和当前聊天位置，"
                            "然后点击「开始聊天」"
                        ),
                    }
                ],
            )

        asyncio.run(scenario())

    def test_endpoint_has_no_nested_functions(self):
        tree = ast.parse(
            VOICE_SERVER_PATH.read_text(encoding="utf-8"),
            filename=str(VOICE_SERVER_PATH),
        )
        endpoint = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "websocket_endpoint"
        )

        nested_names = {
            node.name
            for node in endpoint.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertEqual(nested_names, set())


class VoiceConnectionCleanupTests(unittest.TestCase):
    def test_close_cancels_work_and_consolidates_patient_memory(self):
        async def scenario():
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor",
            )
            session.bind_session("call-cleanup")
            session.lifecycle.current_patient_id = "patient-1"
            session.chat_history.append(
                {"role": "user", "content": "今天星期三"}
            )
            session.processing.pending_audio = np.ones(
                4,
                dtype=np.float32,
            )
            session.processing.pending_source = "live"
            started = asyncio.Event()

            async def active_work():
                started.set()
                await asyncio.Event().wait()

            session.processing.task = asyncio.create_task(active_work())
            await started.wait()

            answer_completion = Mock()
            turn_client = SimpleNamespace(
                close=Mock(
                    side_effect=lambda: asyncio.sleep(0)
                )
            )
            patient_memory_service = Mock()
            messages = []
            ended = []
            cleanup = VoiceConnectionCleanup(
                session,
                answer_completion=answer_completion,
                turn_client=turn_client,
                patient_memory_service=patient_memory_service,
                end_session=lambda *args: ended.append(args),
                logger=messages.append,
            )

            await cleanup.close("client-1")

            self.assertTrue(session.runtime.stop_generate)
            self.assertIsNone(session.processing.pending_audio)
            self.assertEqual(session.processing.pending_source, "")
            self.assertTrue(session.processing.task.cancelled())
            answer_completion.cancel_window.assert_called_once_with()
            turn_client.close.assert_called_once_with()
            patient_memory_service.consolidate.assert_called_once_with(
                session
            )
            patient_memory_service.flush_session.assert_called_once_with(session)
            self.assertEqual(ended, [("call-cleanup", None, None)])
            self.assertTrue(session.lifecycle.sealed)
            self.assertIn(
                "[SoulX] 轮次连接已关闭: client-1",
                messages,
            )

        asyncio.run(scenario())

    def test_close_skips_memory_without_patient_history(self):
        async def scenario():
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor",
            )
            patient_memory_service = Mock()
            cleanup = VoiceConnectionCleanup(
                session,
                answer_completion=Mock(),
                patient_memory_service=patient_memory_service,
                logger=lambda _message: None,
            )

            await cleanup.close("client-2")

            patient_memory_service.consolidate.assert_not_called()

        asyncio.run(scenario())

    def test_close_seals_before_slow_external_cleanup_and_never_notifies_waiting(self):
        async def scenario():
            sent = []
            ended = []
            session = VoiceSession(
                connection=SimpleNamespace(
                    send_json=lambda payload: sent.append(payload)
                ),
                agent=object(),
                owner_username="doctor",
            )
            session.bind_session("call-disconnect")
            session.lifecycle.current_patient_id = "patient-1"

            async def close_turn_client():
                self.assertEqual(ended, [("call-disconnect", None, None)])

            cleanup = VoiceConnectionCleanup(
                session,
                answer_completion=Mock(),
                turn_client=SimpleNamespace(close=close_turn_client),
                patient_memory_service=Mock(),
                end_session=lambda *args: ended.append(args),
                logger=lambda _message: None,
            )

            await cleanup.close("client-3")

            self.assertTrue(session.lifecycle.sealed)
            self.assertEqual(sent, [])

        asyncio.run(scenario())

    def test_close_still_flushes_when_consolidation_fails(self):
        async def scenario():
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor",
            )
            session.bind_session("call-consolidation-failure")
            session.lifecycle.current_patient_id = "patient-1"
            patient_memory_service = Mock()
            patient_memory_service.consolidate.side_effect = RuntimeError(
                "consolidator unavailable"
            )
            cleanup = VoiceConnectionCleanup(
                session,
                answer_completion=Mock(),
                patient_memory_service=patient_memory_service,
                logger=lambda _message: None,
            )

            await cleanup.close("client-consolidation-failure")

            patient_memory_service.consolidate.assert_called_once_with(session)
            patient_memory_service.flush_session.assert_called_once_with(session)

        asyncio.run(scenario())


class AgentResumeStateServiceTests(unittest.TestCase):
    def test_score_payload_rebuilds_legacy_dimension_rows(self):
        payload = AgentResumeStateService.build_score_payload(
            {
                "orientation": {
                    "score": 4,
                    "max_score": 5,
                    "question": "今天星期几？",
                    "answer": "星期三",
                    "evaluation_detail": "答对日期",
                    "created_at": "2026-07-16",
                }
            }
        )

        self.assertEqual(payload["total_score"], 4)
        self.assertEqual(payload["completed_max_score"], 5)
        item = payload["scoring_details"]["dimension_scores"][
            "orientation"
        ]["items"][0]
        self.assertEqual(item["task_id"], "orientation")
        self.assertEqual(item["detail"], "答对日期")

    def test_sync_restores_tasks_history_and_clears_stale_prefetch(self):
        cleared = []
        state = SimpleNamespace(
            session_id=None,
            _active_session_id=None,
            session_data={},
            _task_done=set(),
            _last_generated_question="请开始评估",
            _precomputed_next_task="stale",
            _last_bridge_hint="stale",
            _last_bridge_topic="stale",
            _last_target_question="stale",
            _last_target_task_id="stale",
        )
        agent = SimpleNamespace(
            state=state,
            catalog=SimpleNamespace(
                task_config={
                    "orientation_time": {
                        "dimension_id": "orientation",
                        "max_points": 5,
                    },
                    "memory_recall": {
                        "dimension_id": "memory",
                        "max_points": 3,
                    },
                },
                visual_action_tasks={"copy_pentagons"},
            ),
            tool_gateway=SimpleNamespace(
                memory_tool=SimpleNamespace(
                    get_and_clear_suggestion=lambda: cleared.append(True)
                ),
                cancel_retrieval_prefetch=lambda: cleared.append("prefetch"),
            ),
        )
        service = AgentResumeStateService(
            agent,
            logger=lambda _message: None,
        )
        history = [
            {"role": "user", "content": "星期三"},
            {"role": "assistant", "content": "请记住三个词"},
        ]
        payload = {
            "scoring_details": {
                "dimension_scores": {
                    "orientation": {
                        "items": [
                            {"task_id": "orientation_time"}
                        ]
                    }
                }
            }
        }

        service.sync("call-resume", history, payload)

        self.assertEqual(agent.state.session_id, "call-resume")
        self.assertEqual(agent.state._active_session_id, "call-resume")
        self.assertIs(agent.state.session_data["chat_history"], history)
        self.assertEqual(
            agent.state._task_done,
            {"copy_pentagons", "orientation_time"},
        )
        self.assertEqual(
            agent.state._last_generated_question,
            "请记住三个词",
        )
        self.assertIsNone(agent.state._precomputed_next_task)
        self.assertIn(True, cleared)
        self.assertIn("prefetch", cleared)

    def test_endpoint_uses_profile_and_resume_services(self):
        application = _voice_application_class()
        nested_names = {
            node.name
            for node in ast.walk(application)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        constructed_names = {
            node.func.id
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }

        self.assertTrue(
            {
                "_build_resume_mmse_payload",
                "_sync_agent_resume_state",
                "_normalize_patient_profile",
                "_sync_manual_location_from_profile",
            }.isdisjoint(nested_names)
        )
        self.assertIn("PatientProfileService", constructed_names)
        self.assertIn("AgentResumeStateService", constructed_names)


class SpeechProcessingCoordinatorTests(unittest.TestCase):
    @staticmethod
    def _session():
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor",
        )
        session.bind_session("call-processing")
        return session

    def test_spawn_owns_processing_lifecycle_and_turn_identity(self):
        async def scenario():
            session = self._session()
            calls = []

            class Processor:
                async def process(self, audio, generation, turn_id, **kwargs):
                    calls.append(
                        (
                            np.asarray(audio).copy(),
                            generation,
                            turn_id,
                            kwargs,
                        )
                    )

            coordinator = SpeechProcessingCoordinator(
                session,
                processor=Processor(),
                enabled_full_duplex=True,
                is_ai_speaking=lambda: False,
                stop_playback=lambda: None,
                disconnect_error_type=RuntimeError,
                client_id="client-1",
                logger=lambda _message: None,
            )

            started = await coordinator.spawn(
                np.ones(1600, dtype=np.float32),
                source="测试语音",
                allow_revision=True,
                extra_meta={"source": "test"},
            )
            task = session.processing.task
            await task

            self.assertTrue(started)
            self.assertEqual(calls[0][1], 1)
            self.assertEqual(calls[0][2], "turn_0001")
            self.assertEqual(
                calls[0][3],
                {"extra_meta": {"source": "test"}},
            )
            self.assertFalse(session.processing.is_active)
            self.assertIsNone(session.processing.task)
            self.assertIsNone(session.processing.active_audio)

        asyncio.run(scenario())

    def test_multiple_revisions_replay_only_the_latest_audio(self):
        async def scenario():
            session = self._session()
            first_started = asyncio.Event()
            release_first = asyncio.Event()
            calls = []

            class Processor:
                async def process(self, audio, generation, turn_id, **kwargs):
                    calls.append(
                        (
                            np.asarray(audio).copy(),
                            generation,
                            turn_id,
                            kwargs.get("extra_meta"),
                        )
                    )
                    if len(calls) == 1:
                        first_started.set()
                        await release_first.wait()

            coordinator = SpeechProcessingCoordinator(
                session,
                processor=Processor(),
                enabled_full_duplex=True,
                is_ai_speaking=lambda: False,
                stop_playback=lambda: None,
                disconnect_error_type=RuntimeError,
                client_id="client-1",
                logger=lambda _message: None,
            )

            await coordinator.spawn(
                np.full(1600, 1.0, dtype=np.float32),
                source="第一版",
                allow_revision=True,
            )
            first_task = session.processing.task
            await first_started.wait()
            await coordinator.submit_or_queue(
                np.full(1600, 2.0, dtype=np.float32),
                source="第二版",
                extra_meta={"soulx_text": "第二版"},
            )
            await coordinator.submit_or_queue(
                np.full(1600, 3.0, dtype=np.float32),
                source="第三版",
                extra_meta={"soulx_text": "第三版"},
            )
            release_first.set()
            await first_task
            replay_task = session.processing.task
            if replay_task is not None:
                await replay_task

            self.assertEqual(len(calls), 2)
            np.testing.assert_allclose(calls[0][0], 1.0)
            np.testing.assert_allclose(calls[1][0], 3.0)
            self.assertEqual([call[1] for call in calls], [1, 2])
            self.assertEqual(
                [call[2] for call in calls],
                ["turn_0001", "turn_0002"],
            )
            self.assertEqual(calls[1][3], {"soulx_text": "第三版"})
            self.assertIsNone(session.processing.pending_audio)
            self.assertIsNone(session.processing.pending_extra_meta)

        asyncio.run(scenario())

    def test_interrupt_marks_active_generation_superseded(self):
        async def scenario():
            session = self._session()
            session.processing.is_active = True
            session.processing.revision_enabled = True
            stopped = []

            async def stop_playback():
                stopped.append(True)

            coordinator = SpeechProcessingCoordinator(
                session,
                processor=SimpleNamespace(process=None),
                enabled_full_duplex=True,
                is_ai_speaking=lambda: True,
                stop_playback=stop_playback,
                disconnect_error_type=RuntimeError,
                client_id="client-1",
                logger=lambda _message: None,
            )

            await coordinator.interrupt_active("收到新回答")

            self.assertTrue(session.runtime.stop_generate)
            self.assertFalse(session.processing.revision_enabled)
            self.assertEqual(stopped, [True])

        asyncio.run(scenario())

    def test_endpoint_uses_coordinator_instead_of_processing_closures(self):
        application = _voice_application_class()
        nested_names = {
            node.name
            for node in ast.walk(application)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        coordinator_constructions = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SpeechProcessingCoordinator"
        ]

        self.assertTrue(
            {
                "_spawn_process_speech",
                "_submit_or_queue_speech",
                "_interrupt_active_processing",
                "_queue_pending_processing",
            }.isdisjoint(nested_names)
        )
        self.assertEqual(len(coordinator_constructions), 1)


class PatientMemoryServiceTests(unittest.TestCase):
    def _session(self) -> VoiceSession:
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor-a",
        )
        session.bind_session("call-1")
        session.patient_profile.update(
            {"name": "张阿姨", "age": 72, "gender": "女"}
        )
        return session

    def test_mmse_sync_only_runs_for_cognitive_screening_sessions(self):
        memory = SimpleNamespace(update_mmse_score=Mock())
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=memory,
            logger=Mock(),
        )
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        session.mode = "wellbeing"

        self.assertFalse(service.update_mmse_score(session, 24, ["recall"]))
        memory.update_mmse_score.assert_not_called()

        session.mode = "cognitive_screening"
        self.assertTrue(service.update_mmse_score(session, 24, ["recall"]))
        memory.update_mmse_score.assert_called_once_with(
            "pt-1",
            24,
            ["recall"],
        )

    def test_existing_patient_is_refreshed_linked_and_loaded(self):
        session = self._session()
        update_profile = Mock()
        link_session = Mock()
        assign_patient = Mock()
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=update_profile,
            link_session_patient=link_session,
            assign_patient=assign_patient,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, " pt-existing ")

        self.assertEqual(context.patient_id, "pt-existing")
        self.assertIn("【患者基本信息】姓名：张阿姨", context.memory_card)
        self.assertFalse(context.has_history)
        self.assertEqual(session.lifecycle.current_patient_id, "pt-existing")
        update_profile.assert_called_once_with(
            "pt-existing",
            {"name": "张阿姨", "age": 72, "gender": "女"},
        )
        link_session.assert_called_once_with("call-1", "pt-existing")
        assign_patient.assert_not_called()

    def test_disabled_long_term_memory_keeps_identity_and_current_session_only(self):
        session = self._session()
        session.chat_history.append({"role": "user", "content": "本轮刚说过的话"})
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=None,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-1")

        self.assertEqual(context.patient_id, "pt-1")
        self.assertIn("【患者基本信息】姓名：张阿姨", context.memory_card)
        self.assertFalse(context.has_history)
        self.assertIsNone(session.memory_engine)
        self.assertIn("本轮刚说过的话", service.get_turn_context(session, "现在的话"))
        self.assertFalse(service.capture_turn(session, "用户", "助手"))
        self.assertFalse(service.update_turn_emotion(session, "turn-1", {"calm": 1.0}))
        self.assertFalse(service.update_turn_status(session, "turn-1", turn_state="RESPONDED"))
        self.assertFalse(service.flush_session(session))
        self.assertFalse(service.consolidate(session))

    def test_session_can_disable_configured_long_term_memory(self):
        session = self._session()
        session.long_term_memory_enabled = False
        session.chat_history.append({"role": "user", "content": "本轮内容"})
        memory = Mock()
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=memory,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-1")

        self.assertFalse(context.has_history)
        self.assertIsNone(session.memory_engine)
        self.assertIn("本轮内容", service.get_turn_context(session, "现在的话"))
        self.assertFalse(service.capture_turn(session, "用户", "助手"))
        self.assertFalse(service.consolidate(session))
        memory.get_snapshot.assert_not_called()
        memory.get_relevant_evidence.assert_not_called()
        memory.capture_turn.assert_not_called()

    def test_session_can_freeze_writes_without_disabling_reads(self):
        session = self._session()
        session.long_term_memory_writes_enabled = False
        memory = Mock()
        memory.get_snapshot.return_value = {"facts": [{"value": "喜欢京剧"}]}
        memory.get_relevant_evidence.return_value = "喜欢京剧"
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=memory,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-1")

        self.assertTrue(context.has_history)
        self.assertIn("喜欢京剧", service.get_turn_context(session, "最近听什么"))
        self.assertFalse(service.capture_turn(session, "用户", "助手"))
        self.assertFalse(service.update_turn_emotion(session, "turn-1", {"calm": 1.0}))
        self.assertFalse(service.update_turn_status(session, "turn-1", turn_state="RESPONDED"))
        self.assertFalse(service.flush_session(session))
        self.assertFalse(service.consolidate(session))
        memory.get_relevant_evidence.assert_called_once()
        memory.capture_turn.assert_not_called()

    def test_missing_requested_patient_is_created_and_assigned_to_owner(self):
        session = self._session()
        create_patient = Mock(return_value={"patient_id": "pt-new"})
        assign_patient = Mock()
        service = PatientMemoryService(
            get_patient=lambda patient_id: None,
            create_patient=create_patient,
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            assign_patient=assign_patient,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "missing")

        self.assertEqual(context.patient_id, "pt-new")
        create_patient.assert_called_once_with(
            {"name": "张阿姨", "age": 72, "gender": "女"}
        )
        assign_patient.assert_called_once_with(
            "pt-new",
            "doctor-a",
            can_read=True,
            can_write=True,
            can_voice=True,
            assigned_by="doctor-a",
        )

    def test_new_patient_skips_assignment_without_owner(self):
        session = self._session()
        session.owner_username = ""
        assign_patient = Mock()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(return_value={"patient_id": "pt-new"}),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            assign_patient=assign_patient,
            logger=Mock(),
        )

        context = service.resolve_for_session(session)

        self.assertEqual(context.patient_id, "pt-new")
        assign_patient.assert_not_called()

    def test_empty_profile_does_not_create_or_link_patient(self):
        session = self._session()
        session.patient_profile.clear()
        create_patient = Mock()
        link_session = Mock()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=create_patient,
            update_patient_profile=Mock(),
            link_session_patient=link_session,
            logger=Mock(),
        )

        context = service.resolve_for_session(session)

        self.assertIsNone(context.patient_id)
        self.assertEqual(context.memory_card, "")
        self.assertIsNone(session.lifecycle.current_patient_id)
        create_patient.assert_not_called()
        link_session.assert_not_called()

    def test_existing_patient_hydrates_an_empty_connection_profile(self):
        session = self._session()
        session.patient_profile.clear()
        update_profile = Mock()
        link_session = Mock()
        service = PatientMemoryService(
            get_patient=lambda _patient_id: {
                "patient_id": "pt-existing",
                "name": "张阿姨",
                "age": 72,
                "gender": "女",
                "education_years": 6,
                "extra_profile": {"city": "杭州"},
            },
            create_patient=Mock(),
            update_patient_profile=update_profile,
            link_session_patient=link_session,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-existing")

        self.assertEqual(
            session.patient_profile,
            {
                "name": "张阿姨",
                "age": 72,
                "gender": "女",
                "education_years": 6,
                "city": "杭州",
            },
        )
        self.assertIn("【患者基本信息】姓名：张阿姨", context.memory_card)
        update_profile.assert_not_called()
        link_session.assert_called_once_with("call-1", "pt-existing")

    def test_consolidate_schedules_a_snapshot_with_mmse_summary(self):
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        session.chat_history.extend(
            [{"role": "user", "content": "今天晴天"}]
        )
        long_term_memory = Mock()
        long_term_memory.consolidate_pending_turns.return_value = {
            "consolidated": True,
        }
        called = threading.Event()
        long_term_memory.consolidate_pending_turns.side_effect = (
            lambda *_args, **_kwargs: called.set()
        )
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        result = service.consolidate(
            session,
            total_mmse_score=24,
            cognitive_status="需复查",
        )

        self.assertTrue(result)
        self.assertTrue(called.wait(1))
        long_term_memory.consolidate_pending_turns.assert_called_once_with(
            "pt-1", session_id="call-1"
        )

    def test_consolidate_runs_one_shadow_reflection_after_snapshot(self):
        class Memory:
            def __init__(self):
                self.calls = []
                self.done = threading.Event()

            def consolidate_pending_turns(self, patient_id, *, session_id):
                self.calls.append(("consolidate", patient_id, session_id))

            def reflect_session(self, patient_id, session_id):
                self.calls.append(("reflect", patient_id, session_id))
                self.done.set()

        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        memory = Memory()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=memory,
            logger=Mock(),
        )

        self.assertTrue(service.consolidate(session))
        self.assertTrue(memory.done.wait(1))
        self.assertEqual(
            memory.calls,
            [("consolidate", "pt-1", "call-1"), ("reflect", "pt-1", "call-1")],
        )

    def test_consolidate_returns_without_waiting_for_reflection(self):
        class Memory:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def consolidate_pending_turns(self, patient_id, *, session_id):
                return {"consolidated": True}

            def reflect_session(self, patient_id, session_id):
                self.started.set()
                self.release.wait(1)

        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        memory = Memory()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=memory,
            logger=Mock(),
        )
        returned = threading.Event()
        result = []

        def call_consolidate():
            result.append(service.consolidate(session))
            returned.set()

        worker = threading.Thread(target=call_consolidate, daemon=True)
        worker.start()
        try:
            self.assertTrue(returned.wait(0.5))
            self.assertTrue(memory.started.wait(1))
        finally:
            memory.release.set()
            worker.join(1)

        self.assertEqual(result, [True])

    def test_consolidate_skips_unbound_or_empty_sessions(self):
        session = self._session()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            logger=Mock(),
        )

        self.assertFalse(service.consolidate(session))
        session.lifecycle.current_patient_id = "pt-1"
        self.assertFalse(service.consolidate(session))

    def test_capture_turn_forwards_audio_path_to_long_term_memory(self):
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        long_term_memory = Mock()
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        result = service.capture_turn(
            session,
            "我最近有点焦虑。",
            "我们慢慢聊。",
            {"anxiety": 1.0},
            audio_path="data/voice_calls/user.wav",
        )

        self.assertTrue(result)
        long_term_memory.capture_turn.assert_called_once_with(
            "pt-1",
            "我最近有点焦虑。",
            "我们慢慢聊。",
            session_id="call-1",
            emotion={"anxiety": 1.0},
            audio_path="data/voice_calls/user.wav",
        )

    def test_online_memobase_keeps_static_background_to_basic_profile(self):
        session = self._session()
        long_term_memory = Mock()
        long_term_memory.get_snapshot.return_value = {
            "recent_turns": [{"user_message": "旧对话"}]
        }
        long_term_memory.get_context_for_llm.return_value = "不应长期挂载的旧记忆"
        long_term_memory.is_memobase_available.return_value = True
        long_term_memory.replay_unsynced.return_value = True
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-1")

        self.assertTrue(context.has_history)
        self.assertIn("【患者基本信息】姓名：张阿姨", context.memory_card)
        self.assertNotIn("不应长期挂载", context.memory_card)

    def test_offline_legacy_memory_card_is_wrapped_without_outer_truncation(self):
        session = self._session()
        long_term_memory = Mock()
        long_term_memory.get_snapshot.return_value = {
            "recent_turns": [{"user_message": "旧对话"}]
        }
        long_term_memory.get_context_for_llm.return_value = "旧" * 2000
        long_term_memory.is_memobase_available.return_value = False
        service = PatientMemoryService(
            get_patient=lambda patient_id: {"patient_id": patient_id},
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        context = service.resolve_for_session(session, "pt-1")

        self.assertTrue(context.has_history)
        self.assertIn("旧" * 2000, context.memory_card)
        long_term_memory.replay_unsynced.assert_not_called()

    def test_turn_context_and_flush_are_forwarded_for_bound_patient(self):
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        long_term_memory = Mock()
        long_term_memory.get_relevant_evidence.return_value = "浴室滑倒"
        long_term_memory.flush.return_value = True
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        turn_context = service.get_turn_context(session, "我不敢一个人洗澡")
        self.assertIn("[current_session_memory]", turn_context)
        self.assertIn("[retrieved_long_term_memory]", turn_context)
        self.assertIn("浴室滑倒", turn_context)
        self.assertIn("[current_user_text]", turn_context)
        self.assertTrue(service.flush_session(session))
        long_term_memory.get_relevant_evidence.assert_called_once_with(
            "pt-1",
            "我不敢一个人洗澡",
            exclude_session_id="call-1",
        )
        long_term_memory.flush.assert_called_once_with("pt-1")

    def test_turn_context_does_not_fallback_to_sqlite_when_semantic_lookup_is_empty(self):
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        long_term_memory = Mock()
        long_term_memory.get_relevant_evidence.return_value = ""
        long_term_memory.get_cross_session_context.return_value = "跨会话 SQLite 证据"
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            logger=Mock(),
        )

        context = service.get_turn_context(session, "当前说的话")

        self.assertIn("无通过阈值的跨会话长期事件", context)
        self.assertNotIn("跨会话 SQLite 证据", context)
        long_term_memory.get_relevant_context.assert_not_called()
        long_term_memory.get_cross_session_context.assert_not_called()

    def test_read_only_long_term_memory_retrieves_without_writing(self):
        session = self._session()
        session.lifecycle.current_patient_id = "pt-1"
        long_term_memory = Mock()
        long_term_memory.get_relevant_evidence.return_value = "过去提到过睡眠困难"
        service = PatientMemoryService(
            get_patient=Mock(),
            create_patient=Mock(),
            update_patient_profile=Mock(),
            link_session_patient=Mock(),
            long_term_memory=long_term_memory,
            long_term_memory_writes=False,
            logger=Mock(),
        )

        context = service.get_turn_context(session, "最近还是睡不好")

        self.assertIn("过去提到过睡眠困难", context)
        self.assertFalse(service.capture_turn(session, "用户", "助手"))
        self.assertFalse(service.update_turn_emotion(session, "turn-1", {"calm": 1.0}))
        self.assertFalse(service.update_turn_status(session, "turn-1", turn_state="RESPONDED"))
        self.assertFalse(service.flush_session(session))
        self.assertFalse(service.consolidate(session))
        long_term_memory.capture_turn.assert_not_called()
        long_term_memory.flush.assert_not_called()


class PatientMemoryServiceEndpointWiringTests(unittest.TestCase):
    def test_voice_server_uses_service_instead_of_patient_memory_functions(self):
        server_tree = ast.parse(
            VOICE_SERVER_PATH.read_text(encoding="utf-8"),
            filename=str(VOICE_SERVER_PATH),
        )
        top_level_functions = {
            node.name
            for node in server_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertNotIn("_resolve_patient_for_session", top_level_functions)
        self.assertNotIn("_consolidate_patient_memory_async", top_level_functions)

        application = _voice_application_class()
        lifecycle_injections = [
            keyword
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SessionLifecycleHandler"
            for keyword in node.keywords
            if keyword.arg == "patient_memory_service"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "self"
            and keyword.value.attr == "patient_memory_service"
        ]
        cleanup_injections = [
            keyword
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VoiceConnectionCleanup"
            for keyword in node.keywords
            if keyword.arg == "patient_memory_service"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "self"
            and keyword.value.attr == "patient_memory_service"
        ]

        self.assertEqual(len(lifecycle_injections), 1)
        self.assertEqual(len(cleanup_injections), 1)


if __name__ == "__main__":
    unittest.main()
