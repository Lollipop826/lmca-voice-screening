"""Tests for object-oriented voice message handlers."""

import ast
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import threading
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.tools.voice.soulx_turn_taking import SoulXAudioAccumulator, SoulXTurnState
from src.voice.handlers.live_audio_frame_context import LiveAudioFrameContext
from src.voice.services import SpeechProcessingCoordinator
from src.voice.handlers import (
    AgentOutputPresenter,
    AnswerCompletionMessageHandler,
    ClientControlHandler,
    LiveAudioConfig,
    LiveAudioInputHandler,
    ManualAudioBlobProcessor,
    ManualAudioInputHandler,
    ManualInterruptHandler,
    MessageHandlerAction,
    SessionLifecycleHandler,
    SessionResumeHandler,
    SoulXAudioHandler,
    SpeechTurnConfig,
    SpeechTurnProcessor,
    SpeakerVerificationController,
    TextTurnHandler,
    VisionMessageHandler,
    VoiceFeedbackPresenter,
    VoiceInterruptionController,
    VoiceMessageRouter,
)
from src.voice.answer_completion import AnswerCompletionState
from src.voice.realtime_companion import RealtimeCompanion, RealtimeCompanionConfig, RealtimeTurn
from src.voice.session import VoiceSession, VoiceTurnTakingState


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


class _FakeConnection:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)
        return True


class _FakeVerifier:
    def __init__(self):
        self.is_enrolled = False
        self.threshold = 0.36
        self.speaker_name = None
        self.last_error = None
        self.verify_result = (True, 0.9)
        self.load_result = True
        self.save_result = True
        self.add_result = (True, 1, 8)
        self.reset_count = 0
        self.added_paths = []

    def reset(self):
        self.reset_count += 1
        self.is_enrolled = False

    def save(self, name):
        self.speaker_name = name
        return self.save_result

    def load(self, name):
        if self.load_result:
            self.speaker_name = name
            self.is_enrolled = True
        return self.load_result

    def add_sample(self, path):
        self.added_paths.append(path)
        success, current, needed = self.add_result
        if success and current >= needed:
            self.is_enrolled = True
        return self.add_result

    def verify_from_audio(self, _audio, _sample_rate):
        return self.verify_result


class _FakeManifestStore:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class _FakeHistoryStore:
    def __init__(self):
        self.appended = []

    async def append(self, *args):
        self.appended.append(args)


class _FakeVisionAgent:
    def __init__(self):
        self.dimension_map = {"language": {"name": "语言"}}
        self.state = SimpleNamespace(
            session_id=None,
            session_data={},
            current_dimension=None,
            is_in_comfort_mode=False,
            comfort_turn_count=0,
            _last_task_id=None,
            _last_cognitive_task_id=None,
            _last_generated_question="请开始评估",
            _last_forced_task_id=None,
            _pending_consent_task_id=None,
            _consent_granted_task_id=None,
            _buffer_resume_task_id=None,
            _asked_questions=[],
        )
        self.conversation_policy = SimpleNamespace(
            _get_full_name=self._get_full_name,
        )
        self.task_planner = SimpleNamespace(
            _set_bridge_context=self._set_bridge_context,
        )
        self.process_calls = []

    @staticmethod
    def _get_full_name(profile):
        return profile.get("name") or ""

    def _set_bridge_context(self, *_args, **_kwargs):
        return None

    def process_turn(self, **kwargs):
        self.process_calls.append(kwargs)
        return {"output": "好的，我们继续。"}


class _FakeTextAgent:
    def __init__(
        self,
        *,
        result=None,
        streamed_sentences=None,
        error=None,
    ):
        self.result = result or {"output": "请继续回答。"}
        self.streamed_sentences = list(streamed_sentences or [])
        self.error = error
        self.process_calls = []
        self.background_calls = []
        self.memory_backgrounds = []
        self.turn_background_at_process = None
        self.state = SimpleNamespace(_stream_sentence_cb=None)
        self.tool_gateway = SimpleNamespace(
            mmse_tool=SimpleNamespace(
                _run=lambda **_kwargs: json.dumps(
                    {
                        "success": True,
                        "total_score": 21,
                        "total_max_score": 30,
                    }
                )
            ),
            memory_tool=SimpleNamespace(
                set_turn_background=self.memory_backgrounds.append,
            ),
        )
        self.background_analysis = SimpleNamespace(
            analyze=self._background_analysis,
        )

    def process_turn(self, **kwargs):
        self.process_calls.append(kwargs)
        if self.memory_backgrounds:
            self.turn_background_at_process = self.memory_backgrounds[-1]
        if self.error:
            raise self.error
        for sentence in self.streamed_sentences:
            if self.state._stream_sentence_cb:
                self.state._stream_sentence_cb(sentence)
        return dict(self.result)

    async def _background_analysis(self, topic, history):
        self.background_calls.append((topic, list(history)))


class _FakeAudioStore:
    def __init__(self):
        self.persisted = []
        self.user_persisted = []

    def persist_assistant(self, audio_data, content_text=""):
        self.persisted.append(
            (np.asarray(audio_data).copy(), content_text)
        )

    def persist_user(self, audio_data, asr_text="", extra_meta=None):
        result = {
            "role": "user",
            "asr_text": asr_text,
            "extra_meta": extra_meta or {},
        }
        self.user_persisted.append(
            (np.asarray(audio_data).copy(), result)
        )
        return result


class _FakeSoulXAccumulator:
    def __init__(self, *, complete_audio=None, tail_audio=None):
        self.complete_audio = complete_audio
        self._tail_audio = (
            np.asarray(tail_audio, dtype=np.float32)
            if tail_audio is not None
            else np.array([], dtype=np.float32)
        )
        self.active = False
        self.reset_count = 0
        self.feed_calls = []

    def feed(self, audio, state):
        self.feed_calls.append((np.asarray(audio).copy(), state))
        self.active = state == "nonidle"
        return self.complete_audio if state == "speak" else None

    def tail_audio(self, _seconds):
        return self._tail_audio.copy()

    def reset(self):
        self.reset_count += 1
        self.active = False


class _FakeSoulXSpeaker:
    def __init__(self):
        self.enabled = False
        self.enrolled = False
        self.verifier = None
        self.verify_result = (True, 0.9)
        self.verify_calls = []
        self.diagnostic_count = 0
        self.warning_count = 0

    def log_diagnostic(self):
        self.diagnostic_count += 1

    async def warn_if_disabled(self):
        self.warning_count += 1
        return True

    async def verify(self, audio, sample_rate, **kwargs):
        self.verify_calls.append(
            (np.asarray(audio).copy(), sample_rate, kwargs)
        )
        return self.verify_result


class _FakePatientMemoryService:
    def __init__(self):
        self.resolved = []
        self.consolidated = []
        self.flushed = []

    def resolve_for_session(self, session, patient_id):
        self.resolved.append((session.session_id, patient_id))
        session.lifecycle.current_patient_id = patient_id or "patient-new"
        return SimpleNamespace(
            patient_id=session.lifecycle.current_patient_id,
            memory_card="既往记忆",
            has_history=True,
        )

    def consolidate(self, session, **kwargs):
        self.consolidated.append(
            {
                "session_id": session.session_id,
                "chat_history": list(session.chat_history),
                **kwargs,
            }
        )

    def flush_session(self, session):
        self.flushed.append(session.session_id)
        return True


class VoiceMessageRouterTests(unittest.TestCase):
    def test_dispatches_registered_type_and_leaves_unknown_type_unhandled(self):
        async def scenario():
            received = []
            router = VoiceMessageRouter()

            async def handle(message):
                received.append(message)

            router.register("known", handle)

            self.assertTrue(await router.dispatch({"type": "known", "id": 1}))
            self.assertFalse(await router.dispatch({"type": "unknown"}))
            self.assertEqual(received, [{"type": "known", "id": 1}])

        asyncio.run(scenario())

    def test_duplicate_registration_is_rejected(self):
        router = VoiceMessageRouter()
        router.register("ping", lambda _message: None)

        with self.assertRaisesRegex(ValueError, "already registered"):
            router.register("ping", lambda _message: None)

    def test_stop_action_is_exposed_to_the_connection_loop(self):
        async def scenario():
            router = VoiceMessageRouter()

            async def stop(_message):
                return MessageHandlerAction.STOP_CONNECTION

            router.register("stop", stop)
            result = await router.dispatch({"type": "stop"})

            self.assertTrue(result)
            self.assertTrue(result.stop_connection)

        asyncio.run(scenario())


class ClientControlHandlerTests(unittest.TestCase):
    def test_playback_stop_ack_logs_only_numeric_timing_fields(self):
        async def scenario():
            logs = []
            handler = ClientControlHandler(_FakeConnection(), logger=logs.append)
            await handler.handle_message({
                "type": "client_diag", "area": "tts", "label": "playback-stopped",
                "detail": json.dumps({
                    "server_stop_at_ms": 1000.0, "client_received_at_ms": 1050,
                    "client_stopped_at_ms": 1052, "stop_handler_ms": 2,
                    "generation": 3, "text": "must not be logged",
                }),
            })
            self.assertEqual(len(logs), 1)
            self.assertIn('"stop_handler_ms": 2', logs[0])
            self.assertIn('"server_ack_at_ms":', logs[0])
            self.assertNotIn("must not be logged", logs[0])
        asyncio.run(scenario())

    def test_ping_and_diagnostics_are_handled_without_session_state(self):
        async def scenario():
            connection = _FakeConnection()
            logs = []
            handler = ClientControlHandler(connection, logger=logs.append)

            await handler.handle_message({"type": "ping"})
            await handler.handle_message(
                {
                    "type": "client_diag",
                    "area": "audio",
                    "label": "microphone-ready",
                    "seq": 3,
                    "detail": "ok",
                }
            )
            await handler.handle_message(
                {
                    "type": "client_diag",
                    "area": "tts",
                    "label": "ws:tts_chunk:received",
                }
            )

            self.assertEqual(connection.sent, [{"type": "pong"}])
            self.assertEqual(len(logs), 1)
            self.assertIn("microphone-ready", logs[0])

        asyncio.run(scenario())

    def test_tts_exception_diagnostic_logs_sanitized_detail(self):
        async def scenario():
            connection = _FakeConnection()
            logs = []
            handler = ClientControlHandler(connection, logger=logs.append)

            await handler.handle_message(
                {
                    "type": "client_diag",
                    "area": "tts",
                    "label": "player:addChunk:exception",
                    "seq": 9,
                    "detail": "missing audio\nchunk",
                }
            )

            self.assertEqual(len(logs), 1)
            self.assertIn("detail=missing audio chunk", logs[0])

        asyncio.run(scenario())

    def test_location_update_persists_and_refreshes_cached_location(self):
        async def scenario(config_path):
            connection = _FakeConnection()
            cached = []
            handler = ClientControlHandler(
                connection,
                config_path=config_path,
                reverse_geocode=lambda _lat, _lon: ("浙江省", "杭州市"),
                location_loader=lambda: {
                    "province": "浙江省",
                    "city": "杭州市",
                },
                cache_updater=cached.append,
                logger=lambda _message: None,
            )

            handled = await handler.handle_message(
                {
                    "type": "update_location",
                    "latitude": 30.2,
                    "longitude": 120.1,
                }
            )

            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(handled)
            self.assertEqual(
                saved["location"],
                {
                    "lat": 30.2,
                    "lon": 120.1,
                    "province": "浙江省",
                    "city": "杭州市",
                    "source": "browser-geolocation",
                },
            )
            self.assertEqual(
                cached,
                [{"province": "浙江省", "city": "杭州市"}],
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "deployment.json"
            config_path.write_text(
                json.dumps({"location": {}}),
                encoding="utf-8",
            )
            asyncio.run(scenario(config_path))


class SessionResumeHandlerTests(unittest.TestCase):
    def test_resume_restores_session_history_profile_score_and_agent_state(self):
        async def scenario():
            connection = _FakeConnection()

            class MmseTool:
                def _run(self, **_kwargs):
                    return json.dumps(
                        {
                            "success": True,
                            "total_score": 18,
                            "scoring_details": {
                                "dimension_scores": {"orientation": 5}
                            },
                        }
                    )

            agent = SimpleNamespace(
                tool_gateway=SimpleNamespace(mmse_tool=MmseTool())
            )
            session = VoiceSession(
                connection=connection,
                agent=agent,
                owner_username="doctor",
            )
            session.mode = "cognitive_screening"
            manifest_store = _FakeManifestStore()
            assigned = []
            synced_locations = []
            synced_agents = []
            reset_reasons = []
            resume_data = {
                "ended_at": None,
                "mode": "cognitive_screening",
                "profile": {"name": "患者甲", "age": 72},
                "chat_history": [
                    {"role": "user", "content": "您好"},
                    {"role": "assistant", "content": "您好"},
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "您好",
                        "emotion": "neutral",
                    },
                    {
                        "role": "assistant",
                        "content": "您好",
                        "emotion": None,
                    },
                ],
                "mmse_scores": {"orientation": 5},
            }

            async def reset_turn_session(reason):
                reset_reasons.append(reason)

            handler = SessionResumeHandler(
                connection,
                session=session,
                manifest_store=manifest_store,
                get_resume=lambda _session_id: resume_data,
                assign_session_owner=lambda *args, **kwargs: assigned.append(
                    (args, kwargs)
                ),
                normalize_profile=lambda profile: {
                    **profile,
                    "education_years": 6,
                },
                sync_manual_location=synced_locations.append,
                build_score_payload=lambda scores: {"fallback": scores},
                sync_agent_state=lambda *args: synced_agents.append(args),
                reset_turn_session=reset_turn_session,
                logger=lambda _message: None,
            )

            handled = await handler.handle_message(
                {"type": "resume_session", "session_id": "call-1"}
            )

            self.assertTrue(handled)
            self.assertEqual(session.session_id, "call-1")
            self.assertEqual(session.patient_profile["name"], "患者甲")
            self.assertEqual(session.patient_profile["education_years"], 6)
            self.assertEqual(len(session.chat_history), 2)
            self.assertTrue(session.lifecycle.greeting_sent)
            self.assertEqual(manifest_store.refresh_count, 1)
            self.assertEqual(assigned[0][0], ("call-1", "doctor"))
            self.assertEqual(assigned[0][1], {"overwrite": False})
            self.assertEqual(len(synced_locations), 1)
            self.assertEqual(synced_agents[0][0], "call-1")
            self.assertEqual(synced_agents[0][2]["total_score"], 18)
            self.assertEqual(reset_reasons, ["恢复专项会话"])
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["message", "message", "update_score", "session_resumed"],
            )
            score_payload = next(
                payload
                for payload in connection.sent
                if payload["type"] == "update_score"
            )
            self.assertEqual(score_payload["mode"], "cognitive_screening")

        asyncio.run(scenario())

    def test_wellbeing_resume_does_not_replay_mmse_score(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=SimpleNamespace(
                    tool_gateway=SimpleNamespace(mmse_tool=None)
                ),
                owner_username="doctor",
            )
            synced_agents = []
            resume_data = {
                "ended_at": None,
                "mode": "wellbeing",
                "profile": {"name": "患者甲"},
                "chat_history": [{"role": "user", "content": "最近失眠"}],
                "messages": [
                    {
                        "role": "user",
                        "content": "最近失眠",
                        "emotion": "sad",
                    }
                ],
                "mmse_scores": {"orientation": 5},
            }

            async def reset_turn_session(_reason):
                return None

            handler = SessionResumeHandler(
                connection,
                session=session,
                manifest_store=_FakeManifestStore(),
                get_resume=lambda _session_id: resume_data,
                assign_session_owner=lambda *args, **kwargs: None,
                normalize_profile=lambda profile: dict(profile),
                sync_manual_location=lambda _profile: None,
                build_score_payload=lambda scores: {"fallback": scores},
                sync_agent_state=lambda *args: synced_agents.append(args),
                reset_turn_session=reset_turn_session,
                logger=lambda _message: None,
            )

            handled = await handler.handle_message(
                {"type": "resume_session", "session_id": "call-1"}
            )

            self.assertTrue(handled)
            self.assertEqual(session.mode, "wellbeing")
            self.assertNotIn(
                "update_score",
                [payload["type"] for payload in connection.sent],
            )
            self.assertEqual(synced_agents, [])

        asyncio.run(scenario())

    def test_non_resumable_session_returns_stable_failure_payload(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=SimpleNamespace(
                    tool_gateway=SimpleNamespace(mmse_tool=None)
                ),
                owner_username="doctor",
            )
            handler = SessionResumeHandler(
                connection,
                session=session,
                manifest_store=_FakeManifestStore(),
                get_resume=lambda _session_id: None,
                assign_session_owner=lambda *_args, **_kwargs: None,
                normalize_profile=dict,
                sync_manual_location=lambda _profile: None,
                build_score_payload=lambda scores: scores,
                sync_agent_state=lambda *_args: None,
                reset_turn_session=lambda _reason: None,
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {"type": "resume_session", "session_id": "missing"}
            )

            self.assertEqual(
                connection.sent,
                [
                    {
                        "type": "resume_failed",
                        "reason": "会话不存在或已结束",
                    }
                ],
            )
            self.assertEqual(session.session_id, "")

        asyncio.run(scenario())

    def test_transport_recovery_can_resume_without_replaying_messages(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=SimpleNamespace(
                    tool_gateway=SimpleNamespace(mmse_tool=None)
                ),
                owner_username="doctor",
            )
            resume_data = {
                "ended_at": None,
                "profile": {"name": "患者甲"},
                "chat_history": [
                    {"role": "user", "content": "您好"},
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "您好",
                        "emotion": "neutral",
                    }
                ],
                "mmse_scores": {},
            }
            handler = SessionResumeHandler(
                connection,
                session=session,
                manifest_store=_FakeManifestStore(),
                get_resume=lambda _session_id: resume_data,
                assign_session_owner=lambda *_args, **_kwargs: None,
                normalize_profile=dict,
                sync_manual_location=lambda _profile: None,
                build_score_payload=lambda scores: scores,
                sync_agent_state=lambda *_args: None,
                reset_turn_session=lambda _reason: None,
                logger=lambda _message: None,
            )

            handled = await handler.handle_message(
                {
                    "type": "resume_session",
                    "session_id": "call-1",
                    "replay_history": False,
                }
            )

            self.assertTrue(handled)
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["session_resumed"],
            )
            self.assertEqual(session.chat_history[0]["content"], "您好")

        asyncio.run(scenario())

    def test_resume_authorizes_patient_before_loading_session_contents(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=SimpleNamespace(
                    tool_gateway=SimpleNamespace(mmse_tool=None)
                ),
                owner_username="doctor",
            )
            loaded = []
            handler = SessionResumeHandler(
                connection,
                session=session,
                manifest_store=_FakeManifestStore(),
                get_resume=lambda _session_id: loaded.append(True),
                get_session_binding=lambda _session_id: {
                    "patient_id": "patient-1",
                    "ended_at": None,
                },
                assign_session_owner=lambda *_args, **_kwargs: None,
                normalize_profile=dict,
                sync_manual_location=lambda _profile: None,
                build_score_payload=lambda scores: scores,
                sync_agent_state=lambda *_args: None,
                reset_turn_session=lambda _reason: None,
                authorize_patient=lambda *_args: (_ for _ in ()).throw(
                    PermissionError("PATIENT_ACCESS_DENIED")
                ),
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {"type": "resume_session", "session_id": "call-1"}
            )

            self.assertEqual(loaded, [])
            self.assertEqual(session.session_id, "")
            self.assertEqual(
                connection.sent,
                [{"type": "resume_failed", "reason": "患者访问未授权"}],
            )

        asyncio.run(scenario())


class AnswerCompletionMessageHandlerTests(unittest.TestCase):
    def test_mode_and_manual_check_are_routed_through_state_object(self):
        async def scenario():
            state = AnswerCompletionState()
            notifications = []
            checks = []

            async def notify(state_name, message, **fields):
                notifications.append((state_name, message, fields))

            async def request_check(**fields):
                checks.append(fields)

            handler = AnswerCompletionMessageHandler(
                state,
                notify_status=notify,
                request_check=request_check,
            )

            await handler.handle_message(
                {
                    "type": "set_answer_completion_mode",
                    "enabled": True,
                }
            )
            await handler.handle_message(
                {"type": "request_answer_completion_check"}
            )

            self.assertTrue(state.enabled)
            self.assertEqual(notifications[0][0], "enabled")
            self.assertEqual(checks, [{"trigger": "manual"}])

        asyncio.run(scenario())


class SessionLifecycleHandlerTests(unittest.TestCase):
    def _build_handler(
        self,
        *,
        create_fresh_session=None,
        stream_tts_audio=None,
        authorize_patient=None,
        mode="wellbeing",
    ):
        connection = _FakeConnection()
        memory_backgrounds = []

        class MmseTool:
            def _run(self, **_kwargs):
                return json.dumps(
                    {
                        "success": True,
                        "total_score": 20,
                        "cognitive_status": "轻度认知下降",
                    }
                )

        agent = SimpleNamespace(
            tool_gateway=SimpleNamespace(
                mmse_tool=MmseTool(),
                memory_tool=SimpleNamespace(
                    set_persistent_background=memory_backgrounds.append
                ),
            ),
        )
        session = VoiceSession(
            connection=connection,
            agent=agent,
            owner_username="doctor",
        )
        session.mode = mode
        session.bind_session("call-1")
        answer_completion = AnswerCompletionState()
        history_store = _FakeHistoryStore()
        manifest_store = _FakeManifestStore()
        patient_memory = _FakePatientMemoryService()
        ended = []
        updated_profiles = []
        synced_locations = []
        reset_reasons = []
        vad_resets = []
        waiting_calls = []

        def default_create_fresh():
            session.bind_session("call-2")
            return session.session_id

        async def default_stream_tts(*_args, **kwargs):
            payload = kwargs.get("start_payload")
            if payload:
                await connection.send_json(payload)
            return {
                "samples": 24000,
                "interrupted": False,
                "started": bool(payload),
            }

        async def reset_turn(reason):
            reset_reasons.append(reason)

        async def send_waiting():
            waiting_calls.append(session.session_id)

        handler = SessionLifecycleHandler(
            connection,
            session=session,
            answer_completion=answer_completion,
            history_store=history_store,
            manifest_store=manifest_store,
            patient_memory_service=patient_memory,
            create_fresh_session=(
                create_fresh_session or default_create_fresh
            ),
            end_session=lambda *args: ended.append(args),
            update_profile=lambda *args: updated_profiles.append(args),
            normalize_profile=lambda profile: {
                "name": str(profile.get("name") or "").strip(),
                "gender": profile.get("gender", ""),
                "education_years": 6,
            },
            sync_manual_location=synced_locations.append,
            reset_turn_session=reset_turn,
            reset_vad=lambda: vad_resets.append(True),
            stream_tts_audio=stream_tts_audio or default_stream_tts,
            clean_for_tts=lambda text: text,
            send_waiting_for_info=send_waiting,
            client_id="client-1",
            authorize_patient=authorize_patient,
            now_factory=lambda: 100.0,
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            handler=handler,
            connection=connection,
            session=session,
            answer_completion=answer_completion,
            history_store=history_store,
            manifest_store=manifest_store,
            patient_memory=patient_memory,
            memory_backgrounds=memory_backgrounds,
            ended=ended,
            updated_profiles=updated_profiles,
            synced_locations=synced_locations,
            reset_reasons=reset_reasons,
            vad_resets=vad_resets,
            waiting_calls=waiting_calls,
        )

    def test_start_session_binds_patient_and_sends_personalized_greeting(self):
        async def scenario():
            context = self._build_handler()

            handled = await context.handler.handle_message(
                {
                    "type": "start_session",
                    "patient_id": "patient-1",
                    "profile": {"name": " 张阿姨 ", "gender": "女"},
                    "long_term_memory_enabled": True,
                    "long_term_memory_writes_enabled": False,
                    "emotion_enabled": False,
                }
            )

            self.assertTrue(handled)
            self.assertTrue(context.session.lifecycle.started)
            self.assertTrue(context.session.lifecycle.greeting_sent)
            self.assertEqual(context.session.mode, "wellbeing")
            self.assertEqual(context.session.patient_profile["name"], "张阿姨")
            self.assertEqual(
                context.updated_profiles,
                [
                    (
                        "call-1",
                        {
                            "name": "张阿姨",
                            "gender": "女",
                            "education_years": 6,
                        },
                    )
                ],
            )
            self.assertEqual(
                context.patient_memory.resolved,
                [("call-1", "patient-1")],
            )
            self.assertEqual(context.memory_backgrounds, ["", "既往记忆"])
            self.assertEqual(context.manifest_store.refresh_count, 1)
            session_started = next(
                payload
                for payload in context.connection.sent
                if payload["type"] == "session_started"
            )
            self.assertEqual(session_started["mode"], "wellbeing")
            self.assertTrue(session_started["long_term_memory_enabled"])
            self.assertFalse(session_started["long_term_memory_writes_enabled"])
            self.assertFalse(session_started["emotion_enabled"])
            self.assertFalse(context.session.long_term_memory_writes_enabled)
            self.assertFalse(context.session.emotion_enabled)
            self.assertEqual(context.session.runtime.ai_speaking_until, 101.0)
            self.assertEqual(len(context.session.chat_history), 1)
            self.assertIn("张阿姨女士", context.session.chat_history[0]["content"])
            self.assertEqual(len(context.history_store.appended), 1)
            self.assertEqual(
                [payload["type"] for payload in context.connection.sent],
                [
                    "patient_memory",
                    "session_started",
                    "ai_response",
                    "tts_start",
                    "tts_end",
                ],
            )

        asyncio.run(scenario())

    def test_start_session_rejects_unauthorized_patient_before_binding(self):
        async def scenario():
            context = self._build_handler(
                authorize_patient=lambda *_args: False
            )

            handled = await context.handler.handle_message(
                {
                    "type": "start_session",
                    "patient_id": "patient-denied",
                    "profile": {"name": "不应加载"},
                }
            )

            self.assertTrue(handled)
            self.assertEqual(context.patient_memory.resolved, [])
            self.assertFalse(context.session.lifecycle.started)
            self.assertIsNone(context.session.lifecycle.current_patient_id)
            self.assertEqual(context.session.patient_profile, {})
            self.assertEqual(
                context.connection.sent,
                [{"type": "resume_failed", "reason": "患者访问未授权"}],
            )

        asyncio.run(scenario())

    def test_end_session_consolidates_and_waits_for_next_speech(self):
        async def scenario():
            context = self._build_handler(mode="cognitive_screening")
            context.session.patient_profile.update(
                {"name": "患者甲", "education_years": 6}
            )
            context.session.chat_history.append(
                {"role": "user", "content": "回答"}
            )
            context.session.lifecycle.started = True
            context.session.lifecycle.current_patient_id = "patient-1"
            context.answer_completion.enabled = True
            context.session.processing.revision_enabled = True
            context.session.processing.pending_audio = object()
            context.session.runtime.pending_vision_task = "vision-1"
            context.session.runtime.ai_streaming_tts = True
            context.session.runtime.interrupt_audio_buffer.append(object())
            context.session.runtime.early_vad_armed = True

            result = await context.handler.handle_message(
                {"type": "end_session"}
            )

            self.assertTrue(result)
            self.assertEqual(
                context.ended,
                [("call-1", 20, "轻度认知下降")],
            )
            self.assertEqual(
                context.patient_memory.consolidated[0]["session_id"],
                "call-1",
            )
            self.assertEqual(
                context.patient_memory.consolidated[0]["chat_history"],
                [{"role": "user", "content": "回答"}],
            )
            self.assertEqual(context.patient_memory.flushed, ["call-1"])
            self.assertEqual(context.session.session_id, "call-1")
            self.assertEqual(context.session.mode, "wellbeing")
            self.assertEqual(context.session.chat_history, [])
            self.assertFalse(context.session.lifecycle.started)
            self.assertTrue(context.session.lifecycle.awaiting_next_utterance)
            self.assertFalse(context.answer_completion.enabled)
            self.assertFalse(context.session.processing.revision_enabled)
            self.assertIsNone(context.session.processing.pending_audio)
            self.assertIsNone(context.session.runtime.pending_vision_task)
            self.assertFalse(context.session.runtime.ai_streaming_tts)
            self.assertEqual(
                context.session.runtime.interrupt_audio_buffer,
                [],
            )
            self.assertFalse(context.session.runtime.early_vad_armed)
            self.assertEqual(context.reset_reasons, ["结束评估"])
            self.assertEqual(context.vad_resets, [True])
            self.assertEqual(context.waiting_calls, [])
            self.assertEqual(
                context.connection.sent[-1],
                {
                    "type": "session_waiting_next_speech",
                    "patient_id": "patient-1",
                    "mode": "wellbeing",
                },
            )

        asyncio.run(scenario())

    def test_next_speech_after_cognitive_session_starts_wellbeing_session(self):
        async def scenario():
            context = self._build_handler(mode="cognitive_screening")
            context.session.patient_profile.update({"name": "患者甲"})
            context.session.lifecycle.current_patient_id = "patient-1"
            context.session.lifecycle.awaiting_next_utterance = True
            context.session.lifecycle.sealed = True

            started = await context.handler.start_next_on_speech()

            self.assertTrue(started)
            self.assertEqual(context.session.session_id, "call-2")
            self.assertEqual(context.session.mode, "wellbeing")
            self.assertTrue(context.session.lifecycle.started)
            self.assertEqual(
                context.connection.sent[-1]["type"],
                "session_started",
            )
            self.assertEqual(context.connection.sent[-1]["mode"], "wellbeing")

        asyncio.run(scenario())

    def test_end_session_still_resets_and_waits_when_flush_fails(self):
        async def scenario():
            context = self._build_handler()
            context.session.patient_profile.update({"name": "患者甲"})
            context.session.lifecycle.started = True
            context.session.lifecycle.current_patient_id = "patient-1"

            def fail_flush(_session):
                raise RuntimeError("memobase unavailable")

            context.patient_memory.flush_session = fail_flush

            self.assertTrue(
                await context.handler.handle_message({"type": "end_session"})
            )
            self.assertFalse(context.session.lifecycle.started)
            self.assertTrue(context.session.lifecycle.awaiting_next_utterance)
            self.assertEqual(
                context.connection.sent[-1],
                {
                    "type": "session_waiting_next_speech",
                    "patient_id": "patient-1",
                    "mode": "wellbeing",
                },
            )

        asyncio.run(scenario())

    def test_prepare_patient_waits_for_speech_without_greeting(self):
        async def scenario():
            context = self._build_handler()

            handled = await context.handler.handle_message(
                {
                    "type": "prepare_patient",
                    "patient_id": "patient-1",
                    "profile": {"name": "不应读取"},
                }
            )

            self.assertTrue(handled)
            self.assertEqual(
                context.patient_memory.resolved,
                [("call-1", "patient-1")],
            )
            self.assertEqual(context.session.patient_profile, {})
            self.assertEqual(context.session.chat_history, [])
            self.assertEqual(context.session.lifecycle.current_patient_id, "patient-1")
            self.assertFalse(context.session.lifecycle.started)
            self.assertFalse(context.session.lifecycle.greeting_sent)
            self.assertTrue(context.session.lifecycle.awaiting_next_utterance)
            self.assertEqual(
                [payload["type"] for payload in context.connection.sent],
                ["patient_memory", "session_waiting_next_speech"],
            )

            self.assertTrue(await context.handler.start_next_on_speech())
            self.assertEqual(context.session.session_id, "call-1")
            self.assertTrue(context.session.lifecycle.started)

        asyncio.run(scenario())

    def test_next_speech_creates_a_fresh_session_for_the_same_patient(self):
        async def scenario():
            context = self._build_handler()
            context.session.patient_profile.update({"name": "患者甲"})
            context.session.lifecycle.current_patient_id = "patient-1"
            self.assertTrue(
                await context.handler.handle_message({"type": "end_session"})
            )
            self.assertTrue(await context.handler.start_next_on_speech())
            self.assertEqual(context.session.session_id, "call-2")
            self.assertTrue(context.session.lifecycle.started)
            self.assertFalse(context.session.lifecycle.awaiting_next_utterance)
            self.assertEqual(
                context.patient_memory.resolved[-1],
                ("call-2", "patient-1"),
            )

        asyncio.run(scenario())


class ManualAudioInputHandlerTests(unittest.TestCase):
    def test_chunked_blob_is_reassembled_in_index_order(self):
        async def scenario():
            connection = _FakeConnection()
            processed = []

            async def process_blob(*args, **kwargs):
                processed.append((args, kwargs))

            handler = ManualAudioInputHandler(
                connection,
                uploads={},
                process_recorded_blob=process_blob,
                submit_speech=lambda *_args, **_kwargs: None,
                normalize_doctor_markers=lambda markers: markers or [],
                resample_audio=lambda audio, _src, _dst: audio,
                token_factory=lambda: "generated-id",
                now_factory=lambda: 10.0,
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {
                    "type": "manual_audio_blob_start",
                    "upload_id": "upload-1",
                    "total_chunks": 2,
                    "mime_type": "audio/webm",
                    "doctor_markers": [{"event": "doctor_start"}],
                }
            )
            await handler.handle_message(
                {
                    "type": "manual_audio_blob_chunk",
                    "upload_id": "upload-1",
                    "index": 1,
                    "data": "BBBB",
                }
            )
            await handler.handle_message(
                {
                    "type": "manual_audio_blob_chunk",
                    "upload_id": "upload-1",
                    "index": 0,
                    "data": "AAAA",
                }
            )
            await handler.handle_message(
                {
                    "type": "manual_audio_blob_end",
                    "upload_id": "upload-1",
                }
            )

            self.assertEqual(handler.uploads, {})
            self.assertEqual(processed[0][0], ("AAAABBBB", "audio/webm"))
            self.assertEqual(
                processed[0][1],
                {
                    "source": "manual_audio_blob_chunked",
                    "upload_id": "upload-1",
                    "doctor_markers": [{"event": "doctor_start"}],
                },
            )

        asyncio.run(scenario())

    def test_incomplete_blob_reports_stable_feedback(self):
        async def scenario():
            connection = _FakeConnection()
            handler = ManualAudioInputHandler(
                connection,
                uploads={},
                process_recorded_blob=lambda *_args, **_kwargs: None,
                submit_speech=lambda *_args, **_kwargs: None,
                normalize_doctor_markers=lambda markers: markers or [],
                resample_audio=lambda audio, _src, _dst: audio,
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {
                    "type": "manual_audio_blob_start",
                    "upload_id": "upload-1",
                    "total_chunks": 2,
                }
            )
            await handler.handle_message(
                {
                    "type": "manual_audio_blob_chunk",
                    "upload_id": "upload-1",
                    "index": 0,
                    "data": "AAAA",
                }
            )
            await handler.handle_message(
                {
                    "type": "manual_audio_blob_end",
                    "upload_id": "upload-1",
                }
            )

            self.assertEqual(
                connection.sent[-1],
                {
                    "type": "voice_input_feedback",
                    "reason": "generic",
                    "message": "手动录音分片不完整，请再试一次",
                    "status_text": "手动录音分片不完整，请再试一次",
                    "source": "manual_audio_blob_chunked",
                },
            )

        asyncio.run(scenario())

    def test_pcm_audio_is_normalized_resampled_and_submitted(self):
        async def scenario():
            connection = _FakeConnection()
            submitted = []
            resample_calls = []
            samples = np.array([32767, -32768], dtype=np.int16)

            def resample(audio, source_rate, target_rate):
                resample_calls.append((audio.copy(), source_rate, target_rate))
                return np.repeat(audio, 2)

            async def submit(audio, **kwargs):
                submitted.append((audio.copy(), kwargs))

            handler = ManualAudioInputHandler(
                connection,
                uploads={},
                process_recorded_blob=lambda *_args, **_kwargs: None,
                submit_speech=submit,
                normalize_doctor_markers=lambda markers: [
                    {"normalized": len(markers or [])}
                ],
                resample_audio=resample,
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {
                    "type": "manual_audio",
                    "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
                    "sample_rate": 8000,
                    "doctor_markers": [{"event": "doctor_start"}],
                }
            )

            self.assertEqual(len(resample_calls), 1)
            self.assertEqual(resample_calls[0][1:], (8000, 16000))
            self.assertEqual(connection.sent, [{"type": "vad_end"}])
            np.testing.assert_allclose(
                submitted[0][0],
                np.repeat(samples.astype(np.float32) / 32768.0, 2),
            )
            self.assertEqual(
                submitted[0][1],
                {
                    "source": "手动点击语音",
                    "extra_meta": {
                        "doctor_markers": [{"normalized": 1}],
                        "manual_source": "manual_audio",
                        "sample_rate": 8000,
                    },
                },
            )

        asyncio.run(scenario())

    def test_entrypoint_has_no_inline_manual_audio_message_branches(self):
        application = _voice_application_class()
        inline_types = [
            node.value
            for node in ast.walk(application)
            if isinstance(node, ast.Constant)
            and node.value in ManualAudioInputHandler.MESSAGE_TYPES
        ]

        self.assertEqual(inline_types, [])


class VoiceFeedbackPresenterTests(unittest.TestCase):
    def test_feedback_omits_none_fields_and_keeps_status_text(self):
        async def scenario():
            connection = _FakeConnection()
            presenter = VoiceFeedbackPresenter(connection)

            await presenter.send(
                "speech_too_short",
                "请再说一遍",
                status_text="语音太短",
                duration_s=0.3,
                turn_id=None,
            )

            self.assertEqual(
                connection.sent,
                [
                    {
                        "type": "voice_input_feedback",
                        "reason": "speech_too_short",
                        "message": "请再说一遍",
                        "status_text": "语音太短",
                        "duration_s": 0.3,
                    }
                ],
            )

        asyncio.run(scenario())


class ManualAudioBlobProcessorTests(unittest.TestCase):
    def test_doctor_markers_are_normalized_and_sorted(self):
        markers = ManualAudioBlobProcessor.normalize_doctor_markers(
            [
                {
                    "event": "doctor_end",
                    "at_ms": "20",
                    "duration_ms": "4",
                },
                {
                    "event": "ignored",
                    "at_ms": 1,
                },
                {
                    "event": "doctor_start",
                    "at_ms": -5,
                    "ts": "12.5",
                    "auto_closed": True,
                },
            ]
        )

        self.assertEqual(
            markers,
            [
                {
                    "event": "doctor_start",
                    "at_ms": 0.0,
                    "ts": 12.5,
                    "auto_closed": True,
                },
                {
                    "event": "doctor_end",
                    "at_ms": 20.0,
                    "duration_ms": 4.0,
                },
            ],
        )

    def test_blob_is_decoded_resampled_and_submitted(self):
        async def scenario():
            connection = _FakeConnection()
            submitted = []
            resampled = []

            class Coordinator:
                async def submit_or_queue(self, audio, **kwargs):
                    submitted.append((np.asarray(audio).copy(), kwargs))

            def resample(audio, source_rate, target_rate):
                resampled.append((source_rate, target_rate))
                return np.repeat(audio, 2)

            processor = ManualAudioBlobProcessor(
                connection,
                processing_coordinator=Coordinator(),
                feedback=VoiceFeedbackPresenter(connection),
                decode_recorded_audio_blob=lambda _data, _mime: (
                    np.array([0.1, -0.1], dtype=np.float32),
                    8000,
                ),
                resample_audio=resample,
                audio_signal_stats=lambda audio, rate: {
                    "duration_s": len(audio) / rate,
                    "rms": 0.1,
                    "peak": 0.1,
                    "dbfs": -20.0,
                },
                audio_is_effectively_silent=lambda _stats: False,
                logger=lambda _message: None,
            )

            await processor.process(
                base64.b64encode(b"recording").decode("ascii"),
                "audio/webm",
                "manual_audio_blob",
                upload_id="upload-1",
                doctor_markers=[
                    {"event": "doctor_start", "at_ms": 10}
                ],
            )

            self.assertEqual(resampled, [(8000, 16000)])
            self.assertEqual(
                connection.sent,
                [{"type": "vad_end"}],
            )
            self.assertEqual(len(submitted), 1)
            self.assertEqual(
                submitted[0][1],
                {
                    "source": "手动点击语音",
                    "extra_meta": {
                        "doctor_markers": [
                            {
                                "event": "doctor_start",
                                "at_ms": 10.0,
                            }
                        ],
                        "manual_source": "manual_audio_blob",
                        "mime_type": "audio/webm",
                    },
                },
            )

        asyncio.run(scenario())

    def test_silent_blob_reports_feedback_without_submission(self):
        async def scenario():
            connection = _FakeConnection()
            submitted = []

            class Coordinator:
                async def submit_or_queue(self, *_args, **_kwargs):
                    submitted.append(True)

            processor = ManualAudioBlobProcessor(
                connection,
                processing_coordinator=Coordinator(),
                feedback=VoiceFeedbackPresenter(connection),
                decode_recorded_audio_blob=lambda _data, _mime: (
                    np.ones(1600, dtype=np.float32),
                    16000,
                ),
                resample_audio=lambda audio, _source, _target: audio,
                audio_signal_stats=lambda _audio, _rate: {
                    "duration_s": 0.1,
                    "rms": 0.0,
                    "peak": 0.0,
                    "dbfs": -100.0,
                },
                audio_is_effectively_silent=lambda _stats: True,
                logger=lambda _message: None,
            )

            await processor.process(
                base64.b64encode(b"quiet").decode("ascii"),
                "audio/webm",
                "manual_audio_blob",
            )

            self.assertEqual(submitted, [])
            self.assertEqual(
                connection.sent[-1]["reason"],
                "audio_too_quiet",
            )

        asyncio.run(scenario())


class AgentOutputPresenterTests(unittest.TestCase):
    def test_cognitive_session_presents_image_and_vision_commands(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=_FakeVisionAgent(),
                owner_username="doctor",
            )
            session.mode = "cognitive_screening"
            presenter = AgentOutputPresenter(
                connection,
                session=session,
                now_factory=lambda: 123.0,
                logger=lambda _message: None,
            )

            image_sent = await presenter.send_image_display(
                {
                    "image_display": json.dumps(
                        {"type": "show_image", "image_id": "clock"}
                    )
                },
                source="test",
            )
            vision_sent = await presenter.send_vision_command(
                {
                    "vision_command": {
                        "type": "vision_capture",
                        "task_id": "copy_pentagons",
                        "mode": "manual",
                    }
                },
                source="test",
            )

            self.assertTrue(image_sent)
            self.assertTrue(vision_sent)
            self.assertEqual(
                connection.sent[0]["url"],
                "/api/mmse-image/clock",
            )
            self.assertEqual(
                session.runtime.pending_vision_task,
                "copy_pentagons",
            )
            self.assertEqual(session.runtime.vision_lock_time, 123.0)

        asyncio.run(scenario())

    def test_wellbeing_session_blocks_image_and_vision_commands(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=_FakeVisionAgent(),
                owner_username="doctor",
            )
            session.mode = "wellbeing"
            presenter = AgentOutputPresenter(
                connection,
                session=session,
                now_factory=lambda: 123.0,
                logger=lambda _message: None,
            )

            image_sent = await presenter.send_image_display(
                {
                    "image_display": json.dumps(
                        {"type": "show_image", "image_id": "clock"}
                    )
                },
                source="test",
            )
            vision_sent = await presenter.send_vision_command(
                {
                    "vision_command": {
                        "type": "vision_capture",
                        "task_id": "copy_pentagons",
                        "mode": "manual",
                    }
                },
                source="test",
            )

            self.assertFalse(image_sent)
            self.assertFalse(vision_sent)
            self.assertEqual(connection.sent, [])
            self.assertIsNone(session.runtime.pending_vision_task)

        asyncio.run(scenario())


class ManualInterruptHandlerTests(unittest.TestCase):
    def test_interrupt_stops_playback_and_resets_connection_runtime(self):
        async def scenario():
            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=_FakeVisionAgent(),
                owner_username="doctor",
            )
            session.runtime.early_vad_armed = True
            session.runtime.early_vad_started_during_ai = True
            session.runtime.early_vad_post_tts_chunks = 3
            calls = []

            async def stop_playback():
                calls.append(("stop", {}))

            def reset_interrupt_capture(**kwargs):
                calls.append(("reset", kwargs))

            handler = ManualInterruptHandler(
                session=session,
                stop_playback=stop_playback,
                reset_interrupt_capture=reset_interrupt_capture,
                logger=lambda _message: None,
            )

            handled = await handler.handle_message({"type": "interrupt"})

            self.assertTrue(handled)
            self.assertEqual(
                calls,
                [
                    ("stop", {}),
                    (
                        "reset",
                        {"reset_waiting": True, "reset_vad": True},
                    ),
                ],
            )
            self.assertFalse(session.runtime.early_vad_armed)
            self.assertFalse(
                session.runtime.early_vad_started_during_ai
            )
            self.assertEqual(
                session.runtime.early_vad_post_tts_chunks,
                0,
            )

        asyncio.run(scenario())


class VoiceInterruptionControllerTests(unittest.TestCase):
    @staticmethod
    def _build_controller(*, quick_text="患者回答"):
        connection = _FakeConnection()
        session = VoiceSession(
            connection=connection,
            agent=object(),
            owner_username="doctor",
        )

        class VAD:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        class SoulXAudio:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        class SoulXClient:
            def __init__(self):
                self.reset_ids = []

            async def reset(self, session_id):
                self.reset_ids.append(session_id)

        vad = VAD()
        soulx_audio = SoulXAudio()
        soulx_client = SoulXClient()

        async def quick_asr(_audio):
            return quick_text

        async def judge_answer(_question, _text):
            return {"label": "incomplete", "confidence": 0.8}

        async def judge_interrupt(_text):
            return "complete"

        controller = VoiceInterruptionController(
            connection,
            session=session,
            vad_buffer=vad,
            soulx_client=soulx_client,
            soulx_audio=soulx_audio,
            quick_asr=quick_asr,
            judge_answer_completion=judge_answer,
            judge_interrupt_intent=judge_interrupt,
            extract_latest_assistant_utterance=lambda _history: "请回答",
            normalize_interrupt_text=lambda text: text.replace(" ", ""),
            soulx_session_id_factory=lambda: "soulx-reset-id",
            now_factory=lambda: 10.0,
            playback_stop_delay_s=0.0,
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            connection=connection,
            session=session,
            vad=vad,
            soulx_audio=soulx_audio,
            soulx_client=soulx_client,
            controller=controller,
        )

    def test_capture_judgement_and_speaking_state_are_session_owned(self):
        context = self._build_controller()
        audio = np.ones(3200, dtype=np.float32)
        context.session.runtime.ai_speaking_until = 11.0

        context.controller.remember_judgement(
            audio,
            "等一下",
            "complete",
        )

        self.assertTrue(context.controller.is_ai_speaking())
        self.assertEqual(
            context.controller.reuse_judgement(audio),
            ("等一下", "complete"),
        )
        context.controller.reset_capture(
            reset_waiting=True,
            reset_vad=True,
        )
        self.assertEqual(context.vad.reset_count, 1)
        self.assertEqual(
            context.controller.reuse_judgement(audio),
            (None, None),
        )

    def test_explicit_interrupt_phrase_is_treated_as_complete(self):
        async def scenario():
            context = self._build_controller(quick_text="等一下")

            text, intent = (
                await context.controller.judge_waiting_completion(
                    np.ones(1600, dtype=np.float32)
                )
            )

            self.assertEqual(text, "等一下")
            self.assertEqual(intent, "complete")

        asyncio.run(scenario())

    def test_stop_playback_sends_protocol_and_clears_runtime(self):
        async def scenario():
            context = self._build_controller()
            runtime = context.session.runtime
            runtime.ai_streaming_tts = True
            runtime.ai_speaking_until = 20.0

            await context.controller.stop_playback()

            self.assertEqual(
                context.connection.sent,
                [{"type": "stop_tts"}, {"type": "interrupt"}],
            )
            self.assertTrue(runtime.stop_generate)
            self.assertFalse(runtime.ai_streaming_tts)
            self.assertEqual(runtime.ai_speaking_until, 0.0)

        asyncio.run(scenario())

    def test_reset_turn_taking_resets_local_and_remote_state(self):
        async def scenario():
            context = self._build_controller()
            context.session.turn_taking.soulx_healthy = True

            await context.controller.reset_turn_taking("新会话")

            self.assertEqual(context.soulx_audio.reset_count, 1)
            self.assertEqual(
                context.soulx_client.reset_ids,
                ["soulx-reset-id"],
            )
            self.assertFalse(
                context.session.turn_taking.soulx_healthy
            )

        asyncio.run(scenario())

    def test_entrypoint_has_no_interruption_state_closures(self):
        application = _voice_application_class()
        nested_names = {
            node.name
            for node in ast.walk(application)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertTrue(
            {
                "_is_ai_speaking",
                "_reset_interrupt_capture",
                "_remember_interrupt_judgement",
                "_reuse_interrupt_judgement",
                "_reset_soulx_turn_session",
                "_judge_waiting_completion",
                "_stop_ai_playback",
            }.isdisjoint(nested_names)
        )


class VisionMessageHandlerTests(unittest.TestCase):
    @staticmethod
    def _build_handler(*, evaluate_image=None, mode="wellbeing"):
        connection = _FakeConnection()
        agent = _FakeVisionAgent()
        session = VoiceSession(
            connection=connection,
            agent=agent,
            owner_username="doctor",
        )
        session.mode = mode
        session.bind_session("session-vision")
        session.lifecycle.started = True
        session.patient_profile = {"name": "患者甲"}
        history_store = _FakeHistoryStore()
        presenter = AgentOutputPresenter(
            connection,
            session=session,
            now_factory=lambda: 20.0,
            logger=lambda _message: None,
        )

        async def stream_tts_audio(*_args, **kwargs):
            payload = kwargs.get("start_payload")
            if payload:
                await connection.send_json(payload)
            return {"samples": 24000, "chunks": 2, "started": bool(payload)}

        handler = VisionMessageHandler(
            connection,
            session=session,
            history_store=history_store,
            presenter=presenter,
            stream_tts_audio=stream_tts_audio,
            clean_for_tts=lambda text: text,
            normalize_profile=lambda profile: dict(profile or {}),
            evaluate_image=evaluate_image,
            now_factory=lambda: 30.0,
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            connection=connection,
            agent=agent,
            session=session,
            history_store=history_store,
            handler=handler,
        )

    def test_debug_task_builds_agent_state_and_starts_vision_capture(self):
        async def scenario():
            context = self._build_handler(mode="cognitive_screening")

            handled = await context.handler.handle_message(
                {
                    "type": "debug_trigger_task",
                    "task_id": "language_3step_action",
                }
            )

            self.assertTrue(handled)
            self.assertEqual(
                context.session.runtime.pending_vision_task,
                "language_3step_action",
            )
            self.assertEqual(
                context.agent.state._last_task_id,
                "language_3step_action",
            )
            self.assertEqual(
                [payload["type"] for payload in context.connection.sent],
                [
                    "vision_capture",
                    "ai_response",
                    "tts_start",
                    "tts_end",
                ],
            )
            self.assertEqual(
                context.history_store.appended[0][0],
                "assistant",
            )

        asyncio.run(scenario())

    def test_drawing_evaluation_runs_agent_turn_and_releases_processing(self):
        async def scenario():
            context = self._build_handler(
                evaluate_image=lambda _image, _task: {
                    "is_correct": True,
                    "quality_level": "good",
                },
                mode="cognitive_screening",
            )

            await context.handler.handle_message(
                {
                    "type": "drawing_submit",
                    "image": "base64-image",
                    "mode": "pentagons",
                }
            )

            self.assertEqual(len(context.agent.process_calls), 1)
            self.assertIn(
                "结果：正确（good）",
                context.agent.process_calls[0]["user_input"],
            )
            self.assertFalse(context.session.processing.is_active)
            self.assertEqual(
                [payload["type"] for payload in context.connection.sent],
                [
                    "drawing_evaluating",
                    "drawing_result",
                    "ai_response",
                    "tts_start",
                    "tts_end",
                ],
            )

        asyncio.run(scenario())

    def test_mismatched_vision_result_is_stopped_without_agent_call(self):
        async def scenario():
            context = self._build_handler(mode="cognitive_screening")
            context.session.runtime.pending_vision_task = "expected-task"

            await context.handler.handle_message(
                {
                    "type": "vision_eval_result",
                    "task_id": "stale-task",
                    "result": {"is_correct": True},
                }
            )

            self.assertEqual(context.agent.process_calls, [])
            self.assertEqual(
                context.connection.sent,
                [{"type": "vision_stop", "task_id": "stale-task"}],
            )
            self.assertEqual(
                context.session.runtime.pending_vision_task,
                "expected-task",
            )

        asyncio.run(scenario())

    def test_entrypoint_routes_visual_and_interrupt_messages_to_objects(self):
        application = _voice_application_class()
        extracted_types = (
            VisionMessageHandler.MESSAGE_TYPES
            | ManualInterruptHandler.MESSAGE_TYPES
        )
        inline_type_comparisons = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value in extracted_types
                for comparator in node.comparators
            )
        ]

        self.assertEqual(inline_type_comparisons, [])


class TextTurnHandlerTests(unittest.TestCase):
    @staticmethod
    def _build_handler(
        *,
        streaming,
        result=None,
        streamed_sentences=None,
        error=None,
        mode="wellbeing",
    ):
        connection = _FakeConnection()
        agent = _FakeTextAgent(
            result=result,
            streamed_sentences=streamed_sentences,
            error=error,
        )
        session = VoiceSession(
            connection=connection,
            agent=agent,
            owner_username="doctor",
        )
        session.mode = mode
        session.bind_session("session-text")
        session.patient_profile = {
            "name": "患者乙",
            "education_years": 9,
        }
        history_store = _FakeHistoryStore()
        audio_store = _FakeAudioStore()
        presenter = AgentOutputPresenter(
            connection,
            session=session,
            logger=lambda _message: None,
        )
        tts_calls = []

        async def stream_tts_audio(text, **kwargs):
            tts_calls.append((text, kwargs))
            if kwargs.get("start_payload"):
                await connection.send_json(kwargs["start_payload"])
            return {
                "samples": 24000,
                "chunks": 1,
                "audio_data": np.ones(24000, dtype=np.float32),
                "started": True,
            }

        handler = TextTurnHandler(
            connection,
            session=session,
            history_store=history_store,
            audio_store=audio_store,
            presenter=presenter,
            stream_tts_audio=stream_tts_audio,
            clean_for_tts=lambda text: text,
            normalize_profile=lambda profile: dict(profile or {}),
            use_llm_streaming=streaming,
            now_factory=lambda: 100.0,
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            connection=connection,
            agent=agent,
            session=session,
            history_store=history_store,
            audio_store=audio_store,
            tts_calls=tts_calls,
            handler=handler,
        )

    def test_synchronous_text_turn_updates_history_audio_and_score(self):
        async def scenario():
            context = self._build_handler(
                streaming=False,
                result={"output": "这是同步回复。"},
                mode="cognitive_screening",
            )

            handled = await context.handler.handle_message(
                {"type": "text", "data": "患者文字回答"}
            )

            self.assertTrue(handled)
            self.assertEqual(len(context.agent.process_calls), 1)
            self.assertEqual(
                context.agent.process_calls[0]["patient_profile"]["name"],
                "患者乙",
            )
            self.assertEqual(
                [entry["role"] for entry in context.session.chat_history],
                ["user", "assistant"],
            )
            self.assertEqual(
                context.history_store.appended,
                [
                    ("user", "患者文字回答", None, None, None, "turn_0001"),
                    ("assistant", "这是同步回复。", None, None, None, "turn_0001"),
                ],
            )
            self.assertEqual(len(context.audio_store.persisted), 1)
            self.assertEqual(
                context.audio_store.persisted[0][1],
                "这是同步回复。",
            )
            self.assertIn(
                "update_score",
                [payload["type"] for payload in context.connection.sent],
            )
            self.assertFalse(context.session.processing.is_active)
            self.assertEqual(
                context.session.runtime.ai_speaking_until,
                101.0,
            )

        asyncio.run(scenario())

    def test_wellbeing_text_turn_does_not_call_mmse_or_update_score(self):
        async def scenario():
            context = self._build_handler(
                streaming=False,
                result={"output": "这是陪伴回复。"},
                mode="wellbeing",
            )
            mmse_calls = []

            class MmseTool:
                def _run(self, **kwargs):
                    mmse_calls.append(kwargs)
                    return json.dumps({"success": True, "total_score": 30})

            context.agent.tool_gateway.mmse_tool = MmseTool()

            handled = await context.handler.handle_message(
                {"type": "text", "data": "最近睡不好"}
            )

            self.assertTrue(handled)
            payload_types = [
                payload["type"] for payload in context.connection.sent
            ]
            self.assertNotIn("update_score", payload_types)
            self.assertEqual(mmse_calls, [])

        asyncio.run(scenario())

    def test_text_is_queued_while_a_vision_task_owns_the_turn(self):
        async def scenario():
            context = self._build_handler(
                streaming=False,
                mode="cognitive_screening",
            )
            context.session.runtime.pending_vision_task = "close-eyes"

            await context.handler.handle_message(
                {"type": "text", "data": "我已经做完了"}
            )

            self.assertEqual(context.agent.process_calls, [])
            self.assertEqual(
                context.session.runtime.queued_user_text,
                "我已经做完了",
            )
            ai_messages = [
                payload
                for payload in context.connection.sent
                if payload["type"] == "ai_response"
            ]
            self.assertEqual(ai_messages[-1]["text"], "好的，我看到了，请稍等一下。")
            self.assertFalse(context.session.processing.is_active)

        asyncio.run(scenario())

    def test_streaming_text_turn_emits_chunks_and_persists_combined_audio(self):
        async def scenario():
            context = self._build_handler(
                streaming=True,
                result={"output": "第一句。第二句。"},
                streamed_sentences=["第一句。", "第二句。"],
            )

            await context.handler.handle_message(
                {"type": "text", "data": "继续"}
            )

            payload_types = [
                payload["type"] for payload in context.connection.sent
            ]
            self.assertEqual(payload_types.count("ai_response_chunk"), 2)
            self.assertLess(
                payload_types.index("ai_response"),
                payload_types.index("tts_end"),
            )
            self.assertEqual(len(context.tts_calls), 2)
            self.assertEqual(len(context.audio_store.persisted), 1)
            self.assertEqual(
                context.audio_store.persisted[0][0].size,
                48000,
            )
            tts_end = next(
                payload
                for payload in context.connection.sent
                if payload["type"] == "tts_end"
            )
            self.assertEqual(tts_end["duration"], 2.0)
            self.assertEqual(tts_end["chunks"], 2)

        asyncio.run(scenario())

    def test_agent_failure_finalizes_streaming_bubble_and_releases_state(self):
        async def scenario():
            context = self._build_handler(
                streaming=False,
                error=RuntimeError("agent failed"),
            )

            await context.handler.handle_message(
                {"type": "text", "data": "触发失败"}
            )

            error_responses = [
                payload
                for payload in context.connection.sent
                if payload["type"] == "ai_response"
                and payload.get("error")
            ]
            self.assertEqual(len(error_responses), 2)
            self.assertTrue(error_responses[0]["finalize_chunks"])
            self.assertFalse(context.session.processing.is_active)

        asyncio.run(scenario())

    def test_entrypoint_routes_text_messages_to_handler(self):
        application = _voice_application_class()
        inline_text_comparisons = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "text"
                for comparator in node.comparators
            )
        ]

        self.assertEqual(inline_text_comparisons, [])


class SpeechTurnProcessorTests(unittest.TestCase):
    @staticmethod
    def _build_processor(
        *,
        silent=False,
        streaming=False,
        streamed_sentences=None,
        use_ark_asr=False,
        turn_memory=None,
        memory_timeout_s=0.8,
        mode="wellbeing",
    ):
        connection = _FakeConnection()
        agent = _FakeTextAgent(
            result={"output": "语音轮次回复。"},
            streamed_sentences=streamed_sentences,
        )
        session = VoiceSession(
            connection=connection,
            agent=agent,
            owner_username="doctor",
        )
        session.mode = mode
        session.bind_session("session-speech")
        session.patient_profile = {
            "name": "患者丙",
            "education_years": 6,
        }
        session.processing.generation = 3
        speaker = _FakeSoulXSpeaker()
        audio_store = _FakeAudioStore()
        history_store = _FakeHistoryStore()
        presenter = AgentOutputPresenter(
            connection,
            session=session,
            logger=lambda _message: None,
        )
        feedback = []
        tts_calls = []
        turn_memory_calls = []

        class ASRModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return {"raw": "sensevoice"}

        asr_model = ASRModel()

        async def notify_feedback(*args, **kwargs):
            feedback.append((args, kwargs))

        async def stream_tts_audio(text, **kwargs):
            tts_calls.append((text, kwargs))
            if kwargs.get("start_payload"):
                await connection.send_json(kwargs["start_payload"])
            return {
                "chunks": 1,
                "samples": 24000,
                "first_latency": 0.1,
                "interrupted": False,
                "audio_data": np.ones(24000, dtype=np.float32),
                "started": True,
            }

        def get_turn_memory(session_arg, text):
            turn_memory_calls.append((session_arg.session_id, text))
            return turn_memory

        processor = SpeechTurnProcessor(
            connection,
            session=session,
            speaker=speaker,
            audio_store=audio_store,
            history_store=history_store,
            presenter=presenter,
            stream_tts_audio=stream_tts_audio,
            notify_voice_input_feedback=notify_feedback,
            normalize_profile=lambda profile: dict(profile or {}),
            parse_sensevoice_result=lambda _result: {
                "text": "今天星期三",
                "emotion": "neutral",
                "language": "zh",
                "event": "speech",
            },
            audio_signal_stats=lambda _audio, _rate: {
                "duration_s": 1.0,
                "rms": 0.0 if silent else 0.1,
                "peak": 0.2,
                "dbfs": -20.0,
            },
            audio_is_effectively_silent=lambda _stats: silent,
            asr_model=asr_model,
            tts=None,
            clean_for_tts=lambda text: text,
            config=SpeechTurnConfig(
                use_ark_asr=use_ark_asr,
                use_ark_tts=False,
                use_llm_streaming=streaming,
                vision_lock_timeout=30.0,
                memory_timeout_s=memory_timeout_s,
            ),
            logger=lambda _message: None,
            get_turn_memory=(get_turn_memory if turn_memory is not None else None),
        )
        return SimpleNamespace(
            connection=connection,
            agent=agent,
            session=session,
            speaker=speaker,
            audio_store=audio_store,
            history_store=history_store,
            feedback=feedback,
            tts_calls=tts_calls,
            asr_model=asr_model,
            processor=processor,
            turn_memory_calls=turn_memory_calls,
        )

    def test_silent_audio_stops_before_asr_and_reports_feedback(self):
        async def scenario():
            context = self._build_processor(silent=True)

            await context.processor.process(
                np.zeros(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-silent",
            )

            self.assertEqual(context.asr_model.calls, [])
            self.assertEqual(context.feedback[0][0][0], "audio_too_quiet")
            self.assertEqual(context.agent.process_calls, [])
            self.assertEqual(context.speaker.diagnostic_count, 1)

        asyncio.run(scenario())

    def test_non_streaming_speech_turn_runs_asr_agent_tts_and_persistence(self):
        async def scenario():
            context = self._build_processor(mode="cognitive_screening")
            audio = np.ones(16000, dtype=np.float32)

            await context.processor.process(
                audio,
                process_generation_token=3,
                turn_id="turn-voice",
                extra_meta={"source": "test"},
            )

            self.assertEqual(len(context.asr_model.calls), 1)
            self.assertEqual(len(context.agent.process_calls), 1)
            self.assertEqual(
                context.agent.process_calls[0]["user_input"],
                "今天星期三",
            )
            self.assertEqual(
                [entry["role"] for entry in context.session.chat_history],
                ["user", "assistant"],
            )
            self.assertEqual(len(context.audio_store.user_persisted), 1)
            self.assertEqual(
                context.audio_store.user_persisted[0][1]["extra_meta"],
                {"source": "test"},
            )
            self.assertEqual(len(context.audio_store.persisted), 1)
            payload_types = [
                payload["type"] for payload in context.connection.sent
            ]
            self.assertIn("asr_result", payload_types)
            self.assertIn("ai_response", payload_types)
            self.assertIn("tts_end", payload_types)
            self.assertIn("update_score", payload_types)
            self.assertFalse(context.session.processing.revision_enabled)

        asyncio.run(scenario())

    def test_wellbeing_speech_turn_does_not_call_mmse_or_update_score(self):
        async def scenario():
            context = self._build_processor(mode="wellbeing")
            mmse_calls = []

            class MmseTool:
                def _run(self, **kwargs):
                    mmse_calls.append(kwargs)
                    return json.dumps({"success": True, "total_score": 30})

            context.agent.tool_gateway.mmse_tool = MmseTool()

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-wellbeing",
            )

            payload_types = [
                payload["type"] for payload in context.connection.sent
            ]
            self.assertNotIn("update_score", payload_types)
            self.assertEqual(mmse_calls, [])

        asyncio.run(scenario())

    def test_speech_turn_loads_related_memory_before_agent(self):
        async def scenario():
            context = self._build_processor(turn_memory="浴室滑倒旧事")

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-memory",
            )

            self.assertEqual(
                context.turn_memory_calls,
                [("session-speech", "今天星期三")],
            )
            self.assertEqual(
                context.agent.turn_background_at_process,
                "浴室滑倒旧事",
            )

        asyncio.run(scenario())

    def test_session_emotion_ablation_skips_audio_model_and_reports_disabled(self):
        async def scenario():
            context = self._build_processor()
            context.session.emotion_enabled = False

            with mock.patch(
                "src.voice.handlers.speech_turn_processor."
                "classify_multimodal_with_metadata"
            ) as classifier:
                await context.processor.process(
                    np.ones(16000, dtype=np.float32),
                    process_generation_token=3,
                    turn_id="turn-emotion-off",
                )

            classifier.assert_not_called()
            final = next(
                payload
                for payload in context.connection.sent
                if payload.get("type") == "turn_insight"
                and payload.get("state") == "final"
            )
            self.assertEqual(final["emotion"]["source"], "disabled")
            self.assertEqual(final["emotion"]["analysis_status"], "disabled")

        asyncio.run(scenario())

    def test_agent_receives_final_emotion_after_multimodal_inference(self):
        async def scenario():
            context = self._build_processor()
            final_scores = {
                "joy": 0.02,
                "sadness": 0.82,
                "anger": 0.01,
                "fear": 0.03,
                "anxiety": 0.05,
                "calm": 0.05,
                "confusion": 0.02,
            }
            final_metadata = {
                "source": "emotion2vec_audio+text",
                "audio_model_used": True,
                "inference_ms": 29.0,
            }
            with mock.patch(
                "src.voice.handlers.speech_turn_processor."
                "classify_multimodal_with_metadata",
                return_value=(final_scores, final_metadata),
            ) as classifier:
                await context.processor.process(
                    np.ones(16000, dtype=np.float32),
                    process_generation_token=3,
                    turn_id="turn-final-emotion",
                )

            classifier.assert_called_once()
            self.assertEqual(len(context.agent.process_calls), 1)
            self.assertEqual(
                context.agent.process_calls[0]["current_emotion"],
                "sadness",
            )
            final_insight = [
                payload
                for payload in context.connection.sent
                if payload.get("type") == "turn_insight"
                and payload.get("state") == "final"
            ][-1]
            self.assertEqual(
                final_insight["emotion"]["source"],
                "emotion2vec_audio+text",
            )

        asyncio.run(scenario())

    def test_slow_memory_lookup_skips_long_term_memory_without_sqlite_fallback(self):
        async def scenario():
            context = self._build_processor(memory_timeout_s=0.01)
            started = threading.Event()
            release = threading.Event()

            def slow_lookup(*_args):
                started.set()
                release.wait(1)
                return "过期的检索结果"

            context.processor._get_turn_memory = slow_lookup
            context.processor._get_cross_session_memory = (
                lambda _session: "SQLite 跨会话证据"
            )
            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-memory-timeout",
            )
            self.assertTrue(started.is_set())
            self.assertEqual(
                context.agent.turn_background_at_process,
                "",
            )
            release.set()

        asyncio.run(scenario())

    def test_speech_turn_captures_patient_evidence_once(self):
        async def scenario():
            context = self._build_processor()
            captures = []
            statuses = []
            context.processor._capture_memory_turn = (
                lambda *args, **kwargs: captures.append((args, kwargs))
            )
            context.processor._update_memory_status = (
                lambda *args, **kwargs: statuses.append((args, kwargs))
            )

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-once",
            )

            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0][1]["turn_id"], "turn-once")
            self.assertEqual(captures[0][0][2], "")
            self.assertEqual(
                statuses[0][1],
                {"turn_state": "SAFETY_GATED"},
            )
            self.assertEqual(
                statuses[1][1],
                {"turn_state": "GENERATING"},
            )
            self.assertEqual(
                statuses[-1][1],
                {
                    "assistant_message": "语音轮次回复。",
                    "response_status": "responded",
                    "turn_state": "RESPONDED",
                },
            )

        asyncio.run(scenario())

    def test_cancelling_turn_during_emotion_wait_propagates(self):
        async def scenario():
            context = self._build_processor()
            started = asyncio.Event()

            async def infer():
                started.set()
                await asyncio.Event().wait()

            emotion_task = asyncio.create_task(infer())
            turn = SimpleNamespace(emotion_task=emotion_task)
            owner = asyncio.create_task(context.processor._await_emotion_task(turn))
            await started.wait()
            await asyncio.sleep(0)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            self.assertTrue(emotion_task.cancelled())

        asyncio.run(scenario())

    def test_independently_cancelled_emotion_task_keeps_text_fallback(self):
        async def scenario():
            context = self._build_processor()
            emotion_task = asyncio.create_task(asyncio.sleep(60))
            emotion_task.cancel()
            turn = SimpleNamespace(emotion_task=emotion_task)
            await context.processor._await_emotion_task(turn)
            self.assertIsNone(turn.emotion_task)

        asyncio.run(scenario())

    def test_speech_turn_failure_marks_accepted_turn_failed(self):
        async def scenario():
            context = self._build_processor()
            statuses = []
            context.agent.error = RuntimeError("agent failed")
            context.processor._update_memory_status = (
                lambda *args, **kwargs: statuses.append(kwargs)
            )

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-failed",
            )

            self.assertIn(
                {
                    "response_status": "failed",
                    "turn_state": "FAILED",
                },
                statuses,
            )

        asyncio.run(scenario())

    def test_speech_turn_agent_supersession_marks_turn_cancelled(self):
        async def scenario():
            context = self._build_processor()
            statuses = []
            context.agent.result = {"superseded": True}
            context.processor._update_memory_status = (
                lambda *args, **kwargs: statuses.append(kwargs)
            )

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-cancelled",
            )

            self.assertIn(
                {
                    "response_status": "cancelled",
                    "turn_state": "RESPONSE_CANCELLED",
                },
                statuses,
            )

        asyncio.run(scenario())

    def test_soulx_paraformer_text_skips_second_asr(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-soulx",
                extra_meta={
                    "turn_taking": "soulx",
                    "soulx_text": "我直接说完了",
                    "soulx_asr_buffer": "说完了",
                },
            )

            self.assertEqual(context.asr_model.calls, [])
            self.assertEqual(len(context.agent.process_calls), 1)
            self.assertEqual(
                context.agent.process_calls[0]["user_input"],
                "我直接说完了",
            )
            asr_results = [
                payload
                for payload in context.connection.sent
                if payload["type"] == "asr_result"
            ]
            self.assertEqual(asr_results[0]["text"], "我直接说完了")

        asyncio.run(scenario())

    def test_streaming_asr_without_explicit_final_uses_full_audio_fallback(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)

            class RealtimeTurn:
                stream_error = None
                has_final = False

                async def final_text(self):
                    return "只有暂时文字"

            fallback_calls = []

            async def fallback(_audio):
                fallback_calls.append(True)
                return {
                    "text": "整段最终文字",
                    "emotion": "neutral",
                    "language": "zh",
                    "event": "Speech",
                }

            from src.tools.voice import ark_asr

            with mock.patch.object(
                ark_asr,
                "ark_asr_recognize",
                new=fallback,
            ):
                result = await context.processor._recognize_with_ark(
                    SimpleNamespace(
                        audio_data=np.ones(16000, dtype=np.float32),
                        extra_meta={"realtime_turn": RealtimeTurn()},
                        turn_id="turn-final-fallback",
                    )
                )

            self.assertEqual(result["text"], "整段最终文字")
            self.assertEqual(result["source"], "fallback_final")
            self.assertEqual(fallback_calls, [True])

    def test_explicit_streaming_final_is_used_without_full_audio_fallback(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)

            class RealtimeTurn:
                stream_error = None
                has_final = True

                async def final_text(self):
                    return "流式最终文字"

            from src.tools.voice import ark_asr

            with mock.patch.object(ark_asr, "ark_asr_recognize") as recognize:
                result = await context.processor._recognize_with_ark(
                    SimpleNamespace(
                        audio_data=np.ones(16000, dtype=np.float32),
                        extra_meta={"realtime_turn": RealtimeTurn()},
                        turn_id="turn-stream-final",
                    )
                )

            self.assertEqual(result["text"], "流式最终文字")
            recognize.assert_not_called()

        asyncio.run(scenario())

        asyncio.run(scenario())

    def test_failed_full_audio_fallback_keeps_audio_and_fails_closed(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)
            safety_events = []
            context.processor._record_safety_event = (
                lambda **fields: safety_events.append(fields)
            )

            from src.tools.voice import ark_asr

            async def failed_fallback(_audio):
                raise RuntimeError("network down")

            with mock.patch.object(
                ark_asr,
                "ark_asr_recognize",
                new=failed_fallback,
            ):
                await context.processor.process(
                    np.ones(16000, dtype=np.float32),
                    process_generation_token=3,
                    turn_id="turn-asr-failed",
                )

            self.assertEqual(context.agent.process_calls, [])
            self.assertEqual(len(context.audio_store.user_persisted), 1)
            self.assertEqual(
                context.audio_store.user_persisted[0][1]["asr_text"], ""
            )
            self.assertEqual(safety_events[0]["level"], "unknown")
            self.assertEqual(safety_events[0]["status"], "risk_scan_failed")
            self.assertEqual(
                [payload["type"] for payload in context.connection.sent],
                ["asr_error"],
            )

        asyncio.run(scenario())

    def test_completed_interrupt_asr_is_reused_for_formal_turn(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)
            audio = np.ones(16000, dtype=np.float32)
            from src.tools.voice import ark_asr

            with mock.patch.object(ark_asr, "ark_asr_recognize") as recognize:
                await context.processor.process(
                    audio,
                    process_generation_token=3,
                    turn_id="turn-reused-final",
                    extra_meta={
                        "final_asr_result": {
                            "text": "我心情真的很不好。", "source": "streaming_final",
                            "emotion": "neutral", "language": "zh", "event": "speech",
                        },
                        "final_asr_audio_samples": len(audio),
                    },
                )
            recognize.assert_not_called()
            self.assertEqual(context.agent.process_calls[0]["user_input"], "我心情真的很不好。")

        asyncio.run(scenario())

    def test_cached_partial_or_different_audio_is_recognized_again(self):
        async def scenario(source, sample_count):
            context = self._build_processor(use_ark_asr=True)
            from src.tools.voice import ark_asr

            recognize = mock.AsyncMock(return_value={
                "text": "完整音频的最终文字", "source": "fallback_final",
                "emotion": "neutral", "language": "zh", "event": "speech",
            })
            with mock.patch.object(ark_asr, "ark_asr_recognize", recognize):
                await context.processor.process(
                    np.ones(16000, dtype=np.float32),
                    process_generation_token=3,
                    turn_id="turn-needs-full-asr",
                    extra_meta={
                        "final_asr_result": {"text": "只有片段", "source": source},
                        "final_asr_audio_samples": sample_count,
                    },
                )
            recognize.assert_awaited_once()
            self.assertEqual(context.agent.process_calls[0]["user_input"], "完整音频的最终文字")

        for source, size in (("partial", 16000), ("streaming_final", 8000)):
            with self.subTest(source=source, size=size):
                asyncio.run(scenario(source, size))

    def test_empty_final_asr_reports_feedback_and_keeps_audio_without_answering(self):
        async def scenario():
            context = self._build_processor(use_ark_asr=True)
            from src.tools.voice import ark_asr

            with mock.patch.object(ark_asr, "ark_asr_recognize", mock.AsyncMock(return_value={"text": ""})):
                await context.processor.process(
                    np.ones(16000, dtype=np.float32),
                    process_generation_token=3,
                    turn_id="turn-empty-final",
                )
            self.assertEqual(context.agent.process_calls, [])
            self.assertEqual(len(context.audio_store.user_persisted), 1)
            errors = [item for item in context.connection.sent if item["type"] == "asr_error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["turn_id"], "turn-empty-final")
            self.assertEqual(errors[0]["reason"], "empty_final_transcript")

        asyncio.run(scenario())

    def test_streaming_speech_turn_emits_incremental_text_and_audio(self):
        async def scenario():
            context = self._build_processor(
                streaming=True,
                streamed_sentences=["语音第一句。", "语音第二句。"],
            )

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-streaming-voice",
            )

            payload_types = [
                payload["type"] for payload in context.connection.sent
            ]
            self.assertEqual(payload_types.count("ai_response_chunk"), 2)
            self.assertEqual(len(context.tts_calls), 2)
            self.assertIsNotNone(context.tts_calls[0][1]["start_payload"])
            self.assertIsNone(context.tts_calls[1][1]["start_payload"])
            self.assertEqual(len(context.audio_store.persisted), 1)
            self.assertEqual(
                context.audio_store.persisted[0][0].size,
                48000,
            )
            self.assertEqual(
                [entry["role"] for entry in context.session.chat_history],
                ["user", "assistant"],
            )

        asyncio.run(scenario())

    def test_interrupted_streaming_tts_closes_the_started_playback(self):
        async def scenario():
            context = self._build_processor(
                streaming=True,
                streamed_sentences=["语音第一句。"],
            )
            identity = {
                "session_id": "session-speech",
                "turn_id": "turn-preempted",
                "generation": 3,
                "playback_id": "playback-turn-preempted-3",
            }

            async def stream_tts_audio(_text, **kwargs):
                await context.connection.send_json(
                    {**kwargs["start_payload"], **identity}
                )
                return {
                    "chunks": 1,
                    "samples": 24000,
                    "first_latency": 0.1,
                    "interrupted": True,
                    "interruption_reason": "preempted",
                    "audio_data": np.ones(24000, dtype=np.float32),
                    "output_identity": identity,
                    "started": True,
                }

            context.processor._stream_tts_audio = stream_tts_audio
            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-preempted",
            )

            tts_end = next(
                payload
                for payload in context.connection.sent
                if payload["type"] == "tts_end"
            )
            self.assertEqual(tts_end["reason"], "preempted")
            self.assertEqual(tts_end["playback_id"], identity["playback_id"])

        asyncio.run(scenario())

    def test_streaming_failure_after_tts_start_closes_playback_with_error(self):
        async def scenario():
            context = self._build_processor(
                streaming=True,
                streamed_sentences=["语音第一句。"],
            )
            identity = {
                "session_id": "session-speech",
                "turn_id": "turn-error",
                "generation": 3,
                "playback_id": "playback-turn-error-3",
            }
            context.session.processing.active_turn_id = identity["turn_id"]
            context.session.processing.active_playback_id = identity["playback_id"]

            async def failing_tts(_text, **kwargs):
                await context.connection.send_json(
                    {**kwargs["start_payload"], **identity}
                )
                raise AttributeError("provider output missing")

            context.processor._stream_tts_audio = failing_tts
            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-error",
            )

            tts_ends = [
                payload
                for payload in context.connection.sent
                if payload["type"] == "tts_end"
            ]
            self.assertEqual(len(tts_ends), 1)
            self.assertEqual(tts_ends[0]["reason"], "error")
            self.assertEqual(tts_ends[0]["playback_id"], identity["playback_id"])
            self.assertEqual(tts_ends[0]["session_id"], identity["session_id"])
            self.assertEqual(tts_ends[0]["turn_id"], identity["turn_id"])
            self.assertTrue(
                any(
                    payload["type"] == "ai_response"
                    and payload.get("error")
                    for payload in context.connection.sent
                )
            )

        asyncio.run(scenario())

    def test_streaming_agent_failure_after_first_sentence_closes_playback(self):
        async def scenario():
            context = self._build_processor(
                streaming=True,
            )
            context.session.processing.active_turn_id = "turn-agent-error"
            context.session.processing.active_playback_id = (
                "playback-turn-agent-error-3"
            )

            def fail_after_sentence(**_kwargs):
                callback = context.agent.state._stream_sentence_cb
                if callback:
                    callback("语音第一句。")
                raise AttributeError("agent stream field missing")

            context.agent.process_turn = fail_after_sentence

            await context.processor.process(
                np.ones(16000, dtype=np.float32),
                process_generation_token=3,
                turn_id="turn-agent-error",
            )

            tts_ends = [
                payload
                for payload in context.connection.sent
                if payload["type"] == "tts_end"
            ]
            self.assertEqual(len(tts_ends), 1)
            self.assertEqual(tts_ends[0]["reason"], "error")
            self.assertEqual(
                tts_ends[0]["playback_id"],
                "playback-turn-agent-error-3",
            )

        asyncio.run(scenario())

    def test_generation_change_marks_turn_as_superseded(self):
        context = self._build_processor()

        self.assertFalse(context.processor.is_superseded(3))
        self.assertTrue(context.processor.is_superseded(2))
        context.session.runtime.stop_generate = True
        self.assertTrue(context.processor.is_superseded(3))
        self.assertTrue(context.processor.is_superseded(None))

    def test_entrypoint_constructs_processor_without_nested_process_speech(self):
        application = _voice_application_class()
        nested_process_speech = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "process_speech"
        ]
        processor_injections = [
            keyword
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SpeechProcessingCoordinator"
            for keyword in node.keywords
            if keyword.arg == "processor"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "speech_processor"
        ]

        self.assertEqual(nested_process_speech, [])
        self.assertEqual(len(processor_injections), 1)


class SoulXAudioHandlerTests(unittest.TestCase):
    class Unavailable(RuntimeError):
        pass

    @staticmethod
    def _build_handler(
        *,
        client,
        accumulator,
        is_ai_speaking=lambda: False,
        enabled_full_duplex=True,
        realtime_companion=None,
        barge_in_minimum_chunk_rms=0.0,
        is_processing=None,
        interrupt_active_processing=None,
    ):
        connection = _FakeConnection()
        speaker = _FakeSoulXSpeaker()
        turn_state = VoiceTurnTakingState()
        answer_completion = AnswerCompletionState()
        stopped = []
        reset_vad = []
        submissions = []
        notifications = []
        armed = []

        async def stop_playback():
            stopped.append(True)

        async def notify(*args, **kwargs):
            notifications.append((args, kwargs))

        async def arm():
            armed.append(True)

        async def submit(audio, **kwargs):
            submissions.append((np.asarray(audio).copy(), kwargs))

        handler = SoulXAudioHandler(
            connection,
            client=client,
            accumulator=accumulator,
            turn_state=turn_state,
            speaker=speaker,
            answer_completion=answer_completion,
            reset_local_vad=lambda: reset_vad.append(True),
            is_ai_speaking=is_ai_speaking,
            stop_playback=stop_playback,
            notify_answer_completion=notify,
            arm_answer_completion_window=arm,
            submit_speech=submit,
            audio_signal_stats=lambda _audio, _rate: {"rms": 0.1},
            unavailable_error_type=SoulXAudioHandlerTests.Unavailable,
            server_url="ws://soulx.test/turn",
            enabled_full_duplex=enabled_full_duplex,
            minimum_utterance_rms=0.01,
            barge_in_minimum_chunk_rms=barge_in_minimum_chunk_rms,
            is_processing=is_processing,
            interrupt_active_processing=interrupt_active_processing,
            realtime_companion=realtime_companion,
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            connection=connection,
            speaker=speaker,
            turn_state=turn_state,
            answer_completion=answer_completion,
            stopped=stopped,
            reset_vad=reset_vad,
            submissions=submissions,
            notifications=notifications,
            armed=armed,
            handler=handler,
        )

    def test_unavailable_soulx_falls_back_to_local_vad(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    raise SoulXAudioHandlerTests.Unavailable(
                        "connection refused"
                    )

            accumulator = _FakeSoulXAccumulator()
            context = self._build_handler(
                client=Client(),
                accumulator=accumulator,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertFalse(consumed)
            self.assertEqual(accumulator.reset_count, 1)
            self.assertFalse(context.turn_state.soulx_healthy)
            self.assertEqual(
                context.turn_state.soulx_last_error,
                "connection refused",
            )

        asyncio.run(scenario())

    def test_soulx_not_ready_switches_realtime_companion_to_local_asr(self):
        async def scenario():
            class Companion:
                def __init__(self):
                    self.external_asr = None

                def set_external_asr(self, enabled):
                    self.external_asr = enabled

            companion = Companion()

            class Client:
                retry_ready = False

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
                realtime_companion=companion,
            )
            assert not await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )
            assert companion.external_asr is False

        asyncio.run(scenario())

    def test_small_frames_batch_to_native_chunk_before_process(self):
        async def scenario():
            process_batches = []

            class Client:
                retry_ready = True

                async def process(self, audio):
                    process_batches.append(np.asarray(audio).copy())
                    return SimpleNamespace(
                        state="idle",
                        text="",
                        asr_buffer="",
                    )

            accumulator = _FakeSoulXAccumulator()
            context = self._build_handler(
                client=Client(),
                accumulator=accumulator,
            )

            for value in (1.0, 2.0):
                consumed = await context.handler.handle_audio(
                    np.full(1024, value, dtype=np.float32)
                )
                self.assertTrue(consumed)
            self.assertEqual(process_batches, [])
            self.assertEqual(accumulator.feed_calls, [])

            consumed = await context.handler.handle_audio(
                np.full(1024, 3.0, dtype=np.float32)
            )
            self.assertTrue(consumed)
            self.assertEqual(len(process_batches), 1)
            self.assertEqual(process_batches[0].size, 2560)
            np.testing.assert_allclose(
                process_batches[0],
                np.concatenate(
                    [
                        np.full(1024, 1.0, dtype=np.float32),
                        np.full(1024, 2.0, dtype=np.float32),
                        np.full(512, 3.0, dtype=np.float32),
                    ]
                ),
            )
            self.assertEqual(len(accumulator.feed_calls), 1)

            for value in (4.0, 5.0):
                consumed = await context.handler.handle_audio(
                    np.full(1024, value, dtype=np.float32)
                )
                self.assertTrue(consumed)
            self.assertEqual(len(process_batches), 2)
            np.testing.assert_allclose(
                process_batches[1],
                np.concatenate(
                    [
                        np.full(512, 3.0, dtype=np.float32),
                        np.full(1024, 4.0, dtype=np.float32),
                        np.full(1024, 5.0, dtype=np.float32),
                    ]
                ),
            )
            self.assertEqual(len(accumulator.feed_calls), 2)
            self.assertTrue(
                all(call[0].size == 2560 for call in accumulator.feed_calls)
            )

        asyncio.run(scenario())

    def test_five_512_frames_make_one_native_soulx_chunk(self):
        async def scenario():
            process_batches = []

            class Client:
                retry_ready = True

                async def process(self, audio):
                    process_batches.append(np.asarray(audio).copy())
                    return SimpleNamespace(
                        state="idle",
                        text="",
                        asr_buffer="",
                    )

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
            )

            for value in range(1, 6):
                consumed = await context.handler.handle_audio(
                    np.full(512, value, dtype=np.float32)
                )
                self.assertTrue(consumed)

            self.assertEqual(len(process_batches), 1)
            self.assertEqual(process_batches[0].size, 2560)
            np.testing.assert_allclose(
                process_batches[0],
                np.concatenate(
                    [
                        np.full(512, value, dtype=np.float32)
                        for value in range(1, 6)
                    ]
                ),
            )
            self.assertEqual(context.handler._batch_samples, 0)

        asyncio.run(scenario())

    def test_same_nonidle_state_streams_only_changed_partial_text(self):
        async def scenario():
            responses = iter(
                [
                    SimpleNamespace(
                        state="nonidle",
                        text="",
                        asr_buffer="你",
                    ),
                    SimpleNamespace(
                        state="nonidle",
                        text="",
                        asr_buffer="你好",
                    ),
                    SimpleNamespace(
                        state="nonidle",
                        text="",
                        asr_buffer="你好",
                    ),
                ]
            )

            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return next(responses)

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
            )

            for _ in range(3):
                consumed = await context.handler.handle_audio(
                    np.ones(2560, dtype=np.float32)
                )
                self.assertTrue(consumed)

            self.assertEqual(
                context.connection.sent,
                [
                    {
                        "type": "soulx_state",
                        "state": "nonidle",
                        "text": "你",
                    },
                    {
                        "type": "soulx_state",
                        "state": "nonidle",
                        "text": "你好",
                    },
                ],
            )

        asyncio.run(scenario())

    def test_same_external_state_streams_detailed_wait_progress(self):
        async def scenario():
            responses = iter(
                [
                    SimpleNamespace(
                        state="idle",
                        text="",
                        asr_buffer="今年",
                        raw_state="incomplete",
                        detail_state="incomplete",
                        decision_source="semantic_incomplete",
                        wait_idle_count=0,
                        max_wait_count=22,
                        monitoring_wait_silence=True,
                        speech_detected=True,
                        chunk_rms=0.03,
                    ),
                    SimpleNamespace(
                        state="idle",
                        text="",
                        asr_buffer="今年",
                        raw_state="idle",
                        detail_state="incomplete_wait",
                        decision_source="incomplete_wait",
                        wait_idle_count=1,
                        max_wait_count=22,
                        monitoring_wait_silence=True,
                        speech_detected=True,
                        chunk_rms=0.001,
                    ),
                ]
            )

            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return next(responses)

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
            )
            for _ in range(2):
                self.assertTrue(
                    await context.handler.handle_audio(
                        np.ones(2560, dtype=np.float32)
                    )
                )

            self.assertEqual(len(context.connection.sent), 2)
            self.assertEqual(
                context.connection.sent[0]["detail_state"],
                "incomplete",
            )
            self.assertEqual(
                context.connection.sent[1]["detail_state"],
                "incomplete_wait",
            )
            self.assertEqual(
                context.connection.sent[1]["wait_idle_count"],
                1,
            )

        asyncio.run(scenario())

    def test_nonidle_stops_ai_and_keeps_listening_for_speak(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="nonidle",
                        text="等一下",
                        asr_buffer="等一下",
                    )

            accumulator = _FakeSoulXAccumulator()
            context = self._build_handler(
                client=Client(),
                accumulator=accumulator,
                is_ai_speaking=lambda: True,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertTrue(consumed)
            self.assertEqual(context.stopped, [True])
            self.assertTrue(
                context.turn_state.soulx_barge_in_active
            )
            self.assertEqual(
                context.turn_state.soulx_last_state,
                "nonidle",
            )

        asyncio.run(scenario())

    def test_nonidle_cancels_pending_generation_before_it_starts_playing(self):
        async def scenario():
            session = VoiceSession(connection=_FakeConnection(), agent=None, owner_username="test")
            started = asyncio.Event()

            async def old_generation():
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    session.processing.is_active = False

            task = asyncio.create_task(old_generation())
            session.processing.task = task
            session.processing.is_active = True
            session.processing.revision_enabled = True
            await started.wait()
            coordinator = SpeechProcessingCoordinator(
                session, processor=None, enabled_full_duplex=True,
                is_ai_speaking=lambda: False, stop_playback=mock.AsyncMock(),
                disconnect_error_type=RuntimeError, client_id="test", logger=lambda _: None,
            )

            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SoulXTurnState(state="nonidle", asr_buffer="我今天", chunk_rms=0.08)

            context = self._build_handler(
                client=Client(), accumulator=_FakeSoulXAccumulator(),
                is_processing=lambda: session.processing.is_active,
                interrupt_active_processing=coordinator.interrupt_active,
            )
            self.assertTrue(await context.handler.handle_audio(np.ones(2560, dtype=np.float32)))
            self.assertTrue(task.cancelled())
            self.assertTrue(context.turn_state.soulx_barge_in_active)
            self.assertEqual(context.submissions, [])

        asyncio.run(scenario())

    def test_speak_cancels_pending_generation_before_submitting_next_turn(self):
        async def scenario():
            order = []

            async def cancel(_reason):
                order.append("cancel")

            async def submit(_audio, **_kwargs):
                order.append("submit")

            context = self._build_handler(
                client=None, accumulator=_FakeSoulXAccumulator(),
                is_processing=lambda: True, interrupt_active_processing=cancel,
            )
            context.handler._submit_speech = submit
            await context.handler._handle_speak(
                SoulXTurnState(state="speak", text="我今天心情不太好。"),
                np.ones(16000, dtype=np.float32),
            )
            self.assertEqual(order, ["cancel", "submit"])

        asyncio.run(scenario())

    def test_low_energy_nonidle_does_not_interrupt_when_gate_is_set(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="nonidle",
                        text="嗯",
                        asr_buffer="嗯",
                        chunk_rms=0.002,
                    )

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
                is_ai_speaking=lambda: True,
                barge_in_minimum_chunk_rms=0.01,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertTrue(consumed)
            self.assertEqual(context.stopped, [])
            self.assertFalse(context.turn_state.soulx_barge_in_active)

        asyncio.run(scenario())

    def test_loud_nonidle_still_interrupts_when_gate_is_set(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="nonidle",
                        text="等一下",
                        asr_buffer="等一下",
                        chunk_rms=0.05,
                    )

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
                is_ai_speaking=lambda: True,
                barge_in_minimum_chunk_rms=0.01,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertTrue(consumed)
            self.assertEqual(context.stopped, [True])
            self.assertTrue(context.turn_state.soulx_barge_in_active)

        asyncio.run(scenario())

    def test_missing_chunk_rms_still_interrupts_rather_than_blocking(self):
        """A server that omits chunk_rms must not silently disable barge-in."""

        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="nonidle",
                        text="等一下",
                        asr_buffer="等一下",
                    )

            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(),
                is_ai_speaking=lambda: True,
                barge_in_minimum_chunk_rms=0.01,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertTrue(consumed)
            self.assertEqual(context.stopped, [True])
            self.assertTrue(context.turn_state.soulx_barge_in_active)

        asyncio.run(scenario())

    def test_speak_submits_complete_audio_with_soulx_metadata(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="speak",
                        text="我说完了",
                        asr_buffer="说完了",
                        raw_state="complete",
                        detail_state="complete",
                        decision_source="semantic_complete",
                        wait_idle_count=0,
                        max_wait_count=22,
                        monitoring_wait_silence=False,
                        speech_detected=True,
                        chunk_rms=0.1,
                    )

            complete_audio = np.ones(4000, dtype=np.float32)
            accumulator = _FakeSoulXAccumulator(
                complete_audio=complete_audio
            )
            context = self._build_handler(
                client=Client(),
                accumulator=accumulator,
            )

            consumed = await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )

            self.assertTrue(consumed)
            self.assertEqual(context.reset_vad, [True])
            self.assertEqual(
                context.connection.sent,
                [
                    {
                        "type": "soulx_state",
                        "state": "speak",
                        "text": "我说完了",
                        "raw_state": "complete",
                        "detail_state": "complete",
                        "decision_source": "semantic_complete",
                        "wait_idle_count": 0,
                        "max_wait_count": 22,
                        "monitoring_wait_silence": False,
                        "speech_detected": True,
                        "chunk_rms": 0.1,
                    },
                    {"type": "vad_end"},
                ],
            )
            np.testing.assert_allclose(
                context.submissions[0][0],
                complete_audio,
            )
            self.assertEqual(
                context.submissions[0][1],
                {
                    "source": "SoulX语义轮次",
                    "extra_meta": {
                        "turn_taking": "soulx",
                        "soulx_text": "我说完了",
                        "soulx_asr_buffer": "说完了",
                        "soulx_raw_state": "complete",
                        "soulx_detail_state": "complete",
                        "soulx_decision_source": "semantic_complete",
                        "soulx_wait_idle_count": 0,
                        "soulx_max_wait_count": 22,
                    },
                },
            )

        asyncio.run(scenario())

    def test_speak_passes_soulx_turn_to_formal_processor(self):
        async def scenario():
            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return SimpleNamespace(
                        state="speak",
                        text="完整表达",
                        asr_buffer="完整表达",
                    )

            realtime_turn = object()

            class Companion:
                def __init__(self):
                    self.external_asr = []
                    self.frames = []

                def set_external_asr(self, enabled):
                    self.external_asr.append(enabled)

                async def observe_soulx_frame(
                    self, audio, *, state, text, detail_state, speech_detected,
                    complete_audio
                ):
                    self.frames.append(
                        (audio, state, text, complete_audio, detail_state, speech_detected)
                    )
                    return realtime_turn

            companion = Companion()
            context = self._build_handler(
                client=Client(),
                accumulator=_FakeSoulXAccumulator(
                    complete_audio=np.ones(4000, dtype=np.float32)
                ),
                realtime_companion=companion,
            )

            assert await context.handler.handle_audio(
                np.ones(2560, dtype=np.float32)
            )
            assert companion.external_asr == [True]
            assert len(companion.frames) == 1
            assert companion.frames[0][1:3] == ("speak", "完整表达")
            assert companion.frames[0][4:] == ("", None)
            assert context.submissions[0][1]["extra_meta"]["realtime_turn"] is realtime_turn

        asyncio.run(scenario())

    def test_native_soulx_frames_select_short_acknowledgement_and_allow_speech(self):
        async def scenario():
            state = SoulXTurnState(
                state="nonidle", detail_state="incomplete", speech_detected=True
            )

            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return state

            connection = _FakeConnection()
            spoken = []
            playing = asyncio.Event()

            async def stream_tts(text, **_kwargs):
                spoken.append(text)
                playing.set()
                await asyncio.Event().wait()

            companion = RealtimeCompanion(
                connection,
                session=VoiceSession(connection=connection, agent=None, owner_username="test"),
                patient_memory_service=None,
                stream_tts_audio=stream_tts,
                config=RealtimeCompanionConfig(enabled=True, external_asr=True),
                classify_window=lambda *_args: {},
            )
            context = self._build_handler(
                client=Client(), accumulator=SoulXAudioAccumulator(),
                realtime_companion=companion,
                is_ai_speaking=playing.is_set,
            )
            for _ in range(31):
                await context.handler.handle_audio(np.ones(2560, dtype=np.float32))
                await asyncio.sleep(0)
            self.assertEqual(spoken, [])
            await context.handler.handle_audio(np.ones(2560, dtype=np.float32))
            await asyncio.wait_for(playing.wait(), timeout=1.0)
            for _ in range(3):
                await context.handler.handle_audio(np.ones(2560, dtype=np.float32))
                await asyncio.sleep(0)
            self.assertEqual(spoken, ["嗯嗯"])
            self.assertEqual(context.stopped, [])
            self.assertTrue(companion.backchannel_playing)
            await companion.reset()

        asyncio.run(scenario())

    def test_soulx_pause_prompt_stops_when_patient_resumes_even_after_prior_barge_in(self):
        async def scenario():
            state = SoulXTurnState(
                state="nonidle", detail_state="listening", speech_detected=True
            )

            class Client:
                retry_ready = True

                async def process(self, _audio):
                    return state

            spoken = []
            playing = asyncio.Event()

            async def stream_tts(text, **_kwargs):
                spoken.append(text)
                playing.set()
                await asyncio.Event().wait()

            connection = _FakeConnection()
            companion = RealtimeCompanion(
                connection,
                session=VoiceSession(connection=connection, agent=None, owner_username="test"),
                patient_memory_service=None,
                stream_tts_audio=stream_tts,
                config=RealtimeCompanionConfig(enabled=True, external_asr=True),
                classify_window=lambda *_args: {},
            )
            context = self._build_handler(
                client=Client(), accumulator=SoulXAudioAccumulator(),
                realtime_companion=companion, is_ai_speaking=playing.is_set,
            )
            for _ in range(43):
                await context.handler.handle_audio(np.ones(2560, dtype=np.float32))
            state = SoulXTurnState(
                state="idle", detail_state="incomplete_wait", speech_detected=False
            )
            await context.handler.handle_audio(np.zeros(2560, dtype=np.float32))
            await asyncio.wait_for(playing.wait(), timeout=1.0)
            self.assertEqual(spoken, ["我在听，您慢慢说。"])
            context.turn_state.soulx_barge_in_active = True
            playback = companion._comfort_task
            state = SoulXTurnState(
                state="idle", detail_state="incomplete", speech_detected=True
            )
            await context.handler.handle_audio(np.ones(2560, dtype=np.float32))
            self.assertEqual(context.stopped, [True])
            self.assertTrue(playback.cancelled())
            await companion.reset()

        asyncio.run(scenario())

    def test_entrypoint_delegates_soulx_state_machine_to_handler(self):
        application = _voice_application_class()
        direct_client_process_calls = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "soulx_client"
            and node.func.attr == "process"
        ]
        live_audio_injections = [
            keyword
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LiveAudioInputHandler"
            for keyword in node.keywords
            if keyword.arg == "soulx_audio_handler"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "soulx_handler"
        ]

        self.assertEqual(direct_client_process_calls, [])
        self.assertEqual(len(live_audio_injections), 1)


class _FakeLocalVADBuffer:
    WEAK_SPEECH_THRESHOLD = 0.2

    def __init__(self, *, complete_audio=None):
        self.complete_audio = complete_audio
        self.is_speaking = False
        self.add_calls = []
        self.reset_count = 0

    def add_chunk(self, audio):
        self.add_calls.append(np.asarray(audio).copy())
        return self.complete_audio

    @staticmethod
    def consume_drop_feedback():
        return None, 0.0

    @staticmethod
    def has_speech(_audio):
        return 0.0

    @staticmethod
    def _chunk_rms(_audio):
        return 0.0

    def reset(self):
        self.reset_count += 1
        self.is_speaking = False


class _FakeSoulXRoute:
    def __init__(self, consumed=False):
        self.consumed = consumed
        self.calls = []

    async def handle_audio(self, audio):
        self.calls.append(np.asarray(audio).copy())
        return self.consumed


class LiveAudioInputHandlerTests(unittest.TestCase):
    def test_sparse_noise_cannot_accumulate_into_an_early_stop(self):
        async def scenario():
            context = self._build_handler(started=True)
            handler = context.handler
            handler.config = replace(handler.config, enabled_full_duplex=True)
            handler._is_ai_speaking = lambda *_: True
            handler._stop_playback = mock.AsyncMock()
            handler.vad_buffer._chunk_rms = lambda _: 0.1
            for probability in [0.95, 0.95, 0.05] * 10:
                handler.vad_buffer.has_speech = lambda _, p=probability: p
                await handler.handle_message({
                    "type": "audio", "_audio_float": np.full(512, 0.1, dtype=np.float32),
                })
            handler._stop_playback.assert_not_awaited()
        asyncio.run(scenario())

    def test_seven_voiced_frames_stop_ai_without_waiting_for_recognition(self):
        async def scenario():
            context = self._build_handler(started=True)
            handler = context.handler
            handler.config = replace(handler.config, enabled_full_duplex=True)
            handler._is_ai_speaking = lambda *_: True
            handler._stop_playback = mock.AsyncMock()
            handler._quick_asr = mock.AsyncMock(side_effect=AssertionError("slow ASR"))
            handler.vad_buffer._chunk_rms = lambda _: 0.1
            handler.vad_buffer.has_speech = lambda _: 0.95
            for _ in range(6):
                await handler.handle_message({
                    "type": "audio", "_audio_float": np.full(512, 0.1, dtype=np.float32),
                })
            handler._stop_playback.assert_not_awaited()
            await handler.handle_message({
                "type": "audio", "_audio_float": np.full(512, 0.1, dtype=np.float32),
            })
            handler._stop_playback.assert_awaited_once()
            handler._quick_asr.assert_not_awaited()
            self.assertEqual(context.vad_buffer.reset_count, 0)
        asyncio.run(scenario())

    def test_scheduled_backchannel_is_not_cut_off_by_patient_speech(self):
        async def scenario():
            context = self._build_handler(started=True)
            handler = context.handler
            handler.config = replace(handler.config, enabled_full_duplex=True)
            handler._is_ai_speaking = lambda *_: True
            handler._stop_playback = mock.AsyncMock()
            handler.vad_buffer._chunk_rms = lambda _: 0.1
            handler.vad_buffer.has_speech = lambda _: 0.95
            handler._realtime_companion = SimpleNamespace(
                backchannel_playing=True, observe_frame=mock.AsyncMock(return_value=None),
            )
            for _ in range(12):
                await handler.handle_message({
                    "type": "audio", "_audio_float": np.full(512, 0.1, dtype=np.float32),
                })
            handler._stop_playback.assert_not_awaited()
        asyncio.run(scenario())

    @staticmethod
    def _build_handler(*, started, complete_audio=None, soulx_consumed=False):
        connection = _FakeConnection()
        session = VoiceSession(
            connection=connection,
            agent=object(),
            owner_username="doctor",
        )
        session.lifecycle.started = started
        vad_buffer = _FakeLocalVADBuffer(
            complete_audio=complete_audio
        )
        soulx_route = _FakeSoulXRoute(consumed=soulx_consumed)
        speaker = _FakeSoulXSpeaker()
        answer_completion = AnswerCompletionState()
        history_store = _FakeHistoryStore()
        reset_calls = []
        submissions = []
        feedback = []

        def reset_interrupt_capture(**kwargs):
            reset_calls.append(kwargs)
            session.runtime.reset_interrupt_capture(
                reset_waiting=kwargs.get("reset_waiting", False)
            )
            if kwargs.get("reset_vad"):
                vad_buffer.reset()

        async def submit(audio, **kwargs):
            submissions.append((np.asarray(audio).copy(), kwargs))

        async def notify_feedback(*args, **kwargs):
            feedback.append((args, kwargs))

        handler = LiveAudioInputHandler(
            connection,
            session=session,
            vad_buffer=vad_buffer,
            soulx_audio_handler=soulx_route,
            speaker=speaker,
            answer_completion=answer_completion,
            history_store=history_store,
            is_ai_speaking=lambda *_args: False,
            reset_interrupt_capture=reset_interrupt_capture,
            remember_interrupt_judgement=lambda *_args: None,
            reuse_interrupt_judgement=lambda _audio: (None, None),
            judge_waiting_completion=lambda *_args, **_kwargs: ("", ""),
            interrupt_active_processing=lambda _reason: None,
            stop_playback=lambda: None,
            get_active_processing_audio=lambda: None,
            notify_voice_input_feedback=notify_feedback,
            submit_speech=submit,
            notify_answer_completion=lambda *_args, **_kwargs: None,
            arm_answer_completion_window=lambda: None,
            quick_asr=lambda _audio: "",
            judge_interrupt_intent=lambda _text: "backchannel",
            stream_tts_audio=lambda *_args, **_kwargs: {
                "samples": 0,
                "started": False,
            },
            config=LiveAudioConfig(
                enabled_full_duplex=False,
                pre_end_arm_window_s=0.6,
                minimum_post_tts_chunks=2,
                interrupt_min_duration=0.5,
                interrupt_trigger_probability=0.7,
                interrupt_min_rms=0.01,
                interrupt_min_consecutive_chunks=2,
                interrupt_complete_silence_s=0.8,
                interrupt_min_complete_audio_s=0.4,
            ),
            logger=lambda _message: None,
        )
        return SimpleNamespace(
            connection=connection,
            session=session,
            vad_buffer=vad_buffer,
            soulx_route=soulx_route,
            reset_calls=reset_calls,
            submissions=submissions,
            feedback=feedback,
            handler=handler,
        )

    def test_revision_stops_playback_when_generation_finishes_during_final_asr(self):
        async def scenario():
            context = self._build_handler(started=True)
            processing = context.session.processing
            processing.is_active = True
            processing.revision_enabled = True
            processing.active_audio = np.ones(16000, dtype=np.float32)
            playing = []
            stops = []

            async def stop():
                stops.append(True)
                playing.clear()

            coordinator = SpeechProcessingCoordinator(
                context.session, processor=None, enabled_full_duplex=True,
                is_ai_speaking=lambda: bool(playing), stop_playback=stop,
                disconnect_error_type=RuntimeError, client_id="test", logger=lambda _: None,
            )

            class RealtimeTurn:
                has_final = True
                stream_error = None

                async def final_text(self):
                    # The previous generation completes while the final ASR packet arrives.
                    processing.is_active = False
                    processing.revision_enabled = False
                    processing.active_audio = None
                    playing.append(True)
                    return "我心情真的很不好。"

            handler = context.handler
            handler._is_ai_speaking = lambda *_: bool(playing)
            handler._interrupt_active_processing = coordinator.interrupt_active
            handler._get_active_processing_audio = coordinator.active_audio_copy
            handler._judge_interrupt_intent = mock.AsyncMock(return_value="complete")
            handler._quick_asr = mock.AsyncMock(side_effect=AssertionError("duplicate ASR"))
            frame = LiveAudioFrameContext(
                audio=np.ones(512), complete_audio=np.ones(53760), realtime_turn=RealtimeTurn(),
            )
            await handler._handle_complete_audio(frame)

            self.assertEqual(stops, [True])
            self.assertFalse(playing)
            self.assertEqual(len(context.submissions), 1)
            metadata = context.submissions[0][1]["extra_meta"]
            self.assertEqual(metadata["final_asr_result"]["text"], "我心情真的很不好。")
            self.assertIs(metadata["realtime_turn"], frame.realtime_turn)
            handler._quick_asr.assert_not_awaited()

        asyncio.run(scenario())

    def test_partial_streaming_result_uses_one_full_asr_and_forwards_it(self):
        async def scenario():
            context = self._build_handler(started=True)
            handler = context.handler
            frame = LiveAudioFrameContext(
                audio=np.ones(512), complete_audio=np.ones(16000),
                realtime_turn=SimpleNamespace(
                    has_final=False, stream_error=None,
                    final_text=mock.AsyncMock(return_value="尚未确认的片段"),
                ),
            )
            handler._quick_asr = mock.AsyncMock(return_value="完整表达")
            self.assertEqual(await handler._recognize_complete_audio(frame), "完整表达")
            await handler._handle_complete_audio(frame)
            handler._quick_asr.assert_awaited_once()
            metadata = context.submissions[0][1]["extra_meta"]
            self.assertEqual(metadata["final_asr_result"]["text"], "完整表达")
            self.assertEqual(metadata["final_asr_result"]["source"], "fallback_final")

        asyncio.run(scenario())

    def test_merged_revision_does_not_reuse_only_the_new_fragment_transcript(self):
        async def scenario():
            context = self._build_handler(started=True)
            old_audio = np.ones(8000, dtype=np.float32)
            new_audio = np.full(8000, 2, dtype=np.float32)
            handler = context.handler
            handler._get_active_processing_audio = lambda: old_audio
            handler._interrupt_active_processing = mock.AsyncMock()
            frame = LiveAudioFrameContext(
                audio=np.ones(512), complete_audio=new_audio,
                final_recognition={"text": "补充内容", "source": "streaming_final"},
            )
            await handler._replace_active_processing(frame)
            audio, kwargs = context.submissions[0]
            np.testing.assert_array_equal(audio, np.concatenate([old_audio, new_audio]))
            self.assertNotIn("final_asr_result", kwargs.get("extra_meta", {}))

        asyncio.run(scenario())

    def test_stalled_final_asr_closes_stream_and_falls_back_once(self):
        async def scenario():
            context = self._build_handler(started=True)
            closed = asyncio.Event()

            class StalledStream:
                async def start(self):
                    pass

                async def finish(self):
                    await asyncio.Event().wait()

                async def aclose(self):
                    closed.set()

            companion = RealtimeCompanion(
                context.connection, session=context.session,
                patient_memory_service=None, stream_tts_audio=mock.AsyncMock(),
                config=RealtimeCompanionConfig(enabled=True),
                stream_factory=lambda **_: StalledStream(), logger=lambda _: None,
            )
            turn = RealtimeTurn(companion)
            turn.start()
            turn.finish()
            frame = LiveAudioFrameContext(
                audio=np.ones(512), complete_audio=np.ones(16000), realtime_turn=turn,
            )
            handler = context.handler
            handler._quick_asr = mock.AsyncMock(return_value="完整表达")
            try:
                text = await asyncio.wait_for(handler._recognize_complete_audio(frame), 2.0)
                self.assertEqual(text, "完整表达")
                self.assertTrue(closed.is_set())
                self.assertFalse(turn.has_final)
                await handler._handle_complete_audio(frame)
                handler._quick_asr.assert_awaited_once()
                self.assertEqual(len(context.submissions), 1)
            finally:
                await turn.aclose()

        asyncio.run(scenario())

    def test_late_speaker_check_cannot_interrupt_a_new_generation(self):
        async def scenario():
            context = self._build_handler(started=True)
            handler = context.handler
            handler._is_ai_speaking = lambda *_: True
            handler._stop_playback = mock.AsyncMock()
            handler._interrupt_active_processing = mock.AsyncMock()
            handler._quick_asr = mock.AsyncMock(return_value="过期的输入")
            handler._judge_interrupt_intent = mock.AsyncMock(return_value="complete")

            async def verify(_audio):
                context.session.processing.generation += 1
                return True

            handler._verify_complete_speaker = verify
            frame = LiveAudioFrameContext(audio=np.ones(512), complete_audio=np.ones(16000))
            self.assertTrue(await handler._handle_ai_speaking_utterance(frame))
            handler._stop_playback.assert_not_awaited()
            handler._interrupt_active_processing.assert_not_awaited()
            handler._judge_interrupt_intent.assert_not_awaited()

        asyncio.run(scenario())

    def test_late_revision_cannot_interrupt_a_new_generation(self):
        async def scenario():
            context = self._build_handler(started=True)
            processing = context.session.processing
            processing.is_active = processing.revision_enabled = True

            async def recognize(_audio):
                processing.generation += 1
                return "过期的输入"

            handler = context.handler
            handler._quick_asr = recognize
            handler._judge_interrupt_intent = mock.AsyncMock(return_value="complete")
            handler._interrupt_active_processing = mock.AsyncMock()
            frame = LiveAudioFrameContext(audio=np.ones(512), complete_audio=np.ones(16000))
            await handler._handle_complete_audio(frame)
            self.assertEqual(context.submissions, [])
            handler._interrupt_active_processing.assert_not_awaited()

        asyncio.run(scenario())

    def test_audio_before_session_start_is_safely_discarded(self):
        async def scenario():
            context = self._build_handler(started=False)

            handled = await context.handler.handle_message(
                {
                    "type": "audio",
                    "_audio_float": np.ones(320, dtype=np.float32),
                }
            )

            self.assertTrue(handled)
            self.assertEqual(context.soulx_route.calls, [])
            self.assertEqual(context.vad_buffer.add_calls, [])

        asyncio.run(scenario())

    def test_soulx_consumed_audio_does_not_reach_local_vad(self):
        async def scenario():
            context = self._build_handler(
                started=True,
                soulx_consumed=True,
            )

            await context.handler.handle_message(
                {
                    "type": "audio",
                    "_audio_float": np.ones(320, dtype=np.float32),
                }
            )

            self.assertEqual(len(context.soulx_route.calls), 1)
            self.assertEqual(context.vad_buffer.add_calls, [])

        asyncio.run(scenario())

    def test_completed_local_vad_audio_is_submitted_once(self):
        async def scenario():
            complete_audio = np.ones(4000, dtype=np.float32)
            context = self._build_handler(
                started=True,
                complete_audio=complete_audio,
            )

            await context.handler.handle_message(
                {
                    "type": "audio",
                    "_audio_float": np.ones(320, dtype=np.float32),
                }
            )

            self.assertEqual(
                context.connection.sent,
                [{"type": "vad_end"}],
            )
            self.assertEqual(len(context.submissions), 1)
            np.testing.assert_allclose(
                context.submissions[0][0],
                complete_audio,
            )
            self.assertEqual(
                context.submissions[0][1],
                {"source": "语音输入"},
            )
            self.assertEqual(
                context.reset_calls,
                [{"reset_waiting": True}],
            )

        asyncio.run(scenario())

    def test_full_duplex_stops_before_asr_and_preserves_utterance(self):
        async def scenario():
            class InterruptVAD(_FakeLocalVADBuffer):
                @staticmethod
                def has_speech(_audio):
                    return 0.95

                @staticmethod
                def _chunk_rms(_audio):
                    return 0.1

            connection = _FakeConnection()
            session = VoiceSession(
                connection=connection,
                agent=object(),
                owner_username="doctor",
            )
            session.lifecycle.started = True
            vad_buffer = InterruptVAD()
            reset_calls = []
            stop_calls = []
            submissions = []
            playing = [True]

            def reset_interrupt_capture(**kwargs):
                reset_calls.append(kwargs)
                session.runtime.reset_interrupt_capture(
                    reset_waiting=kwargs.get("reset_waiting", False)
                )
                if kwargs.get("reset_vad"):
                    vad_buffer.reset()

            async def quick_asr(_audio):
                raise AssertionError("ASR must not block playback interruption")

            async def judge_interrupt_intent(_text):
                raise AssertionError("Intent must not block playback interruption")

            async def stop_playback():
                stop_calls.append(True)
                playing.clear()

            async def submit_speech(audio, **_kwargs):
                submissions.append(audio.copy())

            handler = LiveAudioInputHandler(
                connection,
                session=session,
                vad_buffer=vad_buffer,
                soulx_audio_handler=_FakeSoulXRoute(consumed=False),
                speaker=_FakeSoulXSpeaker(),
                answer_completion=AnswerCompletionState(),
                history_store=_FakeHistoryStore(),
                is_ai_speaking=lambda *_args: bool(playing),
                reset_interrupt_capture=reset_interrupt_capture,
                remember_interrupt_judgement=lambda *_args: None,
                reuse_interrupt_judgement=lambda _audio: (None, None),
                judge_waiting_completion=lambda *_args, **_kwargs: (
                    "",
                    "",
                ),
                interrupt_active_processing=lambda _reason: None,
                stop_playback=stop_playback,
                get_active_processing_audio=lambda: None,
                notify_voice_input_feedback=lambda *_args, **_kwargs: None,
                submit_speech=submit_speech,
                notify_answer_completion=lambda *_args, **_kwargs: None,
                arm_answer_completion_window=lambda: None,
                quick_asr=quick_asr,
                judge_interrupt_intent=judge_interrupt_intent,
                stream_tts_audio=lambda *_args, **_kwargs: {
                    "samples": 0,
                    "started": False,
                },
                config=LiveAudioConfig(
                    enabled_full_duplex=True,
                    pre_end_arm_window_s=0.6,
                    minimum_post_tts_chunks=2,
                    interrupt_min_duration=0.1,
                    interrupt_trigger_probability=0.7,
                    interrupt_min_rms=0.01,
                    interrupt_min_consecutive_chunks=1,
                    interrupt_complete_silence_s=0.8,
                    interrupt_min_complete_audio_s=0.4,
                ),
                logger=lambda _message: None,
            )

            await handler.handle_message(
                {
                    "type": "audio",
                    "_audio_float": np.ones(3200, dtype=np.float32),
                }
            )

            self.assertEqual(stop_calls, [True])
            self.assertEqual(
                reset_calls,
                [{"reset_waiting": True}],
            )
            self.assertFalse(session.runtime.waiting_for_complete)
            self.assertEqual(vad_buffer.reset_count, 0)
            self.assertEqual(submissions, [])

            complete_audio = np.arange(16000, dtype=np.float32)
            vad_buffer.complete_audio = complete_audio
            await handler.handle_message({
                "type": "audio", "_audio_float": np.zeros(512, dtype=np.float32),
            })
            self.assertEqual(stop_calls, [True])
            self.assertEqual(len(submissions), 1)
            np.testing.assert_array_equal(submissions[0], complete_audio)

        asyncio.run(scenario())

    def test_entrypoint_routes_audio_without_inline_audio_branch(self):
        application = _voice_application_class()
        inline_audio_comparisons = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "audio"
                for comparator in node.comparators
            )
        ]
        live_handler_bindings = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Tuple)
            and any(
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "LiveAudioInputHandler"
                and item.attr == "MESSAGE_TYPES"
                for item in node.elts
            )
            and any(
                isinstance(item, ast.Name)
                and item.id == "live_handler"
                for item in node.elts
            )
        ]

        self.assertEqual(inline_audio_comparisons, [])
        self.assertEqual(len(live_handler_bindings), 1)


class SpeakerVerificationControllerTests(unittest.TestCase):
    def test_unrelated_message_is_not_consumed(self):
        controller = SpeakerVerificationController(
            _FakeConnection(),
            verifier_factory=lambda: _FakeVerifier(),
        )

        handled = asyncio.run(controller.handle_message({"type": "ping"}))

        self.assertFalse(handled)

    def test_session_flags_are_isolated_even_when_model_is_shared(self):
        verifier = _FakeVerifier()
        first = SpeakerVerificationController(
            _FakeConnection(),
            verifier=verifier,
        )
        second = SpeakerVerificationController(
            _FakeConnection(),
            verifier=verifier,
        )

        first.enabled = True
        first.enrolled = True

        self.assertFalse(second.enabled)
        self.assertFalse(second.enrolled)
        self.assertIs(first.verifier, second.verifier)

    def test_toggle_requires_an_enrollment(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )

            handled = await controller.handle_message(
                {"type": "toggle_speaker_verify", "enabled": True}
            )

            self.assertTrue(handled)
            self.assertFalse(controller.enabled)
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["speaker_verify_status", "speaker_error"],
            )

        asyncio.run(scenario())

    def test_loading_a_speaker_enables_verification_for_this_session(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )

            await controller.handle_message(
                {"type": "load_speaker", "name": "患者甲"}
            )

            self.assertTrue(controller.enrolled)
            self.assertTrue(controller.enabled)
            self.assertEqual(verifier.speaker_name, "患者甲")
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["speaker_loaded", "speaker_verify_status"],
            )

        asyncio.run(scenario())

    def test_non_target_verification_sends_feedback(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            verifier.is_enrolled = True
            verifier.verify_result = (False, 0.42)
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )
            controller.enabled = True
            controller.enrolled = True

            result = await controller.verify(
                np.zeros(1600, dtype=np.float32),
                source="main",
                turn_id="turn-1",
            )

            self.assertEqual(result, (False, 0.42))
            self.assertEqual(connection.sent[-1]["type"], "voice_input_feedback")
            self.assertEqual(connection.sent[-1]["reason"], "speaker_non_target")
            self.assertEqual(connection.sent[-1]["turn_id"], "turn-1")

        asyncio.run(scenario())

    def test_repeated_verifier_issue_is_only_sent_once(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            verifier.last_error = "旧声纹不兼容"
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )
            controller.enabled = True

            audio = np.zeros(1600, dtype=np.float32)
            await controller.verify(audio)
            await controller.verify(audio)

            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["speaker_error"],
            )

        asyncio.run(scenario())

    def test_chunked_enrollment_is_reassembled_and_processed(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            samples = np.array([100, -100, 200, -200], dtype=np.int16)
            encoded = base64.b64encode(samples.tobytes()).decode("ascii")
            midpoint = len(encoded) // 2
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )

            await controller.handle_message(
                {
                    "type": "enroll_speaker_blob_start",
                    "upload_id": "upload-1",
                    "total_chunks": 2,
                    "sample_rate": 16000,
                }
            )
            await controller.handle_message(
                {
                    "type": "enroll_speaker_blob_chunk",
                    "upload_id": "upload-1",
                    "index": 1,
                    "data": encoded[midpoint:],
                }
            )
            await controller.handle_message(
                {
                    "type": "enroll_speaker_blob_chunk",
                    "upload_id": "upload-1",
                    "index": 0,
                    "data": encoded[:midpoint],
                }
            )
            await controller.handle_message(
                {
                    "type": "enroll_speaker_blob_end",
                    "upload_id": "upload-1",
                }
            )

            self.assertEqual(controller.uploads, {})
            self.assertEqual(len(verifier.added_paths), 1)
            self.assertFalse(Path(verifier.added_paths[0]).exists())
            self.assertEqual(connection.sent[-1]["type"], "speaker_sample_added")
            self.assertEqual(connection.sent[-1]["current"], 1)

        asyncio.run(scenario())

    def test_warning_for_disabled_verification_is_sent_once(self):
        async def scenario():
            connection = _FakeConnection()
            verifier = _FakeVerifier()
            verifier.is_enrolled = True
            controller = SpeakerVerificationController(
                connection,
                verifier=verifier,
            )

            first = await controller.warn_if_disabled()
            second = await controller.warn_if_disabled()

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                ["speaker_warning"],
            )

        asyncio.run(scenario())

    def test_entrypoint_delegates_speaker_messages_to_controller(self):
        application = _voice_application_class()
        speaker_types = SpeakerVerificationController.MESSAGE_TYPES

        inline_speaker_branches = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Constant) and node.value in speaker_types
        ]
        speaker_handler_bindings = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Tuple)
            and any(
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "SpeakerVerificationController"
                and item.attr == "MESSAGE_TYPES"
                for item in node.elts
            )
            and any(
                isinstance(item, ast.Name)
                and item.id == "speaker"
                for item in node.elts
            )
        ]
        controller_router_bindings = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VoiceConnectionController"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "router"
        ]

        self.assertEqual(inline_speaker_branches, [])
        self.assertEqual(len(speaker_handler_bindings), 1)
        self.assertEqual(len(controller_router_bindings), 1)


if __name__ == "__main__":
    unittest.main()
