import ast
from pathlib import Path
from unittest import mock
import unittest

from src.voice import (
    VoiceRuntimeState,
    VoiceSession,
    VoiceTurnTakingState,
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


class VoiceSessionTests(unittest.TestCase):
    def test_default_mode_uses_release_flag(self):
        with mock.patch.dict(
            "os.environ",
            {"DEFAULT_SESSION_MODE": "cognitive_screening"},
        ):
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor-a",
            )
        self.assertEqual(session.mode, "cognitive_screening")

        with mock.patch.dict(
            "os.environ",
            {"DEFAULT_SESSION_MODE": "invalid"},
        ):
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor-a",
            )
        self.assertEqual(session.mode, "wellbeing")

    def test_connections_receive_isolated_mutable_state(self):
        first = VoiceSession(connection=object(), agent=object(), owner_username="doctor-a")
        second = VoiceSession(connection=object(), agent=object(), owner_username="doctor-b")

        first.chat_history.append({"role": "user", "content": "你好"})
        first.patient_profile["name"] = "张阿姨"
        first.manual_audio_blob_uploads["upload-1"] = {"chunks": []}
        first.enroll_sample_uploads["enroll-1"] = {"chunks": []}
        first.processing.is_active = True
        first.processing.pending_audio = object()
        first.processing.generation = 3
        first.runtime.pending_vision_task = "task-1"
        first.runtime.interrupt_audio_buffer.append(object())
        first.runtime.ai_streaming_tts = True
        first.turn_taking.soulx_healthy = True
        first.turn_taking.interrupted_asr_prefix = "上一段"
        first.lifecycle.current_patient_id = "patient-1"
        first.lifecycle.started = True
        first.lifecycle.greeting_sent = True

        self.assertEqual(second.chat_history, [])
        self.assertEqual(second.patient_profile, {})
        self.assertEqual(second.manual_audio_blob_uploads, {})
        self.assertEqual(second.enroll_sample_uploads, {})
        self.assertFalse(second.processing.is_active)
        self.assertIsNone(second.processing.pending_audio)
        self.assertEqual(second.processing.generation, 0)
        self.assertIsNone(second.runtime.pending_vision_task)
        self.assertEqual(second.runtime.interrupt_audio_buffer, [])
        self.assertFalse(second.runtime.ai_streaming_tts)
        self.assertFalse(second.turn_taking.soulx_healthy)
        self.assertEqual(second.turn_taking.interrupted_asr_prefix, "")
        self.assertIsNone(second.lifecycle.current_patient_id)
        self.assertFalse(second.lifecycle.started)
        self.assertFalse(second.lifecycle.greeting_sent)

    def test_bind_session_updates_persistence_paths_and_resets_deduplication(self):
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor-a",
        )
        session.chat_history.append({"role": "assistant", "content": "旧消息"})
        session.last_message_key = ("assistant", "旧消息")

        session.bind_session("call_20260715_abcd1234")

        self.assertEqual(session.session_id, "call_20260715_abcd1234")
        self.assertEqual(
            session.history_file,
            "data/voice_calls/call_20260715_abcd1234/messages.json",
        )
        self.assertEqual(session.last_message_key, (None, None))
        self.assertEqual(
            session.chat_history,
            [{"role": "assistant", "content": "旧消息"}],
        )

    def test_runtime_state_owns_interrupt_judgement_and_speaking_status(self):
        runtime = VoiceRuntimeState()
        runtime.ai_speaking_until = 12.0

        self.assertTrue(runtime.is_ai_speaking(11.0))
        self.assertFalse(runtime.is_ai_speaking(13.0))

        runtime.remember_interrupt_judgement(
            audio_samples=3200,
            text="等一下",
            intent="interrupt",
        )
        self.assertEqual(
            runtime.reuse_interrupt_judgement(audio_samples=3200),
            ("等一下", "interrupt"),
        )
        self.assertEqual(
            runtime.reuse_interrupt_judgement(audio_samples=1600),
            (None, None),
        )

        runtime.interrupt_audio_buffer.append(object())
        runtime.interrupt_speech_run = 3
        runtime.waiting_for_complete = True
        runtime.reset_interrupt_capture(reset_waiting=True)

        self.assertEqual(runtime.interrupt_audio_buffer, [])
        self.assertEqual(runtime.interrupt_speech_run, 0)
        self.assertFalse(runtime.waiting_for_complete)
        self.assertEqual(runtime.last_interrupt_text, "")

    def test_turn_ids_are_monotonic_within_a_connection(self):
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor-a",
        )

        self.assertEqual(session.next_turn_id(), "turn_0001")
        self.assertEqual(session.next_turn_id(), "turn_0002")
        self.assertEqual(session.turn_sequence, 2)

    def test_turn_taking_state_tracks_recovery_and_interrupted_asr(self):
        state = VoiceTurnTakingState()

        self.assertTrue(state.mark_soulx_unavailable("connection refused"))
        self.assertFalse(state.mark_soulx_unavailable("connection refused"))
        self.assertEqual(state.soulx_last_error, "connection refused")
        self.assertTrue(state.mark_soulx_connected())
        self.assertFalse(state.mark_soulx_connected())

        previous, changed = state.observe_soulx_state("nonidle")
        self.assertEqual(previous, "")
        self.assertTrue(changed)
        previous, changed = state.observe_soulx_state("nonidle")
        self.assertEqual(previous, "nonidle")
        self.assertFalse(changed)

        state.remember_interrupted_asr(
            "前半句",
            current_time=10.0,
        )
        state.remember_interrupted_asr(
            "后半句",
            current_time=12.0,
            append=True,
        )
        merged, prefix = state.merge_interrupted_asr(
            "补充内容",
            current_time=20.0,
        )
        self.assertEqual(prefix, "前半句，后半句")
        self.assertEqual(merged, "前半句，后半句，补充内容")
        self.assertEqual(state.interrupted_asr_prefix, "")
        self.assertEqual(state.interrupted_asr_time, 0.0)

    def test_expired_interrupted_asr_is_discarded(self):
        state = VoiceTurnTakingState()
        state.remember_interrupted_asr("旧内容", current_time=10.0)

        merged, prefix = state.merge_interrupted_asr(
            "新内容",
            current_time=80.1,
            ttl_seconds=60.0,
        )

        self.assertEqual(merged, "新内容")
        self.assertEqual(prefix, "")

    def test_create_fresh_session_retries_and_binds_the_successful_id(self):
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor-a",
        )
        candidates = iter(["duplicate-id", "fresh-id"])
        created = []

        def create_session(session_id: str) -> None:
            created.append(session_id)
            if session_id == "duplicate-id":
                raise ValueError("duplicate")

        result = session.create_fresh_session(
            create_session,
            lambda: next(candidates),
        )

        self.assertEqual(result, "fresh-id")
        self.assertEqual(created, ["duplicate-id", "fresh-id"])
        self.assertEqual(session.session_id, "fresh-id")
        self.assertEqual(
            session.history_file,
            "data/voice_calls/fresh-id/messages.json",
        )

    def test_reset_conversation_clears_connection_owned_conversation_state(self):
        session = VoiceSession(
            connection=object(),
            agent=object(),
            owner_username="doctor-a",
        )
        session.bind_session("call-1")
        session.chat_history.append({"role": "user", "content": "你好"})
        session.patient_profile["name"] = "张阿姨"
        session.lifecycle.current_patient_id = "patient-1"
        session.lifecycle.started = True
        session.lifecycle.greeting_sent = True
        session.last_message_key = ("user", "你好")

        session.reset_conversation()

        self.assertEqual(session.session_id, "call-1")
        self.assertEqual(session.chat_history, [])
        self.assertEqual(session.patient_profile, {})
        self.assertIsNone(session.lifecycle.current_patient_id)
        self.assertFalse(session.lifecycle.started)
        self.assertFalse(session.lifecycle.greeting_sent)
        self.assertEqual(session.last_message_key, (None, None))


class VoiceSessionEndpointWiringTests(unittest.TestCase):
    def test_websocket_endpoint_uses_a_connection_owned_agent(self):
        server_tree = ast.parse(
            VOICE_SERVER_PATH.read_text(encoding="utf-8"),
            filename=str(VOICE_SERVER_PATH),
        )
        endpoint = next(
            node
            for node in server_tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "websocket_endpoint"
        )
        application = _voice_application_class()
        loaded_names = {
            node.id
            for node in ast.walk(application)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

        self.assertIn("VoiceSession", loaded_names)
        self.assertIn("session_agent", loaded_names)
        self.assertIn("session", loaded_names)
        self.assertNotIn("AGENT", loaded_names)
        create_agent_calls = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr == "models"
            and node.attr == "create_agent"
        ]
        self.assertEqual(len(create_agent_calls), 1)
        delegates_to_application = [
            node
            for node in ast.walk(endpoint)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "VOICE_APPLICATION"
            and node.func.attr == "handle"
        ]
        self.assertEqual(len(delegates_to_application), 1)

        all_names = {
            node.id
            for node in ast.walk(application)
            if isinstance(node, ast.Name)
        }
        legacy_processing_locals = {
            "is_processing",
            "processing_task",
            "processing_revision_enabled",
            "pending_processing_audio",
            "pending_processing_source",
            "active_processing_audio",
            "active_processing_source",
            "processing_generation",
        }
        self.assertTrue(legacy_processing_locals.isdisjoint(all_names))

        legacy_lifecycle_locals = {
            "session_id",
            "history_file",
            "current_patient_id",
            "session_started",
            "has_sent_greeting",
        }
        assigned_names = {
            node.id
            for node in ast.walk(application)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.assertTrue(legacy_lifecycle_locals.isdisjoint(assigned_names))

        runtime_fields = set(VoiceRuntimeState.__dataclass_fields__)
        turn_taking_fields = set(
            VoiceTurnTakingState.__dataclass_fields__
        )
        nonlocal_names = {
            name
            for node in ast.walk(application)
            if isinstance(node, ast.Nonlocal)
            for name in node.names
        }
        self.assertTrue(runtime_fields.isdisjoint(nonlocal_names))
        self.assertTrue(turn_taking_fields.isdisjoint(nonlocal_names))
        self.assertEqual(nonlocal_names, set())

        state_alias_bindings = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "session"
            and any(
                isinstance(target, ast.Name)
                and target.id in {
                    "runtime",
                    "processing",
                    "lifecycle",
                    "turn_taking",
                }
                for target in node.targets
            )
        ]
        cleanup_constructions = [
            node
            for node in ast.walk(application)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VoiceConnectionCleanup"
        ]
        self.assertEqual(state_alias_bindings, [])
        self.assertEqual(len(cleanup_constructions), 1)


if __name__ == "__main__":
    unittest.main()
