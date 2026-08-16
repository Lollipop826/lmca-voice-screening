import ast
import asyncio
from pathlib import Path
import unittest

import numpy as np

from src.voice import (
    AnswerCompletionController,
    AnswerCompletionState,
    VoiceSession,
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


class AnswerCompletionStateTests(unittest.TestCase):
    def test_instances_own_independent_audio_buffers(self):
        first = AnswerCompletionState(enabled=True)
        second = AnswerCompletionState()

        first.append_segment(np.ones(1600, dtype=np.float64))
        first.last_text = "今天晴天"
        first.last_label = "likely_complete"

        self.assertEqual(len(first.segments), 1)
        self.assertEqual(first.duration_s, 0.1)
        self.assertEqual(second.segments, [])
        self.assertEqual(second.duration_s, 0.0)
        self.assertEqual(second.last_text, "")
        self.assertEqual(second.last_label, "")

    def test_append_audio_and_rollback_preserve_accumulated_duration(self):
        state = AnswerCompletionState()

        first_duration = state.append_segment(np.ones(1600))
        second_duration = state.append_segment(np.ones(3200))
        audio = state.audio()
        remaining_duration = state.rollback_last_segment()

        self.assertEqual(first_duration, 0.1)
        self.assertEqual(second_duration, 0.3)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (4800,))
        self.assertEqual(remaining_duration, 0.1)
        self.assertEqual(state.audio().shape, (1600,))

    def test_reset_buffer_clears_semantic_state_and_advances_token(self):
        state = AnswerCompletionState(
            enabled=True,
            last_text="回答",
            last_label="likely_complete",
            observation_token=4,
        )
        state.append_segment(np.ones(800))

        state.reset_buffer()

        self.assertTrue(state.enabled)
        self.assertEqual(state.segments, [])
        self.assertEqual(state.last_text, "")
        self.assertEqual(state.last_label, "")
        self.assertEqual(state.observation_token, 5)
        self.assertIsNone(state.window_task)

    def test_normalize_text_removes_punctuation_and_spacing(self):
        self.assertEqual(
            AnswerCompletionState.normalize_text(" 今天， 是晴天！ "),
            "今天是晴天",
        )


class AnswerCompletionTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_window_cancels_and_releases_active_task(self):
        state = AnswerCompletionState()
        state.window_task = asyncio.create_task(asyncio.sleep(60))

        self.assertTrue(state.has_active_window())
        state.cancel_window()
        await asyncio.sleep(0)

        self.assertIsNone(state.window_task)


class AnswerCompletionControllerTests(unittest.TestCase):
    @staticmethod
    def _build_controller(*, judgement_label="likely_complete"):
        class Connection:
            def __init__(self):
                self.sent = []

            async def send_json(self, payload):
                self.sent.append(payload)
                return True

        class Coordinator:
            def __init__(self):
                self.submitted = []

            async def submit_or_queue(self, audio, **kwargs):
                self.submitted.append((np.asarray(audio).copy(), kwargs))
                return True

        connection = Connection()
        session = VoiceSession(
            connection=connection,
            agent=object(),
            owner_username="doctor",
        )
        state = AnswerCompletionState()
        coordinator = Coordinator()

        async def quick_asr(_audio):
            return "今天是星期三"

        async def judge(_question, _text):
            return {
                "label": judgement_label,
                "confidence": 0.92,
            }

        controller = AnswerCompletionController(
            connection,
            session=session,
            state=state,
            processing_coordinator=coordinator,
            quick_asr=quick_asr,
            judge_answer_completion=judge,
            extract_latest_assistant_utterance=lambda _history: "今天星期几？",
            observation_window_s=1.0,
            logger=lambda _message: None,
        )
        return connection, session, state, coordinator, controller

    def test_disabled_manual_check_reports_status_without_submission(self):
        async def scenario():
            connection, _session, _state, coordinator, controller = (
                self._build_controller()
            )

            accepted = await controller.request_check(trigger="manual")

            self.assertFalse(accepted)
            self.assertEqual(
                connection.sent,
                [
                    {
                        "type": "answer_completion_status",
                        "state": "disabled",
                        "message": "慢答判定模式尚未开启，请先点一次🧠按钮开启。",
                        "trigger": "manual",
                    }
                ],
            )
            self.assertEqual(coordinator.submitted, [])

        asyncio.run(scenario())

    def test_accepted_answer_is_submitted_and_buffer_is_reset(self):
        async def scenario():
            connection, _session, state, coordinator, controller = (
                self._build_controller()
            )
            state.enabled = True
            state.append_segment(np.ones(3200, dtype=np.float32))

            accepted = await controller.request_check(trigger="manual")

            self.assertTrue(accepted)
            self.assertEqual(len(coordinator.submitted), 1)
            self.assertEqual(
                coordinator.submitted[0][1],
                {"source": "慢答判定:likely_complete"},
            )
            self.assertEqual(state.segments, [])
            self.assertEqual(
                [payload["type"] for payload in connection.sent],
                [
                    "answer_completion_status",
                    "answer_completion_status",
                    "vad_end",
                ],
            )

        asyncio.run(scenario())

    def test_incomplete_answer_keeps_audio_for_future_listening(self):
        async def scenario():
            connection, _session, state, coordinator, controller = (
                self._build_controller(judgement_label="incomplete")
            )
            state.enabled = True
            state.append_segment(np.ones(1600, dtype=np.float32))

            accepted = await controller.request_check(trigger="manual")

            self.assertFalse(accepted)
            self.assertEqual(len(state.segments), 1)
            self.assertEqual(coordinator.submitted, [])
            self.assertEqual(
                connection.sent[-1]["state"],
                "continue_listening",
            )

        asyncio.run(scenario())


class AnswerCompletionEndpointWiringTests(unittest.TestCase):
    def test_endpoint_uses_state_object_instead_of_six_legacy_locals(self):
        application = _voice_application_class()
        legacy_names = {
            "answer_completion_mode",
            "answer_completion_segments",
            "answer_completion_last_text",
            "answer_completion_last_label",
            "answer_completion_window_task",
            "answer_completion_observation_token",
        }
        assigned_names = {
            node.id
            for node in ast.walk(application)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        nonlocal_names = {
            name
            for node in ast.walk(application)
            if isinstance(node, ast.Nonlocal)
            for name in node.names
        }
        self.assertTrue(legacy_names.isdisjoint(assigned_names))
        self.assertTrue(legacy_names.isdisjoint(nonlocal_names))

        nested_function_names = {
            node.name
            for node in ast.walk(application)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("_reset_answer_completion_buffer", nested_function_names)
        self.assertNotIn("_append_answer_completion_segment", nested_function_names)
        self.assertNotIn("_get_answer_completion_audio", nested_function_names)

        constructed_names = {
            node.func.id
            for node in ast.walk(application)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("AnswerCompletionState", constructed_names)
        self.assertIn("AnswerCompletionController", constructed_names)
        self.assertTrue(
            {
                "_notify_answer_completion_status",
                "_maybe_arm_answer_completion_window",
                "_request_answer_completion_check",
            }.isdisjoint(nested_function_names)
        )


if __name__ == "__main__":
    unittest.main()
