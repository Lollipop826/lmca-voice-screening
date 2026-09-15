"""Characterization tests for transport-neutral voice connection I/O."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.voice.connection import (
    VoiceConnectionController,
    VoiceConnectionIO,
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


class _FakeTransport:
    def __init__(self, events=None, *, transport_name="WebSocket"):
        self.events = list(events or [])
        self.transport_name = transport_name
        self.cookies = {"aa_session": "token"}
        self.closed = False
        self.accepted = False
        self.close_code = None
        self.sent = []
        self.send_error = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed = True
        self.close_code = code

    async def receive(self):
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    async def send_json(self, payload):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(payload)


class VoiceConnectionIOTests(unittest.TestCase):
    def test_exposes_transport_lifecycle_and_identity(self):
        async def scenario():
            transport = _FakeTransport(transport_name="WebRTC")
            connection = VoiceConnectionIO(transport)

            self.assertEqual(connection.cookies, {"aa_session": "token"})
            self.assertEqual(connection.transport_name, "WebRTC")
            self.assertEqual(connection.client_id, str(id(transport)))
            self.assertFalse(connection.closed)

            await connection.accept()
            await connection.close(code=1008)

            self.assertTrue(transport.accepted)
            self.assertTrue(connection.closed)
            self.assertEqual(transport.close_code, 1008)

        asyncio.run(scenario())

    def test_binary_audio_is_normalized_and_first_frame_is_announced_once(self):
        async def scenario():
            first_samples = np.array([32767, -32768, 0], dtype=np.int16)
            second_samples = np.array([16384, -16384], dtype=np.int16)
            transport = _FakeTransport(
                [
                    {"type": "websocket.receive", "bytes": first_samples.tobytes()},
                    {"type": "websocket.receive", "bytes": second_samples.tobytes()},
                ]
            )
            logs = []
            connection = VoiceConnectionIO(transport, logger=logs.append)

            first = await connection.receive_message()
            second = await connection.receive_message()

            self.assertEqual(first["type"], "audio")
            self.assertEqual(second["type"], "audio")
            np.testing.assert_allclose(
                first["_audio_float"],
                first_samples.astype(np.float32) / 32768.0,
            )
            np.testing.assert_allclose(
                second["_audio_float"],
                second_samples.astype(np.float32) / 32768.0,
            )
            self.assertEqual(connection.audio_frame_count, 2)
            self.assertEqual(connection.audio_sample_count, 5)
            self.assertEqual(
                transport.sent,
                [
                    {
                        "type": "audio_input_started",
                        "samples": 3,
                        "sample_rate": 16000,
                    }
                ],
            )
            self.assertEqual(
                sum("收到首帧音频" in line for line in logs),
                1,
            )

        asyncio.run(scenario())

    def test_text_json_and_legacy_or_float_audio_are_normalized(self):
        async def scenario():
            transport = _FakeTransport(
                [
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "ping"}),
                    },
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "audio", "data": [0, 32767]}),
                    },
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "audio", "data": [0.0, 0.5]}),
                    },
                ]
            )
            connection = VoiceConnectionIO(transport)

            self.assertEqual(await connection.receive_message(), {"type": "ping"})
            legacy_audio = await connection.receive_message()

            self.assertEqual(legacy_audio["type"], "audio")
            self.assertEqual(legacy_audio["data"], [0, 32767])
            np.testing.assert_allclose(
                legacy_audio["_audio_float"],
                np.array([0.0, 32767 / 32768], dtype=np.float32),
            )
            float_audio = await connection.receive_message()
            np.testing.assert_allclose(
                float_audio["_audio_float"],
                np.array([0.0, 0.5], dtype=np.float32),
            )

        asyncio.run(scenario())

    def test_disconnect_and_post_disconnect_runtime_error_end_the_loop(self):
        async def scenario():
            disconnect = VoiceConnectionIO(
                _FakeTransport([{"type": "websocket.disconnect"}]),
                logger=lambda _message: None,
            )
            already_closed = VoiceConnectionIO(
                _FakeTransport([RuntimeError("receive after disconnect")]),
                logger=lambda _message: None,
            )

            self.assertIsNone(await disconnect.receive_message())
            self.assertIsNone(await already_closed.receive_message())

        asyncio.run(scenario())

    def test_safe_send_reports_failure_instead_of_breaking_session(self):
        async def scenario():
            transport = _FakeTransport(transport_name="WebRTC")
            transport.send_error = RuntimeError("channel closed")
            logs = []
            connection = VoiceConnectionIO(transport, logger=logs.append)

            sent = await connection.send_json({"type": "ai_response", "text": "您好"})

            self.assertFalse(sent)
            self.assertTrue(any("WebRTC" in line for line in logs))
            self.assertTrue(any("channel closed" in line for line in logs))

        asyncio.run(scenario())

    def test_output_events_are_correlated_to_the_active_turn(self):
        async def scenario():
            transport = _FakeTransport()
            session = SimpleNamespace(
                session_id="call-output",
                processing=SimpleNamespace(
                    is_active=True,
                    generation=4,
                    active_turn_id="turn_0004",
                    active_playback_id="playback-turn_0004-4",
                ),
            )
            connection = VoiceConnectionIO(transport, session=session)

            for event_type in (
                "ai_response",
                "ai_response_chunk",
                "tts_start",
                "tts_chunk",
                "tts_end",
                "tts_stop",
                "stop_tts",
                "interrupt",
            ):
                await connection.send_json({"type": event_type})

            for payload in transport.sent:
                self.assertEqual(payload["session_id"], "call-output")
                self.assertEqual(payload["turn_id"], "turn_0004")
                self.assertEqual(payload["generation"], 4)
                self.assertEqual(
                    payload["playback_id"],
                    "playback-turn_0004-4",
                )

        asyncio.run(scenario())

    def test_standalone_tts_stream_reuses_one_generated_identity(self):
        async def scenario():
            transport = _FakeTransport()
            session = SimpleNamespace(
                session_id="call-greeting",
                processing=SimpleNamespace(
                    is_active=False,
                    generation=0,
                    active_turn_id="",
                    active_playback_id="",
                ),
            )
            connection = VoiceConnectionIO(transport, session=session)

            for event_type in (
                "ai_response",
                "tts_start",
                "tts_chunk",
                "tts_end",
            ):
                await connection.send_json({"type": event_type})

            identities = [
                (
                    payload["session_id"],
                    payload["turn_id"],
                    payload["generation"],
                    payload["playback_id"],
                )
                for payload in transport.sent
            ]
            self.assertEqual(len(set(identities)), 1)
            self.assertTrue(identities[0][1])
            self.assertTrue(identities[0][3])

        asyncio.run(scenario())

    def test_rejects_late_generation_and_old_playback_events(self):
        async def scenario():
            transport = _FakeTransport()
            session = SimpleNamespace(
                session_id="call-output",
                processing=SimpleNamespace(
                    is_active=True,
                    generation=2,
                    active_turn_id="turn_0002",
                    active_playback_id="playback-turn_0002-2",
                ),
            )
            logs = []
            connection = VoiceConnectionIO(
                transport,
                session=session,
                logger=logs.append,
            )

            self.assertTrue(
                await connection.send_json(
                    {
                        "type": "tts_start",
                        "turn_id": "turn_0002",
                        "generation": 2,
                        "playback_id": "playback-turn_0002-2",
                    }
                )
            )
            self.assertFalse(
                await connection.send_json(
                    {
                        "type": "tts_chunk",
                        "turn_id": "turn_0001",
                        "generation": 1,
                        "playback_id": "playback-turn_0001-1",
                        "chunk": "late",
                    }
                )
            )
            self.assertFalse(
                await connection.send_json(
                    {
                        "type": "tts_chunk",
                        "turn_id": "turn_0002",
                        "generation": 2,
                        "playback_id": "playback-old",
                        "chunk": "old",
                    }
                )
            )
            self.assertTrue(
                await connection.send_json(
                    {
                        "type": "tts_start",
                        "turn_id": "turn_0002",
                        "generation": 2,
                        "playback_id": "playback-new",
                    }
                )
            )
            self.assertFalse(
                await connection.send_json(
                    {
                        "type": "tts_end",
                        "turn_id": "turn_0002",
                        "generation": 2,
                        "playback_id": "playback-old",
                    }
                )
            )
            self.assertEqual(
                [payload["type"] for payload in transport.sent],
                ["tts_start", "tts_start"],
            )
            self.assertEqual(
                connection.current_output_identity()["playback_id"],
                "playback-new",
            )
            self.assertTrue(any("stale_generation" in line for line in logs))
            self.assertTrue(any("stale_playback" in line for line in logs))

        asyncio.run(scenario())

    def test_websocket_endpoint_uses_connection_object_for_io(self):
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

        constructed = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VoiceConnectionIO"
            for node in ast.walk(application)
        )
        connection_receives = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "connection"
            and node.func.attr == "receive_message"
            for node in ast.walk(application)
        )
        controller_construction = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VoiceConnectionController"
            for node in ast.walk(application)
        )
        controller_runs = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "controller"
            and node.func.attr == "run"
            for node in ast.walk(application)
        )
        raw_transport_io = [
            node
            for node in ast.walk(endpoint)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "websocket"
            and node.func.attr in {"receive", "send_json"}
        ]
        legacy_helpers = [
            node
            for node in server_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "send_json_safe"
        ]
        delegates_to_application = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "VOICE_APPLICATION"
            and node.func.attr == "handle"
            for node in ast.walk(endpoint)
        )

        self.assertTrue(constructed)
        self.assertFalse(connection_receives)
        self.assertTrue(controller_construction)
        self.assertTrue(controller_runs)
        self.assertTrue(delegates_to_application)
        self.assertEqual(raw_transport_io, [])
        self.assertEqual(legacy_helpers, [])


class VoiceConnectionControllerTests(unittest.TestCase):
    def test_run_dispatches_until_connection_disconnects(self):
        async def scenario():
            class Connection:
                def __init__(self):
                    self.messages = [
                        {"type": "ping"},
                        {"type": "unknown"},
                        None,
                    ]

                async def receive_message(self):
                    return self.messages.pop(0)

            class Router:
                def __init__(self):
                    self.messages = []

                async def dispatch(self, message):
                    self.messages.append(message)
                    return SimpleNamespace(stop_connection=False)

            connection = Connection()
            router = Router()
            controller = VoiceConnectionController(
                connection,
                router,
            )

            await controller.run()

            self.assertEqual(
                router.messages,
                [{"type": "ping"}, {"type": "unknown"}],
            )

        asyncio.run(scenario())

    def test_stop_result_ends_loop_without_receiving_another_message(self):
        async def scenario():
            class Connection:
                def __init__(self):
                    self.receive_count = 0

                async def receive_message(self):
                    self.receive_count += 1
                    return {"type": "end_session"}

            class Router:
                async def dispatch(self, _message):
                    return SimpleNamespace(stop_connection=True)

            connection = Connection()
            controller = VoiceConnectionController(
                connection,
                Router(),
            )

            await controller.run()

            self.assertEqual(connection.receive_count, 1)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
