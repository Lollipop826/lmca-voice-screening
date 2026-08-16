"""Characterization tests for the WebRTC-to-WebSocket adapter."""

import ast
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.voice import media
from src.voice.media import (
    HybridMediaRegistry,
    HybridVoiceTransport,
    WebRTCPeerSession,
    WebRTCVoiceConnection,
)


VOICE_SERVER_PATH = Path(__file__).resolve().parents[2] / "voice_server.py"
VOICE_CHAT_PATH = Path(__file__).resolve().parents[2] / "static" / "voice_chat.html"


class _FakeDataChannel:
    def __init__(self, label="control", ready_state="open"):
        self.label = label
        self.readyState = ready_state
        self.bufferedAmount = 0
        self.handlers = {}
        self.sent = []

    def on(self, event_name):
        def register(callback):
            self.handlers[event_name] = callback
            return callback

        return register

    def emit(self, event_name, value=None):
        callback = self.handlers[event_name]
        if value is None:
            return callback()
        return callback(value)

    def send(self, payload):
        self.sent.append(payload)


class _FakeAssistantTrack:
    def __init__(self):
        self.clear_count = 0
        self.enqueued = []

    def clear(self):
        self.clear_count += 1

    def enqueue_audio(self, audio, sample_rate):
        self.enqueued.append((audio.copy(), sample_rate))


class _FakeWebSocket:
    def __init__(self):
        self.cookies = {"aa_session": "token"}
        self.incoming = asyncio.Queue()
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed = True


class _FakeMediaSession:
    def __init__(self, transport, peer_id):
        self.transport = transport
        self.peer_id = peer_id
        self.closed_reasons = []

    async def close(self, reason=""):
        self.closed_reasons.append(reason)
        self.transport.detach_media(self.peer_id)


class _FakePeerConnection:
    def __init__(self):
        self.handlers = {}
        self.tracks = []
        self.connectionState = "new"
        self.iceGatheringState = "complete"
        self.localDescription = None
        self.remote_description = None
        self.closed_count = 0

    def on(self, event_name):
        def register(callback):
            self.handlers[event_name] = callback
            return callback

        return register

    def addTrack(self, track):
        self.tracks.append(track)

    async def setRemoteDescription(self, description):
        self.remote_description = description

    async def createAnswer(self):
        return SimpleNamespace(sdp="draft-answer", type="answer")

    async def setLocalDescription(self, _answer):
        self.localDescription = SimpleNamespace(
            sdp="final-answer",
            type="answer",
        )

    async def close(self):
        self.closed_count += 1
        self.connectionState = "closed"


class WebRTCVoiceConnectionTests(unittest.TestCase):
    def test_adapter_is_not_defined_in_the_application_entrypoint(self):
        tree = ast.parse(
            VOICE_SERVER_PATH.read_text(encoding="utf-8"),
            filename=str(VOICE_SERVER_PATH),
        )

        adapter_classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {
                "WebRTCVoiceConnection",
                "WebRTCAssistantAudioTrack",
            }
        ]

        self.assertEqual(adapter_classes, [])

    def test_exposes_websocket_lifecycle_contract(self):
        cookies = {"aa_session": "token"}
        conn = WebRTCVoiceConnection(cookies, peer_id="peer-1")
        cookies["aa_session"] = "changed"

        self.assertEqual(conn.cookies, {"aa_session": "token"})
        self.assertEqual(conn.peer_id, "peer-1")
        self.assertEqual(conn.transport_name, "WebRTC")
        self.assertFalse(conn.closed)
        self.assertIsNone(asyncio.run(conn.accept()))

        asyncio.run(conn.close(code=1000))
        self.assertTrue(conn.closed)
        self.assertEqual(
            asyncio.run(conn.receive()),
            {"type": "websocket.disconnect"},
        )

    def test_data_channel_messages_are_adapted_to_asgi_receive_events(self):
        async def scenario():
            conn = WebRTCVoiceConnection({}, peer_id="peer-2")
            control = _FakeDataChannel()
            conn.attach_data_channel(control)

            control.emit("message", "{\"type\":\"ping\"}")
            text_event = await conn.receive()
            control.emit("message", bytearray(b"\x01\x02"))
            bytes_event = await conn.receive()

            self.assertEqual(
                text_event,
                {"type": "websocket.receive", "text": "{\"type\":\"ping\"}"},
            )
            self.assertEqual(
                bytes_event,
                {"type": "websocket.receive", "bytes": b"\x01\x02"},
            )
            self.assertTrue(conn.is_control_open())

        asyncio.run(scenario())

    def test_only_closing_the_control_channel_disconnects_the_adapter(self):
        async def scenario():
            conn = WebRTCVoiceConnection({}, peer_id="peer-3")
            control = _FakeDataChannel(label="control")
            audio = _FakeDataChannel(label="voice-audio")
            conn.attach_data_channel(control)
            conn.attach_data_channel(audio)

            audio.emit("close")
            self.assertFalse(conn.closed)
            control.emit("close")
            self.assertTrue(conn.closed)
            self.assertEqual(await conn.receive(), {"type": "websocket.disconnect"})

        asyncio.run(scenario())

    def test_microphone_audio_is_chunked_into_websocket_binary_frames(self):
        async def scenario():
            conn = WebRTCVoiceConnection({}, peer_id="peer-audio")
            source = np.concatenate(
                [
                    np.full(1024, 2.0, dtype=np.float32),
                    np.full(1024, -2.0, dtype=np.float32),
                    np.full(10, 0.5, dtype=np.float32),
                ]
            )

            conn.feed_audio_float(source, sample_rate=16000)
            events = [await conn.receive() for _ in range(4)]

            self.assertTrue(
                all(event["type"] == "websocket.receive" for event in events)
            )
            self.assertTrue(
                all(len(event["bytes"]) == 512 * 2 for event in events)
            )
            for event in events[:2]:
                np.testing.assert_array_equal(
                    np.frombuffer(event["bytes"], dtype=np.int16),
                    np.full(512, 32767, dtype=np.int16),
                )
            for event in events[2:]:
                np.testing.assert_array_equal(
                    np.frombuffer(event["bytes"], dtype=np.int16),
                    np.full(512, -32767, dtype=np.int16),
                )
            self.assertEqual(conn._incoming_audio_buffer.size, 10)

        asyncio.run(scenario())

    def test_send_json_requires_an_open_control_channel(self):
        async def scenario():
            conn = WebRTCVoiceConnection({}, peer_id="peer-4")
            with self.assertRaisesRegex(RuntimeError, "not open"):
                await conn.send_json({"type": "ping"})

            conn.attach_data_channel(_FakeDataChannel(ready_state="closed"))
            with self.assertRaisesRegex(RuntimeError, "not open"):
                await conn.send_json({"type": "ping"})

            conn.disconnect()
            with self.assertRaisesRegex(RuntimeError, "is closed"):
                await conn.send_json({"type": "ping"})

        asyncio.run(scenario())

    def test_send_json_serializes_without_mutating_the_callers_payload(self):
        async def scenario():
            conn = WebRTCVoiceConnection({}, peer_id="peer-5")
            control = _FakeDataChannel()
            conn.attach_data_channel(control)
            original = {"type": "ai_response", "text": "您好"}

            await conn.send_json(original)

            self.assertEqual(original, {"type": "ai_response", "text": "您好"})
            self.assertEqual(
                json.loads(control.sent[0]),
                {"type": "ai_response", "text": "您好"},
            )

        asyncio.run(scenario())

    def test_stalled_data_channel_buffer_triggers_peer_recovery(self):
        async def scenario():
            clock = [0.0]
            stalls = []
            conn = WebRTCVoiceConnection(
                {},
                peer_id="peer-stalled",
                monotonic_factory=lambda: clock[0],
                data_channel_stall_timeout_s=4.0,
                data_channel_stall_min_bytes=1024,
                logger=lambda _message: None,
            )
            conn.set_data_channel_stall_handler(stalls.append)
            control = _FakeDataChannel()
            control.bufferedAmount = 2048
            conn.attach_data_channel(control)

            await conn.send_json({"type": "soulx_state", "state": "idle"})
            clock[0] = 5.0
            control.bufferedAmount = 4096

            with self.assertRaisesRegex(
                RuntimeError,
                "DataChannel send buffer stalled",
            ):
                await conn.send_json(
                    {"type": "soulx_state", "state": "nonidle"}
                )

            self.assertEqual(len(stalls), 1)
            self.assertIn("4096 bytes for 5.0s", stalls[0])
            self.assertEqual(len(control.sent), 1)

        asyncio.run(scenario())

    def test_frontend_uses_websocket_control_and_media_only_webrtc(self):
        source = VOICE_CHAT_PATH.read_text(encoding="utf-8")

        self.assertIn("function stageTransportRecovery", source)
        self.assertIn("function stopAutoMicrophoneForTransportRecovery", source)
        self.assertIn("type: 'prepare_patient'", source)
        self.assertIn("patient_id: recovery.patientId", source)
        self.assertNotIn("session_id: recovery.sessionId", source)
        self.assertIn("completeTransportRecovery('session_resumed')", source)
        self.assertIn(
            "ensureAutoListeningAfterSessionReady('session_resumed')",
            source,
        )
        self.assertIn("function createWebRTCMediaTransport", source)
        self.assertIn("media_token: mediaToken", source)
        self.assertIn("control_transport !== 'websocket'", source)
        self.assertNotIn("createDataChannel('voice-events'", source)
        self.assertIn("this.bufferSize = 512", source)

    def test_tts_audio_moves_to_rtp_and_only_announces_the_first_chunk(self):
        async def scenario():
            track = _FakeAssistantTrack()
            conn = WebRTCVoiceConnection(
                {}, peer_id="peer-tts", assistant_track=track
            )
            control = _FakeDataChannel()
            conn.attach_data_channel(control)

            await conn.send_json({"type": "tts_start"})
            audio = np.array([0.25, -0.5], dtype=np.float32)
            encoded = base64.b64encode(audio.tobytes()).decode("ascii")
            await conn.send_json(
                {"type": "tts_chunk", "chunk": encoded, "sample_rate": 24000}
            )
            await conn.send_json(
                {"type": "tts_chunk", "chunk": encoded, "sample_rate": 24000}
            )
            await conn.send_json({"type": "tts_end"})

            self.assertEqual(track.clear_count, 1)
            self.assertEqual(len(track.enqueued), 2)
            np.testing.assert_array_equal(track.enqueued[0][0], audio)
            self.assertEqual(track.enqueued[0][1], 24000)
            self.assertEqual(
                [json.loads(payload) for payload in control.sent],
                [
                    {"type": "tts_start", "media_transport": "webrtc_rtp"},
                    {
                        "type": "tts_chunk",
                        "sample_rate": 24000,
                        "media_transport": "webrtc_rtp",
                    },
                    {"type": "tts_end", "media_transport": "webrtc_rtp"},
                ],
            )

        asyncio.run(scenario())

    def test_disconnect_is_idempotent_and_clears_assistant_audio(self):
        track = _FakeAssistantTrack()
        conn = WebRTCVoiceConnection({}, peer_id="peer-6", assistant_track=track)

        conn.disconnect()
        conn.disconnect()
        conn.feed_message("ignored")

        self.assertTrue(conn.closed)
        self.assertEqual(track.clear_count, 1)
        self.assertEqual(conn._queue.qsize(), 1)


class HybridVoiceTransportTests(unittest.TestCase):
    def test_registry_and_audio_share_the_websocket_owned_session(self):
        async def scenario():
            registry = HybridMediaRegistry()
            websocket = _FakeWebSocket()
            transport = HybridVoiceTransport(
                websocket,
                media_token="a" * 32,
                registry=registry,
                logger=lambda _message: None,
            )

            await transport.accept()
            self.assertTrue(websocket.accepted)
            self.assertIs(registry.get("a" * 32), transport)

            track = _FakeAssistantTrack()
            session = _FakeMediaSession(transport, "media-peer")
            transport.attach_media("media-peer", track, session)
            transport.set_media_ready("media-peer", True)
            transport.feed_audio_float(
                np.full(512, 0.25, dtype=np.float32),
                sample_rate=16000,
            )

            event = await transport.receive()
            samples = np.frombuffer(event["bytes"], dtype=np.int16)
            self.assertEqual(samples.size, 512)
            self.assertTrue(np.all(samples == 8191))

            await transport.close()
            self.assertIsNone(registry.get("a" * 32))
            self.assertTrue(websocket.closed)
            self.assertEqual(
                session.closed_reasons,
                ["control websocket closed"],
            )

        asyncio.run(scenario())

    def test_json_always_uses_websocket_while_tts_audio_moves_to_rtp(self):
        async def scenario():
            registry = HybridMediaRegistry()
            websocket = _FakeWebSocket()
            transport = HybridVoiceTransport(
                websocket,
                media_token="b" * 32,
                registry=registry,
                logger=lambda _message: None,
            )
            await transport.accept()

            await transport.send_json(
                {"type": "soulx_state", "state": "nonidle"}
            )
            track = _FakeAssistantTrack()
            session = _FakeMediaSession(transport, "media-peer")
            transport.attach_media("media-peer", track, session)
            transport.set_media_ready("media-peer", True)
            audio = np.array([0.25, -0.5], dtype=np.float32)
            encoded = base64.b64encode(audio.tobytes()).decode("ascii")
            await transport.send_json({"type": "tts_start"})
            await transport.send_json(
                {"type": "tts_chunk", "chunk": encoded, "sample_rate": 24000}
            )
            await transport.send_json(
                {"type": "tts_chunk", "chunk": encoded, "sample_rate": 24000}
            )
            await transport.send_json({"type": "tts_end"})

            self.assertEqual(
                websocket.sent,
                [
                    {"type": "soulx_state", "state": "nonidle"},
                    {"type": "tts_start", "media_transport": "webrtc_rtp"},
                    {
                        "type": "tts_chunk",
                        "sample_rate": 24000,
                        "media_transport": "webrtc_rtp",
                    },
                    {"type": "tts_end", "media_transport": "webrtc_rtp"},
                ],
            )
            self.assertEqual(len(track.enqueued), 2)
            np.testing.assert_array_equal(track.enqueued[0][0], audio)
            await transport.close()

        asyncio.run(scenario())


class WebRTCPeerSessionTests(unittest.TestCase):
    def test_offer_route_delegates_to_peer_session_without_nested_state(self):
        tree = ast.parse(
            VOICE_SERVER_PATH.read_text(encoding="utf-8"),
            filename=str(VOICE_SERVER_PATH),
        )
        route = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "webrtc_offer"
        )
        nested_names = {
            node.name
            for node in route.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        peer_session_calls = [
            node
            for node in ast.walk(route)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "WebRTCPeerSession"
        ]

        self.assertEqual(nested_names, set())
        self.assertEqual(len(peer_session_calls), 1)

    def test_create_registers_peer_and_builds_transport_adapter(self):
        peer_connection = _FakePeerConnection()
        assistant_track = _FakeAssistantTrack()
        registry = set()

        with (
            patch.object(
                media,
                "build_rtc_configuration",
                return_value="rtc-config",
            ),
            patch.object(
                media,
                "RTCPeerConnection",
                return_value=peer_connection,
            ) as peer_factory,
            patch.object(
                media,
                "WebRTCAssistantAudioTrack",
                return_value=assistant_track,
            ),
        ):
            peer_session = WebRTCPeerSession.create(
                cookies={"aa_session": "token"},
                peer_id="peer-create",
                ice_servers=[{"urls": "stun:example.test"}],
                session_runner=lambda _connection: asyncio.sleep(0),
                peer_registry=registry,
                logger=lambda _message: None,
            )

        peer_factory.assert_called_once_with(configuration="rtc-config")
        self.assertEqual(peer_connection.tracks, [assistant_track])
        self.assertIn(peer_connection, registry)
        self.assertEqual(
            set(peer_connection.handlers),
            {
                "track",
                "datachannel",
                "connectionstatechange",
                "icegatheringstatechange",
            },
        )
        self.assertEqual(
            peer_session.connection.cookies,
            {"aa_session": "token"},
        )
        self.assertEqual(
            peer_session.connection.peer_id,
            "webrtc_peer-create",
        )

    def test_media_only_peer_has_no_datachannel_or_second_voice_session(self):
        async def scenario():
            peer_connection = _FakePeerConnection()
            assistant_track = _FakeAssistantTrack()
            peer_registry = set()
            ready_states = []
            detached = []

            class MediaTransport:
                def attach_media(self, peer_id, track, media_session):
                    self.peer_id = peer_id
                    self.track = track
                    self.media_session = media_session

                def set_media_ready(self, peer_id, ready):
                    ready_states.append((peer_id, ready))

                def detach_media(self, peer_id):
                    detached.append(peer_id)

                def feed_audio_float(self, _audio, _sample_rate):
                    pass

            transport = MediaTransport()
            with (
                patch.object(
                    media,
                    "build_rtc_configuration",
                    return_value="rtc-config",
                ),
                patch.object(
                    media,
                    "RTCPeerConnection",
                    return_value=peer_connection,
                ),
                patch.object(
                    media,
                    "WebRTCAssistantAudioTrack",
                    return_value=assistant_track,
                ),
            ):
                peer_session = WebRTCPeerSession.create_media_only(
                    peer_id="media-only",
                    ice_servers=[],
                    media_transport=transport,
                    peer_registry=peer_registry,
                    logger=lambda _message: None,
                )

            self.assertTrue(peer_session.media_only)
            self.assertIsNone(peer_session.session_task)
            self.assertNotIn("datachannel", peer_connection.handlers)
            self.assertEqual(peer_connection.tracks, [assistant_track])

            peer_connection.connectionState = "connected"
            await peer_connection.handlers["connectionstatechange"]()
            self.assertEqual(ready_states, [("media-only", True)])

            await peer_session.close("test complete")
            self.assertEqual(detached, ["media-only"])
            self.assertNotIn(peer_connection, peer_registry)

        asyncio.run(scenario())

    def test_negotiate_returns_local_description(self):
        async def scenario():
            peer_connection = _FakePeerConnection()
            registry = set()
            peer_session = WebRTCPeerSession(
                peer_connection,
                WebRTCVoiceConnection({}, peer_id="webrtc_peer-negotiate"),
                peer_id="peer-negotiate",
                session_runner=lambda _connection: asyncio.sleep(0),
                peer_registry=registry,
                session_description_factory=lambda **fields: fields,
                logger=lambda _message: None,
            )
            peer_session.register()

            result = await peer_session.negotiate(
                "offer-sdp",
                "offer",
            )

            self.assertEqual(
                peer_connection.remote_description,
                {"sdp": "offer-sdp", "type": "offer"},
            )
            self.assertEqual(
                result,
                {"sdp": "final-answer", "type": "answer"},
            )

        asyncio.run(scenario())

    def test_open_control_channel_starts_one_session_and_closes_peer(self):
        async def scenario():
            peer_connection = _FakePeerConnection()
            registry = set()
            started = asyncio.Event()
            release = asyncio.Event()

            async def run_session(_connection):
                started.set()
                await release.wait()

            connection = WebRTCVoiceConnection(
                {},
                peer_id="webrtc_peer-control",
            )
            peer_session = WebRTCPeerSession(
                peer_connection,
                connection,
                peer_id="peer-control",
                session_runner=run_session,
                peer_registry=registry,
                logger=lambda _message: None,
            )
            peer_session.register()
            channel = _FakeDataChannel(
                label="control",
                ready_state="open",
            )

            peer_connection.handlers["datachannel"](channel)
            await started.wait()
            first_task = peer_session.session_task
            peer_session._start_session_if_ready()

            self.assertIs(peer_session.session_task, first_task)
            release.set()
            await first_task
            await asyncio.sleep(0)
            await peer_session.close_task

            self.assertTrue(connection.closed)
            self.assertEqual(peer_connection.closed_count, 1)
            self.assertNotIn(peer_connection, registry)

            await peer_session.close("duplicate")
            self.assertEqual(peer_connection.closed_count, 1)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
