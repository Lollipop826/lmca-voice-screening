import asyncio
import base64
import json
import unittest

import numpy as np
import websockets

from src.tools.voice.soulx_turn_taking import (
    SoulXAudioAccumulator,
    SoulXTurnState,
    SoulXTurnTakingClient,
)


class SoulXTurnTakingTests(unittest.TestCase):
    def test_legacy_response_without_rms_is_unknown_instead_of_silent(self):
        state = SoulXTurnState.from_message(json.dumps({
            "type": "turn_state", "state": {"state": "nonidle"},
        }))
        self.assertIsNone(state.chunk_rms)

    def test_missing_or_invalid_speech_detection_is_unknown_not_silence(self):
        for value in (None, "false", 0):
            with self.subTest(value=value):
                state = SoulXTurnState.from_message(json.dumps({
                    "type": "turn_state",
                    "state": {
                        "state": "idle", "detail_state": "incomplete",
                        "speech_detected": value,
                    },
                }))
                self.assertIsNone(state.speech_detected)
        state = SoulXTurnState.from_message(json.dumps({
            "type": "turn_state", "state": {"state": "idle"},
        }))
        self.assertIsNone(state.speech_detected)
        state = SoulXTurnState.from_message(json.dumps({
            "type": "turn_state",
            "state": {"state": "idle", "speech_detected": False},
        }))
        self.assertIs(state.speech_detected, False)

    def test_turn_state_parses_soulx_protocol(self):
        state = SoulXTurnState.from_message(
            json.dumps(
                {
                    "type": "turn_state",
                    "session_id": "s1",
                    "state": {
                        "state": "speak",
                        "text": "我说完了",
                        "asr_buffer": "说完了",
                        "raw_state": "idle",
                        "detail_state": "incomplete_timeout",
                        "decision_source": "incomplete_timeout",
                        "wait_idle_count": 22,
                        "max_wait_count": 22,
                        "monitoring_wait_silence": True,
                        "speech_detected": True,
                        "chunk_rms": 0.0123,
                    },
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(state.state, "speak")
        self.assertEqual(state.text, "我说完了")
        self.assertEqual(state.asr_buffer, "说完了")
        self.assertEqual(state.raw_state, "idle")
        self.assertEqual(state.detail_state, "incomplete_timeout")
        self.assertEqual(state.decision_source, "incomplete_timeout")
        self.assertEqual(state.wait_idle_count, 22)
        self.assertEqual(state.max_wait_count, 22)
        self.assertTrue(state.monitoring_wait_silence)
        self.assertTrue(state.speech_detected)
        self.assertAlmostEqual(state.chunk_rms, 0.0123)

    def test_audio_accumulator_keeps_preroll_and_completes_on_speak(self):
        acc = SoulXAudioAccumulator(sample_rate=10, pre_roll_seconds=0.2)
        self.assertIsNone(acc.feed(np.array([0.1], dtype=np.float32), "blank"))
        self.assertIsNone(acc.feed(np.array([0.2], dtype=np.float32), "idle"))
        self.assertIsNone(acc.feed(np.array([0.3], dtype=np.float32), "nonidle"))
        self.assertTrue(acc.active)
        self.assertIsNone(acc.feed(np.array([0.4], dtype=np.float32), "blank"))
        result = acc.feed(np.array([0.5], dtype=np.float32), "speak")
        np.testing.assert_allclose(result, [0.2, 0.3, 0.4, 0.5])
        self.assertFalse(acc.active)

    def test_async_client_uses_float32_audio_contract(self):
        async def scenario():
            observed = {}

            async def handler(ws):
                request = json.loads(await ws.recv())
                observed.update(request)
                audio = np.frombuffer(base64.b64decode(request["audio"]), dtype=np.float32)
                observed["decoded_audio"] = audio
                await ws.send(
                    json.dumps(
                        {
                            "type": "turn_state",
                            "session_id": request["session_id"],
                            "state": {"state": "nonidle", "asr_buffer": "你好"},
                        },
                        ensure_ascii=False,
                    )
                )

            server = await websockets.serve(handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            client = SoulXTurnTakingClient(
                f"ws://127.0.0.1:{port}",
                session_id="voice-test",
                timeout=1.0,
            )
            try:
                result = await client.process(np.array([0.25, -0.5], dtype=np.float32))
            finally:
                await client.close()
                server.close()
                await server.wait_closed()

            self.assertEqual(observed["type"], "audio")
            self.assertEqual(observed["session_id"], "voice-test")
            np.testing.assert_allclose(observed["decoded_audio"], [0.25, -0.5])
            self.assertEqual(result.state, "nonidle")
            self.assertEqual(result.asr_buffer, "你好")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
