"""Backchannels must leave an overlapping reply and its audio stream intact."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.tools.voice.soulx_turn_taking import SoulXAudioAccumulator, SoulXTurnState
from src.voice.answer_completion import AnswerCompletionState
from src.voice.handlers.soulx_audio_handler import SoulXAudioHandler
from src.voice.runtime import VoiceRecognitionService
from src.voice.session import VoiceTurnTakingState


class SoulXBackchannelInterruptionTests(unittest.IsolatedAsyncioTestCase):
    def make_context(self, states, *, speaking=True, processing=False, companion=None):
        context = SimpleNamespace(
            speaking=speaking,
            processing=processing,
            stops=[],
            submissions=[],
            connection=SimpleNamespace(send_json=mock.AsyncMock()),
            turn_state=VoiceTurnTakingState(),
        )

        async def stop(reason="playback"):
            context.stops.append(reason)
            context.speaking = False
            context.processing = False

        async def submit(audio, **kwargs):
            context.submissions.append((audio.copy(), kwargs))

        context.handler = SoulXAudioHandler(
            context.connection,
            client=SimpleNamespace(
                retry_ready=True,
                process=mock.AsyncMock(side_effect=states),
            ),
            accumulator=SoulXAudioAccumulator(),
            turn_state=context.turn_state,
            speaker=SimpleNamespace(enabled=False, enrolled=False, verifier=None),
            answer_completion=AnswerCompletionState(),
            reset_local_vad=lambda: None,
            is_ai_speaking=lambda: context.speaking,
            is_processing=lambda: context.processing,
            stop_playback=stop,
            interrupt_active_processing=stop,
            notify_answer_completion=mock.AsyncMock(),
            arm_answer_completion_window=mock.AsyncMock(),
            submit_speech=submit,
            audio_signal_stats=lambda audio, rate: {"rms": 0.05},
            unavailable_error_type=RuntimeError,
            server_url="ws://soulx.test/turn",
            enabled_full_duplex=True,
            minimum_utterance_rms=0.01,
            barge_in_minimum_chunk_rms=0.012,
            realtime_companion=companion,
            logger=lambda message: None,
        )
        return context

    @staticmethod
    def state(kind, text=""):
        return SoulXTurnState(
            state=kind,
            text=text if kind == "speak" else "",
            asr_buffer=text,
            speech_detected=kind == "nonidle",
            chunk_rms=0.05,
        )

    async def feed(self, context, count=1):
        for _ in range(count):
            self.assertTrue(
                await context.handler.handle_audio(np.full(2560, 0.05, dtype=np.float32))
            )

    def assert_reply_untouched(self, context):
        self.assertEqual(context.stops, [])
        self.assertEqual(context.submissions, [])
        self.assertNotIn(
            "vad_end",
            [call.args[0]["type"] for call in context.connection.send_json.call_args_list],
        )

    async def test_user_backchannels_do_not_stop_playback_or_submit_another_turn(self):
        for text in ("嗯。", "对。", "嗯嗯", "对，对。", "好的！", "是的。", "嗯，对。", "OK."):
            with self.subTest(text=text):
                context = self.make_context(
                    [self.state("nonidle", text), self.state("speak", text)]
                )
                await self.feed(context, 2)
                self.assert_reply_untouched(context)
                self.assertTrue(context.speaking)

    async def test_backchannel_keeps_pending_generation_alive(self):
        context = self.make_context(
            [self.state("nonidle", "对"), self.state("speak", "对。")],
            speaking=False,
            processing=True,
        )
        await self.feed(context, 2)
        self.assert_reply_untouched(context)
        self.assertTrue(context.processing)

    async def test_nonidle_waits_for_text_before_deciding_to_interrupt(self):
        context = self.make_context(
            [
                self.state("nonidle"),
                self.state("nonidle", "嗯"),
                self.state("speak", "嗯。"),
            ]
        )
        await self.feed(context)
        self.assert_reply_untouched(context)
        await self.feed(context, 2)
        self.assert_reply_untouched(context)

    async def test_direct_speak_backchannel_does_not_bypass_the_guard(self):
        context = self.make_context(
            [self.state("idle"), self.state("speak", "对。")]
        )
        await self.feed(context, 2)
        self.assert_reply_untouched(context)

    async def test_overlap_is_remembered_until_final_and_cleared_for_the_next_answer(self):
        context = self.make_context(
            [
                self.state("nonidle", "对"),
                self.state("speak", "对。"),
                self.state("nonidle", "是的"),
                self.state("speak", "是的。"),
            ]
        )
        await self.feed(context)
        context.speaking = False
        await self.feed(context)
        self.assert_reply_untouched(context)
        await self.feed(context, 2)
        self.assertEqual(len(context.submissions), 1)
        self.assertEqual(context.submissions[0][1]["extra_meta"]["soulx_text"], "是的。")

    async def test_short_answers_are_submitted_when_the_assistant_is_listening(self):
        for text in ("对。", "嗯。", "是的。", "好的。"):
            with self.subTest(text=text):
                context = self.make_context(
                    [self.state("nonidle", text), self.state("speak", text)],
                    speaking=False,
                )
                await self.feed(context, 2)
                self.assertEqual(context.stops, [])
                self.assertEqual(len(context.submissions), 1)

    async def test_a_continued_backchannel_prefix_interrupts_as_soon_as_content_arrives(self):
        context = self.make_context(
            [
                self.state("nonidle", "嗯"),
                self.state("nonidle", "嗯，我想换个话题"),
                self.state("speak", "嗯，我想换个话题。"),
            ]
        )
        await self.feed(context)
        self.assert_reply_untouched(context)
        await self.feed(context)
        self.assertEqual(len(context.stops), 1)
        await self.feed(context)
        self.assertEqual(len(context.stops), 1)
        self.assertEqual(len(context.submissions), 1)
        self.assertEqual(context.submissions[0][0].size, 3 * 2560)

    async def test_short_requests_and_corrections_still_interrupt_immediately(self):
        for text in (
            "停", "等一下。", "不对。", "不", "对，但是我不同意。",
            "好像不是这样。", "No.", "对？", "嗯？",
        ):
            with self.subTest(text=text):
                context = self.make_context(
                    [self.state("nonidle", text), self.state("speak", text)]
                )
                await self.feed(context)
                self.assertEqual(len(context.stops), 1)
                await self.feed(context)
                self.assertEqual(len(context.submissions), 1)

    async def test_answer_to_a_listening_prompt_is_not_discarded(self):
        for kind in ("backchannel_playing", "listening_prompt_playing"):
            with self.subTest(kind=kind):
                companion = SimpleNamespace(
                    backchannel_playing=False,
                    listening_prompt_playing=False,
                    observe_soulx_frame=mock.AsyncMock(return_value=None),
                )
                setattr(companion, kind, True)
                context = self.make_context(
                    [self.state("nonidle", "是的"), self.state("speak", "是的。")],
                    companion=companion,
                )
                await self.feed(context, 2)
                self.assertEqual(len(context.submissions), 1)

    async def test_discarded_backchannel_releases_realtime_turn_tasks(self):
        turn = SimpleNamespace(aclose=mock.AsyncMock())
        companion = SimpleNamespace(
            backchannel_playing=False,
            listening_prompt_playing=False,
            observe_soulx_frame=mock.AsyncMock(side_effect=[None, turn]),
        )
        context = self.make_context(
            [self.state("nonidle", "对"), self.state("speak", "对。")],
            companion=companion,
        )
        await self.feed(context, 2)
        self.assert_reply_untouched(context)
        turn.aclose.assert_awaited_once()

    async def test_overlap_is_captured_before_waiting_for_soulx(self):
        context = self.make_context([])
        states = iter([self.state("nonidle", "对"), self.state("speak", "对。")])

        async def process(audio):
            context.speaking = False
            return next(states)

        context.handler.client.process = process
        await self.feed(context, 2)
        self.assert_reply_untouched(context)

    async def test_soulx_failure_clears_overlap_before_a_new_answer(self):
        context = self.make_context(
            [
                self.state("nonidle", "对"),
                RuntimeError("SoulX temporarily unavailable"),
                self.state("nonidle", "是的"),
                self.state("speak", "是的。"),
            ]
        )
        await self.feed(context)
        self.assertFalse(
            await context.handler.handle_audio(np.full(2560, 0.05, dtype=np.float32))
        )
        context.speaking = False
        await self.feed(context, 2)
        self.assertEqual(context.stops, [])
        self.assertEqual(len(context.submissions), 1)

    async def test_session_reset_does_not_discard_a_later_short_answer(self):
        context = self.make_context(
            [
                self.state("nonidle", "对"),
                self.state("nonidle", "是的"),
                self.state("speak", "是的。"),
            ]
        )
        await self.feed(context)
        context.turn_state.reset()
        context.handler.accumulator.reset()
        context.speaking = False
        await self.feed(context, 2)
        self.assertEqual(context.stops, [])
        self.assertEqual(len(context.submissions), 1)


class LocalBackchannelInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_punctuation_and_repeated_acknowledgements_do_not_require_an_llm(self):
        llm = SimpleNamespace(invoke=mock.Mock(return_value="C"))
        service = VoiceRecognitionService(
            SimpleNamespace(agent=SimpleNamespace(llm=llm)), logger=lambda message: None
        )
        for text in ("对。", "嗯，嗯。", "是的。", "好的！", "嗯，对。", "OK."):
            with self.subTest(text=text):
                self.assertEqual(await service.judge_interrupt_intent(text), "backchannel")
        llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
