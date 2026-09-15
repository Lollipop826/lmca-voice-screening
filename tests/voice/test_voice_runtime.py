import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.voice import runtime as voice_runtime
from src.voice.runtime import (
    VADBuffer,
    VoiceModelRuntime,
    VoiceRecognitionService,
    audio_is_effectively_silent,
    audio_signal_stats,
    clean_for_tts,
    decode_recorded_audio_blob,
)


class _FakeVADModel:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.reset_count = 0

    def __call__(self, _chunk, _sample_rate):
        return next(self.probabilities)

    def predict_stateless(self, _chunk, _sample_rate):
        return 0.75

    def reset_states(self):
        self.reset_count += 1


class VoiceRuntimeUtilityTests(unittest.TestCase):
    def test_tts_cleanup_and_audio_stats(self):
        self.assertEqual(
            clean_for_tts("**重点** 和 `代码`"),
            "重点 和 代码",
        )
        stats = audio_signal_stats(
            np.zeros(1600, dtype=np.float32)
        )
        self.assertEqual(stats["duration_s"], 0.1)
        self.assertTrue(audio_is_effectively_silent(stats))

    def test_vad_buffer_owns_injected_model_state(self):
        model = _FakeVADModel([0.9])
        buffer = VADBuffer(
            vad_model=model,
            environ={},
            logger=lambda _message: None,
        )
        chunk = np.ones(512, dtype=np.float32) * 0.1

        self.assertIsNone(buffer.add_chunk(chunk))
        self.assertTrue(buffer.is_speaking)
        self.assertEqual(buffer.has_speech(chunk), 0.9)
        self.assertEqual(buffer.has_speech(chunk), 0.9)

        buffer.reset()
        self.assertEqual(model.reset_count, 1)
        self.assertEqual(buffer.has_speech(chunk), 0.0)

    def test_interrupt_scores_follow_continuous_vad_without_extra_inference(self):
        model = _FakeVADModel([0.2, 0.8, 0.95, 0.1])
        buffer = VADBuffer(vad_model=model, environ={}, logger=lambda _: None)
        for probability in (0.2, 0.8, 0.95, 0.1):
            chunk = np.full(512, 0.1, dtype=np.float32)
            buffer.add_chunk(chunk)
            self.assertEqual(buffer.has_speech(chunk), probability)
            self.assertEqual(buffer.has_speech(chunk), probability)
        self.assertEqual(model.reset_count, 0)

    def test_batched_audio_does_not_count_one_voiced_frame_as_sustained_speech(self):
        model = _FakeVADModel([0.95, 0.05, 0.05, 0.05])
        buffer = VADBuffer(vad_model=model, environ={}, logger=lambda _: None)
        chunk = np.full(2048, 0.1, dtype=np.float32)
        buffer.add_chunk(chunk)
        self.assertEqual(buffer.has_speech(chunk), 0.05)
        self.assertEqual(buffer.has_speech(chunk.copy()), 0.0)

    def test_vad_allows_a_1_2_second_pause_before_finishing(self):
        speech = np.ones(512, dtype=np.float32) * 0.1
        silence = np.zeros(512, dtype=np.float32)
        model = _FakeVADModel([0.9] * 17 + [0.0] * 38)
        buffer = VADBuffer(
            vad_model=model,
            environ={"VAD_END_SILENCE_S": "1.2"},
            logger=lambda _message: None,
        )

        self.assertEqual(buffer.max_silence_chunks, 38)
        for _ in range(17):
            self.assertIsNone(buffer.add_chunk(speech))
        for _ in range(37):
            self.assertIsNone(buffer.add_chunk(silence))
        self.assertIsNotNone(buffer.add_chunk(silence))

    def test_media_recorder_blob_decode_is_mono_and_cleans_temp_files(self):
        created_paths = []

        def create_temp_file(**kwargs):
            temp_file = NamedTemporaryFile(**kwargs)
            created_paths.append(Path(temp_file.name))
            return temp_file

        stereo = np.array(
            [[1.0, 0.0], [0.0, -1.0]],
            dtype=np.float32,
        )
        with (
            patch(
                "src.voice.runtime.tempfile.NamedTemporaryFile",
                side_effect=create_temp_file,
            ),
            patch("src.voice.runtime.subprocess.run") as run_ffmpeg,
            patch(
                "src.voice.runtime.sf.read",
                return_value=(stereo, 16000),
            ),
        ):
            audio, sample_rate = decode_recorded_audio_blob(
                b"browser-audio",
                "audio/mp4",
            )

        np.testing.assert_allclose(
            audio,
            np.array([0.5, -0.5], dtype=np.float32),
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(created_paths[0].suffix, ".mp4")
        self.assertFalse(created_paths[0].exists())
        self.assertFalse(
            Path(str(created_paths[0]) + ".wav").exists()
        )
        run_ffmpeg.assert_called_once()

    def test_vad_loader_accepts_a_model_bundled_with_the_project(self):
        project_model = (
            Path(voice_runtime.__file__).resolve().parents[2]
            / "models"
            / "silero_vad.onnx"
        )
        loaded_model = object()

        with (
            patch(
                "src.voice.runtime.os.path.exists",
                side_effect=lambda path: Path(path) == project_model,
            ),
            patch(
                "src.voice.runtime.SileroVADModel",
                return_value=loaded_model,
            ) as model_type,
        ):
            result = voice_runtime.load_silero_vad(
                logger=lambda _message: None
            )

        self.assertIs(result, loaded_model)
        model_type.assert_called_once_with(str(project_model))


class VoiceModelRuntimeTests(unittest.TestCase):
    def test_runtime_owns_agent_tts_and_readiness(self):
        agents = []
        tts_instances = []

        def agent_factory(**kwargs):
            agent = SimpleNamespace(kwargs=kwargs)
            agents.append(agent)
            return agent

        def tts_factory(**kwargs):
            tts = SimpleNamespace(kwargs=kwargs)
            tts_instances.append(tts)
            return tts

        with TemporaryDirectory() as temp_dir:
            runtime = VoiceModelRuntime(
                base_dir=Path(temp_dir),
                agent_factory=agent_factory,
                agent_type="test",
                tts_factory=tts_factory,
                use_ark_asr=True,
                use_ark_tts=True,
                use_speaker_verifier=False,
                use_local_embedding=False,
                logger=lambda _message: None,
            )
            self.assertFalse(runtime.ready)

            with patch.object(
                runtime,
                "_initialize_location",
                return_value=None,
            ):
                runtime.initialize()

        self.assertTrue(runtime.ready)
        self.assertIs(runtime.agent, agents[0])
        self.assertIs(runtime.tts, tts_instances[0])
        self.assertEqual(runtime.agent.kwargs, {"use_local": False})


class VoiceRecognitionServiceTests(unittest.TestCase):
    @staticmethod
    def _service():
        models = SimpleNamespace(
            use_ark_asr=False,
            asr_model=None,
            agent=SimpleNamespace(
                llm=SimpleNamespace(
                    invoke=lambda _prompt: "C"
                ),
                use_local=False,
            ),
        )
        return VoiceRecognitionService(
            models,
            logger=lambda _message: None,
        )

    def test_interrupt_rules_do_not_need_llm_for_common_phrases(self):
        async def scenario():
            service = self._service()

            self.assertEqual(
                await service.judge_interrupt_intent("嗯"),
                "backchannel",
            )
            self.assertEqual(
                await service.judge_interrupt_intent("等一下"),
                "complete",
            )
            self.assertEqual(
                await service.judge_interrupt_intent("我说错了"),
                "incomplete",
            )

        asyncio.run(scenario())

    def test_text_helpers_are_transport_independent(self):
        service = self._service()

        self.assertEqual(
            service.normalize_interrupt_text(" 今天是三号，对吧？ "),
            "今天是三号，对吧？",
        )
        self.assertEqual(
            service.extract_latest_assistant_utterance(
                [
                    {"role": "assistant", "content": "第一句"},
                    {"role": "user", "content": "回答"},
                    {"role": "assistant", "content": "第二句"},
                ]
            ),
            "第二句",
        )


if __name__ == "__main__":
    unittest.main()
