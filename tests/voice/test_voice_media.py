"""Tests for reusable WebRTC media conversion primitives."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.voice import media


class _FakeAudioFrame:
    def __init__(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = 48000,
        channels: int = 1,
        layout_name: str = "mono",
    ) -> None:
        self._samples = samples
        self.sample_rate = sample_rate
        self.layout = SimpleNamespace(
            name=layout_name,
            channels=[object() for _ in range(channels)],
        )

    def to_ndarray(self):
        return self._samples


class VoiceMediaTests(unittest.TestCase):
    def test_int16_audio_is_normalized_after_channel_layout_handling(self):
        frame = _FakeAudioFrame(
            np.array([[32767, -32768, 0]], dtype=np.int16),
            sample_rate=16000,
        )

        audio, sample_rate = media.audio_frame_to_mono_float(frame)

        self.assertEqual(sample_rate, 16000)
        np.testing.assert_allclose(
            audio,
            np.array([32767 / 32768, -1.0, 0.0], dtype=np.float32),
        )

    def test_packed_stereo_audio_is_downmixed_to_mono(self):
        frame = _FakeAudioFrame(
            np.array([[32767, 32767, -32768, -32768]], dtype=np.int16),
            channels=2,
            layout_name="stereo",
        )

        audio, _sample_rate = media.audio_frame_to_mono_float(frame)

        np.testing.assert_allclose(
            audio,
            np.array([32767 / 32768, -1.0], dtype=np.float32),
        )

    def test_float_audio_is_clipped_without_integer_scaling(self):
        frame = _FakeAudioFrame(
            np.array([[1.5, -1.5, 0.25]], dtype=np.float32)
        )

        audio, _sample_rate = media.audio_frame_to_mono_float(frame)

        np.testing.assert_allclose(
            audio,
            np.array([1.0, -1.0, 0.25], dtype=np.float32),
        )

    def test_resample_fallback_preserves_duration_and_float32_type(self):
        source = np.linspace(-1.0, 1.0, 480, dtype=np.float32)

        with patch.object(media, "soxr", None):
            resampled = media.resample_audio(source, 48000, 16000)

        self.assertEqual(resampled.dtype, np.float32)
        self.assertEqual(resampled.size, 160)

    def test_rtc_configuration_maps_optional_turn_credentials(self):
        created_servers = []

        class FakeIceServer:
            def __init__(self, **kwargs):
                created_servers.append(kwargs)

        class FakeConfiguration:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with (
            patch.object(media, "RTCIceServer", FakeIceServer),
            patch.object(media, "RTCConfiguration", FakeConfiguration),
        ):
            configuration = media.build_rtc_configuration(
                [
                    {"urls": "stun:example.test"},
                    {
                        "urls": "turn:example.test",
                        "username": "doctor",
                        "credential": "secret",
                    },
                ]
            )

        self.assertEqual(
            created_servers,
            [
                {"urls": "stun:example.test"},
                {
                    "urls": "turn:example.test",
                    "username": "doctor",
                    "credential": "secret",
                },
            ],
        )
        self.assertEqual(len(configuration.kwargs["iceServers"]), 2)


if __name__ == "__main__":
    unittest.main()
