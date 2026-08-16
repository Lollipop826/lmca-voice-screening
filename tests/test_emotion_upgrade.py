from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import src.tools.emotion as emotion_api
from src.context_management.emotion_memobase import EmotionMemobase
from src.tools.emotion.audio_emotion_classifier import AudioEmotionClassifier
from src.tools.emotion.emotion_classifier import EMOTION_LABELS, classify_emotion


def _write_audio(path: Path) -> None:
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)


def test_audio_classifier_maps_english_and_chinese_model_labels(tmp_path: Path):
    class FakeModel:
        def __call__(self, _audio_path, **_kwargs):
            return [{
                "labels": ["生气/angry", "neutral", "surprised"],
                "scores": [0.6, 0.25, 0.15],
            }]

    audio_path = tmp_path / "sample.wav"
    _write_audio(audio_path)
    classifier = AudioEmotionClassifier(model=FakeModel())

    result = classifier.classify_audio(str(audio_path))

    assert classifier.model_available
    assert set(result) == set(EMOTION_LABELS)
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert result["anger"] > result["calm"]
    assert result["confusion"] > 0


def test_multimodal_falls_back_to_text_when_audio_model_is_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    class UnavailableClassifier:
        model_available = False

        def classify_audio(self, _audio_path):
            raise AssertionError("unavailable model must not be used as a result")

    audio_path = tmp_path / "sample.wav"
    _write_audio(audio_path)
    monkeypatch.setattr(
        emotion_api,
        "get_audio_emotion_classifier",
        lambda: UnavailableClassifier(),
    )

    expected = classify_emotion("我最近很焦虑")
    actual = emotion_api.classify_multimodal("我最近很焦虑", str(audio_path))

    assert actual == expected


def test_multimodal_metadata_marks_text_fallback(tmp_path: Path, monkeypatch):
    class UnavailableClassifier:
        model_available = False
        model_attempted = True
        last_error = "missing runtime"

    audio_path = tmp_path / "sample.wav"
    _write_audio(audio_path)
    monkeypatch.setattr(
        emotion_api,
        "get_audio_emotion_classifier",
        lambda: UnavailableClassifier(),
    )

    scores, metadata = emotion_api.classify_multimodal_with_metadata(
        "我最近很焦虑", str(audio_path)
    )

    assert scores == classify_emotion("我最近很焦虑")
    assert metadata["source"] == "text_fallback_model_unavailable"
    assert metadata["audio_model_used"] is False


def test_multimodal_metadata_marks_real_audio_use(tmp_path: Path, monkeypatch):
    class AvailableClassifier:
        model_available = True
        model_attempted = True
        last_error = None

        def classify_audio(self, _audio_path):
            return {
                "joy": 0.05, "sadness": 0.75, "anger": 0.05,
                "fear": 0.05, "anxiety": 0.05, "calm": 0.03,
                "confusion": 0.02,
            }

        def get_prosody_features(self, _audio_path):
            return {"pitch_mean": 80, "energy_mean": 0.01, "tempo": 80}

    audio_path = tmp_path / "sample.wav"
    _write_audio(audio_path)
    monkeypatch.setattr(
        emotion_api,
        "get_audio_emotion_classifier",
        lambda: AvailableClassifier(),
    )

    _scores, metadata = emotion_api.classify_multimodal_with_metadata(
        "我今天很开心", str(audio_path)
    )

    assert metadata["source"] == "emotion2vec_audio+text"
    assert metadata["audio_model_used"] is True
    assert metadata["inference_ms"] >= 0


def test_multimodal_uses_audio_and_prosody_when_model_is_available(
    tmp_path: Path,
    monkeypatch,
):
    class AvailableClassifier:
        model_available = True

        def classify_audio(self, _audio_path):
            return {
                "joy": 0.05,
                "sadness": 0.75,
                "anger": 0.05,
                "fear": 0.05,
                "anxiety": 0.05,
                "calm": 0.03,
                "confusion": 0.02,
            }

        def get_prosody_features(self, _audio_path):
            return {"pitch_mean": 80, "energy_mean": 0.01, "tempo": 80}

    audio_path = tmp_path / "sample.wav"
    _write_audio(audio_path)
    monkeypatch.setattr(
        emotion_api,
        "get_audio_emotion_classifier",
        lambda: AvailableClassifier(),
    )

    result = emotion_api.classify_multimodal("我今天很开心", str(audio_path))

    assert set(result) == set(EMOTION_LABELS)
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert result["sadness"] > result["joy"]


def test_emotion_trajectory_is_written_and_queryable(tmp_path: Path):
    memory = EmotionMemobase(storage_path=str(tmp_path / "memory.db"))
    result = memory.capture_turn(
        patient_id="patient-1",
        user_message="我最近有点焦虑。",
        assistant_message="我们慢慢聊。",
        session_id="session-1",
        emotions={
            "joy": 0.05,
            "sadness": 0.10,
            "anger": 0.05,
            "fear": 0.05,
            "anxiety": 0.60,
            "calm": 0.10,
            "confusion": 0.05,
        },
        audio_path="data/example.wav",
    )

    trajectory = memory.get_emotion_trajectory("patient-1")
    snapshot = memory.get_snapshot("patient-1")

    assert result["dominant_emotion"] == "anxiety"
    assert trajectory[0]["session_id"] == "session-1"
    assert trajectory[0]["audio_path"] == "data/example.wav"
    assert trajectory[0]["dominant_emotion"] == "anxiety"
    assert trajectory[0]["arousal"] > 0
    assert snapshot["emotion_trajectory"] == trajectory
