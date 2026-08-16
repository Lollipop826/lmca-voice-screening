"""
情绪识别模块 - 统一API
支持文本情绪识别、语音情绪识别、多模态融合
"""
import os
import time
from .emotion_classifier import EMOTION_LABELS, EmotionClassifier, classify_emotion
from .audio_emotion_classifier import AudioEmotionClassifier, get_audio_emotion_classifier
from .multimodal_fusion import MultimodalEmotionFusion, fuse_emotions

from typing import Any, Dict, Optional
import numpy as np

__all__ = [
    "EMOTION_LABELS",
    "EmotionClassifier",
    "classify_emotion",
    "AudioEmotionClassifier",
    "get_audio_emotion_classifier",
    "MultimodalEmotionFusion",
    "fuse_emotions",
    "classify_multimodal",
    "classify_multimodal_with_metadata",
    "classify_realtime_emotion",
]


def classify_multimodal_with_metadata(
    text: str,
    audio_path: Optional[str] = None,
    audio_weight: float = 0.5,
    text_weight: float = 0.3,
    prosody_weight: float = 0.2,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Return fused scores and the evidence actually used for this call."""
    text_emotions = classify_emotion(text)
    metadata: Dict[str, Any] = {
        "source": "text",
        "audio_model_used": False,
        "inference_ms": 0.0,
        "fallback_reason": None,
    }

    if audio_path is None:
        return text_emotions, metadata

    if not os.path.isfile(audio_path):
        metadata.update(
            source="text_fallback_audio_missing",
            fallback_reason="audio_missing",
        )
        return text_emotions, metadata

    try:
        audio_classifier = get_audio_emotion_classifier()
        if (
            not getattr(audio_classifier, "model_available", True)
            and getattr(audio_classifier, "model_attempted", True)
        ):
            metadata.update(
                source="text_fallback_model_unavailable",
                fallback_reason=getattr(audio_classifier, "last_error", None)
                or "model_unavailable",
            )
            return text_emotions, metadata

        started_at = time.perf_counter()
        audio_emotions = audio_classifier.classify_audio(audio_path)
        metadata["inference_ms"] = (time.perf_counter() - started_at) * 1000
        if not getattr(audio_classifier, "model_available", True):
            print(
                "[classify_multimodal] ⚠️ 语音模型不可用，保持文本情绪结果"
            )
            metadata.update(
                source="text_fallback_model_unavailable",
                fallback_reason=getattr(audio_classifier, "last_error", None)
                or "model_unavailable",
            )
            return text_emotions, metadata

        prosody_features = audio_classifier.get_prosody_features(audio_path)
        fused_emotions = fuse_emotions(
            audio_emotions=audio_emotions,
            text_emotions=text_emotions,
            prosody_features=prosody_features,
            audio_weight=audio_weight,
            text_weight=text_weight,
            prosody_weight=prosody_weight,
        )
        metadata.update(
            source="emotion2vec_audio+text",
            audio_model_used=True,
        )
        return fused_emotions, metadata

    except Exception as exc:
        print(f"[classify_multimodal] ⚠️ 语音情绪识别失败，降级为文本: {exc}")
        metadata.update(
            source="text_fallback_audio_error",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
        return text_emotions, metadata


def classify_multimodal(
    text: str,
    audio_path: Optional[str] = None,
    audio_weight: float = 0.5,
    text_weight: float = 0.3,
    prosody_weight: float = 0.2,
) -> Dict[str, float]:
    """兼容旧调用方，只返回七维情绪分数。"""
    scores, _ = classify_multimodal_with_metadata(
        text,
        audio_path,
        audio_weight,
        text_weight,
        prosody_weight,
    )
    return scores


def classify_realtime_emotion(
    text: str,
    samples: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """Fuse a rolling Emotion2Vec+ window with provisional ASR text."""
    text_emotions = classify_emotion(text)
    classifier = get_audio_emotion_classifier()
    audio_emotions = classifier.classify_samples(samples, sample_rate)
    if not classifier.model_available:
        return text_emotions
    return fuse_emotions(
        audio_emotions=audio_emotions,
        text_emotions=text_emotions,
        audio_weight=0.65,
        text_weight=0.35,
        prosody_weight=0.0,
    )
