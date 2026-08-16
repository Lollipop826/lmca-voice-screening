"""
Audio-based emotion recognition using Emotion2Vec+
语音情绪识别模块 - 基于阿里 Emotion2Vec+ 模型
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import numpy as np

# Emotion2Vec+ 的标签在不同版本/语言配置下可能是英文或中文。
EMOTION2VEC_TO_7D = {
    "happy": "joy",
    "happiness": "joy",
    "开心": "joy",
    "高兴": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "悲伤": "sadness",
    "伤心": "sadness",
    "angry": "anger",
    "anger": "anger",
    "生气": "anger",
    "愤怒": "anger",
    "fearful": "fear",
    "fear": "fear",
    "害怕": "fear",
    "恐惧": "fear",
    "neutral": "calm",
    "中性": "calm",
    "平静": "calm",
    "relaxed": "calm",
    "disgusted": "anger",
    "disgust": "anger",
    "厌恶": "anger",
    "surprised": "confusion",
    "surprise": "confusion",
    "惊讶": "confusion",
    "other": "confusion",
    "其他": "confusion",
    "unknown": "calm",
    "未知": "calm",
}

SYSTEM_EMOTIONS = ("joy", "sadness", "anger", "fear", "anxiety", "calm", "confusion")


class AudioEmotionClassifier:
    """基于 Emotion2Vec+ 的语音情绪识别器"""

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        use_modelscope: bool = True,
        device: str = "cpu",
        model: Any = None,
    ):
        """
        初始化语音情绪分类器

        Args:
            model_name: 模型名称，默认使用 Emotion2Vec+ Large
            use_modelscope: 是否使用 ModelScope（推荐），否则用 Hugging Face
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.model_name = model_name
        self.use_modelscope = use_modelscope
        self.device = device

        self._model = model
        self._model_attempted = model is not None or not use_modelscope
        self._last_error: str | None = None
        self._model_lock = threading.Lock()

        print(f"[AudioEmotionClassifier] 配置: model={model_name}, device={device}")

    def _load_model(self) -> Any:
        """延迟加载模型（首次调用时加载）"""
        if self._model_attempted:
            return self._model

        with self._model_lock:
            if self._model_attempted:
                return self._model

            self._model_attempted = True

            try:
                if self.use_modelscope:
                    # FunASR 直接加载本地/ModelScope 缓存，避免旧模型 requirements.txt
                    # 被 ModelScope 插件解析器误当成包名。
                    from funasr import AutoModel

                    print("[AudioEmotionClassifier] 正在加载 Emotion2Vec+ 模型（FunASR）...")
                    self._model = AutoModel(
                        model=self.model_name,
                        device=self.device,
                        disable_update=True,
                        trust_remote_code=False,
                        log_level="WARNING",
                    )
                    self._last_error = None
                    print("[AudioEmotionClassifier] ✅ 模型加载成功")
                else:
                    self._last_error = "ModelScope 未启用"

            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                print(f"[AudioEmotionClassifier] ❌ 模型加载失败: {e}")
                self._model = None

        return self._model

    @property
    def model_available(self) -> bool:
        """模型已成功加载；不可用时上层应回退到文本结果。"""
        return self._model is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def model_attempted(self) -> bool:
        return self._model_attempted

    def prewarm(self) -> bool:
        return self._load_model() is not None

    def classify_audio(
        self,
        audio_path: str,
        granularity: str = "utterance"
    ) -> Dict[str, float]:
        """
        对音频文件进行情绪识别

        Args:
            audio_path: 音频文件路径
            granularity: 粒度 ("utterance" 整句级, "frame" 帧级)

        Returns:
            7维情绪分布字典 {"joy": 0.8, "sadness": 0.1, ...}
        """
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            print(f"[AudioEmotionClassifier] ⚠️ 音频文件不存在: {audio_path}")
            return self._neutral_emotions()

        # 加载模型
        model = self._load_model()
        if model is None:
            print("[AudioEmotionClassifier] ⚠️ 模型未加载，返回中性情绪")
            return self._neutral_emotions()

        try:
            # 推理
            if hasattr(model, "generate"):
                result = model.generate(
                    input=audio_path,
                    granularity=granularity,
                    extract_embedding=False,
                )
            else:
                result = model(
                    audio_path,
                    granularity=granularity,
                    extract_embedding=False,
                )

            labels, scores = self._extract_scores(result)
            if not labels or not scores:
                self._last_error = "无法解析 Emotion2Vec 输出"
                with self._model_lock:
                    self._model = None
                print(f"[AudioEmotionClassifier] ⚠️ 无法解析模型输出: result_type={type(result).__name__}")
                return self._neutral_emotions()

            # 映射到7维情绪
            emotion_7d = self._map_to_7d(labels, scores)

            return emotion_7d

        except Exception as e:
            self._last_error = type(e).__name__
            print(f"[AudioEmotionClassifier] ❌ 推理失败: {self._last_error}")
            with self._model_lock:
                self._model = None
            return self._neutral_emotions()

    def classify_samples(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        granularity: str = "utterance",
    ) -> Dict[str, float]:
        """Classify a short in-memory window for realtime observation."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return self._neutral_emotions()
        model = self._load_model()
        if model is None:
            return self._neutral_emotions()
        try:
            kwargs = {
                "input": audio,
                "granularity": granularity,
                "extract_embedding": False,
            }
            if hasattr(model, "generate"):
                try:
                    result = model.generate(sample_rate=sample_rate, **kwargs)
                except TypeError:
                    result = model.generate(fs=sample_rate, **kwargs)
            else:
                result = model(audio, sample_rate=sample_rate, **kwargs)
            labels, scores = self._extract_scores(result)
            if labels and scores:
                return self._map_to_7d(labels, scores)
            self._last_error = "无法解析 Emotion2Vec 实时输出"
            with self._model_lock:
                self._model = None
            return self._neutral_emotions()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[AudioEmotionClassifier] 实时窗口推理失败: {exc}")
            with self._model_lock:
                self._model = None
            return self._neutral_emotions()

    @staticmethod
    def _extract_scores(result: Any) -> tuple[list[Any], list[Any]]:
        """兼容 ModelScope 当前常见的 dict/list 两种输出形状。"""
        item = result[0] if isinstance(result, list) and result else result
        if not isinstance(item, dict):
            return [], []
        if "labels" in item and "scores" in item:
            labels = item.get("labels") or []
            scores = item.get("scores") or []
            if isinstance(scores, dict):
                labels, scores = list(scores), list(scores.values())
            return list(labels), list(scores)
        if isinstance(item.get("scores"), dict):
            scores = item["scores"]
            return list(scores), list(scores.values())
        return [], []

    def _map_to_7d(
        self,
        labels: list,
        scores: list
    ) -> Dict[str, float]:
        """
        将 Emotion2Vec+ 的9类情绪映射到系统的7维

        Args:
            labels: Emotion2Vec+ 输出的标签列表
            scores: 对应的置信度列表

        Returns:
            7维情绪分布
        """
        # 初始化7维情绪
        emotion_7d = {e: 0.0 for e in SYSTEM_EMOTIONS}

        # 累加映射
        for label, score in zip(labels, scores):
            label_variants = {
                part.strip().casefold().replace(" ", "")
                for part in str(label).replace("/", "|").split("|")
            }
            target_emotion = next(
                (
                    EMOTION2VEC_TO_7D.get(variant)
                    for variant in label_variants
                    if EMOTION2VEC_TO_7D.get(variant)
                ),
                None,
            )

            if target_emotion:
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_score):
                    emotion_7d[target_emotion] += max(0.0, numeric_score)

        # 特殊处理：从 fearful 推断 anxiety
        # 如果 fear 分数高，anxiety 也应该有一定分数
        if emotion_7d["fear"] > 0.3:
            emotion_7d["anxiety"] = max(
                emotion_7d["anxiety"],
                emotion_7d["fear"] * 0.7
            )

        # 归一化
        total = sum(emotion_7d.values())
        if total > 0:
            emotion_7d = {k: v / total for k, v in emotion_7d.items()}
        else:
            emotion_7d = self._neutral_emotions()

        return emotion_7d

    def _neutral_emotions(self) -> Dict[str, float]:
        """返回中性情绪分布"""
        return {
            "joy": 0.15,
            "sadness": 0.1,
            "anger": 0.05,
            "fear": 0.05,
            "anxiety": 0.15,
            "calm": 0.4,
            "confusion": 0.1,
        }

    def get_prosody_features(self, audio_path: str) -> Dict[str, float]:
        """
        提取音频韵律特征（音高、能量、语速）
        用于情绪强度校准

        Args:
            audio_path: 音频文件路径

        Returns:
            {"pitch_mean": 150.0, "energy_mean": 0.05, "tempo": 120.0}
        """
        try:
            try:
                import librosa

                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                pitches, _ = librosa.piptrack(y=y, sr=sr)
                pitch_values = pitches[pitches > 0]
                pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
                rms = librosa.feature.rms(y=y)[0]
                energy_mean = float(np.mean(rms))
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_values = np.asarray(tempo).reshape(-1)
                tempo = float(tempo_values[0]) if tempo_values.size else 0.0
            except ImportError:
                import soundfile as sf

                y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                y = np.asarray(y, dtype=np.float32)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                pitch_mean = 0.0
                energy_mean = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
                tempo = 0.0

            return {
                "pitch_mean": pitch_mean,
                "energy_mean": energy_mean,
                "tempo": tempo
            }

        except Exception as e:
            print(f"[AudioEmotionClassifier] ⚠️ 韵律特征提取失败: {e}")
            self._last_error = f"{type(e).__name__}: {e}"
            return {"pitch_mean": 0.0, "energy_mean": 0.0, "tempo": 0.0}


# 全局单例
_AUDIO_CLASSIFIER: Optional[AudioEmotionClassifier] = None
_AUDIO_CLASSIFIER_LOCK = threading.Lock()


def get_audio_emotion_classifier() -> AudioEmotionClassifier:
    """获取全局语音情绪分类器（单例模式）"""
    global _AUDIO_CLASSIFIER

    if _AUDIO_CLASSIFIER is None:
        with _AUDIO_CLASSIFIER_LOCK:
            if _AUDIO_CLASSIFIER is None:
                _AUDIO_CLASSIFIER = AudioEmotionClassifier(
                    model_name=os.getenv(
                        "EMOTION2VEC_MODEL", "iic/emotion2vec_plus_large"
                    ),
                    use_modelscope=os.getenv("USE_MODELSCOPE", "false").lower()
                    in {"1", "true", "yes", "on"},
                    device="cuda" if os.getenv("EMOTION_USE_GPU", "false").lower() == "true" else "cpu"
                )

    return _AUDIO_CLASSIFIER
