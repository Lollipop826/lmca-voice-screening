"""
多模态情绪融合模块
Multimodal Emotion Fusion: 语音 + 文本 + 韵律特征
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np

from .emotion_classifier import EMOTION_LABELS


class MultimodalEmotionFusion:
    """多模态情绪融合器"""

    def __init__(
        self,
        audio_weight: float = 0.5,
        text_weight: float = 0.3,
        prosody_weight: float = 0.2,
    ):
        """
        初始化融合器

        Args:
            audio_weight: 语音情绪权重（基于 Emotion2Vec+）
            text_weight: 文本情绪权重（基于文本分类）
            prosody_weight: 韵律特征权重（用于强度调整）
        """
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        self.prosody_weight = prosody_weight

        weights = (audio_weight, text_weight, prosody_weight)
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError("emotion fusion weights must be finite and non-negative")
        if audio_weight + text_weight <= 0:
            raise ValueError("audio_weight and text_weight cannot both be zero")
        self.audio_weight = float(audio_weight)
        self.text_weight = float(text_weight)
        self.prosody_weight = float(prosody_weight)

    def fuse(
        self,
        audio_emotions: Dict[str, float],
        text_emotions: Dict[str, float],
        prosody_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        融合多模态情绪

        Args:
            audio_emotions: 语音情绪 {"joy": 0.6, ...}
            text_emotions: 文本情绪 {"joy": 0.7, ...}
            prosody_features: 韵律特征 {"pitch_mean": 150, "energy_mean": 0.05, "tempo": 120}

        Returns:
            融合后的情绪分布
        """
        audio = self._normalise_input(audio_emotions)
        text = self._normalise_input(text_emotions)

        # 没有韵律特征时，保留完整概率质量给音频和文本，避免把 20% 凭空丢掉。
        if prosody_features:
            prosody = self._prosody_distribution(prosody_features)
            total_weight = self.audio_weight + self.text_weight + self.prosody_weight
            fused = {
                emotion: (
                    audio[emotion] * self.audio_weight
                    + text[emotion] * self.text_weight
                    + prosody[emotion] * self.prosody_weight
                ) / total_weight
                for emotion in EMOTION_LABELS
            }
        else:
            total_weight = self.audio_weight + self.text_weight
            fused = {
                emotion: (
                    audio[emotion] * self.audio_weight
                    + text[emotion] * self.text_weight
                ) / total_weight
                for emotion in EMOTION_LABELS
            }

        # Step 3: 一致性检测（语音和文本冲突时降低置信度）
        consistency = self._calculate_consistency(audio_emotions, text_emotions)
        if consistency < 0.5:
            # 冲突较大时，拉平分布（更保守）
            fused = self._smooth_distribution(fused, strength=0.3)

        # Step 4: 归一化
        return self._normalise_input(fused)

    @staticmethod
    def _normalise_input(scores: Mapping[str, float]) -> Dict[str, float]:
        values = {}
        for emotion in EMOTION_LABELS:
            try:
                value = float(scores.get(emotion, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            values[emotion] = max(0.0, value) if math.isfinite(value) else 0.0
        total = sum(values.values())
        if total <= 0:
            return {emotion: 1.0 / len(EMOTION_LABELS) for emotion in EMOTION_LABELS}
        return {emotion: value / total for emotion, value in values.items()}

    @staticmethod
    def _prosody_distribution(prosody: Mapping[str, float]) -> Dict[str, float]:
        """Turn coarse acoustic features into a conservative seven-class prior."""
        def number(name: str) -> float:
            try:
                value = float(prosody.get(name, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        pitch = float(np.clip((number("pitch_mean") - 100.0) / 100.0, -1.0, 1.0))
        energy = float(np.clip(number("energy_mean") / 0.1, 0.0, 1.0))
        tempo = float(np.clip((number("tempo") - 100.0) / 50.0, 0.0, 1.0))
        scores = {emotion: 0.01 for emotion in EMOTION_LABELS}
        scores["calm"] += 1.0 - energy
        scores["sadness"] += max(0.0, -pitch) * (1.0 - energy)
        scores["joy"] += max(0.0, pitch) * energy
        scores["anger"] += max(0.0, pitch) * energy * 0.6
        scores["anxiety"] += tempo * energy
        scores["fear"] += tempo * energy * 0.5
        return MultimodalEmotionFusion._normalise_input(scores)

    def _adjust_by_prosody(
        self,
        emotions: Dict[str, float],
        prosody: Dict[str, float]
    ) -> Dict[str, float]:
        """
        基于韵律特征调整情绪强度

        规则：
        - 高音调 + 高能量 → 增强 joy/anger
        - 低音调 + 低能量 → 增强 sadness/calm
        - 快语速 + 高能量 → 增强 anxiety/fear
        """
        adjusted = emotions.copy()

        pitch = prosody.get("pitch_mean", 0.0)
        energy = prosody.get("energy_mean", 0.0)
        tempo = prosody.get("tempo", 0.0)

        # 特征归一化（基于经验值）
        pitch_norm = np.clip((pitch - 100) / 100, -1, 1)  # 100-200Hz范围
        energy_norm = np.clip(energy / 0.1, 0, 1)  # 0-0.1范围
        tempo_norm = np.clip((tempo - 100) / 50, 0, 1)  # 100-150 BPM范围

        # 调整规则
        if pitch_norm > 0.3 and energy_norm > 0.3:
            # 高音调高能量 → joy/anger
            adjusted["joy"] *= (1 + 0.2 * pitch_norm)
            adjusted["anger"] *= (1 + 0.15 * energy_norm)

        if pitch_norm < -0.3 and energy_norm < 0.3:
            # 低音调低能量 → sadness/calm
            adjusted["sadness"] *= (1 + 0.2 * abs(pitch_norm))
            adjusted["calm"] *= (1 + 0.15 * (1 - energy_norm))

        if tempo_norm > 0.5 and energy_norm > 0.5:
            # 快语速高能量 → anxiety/fear
            adjusted["anxiety"] *= (1 + 0.3 * tempo_norm)
            adjusted["fear"] *= (1 + 0.2 * energy_norm)

        return adjusted

    def _calculate_consistency(
        self,
        audio_emotions: Dict[str, float],
        text_emotions: Dict[str, float]
    ) -> float:
        """
        计算语音和文本情绪的一致性

        Returns:
            一致性分数 (0-1)，越高越一致
        """
        # 方法：计算两个分布的余弦相似度
        audio_vec = np.array([audio_emotions.get(e, 0) for e in EMOTION_LABELS])
        text_vec = np.array([text_emotions.get(e, 0) for e in EMOTION_LABELS])

        # 余弦相似度
        dot_product = np.dot(audio_vec, text_vec)
        norm_audio = np.linalg.norm(audio_vec)
        norm_text = np.linalg.norm(text_vec)

        if norm_audio == 0 or norm_text == 0:
            return 0.5  # 默认中等一致性

        similarity = dot_product / (norm_audio * norm_text)
        return float(np.clip((similarity + 1) / 2, 0, 1))  # 归一化到[0,1]

    def _smooth_distribution(
        self,
        emotions: Dict[str, float],
        strength: float = 0.3
    ) -> Dict[str, float]:
        """
        平滑情绪分布（降低极端值）

        Args:
            emotions: 原始情绪分布
            strength: 平滑强度 (0-1)

        Returns:
            平滑后的分布
        """
        uniform = 1.0 / len(emotions)
        smoothed = {}

        for emotion, score in emotions.items():
            smoothed[emotion] = score * (1 - strength) + uniform * strength

        return smoothed


# 便捷函数
def fuse_emotions(
    audio_emotions: Dict[str, float],
    text_emotions: Dict[str, float],
    prosody_features: Optional[Dict[str, float]] = None,
    audio_weight: float = 0.5,
    text_weight: float = 0.3,
    prosody_weight: float = 0.2,
) -> Dict[str, float]:
    """
    融合多模态情绪的便捷函数

    Args:
        audio_emotions: 语音情绪
        text_emotions: 文本情绪
        prosody_features: 韵律特征（可选）
        audio_weight: 语音权重
        text_weight: 文本权重

    Returns:
        融合后的7维情绪分布
    """
    fusion = MultimodalEmotionFusion(
        audio_weight=audio_weight,
        text_weight=text_weight,
        prosody_weight=prosody_weight,
    )
    return fusion.fuse(audio_emotions, text_emotions, prosody_features)
