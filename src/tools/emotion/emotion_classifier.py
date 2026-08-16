"""Chinese seven-dimensional emotion classification with a rule fallback."""

from __future__ import annotations

import math
import os
import re
import threading
from typing import Any, Iterator, Mapping


EMOTION_LABELS = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "anxiety",
    "calm",
    "confusion",
)
DEFAULT_MODEL_NAME = os.getenv(
    "EMOTION_MODEL_NAME", "uer/roberta-base-finetuned-dianping-chinese"
)

_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "joy": (
        ("开心", 1.3),
        ("高兴", 1.3),
        ("快乐", 1.3),
        ("幸福", 1.2),
        ("喜悦", 1.2),
        ("满意", 1.0),
        ("欣慰", 1.0),
        ("喜欢", 0.8),
        ("兴奋", 1.1),
        ("哈哈", 0.8),
    ),
    "sadness": (
        ("难过", 1.3),
        ("伤心", 1.3),
        ("悲伤", 1.3),
        ("失落", 1.1),
        ("低落", 1.1),
        ("孤独", 1.0),
        ("寂寞", 1.0),
        ("想哭", 1.1),
        ("绝望", 1.4),
        ("遗憾", 0.8),
    ),
    "anger": (
        ("生气", 1.3),
        ("愤怒", 1.4),
        ("恼火", 1.2),
        ("烦躁", 1.1),
        ("暴躁", 1.1),
        ("气死", 1.3),
        ("火大", 1.2),
        ("憋屈", 1.0),
        ("讨厌", 0.8),
    ),
    "fear": (
        ("害怕", 1.3),
        ("恐惧", 1.4),
        ("恐慌", 1.4),
        ("惊恐", 1.3),
        ("可怕", 1.0),
        ("不敢", 0.9),
        ("吓", 0.8),
    ),
    "anxiety": (
        ("焦虑", 1.4),
        ("紧张", 1.2),
        ("不安", 1.2),
        ("担忧", 1.1),
        ("忧虑", 1.1),
        ("担心", 1.0),
        ("发愁", 1.0),
        ("心慌", 1.1),
        ("忐忑", 1.1),
        ("睡不着", 0.8),
    ),
    "calm": (
        ("平静", 1.3),
        ("放松", 1.3),
        ("安心", 1.2),
        ("踏实", 1.1),
        ("安静", 0.9),
        ("稳定", 0.8),
        ("舒服", 0.8),
        ("释然", 1.1),
        ("平和", 1.1),
    ),
    "confusion": (
        ("困惑", 1.3),
        ("疑惑", 1.2),
        ("糊涂", 1.2),
        ("迷茫", 1.2),
        ("混乱", 1.2),
        ("不明白", 1.0),
        ("搞不懂", 1.0),
        ("记不清", 1.0),
        ("想不起来", 1.0),
        ("不知道", 0.7),
    ),
}

_NEGATIONS = ("从来不", "并不", "不太", "不是", "没有", "不要", "没", "不", "无", "未", "别", "莫", "not", "never")
_LABEL_ALIASES = {
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "positive": "joy",
    "正面": "joy",
    "正向": "joy",
    "积极": "joy",
    "开心": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "negative": "sadness",
    "负面": "sadness",
    "悲伤": "sadness",
    "难过": "sadness",
    "anger": "anger",
    "angry": "anger",
    "愤怒": "anger",
    "生气": "anger",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "恐惧": "fear",
    "害怕": "fear",
    "anxiety": "anxiety",
    "anxious": "anxiety",
    "worried": "anxiety",
    "worry": "anxiety",
    "焦虑": "anxiety",
    "calm": "calm",
    "relaxed": "calm",
    "peaceful": "calm",
    "平静": "calm",
    "放松": "calm",
    "confusion": "confusion",
    "confused": "confusion",
    "uncertain": "confusion",
    "困惑": "confusion",
    "疑惑": "confusion",
}


def _uniform_distribution() -> dict[str, float]:
    result = {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
    result[EMOTION_LABELS[-1]] += 1.0 - sum(result.values())
    return result


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    values: list[float] = []
    for label in EMOTION_LABELS:
        try:
            value = float(scores.get(label, 0.0))
        except (TypeError, ValueError):
            return _uniform_distribution()
        if not math.isfinite(value):
            return _uniform_distribution()
        values.append(value)

    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return _uniform_distribution()

    probabilities = [weight / total for weight in weights]
    probabilities[-1] += 1.0 - sum(probabilities)
    return dict(zip(EMOTION_LABELS, probabilities))


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 5) : start].rstrip()
    return any(prefix.endswith(negation) for negation in _NEGATIONS)


def _rule_scores(text: str) -> dict[str, float]:
    normalized = text.casefold()
    scores = {label: 0.0 for label in EMOTION_LABELS}
    for label, keywords in _KEYWORDS.items():
        for keyword, weight in keywords:
            for match in re.finditer(re.escape(keyword.casefold()), normalized):
                if not _is_negated(normalized, match.start()):
                    scores[label] += weight
    return scores


def _canonical_label(label: Any) -> str | None:
    normalized = re.sub(r"[\s_-]+", "", str(label).strip().casefold())
    return _LABEL_ALIASES.get(normalized)


def _iter_prediction_items(output: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(output, Mapping):
        if "label" in output:
            yield output
        else:
            for label, score in output.items():
                yield {"label": label, "score": score}
        return

    if not isinstance(output, (list, tuple)):
        return
    if len(output) == 1 and isinstance(output[0], (list, tuple)):
        output = output[0]
    if len(output) == 2 and not isinstance(output[0], (list, tuple, Mapping)):
        yield {"label": output[0], "score": output[1]}
        return
    for item in output:
        if isinstance(item, Mapping):
            yield item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            yield {"label": item[0], "score": item[1]}


class EmotionClassifier:
    """Return seven normalized emotion scores; Transformers is opt-in."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_transformers: bool = False,
        model: Any = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.use_transformers = use_transformers
        self.local_files_only = local_files_only
        self._model = model
        self._model_attempted = model is not None or not use_transformers
        self._model_lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model_attempted:
            return self._model
        with self._model_lock:
            if self._model_attempted:
                return self._model
            self._model_attempted = True
            try:
                from transformers import pipeline

                model_kwargs = {"local_files_only": self.local_files_only}
                tokenizer_kwargs = {"local_files_only": self.local_files_only}
                self._model = pipeline(
                    "text-classification",
                    model=self.model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
            except Exception:
                self._model = None
        return self._model

    def _model_label(self, label: Any) -> str | None:
        canonical = _canonical_label(label)
        if canonical:
            return canonical

        normalized = re.sub(r"[\s_-]+", "", str(label).strip().casefold())
        match = re.fullmatch(r"label(\d+)", normalized)
        if not match or self._model is None:
            return None

        model = getattr(self._model, "model", self._model)
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", {}) or {}
        index = match.group(1)
        mapped = id2label.get(index, id2label.get(int(index), None))
        return _canonical_label(mapped) if mapped is not None else None

    def _model_scores(self, text: str) -> dict[str, float]:
        model = self._load_model()
        if model is None:
            return {}
        try:
            output = model(text)
        except Exception:
            with self._model_lock:
                self._model = None
            return {}

        scores: dict[str, float] = {}
        for item in _iter_prediction_items(output):
            label = self._model_label(item.get("label"))
            if label is None:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores[label] = scores.get(label, 0.0) + max(0.0, min(score, 1.0))
        return scores

    def classify(self, text: str | None) -> dict[str, float]:
        """Classify text; empty or unsupported input returns a uniform distribution."""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return _uniform_distribution()

        scores = _rule_scores(text)
        for label, score in self._model_scores(text).items():
            scores[label] += score
        return _softmax(scores)

    predict = classify


_DEFAULT_CLASSIFIER = EmotionClassifier(
    use_transformers=os.getenv("EMOTION_USE_TRANSFORMERS", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"},
    local_files_only=os.getenv("EMOTION_LOCAL_FILES_ONLY", "true")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"},
)


def classify_emotion(text: str | None) -> dict[str, float]:
    """Convenience wrapper using a lazily loaded default classifier."""
    return _DEFAULT_CLASSIFIER.classify(text)


if __name__ == "__main__":
    classifier = EmotionClassifier(use_transformers=False)
    for sample, expected in (
        ("我今天很开心", "joy"),
        ("我最近非常焦虑", "anxiety"),
        ("我现在很平静", "calm"),
    ):
        result = classifier.classify(sample)
        assert set(result) == set(EMOTION_LABELS)
        assert abs(sum(result.values()) - 1.0) < 1e-12
        assert max(result, key=result.get) == expected
    assert abs(sum(classifier.classify(" ").values()) - 1.0) < 1e-12
    print("emotion_classifier self-check passed")
