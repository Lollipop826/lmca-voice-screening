"""Regression tests for Emotion2Vec failure recovery and retrieval provenance.

Both behaviours here previously caused silent, process-wide degradation that
looked healthy from the outside:

* A single inference failure permanently disabled audio emotion recognition,
  so health checks reported the model as prewarmed while every turn fell back
  to text-only classification.
* The query-independent local memory fallback was reported as a semantic
  retrieval hit, which made memory ablation runs look like retrieval worked
  when the dumped items were chosen by recency rather than relevance.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np

from src.tools.emotion.audio_emotion_classifier import AudioEmotionClassifier
from src.voice.handlers.speech_turn_processor import classify_memory_provenance


class _ScriptedModel:
    """Fake Emotion2Vec model whose per-call outcome is scripted."""

    def __init__(self, outcomes: list[object] | None = None):
        self._outcomes = list(outcomes or [])
        self.call_count = 0

    def generate(self, **_kwargs):
        self.call_count += 1
        outcome = self._outcomes.pop(0) if self._outcomes else self._healthy()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def _healthy():
        return [{"labels": ["happy"], "scores": [0.9]}]


def _fake_funasr(models: list[_ScriptedModel]) -> tuple[types.ModuleType, list[dict]]:
    """Stub out only model construction so the real load path still runs.

    Patching ``_load_model`` itself would bypass the backoff bookkeeping that
    these tests exist to verify, so the seam is placed at the FunASR import
    instead.
    """
    remaining = list(models)
    load_calls: list[dict] = []

    def auto_model(**kwargs):
        load_calls.append(kwargs)
        if not remaining:
            raise RuntimeError("model construction unexpectedly repeated")
        return remaining.pop(0)

    module = types.ModuleType("funasr")
    module.AutoModel = auto_model
    return module, load_calls


def _voiced_samples() -> np.ndarray:
    samples = np.zeros(16000, dtype=np.float32)
    samples[::2] = 0.1
    return samples


def test_inference_failure_does_not_permanently_disable_audio_emotion():
    """A transient inference error must not lock the model out for good."""
    failing_model = _ScriptedModel([RuntimeError("cuda hiccup")])
    healthy_model = _ScriptedModel()
    module, load_calls = _fake_funasr([failing_model, healthy_model])
    samples = _voiced_samples()

    with mock.patch.dict(sys.modules, {"funasr": module}):
        classifier = AudioEmotionClassifier(use_modelscope=True, device="cpu")

        classifier.classify_samples(samples)
        assert classifier.model_available is False, "failed model should be dropped"
        assert len(load_calls) == 1

        # Expire the cooldown rather than waiting it out, so the assertion is
        # about recoverability instead of wall-clock timing.
        classifier._reload_blocked_until = 0.0
        recovered = classifier.classify_samples(samples)

    assert classifier.model_available is True, "classifier must recover after failure"
    assert len(load_calls) == 2, "a second load should be attempted after backoff"
    assert healthy_model.call_count == 1
    assert recovered["joy"] > 0.5


def test_reload_is_rate_limited_after_a_failure():
    """Recovery must not reload the large model on every single turn."""
    failing_model = _ScriptedModel([RuntimeError("cuda hiccup")])
    replacement_model = _ScriptedModel()
    module, load_calls = _fake_funasr([failing_model, replacement_model])
    samples = _voiced_samples()

    with mock.patch.dict(sys.modules, {"funasr": module}):
        classifier = AudioEmotionClassifier(use_modelscope=True, device="cpu")
        classifier.classify_samples(samples)
        # Retry immediately, while the cooldown is still active.
        classifier.classify_samples(samples)

    assert len(load_calls) == 1, "reload should wait for the backoff to expire"
    assert replacement_model.call_count == 0
    assert classifier.model_available is False


def test_unparseable_output_also_stays_recoverable():
    """A malformed response is treated like any other transient failure."""
    garbage_model = _ScriptedModel([[{"unexpected": "shape"}]])
    healthy_model = _ScriptedModel()
    module, load_calls = _fake_funasr([garbage_model, healthy_model])
    samples = _voiced_samples()

    with mock.patch.dict(sys.modules, {"funasr": module}):
        classifier = AudioEmotionClassifier(use_modelscope=True, device="cpu")
        classifier.classify_samples(samples)
        assert classifier.model_available is False

        classifier._reload_blocked_until = 0.0
        recovered = classifier.classify_samples(samples)

    assert classifier.model_available is True
    assert len(load_calls) == 2
    assert recovered["joy"] > 0.5


def test_local_fallback_is_not_reported_as_a_semantic_hit():
    """The recency dump must be distinguishable from real semantic retrieval."""
    fallback_context = (
        "[retrieved_long_term_memory]\n"
        "[local_active_memory]\n"
        "1. 来源：SQLite active memory_items；item_id=event-community-walk；"
        "近期事件：上周参加了社区散步活动"
    )

    kind, used_item_ids = classify_memory_provenance(fallback_context)

    assert kind == "local_fallback"
    assert used_item_ids == ["event-community-walk"]


def test_semantic_retrieval_reports_the_gist_ids_that_reached_the_prompt():
    """A semantic hit must name its gists, not report zero used items.

    The renderer previously dropped ``event_gist_id`` when building the prompt
    block, so every genuine semantic hit reported an empty item list.  Ablation
    runs then showed ``memory_used_items=0`` alongside a 6/6 hit rate, which
    made it impossible to tell a working retrieval from a broken one.
    """
    semantic_context = (
        "[retrieved_long_term_memory]\n"
        "1. 摘要：和女儿因为看病的事情起了争执\n"
        "   创建时间：2026-09-01T10:00:00；相似度：0.812"
        "；event_gist_id=gist-daughter-conflict"
    )

    kind, used_item_ids = classify_memory_provenance(semantic_context)

    assert kind == "semantic"
    assert used_item_ids == ["gist-daughter-conflict"]


def test_rendered_semantic_gists_round_trip_into_used_item_ids():
    """Renderer and telemetry must agree on the gist id format.

    Asserting the two halves separately let them drift apart: the parser kept
    looking for an id the renderer had stopped emitting.  Feeding real renderer
    output into the classifier keeps that regression visible.
    """
    from src.context_management.emotion_memobase import EmotionMemobase

    rendered = EmotionMemobase._render_event_gists(
        [
            {
                "content": "和女儿因为看病的事情起了争执",
                "created_at": "2026-09-01T10:00:00",
                "similarity": 0.812,
                "event_gist_id": "gist-daughter-conflict",
            },
            {
                "content": "周三固定和邻居一起散步",
                "created_at": "2026-09-03T09:30:00",
                "similarity": 0.694,
                "event_gist_id": "gist-community-walk",
            },
        ]
    )

    kind, used_item_ids = classify_memory_provenance(rendered)

    assert kind == "semantic"
    assert used_item_ids == ["gist-daughter-conflict", "gist-community-walk"]


def test_semantic_gist_ids_do_not_capture_the_local_fallback_marker():
    """The two provenance paths must not contaminate each other's ids.

    ``event_gist_id`` and ``item_id`` both end in ``item_id``, so an unanchored
    pattern would report local-fallback ids as semantic hits and hide the
    degraded path the ablation protocol is meant to reject.
    """
    fallback_context = (
        "[retrieved_long_term_memory]\n"
        "[local_active_memory]\n"
        "1. 来源：SQLite active memory_items；item_id=event-community-walk；"
        "近期事件：上周参加了社区散步活动"
    )

    kind, used_item_ids = classify_memory_provenance(fallback_context)

    assert kind == "local_fallback"
    assert used_item_ids == ["event-community-walk"]


def test_below_threshold_semantic_search_is_not_a_hit():
    empty_context = "[retrieved_long_term_memory]\n无通过阈值的跨会话长期事件。"

    kind, used_item_ids = classify_memory_provenance(empty_context)

    assert kind == "empty"
    assert used_item_ids == []


def test_absent_memory_context_reports_none():
    assert classify_memory_provenance("") == ("none", [])
    assert classify_memory_provenance(None) == ("none", [])
