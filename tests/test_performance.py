from __future__ import annotations

from statistics import mean
from time import perf_counter

from src.context_management.emotion_memobase import EmotionMemobase


def test_capture_turn_average_latency(tmp_path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    samples = []
    for index in range(10):
        started = perf_counter()
        memory.capture_turn(
            "performance-patient",
            f"这是第{index}次测试对话。",
            "收到，我们继续。",
            session_id=f"session-{index}",
        )
        samples.append(perf_counter() - started)

    assert mean(samples) < 0.5


def test_context_retrieval_average_latency(tmp_path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    for index in range(20):
        memory.capture_turn(
            "performance-patient",
            f"这是第{index}条历史消息。",
            "好的。",
            session_id="history",
        )

    samples = []
    for _ in range(10):
        started = perf_counter()
        assert memory.get_context_for_llm("performance-patient")
        samples.append(perf_counter() - started)

    assert mean(samples) < 0.2
