import pytest

from src.llm import http_client_pool


def test_get_chat_openai_prefers_dashscope(monkeypatch):
    captured = {}

    def fake_dashscope(**kwargs):
        captured.update(kwargs)
        return "dashscope-client"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-flash")
    monkeypatch.setattr(http_client_pool, "get_dashscope_chat_openai", fake_dashscope)

    assert http_client_pool.get_chat_openai(max_tokens=80) == "dashscope-client"
    assert captured["model"] == "qwen3.7-flash"
    assert captured["max_tokens"] == 80


def test_dashscope_client_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        http_client_pool.get_dashscope_chat_openai(model="qwen3.7-flash")
