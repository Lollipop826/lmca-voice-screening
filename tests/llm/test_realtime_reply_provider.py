from src.llm import http_client_pool
from src.tools.agent_tools import comfort_response_tool
from src.tools.agent_tools import question_generation_tool
from src.tools.agent_tools import standard_question_tool


def test_patient_facing_reply_tools_prefer_unified_dashscope_route(monkeypatch):
    for name in (
        "QUESTION_GEN_MODEL",
        "QUESTION_GEN_BALANCED_MODEL",
        "QUESTION_GEN_FAST_MODEL",
        "COMFORT_TOOL_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-flash")
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")

    question_calls = []
    comfort_calls = []
    standard_calls = []

    monkeypatch.setattr(
        question_generation_tool,
        "get_chat_openai",
        lambda **kwargs: question_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        comfort_response_tool,
        "get_chat_openai",
        lambda **kwargs: comfort_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        http_client_pool,
        "get_chat_openai",
        lambda **kwargs: standard_calls.append(kwargs) or object(),
    )

    question_tool = question_generation_tool.QuestionGenerationTool()
    comfort_response_tool.ComfortResponseTool()
    standard_question_tool.StandardQuestionTool(use_local=False)

    assert question_tool._default_model == "qwen3.7-flash"
    assert question_tool._use_volcengine is False
    assert question_calls[0]["model"] == "qwen3.7-flash"
    assert comfort_calls == [{"model": None, "temperature": 0.8, "max_tokens": 150, "timeout": 20, "max_retries": 1}]
    assert standard_calls == [{"model": None, "temperature": 0.7, "max_tokens": 200, "timeout": 10, "max_retries": 1}]
