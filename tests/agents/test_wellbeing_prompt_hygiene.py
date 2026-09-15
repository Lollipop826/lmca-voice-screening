"""Guard the companion prompt against leaking internal signals to the user.

A live session produced "虽然情绪线索显示平静，但……" because the emotion
estimate entered the context as a bare ``【情绪线索】`` block with no usage
note, so the model treated the label as conversational material. These tests
pin down both halves of the fix: the label now carries its own instructions,
and the system prompt forbids repeating internal analysis.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.wellbeing_companion_agent import (
    WellbeingCompanionAgent,
    _emotion_control_block,
    _normalize_emotion_label,
)


class _StreamingLLM:
    def __init__(self, parts):
        self.parts = parts

    def stream(self, _messages):
        for part in self.parts:
            yield SimpleNamespace(content=part)


def _human_message_text(agent: WellbeingCompanionAgent, **overrides) -> str:
    arguments = {
        "text": "我今天心情不太好。",
        "profile": {},
        "chat_history": [],
        "current_emotion": "calm",
    }
    arguments.update(overrides)
    messages = agent._build_messages(
        arguments["text"],
        arguments["profile"],
        arguments["chat_history"],
        arguments["current_emotion"],
    )
    return str(messages[-1].content)


@pytest.fixture
def agent() -> WellbeingCompanionAgent:
    return WellbeingCompanionAgent(llm_factory=lambda: None)


def test_emotion_block_is_labelled_as_internal_and_not_for_disclosure(agent):
    content = _human_message_text(agent)

    assert "【情绪线索】" not in content
    assert "【内部语音情绪控制信号·严禁向用户披露】" in content
    assert "识别结果：平静或无明显情绪（calm）" in content
    assert "不要说出标签" in content


def test_emotion_label_defers_to_the_speaker_own_words(agent):
    """A wrong estimate must not become something the model argues with."""
    content = _human_message_text(agent, current_emotion="calm")

    assert "不是新的事实" in content
    assert "不要推断原因、对象、强度" in content


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("happy", "joy"),
        ("sad", "sadness"),
        ("angry", "anger"),
        ("fearful", "fear"),
        ("neutral", "calm"),
        ("unsupported", "calm"),
    ],
)
def test_emotion_label_aliases_are_normalized(raw, normalized):
    assert _normalize_emotion_label(raw) == normalized


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("joy", "积极感或完成后的释然"),
        ("sadness", "优先倾听"),
        ("anger", "边界、选择权"),
        ("fear", "不要擅自猜测具体担忧"),
        ("anxiety", "缩小到眼前一步"),
        ("confusion", "澄清一个最关键的小点"),
        ("calm", "直接回应用户字面内容"),
    ],
)
def test_each_emotion_has_actionable_bounded_guidance(label, expected):
    block = _emotion_control_block(label)

    assert expected in block
    assert "调节语气、回应顺序和建议力度" in block
    assert "不要说出标签" in block


def test_system_prompt_forbids_repeating_internal_analysis(agent):
    prompt = agent._SYSTEM_PROMPT

    assert "不要提及、复述或引用任何内部分析结果" in prompt
    assert "不要描述你自己的判断过程" in prompt


def test_system_prompt_requires_using_voice_signal_for_ambiguous_text(agent):
    prompt = agent._SYSTEM_PROMPT

    assert "文字没有写明情绪或语义有歧义时" in prompt
    assert "不能因为文字没提情绪就忽略它" in prompt
    assert "调整语气、回应顺序以及建议力度" in prompt
    assert "不使用固定开场" in prompt
    assert "不再复述被拒绝的话术" in prompt
    assert "以用户原话为准" in prompt


def test_current_turn_repetition_is_not_treated_as_old_memory(agent):
    prompt = agent._SYSTEM_PROMPT

    assert "当前表达在近期对话中的重复不算旧记忆" in prompt


def test_current_user_turn_is_removed_from_recent_history(agent):
    content = _human_message_text(
        agent,
        text="这件事终于完成了。",
        chat_history=[
            {"role": "assistant", "content": "你可以慢慢说。"},
            {"role": "user", "content": "这件事终于完成了。"},
        ],
        current_emotion="joy",
    )

    assert "助手：你可以慢慢说。" in content
    assert "对方：这件事终于完成了。" not in content


def test_no_memory_block_explicitly_forbids_inventing_prior_context(agent):
    content = _human_message_text(agent, chat_history=[])

    assert "【旧记忆可用性】无" in content
    assert "不得声称用户在本轮之前提过" in content


def test_system_prompt_offers_several_response_shapes(agent):
    """The old prompt hardcoded reflect-then-ask, which read as mechanical."""
    prompt = agent._SYSTEM_PROMPT

    for shape in ("纯陪伴", "澄清", "具体回应", "顺着说下去"):
        assert shape in prompt
    assert "不要每轮都用同一种" in prompt


def test_system_prompt_treats_questions_as_optional(agent):
    prompt = agent._SYSTEM_PROMPT

    assert "多数回合并不需要提问" in prompt


def test_system_prompt_names_the_filler_phrases_to_avoid(agent):
    prompt = agent._SYSTEM_PROMPT

    for filler in ("我在这儿陪着你", "深呼吸"):
        assert filler in prompt


def test_system_prompt_keeps_the_original_clinical_boundaries(agent):
    """Rewriting the tone must not drop the non-diagnostic guarantees."""
    prompt = agent._SYSTEM_PROMPT

    assert "不是医生" in prompt
    assert "MMSE" in prompt
    assert "不要说教" in prompt
    assert "并允许用户纠正" in prompt


def test_context_still_carries_name_history_and_memory(agent):
    agent.tool_gateway.memory_tool.set_persistent_background("喜欢在傍晚散步")
    content = _human_message_text(
        agent,
        profile={"name": "林秀兰"},
        chat_history=[
            {"role": "user", "content": "昨天孙子来看我了。"},
            {"role": "assistant", "content": "听起来是个热闹的下午。"},
        ],
    )

    assert "【称呼】林秀兰" in content
    assert "对方：昨天孙子来看我了。" in content
    assert "助手：听起来是个热闹的下午。" in content
    assert "喜欢在傍晚散步" in content


def test_streaming_reply_emits_one_safe_first_clause_before_full_sentence():
    llm = _StreamingLLM(
        ["听起来你今天真的有些难受，", "我们先慢慢来，不着急。"]
    )
    agent = WellbeingCompanionAgent(llm_factory=lambda: llm)
    emitted = []
    agent.state._stream_sentence_cb = emitted.append

    output = agent._generate([], should_abort=None)

    assert emitted == [
        "听起来你今天真的有些难受，",
        "我们先慢慢来，不着急。",
    ]
    assert output == "听起来你今天真的有些难受，我们先慢慢来，不着急。"


def test_streaming_reply_keeps_short_and_dependent_clauses_together():
    llm = _StreamingLLM(
        ["路洋，虽然你今天很努力，", "但别人没有看到。"]
    )
    agent = WellbeingCompanionAgent(llm_factory=lambda: llm)
    emitted = []
    agent.state._stream_sentence_cb = emitted.append

    output = agent._generate([], should_abort=None)

    assert emitted == ["路洋，虽然你今天很努力，但别人没有看到。"]
    assert output == emitted[0]


def test_streaming_reply_only_uses_soft_break_for_the_first_segment():
    llm = _StreamingLLM(
        ["第一段文字已经足够自然，", "第二段仍然继续，直到句号。"]
    )
    agent = WellbeingCompanionAgent(llm_factory=lambda: llm)
    emitted = []
    agent.state._stream_sentence_cb = emitted.append

    agent._generate([], should_abort=None)

    assert emitted == ["第一段文字已经足够自然，", "第二段仍然继续，直到句号。"]
