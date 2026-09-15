from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable


_STREAM_STRONG_BREAKS = frozenset("。！？；\n")
_STREAM_SOFT_BREAKS = frozenset("，,")
_FIRST_SPEAKABLE_CLAUSE_MIN_CHARS = 10
_TRAILING_CONNECTIVES = (
    "因为",
    "所以",
    "如果",
    "虽然",
    "但是",
    "不过",
    "而且",
    "并且",
    "然后",
    "以及",
    "或者",
)
_DEPENDENT_CLAUSE_PAIRS = (
    ("虽然", ("但是", "不过", "却")),
    ("因为", ("所以",)),
    ("如果", ("就", "那么")),
    ("尽管", ("但是", "不过", "仍然", "还是")),
    ("不仅", ("而且", "还", "也")),
)

_EMOTION_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "worried": "anxiety",
    "neutral": "calm",
    "relaxed": "calm",
    "surprise": "confusion",
    "surprised": "confusion",
}

_EMOTION_RESPONSE_GUIDANCE = {
    "joy": (
        "积极、喜悦或释然",
        "开场用“听起来像是松了一口气”这类留有余地的措辞，回应积极感或完成后的释然；"
        "再顺着用户想分享的感受展开。不要断言用户特别兴奋、有成就感，也不要补充成功原因。",
    ),
    "sadness": (
        "低落、难过",
        "语气温和，优先倾听，并直接回应用户本轮提到的事或请求。"
        "不必先复述情绪，不使用固定开场，也不要重复上一轮的安抚话术。"
        "禁止祝贺、把完成自动解释成开心；不要急着鼓励、转正向或连续给建议。",
    ),
    "anger": (
        "生气、不满",
        "开场先确认用户希望边界、选择权或意见被尊重，措辞简洁稳定；不要激化冲突，"
        "不要猜测用户在生谁的气或事情缘由。",
    ),
    "fear": (
        "害怕、紧张",
        "开场先温和承接“有点没底、紧张或不安”，但不猜原因；如用户需要帮助，"
        "只给一个轻量可执行的小步骤，"
        "不要擅自猜测具体担忧或风险。",
    ),
    "anxiety": (
        "焦虑、担心",
        "开场先承接紧绷、担心或没底的感觉，帮助缩小到眼前一步；不要诊断，不要假定焦虑原因。",
    ),
    "confusion": (
        "困惑、迟疑",
        "开场先接住困惑或迟疑，再简短复述已知信息并澄清一个最关键的小点；不要一次抛出多个问题。",
    ),
    "calm": (
        "平静或无明显情绪",
        "开场直接回应用户字面内容，整体保持平实；不额外渲染难过、焦虑或愤怒，也不要强行做心理分析。",
    ),
}


def _normalize_emotion_label(value: Any) -> str:
    label = str(value or "calm").strip().casefold()
    label = _EMOTION_ALIASES.get(label, label)
    return label if label in _EMOTION_RESPONSE_GUIDANCE else "calm"


def _emotion_control_block(value: Any) -> str:
    """Turn a classifier label into bounded, actionable response guidance."""
    label = _normalize_emotion_label(value)
    display, guidance = _EMOTION_RESPONSE_GUIDANCE[label]
    return (
        "【内部语音情绪控制信号·严禁向用户披露】\n"
        "状态：已启用；用于调整回应风格，用户明确的诉求和纠正优先。\n"
        f"识别结果：{display}（{label}）\n"
        f"本轮调节：{guidance}\n"
        "使用边界：它用于调节语气、回应顺序和建议力度，不是新的事实；"
        "不要说出标签，不要推断原因、对象、强度或用户未表达的经历。"
    )


def _is_speakable_first_clause(value: str) -> bool:
    """Return whether a comma-terminated prefix is safe to speak on its own."""
    compact = re.sub(r"[\s，,]+", "", str(value or ""))
    if len(compact) < _FIRST_SPEAKABLE_CLAUSE_MIN_CHARS:
        return False
    if any(compact.endswith(word) for word in _TRAILING_CONNECTIVES):
        return False
    for opener, closers in _DEPENDENT_CLAUSE_PAIRS:
        opener_at = compact.find(opener)
        if opener_at < 0:
            continue
        if not any(compact.find(closer, opener_at + len(opener)) >= 0 for closer in closers):
            return False
    return True


def _stream_segment_end(value: str, *, allow_soft_break: bool) -> int | None:
    """Find the earliest complete boundary in an incrementally generated reply."""
    for index, char in enumerate(value):
        if char in _STREAM_STRONG_BREAKS:
            return index + 1
        if (
            allow_soft_break
            and char in _STREAM_SOFT_BREAKS
            and _is_speakable_first_clause(value[: index + 1])
        ):
            return index + 1
    return None


@dataclass
class WellbeingSessionState:
    session_id: str | None = None
    session_data: dict[str, Any] = field(default_factory=dict)
    _active_session_id: str | None = None
    _stream_sentence_cb: Callable[[str], Any] | None = None
    _task_done: set[str] = field(default_factory=set)


class WellbeingMemoryTool:
    """Small, explicit memory surface for the companion Agent.

    Long-term memory is loaded by ``PatientMemoryService``. This object only
    carries the bounded blocks prepared for the current connection and turn.
    """

    def __init__(self) -> None:
        self._persistent_background = ""
        self._turn_background = ""

    def set_persistent_background(self, value: Any) -> None:
        self._persistent_background = str(value or "")[:4000]

    def set_turn_background(self, value: Any) -> None:
        self._turn_background = str(value or "")[:3000]

    def get_context(
        self,
        conversation_history: Iterable[dict[str, Any]] | None = None,
        _agent_state: Any = None,
    ) -> dict[str, Any]:
        recent = [
            item for item in list(conversation_history or [])[-8:]
            if isinstance(item, dict) and item.get("content")
        ]
        return {
            "summary": "\n".join(
                part
                for part in (self._persistent_background, self._turn_background)
                if part
            )[:6000],
            "recent": recent,
        }

    def reset(self) -> None:
        self._turn_background = ""

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "available": False,
            "status": "idle",
            "current_summary": "",
            "current_topics": [],
            "summary_len": 0,
            "summarized_up_to": 0,
            "last_elapsed": 0,
            "history": [],
        }


class _NoopBackgroundAnalysis:
    async def analyze(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class WellbeingCompanionAgent:
    """Non-diagnostic, support-first Agent used by the default mode."""

    mode = "wellbeing"

    _SYSTEM_PROMPT = """你是心理健康陪伴助手，不是医生，也不做疾病诊断或认知筛查。

【绝对不要做的事】
不要提及、复述或引用任何内部分析结果、字段名、标签、分数或置信度。对方看不到这些，
说出来会让人觉得自己正在被分析。也不要描述你自己的判断过程或选了哪种回应方式。
不要提出MMSE、认知题、分数、教育年限或筛查结论。不要说教，不要保证治疗效果。

【怎么回应】
先判断对方此刻需要什么，再从下面挑一种最贴合的方式。不要每轮都用同一种：
- 纯陪伴：只接住对方说的话，不提问、不建议。对方正在倾诉时用这种。
- 澄清：处境还不清楚时，问一个具体的小问题。
- 具体回应：对方问了实际问题或明确想要办法时，给一个可执行的小步骤。
- 顺着说下去：对方在讲一件具体的事，就接着这件事聊，不要硬拐回情绪。

【如何使用语音情绪】
内部语音情绪控制信号来自说话语气，不是用户说出的对话内容。用户文字没有写明情绪或语义有歧义时，
按“本轮调节”调整语气、回应顺序以及建议力度，不能因为文字没提情绪就忽略它。
直接回应用户本轮的内容和诉求，不要求每轮先说一句情绪描述，不使用固定开场。
它只负责调节回应方式：不要说出标签，不要猜测情绪的原因、对象、强度或未表达的经历。
如果用户明确说出的感受与信号直接冲突，以用户原话为准；不要向用户解释这个冲突。
用户纠正了你的说法或要求不要重复时，立即调整，不再复述被拒绝的话术。

【记忆】
只有“受控记忆背景”明确给出当前轮之前的具体事实时，才能自然引用旧记忆，并允许用户纠正。
没有该背景时，禁止声称用户过去说过、做过或感受过某件事；当前表达在近期对话中的重复不算旧记忆。

【说话方式】
像熟人聊天，不像客服。避免“我在这儿陪着你”“你并不孤单”“要不要深呼吸”这类空泛的安抚套话。
回复一到三段、总计不超过120字。多数回合并不需要提问；确实需要澄清时才问，最多一个问题。
"""

    def __init__(
        self,
        use_local: bool = False,
        *,
        llm_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.use_local = bool(use_local)
        self.state = WellbeingSessionState()
        self.tool_gateway = SimpleNamespace(
            memory_tool=WellbeingMemoryTool(),
            mmse_tool=None,
        )
        self.background_analysis = _NoopBackgroundAnalysis()
        self.session_lifecycle = None
        self.llm = None
        self._llm_factory = llm_factory

    def _get_llm(self):
        if self.llm is None:
            factory = self._llm_factory
            if factory is None:
                from src.llm.http_client_pool import get_chat_openai

                factory = lambda: get_chat_openai(
                    temperature=0.45,
                    max_tokens=240,
                    timeout=15,
                    max_retries=1,
                )
            self.llm = factory()
        return self.llm

    def process_turn(
        self,
        user_input: str,
        dimension: dict[str, Any] | None = None,
        session_id: str | None = None,
        patient_profile: dict[str, Any] | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        current_emotion: str = "neutral",
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        del dimension
        text = str(user_input or "").strip()
        self.state.session_id = session_id
        self.state._active_session_id = session_id
        self.state.session_data["session_id"] = session_id
        if should_abort and should_abort():
            return {"output": "", "superseded": True}

        messages = self._build_messages(
            text,
            patient_profile or {},
            chat_history or [],
            current_emotion,
        )
        output = self._generate(messages, should_abort=should_abort)
        if should_abort and should_abort():
            return {"output": output, "superseded": True}
        return {
            "output": output or "我在听，你可以按自己的节奏继续说。",
            "mode": self.mode,
            "selected_topic": None,
        }

    def _build_messages(
        self,
        text: str,
        profile: dict[str, Any],
        chat_history: list[dict[str, Any]],
        current_emotion: str,
    ) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        memory = self.tool_gateway.memory_tool.get_context()
        recent_lines = []
        recent_items = list(chat_history[-8:])
        if (
            recent_items
            and recent_items[-1].get("role") == "user"
            and str(recent_items[-1].get("content") or "").strip() == text
        ):
            recent_items.pop()
        for item in recent_items:
            role = "对方" if item.get("role") == "user" else "助手"
            content = str(item.get("content") or "").strip()
            if content:
                recent_lines.append(f"{role}：{content[:500]}")
        name = str(profile.get("name") or "").strip()
        context = [
            "【当前表达】" + text[:4000],
            _emotion_control_block(current_emotion),
        ]
        if name:
            context.append("【称呼】" + name[:40])
        if recent_lines:
            context.append("【当前会话近期内容】\n" + "\n".join(recent_lines))
        if memory.get("summary"):
            context.append("【受控记忆背景】\n" + str(memory["summary"])[:6000])
        else:
            context.append(
                "【旧记忆可用性】无。不得声称用户在本轮之前提过、做过或感受过任何事。"
            )
        return [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(context)),
        ]

    def _generate(
        self,
        messages: list[Any],
        *,
        should_abort: Callable[[], bool] | None,
    ) -> str:
        llm = self._get_llm()
        callback = self.state._stream_sentence_cb
        if callable(callback) and callable(getattr(llm, "stream", None)):
            chunks: list[str] = []
            pending = ""
            emitted_any = False
            for chunk in llm.stream(messages):
                if should_abort and should_abort():
                    break
                content = getattr(chunk, "content", chunk)
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text") or part)
                        if isinstance(part, dict)
                        else str(part)
                        for part in content
                    )
                pending += str(content or "")
                while True:
                    segment_end = _stream_segment_end(
                        pending,
                        allow_soft_break=not emitted_any,
                    )
                    if segment_end is None:
                        break
                    segment = pending[:segment_end].strip()
                    pending = pending[segment_end:]
                    if segment:
                        chunks.append(segment)
                        callback(segment)
                        emitted_any = True
            if pending.strip():
                chunks.append(pending.strip())
                callback(pending.strip())
            return "".join(chunks).strip()

        result = llm.invoke(messages)
        content = getattr(result, "content", result)
        output = str(content or "").strip()
        if callable(callback) and output:
            for sentence in re.split(r"(?<=[。！？；])|\n+", output):
                sentence = sentence.strip()
                if sentence:
                    callback(sentence)
        return output
