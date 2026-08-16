from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable


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
你的任务是先准确反映对方刚才表达的感受或事实，再用温和、简短的问题澄清对方此刻更需要什么。
当对方明显难受时，先承接情绪，再询问是否愿意一起梳理或尝试一个很小的步骤；不要说教、不要保证治疗效果。
只有相关且自然时，才引用受控的旧记忆，并使用“你之前提到过……，如果我记错了请告诉我”的可纠正表述。
不要提出MMSE、认知题、分数、教育年限或筛查结论。回复控制在一到三段、总计不超过120字，结尾最多保留一个问题。
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
                    temperature=0.65,
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
        for item in chat_history[-8:]:
            role = "对方" if item.get("role") == "user" else "助手"
            content = str(item.get("content") or "").strip()
            if content:
                recent_lines.append(f"{role}：{content[:500]}")
        name = str(profile.get("name") or "").strip()
        context = [
            "【当前表达】" + text[:4000],
            "【情绪线索】" + str(current_emotion or "neutral"),
        ]
        if name:
            context.append("【称呼】" + name[:40])
        if recent_lines:
            context.append("【当前会话近期内容】\n" + "\n".join(recent_lines))
        if memory.get("summary"):
            context.append("【受控记忆背景】\n" + str(memory["summary"])[:6000])
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
                    match = re.search(r"(.+?[。！？；\n])", pending, re.S)
                    if not match:
                        break
                    sentence = match.group(1).strip()
                    pending = pending[match.end():]
                    if sentence:
                        chunks.append(sentence)
                        callback(sentence)
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
