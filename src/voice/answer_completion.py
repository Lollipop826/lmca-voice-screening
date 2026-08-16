from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np


@dataclass
class AnswerCompletionState:
    """Mutable slow-answer completion state for one voice connection."""

    enabled: bool = False
    segments: list[np.ndarray] = field(default_factory=list)
    last_text: str = ""
    last_label: str = ""
    window_task: asyncio.Task | None = None
    observation_token: int = 0

    @property
    def duration_s(self) -> float:
        if not self.segments:
            return 0.0
        return sum(len(segment) for segment in self.segments) / 16000.0

    def append_segment(self, audio_data: np.ndarray) -> float:
        segment = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if segment.size:
            self.segments.append(segment)
        return self.duration_s

    def rollback_last_segment(self) -> float:
        if self.segments:
            self.segments.pop()
        return self.duration_s

    def audio(self) -> np.ndarray | None:
        if not self.segments:
            return None
        return np.concatenate(self.segments).astype(np.float32, copy=False)

    def has_active_window(self) -> bool:
        return self.window_task is not None and not self.window_task.done()

    def cancel_window(self) -> None:
        if self.window_task is not None and not self.window_task.done():
            self.window_task.cancel()
        self.window_task = None

    def reset_buffer(self) -> None:
        self.cancel_window()
        self.observation_token += 1
        self.segments.clear()
        self.last_text = ""
        self.last_label = ""

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(
            r"[\s，。！？、,.!?；;：“”\"'‘’（）()【】\[\]…·]+",
            "",
            str(text or "").strip().lower(),
        )


class AnswerCompletionController:
    """Judge slow answers, manage observation windows and submit accepted audio."""

    _ACCEPTED_LABELS = {
        "likely_complete",
        "ask_repeat",
        "explicit_no_answer",
    }

    def __init__(
        self,
        connection,
        *,
        session,
        state: AnswerCompletionState,
        processing_coordinator,
        quick_asr: Callable[[np.ndarray], Any],
        judge_answer_completion: Callable[[str, str], Any],
        extract_latest_assistant_utterance: Callable[[list], str],
        observation_window_s: float,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.state = state
        self.processing_coordinator = processing_coordinator
        self._quick_asr = quick_asr
        self._judge_answer_completion = judge_answer_completion
        self._extract_latest_assistant_utterance = (
            extract_latest_assistant_utterance
        )
        self.observation_window_s = float(observation_window_s)
        self._log = logger

    async def notify(self, state_name: str, message: str, **fields) -> None:
        await self.connection.send_json(
            {
                "type": "answer_completion_status",
                "state": state_name,
                "message": message,
                **fields,
            }
        )

    async def maybe_arm_window(self) -> bool:
        state = self.state
        if not state.enabled or self.session.processing.is_active:
            return False

        audio_data = state.audio()
        if audio_data is None or len(audio_data) == 0:
            return False

        had_active_window = state.has_active_window()
        previous_text = state.last_text
        previous_label = state.last_label
        previous_normalized = state.normalize_text(previous_text)
        text = await self._quick_asr(audio_data)
        current_normalized = state.normalize_text(text)

        if (
            had_active_window
            and previous_label in self._ACCEPTED_LABELS
        ):
            if not current_normalized:
                remaining_duration = state.rollback_last_segment()
                self._log(
                    "[作答判定] 观察窗口内新增片段未形成有效文本，"
                    "继续沿用当前观察窗口"
                )
                await self.notify(
                    "observation_window",
                    "观察窗口内未检测到新的有效表达，继续等待患者是否补充。",
                    text=previous_text,
                    label=previous_label,
                    wait_s=self.observation_window_s,
                    duration_s=remaining_duration,
                )
                return True
            if (
                previous_normalized
                and current_normalized == previous_normalized
            ):
                remaining_duration = state.rollback_last_segment()
                self._log(
                    "[作答判定] 观察窗口内新增片段未带来新语义，"
                    f"忽略本段: text_chars={len(previous_text)}"
                )
                await self.notify(
                    "observation_window",
                    "观察窗口内未检测到新的有效补充，继续等待。",
                    text=previous_text,
                    label=previous_label,
                    wait_s=self.observation_window_s,
                    duration_s=remaining_duration,
                )
                return True

        if not text or not text.strip():
            state.cancel_window()
            await self.notify(
                "continue_listening",
                "当前累计语音还不够清晰，系统将继续聆听。",
                label="uncertain",
                duration_s=len(audio_data) / 16000.0,
            )
            return False

        judgement = await self._judge(text)
        state.last_text = text
        state.last_label = judgement["label"]
        if judgement["label"] not in self._ACCEPTED_LABELS:
            state.cancel_window()
            await self.notify(
                "continue_listening",
                "模型判断患者可能还没说完，系统继续聆听。",
                text=text,
                label=judgement["label"],
                confidence=judgement["confidence"],
                duration_s=len(audio_data) / 16000.0,
            )
            return False

        state.cancel_window()
        state.observation_token += 1
        observation_token = state.observation_token
        wait_s = self.observation_window_s
        if wait_s <= 0:
            self._log(
                f"[作答判定] 初判 {judgement['label']} "
                "已满足提交条件，立即提交。"
            )
            await self.notify(
                "auto_committing",
                "模型判定作答已完整，正在立即提交处理。",
                text=text,
                label=judgement["label"],
                confidence=judgement["confidence"],
                duration_s=len(audio_data) / 16000.0,
            )
            await self.request_check(trigger="auto_commit")
            return True

        state.window_task = asyncio.create_task(
            self._delayed_commit(observation_token, wait_s)
        )
        await self.notify(
            "observation_window",
            f"模型初判患者可能已说完，进入 {wait_s:.0f}s "
            "观察窗口；如继续说话会自动延后。",
            text=text,
            label=judgement["label"],
            confidence=judgement["confidence"],
            wait_s=wait_s,
            duration_s=len(audio_data) / 16000.0,
        )
        return True

    async def request_check(self, trigger: str = "manual") -> bool:
        state = self.state
        if not state.enabled:
            await self.notify(
                "disabled",
                "慢答判定模式尚未开启，请先点一次🧠按钮开启。",
                trigger=trigger,
            )
            return False
        if self.session.processing.is_active:
            await self.notify(
                "busy",
                "当前正在处理中，请稍后再做慢答判定。",
                trigger=trigger,
            )
            return False

        state.cancel_window()
        audio_data = state.audio()
        if audio_data is None or len(audio_data) == 0:
            await self.notify(
                "empty",
                "当前还没有可判定的回答，请继续聆听患者作答。",
                trigger=trigger,
            )
            return False

        duration_s = len(audio_data) / 16000.0
        await self.notify(
            "judging",
            "正在判断患者是否已经表达完整，请稍候...",
            trigger=trigger,
            duration_s=duration_s,
        )
        text = await self._quick_asr(audio_data)
        if not text or not text.strip():
            await self.notify(
                "continue_listening",
                "当前累计语音还不够清晰，建议继续聆听后再判定。",
                trigger=trigger,
                label="uncertain",
                duration_s=duration_s,
            )
            return False

        judgement = await self._judge(text)
        state.last_text = text
        state.last_label = judgement["label"]
        if judgement["label"] in self._ACCEPTED_LABELS:
            await self.notify(
                "accepted",
                "慢答判定通过，开始处理当前回答。",
                trigger=trigger,
                text=text,
                label=judgement["label"],
                confidence=judgement["confidence"],
                duration_s=duration_s,
            )
            await self.connection.send_json({"type": "vad_end"})
            state.reset_buffer()
            await self.processing_coordinator.submit_or_queue(
                audio_data,
                source=f"慢答判定:{judgement['label']}",
            )
            return True

        message = (
            "模型判断患者可能还没说完，请继续聆听后再点一次🧠按钮。"
        )
        if judgement["label"] == "uncertain":
            message = (
                "模型暂时无法确认是否说完，请继续聆听后再做一次判定。"
            )
        await self.notify(
            "continue_listening",
            message,
            trigger=trigger,
            text=text,
            label=judgement["label"],
            confidence=judgement["confidence"],
            duration_s=duration_s,
        )
        return False

    async def _judge(self, text: str) -> dict:
        question_text = self._extract_latest_assistant_utterance(
            self.session.chat_history
        )
        result = self._judge_answer_completion(question_text, text)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def _delayed_commit(
        self,
        expected_token: int,
        wait_s: float,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(wait_s)
            if (
                not self.state.enabled
                or expected_token != self.state.observation_token
            ):
                return
            await self.request_check(trigger="observation_window")
        except asyncio.CancelledError:
            return
        finally:
            if self.state.window_task is current_task:
                self.state.window_task = None
