from __future__ import annotations

import os
from .message_handler_action import MessageHandlerAction
from collections.abc import Callable
from typing import Any
import asyncio
import json
import time

from ..modes import (
    COGNITIVE_SCREENING,
    WELLBEING,
    is_cognitive_screening,
    normalize_session_mode,
)


def _cognitive_enabled() -> bool:
    return os.getenv("ENABLE_COGNITIVE_SCREENING", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _emotion_enabled() -> bool:
    return os.getenv("ENABLE_EMOTION", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _long_term_memory_writes_enabled() -> bool:
    return os.getenv(
        "ENABLE_LONG_TERM_MEMORY_WRITES", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}

class SessionLifecycleHandler:
    """Start and end assessments while keeping all connection-owned state aligned."""

    MESSAGE_TYPES = {"start_session", "prepare_patient", "end_session"}

    def __init__(
        self,
        connection,
        *,
        session,
        answer_completion,
        history_store,
        manifest_store,
        patient_memory_service,
        create_fresh_session: Callable[..., str],
        end_session: Callable[[str, Any, Any], Any],
        update_profile: Callable[[str, dict], Any],
        normalize_profile: Callable[[dict | None], dict],
        sync_manual_location: Callable[[dict], Any],
        reset_turn_session: Callable[[str], Any],
        reset_vad: Callable[[], Any],
        stream_tts_audio: Callable[..., Any],
        clean_for_tts: Callable[[str], str],
        send_waiting_for_info: Callable[[], Any],
        client_id: str,
        cancel_active_processing: Callable[..., Any] | None = None,
        reset_realtime: Callable[..., Any] | None = None,
        authorize_patient: Callable[[str, str], Any] | None = None,
        now_factory: Callable[[], float] = time.time,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.answer_completion = answer_completion
        self.history_store = history_store
        self.manifest_store = manifest_store
        self.patient_memory_service = patient_memory_service
        self._create_fresh_session = create_fresh_session
        self._end_session = end_session
        self._update_profile = update_profile
        self._normalize_profile = normalize_profile
        self._sync_manual_location = sync_manual_location
        self._reset_turn_session = reset_turn_session
        self._reset_vad = reset_vad
        self._stream_tts_audio = stream_tts_audio
        self._clean_for_tts = clean_for_tts
        self._send_waiting_for_info = send_waiting_for_info
        self._client_id = client_id
        self._cancel_active_processing = cancel_active_processing
        self._reset_realtime = reset_realtime
        self._authorize_patient = authorize_patient
        self._now = now_factory
        self._log = logger

    async def handle_message(self, message: dict):
        message_type = message.get("type")
        if message_type == "start_session":
            await self._start(message)
            return True
        if message_type == "prepare_patient":
            return await self._prepare_patient(message)
        if message_type == "end_session":
            return await self._end()
        return False

    async def _prepare_patient(self, message: dict) -> bool:
        """Bind a reconnecting client to an existing patient without reopening it."""
        patient_id = str(message.get("patient_id") or "").strip()
        lifecycle = self.session.lifecycle
        if not patient_id:
            await self.connection.send_json(
                {"type": "resume_failed", "reason": "缺少患者标识"}
            )
            return False
        if lifecycle.started:
            await self.connection.send_json(
                {"type": "resume_failed", "reason": "当前会话仍在进行"}
            )
            return False
        if self._authorize_patient is not None:
            try:
                allowed = self._authorize_patient(patient_id, "voice")
                if hasattr(allowed, "__await__"):
                    allowed = await allowed
                if allowed is False:
                    raise PermissionError("PATIENT_ACCESS_DENIED")
            except Exception as exc:
                self._log(f"[Auth] 拒绝患者绑定: {type(exc).__name__}")
                await self.connection.send_json(
                    {"type": "resume_failed", "reason": "患者访问未授权"}
                )
                return False

        self.session.chat_history.clear()
        self.session.patient_profile.clear()
        self.session.last_message_key = (None, None)
        self.session.accepted_turn_ids.clear()
        self.session.history_message_keys.clear()
        lifecycle.reset()
        loaded = await self._load_patient_memory(
            {"patient_id": patient_id},
            self.session.patient_profile,
        )
        if not loaded:
            lifecycle.reset()
            await self.connection.send_json(
                {"type": "resume_failed", "reason": "患者不存在或无法加载"}
            )
            return False

        lifecycle.awaiting_next_utterance = True
        self.manifest_store.refresh()
        await self.connection.send_json(
            {
                "type": "session_waiting_next_speech",
                "patient_id": lifecycle.current_patient_id,
                "mode": normalize_session_mode(self.session.mode),
            }
        )
        return True

    async def _end(self):
        ended_session_id = self.session.session_id
        total_mmse_score = None
        cognitive_status = None
        if self._cancel_active_processing is not None:
            result = self._cancel_active_processing("会话结束")
            await self._await_if_needed(result)
        await self._play_closing_message()
        try:
            summary_data = {}
            if is_cognitive_screening(self.session.mode):
                mmse_tool = self.session.agent.tool_gateway.mmse_tool
                if mmse_tool is not None:
                    summary_json = mmse_tool._run(
                        session_id=ended_session_id,
                        action="summary",
                        dimension_id="",
                        score=0,
                        education_years=(
                            self.session.patient_profile or {}
                        ).get("education_years"),
                    )
                    summary_data = json.loads(summary_json)
                    if summary_data.get("success"):
                        total_mmse_score = summary_data.get("total_score")
                        cognitive_status = summary_data.get("cognitive_status")
            self._end_session(
                ended_session_id,
                total_mmse_score,
                cognitive_status,
            )
            self.session.lifecycle.sealed = True
            if total_mmse_score is not None:
                weak_dimensions = self._weak_dimensions(summary_data)
                synced = False
                lifecycle = getattr(
                    self.session.agent,
                    "session_lifecycle",
                    None,
                )
                sync_mmse = getattr(lifecycle, "sync_mmse_score", None)
                if callable(sync_mmse):
                    synced = sync_mmse(
                        self.session.lifecycle.current_patient_id,
                        total_mmse_score,
                        weak_dimensions,
                        session_id=ended_session_id,
                    )
                if not synced:
                    updater = getattr(
                        self.patient_memory_service,
                        "update_mmse_score",
                        None,
                    )
                    if callable(updater):
                        updater(
                            self.session,
                            total_mmse_score,
                            weak_dimensions,
                        )
            self._log(
                f"[结束] ✅ 会话已结束: {ended_session_id} "
                f"({total_mmse_score}, {cognitive_status})"
            )
        except Exception as exc:
            self._log(f"[结束] ⚠️ 结束会话失败: {type(exc).__name__}")

        try:
            self.patient_memory_service.consolidate(
                self.session,
                total_mmse_score=total_mmse_score,
                cognitive_status=cognitive_status,
            )
        except Exception as exc:
            self._log(f"[PatientMemory] ⚠️ 触发沉淀失败: {type(exc).__name__}")

        flusher = getattr(self.patient_memory_service, "flush_session", None)
        if callable(flusher):
            try:
                flusher(self.session)
            except Exception as exc:
                self._log(f"[Memobase] ⚠️ 会话刷新投递失败: {type(exc).__name__}")

        self._reset_after_end()
        await self._await_if_needed(self._reset_turn_session("结束评估"))
        if self._reset_realtime is not None:
            await self._await_if_needed(self._reset_realtime())
        await self.connection.send_json(
            {
                "type": "session_waiting_next_speech",
                "patient_id": self.session.lifecycle.current_patient_id,
                "mode": normalize_session_mode(self.session.mode),
            }
        )
        return True

    @staticmethod
    def _weak_dimensions(summary_data: dict) -> list[str]:
        scores = (
            summary_data.get("scoring_details", {})
            .get("dimension_scores", {})
        )
        return [
            str(dimension_id)
            for dimension_id, item in scores.items()
            if int(item.get("score", 0) or 0)
            < int(item.get("max_score", 0) or 0)
        ]

    async def _start(self, message: dict) -> None:
        requested_mode = normalize_session_mode(
            message.get("mode"),
            default=normalize_session_mode(self.session.mode),
        )
        current_mode = normalize_session_mode(self.session.mode)
        if requested_mode != current_mode and self.session.lifecycle.started:
            await self.connection.send_json(
                {
                    "type": "mode_change_rejected",
                    "mode": current_mode,
                    "reason": "会话开始后不能切换模式，请先结束当前会话",
                }
            )
            return
        if requested_mode == COGNITIVE_SCREENING and not _cognitive_enabled():
            await self.connection.send_json(
                {"type": "mode_change_rejected", "reason": "认知筛查专项当前未启用"}
            )
            return

        if self.session.lifecycle.awaiting_next_utterance:
            try:
                self._create_fresh(mode=requested_mode)
                self.session.reset_conversation()
                self._set_agent_mode(requested_mode)
            except Exception as exc:
                await self.connection.send_json(
                    {"type": "resume_failed", "reason": str(exc)}
                )
                return
            self.session.chat_history.clear()
            self.session.lifecycle.awaiting_next_utterance = False
            self.session.lifecycle.sealed = False
        elif requested_mode != current_mode:
            try:
                self._create_fresh(mode=requested_mode)
                self.session.reset_conversation()
                self._set_agent_mode(requested_mode)
            except Exception as exc:
                await self.connection.send_json(
                    {"type": "resume_failed", "reason": str(exc)}
                )
                return
        if bool(message.get("force_new_session")) and (
            self.session.chat_history
            or self.session.lifecycle.greeting_sent
            or self.session.lifecycle.started
        ):
            previous_session_id = self.session.session_id
            self._create_fresh(mode=requested_mode)
            self.session.reset_conversation()
            self._set_agent_mode(requested_mode)
            self.answer_completion.enabled = False
            self.session.runtime.reset_for_new_session()
            self.answer_completion.reset_buffer()
            self._reset_vad()
            await self._await_if_needed(
                self._reset_turn_session("开始新评估")
            )
            self._log(
                f"[开始] 🆕 用户请求新评估，已从会话 "
                f"{previous_session_id} 切换到 {self.session.session_id}"
            )

        profile = self._normalize_profile(message.get("profile", {}))
        self.session.long_term_memory_enabled = (
            message.get("long_term_memory_enabled", True) is not False
        )
        self.session.long_term_memory_writes_enabled = (
            message.get("long_term_memory_writes_enabled", True) is not False
        )
        self.session.emotion_enabled = (
            message.get("emotion_enabled", True) is not False
        )
        self.session.patient_profile.clear()
        self.session.patient_profile.update(profile)
        self._log(f"\n[开始] 收到用户档案 fields={len(profile)}")

        loaded = await self._load_patient_memory(message, profile)
        if not loaded:
            self.session.patient_profile.clear()
            self.session.lifecycle.current_patient_id = None
            self.session.lifecycle.started = False
            self.session.lifecycle.greeting_sent = False
            self.session.lifecycle.awaiting_next_utterance = False
            await self.connection.send_json(
                {"type": "resume_failed", "reason": "患者访问未授权"}
            )
            return

        self.session.lifecycle.started = True
        self.session.lifecycle.sealed = False

        try:
            self._update_profile(self.session.session_id, profile)
        except Exception as exc:
            self._log(f"[DB] ⚠️ 保存患者信息失败: {type(exc).__name__}")

        self._sync_manual_location(profile)
        self.manifest_store.refresh()
        await self.connection.send_json(
            {
                "type": "session_started",
                "session_id": self.session.session_id,
                "profile": profile,
                "mode": normalize_session_mode(self.session.mode),
                "long_term_memory_enabled": self.session.long_term_memory_enabled,
                "long_term_memory_writes_enabled": (
                    self.session.long_term_memory_writes_enabled
                    and _long_term_memory_writes_enabled()
                ),
                "emotion_enabled": (
                    self.session.emotion_enabled and _emotion_enabled()
                ),
            }
        )

        if not self.session.lifecycle.greeting_sent and profile:
            await self._send_greeting(profile)

    async def start_next_on_speech(self) -> bool:
        """Create a fresh session only when the retained patient speaks again."""
        lifecycle = self.session.lifecycle
        if lifecycle.started:
            return True
        if not lifecycle.awaiting_next_utterance:
            return False
        profile = dict(self.session.patient_profile)
        patient_id = lifecycle.current_patient_id
        try:
            if lifecycle.sealed:
                self._create_fresh(mode=WELLBEING)
                self._set_agent_mode(WELLBEING)
            lifecycle.started = True
            lifecycle.greeting_sent = False
            lifecycle.awaiting_next_utterance = False
            lifecycle.sealed = False
            self._update_profile(self.session.session_id, profile)
            await self._load_patient_memory(
                {"patient_id": patient_id},
                profile,
            )
            self._sync_manual_location(profile)
            self.manifest_store.refresh()
            await self.connection.send_json(
                {
                    "type": "session_started",
                    "session_id": self.session.session_id,
                    "profile": profile,
                    "mode": normalize_session_mode(self.session.mode),
                    "auto_started": True,
                    "long_term_memory_enabled": (
                        self.session.long_term_memory_enabled
                    ),
                    "long_term_memory_writes_enabled": (
                        self.session.long_term_memory_writes_enabled
                        and _long_term_memory_writes_enabled()
                    ),
                    "emotion_enabled": (
                        self.session.emotion_enabled and _emotion_enabled()
                    ),
                }
            )
            return True
        except Exception as exc:
            self._log(f"[结束] ❌ 下次开口创建会话失败: {type(exc).__name__}")
            await self.connection.send_json(
                {"type": "resume_failed", "reason": str(exc)}
            )
            return False

    async def _load_patient_memory(
        self,
        message: dict,
        profile: dict,
    ) -> bool:
        try:
            bind_agent = getattr(self.patient_memory_service, "bind_agent", None)
            if callable(bind_agent):
                bind_agent(
                    self.session.agent,
                    enabled=(
                        self.session.long_term_memory_enabled
                        and self.session.long_term_memory_writes_enabled
                    ),
                )
            memory_tool = self.session.agent.tool_gateway.memory_tool
            setter = getattr(memory_tool, "set_persistent_background", None)
            if callable(setter):
                setter("")
            turn_setter = getattr(memory_tool, "set_turn_background", None)
            if callable(turn_setter):
                turn_setter("")
            requested_patient_id = (
                message.get("patient_id") or profile.get("patient_id")
            )
            if requested_patient_id and self._authorize_patient is not None:
                allowed = self._authorize_patient(str(requested_patient_id), "voice")
                if hasattr(allowed, "__await__"):
                    allowed = await allowed
                if allowed is False:
                    raise PermissionError("PATIENT_ACCESS_DENIED")
            patient_context = await asyncio.to_thread(
                self.patient_memory_service.resolve_for_session,
                self.session,
                requested_patient_id,
            )
            if not patient_context.patient_id:
                return False
            if callable(setter):
                setter(patient_context.memory_card)
            await self.connection.send_json(
                {
                    "type": "patient_memory",
                    "patient_id": patient_context.patient_id,
                    "has_history": patient_context.has_history,
                }
            )
            return True
        except Exception as exc:
            self._log(
                f"[PatientMemory] ⚠️ 记忆卡注入失败（不影响筛查）: {exc}"
            )
            return False

    def _set_agent_mode(self, mode: str) -> None:
        normalized = normalize_session_mode(mode)
        setter = getattr(self.session.agent, "set_mode", None)
        if callable(setter):
            setter(normalized)
        self.session.mode = normalized

    def _create_fresh(self, *, mode: str) -> str:
        normalized = normalize_session_mode(mode)
        try:
            session_id = self._create_fresh_session(mode=normalized)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                raise
            session_id = self._create_fresh_session()
        resetter = getattr(self.session.agent, "reset_for_session", None)
        if callable(resetter):
            resetter(normalized)
        self.session.mode = normalized
        return session_id

    async def _send_greeting(self, profile: dict) -> None:
        greeting_name = self._greeting_name(profile)
        if is_cognitive_screening(self.session.mode):
            welcome_message = (
                f"{greeting_name}，您好。接下来是认知筛查专项，"
                "请按自己知道的回答，不确定时直接告诉我就好。"
            )
        else:
            welcome_message = (
                f"{greeting_name}，您好。我会陪您聊一会儿。"
                "您可以从最近的心情或正在困扰您的事情说起，按自己的节奏就好。"
            )
        try:
            await self.connection.send_json(
                {"type": "ai_response", "text": welcome_message}
            )
            self._log("[开场] 🎵 流式语音合成 (TTS) - 情感: neutral...")
            tts_text = self._clean_for_tts(welcome_message)
            tts_result = self._stream_tts_audio(
                tts_text,
                content_text=welcome_message,
                emotion="neutral",
                label="-greeting",
                event_type="tts_chunk",
                allow_interrupt=False,
                include_dtype=False,
                start_payload={"type": "tts_start", "text": tts_text},
            )
            tts_result = await self._await_if_needed(tts_result)
            if tts_result.get("interrupted"):
                self.session.runtime.ai_streaming_tts = False
                self._log(
                    "[开场] ⚠️ 客户端连接已断开，开场语音未完整发送"
                )
                return

            total_samples = tts_result["samples"]
            total_duration = (
                total_samples / 24000.0 if total_samples > 0 else 0.0
            )
            if tts_result.get("started"):
                end_payload = {"type": "tts_end", "duration": total_duration}
                end_payload.update(tts_result.get("output_identity") or {})
                if tts_result.get("interrupted"):
                    end_payload["reason"] = (
                        tts_result.get("interruption_reason") or "cancelled"
                    )
                await self.connection.send_json(end_payload)
            self.session.runtime.ai_speaking_until = (
                self._now() + total_duration
            )
            self.session.runtime.ai_streaming_tts = False
            self._log(
                f"[开场] ✅ 个性化开场问候已发送给{greeting_name}，"
                f"AI将说话 {total_duration:.1f}s"
            )
            self.session.lifecycle.greeting_sent = True
            self.session.lifecycle.started = True
            self.session.chat_history.append(
                {"role": "assistant", "content": welcome_message}
            )
            await self.history_store.append("assistant", welcome_message)
            self._log(
                f"[开场] 📝 开场白已写入 chat_history "
                f"(当前历史长度: {len(self.session.chat_history)})"
            )
        except Exception as exc:
            self._log(f"[错误] 发送开场问候失败: {type(exc).__name__}")

    async def _play_closing_message(self) -> None:
        message = "我们这次就先聊到这里"
        try:
            await self.connection.send_json(
                {"type": "ai_response", "text": message, "closing": True}
            )
            result = await self._await_if_needed(
                self._stream_tts_audio(
                    self._clean_for_tts(message),
                    content_text=message,
                    emotion="gentle",
                    label="-session-end",
                    event_type="tts_chunk",
                    allow_interrupt=False,
                    include_dtype=True,
                    start_payload={
                        "type": "tts_start",
                        "text": message,
                        "closing": True,
                    },
                )
            )
            duration = result["samples"] / 24000.0 if result["samples"] else 0.0
            if result.get("started"):
                end_payload = {
                    "type": "tts_end",
                    "duration": duration,
                    "closing": True,
                }
                end_payload.update(result.get("output_identity") or {})
                if result.get("interrupted"):
                    end_payload["reason"] = (
                        result.get("interruption_reason") or "cancelled"
                    )
                await self.connection.send_json(end_payload)
        except Exception as exc:
            self._log(f"[结束] ⚠️ 结束语播报失败: {type(exc).__name__}")

    def _reset_after_end(self) -> None:
        self.session.chat_history.clear()
        self.session.last_message_key = (None, None)
        self.session.accepted_turn_ids.clear()
        self.session.history_message_keys.clear()
        self.session.lifecycle.started = False
        self.session.lifecycle.greeting_sent = False
        self._set_agent_mode(WELLBEING)
        self.session.lifecycle.awaiting_next_utterance = bool(
            self.session.lifecycle.current_patient_id
            and self.session.patient_profile
        )
        self.answer_completion.enabled = False
        self.session.processing.reset_session_buffers()
        self.session.runtime.reset_after_end_session()
        self.answer_completion.reset_buffer()
        self._reset_vad()

    @staticmethod
    def _greeting_name(profile: dict) -> str:
        name = profile.get("name", "")
        gender = profile.get("gender", "")
        if not name:
            return "您"
        if gender == "女":
            return f"{name}女士"
        if gender == "男":
            return f"{name}先生"
        return name

    @staticmethod
    async def _await_if_needed(result):
        if hasattr(result, "__await__"):
            return await result
        return result
