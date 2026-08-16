from __future__ import annotations

from collections.abc import Callable
from typing import Any
import json

from ..modes import (
    COGNITIVE_SCREENING,
    LEGACY,
    is_cognitive_screening,
    normalize_session_mode,
)

class SessionResumeHandler:
    """Restore one persisted assessment into the connection-owned VoiceSession."""

    MESSAGE_TYPES = {"resume_session"}

    def __init__(
        self,
        connection,
        *,
        session,
        manifest_store,
        get_resume: Callable[[str], dict | None],
        get_session_binding: Callable[[str], dict | None] | None = None,
        assign_session_owner: Callable[..., Any],
        normalize_profile: Callable[[dict | None], dict],
        sync_manual_location: Callable[[dict], Any],
        build_score_payload: Callable[[dict], dict],
        sync_agent_state: Callable[[str, list, dict], Any],
        reset_turn_session: Callable[[str], Any],
        mmse_tool_factory: Callable[[], Any] | None = None,
        authorize_patient: Callable[[str, str], Any] | None = None,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.manifest_store = manifest_store
        self._get_resume = get_resume
        self._get_session_binding = get_session_binding
        self._assign_session_owner = assign_session_owner
        self._normalize_profile = normalize_profile
        self._sync_manual_location = sync_manual_location
        self._build_score_payload = build_score_payload
        self._sync_agent_state = sync_agent_state
        self._reset_turn_session = reset_turn_session
        self._mmse_tool_factory = (
            mmse_tool_factory or self._default_mmse_tool_factory
        )
        self._authorize_patient = authorize_patient
        self._log = logger

    async def handle_message(self, message: dict) -> bool:
        if message.get("type") != "resume_session":
            return False
        await self._resume(
            str(message.get("session_id") or ""),
            replay_history=bool(message.get("replay_history", True)),
        )
        return True

    async def _resume(
        self,
        resume_id: str,
        *,
        replay_history: bool = True,
    ) -> None:
        self._log(f"\n[恢复] 尝试恢复会话: {resume_id}")
        try:
            if self._get_session_binding is not None:
                binding = self._get_session_binding(resume_id)
                if not binding or binding.get("ended_at") is not None:
                    await self.connection.send_json(
                        {
                            "type": "resume_failed",
                            "reason": "会话不存在或已结束",
                        }
                    )
                    return
                patient_id = str(binding.get("patient_id") or "").strip()
                if not patient_id:
                    await self.connection.send_json(
                        {
                            "type": "resume_failed",
                            "reason": "患者访问未授权",
                        }
                    )
                    return
                if self._authorize_patient is not None:
                    try:
                        allowed = self._authorize_patient(patient_id, "voice")
                        if hasattr(allowed, "__await__"):
                            allowed = await allowed
                        if allowed is False:
                            raise PermissionError("PATIENT_ACCESS_DENIED")
                    except Exception as exc:
                        self._log(f"[Auth] 拒绝会话恢复: {type(exc).__name__}")
                        await self.connection.send_json(
                            {
                                "type": "resume_failed",
                                "reason": "患者访问未授权",
                            }
                        )
                        return
            resume_data = self._get_resume(resume_id)
            if not self._is_resumable(resume_data):
                await self.connection.send_json(
                    {
                        "type": "resume_failed",
                        "reason": "会话不存在或已结束",
                    }
                )
                return

            profile = self._normalize_profile(resume_data.get("profile") or {})
            patient_id = str(
                profile.get("patient_id")
                or resume_data.get("patient_id")
                or ""
            ).strip()
            if self._get_session_binding is None and patient_id and self._authorize_patient is not None:
                allowed = self._authorize_patient(patient_id, "voice")
                if hasattr(allowed, "__await__"):
                    allowed = await allowed
                if allowed is False:
                    raise PermissionError("PATIENT_ACCESS_DENIED")
            self.session.bind_session(resume_id)
            resumed_mode = normalize_session_mode(
                resume_data.get("mode"),
                default=LEGACY,
            )
            setter = getattr(self.session.agent, "set_mode", None)
            if callable(setter):
                setter(resumed_mode)
            self.session.mode = resumed_mode
            self._assign_session_owner(
                self.session.session_id,
                self.session.owner_username,
                overwrite=False,
            )
            self.session.patient_profile.clear()
            self.session.patient_profile.update(profile)
            self._sync_manual_location(self.session.patient_profile)
            self.session.chat_history.clear()
            self.session.chat_history.extend(resume_data["chat_history"])
            self.session.accepted_turn_ids.clear()
            self.session.history_message_keys.clear()
            for saved_message in resume_data["messages"]:
                turn_id = str(saved_message.get("turn_id") or "").strip()
                if not turn_id:
                    continue
                role = str(saved_message.get("role") or "")
                content = str(saved_message.get("content") or "")
                self.session.observe_turn_id(turn_id)
                self.session.history_message_keys.add((role, content, turn_id))
                if role == "user":
                    self.session.accepted_turn_ids.add(turn_id)
            self.session.lifecycle.greeting_sent = True
            self.session.lifecycle.started = True
            self.session.lifecycle.awaiting_next_utterance = False
            self.session.lifecycle.sealed = False
            self.manifest_store.refresh()

            if replay_history:
                for saved_message in resume_data["messages"]:
                    await self.connection.send_json(
                        {
                            "type": "message",
                            "role": saved_message["role"],
                            "text": saved_message["content"],
                            "emotion": saved_message.get("emotion"),
                        }
                    )

            detailed_score_payload = None
            if is_cognitive_screening(self.session.mode) and resume_data.get("mmse_scores"):
                detailed_score_payload = self._load_detailed_score_payload(
                    resume_data
                )
                score_payload = (
                    detailed_score_payload
                    or self._build_score_payload(resume_data["mmse_scores"])
                )
                await self.connection.send_json(
                    {
                        "type": "update_score",
                        "mode": self.session.mode,
                        "data": score_payload,
                    }
                )

            if is_cognitive_screening(self.session.mode):
                resume_score_payload = (
                    detailed_score_payload
                    or self._build_score_payload(resume_data.get("mmse_scores") or {})
                )
                self._sync_agent_state(
                    self.session.session_id,
                    self.session.chat_history,
                    resume_score_payload,
                )
            reset_result = self._reset_turn_session("恢复专项会话")
            if hasattr(reset_result, "__await__"):
                await reset_result

            await self.connection.send_json(
                {
                    "type": "session_resumed",
                    "session_id": self.session.session_id,
                    "patient_name": self.session.patient_profile.get("name", ""),
                    "profile": self.session.patient_profile,
                    "mode": self.session.mode,
                    "message_count": len(resume_data["messages"]),
                }
            )
            self._log(
                f"[恢复] ✅ 会话 {self.session.session_id} 已恢复，"
                f"{len(self.session.chat_history)} 条历史"
            )
        except Exception as exc:
            self._log(f"[恢复] ❌ 恢复失败: {type(exc).__name__}")
            await self.connection.send_json(
                {"type": "resume_failed", "reason": str(exc)}
            )

    def _load_detailed_score_payload(
        self,
        resume_data: dict,
    ) -> dict | None:
        try:
            mmse_tool = self.session.agent.tool_gateway.mmse_tool
            if not mmse_tool:
                mmse_tool = self._mmse_tool_factory()
            summary_json = mmse_tool._run(
                session_id=self.session.session_id,
                action="summary",
                dimension_id="orientation",
                score=0,
                education_years=(
                    self.session.patient_profile or {}
                ).get("education_years"),
            )
            summary_data = json.loads(summary_json)
            dimension_scores = (
                (summary_data.get("scoring_details") or {}).get(
                    "dimension_scores"
                )
                or {}
            )
            if summary_data.get("success") and (
                dimension_scores
                or int(summary_data.get("total_score") or 0) > 0
            ):
                return summary_data
            self._log("[恢复] MMSE文件summary为空，使用SQLite评分兜底")
        except Exception as exc:
            self._log(f"[恢复] ⚠️ MMSE细分恢复失败，回退旧payload: {type(exc).__name__}")
        return None

    @staticmethod
    def _is_resumable(resume_data: dict | None) -> bool:
        return bool(
            resume_data
            and not resume_data.get("ended_at")
            and len(resume_data.get("messages", [])) > 0
        )

    @staticmethod
    def _default_mmse_tool_factory():
        from src.tools.agent_tools import MMSEScoringTool

        return MMSEScoringTool()
