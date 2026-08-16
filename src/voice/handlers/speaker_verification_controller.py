from __future__ import annotations

from collections.abc import Callable
from typing import Any
import asyncio
import base64
import numpy as np
import os
import soundfile as sf
import tempfile
import uuid

def _default_verifier_factory():
    from src.tools.voice.speaker_verification import get_speaker_verifier

    return get_speaker_verifier()

def _default_speaker_list() -> list[str]:
    from src.tools.voice.speaker_verification import SpeakerVerifier

    return SpeakerVerifier.list_saved_speakers()

class SpeakerVerificationController:
    """Own speaker-verification state and messages for one voice connection."""

    MESSAGE_TYPES = {
        "reset_speaker",
        "toggle_speaker_verify",
        "update_speaker_threshold",
        "save_speaker",
        "list_speakers",
        "load_speaker",
        "enroll_speaker_blob_start",
        "enroll_speaker_blob_chunk",
        "enroll_speaker_blob_end",
        "enroll_speaker_sample",
    }

    def __init__(
        self,
        connection,
        *,
        verifier=None,
        verifier_factory: Callable[[], Any] = _default_verifier_factory,
        list_speakers: Callable[[], list[str]] = _default_speaker_list,
        uploads: dict[str, dict] | None = None,
        logger=print,
    ) -> None:
        self.connection = connection
        self.verifier = verifier
        self._verifier_factory = verifier_factory
        self._list_speakers = list_speakers
        self.uploads = uploads if uploads is not None else {}
        self._log = logger

        self.enabled = False
        self.enrolled = False
        self.warning_sent = False
        self._last_issue = ""

        self._handlers = {
            "reset_speaker": self._handle_reset,
            "toggle_speaker_verify": self._handle_toggle,
            "update_speaker_threshold": self._handle_threshold,
            "save_speaker": self._handle_save,
            "list_speakers": self._handle_list,
            "load_speaker": self._handle_load,
            "enroll_speaker_blob_start": self._handle_blob_start,
            "enroll_speaker_blob_chunk": self._handle_blob_chunk,
            "enroll_speaker_blob_end": self._handle_blob_end,
            "enroll_speaker_sample": self._handle_sample,
        }

    async def handle_message(self, message: dict) -> bool:
        """Handle one speaker-related message and report whether it was consumed."""
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            return False
        await handler(message)
        return True

    async def ensure_ready(self, action: str, *, notify: bool = True) -> bool:
        if self.verifier is not None:
            return True
        try:
            self._log("[声纹] 🚀 按需加载声纹验证器 (ONNX Runtime)...")
            self.verifier = self._verifier_factory()
            self._log("[声纹] ✅ 声纹验证器加载完成")
            self.log_state("config", action=action, status="loaded")
            return True
        except Exception as exc:
            self.enrolled = False
            message = f"声纹模型加载失败：{type(exc).__name__}"
            self._log("[声纹] ❌ 声纹模型加载失败")
            self.log_state(
                "config",
                action=action,
                status="failed",
                error=message,
            )
            if notify:
                await self.connection.send_json(
                    {"type": "speaker_error", "message": message}
                )
            return False

    async def verify(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        *,
        source: str = "unknown",
        turn_id: str = "",
    ) -> tuple[bool, float]:
        audio_seconds = (
            len(audio_data) / float(sample_rate) if sample_rate else 0.0
        )
        if not self.enabled:
            await self._notify_issue("")
            self.log_state(
                "gate",
                source=source,
                decision="allow",
                reason="disabled",
                audio_s=audio_seconds,
            )
            return True, 1.0

        if self.verifier is None:
            message = "声纹验证已开启，但验证器未加载"
            await self._notify_issue(message)
            self.log_state(
                "gate",
                source=source,
                decision="block",
                reason="verifier_unloaded",
                audio_s=audio_seconds,
                error=message,
            )
            return False, float("nan")

        if not self.enrolled or not self.verifier.is_enrolled:
            self.enrolled = False
            message = (
                getattr(self.verifier, "last_error", None)
                or "声纹验证已开启，但当前未加载有效声纹，请重新加载或注册"
            )
            await self._notify_issue(message)
            self.log_state(
                "gate",
                source=source,
                decision="block",
                reason="no_active_enrollment",
                audio_s=audio_seconds,
                error=message,
            )
            return False, float("nan")

        is_target, similarity = await asyncio.to_thread(
            self.verifier.verify_from_audio,
            audio_data,
            sample_rate,
        )
        if not np.isfinite(similarity):
            if not self.verifier.is_enrolled:
                self.enrolled = False
            message = (
                getattr(self.verifier, "last_error", None)
                or "声纹验证不可用，请重新加载或注册声纹"
            )
            await self._notify_issue(message)
            self.log_state(
                "gate",
                source=source,
                decision="block",
                reason="verification_unavailable",
                audio_s=audio_seconds,
                similarity=similarity,
                error=message,
            )
            return False, similarity

        await self._notify_issue("")
        self.log_state(
            "gate",
            source=source,
            decision="allow" if is_target else "block",
            reason="target_match" if is_target else "non_target",
            audio_s=audio_seconds,
            similarity=similarity,
        )
        if not is_target:
            await self.connection.send_json(
                {
                    "type": "voice_input_feedback",
                    "reason": "speaker_non_target",
                    "message": (
                        "当前说话人与已加载声纹不匹配，这句没有处理，"
                        "请让已注册的人再说一遍。"
                    ),
                    "status_text": "未通过声纹验证，请由已注册说话人发言",
                    "similarity": similarity,
                    "source": source,
                    "turn_id": turn_id,
                }
            )
        return is_target, similarity

    async def warn_if_disabled(self) -> bool:
        """Warn once when an enrollment exists but verification is disabled."""
        if (
            self.warning_sent
            or self.enabled
            or self.verifier is None
            or not self.verifier.is_enrolled
        ):
            return False
        self.warning_sent = True
        message = (
            "已加载患者声纹但未开启声纹验证，旁人说话会被当作患者回答；"
            "建议在设置中开启声纹验证。"
        )
        self._log("[声纹] ⚠️ 声纹验证未启用")
        await self.connection.send_json(
            {"type": "speaker_warning", "message": message}
        )
        return True

    def log_diagnostic(self) -> None:
        self._log(
            f"[声纹调试] SPEAKER_VERIFIER={self.verifier is not None}, "
            f"speaker_enrolled={self.enrolled}, "
            f"is_enrolled="
            f"{self.verifier.is_enrolled if self.verifier else 'N/A'}, "
            f"speaker_verify_enabled={self.enabled}"
        )

    def log_state(self, event: str, **fields) -> None:
        payload = {
            "event": event,
            "enabled": self.enabled,
            "loaded": self.verifier is not None,
            "session_enrolled": self.enrolled,
            "verifier_enrolled": (
                self.verifier.is_enrolled if self.verifier else False
            ),
            "speaker_registered": bool(
                getattr(self.verifier, "speaker_name", None)
                if self.verifier
                else False
            ),
            "threshold": (
                getattr(self.verifier, "threshold", None)
                if self.verifier
                else None
            ),
        }
        payload.update(fields)
        parts = []
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, float):
                rendered = "nan" if np.isnan(value) else f"{value:.3f}"
            elif isinstance(value, str):
                rendered = f"text_chars={len(value)}"
            else:
                rendered = value
            parts.append(f"{key}={rendered}")
        self._log("[声纹][State] " + " | ".join(parts))

    async def _notify_issue(self, message: str) -> None:
        if not message:
            self._last_issue = ""
            return
        if message == self._last_issue:
            return
        self._last_issue = message
        self._log(f"[声纹] ⛔ issue_chars={len(message)}")
        await self.connection.send_json(
            {"type": "speaker_error", "message": message}
        )

    async def _handle_reset(self, _message: dict) -> None:
        self.enabled = False
        if not await self.ensure_ready("reset_speaker", notify=False):
            self.enrolled = False
            await self.connection.send_json(
                {
                    "type": "speaker_verify_status",
                    "enabled": False,
                    "message": "声纹验证已关闭",
                }
            )
            await self.connection.send_json(
                {
                    "type": "speaker_error",
                    "message": "声纹模型未就绪，无法重置，请检查服务器模型文件",
                }
            )
            return

        self.verifier.reset()
        self.enrolled = False
        self._log("[声纹] 🔄 用户请求重置声纹")
        self.log_state("config", action="reset")
        await self.connection.send_json(
            {"type": "speaker_reset", "message": "声纹已重置，请重新录入"}
        )
        await self.connection.send_json(
            {
                "type": "speaker_verify_status",
                "enabled": False,
                "message": "声纹验证已关闭",
            }
        )

    async def _handle_toggle(self, message: dict) -> None:
        requested_enabled = bool(message.get("enabled", False))
        self.enabled = requested_enabled
        if requested_enabled:
            if not await self.ensure_ready("toggle_speaker_verify"):
                self.enabled = False
            elif not (
                self.enrolled
                and self.verifier
                and self.verifier.is_enrolled
            ):
                self.enabled = False

        status = "开启" if self.enabled else "关闭"
        self._log(f"[声纹] 🔊 用户{status}了声纹验证")
        self.log_state("config", action="toggle", status=status)
        await self.connection.send_json(
            {
                "type": "speaker_verify_status",
                "enabled": self.enabled,
                "message": f"声纹验证已{status}",
            }
        )
        if requested_enabled and not self.enabled:
            detail = (
                getattr(self.verifier, "last_error", None)
                if self.verifier
                else None
            )
            await self.connection.send_json(
                {
                    "type": "speaker_error",
                    "message": (
                        detail
                        or "没有有效声纹，验证未开启；请先重新录入或加载兼容声纹"
                    ),
                }
            )

    async def _handle_threshold(self, message: dict) -> None:
        new_threshold = message.get("threshold", 0.7)
        if self.verifier:
            self.verifier.threshold = new_threshold
            self._log(f"[声纹] 🎚️ 阈值已更新为: {new_threshold}")
            self.log_state(
                "config",
                action="threshold",
                status="updated",
                threshold=new_threshold,
            )
            await self.connection.send_json(
                {
                    "type": "threshold_updated",
                    "threshold": new_threshold,
                    "message": f"阈值已更新为 {new_threshold:.2f}",
                }
            )

    async def _handle_save(self, message: dict) -> None:
        speaker_name = message.get("name", "")
        if speaker_name and self.verifier and self.verifier.is_enrolled:
            if self.verifier.save(speaker_name):
                await self.connection.send_json(
                    {
                        "type": "speaker_saved",
                        "name": speaker_name,
                        "message": f"✅ 声纹已保存为「{speaker_name}」",
                    }
                )
            else:
                await self.connection.send_json(
                    {"type": "speaker_error", "message": "保存失败，请重试"}
                )
            return
        await self.connection.send_json(
            {"type": "speaker_error", "message": "请先完成声纹注册"}
        )

    async def _handle_list(self, _message: dict) -> None:
        await self.connection.send_json(
            {
                "type": "speakers_list",
                "speakers": self._list_speakers(),
            }
        )

    async def _handle_load(self, message: dict) -> None:
        speaker_name = message.get("name", "")
        if not speaker_name:
            await self.connection.send_json(
                {"type": "speaker_error", "message": "请选择要加载的声纹"}
            )
            return
        if not await self.ensure_ready("load_speaker"):
            return

        if self.verifier.load(speaker_name):
            self.enrolled = True
            self.enabled = True
            self.log_state(
                "config",
                action="load",
                status="success",
                requested_speaker=speaker_name,
            )
            await self.connection.send_json(
                {
                    "type": "speaker_loaded",
                    "name": speaker_name,
                    "message": f"✅ 已加载声纹「{speaker_name}」",
                }
            )
            await self.connection.send_json(
                {
                    "type": "speaker_verify_status",
                    "enabled": True,
                    "message": "已自动开启声纹验证",
                }
            )
            return

        self.enrolled = False
        self.enabled = False
        load_error = (
            getattr(self.verifier, "last_error", None)
            or f"加载失败，找不到「{speaker_name}」的声纹"
        )
        reenroll_message = (
            f"{load_error}。声纹验证已自动关闭，普通对话不会被拦截；"
            "请点击“开始重新录制”采集8句新声纹。"
        )
        self.log_state(
            "config",
            action="load",
            status="failed",
            requested_speaker=speaker_name,
            error=load_error,
        )
        await self.connection.send_json(
            {
                "type": "speaker_verify_status",
                "enabled": False,
                "message": "旧声纹不兼容，验证已自动关闭",
            }
        )
        await self.connection.send_json(
            {"type": "speaker_error", "message": reenroll_message}
        )
        await self.connection.send_json(
            {
                "type": "speaker_reenroll_required",
                "name": speaker_name,
                "message": reenroll_message,
            }
        )

    async def _handle_blob_start(self, message: dict) -> None:
        upload_id = str(message.get("upload_id") or uuid.uuid4().hex[:8])
        self.uploads[upload_id] = {
            "total_chunks": max(
                0,
                self._safe_int(message.get("total_chunks"), 0),
            ),
            "chunks": [],
            "sample_rate": self._safe_int(
                message.get("sample_rate"),
                16000,
            ),
            "restart": bool(message.get("restart")),
        }

    async def _handle_blob_chunk(self, message: dict) -> None:
        upload_state = self.uploads.get(str(message.get("upload_id") or ""))
        if upload_state is None:
            return
        index = max(
            0,
            self._safe_int(
                message.get("index"),
                len(upload_state["chunks"]),
            ),
        )
        chunks = upload_state["chunks"]
        while len(chunks) <= index:
            chunks.append(None)
        chunks[index] = str(message.get("data") or "")

    async def _handle_blob_end(self, message: dict) -> None:
        upload_state = self.uploads.pop(
            str(message.get("upload_id") or ""),
            None,
        )
        if upload_state is None:
            await self.connection.send_json(
                {
                    "type": "speaker_error",
                    "message": "声纹样本分片上传状态丢失，请重试当前句",
                }
            )
            return

        expected = upload_state["total_chunks"] or len(
            upload_state["chunks"]
        )
        chunks = upload_state["chunks"]
        if len(chunks) < expected or any(
            part is None for part in chunks[:expected]
        ):
            await self.connection.send_json(
                {
                    "type": "speaker_error",
                    "message": "声纹样本分片不完整，请重试当前句",
                }
            )
            return

        await self._handle_sample(
            {
                "type": "enroll_speaker_sample",
                "audio": "".join(chunks[:expected]),
                "sample_rate": upload_state["sample_rate"],
                "restart": upload_state["restart"],
            }
        )

    async def _handle_sample(self, message: dict) -> None:
        audio_base64 = message.get("audio")
        if not audio_base64:
            await self.connection.send_json(
                {"type": "speaker_error", "message": "未收到录音样本，请重试"}
            )
            return
        if not await self.ensure_ready("enroll_speaker_sample"):
            return

        temp_path = None
        try:
            if message.get("restart"):
                self.verifier.reset()
                self.enrolled = False
            if self.enabled or message.get("restart"):
                self.enabled = False
                await self.connection.send_json(
                    {
                        "type": "speaker_verify_status",
                        "enabled": False,
                        "message": "采集新声纹期间，验证已关闭",
                    }
                )

            audio_bytes = base64.b64decode(audio_base64)
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            sample_rate = int(message.get("sample_rate") or 16000)
            if sample_rate < 8000 or sample_rate > 192000:
                raise ValueError(f"不支持的采样率: {sample_rate}")

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            )
            temp_path = temp_file.name
            temp_file.close()
            sf.write(temp_path, audio_array, sample_rate)
            success, current, needed = self.verifier.add_sample(temp_path)
        except Exception as exc:
            self.enrolled = False
            self.log_state(
                "config",
                action="enroll",
                status="failed",
                error=str(exc),
            )
            await self.connection.send_json(
                {
                    "type": "speaker_error",
                    "message": f"样本处理失败：{exc}",
                }
            )
            return
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        if success and self.verifier.is_enrolled:
            self.enrolled = True
            self.enabled = True
            self.log_state(
                "config",
                action="enroll",
                status="ready",
                current=current,
                needed=needed,
            )
            await self.connection.send_json(
                {
                    "type": "speaker_enrolled",
                    "message": f"✅ 声纹注册完成 ({current}个样本)",
                    "current": current,
                    "needed": needed,
                }
            )
            await self.connection.send_json(
                {
                    "type": "speaker_verify_status",
                    "enabled": True,
                    "message": "声纹录入完成，已自动开启验证",
                }
            )
            return

        if success:
            self.enrolled = False
            self.log_state(
                "config",
                action="enroll",
                status="collecting",
                current=current,
                needed=needed,
            )
            await self.connection.send_json(
                {
                    "type": "speaker_sample_added",
                    "message": f"已添加 {current}/{needed} 个样本",
                    "current": current,
                    "needed": needed,
                }
            )
            return

        self.enrolled = False
        error = (
            getattr(self.verifier, "last_error", None)
            or "样本添加失败"
        )
        self.log_state(
            "config",
            action="enroll",
            status="failed",
            error=error,
        )
        await self.connection.send_json(
            {
                "type": "speaker_error",
                "message": (
                    getattr(self.verifier, "last_error", None)
                    or "样本添加失败，请重试"
                ),
            }
        )

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (ValueError, TypeError):
            return default
