from __future__ import annotations

import asyncio
import base64
import time
import uuid
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from src.voice_modes import is_cognitive_screening

from .session import VoiceSession


_CURRENT_SESSION_MEMORY_MESSAGES = 6
_CURRENT_SESSION_MESSAGE_CHARS = 500


def _context_block(name: str, content: str) -> str:
    content = str(content or "").strip()
    return f"[{name}]\n{content}" if content else f"[{name}]\n无。"


class TTSPrewarmHandle:
    """A one-shot async TTS prewarm task that can be consumed by nested senders."""

    def __init__(self, task=None) -> None:
        self._task = task

    @classmethod
    def start(cls, tts, *, enabled: bool):
        if not enabled or tts is None:
            return cls()
        prewarm = getattr(tts, "prewarm", None)
        if not callable(prewarm):
            return cls()
        return cls(asyncio.create_task(prewarm()))

    @property
    def pending(self) -> bool:
        return self._task is not None

    async def consume(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        try:
            await task
        except Exception:
            pass


@dataclass
class _TTSLease:
    priority: int
    cancelled: bool = False


class TTSOutputArbiter:
    """Preempt low-priority speech without making higher priority speech wait."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active: _TTSLease | None = None

    async def acquire(self, priority: int) -> _TTSLease | None:
        async with self._lock:
            active = self._active
            if active is not None and priority < active.priority:
                return None
            if active is not None:
                active.cancelled = True
            lease = _TTSLease(int(priority))
            self._active = lease
            return lease

    async def release(self, lease: _TTSLease) -> None:
        async with self._lock:
            if self._active is lease:
                self._active = None

    @staticmethod
    def cancelled(lease: _TTSLease) -> bool:
        return lease.cancelled


class SpeechProcessingCoordinator:
    """Serialize speech turns and replay only the latest queued revision."""

    def __init__(
        self,
        session: VoiceSession,
        *,
        processor,
        enabled_full_duplex: bool,
        is_ai_speaking: Callable[[], bool],
        stop_playback: Callable[[], Any],
        disconnect_error_type: type[Exception],
        client_id: str,
        logger=print,
    ) -> None:
        self.session = session
        self.processor = processor
        self.enabled_full_duplex = bool(enabled_full_duplex)
        self._is_ai_speaking = is_ai_speaking
        self._stop_playback = stop_playback
        self._disconnect_error_type = disconnect_error_type
        self.client_id = client_id
        self._log = logger

    def queue_pending(
        self,
        audio_data: np.ndarray,
        source: str,
        extra_meta: dict | None = None,
    ) -> None:
        processing = self.session.processing
        processing.pending_audio = (
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        processing.pending_source = source
        processing.pending_extra_meta = (
            dict(extra_meta) if isinstance(extra_meta, dict) else None
        )

    def active_audio_copy(self) -> np.ndarray | None:
        active_audio = self.session.processing.active_audio
        if active_audio is None:
            return None
        return (
            np.asarray(active_audio, dtype=np.float32)
            .reshape(-1)
            .copy()
        )

    async def interrupt_active(self, reason: str) -> None:
        processing = self.session.processing
        runtime = self.session.runtime
        if not processing.is_active:
            return
        runtime.stop_generate = True
        processing.generation += 1
        processing.revision_enabled = False
        if self._is_ai_speaking():
            self._log(
                f"[全双工] 🛑 {reason}，停止当前AI播放与处理"
            )
            result = self._stop_playback()
            if hasattr(result, "__await__"):
                await result
        else:
            self._log(
                f"[全双工] 🛑 {reason}，"
                "当前处理结果作废，等待用户补充后的最新版"
            )

    async def cancel_active(self, reason: str) -> None:
        """Cancel work before sealing a session so an old turn cannot cross-write."""
        processing = self.session.processing
        processing.pending_audio = None
        processing.pending_source = ""
        processing.pending_extra_meta = None
        processing.revision_enabled = False
        self.session.runtime.stop_generate = True
        task = processing.task
        if task is not None and not task.done():
            self._log(f"[TURN] ⏹️ {reason}，取消当前语音处理")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        processing.task = None
        processing.is_active = False
        processing.active_audio = None
        processing.active_source = ""

    async def submit_or_queue(
        self,
        audio_data: np.ndarray,
        *,
        source: str,
        extra_meta: dict | None = None,
    ) -> bool:
        processing = self.session.processing
        if processing.task is not None and not processing.task.done():
            self.queue_pending(audio_data, source, extra_meta)
            self._log(
                f"[全双工] 🔁 当前仍在处理旧语音，"
                f"已登记新的{source}待重算"
            )
            return True
        return await self.spawn(
            audio_data,
            source=source,
            allow_revision=self.enabled_full_duplex,
            extra_meta=extra_meta,
        )

    async def spawn(
        self,
        audio_data: np.ndarray,
        *,
        source: str,
        wait_for_previous: bool = False,
        allow_revision: bool = False,
        extra_meta: dict | None = None,
    ) -> bool:
        processing = self.session.processing
        runtime = self.session.runtime
        if processing.task is not None and not processing.task.done():
            if wait_for_previous:
                for _ in range(40):
                    if processing.task is None or processing.task.done():
                        break
                    await asyncio.sleep(0.02)
        if processing.task is not None and not processing.task.done():
            self._log(f"[FLOW] ⚠️ 仍在处理上一段语音，跳过新的{source}")
            if allow_revision:
                self.queue_pending(audio_data, source, extra_meta)
                self._log(
                    f"[全双工] 🔁 已覆盖为最新一版{source}，"
                    "待当前处理退出后自动重算"
                )
            return False

        processing.is_active = True
        runtime.stop_generate = False
        processing.revision_enabled = allow_revision
        processing.active_audio = (
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        processing.active_source = source
        processing.generation += 1
        generation = processing.generation
        turn_id = self.session.next_turn_id()
        processing.active_turn_id = turn_id
        processing.active_playback_id = f"playback-{turn_id}-{generation}"
        self.session.turn_generations[turn_id] = generation
        self._log(
            f"[TURN] ▶️ start turn_id={turn_id} source={source} "
            f"generation={generation} "
            f"audio_s={len(processing.active_audio) / 16000:.2f}"
        )

        processing.task = asyncio.create_task(
            self._run(
                audio_data,
                source=source,
                generation=generation,
                turn_id=turn_id,
                extra_meta=extra_meta,
            )
        )
        return True

    async def _run(
        self,
        audio_data: np.ndarray,
        *,
        source: str,
        generation: int,
        turn_id: str,
        extra_meta: dict | None,
    ) -> None:
        processing = self.session.processing
        try:
            await self.processor.process(
                audio_data,
                generation,
                turn_id,
                extra_meta=extra_meta,
            )
        except self._disconnect_error_type:
            self._log(
                f"[断开] 客户端在{source}处理中断开 {self.client_id}"
            )
        except Exception as exc:
            self._log(f"[错误] 处理{source}时出错: {type(exc).__name__}")
        finally:
            processing.is_active = False
            processing.task = None
            processing.revision_enabled = False
            processing.active_audio = None
            processing.active_source = ""
            processing.active_turn_id = ""
            processing.active_playback_id = ""
            if processing.pending_audio is not None:
                replay_audio = processing.pending_audio
                replay_source = (
                    processing.pending_source or "语音输入(重算)"
                )
                replay_extra_meta = processing.pending_extra_meta
                processing.pending_audio = None
                processing.pending_source = ""
                processing.pending_extra_meta = None
                self._log(
                    f"[全双工] 🔁 启动补充后的重处理: {replay_source}"
                )
                await self.spawn(
                    replay_audio,
                    source=replay_source,
                    allow_revision=self.enabled_full_duplex,
                    extra_meta=replay_extra_meta,
                )


class VoiceTTSStreamer:
    """Stream TTS chunks to the active transport and persist assistant audio."""

    def __init__(
        self,
        connection,
        *,
        session: VoiceSession,
        audio_store,
        tts,
        clean_for_tts: Callable[[str], str],
        sample_rate: int = 24000,
        now_factory: Callable[[], float] = time.time,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self.audio_store = audio_store
        self.tts = tts
        self._clean_for_tts = clean_for_tts
        self.sample_rate = int(sample_rate)
        self._now = now_factory
        self._log = logger
        self._arbiter = TTSOutputArbiter()

    @staticmethod
    def _priority(label: str, companion: bool, priority: int | None) -> int:
        if priority is not None:
            return int(priority)
        if "safety" in str(label or "").casefold():
            return 3
        return 1 if companion else 2

    @staticmethod
    def _empty_result(*, interrupted: bool, reason: str = "") -> dict:
        return {
            "chunks": 0,
            "samples": 0,
            "first_latency": 0.0,
            "interrupted": interrupted,
            "interruption_reason": reason,
            "audio_meta": None,
            "audio_data": None,
            "output_identity": None,
            "started": False,
        }

    def _output_identity(self) -> dict[str, Any]:
        current_identity = getattr(
            self.connection,
            "current_output_identity",
            None,
        )
        if callable(current_identity):
            identity = current_identity()
            if identity:
                return identity
        processing = self.session.processing
        turn_id = str(
            processing.active_turn_id
            or f"system-{self.session.session_id or 'session'}"
        )
        generation = int(processing.generation or 0)
        return {
            "session_id": self.session.session_id,
            "turn_id": turn_id,
            "generation": generation,
            "playback_id": str(
                processing.active_playback_id
                or f"playback-{turn_id}-{generation}"
            ),
        }

    async def stream(
        self,
        tts_text: str,
        content_text: str = "",
        emotion: str = "neutral",
        label: str = "",
        event_type: str = "tts_chunk",
        allow_interrupt: bool = False,
        include_dtype: bool = True,
        persist: bool = True,
        companion: bool = False,
        session_id: str | None = None,
        turn_id: str | None = None,
        generation: int | None = None,
        playback_id: str | None = None,
        priority: int | None = None,
        start_payload: dict[str, Any] | None = None,
        cancel_as_result: bool = False,
    ) -> dict:
        clean_text = self._clean_for_tts(tts_text)
        if not clean_text:
            return self._empty_result(interrupted=False)
        if allow_interrupt and self.session.runtime.stop_generate:
            return self._empty_result(interrupted=True, reason="cancelled")
        lease = await self._arbiter.acquire(
            self._priority(label, companion, priority)
        )
        if lease is None:
            return self._empty_result(interrupted=True, reason="preempted")
        try:
            return await self._stream_unlocked(
                tts_text,
                clean_text=clean_text,
                content_text=content_text,
                emotion=emotion,
                label=label,
                event_type=event_type,
                allow_interrupt=allow_interrupt,
                include_dtype=include_dtype,
                persist=persist,
                companion=companion,
                session_id=session_id,
                turn_id=turn_id,
                generation=generation,
                playback_id=playback_id,
                lease=lease,
                start_payload=start_payload,
                cancel_as_result=cancel_as_result,
            )
        finally:
            await self._arbiter.release(lease)

    async def _stream_unlocked(
        self,
        tts_text: str,
        clean_text: str | None = None,
        content_text: str = "",
        emotion: str = "neutral",
        label: str = "",
        event_type: str = "tts_chunk",
        allow_interrupt: bool = False,
        include_dtype: bool = True,
        persist: bool = True,
        companion: bool = False,
        session_id: str | None = None,
        turn_id: str | None = None,
        generation: int | None = None,
        playback_id: str | None = None,
        lease: _TTSLease | None = None,
        start_payload: dict[str, Any] | None = None,
        cancel_as_result: bool = False,
    ) -> dict:
        clean_text = clean_text if clean_text is not None else self._clean_for_tts(tts_text)
        if not clean_text:
            return self._empty_result(interrupted=False)

        started_at = self._now()
        chunk_count = 0
        total_samples = 0
        first_latency = 0.0
        interrupted = False
        interruption_reason = ""
        first_chunk = True
        collected_chunks = []
        started = start_payload is None
        runtime = self.session.runtime
        if allow_interrupt and runtime.stop_generate:
            return self._empty_result(interrupted=True, reason="cancelled")
        runtime.recent_tts_texts = [
            item
            for item in runtime.recent_tts_texts
            if started_at - item.get("started_at", started_at) <= 10.0
        ]
        runtime.recent_tts_texts.append(
            {"text": clean_text, "started_at": started_at}
        )
        runtime.ai_streaming_tts = True
        runtime.realtime_comfort_playing = companion
        output_identity = self._output_identity()
        if session_id is not None:
            output_identity["session_id"] = str(session_id)
        if turn_id is not None:
            output_identity["turn_id"] = str(turn_id)
        if generation is not None:
            output_identity["generation"] = int(generation)
        if playback_id is not None:
            output_identity["playback_id"] = str(playback_id)
        if companion and playback_id is None:
            output_identity["playback_id"] = (
                f"companion-{output_identity['turn_id']}-"
                f"{uuid.uuid4().hex[:10]}"
            )
        try:
            if lease is not None and self._arbiter.cancelled(lease):
                runtime.ai_streaming_tts = False
                runtime.realtime_comfort_playing = False
                return self._empty_result(interrupted=True, reason="preempted")
            if start_payload is not None:
                start_event = dict(start_payload)
                start_event.update(output_identity)
                if not await self.connection.send_json(start_event):
                    runtime.ai_streaming_tts = False
                    runtime.realtime_comfort_playing = False
                    return self._empty_result(interrupted=True, reason="cancelled")
                started = True
            async for audio_chunk in self.tts.text_to_speech_streaming(
                clean_text,
                emotion=emotion,
            ):
                if lease is not None and self._arbiter.cancelled(lease):
                    self._log(f"[TTS{label}] ⚠️ 被更高优先级输出抢占")
                    interrupted = True
                    interruption_reason = "preempted"
                    break
                if allow_interrupt and runtime.stop_generate:
                    self._log(
                        f"[TTS{label}] ⚠️ 流式生成中检测到打断，停止"
                    )
                    interrupted = True
                    interruption_reason = "cancelled"
                    break
                chunk_array = (
                    np.asarray(audio_chunk, dtype=np.float32)
                    .reshape(-1)
                )
                if chunk_array.size == 0:
                    continue
                collected_chunks.append(chunk_array.copy())
                payload = {
                    "type": event_type,
                    "sample_rate": self.sample_rate,
                    **output_identity,
                }
                chunk_base64 = base64.b64encode(
                    chunk_array.tobytes()
                ).decode("utf-8")
                if event_type == "tts_audio":
                    payload["audio"] = chunk_base64
                    payload["format"] = "pcm"
                else:
                    payload["chunk"] = chunk_base64
                    if include_dtype:
                        payload["dtype"] = "float32"
                if not await self.connection.send_json(payload):
                    interrupted = True
                    interruption_reason = "cancelled"
                    break
                await asyncio.sleep(0)
                chunk_count += 1
                total_samples += len(chunk_array)
                if first_chunk:
                    first_latency = self._now() - started_at
                    self._log(
                        f"[TTS{label}] 🎵 首块! "
                        f"延迟={first_latency:.2f}s"
                    )
                    first_chunk = False
        except asyncio.CancelledError:
            if not cancel_as_result:
                raise
            interrupted = True
            interruption_reason = "cancelled"
        finally:
            if lease is None or not self._arbiter.cancelled(lease):
                runtime.ai_streaming_tts = False
                runtime.realtime_comfort_playing = False

        audio_meta = None
        audio_data = (
            np.concatenate(collected_chunks)
            if collected_chunks
            else None
        )
        if persist and audio_data is not None:
            audio_meta = self.audio_store.persist_assistant(
                audio_data,
                content_text=content_text or tts_text,
            )
        return {
            "chunks": chunk_count,
            "samples": total_samples,
            "first_latency": first_latency,
            "interrupted": interrupted,
            "interruption_reason": interruption_reason,
            "audio_meta": audio_meta,
            "audio_data": audio_data,
            "output_identity": dict(output_identity),
            "started": started,
        }


class AgentResumeStateService:
    """Rebuild score payloads and synchronize restored task state into an Agent."""

    def __init__(self, agent, *, logger=print) -> None:
        self.agent = agent
        self._log = logger

    @staticmethod
    def build_score_payload(resume_scores: dict) -> dict:
        dimension_scores = {}
        total_score = 0
        completed_max_score = 0
        for dimension_id, score_data in (resume_scores or {}).items():
            dimension_score = int(score_data.get("score", 0) or 0)
            dimension_max = int(score_data.get("max_score", 0) or 0)
            total_score += dimension_score
            completed_max_score += dimension_max
            question = score_data.get("question") or ""
            answer = score_data.get("answer") or ""
            detail = score_data.get("evaluation_detail") or ""
            dimension_scores[dimension_id] = {
                "score": dimension_score,
                "max_score": dimension_max,
                "question": question,
                "answer": answer,
                "evaluation_detail": detail,
                "items": (
                    [
                        {
                            "task_id": dimension_id,
                            "base_task_id": dimension_id,
                            "label": dimension_id,
                            "score": dimension_score,
                            "max_score": dimension_max,
                            "question": question,
                            "answer": answer,
                            "evaluation_detail": detail,
                            "detail": detail,
                            "timestamp": (
                                score_data.get("created_at") or ""
                            ),
                            "records_count": 1,
                        }
                    ]
                    if dimension_max or question or answer or detail
                    else []
                ),
            }
        return {
            "success": True,
            "total_score": total_score,
            "total_max_score": 35,
            "completed_max_score": completed_max_score,
            "completed_dimensions": list((resume_scores or {}).keys()),
            "scoring_details": {
                "dimension_scores": dimension_scores,
            },
        }

    def sync(
        self,
        resume_id: str,
        resume_chat_history: list[dict],
        resume_payload: dict | None,
    ) -> None:
        agent = self.agent
        try:
            state = agent.state
            state.session_id = resume_id
            state._active_session_id = resume_id
            state.session_data["session_id"] = resume_id
            state.session_data["chat_history"] = resume_chat_history

            scoring_details = (
                (resume_payload or {}).get("scoring_details") or {}
            )
            dimension_scores = (
                scoring_details.get("dimension_scores") or {}
            )
            if isinstance(state._task_done, set):
                task_config = agent.catalog.task_config
                valid_tasks = set(task_config)
                dimension_to_tasks = {}
                for task_id, config in task_config.items():
                    dimension_id = config.get("dimension_id")
                    if dimension_id:
                        dimension_to_tasks.setdefault(
                            dimension_id,
                            [],
                        ).append(task_id)

                restored_tasks = set(
                    agent.catalog.visual_action_tasks
                )
                for dimension_data in dimension_scores.values():
                    tasks = (
                        dimension_data.get("tasks")
                        if isinstance(dimension_data, dict)
                        else None
                    )
                    items = (
                        dimension_data.get("items")
                        if isinstance(dimension_data, dict)
                        else None
                    )
                    task_ids = []
                    if isinstance(tasks, dict):
                        task_ids.extend(tasks.keys())
                    if isinstance(items, list):
                        task_ids.extend(
                            item.get("base_task_id")
                            or item.get("task_id")
                            for item in items
                            if isinstance(item, dict)
                        )
                    for task_id in task_ids:
                        base_task_id = str(task_id or "").split("::")[0]
                        if base_task_id in valid_tasks:
                            restored_tasks.add(base_task_id)

                for dimension_id, dimension_data in (
                    dimension_scores or {}
                ).items():
                    if not isinstance(dimension_data, dict):
                        continue
                    if (
                        dimension_data.get("tasks")
                        or dimension_data.get("items")
                    ):
                        continue
                    dimension_max = int(
                        dimension_data.get("max_score", 0) or 0
                    )
                    standard_total = sum(
                        int(
                            task_config.get(task_id, {}).get(
                                "max_points",
                                0,
                            )
                            or 0
                        )
                        for task_id in dimension_to_tasks.get(
                            dimension_id,
                            [],
                        )
                    )
                    if (
                        standard_total > 0
                        and dimension_max >= standard_total
                    ):
                        restored_tasks.update(
                            dimension_to_tasks.get(dimension_id, [])
                        )
                state._task_done = restored_tasks

            last_assistant = ""
            for history_item in reversed(resume_chat_history or []):
                if (
                    history_item.get("role") == "assistant"
                    and history_item.get("content")
                ):
                    last_assistant = str(
                        history_item.get("content") or ""
                    )
                    break
            if last_assistant:
                state._last_generated_question = last_assistant

            state._precomputed_next_task = None
            state._last_bridge_hint = None
            state._last_bridge_topic = None
            state._last_target_question = None
            state._last_target_task_id = None
            gateway = agent.tool_gateway
            cancel_prefetch = getattr(
                gateway,
                "cancel_retrieval_prefetch",
                None,
            )
            if callable(cancel_prefetch):
                cancel_prefetch()
            memory_tool = gateway.memory_tool
            if (
                memory_tool is not None
                and hasattr(memory_tool, "get_and_clear_suggestion")
            ):
                try:
                    memory_tool.get_and_clear_suggestion()
                except Exception:
                    pass

            restored_count = len(state._task_done)
            self._log(
                "[恢复] ✅ Agent任务状态已同步: "
                f"active_session={resume_id}, "
                f"restored_tasks={restored_count}"
            )
        except Exception as exc:
            self._log(f"[恢复] ⚠️ 同步Agent恢复状态失败: {type(exc).__name__}")


class PatientProfileService:
    """Normalize patient fields and mirror manually entered location context."""

    _LOCATION_KEYS = (
        "province",
        "city",
        "district",
        "hospital",
        "hospital_name",
        "department",
        "floor",
        "bed_number",
        "place",
    )

    def __init__(self, *, logger=print) -> None:
        self._log = logger

    def normalize(self, raw_profile: dict | None) -> dict:
        profile = dict(raw_profile or {})
        normalized = {
            "name": str(profile.get("name", "") or "").strip(),
            "age": self._safe_int(profile.get("age"), 70),
            "gender": str(
                profile.get("gender") or profile.get("sex") or ""
            ).strip(),
            "education_years": self._safe_int(
                profile.get("education_years"),
                6,
            ),
        }
        for key in self._LOCATION_KEYS:
            value = profile.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                normalized[key] = value
        if (
            not normalized.get("hospital")
            and normalized.get("hospital_name")
        ):
            normalized["hospital"] = normalized["hospital_name"]
        if not normalized.get("place"):
            normalized["place"] = " ".join(
                part
                for part in (
                    normalized.get("hospital"),
                    normalized.get("department"),
                    normalized.get("bed_number"),
                )
                if part
            ).strip()
        patient_id = str(profile.get("patient_id") or "").strip()
        if patient_id:
            normalized["patient_id"] = patient_id
        return {
            key: value
            for key, value in normalized.items()
            if value not in ("", None)
        }

    def sync_location(self, profile: dict | None) -> None:
        if not profile or not any(
            profile.get(key) for key in self._LOCATION_KEYS
        ):
            return
        try:
            from src.utils.location_service import (
                get_location_from_config,
                get_realtime_context,
                save_location_to_config,
            )
            import src.utils.location_service as location_service

            current_location = get_location_from_config() or {}
            manual_location = dict(current_location)
            manual_location.update(
                {
                    "province": profile.get(
                        "province",
                        current_location.get("province", ""),
                    ),
                    "city": profile.get(
                        "city",
                        current_location.get("city", ""),
                    ),
                    "district": profile.get(
                        "district",
                        current_location.get("district", ""),
                    ),
                    "hospital": (
                        profile.get("hospital")
                        or profile.get("hospital_name")
                        or current_location.get("hospital", "")
                    ),
                    "department": profile.get(
                        "department",
                        current_location.get("department", ""),
                    ),
                    "floor": profile.get(
                        "floor",
                        current_location.get("floor", ""),
                    ),
                    "bed_number": profile.get(
                        "bed_number",
                        current_location.get("bed_number", ""),
                    ),
                    "place": profile.get(
                        "place",
                        current_location.get("place", ""),
                    ),
                    "source": "manual-profile",
                }
            )
            if not manual_location.get("place"):
                manual_location["place"] = " ".join(
                    part
                    for part in (
                        manual_location.get("hospital"),
                        manual_location.get("department"),
                        manual_location.get("bed_number"),
                    )
                    if part
                ).strip()
            save_location_to_config(manual_location)
            location_service._cached_location = manual_location
            location_service._cached_weather = None
            location_service._weather_update_time = None
            get_realtime_context(fetch_weather=False)
            self._log("[位置] ✅ 已同步手填位置信息")
        except Exception as exc:
            self._log(f"[位置] ⚠️ 同步手填位置信息失败: {type(exc).__name__}")

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (ValueError, TypeError):
            return default


class VoiceSessionBootstrap:
    """Create a persisted session identity and present its initial waiting state."""

    def __init__(
        self,
        connection,
        *,
        session: VoiceSession,
        create_session: Callable[..., Any],
        now_factory: Callable[[], datetime] = datetime.now,
        token_factory: Callable[[], str] | None = None,
        logger=print,
    ) -> None:
        self.connection = connection
        self.session = session
        self._create_session = create_session
        self._now = now_factory
        self._token_factory = token_factory or (
            lambda: uuid.uuid4().hex[:8]
        )
        self._pending_mode: str | None = None
        self._log = logger

    def create_fresh(self, mode: str | None = None) -> str:
        from .modes import WELLBEING, normalize_session_mode

        target_mode = normalize_session_mode(
            mode if mode is not None else self.session.mode,
            default=WELLBEING,
        )
        self._pending_mode = target_mode
        try:
            session_id = self.session.create_fresh_session(
                self._persist,
                self.generate_session_id,
            )
            self.session.mode = target_mode
            return session_id
        finally:
            self._pending_mode = None

    def _persist(self, candidate_session_id: str) -> None:
        try:
            kwargs = {"owner_username": self.session.owner_username}
            if self._pending_mode and self._pending_mode != "wellbeing":
                kwargs["mode"] = self._pending_mode
            self._create_session(candidate_session_id, **kwargs)
        except Exception as exc:
            self._log(
                f"[DB] ⚠️ 创建会话失败({candidate_session_id}): {exc}"
            )
            raise

    def generate_session_id(self) -> str:
        return (
            f"call_{self._now().strftime('%Y%m%d_%H%M%S')}_"
            f"{self._token_factory()}"
        )

    async def send_waiting_for_info(self) -> None:
        await self.connection.send_json(
            {
                "type": "waiting_for_info",
                "session_id": self.session.session_id,
                "message": (
                    "请先填写患者基本信息和当前聊天位置，"
                    "然后点击「开始聊天」"
                ),
            }
        )


class VoiceConnectionCleanup:
    """Release per-connection work and preserve patient memory on disconnect."""

    def __init__(
        self,
        session: VoiceSession,
        *,
        answer_completion,
        turn_client=None,
        patient_memory_service=None,
        end_session: Callable[..., Any] | None = None,
        realtime_companion=None,
        logger=print,
    ) -> None:
        self.session = session
        self.answer_completion = answer_completion
        self.turn_client = turn_client
        self.patient_memory_service = patient_memory_service
        self._end_session = end_session
        self._realtime_companion = realtime_companion
        self._log = logger

    async def close(self, client_id: str) -> None:
        runtime = self.session.runtime
        processing = self.session.processing
        lifecycle = self.session.lifecycle

        runtime.stop_generate = True
        processing.pending_audio = None
        processing.pending_source = ""
        processing.pending_extra_meta = None
        self.answer_completion.cancel_window()

        if processing.task is not None and not processing.task.done():
            processing.task.cancel()
            await asyncio.gather(
                processing.task,
                return_exceptions=True,
            )

        # Disconnect has no client to notify; seal the local fact before any
        # external transport cleanup can delay or fail.
        if (
            self._end_session is not None
            and lifecycle.current_patient_id
            and not lifecycle.sealed
        ):
            try:
                self._end_session(self.session.session_id, None, None)
                lifecycle.sealed = True
            except Exception as exc:
                self._log(f"[断开] ⚠️ 封存SQLite会话失败: {type(exc).__name__}")

        if self.turn_client is not None:
            await self.turn_client.close()
            self._log(f"[SoulX] 轮次连接已关闭: {client_id}")

        if self._realtime_companion is not None:
            result = self._realtime_companion.close()
            if hasattr(result, "__await__"):
                await result

        if self.patient_memory_service is not None and lifecycle.current_patient_id:
            consolidate = getattr(self.patient_memory_service, "consolidate", None)
            if callable(consolidate):
                try:
                    consolidate(self.session)
                except Exception as exc:
                    self._log(f"[PatientMemory] ⚠️ 断连后台整理投递失败: {type(exc).__name__}")

            flush = getattr(self.patient_memory_service, "flush_session", None)
            if callable(flush):
                try:
                    flush(self.session)
                except Exception as exc:
                    self._log(f"[PatientMemory] ⚠️ 断连刷新投递失败: {type(exc).__name__}")


@dataclass(frozen=True)
class PatientMemoryContext:
    """Patient identity and cross-session memory loaded for one assessment."""

    patient_id: str | None = None
    memory_card: str = ""
    has_history_data: bool = False

    @property
    def has_history(self) -> bool:
        return self.has_history_data


class PatientMemoryService:
    """Own patient resolution, session linking, memory loading and consolidation."""

    def __init__(
        self,
        *,
        get_patient: Callable[[str], dict[str, Any] | None],
        create_patient: Callable[[dict[str, Any]], dict[str, Any]],
        update_patient_profile: Callable[[str, dict[str, Any]], Any],
        link_session_patient: Callable[[str, str], Any],
        assign_patient: Callable[..., Any] | None = None,
        long_term_memory: Any = None,
        long_term_memory_writes: bool = True,
        logger: Callable[[str], Any] = print,
    ) -> None:
        self._get_patient = get_patient
        self._create_patient = create_patient
        self._update_patient_profile = update_patient_profile
        self._link_session_patient = link_session_patient
        self._assign_patient = assign_patient
        self._long_term_memory = long_term_memory
        self._long_term_memory_writes = bool(long_term_memory_writes)
        self._log = logger
        self._consolidation_lock = threading.Lock()
        self._pending_consolidations: set[tuple[str, str]] = set()

    @staticmethod
    def _enabled(session: VoiceSession) -> bool:
        return bool(getattr(session, "long_term_memory_enabled", True))

    def resolve_for_session(
        self,
        session: VoiceSession,
        requested_patient_id: str | None = None,
    ) -> PatientMemoryContext:
        """Resolve/create a patient, link it to the session and load its memory card."""
        started_at = time.perf_counter()
        profile = dict(session.patient_profile or {})
        restored_profile = False
        if not profile and requested_patient_id:
            profile = self._existing_patient_profile(requested_patient_id)
            if profile:
                session.patient_profile.update(profile)
                restored_profile = True
        patient_id = self._resolve_patient_id(
            profile,
            requested_patient_id,
            owner_username=session.owner_username,
            refresh_profile=not restored_profile,
        )
        session.lifecycle.current_patient_id = patient_id
        memory_enabled = self._enabled(session)
        session.memory_engine = self._long_term_memory if memory_enabled else None

        if not patient_id:
            return PatientMemoryContext()

        try:
            self._link_session_patient(session.session_id, patient_id)
        except Exception as exc:
            self._log(f"[PatientMemory] ⚠️ 关联会话-患者失败: {type(exc).__name__}")

        basic_context = self._basic_patient_context(profile)
        fallback_context = ""
        has_history = False
        try:
            if memory_enabled and self._long_term_memory is not None:
                snapshot = self._long_term_memory.get_snapshot(patient_id)
                has_history = any(
                    snapshot.get(key)
                    for key in (
                        "recent_turns",
                        "facts",
                        "preferences",
                        "events",
                        "mmse_history",
                        "comfort_strategies",
                        "narrative",
                    )
                )
                card_getter = getattr(
                    type(self._long_term_memory), "get_authoritative_card", None
                )
                if callable(card_getter):
                    value = self._long_term_memory.get_authoritative_card(patient_id)
                    fallback_context = value if isinstance(value, str) else ""
                else:
                    # Do not attach the legacy narrative while the semantic store is online.
                    available = getattr(self._long_term_memory, "is_memobase_available", None)
                    online = bool(available()) if callable(available) else False
                    if not online:
                        legacy_getter = getattr(self._long_term_memory, "get_context_for_llm", None)
                        if callable(legacy_getter):
                            fallback_context = str(legacy_getter(patient_id) or "")
        except Exception as exc:
            self._log(f"[PatientMemory] ⚠️ 记忆卡拼装失败: {type(exc).__name__}")
        prewarm = (
            getattr(self._long_term_memory, "prewarm_semantic_retrieval", None)
            if memory_enabled else None
        )
        if callable(prewarm):
            try:
                prewarm(patient_id)
            except Exception as exc:
                self._log(f"[Memobase] ⚠️ 语义检索预热投递失败: {type(exc).__name__}")

        memory_card = _context_block(
            "confirmed_profile",
            "\n".join(part for part in (basic_context, fallback_context) if part),
        )
        self._log(
            f"[Latency] session={session.session_id or '-'} "
            f"memory_card_ms={(time.perf_counter() - started_at) * 1000:.1f} "
            f"memory_card_chars={len(memory_card)}"
        )
        return PatientMemoryContext(
            patient_id=patient_id,
            memory_card=memory_card,
            has_history_data=has_history,
        )

    def bind_agent(self, agent: Any, *, enabled: bool = True) -> None:
        """让筛查生命周期可以同步最终MMSE到长程记忆。"""
        lifecycle = getattr(agent, "session_lifecycle", None)
        setter = getattr(lifecycle, "set_memory_manager", None)
        if callable(setter):
            setter(
                self._long_term_memory
                if enabled and self._long_term_memory_writes else None
            )

    def capture_turn(
        self,
        session: VoiceSession,
        user_message: str,
        assistant_message: str,
        emotion: Any = None,
        audio_path: str | None = None,
        turn_id: str | None = None,
        *,
        emotion_source: str | None = None,
        analysis_status: str = "final",
    ) -> bool:
        patient_id = session.lifecycle.current_patient_id
        if (
            not patient_id
            or self._long_term_memory is None
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            return False
        try:
            kwargs = {"turn_id": turn_id} if turn_id else {}
            if emotion_source:
                kwargs["emotion_source"] = emotion_source
            if analysis_status != "final":
                kwargs["analysis_status"] = analysis_status
            self._long_term_memory.capture_turn(
                patient_id,
                user_message,
                assistant_message,
                session_id=session.session_id,
                emotion=emotion,
                audio_path=audio_path,
                **kwargs,
            )
            return True
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 轮次捕获失败: {type(exc).__name__}")
            return False

    def update_turn_emotion(
        self,
        session: VoiceSession,
        turn_id: str,
        emotions: Mapping[str, Any],
        *,
        audio_path: str | None = None,
        emotion_source: str | None = None,
        analysis_status: str = "final",
    ) -> bool:
        patient_id = session.lifecycle.current_patient_id
        updater = getattr(self._long_term_memory, "update_emotion_analysis", None)
        if (
            not patient_id
            or not callable(updater)
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            return False
        try:
            kwargs = {"audio_path": audio_path}
            if emotion_source:
                kwargs["emotion_source"] = emotion_source
            if analysis_status != "final":
                kwargs["analysis_status"] = analysis_status
            updater(
                patient_id,
                turn_id,
                emotions,
                session_id=session.session_id,
                **kwargs,
            )
            return True
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 最终情绪回写失败: {type(exc).__name__}")
            return False

    def update_turn_status(
        self,
        session: VoiceSession,
        turn_id: str,
        *,
        assistant_message: str | None = None,
        response_status: str | None = None,
        turn_state: str | None = None,
    ) -> bool:
        patient_id = session.lifecycle.current_patient_id
        updater = getattr(self._long_term_memory, "update_turn_status", None)
        if (
            not patient_id
            or not callable(updater)
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            return False
        try:
            return bool(
                updater(
                    patient_id,
                    turn_id,
                    session_id=session.session_id,
                    assistant_message=assistant_message,
                    response_status=response_status,
                    turn_state=turn_state,
                )
            )
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ 轮次状态回写失败: {type(exc).__name__}")
            return False

    def get_turn_context(self, session: VoiceSession, current_text: str) -> str:
        patient_id = session.lifecycle.current_patient_id
        current_text = str(current_text or "").strip()
        if (
            not patient_id
            or self._long_term_memory is None
            or not self._enabled(session)
        ):
            return self._format_turn_context(
                session,
                current_text,
                retrieved_long_term_memory="",
                retrieval_source="none",
            )
        context = ""
        retrieval_source = "none"
        try:
            getter = getattr(self._long_term_memory, "get_relevant_evidence", None)
            if callable(getter):
                context = getter(
                    patient_id,
                    current_text,
                    exclude_session_id=session.session_id,
                )
                if not isinstance(context, str):
                    context = ""
                retrieval_source = "memobase" if context else "empty"
            else:
                retrieval_source = "unsupported"
        except Exception as exc:
            self._log(f"[Memobase] ⚠️ 本轮旧事检索失败: {type(exc).__name__}")
            retrieval_source = "error"
        return self._format_turn_context(
            session,
            current_text,
            retrieved_long_term_memory=context,
            retrieval_source=retrieval_source,
        )

    def _format_turn_context(
        self,
        session: VoiceSession,
        current_text: str,
        *,
        retrieved_long_term_memory: str,
        retrieval_source: str,
    ) -> str:
        current_session = self._current_session_memory(session)
        long_term = str(retrieved_long_term_memory or "").strip()
        if not long_term:
            long_term = "[retrieved_long_term_memory]\n无通过阈值的跨会话长期事件。"
        elif not long_term.startswith("[retrieved_long_term_memory]"):
            long_term = _context_block("retrieved_long_term_memory", long_term)
        state_bits = [f"长期记忆检索来源：{retrieval_source}"]
        runtime = getattr(session, "runtime", None)
        if getattr(runtime, "high_risk_detected", False):
            state_bits.append("本轮存在安全风险信号。")
        return "\n\n".join(
            [
                _context_block("current_session_memory", current_session),
                long_term,
                _context_block("current_user_text", current_text),
                _context_block("current_state", "；".join(state_bits)),
            ]
        )

    @staticmethod
    def _current_session_memory(session: VoiceSession) -> str:
        lines = []
        for message in (getattr(session, "chat_history", []) or [])[-_CURRENT_SESSION_MEMORY_MESSAGES:]:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if not role or not content:
                continue
            lines.append(f"{role}: {content[:_CURRENT_SESSION_MESSAGE_CHARS]}")
        return "\n".join(lines)

    def get_cross_session_context(self, session: VoiceSession) -> str:
        return self._format_turn_context(
            session,
            "",
            retrieved_long_term_memory="",
            retrieval_source="disabled",
        )

    def flush_session(self, session: VoiceSession) -> bool:
        patient_id = session.lifecycle.current_patient_id
        if (
            not patient_id
            or self._long_term_memory is None
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            return False
        try:
            flush = getattr(self._long_term_memory, "flush", None)
            if not callable(flush):
                return False
            threading.Thread(
                target=flush,
                args=(patient_id,),
                name="memory-flush",
                daemon=True,
            ).start()
            return True
        except Exception as exc:
            self._log(f"[Memobase] ⚠️ 会话刷新失败: {type(exc).__name__}")
            return False

    def update_mmse_score(
        self,
        session: VoiceSession,
        score: Any,
        weak_dimensions: list[str] | None = None,
    ) -> bool:
        if not is_cognitive_screening(getattr(session, "mode", None)):
            return False
        patient_id = session.lifecycle.current_patient_id
        if (
            not patient_id
            or self._long_term_memory is None
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            return False
        try:
            self._long_term_memory.update_mmse_score(
                patient_id,
                score,
                weak_dimensions or [],
            )
            return True
        except Exception as exc:
            self._log(f"[EmotionMemory] ⚠️ MMSE同步失败: {type(exc).__name__}")
            return False

    def consolidate(
        self,
        session: VoiceSession,
        *,
        total_mmse_score: Any = None,
        cognitive_status: Any = None,
    ) -> bool:
        """Schedule non-blocking consolidation for the current patient session."""
        patient_id = str(session.lifecycle.current_patient_id or "").strip()
        session_id = str(session.session_id or "").strip()
        if not patient_id or not session_id:
            return False

        if (
            self._long_term_memory is None
            or not self._long_term_memory_writes
            or not self._enabled(session)
        ):
            self._log("[EmotionMemory] ⚠️ 新记忆服务未配置，跳过整理")
            return False

        consolidator = getattr(self._long_term_memory, "consolidate_pending_turns", None)
        if not callable(consolidator):
            self._log("[EmotionMemory] ⚠️ 新记忆整理接口不可用")
            return False
        reflector = getattr(type(self._long_term_memory), "reflect_session", None)
        key = (patient_id, session_id)
        with self._consolidation_lock:
            if key in self._pending_consolidations:
                return True
            self._pending_consolidations.add(key)

        def run() -> None:
            try:
                try:
                    consolidator(patient_id, session_id=session_id)
                except Exception as exc:
                    self._log(f"[EmotionMemory] ⚠️ 增量整理失败: {type(exc).__name__}")
                if callable(reflector):
                    try:
                        self._long_term_memory.reflect_session(patient_id, session_id)
                    except Exception as exc:
                        self._log(f"[EmotionMemory] ⚠️ 会话反思失败: {type(exc).__name__}")
            finally:
                with self._consolidation_lock:
                    self._pending_consolidations.discard(key)

        threading.Thread(
            target=run,
            name=f"memory-consolidate-{session_id}",
            daemon=True,
        ).start()
        return True

    def _existing_patient_profile(self, patient_id: str) -> dict[str, Any]:
        try:
            patient = self._get_patient(str(patient_id).strip())
        except Exception as exc:
            self._log(f"[PatientMemory] ⚠️ 查询患者失败: {type(exc).__name__}")
            return {}
        if not patient:
            return {}
        profile = dict(patient.get("extra_profile") or {})
        for key in ("name", "gender", "age", "education_years"):
            value = patient.get(key)
            if value not in (None, ""):
                profile[key] = value
        return profile

    def _resolve_patient_id(
        self,
        profile: dict[str, Any],
        requested_patient_id: str | None,
        *,
        owner_username: str | None = None,
        refresh_profile: bool = True,
    ) -> str | None:
        patient_id = None
        requested_id = str(requested_patient_id or "").strip()
        if requested_id:
            try:
                patient = self._get_patient(requested_id)
            except Exception as exc:
                patient = None
                self._log(f"[PatientMemory] ⚠️ 查询患者失败: {type(exc).__name__}")
            if patient:
                patient_id = str(patient["patient_id"])
                if refresh_profile:
                    try:
                        self._update_patient_profile(patient_id, profile)
                    except Exception as exc:
                        self._log(f"[PatientMemory] ⚠️ 刷新患者档案失败: {type(exc).__name__}")
            else:
                self._log("[PatientMemory] ⚠️ 患者标识不存在，按新患者处理")

        if patient_id is None:
            name = str(profile.get("name") or "").strip()
            if name:
                try:
                    patient_id = str(self._create_patient(profile)["patient_id"])
                    owner_username = str(owner_username or "").strip()
                    if owner_username and self._assign_patient is not None:
                        try:
                            self._assign_patient(
                                patient_id,
                                owner_username,
                                can_read=True,
                                can_write=True,
                                can_voice=True,
                                assigned_by=owner_username,
                            )
                        except Exception as exc:
                            self._log(
                                f"[PatientMemory] ⚠️ 新患者自动授权失败: {type(exc).__name__}"
                            )
                    self._log(f"[PatientMemory] 🆕 自动建档完成: patient_id={patient_id}")
                except Exception as exc:
                    self._log(f"[PatientMemory] ⚠️ 自动建档失败: {type(exc).__name__}")
        return patient_id

    @staticmethod
    def _basic_patient_context(profile: dict[str, Any]) -> str:
        labels = {
            "name": "姓名",
            "age": "年龄",
            "gender": "性别",
            "education_years": "受教育年限",
            "location": "当前位置",
        }
        values = [
            f"{label}：{profile.get(key)}"
            for key, label in labels.items()
            if profile.get(key) not in (None, "")
        ]
        return ("【患者基本信息】" + "；".join(values))[:400] if values else ""
