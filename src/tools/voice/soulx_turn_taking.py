"""Async client and audio accumulator for SoulX-Duplug turn taking.

The SoulX server accepts float32 mono PCM over ``/turn`` and returns one of
``blank``, ``idle``, ``nonidle`` or ``speak`` plus backward-compatible detail
metadata describing the internal semantic state and decision source.
Transcription and downstream dialogue remain in the 8502 voice server.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import websockets


class SoulXUnavailable(RuntimeError):
    """Raised when the SoulX service cannot serve the current audio chunk."""


@dataclass(frozen=True)
class SoulXTurnState:
    state: str
    text: str = ""
    asr_buffer: str = ""
    asr_segment: str = ""
    raw_state: str = ""
    detail_state: str = ""
    decision_source: str = ""
    wait_idle_count: int = 0
    max_wait_count: int = 0
    monitoring_wait_silence: bool = False
    speech_detected: bool = False
    chunk_rms: float = 0.0
    raw: dict[str, Any] | None = None

    @classmethod
    def from_message(cls, message: str | bytes) -> "SoulXTurnState":
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        payload = json.loads(message)
        state_payload = payload.get("state")
        if payload.get("type") != "turn_state" or not isinstance(state_payload, dict):
            raise ValueError(f"unexpected SoulX response: {payload!r}")
        state = str(state_payload.get("state") or "").strip().lower()
        if state not in {"blank", "idle", "nonidle", "speak"}:
            raise ValueError(f"unknown SoulX state: {state!r}")
        return cls(
            state=state,
            text=str(state_payload.get("text") or ""),
            asr_buffer=str(state_payload.get("asr_buffer") or ""),
            asr_segment=str(state_payload.get("asr_segment") or ""),
            raw_state=str(state_payload.get("raw_state") or ""),
            detail_state=str(state_payload.get("detail_state") or ""),
            decision_source=str(state_payload.get("decision_source") or ""),
            wait_idle_count=int(state_payload.get("wait_idle_count") or 0),
            max_wait_count=int(state_payload.get("max_wait_count") or 0),
            monitoring_wait_silence=bool(
                state_payload.get("monitoring_wait_silence", False)
            ),
            speech_detected=bool(state_payload.get("speech_detected", False)),
            chunk_rms=float(state_payload.get("chunk_rms") or 0.0),
            raw=payload,
        )


class SoulXTurnTakingClient:
    """One persistent SoulX websocket connection per 8502 voice session."""

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000/turn",
        *,
        session_id: str | None = None,
        timeout: float = 3.0,
        retry_interval: float = 5.0,
    ) -> None:
        self.server_url = server_url
        self.session_id = session_id or f"voice8502_{uuid.uuid4().hex}"
        self.timeout = max(0.2, float(timeout))
        self.retry_interval = max(0.2, float(retry_interval))
        self._ws = None
        self._lock = asyncio.Lock()
        self._retry_after = 0.0
        self.last_error = ""

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def retry_ready(self) -> bool:
        return time.monotonic() >= self._retry_after

    async def _close_unlocked(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def reset(self, session_id: str | None = None) -> None:
        async with self._lock:
            await self._close_unlocked()
            self.session_id = session_id or f"voice8502_{uuid.uuid4().hex}"
            self._retry_after = 0.0
            self.last_error = ""

    async def _connect_unlocked(self) -> None:
        if not self.retry_ready:
            raise SoulXUnavailable(self.last_error or "SoulX reconnect cooldown")
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.server_url,
                    open_timeout=self.timeout,
                    close_timeout=1.0,
                    max_size=2 * 1024 * 1024,
                    # SoulX 是本机/局域网服务，绝不走系统 HTTP 代理
                    # （websockets>=14 默认 proxy=True 会读取 HTTP_PROXY 环境变量）
                    proxy=None,
                ),
                timeout=self.timeout,
            )
            self.last_error = ""
        except Exception as exc:
            self._ws = None
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._retry_after = time.monotonic() + self.retry_interval
            raise SoulXUnavailable(self.last_error) from exc

    async def process(self, audio_chunk: np.ndarray) -> SoulXTurnState:
        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return SoulXTurnState(state="blank")

        payload = json.dumps(
            {
                "type": "audio",
                "session_id": self.session_id,
                "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
            },
            ensure_ascii=False,
        )

        # 背压探针：实时预算 = 本次请求携带的音频时长（16 样本/ms@16k）。
        # 主循环串行消费，"抢锁等待 + 往返"一旦超过预算，队列就会单调积压、
        # 实时预测越来越滞后。仅在超阈值时打一行，避免刷屏。
        _budget_ms = max(64.0, audio.size / 16.0)
        _lock_wait_start = time.perf_counter()
        async with self._lock:
            _lock_wait_ms = (time.perf_counter() - _lock_wait_start) * 1000.0
            if self._ws is None:
                await self._connect_unlocked()
            try:
                _rt_start = time.perf_counter()
                await asyncio.wait_for(self._ws.send(payload), timeout=self.timeout)
                response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
                _total_ms = (time.perf_counter() - _lock_wait_start) * 1000.0
                if _total_ms > _budget_ms:
                    _rt_ms = (time.perf_counter() - _rt_start) * 1000.0
                    print(
                        f"[SoulX] ⚠️ 背压: 总{_total_ms:.0f}ms > 预算{_budget_ms:.0f}ms "
                        f"(抢锁{_lock_wait_ms:.0f}ms + 往返{_rt_ms:.0f}ms)"
                    )
                return SoulXTurnState.from_message(response)
            except Exception as exc:
                await self._close_unlocked()
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._retry_after = time.monotonic() + self.retry_interval
                raise SoulXUnavailable(self.last_error) from exc


class SoulXAudioAccumulator:
    """Retain the original PCM while SoulX decides when a turn completes."""

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        pre_roll_seconds: float = 0.48,
        max_utterance_seconds: float = 180.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.pre_roll_samples = max(0, int(self.sample_rate * pre_roll_seconds))
        self.max_utterance_samples = max(
            self.sample_rate,
            int(self.sample_rate * max_utterance_seconds),
        )
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._active_chunks: list[np.ndarray] = []
        self._active_samples = 0

    @property
    def active(self) -> bool:
        return bool(self._active_chunks)

    @property
    def duration_seconds(self) -> float:
        return self._active_samples / float(self.sample_rate)

    def reset(self) -> None:
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._active_chunks = []
        self._active_samples = 0

    def tail_audio(self, seconds: float) -> np.ndarray:
        """Return the most recent ``seconds`` of accumulated audio.

        Used for fast speaker checks at barge-in time: active chunks first
        (they already include the pre-roll captured at speech onset), falling
        back to the rolling pre-roll when no turn is active.
        """
        max_samples = max(0, int(self.sample_rate * seconds))
        if max_samples == 0:
            return np.zeros(0, dtype=np.float32)
        source = self._active_chunks if self._active_chunks else [self._pre_roll]
        collected: list[np.ndarray] = []
        remaining = max_samples
        for chunk in reversed(source):
            if remaining <= 0:
                break
            take = chunk[-remaining:] if chunk.size > remaining else chunk
            collected.append(take)
            remaining -= int(take.size)
        if not collected:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(collected[::-1]).astype(np.float32, copy=False)

    def _append_active(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        self._active_chunks.append(audio.copy())
        self._active_samples += int(audio.size)
        while self._active_samples > self.max_utterance_samples and self._active_chunks:
            overflow = self._active_samples - self.max_utterance_samples
            first = self._active_chunks[0]
            if first.size <= overflow:
                self._active_chunks.pop(0)
                self._active_samples -= int(first.size)
            else:
                self._active_chunks[0] = first[overflow:].copy()
                self._active_samples -= overflow

    def _append_pre_roll(self, audio: np.ndarray) -> None:
        if self.pre_roll_samples <= 0:
            self._pre_roll = np.zeros(0, dtype=np.float32)
            return
        self._pre_roll = np.concatenate([self._pre_roll, audio]).astype(np.float32, copy=False)
        if self._pre_roll.size > self.pre_roll_samples:
            self._pre_roll = self._pre_roll[-self.pre_roll_samples :].copy()

    def feed(self, audio_chunk: np.ndarray, state: str) -> np.ndarray | None:
        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        normalized_state = str(state or "").strip().lower()

        if self.active:
            self._append_active(audio)
        else:
            self._append_pre_roll(audio)
            if normalized_state in {"nonidle", "speak"}:
                self._append_active(self._pre_roll)
                self._pre_roll = np.zeros(0, dtype=np.float32)

        if normalized_state != "speak":
            return None

        if not self.active:
            return None
        utterance = np.concatenate(self._active_chunks).astype(np.float32, copy=False)
        self.reset()
        return utterance
