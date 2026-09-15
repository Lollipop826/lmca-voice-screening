"""Regression tests for ArkTTS connection reuse after an interrupted session.

Background: consumers stop playback with ``break`` inside ``async for``, which
leaves the async generator suspended at ``yield``. The interrupted session's
trailing audio frames and its SessionFinished frame stay unread on the socket.
Reusing that connection made the next StartSession read those stale frames,
raising RuntimeError and forcing a ~1.6s reconnect on the critical path.

A bare ``break`` does not run the generator's ``finally`` -- that only happens
when the generator is finalized. ``VoiceTTSStreamer`` therefore calls
``aclose()`` explicitly, which is what these tests exercise.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

os.environ.setdefault("VOLC_APP_ID", "test-app-id")
os.environ.setdefault("VOLC_ACCESS_TOKEN", "test-token")

from src.tools.voice import ark_tts
from src.tools.voice.ark_tts import ArkTTS


def _server_audio_frame(pcm_bytes: bytes) -> bytes:
    """Build an audio-only server frame (msg_type 0xB0)."""
    session_id = b"session"
    return (
        bytes([0x11, 0xB0, 0x10, 0x00])
        + struct.pack(">i", ark_tts._EVT_TTS_RESPONSE)
        + struct.pack(">I", len(session_id))
        + session_id
        + struct.pack(">I", len(pcm_bytes))
        + pcm_bytes
    )


def _server_json_frame(event: int, payload: dict) -> bytes:
    """Build a full-server JSON frame (msg_type 0x90)."""
    session_id = b"session"
    raw = json.dumps(payload).encode()
    return (
        bytes([0x11, 0x90, 0x10, 0x00])
        + struct.pack(">i", event)
        + struct.pack(">I", len(session_id))
        + session_id
        + struct.pack(">I", len(raw))
        + raw
    )


def _session_frames(audio_frame_count: int, *, finished: bool = True) -> list[bytes]:
    """One scripted session: SessionStarted, some audio, then SessionFinished."""
    pcm = b"\x00\x01" * 2400
    frames = [_server_json_frame(ark_tts._EVT_SESSION_STARTED, {})]
    frames.extend(_server_audio_frame(pcm) for _ in range(audio_frame_count))
    if finished:
        frames.append(_server_json_frame(ark_tts._EVT_SESSION_FINISHED, {}))
    return frames


class FakeWebSocket:
    """Minimal stand-in that replays a scripted sequence of server frames."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self.sent: list[bytes] = []
        self.closed = False
        self.state = ark_tts.State.OPEN

    async def send(self, data: bytes) -> None:
        self.sent.append(data)

    async def recv(self) -> bytes:
        if not self._frames:
            # Mimic a server that has gone quiet rather than closing, so the
            # drain loop has to rely on its own deadline.
            await asyncio.sleep(3600)
        return self._frames.pop(0)

    async def ping(self):
        if self.closed:
            raise ConnectionError("socket closed")
        pong_waiter = asyncio.get_running_loop().create_future()
        pong_waiter.set_result(0.001)
        return pong_waiter

    async def close(self) -> None:
        self.closed = True
        self.state = ark_tts.State.CLOSED

    @property
    def remaining_frames(self) -> int:
        return len(self._frames)


async def _consume_one_chunk_then_close(tts: ArkTTS, text: str) -> None:
    """Reproduce how VoiceTTSStreamer aborts playback: break, then aclose."""
    stream = tts.text_to_speech_streaming(text)
    async for _chunk in stream:
        break
    await stream.aclose()


def _client_event_and_payload(frame: bytes) -> tuple[int, dict]:
    """Decode an outbound session frame for protocol-shape assertions."""
    event = struct.unpack(">i", frame[4:8])[0]
    sid_size = struct.unpack(">I", frame[8:12])[0]
    payload_size_pos = 12 + sid_size
    payload_size = struct.unpack(
        ">I", frame[payload_size_pos:payload_size_pos + 4]
    )[0]
    payload_start = payload_size_pos + 4
    payload = frame[payload_start:payload_start + payload_size]
    return event, json.loads(payload or b"{}")


def test_interrupted_session_drains_tail_and_keeps_connection():
    """Aborting playback must leave the connection clean and reusable."""

    async def scenario():
        tts = ArkTTS()
        socket = FakeWebSocket(_session_frames(audio_frame_count=3))
        tts._ws = socket

        await _consume_one_chunk_then_close(tts, "你好，今天怎么样？")

        assert tts._ws is socket, "干净排空后应保留连接复用，而不是丢弃重建"
        assert socket.remaining_frames == 0, "会话残留帧应已被读空"
        assert not socket.closed, "排空成功不应关闭连接"
        assert any(
            _client_event_and_payload(frame)[0] == ark_tts._EVT_CANCEL_SESSION
            for frame in socket.sent
        ), "打断时应主动发送 CancelSession，停止服务端继续合成"

    asyncio.run(scenario())


def test_synthesis_after_interrupt_does_not_reconnect():
    """The synthesis following an interruption must not rebuild the socket."""

    async def scenario():
        tts = ArkTTS()
        socket = FakeWebSocket(
            _session_frames(audio_frame_count=2)
            + _session_frames(audio_frame_count=1)
        )
        tts._ws = socket

        reconnects = 0

        async def _fail_on_connect(*args, **kwargs):
            nonlocal reconnects
            reconnects += 1
            raise AssertionError("打断后不应重连（这正是 1.6s 延迟的来源）")

        original_connect = ark_tts.websockets.connect
        ark_tts.websockets.connect = _fail_on_connect
        try:
            await _consume_one_chunk_then_close(tts, "第一句被打断")

            chunks = [
                chunk
                async for chunk in tts.text_to_speech_streaming("第二句正常播完")
            ]
        finally:
            ark_tts.websockets.connect = original_connect

        assert chunks, "打断后的下一次合成应正常产出音频"
        assert reconnects == 0
        assert tts._ws is socket, "整个过程应复用同一条连接"

    asyncio.run(scenario())


def test_stalled_drain_quarantines_connection_off_critical_path():
    """A server still producing audio must not block the caller on drain."""

    async def scenario():
        tts = ArkTTS()
        # SessionFinished never arrives, so the drain loop hits its deadline.
        socket = FakeWebSocket(
            _session_frames(audio_frame_count=2, finished=False)
        )
        tts._ws = socket

        quarantined: list[object] = []
        original_schedule = ArkTTS._schedule_reconnect
        ArkTTS._schedule_reconnect = (
            lambda self, stale: quarantined.append(stale)
        )
        original_deadline = ark_tts._DRAIN_DEADLINE_S
        ark_tts._DRAIN_DEADLINE_S = 0.05

        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await _consume_one_chunk_then_close(tts, "排空超时的情况")
        finally:
            ArkTTS._schedule_reconnect = original_schedule
            ark_tts._DRAIN_DEADLINE_S = original_deadline
        elapsed = loop.time() - started

        assert tts._ws is None, "排空超时的连接必须摘除，不能留给下一次复用"
        assert quarantined == [socket], "废弃连接应交给后台重建"
        assert elapsed < 1.0, f"排空不应拖住调用方，实际 {elapsed:.2f}s"

    asyncio.run(scenario())


def test_connection_health_check_waits_for_pong():
    """A successful ping send alone must not mark the connection healthy."""

    class DeferredPongWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__([])
            self.pong_waiter = None

        async def ping(self):
            self.pong_waiter = asyncio.get_running_loop().create_future()
            return self.pong_waiter

    async def scenario():
        tts = ArkTTS()
        socket = DeferredPongWebSocket()
        tts._ws = socket

        health_check = asyncio.create_task(tts._ensure_connected())
        await asyncio.sleep(0)

        assert socket.pong_waiter is not None
        assert not health_check.done(), "收到 Pong 前不应认为连接健康"

        socket.pong_waiter.set_result(0.004)
        await health_check
        assert tts._ws is socket

    asyncio.run(scenario())


def test_recovery_connects_without_session_lock_and_is_reused():
    """Background recovery must progress while the old session owns the lock."""

    async def scenario():
        tts = ArkTTS()
        stale = FakeWebSocket([])
        replacement = FakeWebSocket([])
        tts._ws = None
        connect_started = asyncio.Event()
        allow_connect = asyncio.Event()
        connect_calls = 0

        async def _open_replacement():
            nonlocal connect_calls
            connect_calls += 1
            connect_started.set()
            await allow_connect.wait()
            return replacement

        tts._open_connection = _open_replacement

        async with tts._ws_lock:
            recovery = tts._schedule_reconnect(stale)
            await asyncio.wait_for(connect_started.wait(), timeout=0.2)
            ensure_connected = asyncio.create_task(tts._ensure_connected())
            allow_connect.set()
            await asyncio.wait_for(ensure_connected, timeout=0.2)

        await recovery
        assert connect_calls == 1, "下一请求应复用恢复任务，不能重复建连"
        assert tts._ws is replacement
        assert stale.closed

    asyncio.run(scenario())


def test_text_is_sent_only_in_task_request():
    """StartSession config must not duplicate TaskRequest synthesis text."""

    async def scenario():
        tts = ArkTTS()
        socket = FakeWebSocket(_session_frames(audio_frame_count=1))
        tts._ws = socket
        text = "这段文字只发送一次"

        chunks = [
            chunk async for chunk in tts.text_to_speech_streaming(text)
        ]
        assert chunks

        decoded = [_client_event_and_payload(frame) for frame in socket.sent]
        start_payload = next(
            payload for event, payload in decoded
            if event == ark_tts._EVT_START_SESSION
        )
        task_payload = next(
            payload for event, payload in decoded
            if event == ark_tts._EVT_TASK_REQUEST
        )

        assert "text" not in start_payload["req_params"]
        assert task_payload == {"req_params": {"text": text}}

    asyncio.run(scenario())


def test_completed_session_does_not_trigger_drain():
    """A fully consumed session is already clean; drain must not run."""

    async def scenario():
        tts = ArkTTS()
        socket = FakeWebSocket(_session_frames(audio_frame_count=2))
        tts._ws = socket

        drain_calls = 0
        original_drain = ArkTTS._discard_session_tail

        async def _counting_drain(self):
            nonlocal drain_calls
            drain_calls += 1
            await original_drain(self)

        ArkTTS._discard_session_tail = _counting_drain
        try:
            chunks = [
                chunk async for chunk in tts.text_to_speech_streaming("完整播完")
            ]
        finally:
            ArkTTS._discard_session_tail = original_drain

        assert chunks, "正常播放应产出音频"
        assert drain_calls == 0, "正常读到 SessionFinished 时不应再排空"
        assert tts._ws is socket

    asyncio.run(scenario())
