from __future__ import annotations

import asyncio
import gzip
import json
import struct

from src.tools.voice import ark_asr, ws_proxy


def test_extract_text_prefers_cumulative_result_text_over_utterances():
    payload = {
        "result": {
            "text": "第一句。第二句？",
            "utterances": [
                {"text": "第一句。"},
                {"text": "第二句？"},
            ],
        }
    }

    assert ark_asr._extract_text_from_payload(payload) == "第一句。第二句？"


def test_extract_text_falls_back_to_utterances_when_result_text_is_empty():
    payload = {
        "result": {
            "text": "",
            "utterances": [
                {"text": "第一句。"},
                {"text": "第二句？"},
            ],
        }
    }

    assert ark_asr._extract_text_from_payload(payload) == "第一句。第二句？"


def test_timing_log_excludes_transcript(capsys):
    transcript = "sensitive transcript"
    ark_asr._print_timing_summary(
        "success",
        1,
        1,
        {"audio_seconds": 1.0, "chunk_count": 1},
        text=transcript,
    )
    output = capsys.readouterr().out
    assert transcript not in output
    assert f"text_chars={len(transcript)}" in output


def test_streaming_session_accepts_bigmodel_async_and_api_key(monkeypatch):
    class Socket:
        def __init__(self):
            self.sent = []
            self.closed = False

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            await asyncio.Future()

        async def close(self):
            self.closed = True

    async def scenario():
        socket = Socket()
        connected = {}

        async def connect(*args, **kwargs):
            connected["args"] = args
            connected["headers"] = kwargs["additional_headers"]
            return socket

        session = ark_asr.ArkASRStreamingSession(ws_connect=connect)
        await session.start()
        assert struct.unpack(">i", socket.sent[0][4:8])[0] == 1
        config = json.loads(gzip.decompress(socket.sent[0][12:]))
        assert connected["headers"]["X-Api-Key"] == "test-key"
        assert "X-Api-App-Key" not in connected["headers"]
        assert config["request"]["show_utterances"] is True
        await session.aclose()
        assert socket.closed

    monkeypatch.setattr(ark_asr, "_ARK_ASR_MODE", "bigmodel_async")
    monkeypatch.setenv("VOLC_API_KEY", "test-key")
    monkeypatch.delenv("VOLC_APP_ID", raising=False)
    monkeypatch.delenv("VOLC_ACCESS_TOKEN", raising=False)
    asyncio.run(scenario())


def test_streaming_session_ignores_ambient_proxy_variables(monkeypatch):
    """An overseas proxy adds ~1.4s to the handshake, so it must not leak in.

    ``websockets`` 14 made ``proxy=True`` the default, which silently routes
    every connection through ``HTTPS_PROXY`` / ``ALL_PROXY``. The Volcengine
    endpoints are domestic, so inheriting that default is a pure latency loss.
    """
    class Socket:
        async def send(self, data):
            pass

        async def recv(self):
            await asyncio.Future()

        async def close(self):
            pass

    async def scenario() -> dict:
        connected = {}

        async def connect(*args, **kwargs):
            connected.update(kwargs)
            return Socket()

        session = ark_asr.ArkASRStreamingSession(ws_connect=connect)
        await session.start()
        await session.aclose()
        return connected

    monkeypatch.setattr(ark_asr, "_ARK_ASR_MODE", "bigmodel")
    monkeypatch.setenv("VOLC_API_KEY", "test-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:39677")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:39677")
    monkeypatch.delenv("VOLC_WS_PROXY", raising=False)

    connected = asyncio.run(scenario())
    assert connected["proxy"] is None

    monkeypatch.setenv("VOLC_WS_PROXY", "system")
    assert asyncio.run(scenario())["proxy"] is True

    monkeypatch.setenv("VOLC_WS_PROXY", "http://127.0.0.1:7897")
    assert asyncio.run(scenario())["proxy"] == "http://127.0.0.1:7897"


def test_proxy_argument_is_withheld_on_websockets_before_14(monkeypatch):
    """``requirements.txt`` still allows websockets 12/13, which reject it."""
    monkeypatch.setattr(ws_proxy.websockets, "__version__", "13.1")
    assert ws_proxy._websockets_supports_proxy_argument() is False

    monkeypatch.setattr(ws_proxy.websockets, "__version__", "14.0")
    assert ws_proxy._websockets_supports_proxy_argument() is True

    monkeypatch.setattr(ws_proxy.websockets, "__version__", "not-a-version")
    assert ws_proxy._websockets_supports_proxy_argument() is False


def test_streaming_support_matches_the_selected_mode(monkeypatch):
    monkeypatch.setattr(ark_asr, "_ARK_ASR_MODE", "bigmodel_async")
    assert ark_asr.ark_asr_streaming_supported()
    monkeypatch.setattr(ark_asr, "_ARK_ASR_MODE", "bigmodel_nostream")
    assert not ark_asr.ark_asr_streaming_supported()


def test_bigmodel_requests_use_ordered_sequences():
    config = ark_asr._full_client_request(b"{}")
    middle = ark_asr._audio_request(b"middle", sequence=2)
    last = ark_asr._audio_request(b"", sequence=3, last=True)

    assert config[1] == 0x11
    assert struct.unpack(">i", config[4:8])[0] == 1
    assert middle[1] == 0x21
    assert struct.unpack(">i", middle[4:8])[0] == 2
    assert last[1] == 0x23
    assert struct.unpack(">i", last[4:8])[0] == -3
