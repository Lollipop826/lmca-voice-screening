import asyncio
from unittest.mock import patch

import httpx

from src.voice.health import SoulXHealthProbe


def test_health_requires_a_ready_soulx_model_and_recovers_after_failure():
    async def scenario():
        responses = [None, {"service": "soulx-turn-taking", "ready": False},
                     {"service": "another-service", "ready": True},
                     {"service": "soulx-turn-taking", "ready": True}]

        def respond(request):
            assert request.url == "http://127.0.0.1:8001/health"
            payload = responses.pop(0)
            if payload is None:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=payload)

        client_type = httpx.AsyncClient
        probe = SoulXHealthProbe("ws://127.0.0.1:8001/turn", cache_s=0)
        with patch("src.voice.health.httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(respond), **kwargs,
        )):
            for expected in (False, False, False, True):
                result = await probe.check()
                assert result["available"] is expected
                assert bool(result["error"]) is not expected

    asyncio.run(scenario())


def test_concurrent_health_requests_share_a_probe_without_creating_audio_sessions():
    async def scenario():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"service": "soulx-turn-taking", "ready": True})

        client_type = httpx.AsyncClient
        probe = SoulXHealthProbe("wss://example.test/soulx/turn")
        with patch("src.voice.health.httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(respond), **kwargs,
        )):
            results = await asyncio.gather(probe.check(), probe.check(), probe.check())
        assert all(result["available"] for result in results)
        assert len(requests) == 1
        assert requests[0].method == "GET"
        assert requests[0].url == "https://example.test/soulx/health"

    asyncio.run(scenario())
