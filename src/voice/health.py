"""Live dependency checks kept separate from model warm-up state."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlsplit, urlunsplit

import httpx


class SoulXHealthProbe:
    """Check the model-ready endpoint without feeding audio or creating a turn."""

    def __init__(self, server_url: str, *, timeout_s: float = 1.0, cache_s: float = 2.0):
        url = urlsplit(server_url)
        self.health_url = urlunsplit((
            "https" if url.scheme == "wss" else "http",
            url.netloc,
            url.path.rsplit("/", 1)[0] + "/health",
            "",
            "",
        ))
        self.timeout_s = timeout_s
        self.cache_s = cache_s
        self._cached: dict | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def check(self) -> dict:
        async with self._lock:
            if self._cached is not None and time.monotonic() < self._expires_at:
                return dict(self._cached)
            started = time.monotonic()
            available = False
            error = None
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s, trust_env=False,
                ) as client:
                    response = await client.get(self.health_url)
                    response.raise_for_status()
                    payload = response.json()
                available = (
                    payload.get("service") == "soulx-turn-taking"
                    and payload.get("ready") is True
                )
                if not available:
                    error = "SoulX model is not ready or health protocol does not match"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._cached = {
                "available": available,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "error": error,
            }
            self._expires_at = time.monotonic() + self.cache_s
            return dict(self._cached)
