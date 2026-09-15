from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Hashable


class RetrievalBusyError(RuntimeError):
    pass


class RetrievalUnavailableError(RuntimeError):
    pass


class RetrievalCache:
    """Bound cached results and concurrent loads without queueing new queries."""

    def __init__(
        self,
        *,
        ttl_s: float = 60.0,
        max_entries: int = 128,
        max_inflight: int = 2,
        wait_timeout_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_entries = max(0, int(max_entries))
        self.max_inflight = max(1, int(max_inflight))
        self.wait_timeout_s = max(0.01, float(wait_timeout_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._inflight: dict[Hashable, Future] = {}

    def get(self, key: Hashable, loader: Callable[[], Any]) -> tuple[Any, str]:
        with self._lock:
            now = self._clock()
            expired = [
                cached_key
                for cached_key, (expires_at, _) in self._entries.items()
                if expires_at <= now
            ]
            for cached_key in expired:
                self._entries.pop(cached_key, None)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return copy.deepcopy(entry[1]), "hit"
            pending = self._inflight.get(key)
            owner = pending is None
            if owner:
                if len(self._inflight) >= self.max_inflight:
                    raise RetrievalBusyError("semantic retrieval concurrency limit")
                pending = Future()
                self._inflight[key] = pending

        if not owner:
            return copy.deepcopy(pending.result(timeout=self.wait_timeout_s)), "shared"

        try:
            value = loader()
            stored = copy.deepcopy(value)
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            pending.set_exception(exc)
            raise

        with self._lock:
            if stored and self.ttl_s > 0 and self.max_entries > 0:
                self._entries[key] = (self._clock() + self.ttl_s, stored)
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            self._inflight.pop(key, None)
        pending.set_result(stored)
        return value, "miss"


class RetrievalHttpClient:
    """Apply a read-path deadline while reusing the SDK's connection pool."""

    def __init__(self, client: Any, timeout_s: float) -> None:
        self._client = client
        self._deadline = time.monotonic() + max(0.01, float(timeout_s))

    def _request(self, method: str, url: str, **kwargs: Any):
        import httpx

        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("semantic retrieval HTTP budget exhausted")
        kwargs["timeout"] = httpx.Timeout(
            remaining,
            connect=min(0.25, remaining),
            pool=min(0.1, remaining),
        )
        return getattr(self._client, method)(url, **kwargs)

    def get(self, url: str, **kwargs: Any):
        return self._request("get", url, **kwargs)

    def post(self, url: str, **kwargs: Any):
        return self._request("post", url, **kwargs)
