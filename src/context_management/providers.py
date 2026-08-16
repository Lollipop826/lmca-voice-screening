"""Pluggable context providers for isolated context-management experiments.

``baseline`` keeps using the application's existing ``ConversationMemoryTool``.
Every other configured variant is an HTTP service that receives only the
conversation/state contract and returns a context bundle.  The adapter mirrors
the methods the screening agent already expects, so no private agent prompt or
implementation has to be shared with the context-service author.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse


_VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_BASELINE_VARIANT = "baseline"


class ContextVariantConfigurationError(ValueError):
    """The server-side context variant configuration is invalid."""


class UnknownContextVariantError(LookupError):
    """A caller requested a variant that the server has not configured."""


def normalize_context_variant_name(value: str) -> str:
    """Normalize and validate a public context variant name."""

    normalized = str(value or "").strip().lower()
    if not normalized or not _VARIANT_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "context_variant must use 1-64 lowercase letters, numbers, '.', '_' or '-'"
        )
    return normalized


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _json_safe(value: Any) -> Any:
    """Convert agent state (which contains sets) into a JSON-safe structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=lambda item: str(item))
    return str(value)


def _compact_json(value: Any, limit: int = 1600) -> str:
    text = json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _trusted_agent_state_text(agent_state: Mapping[str, Any] | None) -> str:
    """Keep task-critical state under the main application's control."""

    state = agent_state or {}
    lines: list[str] = []
    task_done = state.get("task_done")
    if task_done:
        lines.append(f"已完成任务：{_compact_json(task_done, 900)}")
    memory_words = state.get("memory_words")
    if memory_words:
        lines.append(f"登记的记忆词：{_compact_json(memory_words, 300)}")
    persona_hooks = state.get("persona_hooks")
    if persona_hooks:
        lines.append(f"患者兴趣/习惯：{_compact_json(persona_hooks, 500)}")
    mmse_total = state.get("mmse_total_score")
    if mmse_total is not None:
        lines.append(f"当前MMSE得分：{mmse_total}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RemoteContextVariantConfig:
    name: str
    url: str
    api_key: str = ""
    timeout_seconds: float = 1.5
    max_recent_messages: int = 12
    max_response_bytes: int = 262_144
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> "RemoteContextVariantConfig":
        normalized_name = normalize_context_variant_name(name)
        if normalized_name == _BASELINE_VARIANT:
            raise ContextVariantConfigurationError("'baseline' is reserved")

        url = str(value.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ContextVariantConfigurationError(
                f"context variant '{normalized_name}' needs an absolute http(s) URL"
            )
        if parsed.username or parsed.password:
            raise ContextVariantConfigurationError(
                f"context variant '{normalized_name}' URL must not contain credentials"
            )

        api_key_env = str(value.get("api_key_env") or "").strip()
        api_key = str(value.get("api_key") or "").strip()
        if api_key_env:
            api_key = str(environ.get(api_key_env) or "").strip()

        return cls(
            name=normalized_name,
            url=url,
            api_key=api_key,
            timeout_seconds=_clamp_float(value.get("timeout_seconds"), 1.5, 0.05, 10.0),
            max_recent_messages=_clamp_int(value.get("max_recent_messages"), 12, 1, 40),
            max_response_bytes=_clamp_int(
                value.get("max_response_bytes"), 262_144, 4_096, 2_097_152
            ),
            failure_threshold=_clamp_int(value.get("failure_threshold"), 3, 1, 20),
            cooldown_seconds=_clamp_float(value.get("cooldown_seconds"), 30.0, 1.0, 600.0),
        )


class ContextVariantRegistry:
    """Server-owned allow-list of context implementations."""

    def __init__(
        self,
        variants: Optional[Mapping[str, RemoteContextVariantConfig]] = None,
        default_variant: str = _BASELINE_VARIANT,
    ) -> None:
        self._variants = dict(variants or {})
        self.default_variant = normalize_context_variant_name(default_variant)
        if self.default_variant != _BASELINE_VARIANT and self.default_variant not in self._variants:
            raise ContextVariantConfigurationError(
                f"default context variant '{self.default_variant}' is not configured"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        *,
        strict: bool = True,
    ) -> "ContextVariantRegistry":
        env = environ if environ is not None else os.environ
        raw = str(env.get("CONTEXT_PROVIDER_VARIANTS_JSON") or "").strip()
        raw_variants: dict[str, Any] = {}
        if raw:
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ContextVariantConfigurationError(
                        f"CONTEXT_PROVIDER_VARIANTS_JSON is invalid JSON: {exc.msg}"
                    ) from exc
                decoded = {}
            if not isinstance(decoded, dict):
                if strict:
                    raise ContextVariantConfigurationError(
                        "CONTEXT_PROVIDER_VARIANTS_JSON must be a JSON object"
                    )
                decoded = {}
            raw_variants.update(decoded)

        # A simple single-service configuration is convenient for the first
        # candidate; JSON remains available when several versions coexist.
        single_url = str(env.get("CONTEXT_PROVIDER_URL") or "").strip()
        if single_url:
            single_name = str(env.get("CONTEXT_PROVIDER_VARIANT") or "candidate-v1").strip()
            raw_variants.setdefault(
                single_name,
                {
                    "url": single_url,
                    "api_key_env": "CONTEXT_PROVIDER_API_KEY",
                    "timeout_seconds": env.get("CONTEXT_PROVIDER_TIMEOUT_SECONDS", "1.5"),
                    "max_recent_messages": env.get("CONTEXT_PROVIDER_MAX_RECENT_MESSAGES", "12"),
                    "failure_threshold": env.get("CONTEXT_PROVIDER_FAILURE_THRESHOLD", "3"),
                    "cooldown_seconds": env.get("CONTEXT_PROVIDER_COOLDOWN_SECONDS", "30"),
                },
            )

        variants: dict[str, RemoteContextVariantConfig] = {}
        try:
            for name, value in raw_variants.items():
                if not isinstance(value, Mapping):
                    raise ContextVariantConfigurationError(
                        f"context variant '{name}' configuration must be an object"
                    )
                config = RemoteContextVariantConfig.from_mapping(str(name), value, env)
                variants[config.name] = config
            default_variant = str(
                env.get("CONTEXT_PROVIDER_DEFAULT_VARIANT") or _BASELINE_VARIANT
            )
            return cls(variants=variants, default_variant=default_variant)
        except (ContextVariantConfigurationError, ValueError):
            if strict:
                raise
            return cls()

    @property
    def available_variants(self) -> list[str]:
        return [_BASELINE_VARIANT, *sorted(self._variants)]

    def resolve(self, requested: Optional[str]) -> str:
        variant = self.default_variant if requested is None or not str(requested).strip() else requested
        normalized = normalize_context_variant_name(str(variant))
        if normalized != _BASELINE_VARIANT and normalized not in self._variants:
            raise UnknownContextVariantError(normalized)
        return normalized

    def install(
        self,
        agent: Any,
        *,
        variant: str,
        session_id: str,
        patient_profile: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        resolved = self.resolve(variant)
        baseline_memory = agent.tool_gateway.memory_tool
        if resolved == _BASELINE_VARIANT:
            return baseline_memory
        adapter = RemoteContextMemoryTool(
            baseline_memory=baseline_memory,
            config=self._variants[resolved],
            session_id=session_id,
            patient_profile=patient_profile,
        )
        agent.tool_gateway.memory_tool = adapter
        return adapter


class RemoteContextMemoryTool:
    """MemoryTool-compatible adapter backed by a sandboxed HTTP service.

    Remote failures never fail the screening turn: the adapter falls back to
    the existing memory tool and opens a short circuit after repeated errors.
    """

    schema_version = "1.1"

    def __init__(
        self,
        *,
        baseline_memory: Any,
        config: RemoteContextVariantConfig,
        session_id: str,
        patient_profile: Optional[Mapping[str, Any]] = None,
        urlopen: Optional[Callable[..., Any]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._baseline = baseline_memory
        self._config = config
        self._session_id = session_id
        self._patient_profile = dict(patient_profile or {})
        self._urlopen = urlopen or urllib.request.urlopen
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._persistent_background = ""
        self._turn_background = ""
        self._pending_task_context: Optional[dict[str, Any]] = None
        self._remote_topics: list[str] = []
        self._remote_suggestion: Optional[dict[str, Any]] = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._cache_key = ""
        self._cache_context: Optional[dict[str, Any]] = None
        self._diagnostics: dict[str, Any] = {
            "provider": "remote",
            "context_variant": config.name,
            "status": "not_called",
            "fallback": False,
            "latency_ms": 0,
        }

    def __getattr__(self, name: str) -> Any:
        # Preserve compatibility with non-context diagnostics exposed by the
        # original ConversationMemoryTool.
        return getattr(self._baseline, name)

    def set_persistent_background(self, text: str) -> None:
        """患者跨会话记忆卡：由本适配器统一注入到 summary 头部。

        记忆卡属于主项目可信数据（与 trusted agent state 同理），不依赖候选
        服务返回：无论候选成功还是回退 baseline，都由本适配器 prepend，保证
        单一数据源、不重复注入。同时也放进请求体供候选算法参考。
        """
        with self._lock:
            self._persistent_background = str(text or "").strip()
            self._cache_key = ""
            self._cache_context = None

    def set_turn_background(self, text: str) -> None:
        with self._lock:
            self._turn_background = str(text or "").strip()
            self._cache_key = ""
            self._cache_context = None

    def _apply_background(self, summary: str) -> str:
        summary = str(summary or "").strip()
        parts = [self._persistent_background, self._turn_background, summary]
        return "\n".join(part for part in parts if part)

    def _request_payload(
        self,
        chat_history: list[dict[str, str]],
        agent_state: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_variant": self._config.name,
            "session_id": self._session_id,
            "patient": _json_safe(self._patient_profile),
            "patient_memory": self._persistent_background or None,
            "messages": _json_safe(chat_history),
            "agent_state": _json_safe(agent_state or {}),
            "task_context": _json_safe(self._pending_task_context or {}),
            "limits": {
                "max_recent_messages": self._config.max_recent_messages,
                "max_summary_characters": 4000,
            },
        }

    def _decode_response(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("context response must be a JSON object")

        summary = str(payload.get("summary") or "").strip()
        if len(summary) > 4000:
            summary = summary[:4000]

        raw_recent = payload.get("recent_messages", payload.get("recent", []))
        if not isinstance(raw_recent, list):
            raise ValueError("recent_messages must be an array")
        recent: list[dict[str, str]] = []
        for item in raw_recent[-self._config.max_recent_messages :]:
            if not isinstance(item, Mapping):
                raise ValueError("each recent message must be an object")
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                raise ValueError("recent message needs a valid role and non-empty content")
            recent.append({"role": role, "content": content[:4000]})

        raw_topics = payload.get("discussed_topics", [])
        if not isinstance(raw_topics, list):
            raise ValueError("discussed_topics must be an array")
        topics = [str(item).strip()[:200] for item in raw_topics if str(item).strip()][:100]

        raw_suggestion = payload.get("next_task_suggestion", payload.get("next_suggestion"))
        suggestion = dict(raw_suggestion) if isinstance(raw_suggestion, Mapping) else None

        facts = payload.get("facts")
        asked_questions = payload.get("asked_questions")
        summary_parts = [summary] if summary else []
        if facts:
            summary_parts.append(f"关键事实：{_compact_json(facts)}")
        if asked_questions:
            summary_parts.append(f"已问问题：{_compact_json(asked_questions)}")

        estimated_tokens = payload.get("estimated_tokens")
        try:
            estimated_tokens = int(estimated_tokens) if estimated_tokens is not None else None
        except (TypeError, ValueError):
            estimated_tokens = None

        return {
            "summary": "\n".join(summary_parts),
            "recent": recent,
            "topics": topics,
            "suggestion": suggestion,
            "provider_version": str(payload.get("version") or "").strip()[:100] or None,
            "estimated_tokens": estimated_tokens,
        }

    def _call_remote(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "ADScreening-ContextClient/1.0",
            "X-Context-Variant": self._config.name,
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        request = urllib.request.Request(
            self._config.url,
            data=body,
            headers=headers,
            method="POST",
        )
        with self._urlopen(request, timeout=self._config.timeout_seconds) as response:
            raw = response.read(self._config.max_response_bytes + 1)
        if len(raw) > self._config.max_response_bytes:
            raise ValueError("context response is too large")
        return self._decode_response(json.loads(raw.decode("utf-8")))

    def _fallback_context(
        self,
        chat_history: list[dict[str, str]],
        agent_state: Optional[Mapping[str, Any]],
        *,
        status: str,
        latency_ms: int = 0,
        error_type: Optional[str] = None,
    ) -> dict[str, Any]:
        context = self._baseline.get_context(chat_history, agent_state)
        context = {**context, "summary": self._apply_background(context.get("summary", ""))}
        self._diagnostics = {
            "provider": "remote",
            "context_variant": self._config.name,
            "status": status,
            "fallback": True,
            "latency_ms": latency_ms,
            "error": error_type,
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": self._monotonic() < self._circuit_open_until,
            "input_messages": len(chat_history),
            "output_messages": len(context.get("recent") or []),
        }
        return context

    def get_context(
        self,
        chat_history: list[dict[str, str]],
        agent_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = self._monotonic()
            if now < self._circuit_open_until:
                return self._fallback_context(
                    chat_history,
                    agent_state,
                    status="circuit_open",
                )

            request_payload = self._request_payload(chat_history, agent_state)
            cache_key = hashlib.sha256(
                json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if cache_key == self._cache_key and self._cache_context is not None:
                cached = {
                    "summary": self._cache_context["summary"],
                    "recent": list(self._cache_context["recent"]),
                }
                self._diagnostics = {
                    **self._diagnostics,
                    "status": "cache_hit",
                    "fallback": False,
                    "latency_ms": 0,
                }
                return cached

            started = self._monotonic()
            try:
                remote = self._call_remote(request_payload)
            except Exception as exc:
                latency_ms = max(0, round((self._monotonic() - started) * 1000))
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._config.failure_threshold:
                    self._circuit_open_until = self._monotonic() + self._config.cooldown_seconds
                print(
                    f"[ContextProvider] ⚠️ {self._config.name} 调用失败，回退 baseline: "
                    f"{type(exc).__name__}: {exc}"
                )
                return self._fallback_context(
                    chat_history,
                    agent_state,
                    status="fallback",
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                )

            latency_ms = max(0, round((self._monotonic() - started) * 1000))
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._remote_topics = remote["topics"]
            self._remote_suggestion = remote["suggestion"]

            trusted_state = _trusted_agent_state_text(agent_state)
            summary = remote["summary"]
            if trusted_state:
                summary = f"{summary}\n{trusted_state}".strip()
            summary = self._apply_background(summary)
            context = {"summary": summary, "recent": remote["recent"]}
            self._cache_key = cache_key
            self._cache_context = {"summary": summary, "recent": list(remote["recent"])}
            self._diagnostics = {
                "provider": "remote",
                "context_variant": self._config.name,
                "status": "ok",
                "fallback": False,
                "latency_ms": latency_ms,
                "provider_version": remote["provider_version"],
                "estimated_tokens": remote["estimated_tokens"],
                "input_messages": len(chat_history),
                "output_messages": len(remote["recent"]),
                "summary_characters": len(summary),
                "consecutive_failures": 0,
                "circuit_open": False,
            }
            return context

    def update_async(
        self,
        chat_history: list[dict[str, str]],
        task_context: Optional[dict[str, Any]] = None,
    ) -> None:
        # The remote service is stateless by contract and receives the complete
        # visible history on the next build call.  Retain only the task hint and
        # invalidate the request cache.
        with self._lock:
            self._pending_task_context = dict(task_context or {}) or None
            self._cache_key = ""
            self._cache_context = None

    def update(
        self,
        chat_history: list[dict[str, str]],
        task_context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.update_async(chat_history, task_context)

    def reset(self) -> None:
        with self._lock:
            self._pending_task_context = None
            self._turn_background = ""
            self._remote_topics = []
            self._remote_suggestion = None
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._cache_key = ""
            self._cache_context = None
            self._diagnostics.update(
                {"status": "not_called", "fallback": False, "latency_ms": 0}
            )
        self._baseline.reset()

    def get_discussed_topics(self) -> list[str]:
        with self._lock:
            if self._remote_topics:
                return list(self._remote_topics)
        getter = getattr(self._baseline, "get_discussed_topics", None)
        return list(getter() or []) if callable(getter) else []

    def get_discussed_topics_snapshot(self) -> list[str]:
        with self._lock:
            if self._remote_topics:
                return list(self._remote_topics)
        getter = getattr(self._baseline, "get_discussed_topics_snapshot", None)
        return list(getter() or []) if callable(getter) else []

    def get_discussed_topics_str(self) -> str:
        topics = self.get_discussed_topics()
        return "、".join(topics) if topics else "无"

    def get_discussed_topics_str_snapshot(self) -> str:
        topics = self.get_discussed_topics_snapshot()
        return "、".join(topics) if topics else "无"

    def get_and_clear_suggestion(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._remote_suggestion:
                suggestion = dict(self._remote_suggestion)
                self._remote_suggestion = None
                return suggestion
        getter = getattr(self._baseline, "get_and_clear_suggestion", None)
        return getter() if callable(getter) else None

    def get_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._diagnostics)
