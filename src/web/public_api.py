from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from src.context_management import (
    ContextVariantRegistry,
    UnknownContextVariantError,
    normalize_context_variant_name,
)
from src.db import database
from src.voice_modes import COGNITIVE_SCREENING


def _cognitive_screening_enabled() -> bool:
    return os.getenv("ENABLE_COGNITIVE_SCREENING", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_context_registry(*, logger=print) -> ContextVariantRegistry:
    try:
        return ContextVariantRegistry.from_environment(strict=True)
    except Exception as exc:
        logger(
            "[ContextProvider] ⚠️ 候选上下文配置无效，"
            f"仅启用 baseline: {exc}"
        )
        return ContextVariantRegistry()


class PublicApiPatientProfile(BaseModel):
    patient_id: Optional[str] = Field(default=None, max_length=80)
    name: Optional[str] = Field(default=None, max_length=80)
    age: Optional[int] = Field(default=None, ge=0, le=150)
    gender: Optional[str] = Field(default=None, max_length=32)
    education_years: Optional[int] = Field(default=None, ge=0, le=40)


class PublicApiSessionCreateRequest(BaseModel):
    patient: PublicApiPatientProfile = Field(
        default_factory=PublicApiPatientProfile
    )
    context_variant: Optional[str] = Field(default=None, max_length=64)

    @field_validator("context_variant")
    @classmethod
    def _validate_context_variant(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        return normalize_context_variant_name(value)


class PublicApiTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    emotion: str = Field(default="neutral", max_length=64)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class PublicApiVisionRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=120)
    image: Optional[str] = Field(default=None, max_length=16_000_000)
    video: Optional[str] = Field(default=None, max_length=32_000_000)
    frames: list[str] = Field(default_factory=list, max_length=32)
    mime_type: str = Field(default="video/webm", max_length=128)
    context: str = Field(default="", max_length=4000)

    @field_validator("task_id")
    @classmethod
    def _strip_task_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task_id must not be blank")
        return value


class PublicApiKeyService:
    """Validate bootstrap/managed API keys and enforce per-key rate limits."""

    def __init__(
        self,
        *,
        key_file: Path,
        repository=database,
        environ=None,
        monotonic: Callable[[], float] = time.monotonic,
        logger=print,
    ) -> None:
        self.key_file = Path(key_file)
        self.repository = repository
        self.environ = environ if environ is not None else os.environ
        self._monotonic = monotonic
        self._log = logger
        self._rate_lock = threading.Lock()
        self._rate_buckets: dict[str, deque[float]] = {}

    def require(self, request: Request) -> str:
        supplied_key = self._extract(request)
        if not supplied_key:
            raise HTTPException(status_code=401, detail="INVALID_API_KEY")

        supplied_hash = hashlib.sha256(
            supplied_key.encode("utf-8")
        ).hexdigest()
        configured_keys = self._load_bootstrap_keys()
        if any(
            hmac.compare_digest(supplied_key, key)
            for key in configured_keys
        ):
            key_id = f"bootstrap:{supplied_hash[:16]}"
        else:
            managed_key = (
                self.repository.get_active_public_api_key_by_hash(
                    supplied_hash
                )
            )
            if not managed_key:
                raise HTTPException(
                    status_code=401,
                    detail="INVALID_API_KEY",
                )
            key_id = str(managed_key["key_id"])
            self.repository.touch_public_api_key(key_id)

        self._apply_rate_limit(key_id)
        return key_id

    def _extract(self, request: Request) -> str:
        authorization = request.headers.get(
            "Authorization",
            "",
        ).strip()
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return request.headers.get("X-API-Key", "").strip()

    def _load_bootstrap_keys(self) -> list[str]:
        configured = str(
            self.environ.get("PUBLIC_API_KEYS") or ""
        ).strip()
        if configured:
            return [
                key.strip()
                for key in configured.split(",")
                if key.strip()
            ]
        try:
            if self.key_file.exists():
                key = self.key_file.read_text(
                    encoding="utf-8"
                ).strip()
                if key:
                    return [key]
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            key = f"adsk_{secrets.token_urlsafe(32)}"
            self.key_file.write_text(key + "\n", encoding="utf-8")
            try:
                os.chmod(self.key_file, 0o600)
            except OSError:
                pass
            self._log(
                "[Public API] 已生成 bootstrap API Key："
                f"{self.key_file}"
            )
            return [key]
        except Exception as exc:
            self._log(
                f"[Public API] ⚠️ 无法读取或生成 API Key: {exc}"
            )
            return []

    def _apply_rate_limit(self, key_id: str) -> None:
        try:
            per_minute = max(
                1,
                int(
                    self.environ.get(
                        "PUBLIC_API_RATE_LIMIT_PER_MINUTE",
                        "60",
                    )
                ),
            )
        except ValueError:
            per_minute = 60
        now = self._monotonic()
        with self._rate_lock:
            bucket = self._rate_buckets.setdefault(key_id, deque())
            while bucket and now - bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="RATE_LIMITED",
                )
            bucket.append(now)


@dataclass
class PublicApiScreeningSession:
    agent: Any
    patient_profile: dict[str, Any]
    patient_id: str | None = None
    context_variant: str = "baseline"
    context_memory: Any = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    last_active_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PublicApiSessionStore:
    """Own capacity, expiry and locking for public screening sessions."""

    def __init__(
        self,
        *,
        environ=None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.environ = environ if environ is not None else os.environ
        self._monotonic = monotonic
        self._sessions: dict[str, PublicApiScreeningSession] = {}
        self._lock = asyncio.Lock()

    @property
    def ttl_seconds(self) -> int:
        try:
            return max(
                300,
                int(
                    self.environ.get(
                        "PUBLIC_API_SESSION_TTL_SECONDS",
                        "3600",
                    )
                ),
            )
        except ValueError:
            return 3600

    @property
    def max_sessions(self) -> int:
        try:
            return max(
                1,
                int(
                    self.environ.get(
                        "PUBLIC_API_MAX_SESSIONS",
                        "100",
                    )
                ),
            )
        except ValueError:
            return 100

    async def add(
        self,
        session_id: str,
        state: PublicApiScreeningSession,
    ) -> None:
        async with self._lock:
            self._evict_expired_locked()
            if len(self._sessions) >= self.max_sessions:
                raise HTTPException(
                    status_code=503,
                    detail="PUBLIC_API_SESSION_CAPACITY_REACHED",
                )
            self._sessions[session_id] = state

    async def get(
        self,
        session_id: str,
    ) -> PublicApiScreeningSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    def expired(self, state: PublicApiScreeningSession) -> bool:
        return (
            self._monotonic() - state.last_active_at
            > self.ttl_seconds
        )

    def touch(self, state: PublicApiScreeningSession) -> None:
        state.last_active_at = self._monotonic()

    def _evict_expired_locked(self) -> None:
        now = self._monotonic()
        for session_id, state in list(self._sessions.items()):
            if now - state.last_active_at > self.ttl_seconds:
                self._sessions.pop(session_id, None)


class PublicApiController:
    """Register the authenticated external screening API."""

    def __init__(
        self,
        *,
        key_service: PublicApiKeyService,
        session_store: PublicApiSessionStore,
        context_registry: ContextVariantRegistry,
        agent_factory: Callable[[], Any],
        service_ready: Callable[[], bool],
        create_session: Callable[..., Any],
        static_dir: Path,
        memory: Any = None,
    ) -> None:
        self.key_service = key_service
        self.session_store = session_store
        self.context_registry = context_registry
        self._agent_factory = agent_factory
        self._service_ready = service_ready
        self._create_session = create_session
        self.memory = memory
        self.static_dir = Path(static_dir)
        self.router = APIRouter()
        self._register_routes()

    def install(self, app) -> None:
        app.include_router(self.router)

    async def health(self, request: Request):
        self.key_service.require(request)
        return {
            "status": "ok",
            "service": "ad-screening-api",
            "version": "v1",
            "capabilities": [
                "screening",
                "vision_evaluate",
                "context_variants",
            ],
            "context_variants": (
                self.context_registry.available_variants
            ),
            "default_context_variant": (
                self.context_registry.default_variant
            ),
        }

    async def context_variants(self, request: Request):
        self.key_service.require(request)
        return {
            "variants": self.context_registry.available_variants,
            "default": self.context_registry.default_variant,
        }

    async def developer_portal(self):
        return FileResponse(
            self.static_dir / "api_portal.html",
            media_type="text/html; charset=utf-8",
            headers={
                "Cache-Control": (
                    "no-cache, no-store, must-revalidate"
                )
            },
        )

    async def docs_redirect(self):
        return RedirectResponse(url="/developers", status_code=307)

    async def create_screening_session(
        self,
        payload: PublicApiSessionCreateRequest,
        request: Request,
    ):
        key_id = self.key_service.require(request)
        if not _cognitive_screening_enabled():
            raise HTTPException(
                status_code=403,
                detail="COGNITIVE_SCREENING_DISABLED",
            )
        if not self._service_ready():
            raise HTTPException(
                status_code=503,
                detail="SERVICE_STARTING",
            )

        profile = payload.patient.model_dump(exclude_none=True)
        if "patient_id" in profile:
            raise HTTPException(
                status_code=403,
                detail="PUBLIC_API_PATIENT_SCOPE_REQUIRED",
            )
        session_id = f"api_{uuid.uuid4().hex}"
        requested_variant = (
            payload.context_variant
            or request.headers.get("X-Context-Variant")
            or None
        )
        try:
            context_variant = self.context_registry.resolve(
                requested_variant
            )
        except (UnknownContextVariantError, ValueError):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "UNKNOWN_CONTEXT_VARIANT",
                    "requested": requested_variant,
                    "available": (
                        self.context_registry.available_variants
                    ),
                },
            )

        agent = await asyncio.to_thread(self._agent_factory)
        context_memory = self.context_registry.install(
            agent,
            variant=context_variant,
            session_id=session_id,
            patient_profile=profile,
        )
        state = PublicApiScreeningSession(
            agent=agent,
            patient_profile=profile,
            patient_id=profile.get("patient_id"),
            context_variant=context_variant,
            context_memory=context_memory,
        )
        await self.session_store.add(session_id, state)

        try:
            self._create_session(
                session_id,
                profile=profile,
                owner_username=f"api:{key_id}",
                mode=COGNITIVE_SCREENING,
            )
        except Exception:
            await self.session_store.remove(session_id)
            raise

        return {
            "session_id": session_id,
            "context_variant": context_variant,
            "expires_in_seconds": self.session_store.ttl_seconds,
            "message": (
                "会话已创建。请使用 "
                "/v1/screening/sessions/{session_id}/turn "
                "提交患者回答。"
            ),
        }

    async def screening_turn(
        self,
        session_id: str,
        payload: PublicApiTurnRequest,
        request: Request,
    ):
        self.key_service.require(request)
        if not _cognitive_screening_enabled():
            raise HTTPException(
                status_code=403,
                detail="COGNITIVE_SCREENING_DISABLED",
            )
        state = await self.session_store.get(session_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail="PUBLIC_API_SESSION_NOT_FOUND",
            )

        async with state.lock:
            if self.session_store.expired(state):
                await self.session_store.remove(session_id)
                raise HTTPException(
                    status_code=410,
                    detail="PUBLIC_API_SESSION_EXPIRED",
                )
            if state.patient_id and self.memory is not None:
                try:
                    memory_context = self.memory.get_context_for_llm(
                        state.patient_id
                    )
                    setter = getattr(
                        state.agent.tool_gateway.memory_tool,
                        "set_persistent_background",
                        None,
                    )
                    if callable(setter):
                        setter(memory_context)
                except Exception:
                    pass
            result = await asyncio.to_thread(
                state.agent.process_turn,
                user_input=payload.message,
                session_id=session_id,
                patient_profile=state.patient_profile,
                chat_history=state.chat_history,
                current_emotion=payload.emotion,
            )
            response_text = str(
                result.get("output") or result.get("response") or ""
            )
            state.chat_history.extend(
                [
                    {"role": "user", "content": payload.message},
                    {
                        "role": "assistant",
                        "content": response_text,
                    },
                ]
            )
            state.chat_history = state.chat_history[-40:]
            if state.patient_id and self.memory is not None:
                try:
                    self.memory.capture_turn(
                        state.patient_id,
                        payload.message,
                        response_text,
                        session_id=session_id,
                        emotion=payload.emotion,
                    )
                except Exception:
                    pass
            self.session_store.touch(state)
            diagnostics_getter = getattr(
                state.context_memory,
                "get_diagnostics",
                None,
            )
            context_diagnostics = (
                diagnostics_getter()
                if callable(diagnostics_getter)
                else {
                    "provider": "builtin",
                    "context_variant": state.context_variant,
                    "status": "ok",
                    "fallback": False,
                }
            )

        return {
            "session_id": session_id,
            "context_variant": state.context_variant,
            "context_diagnostics": context_diagnostics,
            "response": response_text,
            "task_id": (
                result.get("task_id")
                or result.get("current_task_id")
            ),
            "dimension_id": result.get("dimension_id"),
            "is_comfort_mode": bool(
                result.get("is_comfort_mode", False)
            ),
            "is_complete": bool(result.get("is_complete", False)),
            "image_display": result.get("image_display_command"),
            "vision_command": result.get("vision_command"),
        }

    async def vision_evaluate(
        self,
        payload: PublicApiVisionRequest,
        request: Request,
    ):
        self.key_service.require(request)
        if not _cognitive_screening_enabled():
            raise HTTPException(
                status_code=403,
                detail="COGNITIVE_SCREENING_DISABLED",
            )
        if not (payload.video or payload.image or payload.frames):
            raise HTTPException(
                status_code=400,
                detail="IMAGE_VIDEO_OR_FRAMES_REQUIRED",
            )
        from src.tools.agent_tools.vision_evaluation_tool import (
            evaluate_hybrid,
            evaluate_image_with_vlm,
        )

        try:
            if payload.video or payload.frames:
                result = await asyncio.to_thread(
                    evaluate_hybrid,
                    task_id=payload.task_id,
                    video_base64=payload.video or "",
                    mime_type=payload.mime_type,
                    frames_base64=payload.frames,
                    extra_context=payload.context,
                )
            else:
                result = await asyncio.to_thread(
                    evaluate_image_with_vlm,
                    image_base64=payload.image or "",
                    task_id=payload.task_id,
                    extra_context=payload.context,
                )
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse(
                {
                    "success": False,
                    "error": "VISION_EVALUATION_FAILED",
                    "detail": str(exc),
                },
                status_code=500,
            )

    def _register_routes(self) -> None:
        self.router.add_api_route(
            "/v1/health",
            self.health,
            methods=["GET"],
            tags=["Public API"],
        )
        self.router.add_api_route(
            "/v1/context/variants",
            self.context_variants,
            methods=["GET"],
            tags=["Public API"],
        )
        self.router.add_api_route(
            "/developers",
            self.developer_portal,
            methods=["GET"],
            include_in_schema=False,
        )
        self.router.add_api_route(
            "/api-docs",
            self.docs_redirect,
            methods=["GET"],
            include_in_schema=False,
        )
        self.router.add_api_route(
            "/v1/screening/sessions",
            self.create_screening_session,
            methods=["POST"],
            status_code=201,
            tags=["Public API"],
        )
        self.router.add_api_route(
            "/v1/screening/sessions/{session_id}/turn",
            self.screening_turn,
            methods=["POST"],
            tags=["Public API"],
        )
        self.router.add_api_route(
            "/v1/vision/evaluate",
            self.vision_evaluate,
            methods=["POST"],
            tags=["Public API"],
        )
