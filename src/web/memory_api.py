from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.db.database import MemoryRevisionConflict, record_patient_audit_event


__all__ = [
    "ComfortStrategyRequest",
    "MMSERecordRequest",
    "MemoryApiController",
    "MemoryItemDeleteRequest",
    "MemoryItemVersionRequest",
    "MemoryItemUpdateRequest",
    "MemoryRevisionConflict",
    "MemoryUpdateRequest",
    "MirrorOpRequeueRequest",
]


PatientId = Annotated[str, Path(min_length=1, max_length=120)]
MemoryString = Annotated[str, Field(min_length=1, max_length=128)]


def _enabled(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: dict[str, Any] = Field(min_length=1, max_length=16)
    expected_revision: int = Field(ge=0)

    @field_validator("updates")
    @classmethod
    def _validate_update_keys(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = key.strip()
            if not normalized_key or len(normalized_key) > 64:
                raise ValueError(
                    "update keys must be 1-64 non-whitespace characters"
                )
            normalized[normalized_key] = item
        return normalized


class MMSERecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=35)
    weak_dimensions: list[MemoryString] = Field(
        default_factory=list,
        max_length=16,
    )
    date: str | None = Field(default=None, max_length=32)

    @field_validator("weak_dimensions")
    @classmethod
    def _strip_dimensions(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("weak_dimensions must not contain blanks")
        return normalized

    @field_validator("date")
    @classmethod
    def _strip_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ComfortStrategyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(min_length=1, max_length=500)
    effective: bool

    @field_validator("strategy")
    @classmethod
    def _strip_strategy(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("strategy must not be blank")
        return value


class MemoryItemUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1000)
    expected_version: int = Field(ge=1)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class MemoryItemVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)


class MemoryItemDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    deletion_token: str = Field(min_length=8, max_length=128)

    @field_validator("deletion_token")
    @classmethod
    def _strip_token(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deletion_token must not be blank")
        return value


class MirrorOpRequeueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="api_manual", min_length=1, max_length=200)


class MemoryApiController:
    """Register the patient memory API around an injected memory service."""

    def __init__(
        self,
        memory: Any,
        auth: Any = None,
        audit_event: Any = record_patient_audit_event,
    ) -> None:
        self.memory = memory
        self.auth = auth
        self.audit_event = audit_event
        self.router = APIRouter()
        self._register_routes()

    def install(self, app) -> None:
        app.include_router(self.router)

    def _authorize(
        self,
        request: Request,
        patient_id: str,
        action: str,
    ) -> str:
        patient_id = str(patient_id or "").strip()
        if self.auth is None:
            raise HTTPException(
                status_code=503,
                detail="MEMORY_API_AUTH_NOT_CONFIGURED",
            )
        require = getattr(self.auth, "require_patient_access", None)
        if callable(require):
            require(request, patient_id, action)
            return patient_id
        require_user = getattr(self.auth, "require_authenticated_user", None)
        authorize = getattr(self.auth, "authorize_patient", None)
        if callable(require_user) and callable(authorize):
            authorize(require_user(request), patient_id, action)
            return patient_id
        if callable(self.auth):
            self.auth(request, patient_id, action)
            return patient_id
        raise HTTPException(
            status_code=500,
            detail="INVALID_MEMORY_API_AUTH_DEPENDENCY",
        )

    def _authorize_admin(self, request: Request) -> None:
        require_admin = getattr(self.auth, "require_admin_user", None)
        if not callable(require_admin):
            raise HTTPException(status_code=503, detail="MEMORY_API_AUTH_NOT_CONFIGURED")
        require_admin(request)

    def _call_memory(self, method_name: str, *args, **kwargs):
        method = getattr(self.memory, method_name, None)
        if not callable(method):
            raise HTTPException(
                status_code=501,
                detail=f"MEMORY_METHOD_NOT_SUPPORTED:{method_name}",
            )
        try:
            return method(*args, **kwargs)
        except HTTPException:
            raise
        except MemoryRevisionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "MEMORY_REVISION_CONFLICT",
                    "current_revision": exc.current_revision,
                },
            ) from exc
        except TypeError as exc:
            if "expected_revision" in str(exc):
                raise HTTPException(
                    status_code=501,
                    detail="MEMORY_REVISION_NOT_SUPPORTED",
                ) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @staticmethod
    def _memory_item_view(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "item_id", "category", "content", "status", "version",
                "observed_at", "valid_until", "sensitivity", "confidence",
                "created_at", "updated_at", "source",
            )
        }

    @staticmethod
    def _session_reflection_view(reflection: dict[str, Any]) -> dict[str, Any]:
        view = {
            key: reflection.get(key)
            for key in (
                "session_id", "status", "session_summary", "attempt_count",
                "created_at", "updated_at", "completed_at",
            )
        }
        if reflection.get("status") in {"failed", "dead_letter"}:
            error = str(reflection.get("last_error") or "").lower()
            if "timeout" in error or "timed out" in error:
                reason = "模型服务响应超时"
            elif any(token in error for token in ("json", "decode", "parse")):
                reason = "模型返回格式异常"
            elif any(token in error for token in ("evidence", "source_turn", "patient turn")):
                reason = "摘要证据校验未通过"
            else:
                reason = "后台整理暂时不可用"
            view["failure_reason"] = reason
            view["next_retry_at"] = reflection.get("next_retry_at")
            view["error_code"] = reflection.get("error_code")
        return view

    def _call_memory_item(self, patient_id: str, item_id: str, method_name: str, *args, **kwargs):
        try:
            return self._call_memory(method_name, patient_id, item_id, *args, **kwargs)
        except HTTPException as exc:
            if exc.status_code == 409 and isinstance(exc.detail, dict):
                try:
                    current = self._call_memory("get_memory_item", patient_id, item_id)
                    exc.detail["current_item"] = self._memory_item_view(current)
                except HTTPException:
                    pass
            raise
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            current_revision = getattr(exc, "current_revision", None)
            if current_revision is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "MEMORY_REVISION_CONFLICT",
                        "current_revision": int(current_revision),
                    },
                ) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _actor_username(self, request: Request) -> str | None:
        current_user = getattr(self.auth, "current_user", None)
        if callable(current_user):
            try:
                user = current_user(request)
                return str((user or {}).get("username") or "").strip() or None
            except Exception:
                return None
        return None

    def _audit_memory_item(
        self,
        request: Request,
        patient_id: str,
        action: str,
        item: dict[str, Any],
    ) -> None:
        if not callable(self.audit_event):
            return
        revision = item.get("version")
        self.audit_event(
            actor_username=self._actor_username(request),
            patient_id=patient_id,
            action=action,
            outcome="success",
            object_revision=int(revision) if revision is not None else None,
        )

    async def list_memory_items(
        self,
        patient_id: PatientId,
        request: Request,
        include_deleted: bool = Query(default=False),
        mode: str | None = Query(default=None, max_length=64),
    ):
        patient_id = self._authorize(request, patient_id, "read")
        items = self._call_memory(
            "list_memory_items",
            patient_id,
            include_deleted=include_deleted,
            mode=mode,
        )
        return {"success": True, "items": [self._memory_item_view(item) for item in items]}

    async def update_memory_item(
        self,
        patient_id: PatientId,
        item_id: str,
        payload: MemoryItemUpdateRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        item = self._call_memory_item(
            patient_id,
            item_id,
            "update_memory_item",
            payload.content,
            payload.expected_version,
            updated_by="user",
        )
        self._audit_memory_item(
            request,
            patient_id,
            "memory_item_updated",
            item,
        )
        return {"success": True, "item": self._memory_item_view(item)}

    async def confirm_memory_item(
        self,
        patient_id: PatientId,
        item_id: str,
        payload: MemoryItemVersionRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        item = self._call_memory_item(
            patient_id, item_id, "confirm_memory_item", payload.expected_version, confirmed_by="user"
        )
        self._audit_memory_item(request, patient_id, "memory_item_confirmed", item)
        return {"success": True, "item": self._memory_item_view(item)}

    async def reject_memory_item(
        self,
        patient_id: PatientId,
        item_id: str,
        payload: MemoryItemVersionRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        item = self._call_memory_item(
            patient_id, item_id, "reject_memory_item", payload.expected_version, rejected_by="user"
        )
        self._audit_memory_item(request, patient_id, "memory_item_rejected", item)
        return {"success": True, "item": self._memory_item_view(item)}

    async def stop_memory_item(
        self,
        patient_id: PatientId,
        item_id: str,
        payload: MemoryItemVersionRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        item = self._call_memory_item(
            patient_id, item_id, "stop_memory_item", payload.expected_version, stopped_by="user"
        )
        self._audit_memory_item(request, patient_id, "memory_item_stopped", item)
        return {"success": True, "item": self._memory_item_view(item)}

    async def delete_memory_item(
        self,
        patient_id: PatientId,
        item_id: str,
        payload: MemoryItemDeleteRequest,
        request: Request,
    ):
        if not _enabled("ENABLE_PATIENT_MEMORY_DELETE"):
            raise HTTPException(status_code=403, detail="MEMORY_DELETE_DISABLED")
        patient_id = self._authorize(request, patient_id, "write")
        item = self._call_memory_item(
            patient_id,
            item_id,
            "delete_memory_item",
            payload.expected_version,
            payload.deletion_token,
            deleted_by="user",
        )
        self._audit_memory_item(
            request,
            patient_id,
            "memory_item_deleted",
            item,
        )
        return {"success": True, "item": self._memory_item_view(item)}

    async def get_patient_memory(
        self,
        patient_id: PatientId,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "read")
        memory = self._call_memory("get_memory_for_user", patient_id)
        return {"success": True, "data": memory}

    async def get_llm_context(
        self,
        patient_id: PatientId,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "read")
        context = self._call_memory("get_context_for_llm", patient_id)
        return {"success": True, "context": context}

    async def get_emotion_trajectory(
        self,
        patient_id: PatientId,
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
        session_id: str | None = Query(default=None, max_length=120),
    ):
        patient_id = self._authorize(request, patient_id, "read")
        trajectory = self._call_memory(
            "get_emotion_trajectory",
            patient_id,
            limit=limit,
            session_id=session_id,
        )
        return {"success": True, "data": trajectory}

    async def get_session_reflection(
        self,
        patient_id: PatientId,
        request: Request,
        session_id: str = Query(min_length=1, max_length=120),
    ):
        patient_id = self._authorize(request, patient_id, "read")
        reflection = self._call_memory(
            "get_session_reflection",
            patient_id,
            session_id.strip(),
        )
        data = self._session_reflection_view(reflection)
        if reflection.get("status") != "succeeded":
            previous = self._call_memory(
                "get_latest_session_reflection",
                patient_id,
                exclude_session_id=session_id.strip(),
            )
            data["previous_reflection"] = (
                self._session_reflection_view(previous)
                if previous.get("status") != "missing"
                else None
            )
        return {
            "success": True,
            "data": data,
        }

    async def requeue_session_reflection(
        self,
        patient_id: PatientId,
        request: Request,
        session_id: str = Query(min_length=1, max_length=120),
    ):
        patient_id = self._authorize(request, patient_id, "write")
        result = self._call_memory(
            "requeue_session_reflection", patient_id, session_id.strip(), reason="api_manual"
        )
        return {"success": bool(result), "data": {"status": "pending" if result else "missing"}}

    async def get_memory_worker_status(self, patient_id: PatientId, request: Request):
        patient_id = self._authorize(request, patient_id, "read")
        status = self._call_memory("get_memory_worker_status", patient_id)
        return {"success": True, "data": status}

    async def check_memobase_consistency(self, patient_id: PatientId, request: Request):
        patient_id = self._authorize(request, patient_id, "read")
        result = self._call_memory("check_memobase_consistency", patient_id)
        return {"success": True, "data": result}

    async def get_admin_memory_worker_status(self, request: Request):
        self._authorize_admin(request)
        status = self._call_memory("get_memory_worker_status")
        return {"success": True, "data": status}

    async def requeue_mirror_op(
        self,
        patient_id: PatientId,
        op_id: str,
        payload: MirrorOpRequeueRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        result = self._call_memory(
            "requeue_mirror_op", op_id, payload.reason, patient_id=patient_id
        )
        return {"success": bool(result), "data": {"status": "pending" if result else "missing"}}

    async def update_patient_memory(
        self,
        patient_id: PatientId,
        payload: MemoryUpdateRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        memory = self._call_memory(
            "update_memory_by_user",
            patient_id,
            payload.updates,
            expected_revision=payload.expected_revision,
        )
        return {"success": True, "data": memory}

    async def record_mmse(
        self,
        patient_id: PatientId,
        payload: MMSERecordRequest,
        request: Request,
    ):
        if not _enabled("ENABLE_COGNITIVE_SCREENING"):
            raise HTTPException(
                status_code=403,
                detail="COGNITIVE_SCREENING_DISABLED",
            )
        patient_id = self._authorize(request, patient_id, "write")
        kwargs = {"date": payload.date} if payload.date is not None else {}
        result = self._call_memory(
            "update_mmse_score",
            patient_id,
            payload.score,
            payload.weak_dimensions,
            **kwargs,
        )
        return {"success": True, "data": result}

    async def add_comfort_strategy(
        self,
        patient_id: PatientId,
        payload: ComfortStrategyRequest,
        request: Request,
    ):
        patient_id = self._authorize(request, patient_id, "write")
        result = self._call_memory(
            "add_comfort_strategy",
            patient_id,
            payload.strategy,
            payload.effective,
        )
        return {"success": True, "data": result}

    def _register_routes(self) -> None:
        routes = (
            (
                "/api/memory/{patient_id}/context",
                self.get_llm_context,
                ["GET"],
                "get_llm_context",
            ),
            (
                "/api/memory/{patient_id}/emotion-trajectory",
                self.get_emotion_trajectory,
                ["GET"],
                "get_emotion_trajectory",
            ),
            (
                "/api/memory/{patient_id}/session-reflection",
                self.get_session_reflection,
                ["GET"],
                "get_session_reflection",
            ),
            (
                "/api/memory/{patient_id}/session-reflection/requeue",
                self.requeue_session_reflection,
                ["POST"],
                "requeue_session_reflection",
            ),
            (
                "/api/memory/{patient_id}/worker-status",
                self.get_memory_worker_status,
                ["GET"],
                "get_memory_worker_status",
            ),
            (
                "/api/memory/{patient_id}/mirror-consistency",
                self.check_memobase_consistency,
                ["GET"],
                "check_memobase_consistency",
            ),
            (
                "/api/memory/{patient_id}/mirror-ops/{op_id}/requeue",
                self.requeue_mirror_op,
                ["POST"],
                "requeue_mirror_op",
            ),
            (
                "/api/admin/memory/worker-status",
                self.get_admin_memory_worker_status,
                ["GET"],
                "get_admin_memory_worker_status",
            ),
            (
                "/api/memory/{patient_id}/items",
                self.list_memory_items,
                ["GET"],
                "list_memory_items",
            ),
            (
                "/api/memory/{patient_id}/items/{item_id}",
                self.update_memory_item,
                ["PATCH"],
                "patch_memory_item",
            ),
            (
                "/api/memory/{patient_id}/items/{item_id}/confirm",
                self.confirm_memory_item,
                ["POST"],
                "confirm_memory_item",
            ),
            (
                "/api/memory/{patient_id}/items/{item_id}/reject",
                self.reject_memory_item,
                ["POST"],
                "reject_memory_item",
            ),
            (
                "/api/memory/{patient_id}/items/{item_id}/stop-using",
                self.stop_memory_item,
                ["POST"],
                "stop_memory_item",
            ),
            (
                "/api/memory/{patient_id}/items/{item_id}",
                self.delete_memory_item,
                ["DELETE"],
                "delete_memory_item",
            ),
            (
                "/api/memory/{patient_id}",
                self.get_patient_memory,
                ["GET"],
                "get_patient_memory",
            ),
            (
                "/api/memory/{patient_id}",
                self.update_patient_memory,
                ["PATCH"],
                "patch_patient_memory",
            ),
            (
                "/api/memory/{patient_id}",
                self.update_patient_memory,
                ["PUT"],
                "put_patient_memory",
            ),
            (
                "/api/memory/{patient_id}/mmse",
                self.record_mmse,
                ["POST"],
                "record_mmse",
            ),
            (
                "/api/memory/{patient_id}/comfort-strategies",
                self.add_comfort_strategy,
                ["POST"],
                "add_comfort_strategy",
            ),
        )
        for path, endpoint, methods, operation_id in routes:
            self.router.add_api_route(
                path,
                endpoint,
                methods=methods,
                tags=["Memory API"],
                operation_id=operation_id,
            )
