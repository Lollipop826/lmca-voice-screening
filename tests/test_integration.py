from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import HTTPException

from src.agents.screening.catalog import ScreeningTaskCatalog
from src.agents.screening.session_lifecycle import ScreeningSessionLifecycle
from src.agents.screening.state import ScreeningSessionState
from src.context_management.emotion_memobase import EmotionMemobase
from src.web.memory_api import MemoryApiController
from src.web.public_api import (
    PublicApiController,
    PublicApiKeyService,
    PublicApiPatientProfile,
    PublicApiSessionCreateRequest,
    PublicApiSessionStore,
)


class _FakeKeyRepository:
    def get_active_public_api_key_by_hash(self, _value):
        return None

    def touch_public_api_key(self, _value):
        raise AssertionError("bootstrap key must not touch repository")


def test_screening_lifecycle_can_sync_mmse_to_long_term_memory(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    lifecycle = ScreeningSessionLifecycle(
        state=ScreeningSessionState(),
        catalog=ScreeningTaskCatalog(),
        memory_tool_provider=lambda: None,
        reset_classifier=lambda: None,
    )
    lifecycle.set_memory_manager(memory)

    assert lifecycle.sync_mmse_score("pt-1", 24, ["recall"])
    assert memory.get_memory_for_user("pt-1")["mmse_history"][0]["score"] == 24


def test_memory_api_exposes_view_edit_mmse_and_comfort_routes(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))
    controller = MemoryApiController(memory)
    paths = {route.path for route in controller.router.routes}

    assert "/api/memory/{patient_id}" in paths
    assert "/api/memory/{patient_id}/context" in paths
    assert "/api/memory/{patient_id}/session-reflection" in paths
    assert "/api/memory/{patient_id}/items" in paths
    assert "/api/memory/{patient_id}/items/{item_id}" in paths
    assert "/api/memory/{patient_id}/items/{item_id}/confirm" in paths
    assert "/api/memory/{patient_id}/items/{item_id}/reject" in paths
    assert "/api/memory/{patient_id}/items/{item_id}/stop-using" in paths
    assert "/api/memory/{patient_id}/mmse" in paths
    assert "/api/memory/{patient_id}/comfort-strategies" in paths


def test_public_api_rejects_patient_memory_without_explicit_scope(tmp_path: Path):
    memory = EmotionMemobase(str(tmp_path / "memory.db"))

    with TemporaryDirectory() as temp_dir:
        controller = PublicApiController(
            key_service=PublicApiKeyService(
                key_file=Path(temp_dir) / ".key",
                repository=_FakeKeyRepository(),
                environ={"PUBLIC_API_KEYS": "test-key"},
                logger=lambda _message: None,
            ),
            session_store=PublicApiSessionStore(),
            context_registry=SimpleNamespace(
                resolve=lambda value: value or "baseline",
                install=lambda *args, **kwargs: None,
                available_variants=["baseline"],
                default_variant="baseline",
            ),
            agent_factory=lambda: None,
            service_ready=lambda: True,
            create_session=lambda *_args, **_kwargs: None,
            static_dir=Path(temp_dir),
            memory=memory,
        )

        async def scenario():
            request = SimpleNamespace(headers={"X-API-Key": "test-key"})
            try:
                await controller.create_screening_session(
                    PublicApiSessionCreateRequest(
                        patient=PublicApiPatientProfile(
                            patient_id="pt-1",
                            name="张阿姨",
                        )
                    ),
                    request,
                )
            except HTTPException as exc:
                assert exc.status_code == 403
                assert exc.detail == "PUBLIC_API_PATIENT_SCOPE_REQUIRED"
            else:
                raise AssertionError("public key must not access patient memory")

        asyncio.run(scenario())

    assert memory.get_context_for_llm("pt-1") == ""
