import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from fastapi import HTTPException

from src.context_management import ContextVariantRegistry
from src.web.public_api import (
    PublicApiController,
    PublicApiKeyService,
    PublicApiPatientProfile,
    PublicApiScreeningSession,
    PublicApiSessionCreateRequest,
    PublicApiSessionStore,
    PublicApiTurnRequest,
    PublicApiVisionRequest,
)


class _PublicApiRepository:
    def get_active_public_api_key_by_hash(self, _key_hash):
        return None

    def touch_public_api_key(self, _key_id):
        raise AssertionError("bootstrap key must not touch managed key")


class PublicApiKeyServiceTests(unittest.TestCase):
    def test_accepts_configured_bearer_key_and_rate_limits_it(self):
        current_time = [10.0]
        with TemporaryDirectory() as temp_dir:
            service = PublicApiKeyService(
                key_file=Path(temp_dir) / ".public_api_key",
                repository=_PublicApiRepository(),
                environ={
                    "PUBLIC_API_KEYS": "test-key",
                    "PUBLIC_API_RATE_LIMIT_PER_MINUTE": "1",
                },
                monotonic=lambda: current_time[0],
                logger=lambda _message: None,
            )
            request = SimpleNamespace(
                headers={"Authorization": "Bearer test-key"}
            )

            key_id = service.require(request)

            self.assertTrue(key_id.startswith("bootstrap:"))
            with self.assertRaises(HTTPException) as caught:
                service.require(request)
            self.assertEqual(caught.exception.status_code, 429)

            current_time[0] += 61
            self.assertEqual(service.require(request), key_id)

    def test_rejects_missing_key(self):
        with TemporaryDirectory() as temp_dir:
            service = PublicApiKeyService(
                key_file=Path(temp_dir) / ".public_api_key",
                repository=_PublicApiRepository(),
                environ={"PUBLIC_API_KEYS": "test-key"},
                logger=lambda _message: None,
            )
            request = SimpleNamespace(headers={})

            with self.assertRaises(HTTPException) as caught:
                service.require(request)

            self.assertEqual(caught.exception.status_code, 401)


class PublicApiSessionStoreTests(unittest.TestCase):
    def test_store_evicts_expired_session_before_capacity_check(self):
        async def scenario():
            current_time = [0.0]
            store = PublicApiSessionStore(
                environ={
                    "PUBLIC_API_SESSION_TTL_SECONDS": "300",
                    "PUBLIC_API_MAX_SESSIONS": "1",
                },
                monotonic=lambda: current_time[0],
            )
            first = PublicApiScreeningSession(
                agent=object(),
                patient_profile={},
                last_active_at=0.0,
            )
            await store.add("first", first)
            current_time[0] = 301.0
            second = PublicApiScreeningSession(
                agent=object(),
                patient_profile={},
                last_active_at=301.0,
            )

            await store.add("second", second)

            self.assertIsNone(await store.get("first"))
            self.assertIs(await store.get("second"), second)

        asyncio.run(scenario())


class PublicApiControllerTests(unittest.TestCase):
    def test_controller_registers_complete_external_api(self):
        with TemporaryDirectory() as temp_dir:
            controller = PublicApiController(
                key_service=PublicApiKeyService(
                    key_file=Path(temp_dir) / ".key",
                    repository=_PublicApiRepository(),
                    environ={"PUBLIC_API_KEYS": "test-key"},
                    logger=lambda _message: None,
                ),
                session_store=PublicApiSessionStore(),
                context_registry=ContextVariantRegistry(),
                agent_factory=lambda: object(),
                service_ready=lambda: True,
                create_session=lambda *_args, **_kwargs: None,
                static_dir=Path(temp_dir),
            )

        paths = {route.path for route in controller.router.routes}

        self.assertEqual(
            paths,
            {
                "/v1/health",
                "/v1/context/variants",
                "/developers",
                "/api-docs",
                "/v1/screening/sessions",
                "/v1/screening/sessions/{session_id}/turn",
                "/v1/vision/evaluate",
            },
        )

    def test_create_session_owns_agent_context_and_persistence(self):
        async def scenario():
            created = []
            agent = SimpleNamespace(
                tool_gateway=SimpleNamespace(memory_tool=object())
            )
            with TemporaryDirectory() as temp_dir:
                controller = PublicApiController(
                    key_service=PublicApiKeyService(
                        key_file=Path(temp_dir) / ".key",
                        repository=_PublicApiRepository(),
                        environ={"PUBLIC_API_KEYS": "test-key"},
                        logger=lambda _message: None,
                    ),
                    session_store=PublicApiSessionStore(),
                    context_registry=ContextVariantRegistry(),
                    agent_factory=lambda: agent,
                    service_ready=lambda: True,
                    create_session=lambda session_id, **kwargs: (
                        created.append((session_id, kwargs))
                    ),
                    static_dir=Path(temp_dir),
                )
                request = SimpleNamespace(
                    headers={"X-API-Key": "test-key"}
                )
                payload = PublicApiSessionCreateRequest(
                    patient=PublicApiPatientProfile(
                        name="张阿姨",
                        age=72,
                    )
                )

                result = await controller.create_screening_session(
                    payload,
                    request,
                )

            session_id = result["session_id"]
            state = await controller.session_store.get(session_id)
            self.assertIs(state.agent, agent)
            self.assertEqual(
                state.patient_profile,
                {"name": "张阿姨", "age": 72},
            )
            self.assertEqual(created[0][0], session_id)
            self.assertEqual(
                created[0][1]["profile"],
                {"name": "张阿姨", "age": 72},
            )
            self.assertTrue(
                created[0][1]["owner_username"].startswith(
                    "api:bootstrap:"
                )
            )
            self.assertEqual(created[0][1]["mode"], "cognitive_screening")

        asyncio.run(scenario())

    def test_cognitive_release_flag_blocks_screening_operations(self):
        async def scenario():
            with TemporaryDirectory() as temp_dir:
                store = PublicApiSessionStore()
                await store.add(
                    "session-1",
                    PublicApiScreeningSession(
                        agent=SimpleNamespace(
                            process_turn=lambda **_kwargs: {
                                "output": "should not run"
                            }
                        ),
                        patient_profile={},
                    ),
                )
                controller = PublicApiController(
                    key_service=PublicApiKeyService(
                        key_file=Path(temp_dir) / ".key",
                        repository=_PublicApiRepository(),
                        environ={"PUBLIC_API_KEYS": "test-key"},
                        logger=lambda _message: None,
                    ),
                    session_store=store,
                    context_registry=ContextVariantRegistry(),
                    agent_factory=lambda: (_ for _ in ()).throw(AssertionError()),
                    service_ready=lambda: (_ for _ in ()).throw(AssertionError()),
                    create_session=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError()
                    ),
                    static_dir=Path(temp_dir),
                )
                request = SimpleNamespace(headers={"X-API-Key": "test-key"})

                with mock.patch.dict(
                    "os.environ",
                    {"ENABLE_COGNITIVE_SCREENING": "false"},
                ):
                    calls = (
                        lambda: controller.create_screening_session(
                            PublicApiSessionCreateRequest(
                                patient=PublicApiPatientProfile(name="张阿姨")
                            ),
                            request,
                        ),
                        lambda: controller.screening_turn(
                            "session-1",
                            PublicApiTurnRequest(message="你好"),
                            request,
                        ),
                        lambda: controller.vision_evaluate(
                            PublicApiVisionRequest(
                                image="base64",
                                task_id="clock",
                            ),
                            request,
                        ),
                    )
                    for call in calls:
                        with self.assertRaises(HTTPException) as caught:
                            await call()
                        self.assertEqual(caught.exception.status_code, 403)
                        self.assertEqual(
                            caught.exception.detail,
                            "COGNITIVE_SCREENING_DISABLED",
                        )

        asyncio.run(scenario())

    def test_public_api_rejects_patient_memory_without_explicit_scope(self):
        async def scenario():
            created = []
            with TemporaryDirectory() as temp_dir:
                controller = PublicApiController(
                    key_service=PublicApiKeyService(
                        key_file=Path(temp_dir) / ".key",
                        repository=_PublicApiRepository(),
                        environ={"PUBLIC_API_KEYS": "test-key"},
                        logger=lambda _message: None,
                    ),
                    session_store=PublicApiSessionStore(),
                    context_registry=ContextVariantRegistry(),
                    agent_factory=lambda: (_ for _ in ()).throw(AssertionError()),
                    service_ready=lambda: True,
                    create_session=lambda *args, **kwargs: created.append(args),
                    static_dir=Path(temp_dir),
                )
                request = SimpleNamespace(headers={"X-API-Key": "test-key"})
                payload = PublicApiSessionCreateRequest(
                    patient=PublicApiPatientProfile(patient_id="patient-1")
                )

                with self.assertRaises(HTTPException) as caught:
                    await controller.create_screening_session(payload, request)

            self.assertEqual(caught.exception.status_code, 403)
            self.assertEqual(
                caught.exception.detail,
                "PUBLIC_API_PATIENT_SCOPE_REQUIRED",
            )
            self.assertEqual(created, [])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
