import asyncio
from types import SimpleNamespace
import unittest

from src.voice.application import VoiceEndpointApplication


class _FakeTransport:
    transport_name = "WebSocket"

    def __init__(self, token="token"):
        self.cookies = {"aa_session": token}
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed = True
        self.close_code = code


class _FakeAuth:
    cookie_name = "aa_session"

    def __init__(self, user):
        self.user = user

    def user_from_session_token(self, _token):
        return self.user


class _FakeModels:
    def __init__(self, *, agent=None, error=None):
        self.agent = agent
        self.error = error
        self.create_count = 0

    def create_agent(self):
        self.create_count += 1
        if self.error:
            raise self.error
        return self.agent


def _build_application(*, user, models):
    return VoiceEndpointApplication(
        auth=_FakeAuth(user),
        models=models,
        recognition=object(),
        patient_memory_service=object(),
        config=SimpleNamespace(
            enable_full_duplex=True,
            use_soulx_turn_taking=False,
            soulx_turn_url="ws://127.0.0.1:8000/turn",
        ),
        logger=lambda _message: None,
    )


class VoiceEndpointApplicationTests(unittest.TestCase):
    def test_unauthenticated_transport_is_rejected_before_accept(self):
        async def scenario():
            transport = _FakeTransport(token="invalid")
            models = _FakeModels(agent=object())
            application = _build_application(
                user=None,
                models=models,
            )

            await application.handle(transport)

            self.assertFalse(transport.accepted)
            self.assertTrue(transport.closed)
            self.assertEqual(transport.close_code, 1008)
            self.assertEqual(models.create_count, 0)

        asyncio.run(scenario())

    def test_agent_creation_failure_closes_accepted_connection(self):
        async def scenario():
            transport = _FakeTransport()
            models = _FakeModels(error=RuntimeError("agent unavailable"))
            application = _build_application(
                user={"username": "doctor-a"},
                models=models,
            )

            await application.handle(transport)

            self.assertTrue(transport.accepted)
            self.assertTrue(transport.closed)
            self.assertEqual(transport.close_code, 1011)
            self.assertEqual(models.create_count, 1)

        asyncio.run(scenario())

    def test_connection_owns_agent_and_always_runs_cleanup(self):
        async def scenario():
            transport = _FakeTransport()
            agent = object()
            models = _FakeModels(agent=agent)
            application = _build_application(
                user={"username": "doctor-a"},
                models=models,
            )
            captured = {}

            class Controller:
                async def run(self):
                    captured["controller_ran"] = True

            class Cleanup:
                async def close(self, client_id):
                    captured["cleanup_client_id"] = client_id

            async def build_components(
                connection,
                session,
                client_id,
                authenticated_user,
            ):
                captured["connection"] = connection
                captured["session"] = session
                captured["client_id"] = client_id
                captured["authenticated_user"] = authenticated_user
                return Controller(), Cleanup()

            application._build_components = build_components

            await application.handle(transport)

            self.assertTrue(transport.accepted)
            self.assertIs(captured["session"].agent, agent)
            self.assertEqual(
                captured["session"].owner_username,
                "doctor-a",
            )
            self.assertEqual(
                captured["authenticated_user"],
                {"username": "doctor-a"},
            )
            self.assertTrue(captured["controller_ran"])
            self.assertEqual(
                captured["cleanup_client_id"],
                captured["client_id"],
            )

        asyncio.run(scenario())

    def test_component_build_failure_closes_connection(self):
        async def scenario():
            transport = _FakeTransport()
            application = _build_application(
                user={"username": "doctor-a"},
                models=_FakeModels(agent=object()),
            )

            async def build_components(*_args):
                raise RuntimeError("wiring failed")

            application._build_components = build_components
            await application.handle(transport)

            self.assertTrue(transport.accepted)
            self.assertTrue(transport.closed)
            self.assertEqual(transport.close_code, 1011)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
