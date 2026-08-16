import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.web import AuthController, AuthService


class _AuthRepository:
    def __init__(self):
        self.users = {
            "doctor": {
                "username": "doctor",
                "display_name": "王医生",
                "role": "user",
            }
        }

    def get_user_by_username(self, username):
        return self.users.get(username)


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.repository = _AuthRepository()
        self.auth = AuthService(
            repository=self.repository,
            data_dir=Path(self.temp_dir.name),
            logger=lambda _message: None,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_password_hash_and_signed_session_round_trip(self):
        password_hash, salt = self.auth.hash_password("secret123")

        self.assertTrue(
            self.auth.verify_password(
                "secret123",
                password_hash,
                salt,
            )
        )
        self.assertFalse(
            self.auth.verify_password(
                "wrong",
                password_hash,
                salt,
            )
        )

        token = self.auth.sign_session_token("doctor")
        self.assertEqual(
            self.auth.verify_session_token(token),
            "doctor",
        )
        self.assertEqual(
            self.auth.user_from_session_token(token)["username"],
            "doctor",
        )

    def test_current_user_exposes_only_public_identity(self):
        token = self.auth.sign_session_token("doctor")
        request = SimpleNamespace(
            cookies={self.auth.cookie_name: token}
        )

        user = self.auth.current_user(request)

        self.assertEqual(
            user,
            {
                "username": "doctor",
                "display_name": "王医生",
                "role": "user",
            },
        )

    def test_rejects_unsafe_post_login_paths(self):
        self.assertEqual(
            self.auth.sanitize_next_path("//example.com"),
            "/",
        )
        self.assertEqual(
            self.auth.sanitize_next_path("/api/admin/users"),
            "/",
        )
        self.assertEqual(
            self.auth.sanitize_next_path("/history"),
            "/history",
        )


class AuthControllerTests(unittest.TestCase):
    def test_controller_registers_existing_auth_and_admin_urls(self):
        with TemporaryDirectory() as temp_dir:
            repository = _AuthRepository()
            auth = AuthService(
                repository=repository,
                data_dir=Path(temp_dir),
                logger=lambda _message: None,
            )
            controller = AuthController(
                auth,
                static_dir=Path(temp_dir),
                repository=repository,
            )

        route_methods = {
            (
                route.path,
                tuple(sorted(route.methods or [])),
            )
            for route in controller.router.routes
        }

        self.assertIn(("/login", ("GET",)), route_methods)
        self.assertIn(("/api/auth/login", ("POST",)), route_methods)
        self.assertIn(("/api/admin/users", ("GET",)), route_methods)
        self.assertIn(("/api/admin/users", ("POST",)), route_methods)
        self.assertIn(("/history", ("GET",)), route_methods)
        self.assertIn(("/legacy", ("GET",)), route_methods)

    def test_guard_keeps_public_paths_and_rejects_private_api(self):
        async def scenario():
            with TemporaryDirectory() as temp_dir:
                auth = AuthService(
                    repository=_AuthRepository(),
                    data_dir=Path(temp_dir),
                    logger=lambda _message: None,
                )
                controller = AuthController(
                    auth,
                    static_dir=Path(temp_dir),
                    repository=_AuthRepository(),
                )
                called = []

                async def call_next(_request):
                    called.append(True)
                    return "ok"

                public_request = SimpleNamespace(
                    url=SimpleNamespace(path="/login"),
                    cookies={},
                )
                private_request = SimpleNamespace(
                    url=SimpleNamespace(path="/api/sessions"),
                    cookies={},
                )

                public_result = await controller.guard(
                    public_request,
                    call_next,
                )
                private_result = await controller.guard(
                    private_request,
                    call_next,
                )

                self.assertEqual(public_result, "ok")
                self.assertEqual(called, [True])
                self.assertEqual(private_result.status_code, 401)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
