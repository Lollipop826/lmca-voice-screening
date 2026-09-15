import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src.web import (
    ApplicationHttpController,
    ApplicationLogBroker,
    AuthService,
)


class _Repository:
    def get_user_by_username(self, username):
        return {"username": username, "role": "admin"}

    def list_accessible_patients(self, **_kwargs):
        return [{"patient_id": "pt-1", "name": "张阿姨"}]


class ApplicationLogBrokerTests(unittest.TestCase):
    def test_snapshot_is_bounded_and_subscription_receives_new_lines(self):
        broker = ApplicationLogBroker(max_lines=2)
        broker.append("first")
        broker.append("second")
        queue, backlog = broker.subscribe(backlog_size=10)

        broker.append("third")

        self.assertEqual(len(broker.snapshot()), 2)
        self.assertTrue(broker.snapshot()[0].endswith("second"))
        self.assertEqual(len(backlog), 2)
        self.assertTrue(
            asyncio.run(queue.get()).endswith("third")
        )
        broker.unsubscribe(queue)


class ApplicationHttpControllerTests(unittest.TestCase):
    def _build_controller(self, temp_dir: str):
        auth = AuthService(
            repository=_Repository(),
            data_dir=Path(temp_dir) / "data",
            logger=lambda _message: None,
        )
        return ApplicationHttpController(
            auth=auth,
            logs=ApplicationLogBroker(),
            static_dir=Path(temp_dir) / "static",
            voice_calls_dir=Path(temp_dir) / "voice_calls",
            get_agent=lambda: SimpleNamespace(
                tool_gateway=SimpleNamespace(
                    memory_tool=SimpleNamespace(
                        get_snapshot=lambda: {"ok": True}
                    )
                )
            ),
            repository=_Repository(),
            environ={},
            logger=lambda _message: None,
        )

    def test_controller_registers_ui_session_and_operational_routes(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)

        route_methods = {}
        for route in controller.router.routes:
            route_methods.setdefault(route.path, set()).update(
                route.methods or []
            )

        self.assertEqual(route_methods["/"], {"GET"})
        self.assertEqual(
            route_methods["/api/location"],
            {"GET", "POST"},
        )
        self.assertEqual(
            route_methods["/api/model"],
            {"GET", "POST"},
        )
        self.assertEqual(
            route_methods[
                "/api/sessions/{session_id}/export/csv"
            ],
            {"GET"},
        )
        self.assertEqual(
            route_methods["/api/vision-evaluate"],
            {"POST"},
        )
        self.assertEqual(route_methods["/api/patients"], {"GET"})

    def test_list_patients_returns_authenticated_records(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)
            request = SimpleNamespace(
                cookies={
                    controller.auth.cookie_name:
                    controller.auth.sign_session_token("admin")
                }
            )

            result = asyncio.run(controller.list_patients(request))

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["patients"][0]["patient_id"], "pt-1")

    def test_model_status_prefers_dashscope_over_configured_ark_fallback(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)
            controller.environ.update({
                "DASHSCOPE_API_KEY": "test-dashscope-key",
                "DASHSCOPE_CHAT_MODEL": "qwen3.7-flash",
                "ARK_API_KEY": "test-ark-key",
            })
            result = asyncio.run(controller.get_model())

        self.assertEqual(result, {
            "model": "qwen3.7-flash",
            "options": ["qwen3.7-flash"],
            "provider": "dashscope",
        })

    def test_dashscope_model_switch_cannot_report_a_false_ark_switch(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)
            controller.environ.update({
                "DASHSCOPE_API_KEY": "test-dashscope-key",
                "ARK_API_KEY": "test-ark-key",
            })
            request = SimpleNamespace(json=AsyncMock(return_value={
                "model": "doubao-seed-2-0-lite-260215",
            }))
            with patch("src.llm.http_client_pool.switch_ark_model") as switch:
                result = asyncio.run(controller.switch_model(request))
                switch.assert_not_called()

            self.assertEqual(result.status_code, 400)
            self.assertEqual(asyncio.run(controller.get_model())["model"], "qwen3.7-flash")

    def test_ark_model_switch_still_uses_ark_when_dashscope_is_not_configured(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)
            controller.environ["ARK_API_KEY"] = "test-ark-key"
            selected = "doubao-seed-2-0-lite-260215"
            request = SimpleNamespace(json=AsyncMock(return_value={"model": selected}))
            with patch("src.llm.http_client_pool.switch_ark_model", return_value=selected) as switch:
                result = asyncio.run(controller.switch_model(request))
                switch.assert_called_once_with(selected)

        self.assertTrue(result["success"])
        self.assertEqual(result["model"], selected)
        self.assertEqual(result["provider"], "volcengine")

    def test_managed_audio_path_cannot_escape_voice_call_directory(self):
        with TemporaryDirectory() as temp_dir:
            controller = self._build_controller(temp_dir)
            voice_dir = Path(temp_dir) / "voice_calls"
            voice_dir.mkdir()
            inside = voice_dir / "audio.wav"
            inside.write_bytes(b"audio")
            outside = Path(temp_dir) / "outside.wav"
            outside.write_bytes(b"audio")

            self.assertEqual(
                controller._resolve_managed_audio_path(inside),
                inside.resolve(),
            )
            self.assertIsNone(
                controller._resolve_managed_audio_path(outside)
            )

    def test_reverse_location_parser_normalizes_amap_shapes(self):
        location = ApplicationHttpController._parse_reverse_location(
            {
                "status": "1",
                "regeocode": {
                    "formatted_address": "上海市徐汇区某路",
                    "addressComponent": {
                        "province": "上海市",
                        "city": [],
                        "district": "徐汇区",
                        "adcode": "310104",
                        "neighborhood": {"name": "某小区"},
                        "streetNumber": {"street": "某路"},
                    },
                },
            }
        )

        self.assertEqual(location["province"], "上海市")
        self.assertEqual(location["city"], "")
        self.assertEqual(location["neighborhood"], "某小区")
        self.assertEqual(location["source"], "amap-gps")


if __name__ == "__main__":
    unittest.main()
