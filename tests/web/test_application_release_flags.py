import asyncio
from types import SimpleNamespace

from src.web.application import ApplicationHttpController, ApplicationLogBroker


def test_mmse_image_release_flag_blocks_static_asset(tmp_path):
    controller = ApplicationHttpController(
        auth=SimpleNamespace(current_user=lambda _request: {"username": "doctor"}),
        logs=ApplicationLogBroker(),
        static_dir=tmp_path,
        voice_calls_dir=tmp_path,
        get_agent=lambda: object(),
        environ={"ENABLE_COGNITIVE_SCREENING": "false"},
        logger=lambda _message: None,
    )

    response = asyncio.run(controller.mmse_image("clock"))

    assert response.status_code == 403
