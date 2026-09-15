from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import json
import math
import time

class ClientControlHandler:
    """Handle lightweight connection controls that do not belong to a session turn."""

    MESSAGE_TYPES = {"ping", "client_diag", "update_location"}

    _NOISY_TTS_LABELS = (
        "unlock:html-silent-play-failed",
        "unlock:html-silent-play-ok",
        "player:source:onended",
        "player:playBuffer:start",
        "player:addChunk:webAudio-scheduled",
        "worklet:chunk-posted",
        "scriptProcessor:chunk-queued",
        "ws:tts_chunk:received",
        "ws:tts_chunk:queued",
    )

    def __init__(
        self,
        connection,
        *,
        config_path: str | Path = "config/deployment.json",
        reverse_geocode: Callable[[Any, Any], tuple[str, str]] | None = None,
        location_loader: Callable[[], dict | None] | None = None,
        cache_updater: Callable[[dict], None] | None = None,
        logger=print,
    ) -> None:
        self.connection = connection
        self.config_path = Path(config_path)
        self._reverse_geocode = reverse_geocode or self._default_reverse_geocode
        self._location_loader = location_loader or self._default_location_loader
        self._cache_updater = cache_updater or self._default_cache_updater
        self._log = logger
        self._handlers = {
            "ping": self._handle_ping,
            "client_diag": self._handle_client_diag,
            "update_location": self._handle_update_location,
        }

    async def handle_message(self, message: dict) -> bool:
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            return False
        await handler(message)
        return True

    async def _handle_ping(self, _message: dict) -> None:
        await self.connection.send_json({"type": "pong"})

    async def _handle_client_diag(self, message: dict) -> None:
        area = str(message.get("area") or "client")[:32]
        label = str(message.get("label") or "-")[:96]
        seq = message.get("seq")
        detail = str(message.get("detail") or "")[:1200]
        if area == "tts" and any(
            label.startswith(prefix) for prefix in self._NOISY_TTS_LABELS
        ):
            return
        detail_preview = " ".join(detail.split())[:320]
        if area == "tts" and label == "playback-stopped":
            try:
                payload = json.loads(detail)
            except (TypeError, ValueError):
                payload = {}
            if isinstance(payload, dict):
                timings = {
                    key: value for key, value in payload.items()
                    if key in {
                        "server_stop_at_ms", "client_received_at_ms",
                        "client_stopped_at_ms", "stop_handler_ms", "generation",
                    }
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                }
                timings["server_ack_at_ms"] = round(time.time() * 1000, 1)
                self._log(f"[全双工停播确认] {json.dumps(timings)}")
            return
        if area == "tts" and (
            "exception" in label.casefold()
            or "failed" in label.casefold()
        ):
            self._log(
                f"[CLIENT-DIAG][{area}] #{seq} {label} | "
                f"detail={detail_preview or '-'}"
            )
            return
        self._log(
            f"[CLIENT-DIAG][{area}] #{seq} {label} | "
            f"detail_chars={len(detail)}"
        )

    async def _handle_update_location(self, message: dict) -> None:
        latitude = message.get("latitude")
        longitude = message.get("longitude")
        if not latitude or not longitude:
            return

        self._log("[位置] 🌍 收到浏览器位置更新")
        city_name = message.get("city")
        province_name = message.get("province")
        self._log(
            "[位置] 📦 浏览器位置字段: "
            f"city_present={bool(city_name)} province_present={bool(province_name)}"
        )

        if not city_name:
            self._log("[位置] ⚠️ 浏览器未传城市名，尝试反向编码...")
            try:
                province_name, city_name = self._reverse_geocode(
                    latitude,
                    longitude,
                )
                self._log("[位置] 🗺️ Nominatim反向编码成功")
            except Exception as exc:
                self._log(f"[位置] ⚠️ 反向地理编码失败: {type(exc).__name__}")
        else:
            self._log("[位置] ✅ 使用浏览器传来的城市信息")

        if self.config_path.exists():
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            config["location"]["lat"] = latitude
            config["location"]["lon"] = longitude
            if province_name:
                config["location"]["province"] = province_name
            if city_name:
                config["location"]["city"] = city_name
            config["location"]["source"] = "browser-geolocation"
            self.config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )

        updated_location = self._location_loader()
        if updated_location:
            self._cache_updater(updated_location)
            self._log("[位置] ✅ 已更新并刷新缓存")
        else:
            self._log("[位置] ⚠️ 配置文件读取失败")

    @staticmethod
    def _default_reverse_geocode(
        latitude,
        longitude,
    ) -> tuple[str, str]:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "format": "json",
                    "accept-language": "zh",
                },
                headers={"User-Agent": "ADScreeningApp/1.0"},
            )
            if response.status_code != 200:
                return "", ""
            address = response.json().get("address", {})
            province = address.get("state", address.get("province", ""))
            city = address.get(
                "city",
                address.get("county", address.get("town", "")),
            )
            return province, city

    @staticmethod
    def _default_location_loader() -> dict | None:
        from src.utils.location_service import get_location_from_config

        return get_location_from_config()

    @staticmethod
    def _default_cache_updater(location: dict) -> None:
        import src.utils.location_service as location_service

        location_service._cached_location = location
