from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from src.db import database
from src.voice.media import resample_audio

from .auth import AuthService
from .page_responses import message_page


def _cognitive_screening_enabled(environ=os.environ) -> bool:
    return environ.get("ENABLE_COGNITIVE_SCREENING", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _CapturedStream:
    def __init__(self, original, broker: "ApplicationLogBroker"):
        self._original = original
        self._broker = broker
        self._pending = ""

    def write(self, data):
        try:
            self._original.write(data)
        except Exception:
            pass
        if not data:
            return len(data) if isinstance(data, str) else 0
        try:
            self._pending += data
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._broker.append(line.rstrip("\r"))
        except Exception:
            pass
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._original, name)


class ApplicationLogBroker:
    """Capture process output and fan it out to admin SSE subscribers."""

    def __init__(self, *, max_lines: int = 3000) -> None:
        self._buffer: deque[str] = deque(maxlen=max_lines)
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()

    def install_stdio(self) -> None:
        if not isinstance(sys.stdout, _CapturedStream):
            sys.stdout = _CapturedStream(sys.stdout, self)
        if not isinstance(sys.stderr, _CapturedStream):
            sys.stderr = _CapturedStream(sys.stderr, self)

    def append(self, line: str) -> None:
        if not line:
            return
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {line}"
        with self._lock:
            self._buffer.append(stamped)
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(stamped)
            except Exception:
                pass

    def snapshot(self, limit: int = 500) -> list[str]:
        with self._lock:
            return list(self._buffer)[-max(1, min(limit, 3000)) :]

    def subscribe(
        self,
        *,
        backlog_size: int = 300,
        queue_size: int = 2000,
    ) -> tuple[asyncio.Queue, list[str]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        with self._lock:
            backlog = list(self._buffer)[-backlog_size:]
            self._subscribers.append(queue)
        return queue, backlog

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    async def iter_events(
        self,
        request: Request,
        queue: asyncio.Queue,
        backlog: list[str],
    ):
        try:
            for line in backlog:
                yield self._event(line)
            while True:
                if await request.is_disconnected():
                    return
                try:
                    line = await asyncio.wait_for(
                        queue.get(),
                        timeout=15.0,
                    )
                    yield self._event(line)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            self.unsubscribe(queue)

    @staticmethod
    def _event(line: str) -> str:
        payload = json.dumps(
            {"line": line},
            ensure_ascii=False,
        )
        return f"data: {payload}\n\n"


class ApplicationHttpController:
    """Serve the authenticated application UI and operational HTTP APIs."""

    _NO_CACHE_HEADERS = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }

    def __init__(
        self,
        *,
        auth: AuthService,
        logs: ApplicationLogBroker,
        static_dir: Path,
        voice_calls_dir: Path,
        get_agent: Callable[[], Any],
        memory_service: Any = None,
        repository=database,
        environ=None,
        logger=print,
    ) -> None:
        self.auth = auth
        self.logs = logs
        self.static_dir = Path(static_dir)
        self.voice_calls_dir = Path(voice_calls_dir).resolve()
        self._get_agent = get_agent
        self.memory_service = memory_service
        self.repository = repository
        self.environ = environ if environ is not None else os.environ
        self._log = logger
        self.router = APIRouter()
        self._register_routes()

    def install(self, app) -> None:
        app.include_router(self.router)

    async def open_three_step_action_demo(self):
        return RedirectResponse(
            url="/static/test_3step_action_demo.html",
            status_code=307,
        )

    async def open_test_pages(self):
        return RedirectResponse(
            url="/static/test_camera.html",
            status_code=307,
        )

    async def logs_snapshot(
        self,
        request: Request,
        limit: int = 500,
    ):
        self.auth.require_admin_user(request)
        return {"lines": self.logs.snapshot(limit)}

    async def logs_stream(self, request: Request):
        self.auth.require_admin_user(request)
        queue, backlog = self.logs.subscribe()
        return StreamingResponse(
            self.logs.iter_events(request, queue, backlog),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def list_sessions(
        self,
        request: Request,
        limit: int = 50,
        offset: int = 0,
        name: str = None,
    ):
        user = self.auth.require_authenticated_user(request)
        try:
            sessions = self.repository.list_sessions(
                limit=limit,
                offset=offset,
                name_filter=name,
                actor_username=str(user.get("username") or ""),
                is_admin=self.auth.is_admin_user(user),
            )
            return {
                "success": True,
                "sessions": sessions,
                "count": len(sessions),
            }
        except Exception as exc:
            return self._error(str(exc), 500)

    async def list_patients(
        self,
        request: Request,
        q: str = "",
        limit: int = 100,
    ):
        user = self.auth.require_authenticated_user(request)
        try:
            patients = self.repository.list_accessible_patients(
                query=q,
                limit=limit,
                actor_username=str(user.get("username") or ""),
                is_admin=self.auth.is_admin_user(user),
            )
            return {
                "success": True,
                "patients": patients,
                "count": len(patients),
            }
        except Exception as exc:
            return self._error(str(exc), 500)

    async def active_sessions(self, request: Request):
        self.auth.require_admin_user(request)
        try:
            return {
                "success": True,
                "sessions": self.repository.get_active_sessions(),
            }
        except Exception as exc:
            return self._error(str(exc), 500)

    async def get_session(
        self,
        request: Request,
        session_id: str,
    ):
        self.auth.require_session_access(request, session_id)
        try:
            detail = self.repository.get_session_detail(session_id)
            if not detail:
                return self._error("会话不存在", 404)
            detail["audio_files"] = [
                self._build_session_audio_payload(session_id, audio)
                for audio in self.repository.list_audio_records(
                    session_id
                )
            ]
            detail["exports"] = {
                "merged_audio_url": (
                    f"/api/sessions/{session_id}/"
                    "export/audio-merged.wav"
                ),
                "audio_clips_zip_url": (
                    f"/api/sessions/{session_id}/"
                    "export/audio-clips.zip"
                ),
            }
            return {"success": True, **detail}
        except Exception as exc:
            return self._error(str(exc), 500)

    async def delete_session(
        self,
        request: Request,
        session_id: str,
    ):
        self.auth.require_admin_user(request)
        try:
            session_id = str(session_id or "").strip()
            detail = self.repository.get_session_detail(session_id)
            if not detail:
                return self._error("会话不存在", 404)
            patient_id = str((detail.get("session") or {}).get("patient_id") or "").strip()
            memory_cleanup = None
            agent = self._get_agent()
            memory_service = self.memory_service or getattr(agent, "patient_memory_service", None)
            memory = getattr(memory_service, "_long_term_memory", None)
            if memory is None:
                memory = memory_service if callable(getattr(memory_service, "delete_session_data", None)) else None
            if patient_id and memory is not None:
                cleanup = getattr(memory, "delete_session_data", None)
                if callable(cleanup):
                    memory_cleanup = cleanup(session_id)
            result = self.repository.delete_session(session_id)
            if not result:
                return self._error("会话不存在", 404)

            file_cleanup = self._cleanup_session_files(session_id)
            cleanup_error = file_cleanup.get("error")
            if cleanup_error:
                self._log(
                    f"[SessionDelete] ⚠️ 会话 {session_id} 数据已删，"
                    f"文件清理失败: {cleanup_error}"
                )

            response = {
                "success": True,
                "session_id": str(session_id),
                "deleted_counts": result.get("deleted_counts", {}),
                "memory_cleanup": memory_cleanup,
                "file_cleanup": "partial" if cleanup_error else "done",
                "file_cleanup_counts": file_cleanup.get("removed", {}),
            }
            if cleanup_error:
                response["file_cleanup_error"] = cleanup_error
            self._log(f"[SessionDelete] ✅ 已删除会话 {session_id}")
            return response
        except ValueError as exc:
            if str(exc) == "SESSION_ACTIVE":
                return self._error("会话进行中，结束后才能删除", 409)
            return self._error(str(exc), 400)
        except Exception as exc:
            return self._error(str(exc), 500)

    def _cleanup_session_files(self, session_id: str) -> dict[str, Any]:
        """删除会话目录及旧版 session sidecar，始终限制在 data 根目录内。"""
        session_id = str(session_id or "").strip()
        if not session_id or Path(session_id).name != session_id or session_id in {".", ".."}:
            raise ValueError("会话标识无效")

        data_dir = self.voice_calls_dir.parent
        targets = {
            "voice_calls": self.voice_calls_dir / session_id,
            "conversations": data_dir / "conversations" / f"{session_id}.json",
            "mmse_scores": data_dir / "mmse_scores" / f"{session_id}_mmse.json",
            "cognitive_performance": data_dir / "cognitive_performance" / f"{session_id}_performance.json",
        }
        removed: dict[str, int] = {}
        errors: list[str] = []
        for kind, target in targets.items():
            try:
                resolved = target.resolve()
                root = self.voice_calls_dir if kind == "voice_calls" else data_dir
                resolved.relative_to(root)
                if target.is_symlink():
                    raise ValueError(f"拒绝删除符号链接: {target}")
                if target.is_dir():
                    shutil.rmtree(target)
                    removed[kind] = 1
                elif target.is_file():
                    target.unlink()
                    removed[kind] = 1
            except Exception as exc:
                errors.append(f"{kind}: {exc}")
        return {
            "removed": removed,
            "error": "; ".join(errors) if errors else None,
        }

    async def get_session_audio(
        self,
        request: Request,
        session_id: str,
        audio_id: int,
        download: bool = False,
    ):
        self.auth.require_session_access(request, session_id)
        try:
            audio = self.repository.get_audio_record(
                session_id,
                audio_id,
            )
            if not audio:
                return self._error("录音不存在", 404)
            audio_path = self._resolve_managed_audio_path(
                audio.get("file_path")
            )
            if not audio_path:
                return self._error(
                    "录音文件不存在或不在受控目录",
                    404,
                )
            if download:
                return FileResponse(
                    str(audio_path),
                    media_type="audio/wav",
                    filename=audio_path.name,
                )
            return FileResponse(
                str(audio_path),
                media_type="audio/wav",
            )
        except Exception as exc:
            return self._error(str(exc), 500)

    async def export_merged_audio(
        self,
        request: Request,
        session_id: str,
    ):
        self.auth.require_session_access(request, session_id)
        try:
            detail = self.repository.get_session_detail(session_id)
            if not detail:
                return self._error("会话不存在", 404)
            merged = self._build_merged_session_audio(session_id)
            if not merged:
                return self._error(
                    "当前会话暂无可导出的音频",
                    404,
                )
            audio_array, sample_rate, _rows = merged
            output = io.BytesIO()
            sf.write(
                output,
                audio_array,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            patient_name = detail["session"].get("patient_name")
            filename = (
                f"会话总音频_{self._safe_download_name(patient_name)}_"
                f"{session_id[:16]}.wav"
            )
            return Response(
                content=output.getvalue(),
                media_type="audio/wav",
                headers=self._download_headers(filename),
            )
        except Exception as exc:
            return self._error(str(exc), 500)

    async def export_audio_clips(
        self,
        request: Request,
        session_id: str,
    ):
        self.auth.require_session_access(request, session_id)
        try:
            detail = self.repository.get_session_detail(session_id)
            if not detail:
                return self._error("会话不存在", 404)
            archive = self._build_audio_clips_zip(session_id)
            if not archive:
                return self._error(
                    "当前会话暂无可导出的音频片段",
                    404,
                )
            patient_name = detail["session"].get("patient_name")
            filename = (
                f"音频片段_{self._safe_download_name(patient_name)}_"
                f"{session_id[:16]}.zip"
            )
            return Response(
                content=archive,
                media_type="application/zip",
                headers=self._download_headers(filename),
            )
        except Exception as exc:
            return self._error(str(exc), 500)

    async def export_session_csv(
        self,
        request: Request,
        session_id: str,
    ):
        self.auth.require_session_access(request, session_id)
        try:
            detail = self.repository.get_session_detail(session_id)
            if not detail:
                return self._error("会话不存在", 404)
            output = io.StringIO()
            output.write("\ufeff")
            writer = csv.writer(output)
            session_info = detail.get("session", {})
            writer.writerow(
                ["患者姓名", session_info.get("patient_name", "未知")]
            )
            writer.writerow(["年龄", session_info.get("patient_age", "")])
            writer.writerow(
                ["性别", session_info.get("patient_gender", "")]
            )
            writer.writerow(
                ["会话时间", session_info.get("created_at", "")]
            )
            writer.writerow(
                ["结束时间", session_info.get("ended_at", "")]
            )
            writer.writerow([])
            self._write_scores(writer, detail.get("mmse_scores", []))
            writer.writerow([])
            writer.writerow(["=== 对话记录 ==="])
            writer.writerow(["序号", "角色", "内容", "时间"])
            for index, message in enumerate(
                detail.get("messages", []),
                start=1,
            ):
                role = (
                    "患者"
                    if message.get("role") == "user"
                    else "AI助手"
                )
                writer.writerow(
                    [
                        index,
                        role,
                        message.get("content", ""),
                        message.get("created_at", ""),
                    ]
                )
            patient_name = session_info.get("patient_name", "未知")
            filename = (
                f"评估记录_{patient_name}_{session_id[:16]}.csv"
            )
            return Response(
                content=output.getvalue(),
                media_type="text/csv; charset=utf-8",
                headers=self._download_headers(filename),
            )
        except Exception as exc:
            return self._error(str(exc), 500)

    async def index(self, request: Request):
        if not self.auth.current_user(request):
            return RedirectResponse(
                url="/login?next=/",
                status_code=307,
            )
        return self._static_html(
            "voice_chat.html",
            "语音页面未找到",
        )

    async def mmse_image(self, image_id: str):
        if not _cognitive_screening_enabled(self.environ):
            return JSONResponse(
                {"error": "COGNITIVE_SCREENING_DISABLED"},
                status_code=403,
            )
        safe_id = image_id.replace("/", "").replace("..", "")
        image_path = (
            self.static_dir / "mmse_images" / f"{safe_id}.png"
        )
        if not image_path.exists():
            return JSONResponse(
                {"error": f"图片不存在: {safe_id}"},
                status_code=404,
            )
        return FileResponse(
            str(image_path),
            media_type="image/png",
        )

    async def memory_snapshot(self):
        try:
            return JSONResponse(
                self._get_agent().tool_gateway.memory_tool.get_snapshot()
            )
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
            )

    async def get_location(self, request: Request):
        from src.utils import location_service

        client_ip = (
            request.headers.get("X-Forwarded-For", "")
            .split(",")[0]
            .strip()
        )
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")
        if not client_ip and request.client:
            client_ip = request.client.host
        location = await self._locate_client_ip(client_ip)
        if location:
            self._save_location(location_service, location)
            context = location_service.get_realtime_context()
            return {
                "success": True,
                "location": location,
                "weather": context.get("weather"),
                "client_ip": client_ip,
            }
        return {
            "success": True,
            "location": location_service.get_deployment_location(),
            "weather": None,
            "client_ip": client_ip,
            "note": "fallback_to_server_ip",
        }

    async def set_location(self, request: Request):
        from src.utils import location_service

        body = await request.json()
        latitude = body.get("lat")
        longitude = body.get("lon")
        amap_key = self.environ.get("AMAP_KEY", "")
        if not (latitude and longitude and amap_key):
            return self._error("缺少 lat/lon 或 AMAP_KEY", 400)
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    "https://restapi.amap.com/v3/geocode/regeo",
                    params={
                        "key": amap_key,
                        "location": f"{longitude},{latitude}",
                    },
                )
            location = self._parse_reverse_location(response.json())
            if location:
                self._save_location(location_service, location)
                context = location_service.get_realtime_context()
                self._log(
                    "[LocationAPI] ✅ GPS定位: "
                    f"{location['province']} {location['city']} "
                    f"{location['district']} "
                    f"{location.get('neighborhood', '')} "
                    f"({location.get('formatted_address', '')})"
                )
                return {
                    "success": True,
                    "location": location,
                    "weather": context.get("weather"),
                }
        except Exception as exc:
            self._log(f"[LocationAPI] ⚠️ 逆地理编码失败: {exc}")
            return self._error(str(exc), 500)
        return self._error("逆地理编码无结果", 500)

    async def get_model(self):
        from src.llm.http_client_pool import (
            _ARK_MODEL_OPTIONS,
            get_active_ark_model,
        )

        if self.environ.get("DASHSCOPE_API_KEY"):
            model = self.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.7-flash")
            return {"model": model, "options": [model], "provider": "dashscope"}
        if not self.environ.get("ARK_API_KEY"):
            model = self.environ.get(
                "TOPIC_SELECTION_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
            )
            return {"model": model, "options": [model], "provider": "siliconflow"}
        return {
            "model": get_active_ark_model(),
            "options": _ARK_MODEL_OPTIONS,
            "provider": "volcengine",
        }

    async def switch_model(self, request: Request):
        from src.llm.http_client_pool import switch_ark_model

        try:
            body = await request.json()
            current = await self.get_model()
            requested = body.get("model", "")
            if current["provider"] == "volcengine":
                result = switch_ark_model(requested)
            elif requested == current["model"]:
                result = requested
            else:
                raise ValueError(f"当前对话使用 {current['model']}，暂不支持切换到该模型")
            return {
                **current,
                "success": True,
                "model": result,
            }
        except ValueError as exc:
            return self._error(str(exc), 400)

    async def vision_evaluate(self, request: Request):
        from src.tools.agent_tools.vision_evaluation_tool import (
            evaluate_hybrid,
            evaluate_image_with_vlm,
        )

        try:
            body = await request.json()
            video = body.get("video", "")
            image = body.get("image", "")
            frames = body.get("frames", [])
            if video or frames:
                result = await asyncio.to_thread(
                    evaluate_hybrid,
                    task_id=body.get("task_id", ""),
                    video_base64=video,
                    mime_type=body.get("mime_type", "video/webm"),
                    frames_base64=frames,
                    extra_context=body.get("context", ""),
                )
            elif image:
                result = await asyncio.to_thread(
                    evaluate_image_with_vlm,
                    image_base64=image,
                    task_id=body.get("task_id", ""),
                    extra_context=body.get("context", ""),
                )
            else:
                result = {
                    "success": False,
                    "error": "未收到视频、帧或图片数据",
                    "is_correct": None,
                    "quality_level": "unknown",
                }
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse(
                {
                    "success": False,
                    "error": str(exc),
                    "is_correct": None,
                    "quality_level": "unknown",
                },
                status_code=500,
            )

    async def _locate_client_ip(
        self,
        client_ip: str,
    ) -> dict[str, Any] | None:
        amap_key = self.environ.get("AMAP_KEY", "")
        if (
            not amap_key
            or not client_ip
            or client_ip in {"127.0.0.1", "::1"}
        ):
            return None
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    "https://restapi.amap.com/v3/ip",
                    params={"key": amap_key, "ip": client_ip},
                )
            data = response.json()
            if (
                response.status_code == 200
                and data.get("status") == "1"
                and data.get("city")
            ):
                location = {
                    "province": self._string(data.get("province")),
                    "city": self._string(data.get("city")),
                    "district": "",
                    "place": "家中",
                    "adcode": data.get("adcode", ""),
                    "source": "amap-client-ip",
                }
                self._log(
                    "[LocationAPI] ✅ 客户端IP定位: "
                    f"{client_ip} → {location['province']} "
                    f"{location['city']}"
                )
                return location
        except Exception as exc:
            self._log(f"[LocationAPI] ⚠️ 客户端IP定位失败: {exc}")
        return None

    @classmethod
    def _parse_reverse_location(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        if data.get("status") != "1":
            return None
        reverse = data.get("regeocode", {})
        address = reverse.get("addressComponent", {})
        neighborhood = address.get("neighborhood", {})
        street_number = address.get("streetNumber", {})
        return {
            "province": cls._string(address.get("province")),
            "city": cls._string(address.get("city")),
            "district": cls._string(address.get("district")),
            "neighborhood": (
                cls._string(neighborhood.get("name"))
                if isinstance(neighborhood, dict)
                else ""
            ),
            "street": (
                cls._string(street_number.get("street"))
                if isinstance(street_number, dict)
                else ""
            ),
            "formatted_address": cls._string(
                reverse.get("formatted_address")
            ),
            "place": "家中",
            "adcode": address.get("adcode", ""),
            "source": "amap-gps",
        }

    @staticmethod
    def _save_location(location_service, location: dict) -> None:
        location_service.save_location_to_config(location)
        location_service._cached_location = location
        location_service._cached_weather = None
        location_service._weather_update_time = None

    def _build_session_audio_payload(
        self,
        session_id: str,
        audio: dict,
    ) -> dict:
        item = dict(audio)
        role = str(item.get("role") or "user")
        item["role"] = role
        item["role_label"] = (
            "系统语音" if role == "assistant" else "用户语音"
        )
        item["audio_url"] = (
            f"/api/sessions/{session_id}/audio/{item.get('id')}"
        )
        item["download_url"] = f"{item['audio_url']}?download=1"
        item["file_name"] = Path(
            item.get("file_path", "")
        ).name
        item["display_text"] = (
            item.get("content_text")
            or item.get("asr_text")
            or item["file_name"]
        )
        return item

    def _build_merged_session_audio(
        self,
        session_id: str,
        target_sample_rate: int = 24000,
    ) -> tuple[np.ndarray, int, list[dict]] | None:
        audio_rows = self.repository.list_audio_records(session_id)
        if not audio_rows:
            return None
        parts: list[np.ndarray] = []
        export_rows: list[dict] = []
        separator = np.zeros(
            int(target_sample_rate * 0.18),
            dtype=np.float32,
        )
        for row in audio_rows:
            audio_path = self._resolve_managed_audio_path(
                row.get("file_path")
            )
            if not audio_path:
                continue
            audio, sample_rate = self._load_audio_mono(audio_path)
            normalized = resample_audio(
                audio,
                sample_rate,
                target_sample_rate,
            )
            if normalized.size == 0:
                continue
            if parts:
                parts.append(separator)
            parts.append(normalized)
            export_rows.append(dict(row))
        if not parts:
            return None
        return (
            np.concatenate(parts).astype(np.float32),
            target_sample_rate,
            export_rows,
        )

    def _build_audio_clips_zip(
        self,
        session_id: str,
    ) -> bytes | None:
        audio_rows = self.repository.list_audio_records(session_id)
        if not audio_rows:
            return None
        buffer = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(
            buffer,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for index, row in enumerate(audio_rows, start=1):
                audio_path = self._resolve_managed_audio_path(
                    row.get("file_path")
                )
                if not audio_path:
                    continue
                role = str(row.get("role") or "user")
                timestamp = self._safe_download_name(
                    str(row.get("created_at") or "")
                )
                clip_name = (
                    f"{index:03d}_{role}_{timestamp}"
                    f"{audio_path.suffix or '.wav'}"
                )
                archive.write(audio_path, arcname=clip_name)
                manifest.append(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "role": role,
                        "created_at": row.get("created_at"),
                        "duration_s": row.get("duration_s"),
                        "file_name": clip_name,
                        "content_text": row.get("content_text"),
                        "asr_text": row.get("asr_text"),
                    }
                )
            if manifest:
                archive.writestr(
                    "clips_manifest.json",
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
        return buffer.getvalue() if manifest else None

    def _resolve_managed_audio_path(
        self,
        file_path,
    ) -> Path | None:
        if not file_path:
            return None
        try:
            audio_path = Path(file_path).expanduser().resolve()
            audio_path.relative_to(self.voice_calls_dir)
        except Exception:
            return None
        if not audio_path.exists() or not audio_path.is_file():
            return None
        return audio_path

    @staticmethod
    def _load_audio_mono(file_path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(
            file_path,
            dtype="float32",
            always_2d=False,
        )
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array.reshape(-1), int(sample_rate)

    @staticmethod
    def _write_scores(writer, scores: list[dict]) -> None:
        labels = {
            "orientation": "定向力",
            "registration": "即时记忆",
            "attention_calculation": "注意与计算",
            "recall": "回忆能力",
            "language": "语言能力",
            "copy": "视空间",
        }
        writer.writerow(["=== MMSE 评分 ==="])
        writer.writerow(["维度", "得分", "满分"])
        total = 0
        total_max = 0
        for score in scores:
            dimension_id = score.get("dimension_id", "")
            value = score.get("score", 0)
            maximum = score.get("max_score", 0)
            writer.writerow(
                [
                    labels.get(dimension_id, dimension_id),
                    value,
                    maximum,
                ]
            )
            total += value
            total_max += maximum
        writer.writerow(["总分", total, total_max])

    def _static_html(
        self,
        filename: str,
        missing_message: str,
    ) -> HTMLResponse:
        html_file = self.static_dir / filename
        if html_file.exists():
            return HTMLResponse(
                content=html_file.read_text(encoding="utf-8"),
                headers=self._NO_CACHE_HEADERS,
            )
        return message_page(
            missing_message,
            "页面暂时无法打开。请返回陪伴首页；如果仍无法访问，请联系管理员。",
            status_code=404,
        )

    def _register_routes(self) -> None:
        routes = (
            (
                "/demo/3step-action",
                self.open_three_step_action_demo,
                ["GET"],
            ),
            ("/demo/tests", self.open_test_pages, ["GET"]),
            ("/api/logs", self.logs_snapshot, ["GET"]),
            ("/api/logs/stream", self.logs_stream, ["GET"]),
            ("/api/sessions", self.list_sessions, ["GET"]),
            ("/api/patients", self.list_patients, ["GET"]),
            (
                "/api/sessions/active",
                self.active_sessions,
                ["GET"],
            ),
            (
                "/api/sessions/{session_id}",
                self.get_session,
                ["GET"],
            ),
            (
                "/api/sessions/{session_id}",
                self.delete_session,
                ["DELETE"],
            ),
            (
                "/api/sessions/{session_id}/audio/{audio_id}",
                self.get_session_audio,
                ["GET"],
            ),
            (
                "/api/sessions/{session_id}/export/"
                "audio-merged.wav",
                self.export_merged_audio,
                ["GET"],
            ),
            (
                "/api/sessions/{session_id}/export/"
                "audio-clips.zip",
                self.export_audio_clips,
                ["GET"],
            ),
            (
                "/api/sessions/{session_id}/export/csv",
                self.export_session_csv,
                ["GET"],
            ),
            ("/", self.index, ["GET"]),
            (
                "/api/mmse-image/{image_id}",
                self.mmse_image,
                ["GET"],
            ),
            (
                "/api/memory-snapshot",
                self.memory_snapshot,
                ["GET"],
            ),
            ("/api/location", self.get_location, ["GET"]),
            ("/api/location", self.set_location, ["POST"]),
            ("/api/model", self.get_model, ["GET"]),
            ("/api/model", self.switch_model, ["POST"]),
            (
                "/api/vision-evaluate",
                self.vision_evaluate,
                ["POST"],
            ),
        )
        for path, endpoint, methods in routes:
            self.router.add_api_route(
                path,
                endpoint,
                methods=methods,
            )

    @staticmethod
    def _safe_download_name(
        value: str,
        fallback: str = "未知",
    ) -> str:
        cleaned = re.sub(
            r'[\\/:*?"<>|]+',
            "_",
            str(value or "").strip(),
        )
        cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
        return cleaned or fallback

    @staticmethod
    def _download_headers(filename: str) -> dict:
        return {
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{quote(filename)}"
            )
        }

    @staticmethod
    def _string(value: Any) -> str:
        return value if isinstance(value, str) else ""

    @staticmethod
    def _error(error: str, status_code: int) -> JSONResponse:
        return JSONResponse(
            {"success": False, "error": error},
            status_code=status_code,
        )
