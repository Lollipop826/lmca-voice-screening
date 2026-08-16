from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from src.db import database


class AuthService:
    """Own password hashing, signed login sessions and access checks."""

    _PATIENT_ACTIONS = {"read", "write", "voice"}

    def __init__(
        self,
        *,
        repository=database,
        data_dir: Path,
        cookie_name: str = "aa_session",
        token_ttl: timedelta = timedelta(days=14),
        pbkdf2_iterations: int = 120_000,
        logger=print,
    ) -> None:
        self.repository = repository
        self.data_dir = Path(data_dir)
        self.cookie_name = cookie_name
        self.token_ttl = token_ttl
        self.pbkdf2_iterations = int(pbkdf2_iterations)
        self._log = logger
        self._secret = self._load_secret()

    def initialize_bootstrap_admin(self) -> dict[str, Any] | None:
        admin = self.repository.ensure_bootstrap_admin()
        if admin:
            self._log(
                f"[Auth] ✅ 管理员已就绪: {admin.get('username')}"
            )
        return admin

    def hash_password(
        self,
        password: str,
        salt: bytes | None = None,
    ) -> tuple[str, str]:
        if salt is None:
            salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.pbkdf2_iterations,
        )
        return digest.hex(), salt.hex()

    def verify_password(
        self,
        password: str,
        stored_hash_hex: str,
        stored_salt_hex: str,
    ) -> bool:
        try:
            salt = bytes.fromhex(stored_salt_hex)
        except ValueError:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.pbkdf2_iterations,
        )
        return hmac.compare_digest(digest.hex(), stored_hash_hex)

    def sign_session_token(self, username: str) -> str:
        expires_at = int(
            (datetime.now(timezone.utc) + self.token_ttl).timestamp()
        )
        payload = f"{username}|{expires_at}"
        mac = hmac.new(
            self._secret,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}|{mac}"

    def verify_session_token(self, token: str) -> str | None:
        if not token or token.count("|") != 2:
            return None
        username, expires_str, mac = token.split("|", 2)
        expected_mac = hmac.new(
            self._secret,
            f"{username}|{expires_str}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(mac, expected_mac):
            return None
        try:
            if datetime.now(timezone.utc).timestamp() > float(expires_str):
                return None
        except ValueError:
            return None
        return username

    def user_from_session_token(self, token: str) -> dict | None:
        username = self.verify_session_token(token)
        if not username:
            return None
        try:
            return self.repository.get_user_by_username(username)
        except Exception:
            return None

    def current_user(self, request: Request) -> dict | None:
        token = request.cookies.get(self.cookie_name, "")
        user = self.user_from_session_token(token)
        return self.build_public_user(user) if user else None

    @staticmethod
    def build_public_user(user: dict) -> dict:
        return {
            "username": user.get("username"),
            "display_name": (
                user.get("display_name") or user.get("username")
            ),
            "role": user.get("role") or "user",
        }

    @staticmethod
    def is_admin_user(user: dict | None) -> bool:
        return (
            str((user or {}).get("role") or "").strip().lower()
            == "admin"
        )

    def require_authenticated_user(self, request: Request) -> dict:
        user = self.current_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="UNAUTHORIZED")
        return user

    def require_admin_user(self, request: Request) -> dict:
        user = self.require_authenticated_user(request)
        if not self.is_admin_user(user):
            raise HTTPException(status_code=403, detail="ADMIN_REQUIRED")
        return user

    def require_session_access(
        self,
        request: Request,
        session_id: str,
    ) -> dict:
        user = self.require_authenticated_user(request)
        if self.is_admin_user(user):
            return user
        checker = getattr(self.repository, "can_user_access_session", None)
        if not callable(checker):
            raise HTTPException(status_code=403, detail="SESSION_ACCESS_DENIED")
        try:
            allowed = checker(session_id, str(user.get("username") or ""))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="SESSION_AUTH_UNAVAILABLE",
            ) from exc
        if not allowed:
            raise HTTPException(status_code=403, detail="SESSION_ACCESS_DENIED")
        return user

    def authorize_patient(
        self,
        actor: dict | None,
        patient_id: str,
        action: str,
    ) -> dict:
        """验证已认证主体对一个患者的单项权限。"""
        action = str(action or "").strip().lower()
        if action not in self._PATIENT_ACTIONS:
            raise ValueError(f"unsupported patient action: {action}")
        patient_id = str(patient_id or "").strip()
        username = str((actor or {}).get("username") or "").strip()
        if not username:
            raise HTTPException(status_code=401, detail="UNAUTHORIZED")
        try:
            patient = self.repository.get_patient(patient_id) if patient_id else None
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="PATIENT_AUTH_UNAVAILABLE",
            ) from exc
        if patient is None:
            if self.is_admin_user(actor):
                raise HTTPException(status_code=404, detail="PATIENT_NOT_FOUND")
            raise HTTPException(
                status_code=403,
                detail="PATIENT_ACCESS_DENIED",
            )
        if self.is_admin_user(actor):
            return actor
        try:
            assignment = self.repository.get_patient_assignment(
                patient_id,
                username,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="PATIENT_AUTH_UNAVAILABLE",
            ) from exc
        allowed = assignment and (
            assignment.get(f"can_{action}") is True
            or assignment.get(f"can_{action}") == 1
        )
        if not allowed or assignment.get("revoked_at"):
            raise HTTPException(
                status_code=403,
                detail="PATIENT_ACCESS_DENIED",
            )
        return actor

    def require_patient_access(
        self,
        request: Request,
        patient_id: str,
        action: str,
    ) -> dict:
        return self.authorize_patient(
            self.require_authenticated_user(request),
            patient_id,
            action,
        )

    def set_auth_cookie(
        self,
        response: Response,
        token: str,
        request: Request,
    ) -> None:
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=int(self.token_ttl.total_seconds()),
            httponly=True,
            samesite="lax",
            secure=self._request_scheme(request) == "https",
            path="/",
        )

    def clear_auth_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=self.cookie_name,
            path="/",
            samesite="lax",
        )

    @staticmethod
    def sanitize_next_path(
        next_path: str | None,
        default: str = "/",
    ) -> str:
        candidate = str(next_path or "").strip()
        if (
            not candidate
            or not candidate.startswith("/")
            or candidate.startswith("//")
            or candidate.startswith("/api/")
            or candidate.startswith("/ws")
        ):
            return default
        return candidate

    @staticmethod
    def valid_username(username: str) -> bool:
        return bool(
            username
            and 3 <= len(username) <= 24
            and re.fullmatch(r"[A-Za-z0-9_.\-]+", username)
        )

    @staticmethod
    def valid_password(password: str) -> bool:
        return bool(password and 6 <= len(password) <= 64)

    @staticmethod
    def normalize_user_role(
        role: str | None,
        default: str = "user",
    ) -> str:
        value = str(role or "").strip().lower()
        return value if value in {"admin", "user"} else default

    @staticmethod
    def _request_scheme(request: Request) -> str:
        forwarded_proto = (
            request.headers.get("x-forwarded-proto") or ""
        ).split(",")[0].strip()
        return forwarded_proto or request.url.scheme or "http"

    def _load_secret(self) -> bytes:
        env_secret = os.getenv("AUTH_SECRET_KEY", "").strip()
        if env_secret:
            return env_secret.encode("utf-8")
        key_file = self.data_dir / ".auth_secret"
        try:
            if key_file.exists():
                return key_file.read_bytes()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            new_secret = secrets.token_bytes(32)
            key_file.write_bytes(new_secret)
            try:
                os.chmod(key_file, 0o600)
            except OSError:
                pass
            return new_secret
        except Exception as exc:
            self._log(
                "[AUTH] ⚠️ 读取密钥失败，使用进程临时密钥: "
                f"{exc}"
            )
            return secrets.token_bytes(32)


class AuthController:
    """Register login, account, admin and access-guard HTTP endpoints."""

    _NO_CACHE_HEADERS = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }

    def __init__(
        self,
        auth: AuthService,
        *,
        static_dir: Path,
        legacy_system_url: str = "",
        repository=database,
    ) -> None:
        self.auth = auth
        self.static_dir = Path(static_dir)
        self.legacy_system_url = str(legacy_system_url or "").strip()
        self.repository = repository
        self.router = APIRouter()
        self._register_routes()

    def install(self, app) -> None:
        app.include_router(self.router)
        app.middleware("http")(self.guard)

    async def guard(self, request: Request, call_next):
        path = request.url.path or "/"
        public_prefixes = (
            "/static/",
            "/login",
            "/legacy",
            "/health",
            "/api/auth/",
            "/api/mmse-image/",
        )
        if any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in public_prefixes
        ):
            return await call_next(request)
        if path.startswith("/api/") and not self.auth.current_user(request):
            return JSONResponse(
                {"success": False, "error": "UNAUTHORIZED"},
                status_code=401,
            )
        return await call_next(request)

    async def login_page(self, request: Request, next: str = "/"):
        target = self.auth.sanitize_next_path(next)
        if self.auth.current_user(request):
            return RedirectResponse(url=target, status_code=307)
        html_file = self.static_dir / "login.html"
        if not html_file.exists():
            return HTMLResponse(
                "<h1>登录页面未找到</h1>",
                status_code=404,
            )
        html = html_file.read_text(encoding="utf-8").replace(
            "__NEXT_PATH__",
            json.dumps(target, ensure_ascii=False),
        )
        return HTMLResponse(
            content=html,
            headers=self._NO_CACHE_HEADERS,
        )

    async def me(self, request: Request):
        user = self.auth.current_user(request)
        if not user:
            return JSONResponse(
                {"success": False, "error": "UNAUTHORIZED"},
                status_code=401,
            )
        return {"success": True, "user": user}

    async def register(self, request: Request):
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        display_name = str(payload.get("display_name") or "").strip()
        if not self.auth.valid_username(username):
            return self._error("INVALID_USERNAME", 400)
        if not self.auth.valid_password(password):
            return self._error("INVALID_PASSWORD", 400)
        if self.repository.get_user_by_username(username):
            return self._error("USERNAME_EXISTS", 409)
        password_hash, salt = self.auth.hash_password(password)
        try:
            self.repository.create_user(
                username,
                password_hash,
                salt,
                display_name=display_name or username,
            )
            self.repository.update_user_last_login(username)
            self.repository.ensure_bootstrap_admin()
            created_user = (
                self.repository.get_user_by_username(username)
                or {
                    "username": username,
                    "display_name": display_name or username,
                    "role": "user",
                }
            )
            response = JSONResponse(
                {
                    "success": True,
                    "user": self.auth.build_public_user(created_user),
                }
            )
            self.auth.set_auth_cookie(
                response,
                self.auth.sign_session_token(username),
                request,
            )
            return response
        except Exception as exc:
            return self._error(str(exc), 500)

    async def login(self, request: Request):
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        user = self.repository.get_user_by_username(username)
        if (
            not user
            or not self.auth.verify_password(
                password,
                str(user.get("password_hash") or ""),
                str(user.get("salt") or ""),
            )
        ):
            return self._error("INVALID_CREDENTIALS", 401)
        self.repository.update_user_last_login(username)
        self.repository.ensure_bootstrap_admin()
        refreshed = self.repository.get_user_by_username(username) or user
        response = JSONResponse(
            {
                "success": True,
                "user": self.auth.build_public_user(refreshed),
            }
        )
        self.auth.set_auth_cookie(
            response,
            self.auth.sign_session_token(username),
            request,
        )
        return response

    async def logout(self):
        response = JSONResponse({"success": True})
        self.auth.clear_auth_cookie(response)
        return response

    async def change_password(self, request: Request):
        current_user = self.auth.user_from_session_token(
            request.cookies.get(self.auth.cookie_name, "")
        )
        if not current_user:
            return self._error("UNAUTHORIZED", 401)
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        if not self.auth.verify_password(
            str(payload.get("current_password") or ""),
            str(current_user.get("password_hash") or ""),
            str(current_user.get("salt") or ""),
        ):
            return self._error("INVALID_CURRENT_PASSWORD", 400)
        new_password = str(payload.get("new_password") or "")
        if not self.auth.valid_password(new_password):
            return self._error("INVALID_PASSWORD", 400)
        password_hash, salt = self.auth.hash_password(new_password)
        self.repository.update_user_password(
            str(current_user.get("username") or ""),
            password_hash,
            salt,
        )
        return JSONResponse({"success": True})

    async def admin_users(self, request: Request):
        access_error = self._admin_access_error(request)
        if access_error:
            return access_error
        return {
            "success": True,
            "users": self.repository.list_users(),
        }

    async def list_public_api_keys(self, request: Request):
        self.auth.require_admin_user(request)
        return {
            "success": True,
            "keys": self.repository.list_public_api_keys(),
        }

    async def create_public_api_key(self, request: Request):
        admin = self.auth.require_admin_user(request)
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        name = str(payload.get("name") or "").strip()
        if len(name) < 2 or len(name) > 80:
            return self._error("INVALID_KEY_NAME", 400)
        secret = f"adsk_live_{secrets.token_urlsafe(32)}"
        key = self.repository.create_public_api_key(
            key_id=f"key_{uuid.uuid4().hex}",
            name=name,
            key_prefix=f"{secret[:18]}…",
            key_hash=hashlib.sha256(
                secret.encode("utf-8")
            ).hexdigest(),
            created_by=str(admin.get("username") or ""),
        )
        return {
            "success": True,
            "key": key,
            "api_key": secret,
            "warning": (
                "请立即保存此 API Key；"
                "关闭提示后将无法再次查看完整内容。"
            ),
        }

    async def revoke_public_api_key(
        self,
        key_id: str,
        request: Request,
    ):
        self.auth.require_admin_user(request)
        if not re.fullmatch(r"key_[a-f0-9]{32}", key_id):
            return self._error("KEY_NOT_FOUND", 404)
        if not self.repository.revoke_public_api_key(key_id):
            return self._error("KEY_NOT_FOUND_OR_REVOKED", 404)
        return {"success": True}

    async def create_user(self, request: Request):
        access_error = self._admin_access_error(request)
        if access_error:
            return access_error
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        username = str(payload.get("username") or "").strip()
        display_name = str(payload.get("display_name") or "").strip()
        password = str(payload.get("password") or "")
        role = self.auth.normalize_user_role(payload.get("role"))
        if not self.auth.valid_username(username):
            return self._error("INVALID_USERNAME", 400)
        if not self.auth.valid_password(password):
            return self._error("INVALID_PASSWORD", 400)
        if self.repository.get_user_by_username(username):
            return self._error("USERNAME_EXISTS", 409)
        password_hash, salt = self.auth.hash_password(password)
        created = self.repository.create_user(
            username,
            password_hash,
            salt,
            display_name=display_name or username,
            role=role,
        )
        return {"success": True, "user": created}

    async def update_user_profile(
        self,
        username: str,
        request: Request,
    ):
        access_error = self._admin_access_error(request)
        if access_error:
            return access_error
        target_user = self.repository.get_user_by_username(username)
        if not target_user:
            return self._error("USER_NOT_FOUND", 404)
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        display_name = payload.get("display_name")
        if display_name is not None:
            display_name = str(display_name).strip()
            if not display_name:
                return self._error("INVALID_DISPLAY_NAME", 400)
        normalized_role = None
        if payload.get("role") is not None:
            normalized_role = self.auth.normalize_user_role(
                payload.get("role"),
                default="",
            )
            if not normalized_role:
                return self._error("INVALID_ROLE", 400)
            if (
                target_user.get("role") == "admin"
                and normalized_role != "admin"
                and self.repository.count_admin_users() <= 1
            ):
                return self._error("LAST_ADMIN_LOCKED", 400)
        updated = self.repository.update_user_profile(
            username,
            display_name=display_name,
            role=normalized_role,
        )
        return {"success": True, "user": updated}

    async def update_user_password(
        self,
        username: str,
        request: Request,
    ):
        access_error = self._admin_access_error(request)
        if access_error:
            return access_error
        if not self.repository.get_user_by_username(username):
            return self._error("USER_NOT_FOUND", 404)
        payload = await self._json_payload(request)
        if isinstance(payload, JSONResponse):
            return payload
        new_password = str(payload.get("new_password") or "")
        if not self.auth.valid_password(new_password):
            return self._error("INVALID_PASSWORD", 400)
        password_hash, salt = self.auth.hash_password(new_password)
        self.repository.update_user_password(
            username,
            password_hash,
            salt,
        )
        return {"success": True}

    async def history_page(self, request: Request):
        user = self.auth.current_user(request)
        if not user:
            return RedirectResponse(
                url="/login?next=/history",
                status_code=307,
            )
        return self._static_html("history.html", "历史记录页面未找到")

    async def admin_page(self, request: Request):
        user = self.auth.current_user(request)
        if not user:
            return RedirectResponse(
                url="/login?next=/admin",
                status_code=307,
            )
        if not self.auth.is_admin_user(user):
            return HTMLResponse(
                "<h1>无权限访问管理员后台</h1>"
                '<p><a href="/">返回评估首页</a></p>',
                status_code=403,
            )
        return self._static_html("admin.html", "管理员页面未找到")

    async def admin_usage(self, request: Request):
        access_error = self._admin_access_error(request)
        if access_error:
            return access_error
        try:
            return {
                "success": True,
                "data": self.repository.get_admin_usage_snapshot(),
            }
        except Exception as exc:
            return self._error(str(exc), 500)

    async def legacy_entry(self, request: Request):
        return RedirectResponse(
            url=self._resolve_legacy_system_url(request),
            status_code=307,
        )

    def _register_routes(self) -> None:
        routes = (
            ("/login", self.login_page, ["GET"]),
            ("/api/auth/me", self.me, ["GET"]),
            ("/api/auth/register", self.register, ["POST"]),
            ("/api/auth/login", self.login, ["POST"]),
            ("/api/auth/logout", self.logout, ["POST"]),
            (
                "/api/auth/change-password",
                self.change_password,
                ["POST"],
            ),
            ("/api/admin/users", self.admin_users, ["GET"]),
            (
                "/api/admin/public-api-keys",
                self.list_public_api_keys,
                ["GET"],
            ),
            (
                "/api/admin/public-api-keys",
                self.create_public_api_key,
                ["POST"],
            ),
            (
                "/api/admin/public-api-keys/{key_id}",
                self.revoke_public_api_key,
                ["DELETE"],
            ),
            ("/api/admin/users", self.create_user, ["POST"]),
            (
                "/api/admin/users/{username}/profile",
                self.update_user_profile,
                ["POST"],
            ),
            (
                "/api/admin/users/{username}/password",
                self.update_user_password,
                ["POST"],
            ),
            ("/history", self.history_page, ["GET"]),
            ("/admin", self.admin_page, ["GET"]),
            ("/api/admin/usage", self.admin_usage, ["GET"]),
            ("/legacy", self.legacy_entry, ["GET"]),
        )
        for path, endpoint, methods in routes:
            self.router.add_api_route(
                path,
                endpoint,
                methods=methods,
            )

    async def _json_payload(self, request: Request):
        try:
            return await request.json()
        except Exception:
            return self._error("INVALID_JSON", 400)

    def _admin_access_error(
        self,
        request: Request,
    ) -> JSONResponse | None:
        user = self.auth.current_user(request)
        if not user:
            return self._error("UNAUTHORIZED", 401)
        if not self.auth.is_admin_user(user):
            return self._error("FORBIDDEN", 403)
        return None

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
        return HTMLResponse(
            f"<h1>{missing_message}</h1>",
            status_code=404,
        )

    def _resolve_legacy_system_url(self, request: Request) -> str:
        if self.legacy_system_url:
            return self.legacy_system_url
        forwarded_proto = (
            request.headers.get("x-forwarded-proto") or ""
        ).split(",")[0].strip()
        scheme = forwarded_proto or request.url.scheme or "http"
        host_header = (
            request.headers.get("host") or ""
        ).split(",")[0].strip()
        hostname = request.url.hostname or "127.0.0.1"
        if host_header:
            if host_header.startswith("["):
                closing = host_header.find("]")
                if closing != -1:
                    hostname = host_header[: closing + 1]
            else:
                hostname = host_header.split(":")[0] or hostname
        return f"{scheme}://{hostname}:8001/app/login/"

    @staticmethod
    def _error(error: str, status_code: int) -> JSONResponse:
        return JSONResponse(
            {"success": False, "error": error},
            status_code=status_code,
        )
