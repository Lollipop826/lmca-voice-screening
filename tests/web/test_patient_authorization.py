import asyncio
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import HTTPException

from src.db import database
from src.web.auth import AuthService
from src.web.memory_api import (
    MemoryApiController,
    MemoryItemDeleteRequest,
    MemoryItemVersionRequest,
    MemoryItemUpdateRequest,
    MemoryRevisionConflict,
    MMSERecordRequest,
    MemoryUpdateRequest,
)


def test_database_patient_assignment_revoke_and_audit():
    with TemporaryDirectory() as temp_dir:
        old_path = database.DB_PATH
        database.DB_PATH = str(Path(temp_dir) / "auth.db")
        try:
            database.init_db()
            database.create_user("doctor", "hash", "salt")
            patient = database.create_patient({"name": "张阿姨"})
            assignment = database.assign_patient(
                patient["patient_id"],
                "doctor",
                can_read=True,
                assigned_by="admin",
            )

            assert assignment["can_read"] == 1
            assert database.get_patient_assignment(
                patient["patient_id"], "doctor"
            )["revoked_at"] is None
            assert database.revoke_patient_assignment(
                patient["patient_id"],
                "doctor",
                revoked_by="admin",
            )
            assert database.get_patient_assignment(
                patient["patient_id"], "doctor"
            )["revoked_at"]
            events = database.list_patient_audit_events(patient["patient_id"])
            assert [event["action"] for event in events[:2]] == [
                "patient_assignment_revoked",
                "patient_assignment_granted",
            ]
            assert all("张阿姨" not in str(event) for event in events)
        finally:
            database.DB_PATH = old_path


def test_list_accessible_patients_filters_by_voice_assignment():
    with TemporaryDirectory() as temp_dir:
        old_path = database.DB_PATH
        database.DB_PATH = str(Path(temp_dir) / "patients.db")
        try:
            database.init_db()
            database.create_user("doctor", "hash", "salt")
            voice_patient = database.create_patient({"name": "可绑定患者"})
            read_only_patient = database.create_patient({"name": "仅查看患者"})
            revoked_patient = database.create_patient({"name": "已撤销患者"})
            database.assign_patient(
                voice_patient["patient_id"],
                "doctor",
                can_read=True,
                can_voice=True,
            )
            database.assign_patient(
                read_only_patient["patient_id"],
                "doctor",
                can_read=True,
            )
            database.assign_patient(
                revoked_patient["patient_id"],
                "doctor",
                can_read=True,
                can_voice=True,
            )
            database.revoke_patient_assignment(
                revoked_patient["patient_id"],
                "doctor",
            )

            visible = database.list_accessible_patients(
                actor_username="doctor",
                is_admin=False,
            )
            assert [item["name"] for item in visible] == ["可绑定患者"]
            assert len(
                database.list_accessible_patients(is_admin=True)
            ) == 3
        finally:
            database.DB_PATH = old_path


class _AuthRepository:
    def __init__(self):
        self.patient = {"patient_id": "pt-1", "name": "张阿姨"}
        self.assignments = {}

    def get_user_by_username(self, username):
        return {"username": username, "role": "user"}

    def get_patient(self, patient_id):
        return self.patient if patient_id == "pt-1" else None

    def get_patient_assignment(self, patient_id, username):
        return self.assignments.get((patient_id, username))


def test_auth_service_denies_unknown_and_unassigned_without_patient_data(tmp_path):
    repository = _AuthRepository()
    auth = AuthService(repository=repository, data_dir=tmp_path, logger=lambda _: None)
    actor = {"username": "doctor", "role": "user"}

    with pytest.raises(HTTPException) as unknown:
        auth.authorize_patient(actor, "missing", "read")
    assert unknown.value.status_code == 403
    assert unknown.value.detail == "PATIENT_ACCESS_DENIED"

    with pytest.raises(HTTPException) as unassigned:
        auth.authorize_patient(actor, "pt-1", "read")
    assert unassigned.value.status_code == 403
    assert unassigned.value.detail == "PATIENT_ACCESS_DENIED"

    repository.assignments[("pt-1", "doctor")] = {
        "can_read": 1,
        "can_write": 0,
        "can_voice": 0,
        "revoked_at": None,
    }
    assert auth.authorize_patient(actor, "pt-1", "read") == actor
    with pytest.raises(HTTPException) as write_denied:
        auth.authorize_patient(actor, "pt-1", "write")
    assert write_denied.value.detail == "PATIENT_ACCESS_DENIED"


def test_admin_can_access_existing_patient_but_not_unknown(tmp_path):
    repository = _AuthRepository()
    auth = AuthService(repository=repository, data_dir=tmp_path, logger=lambda _: None)
    admin = {"username": "admin", "role": "admin"}

    assert auth.authorize_patient(admin, "pt-1", "write") == admin
    with pytest.raises(HTTPException) as missing:
        auth.authorize_patient(admin, "missing", "read")
    assert missing.value.status_code == 404


def test_revoked_assignment_denies_subsequent_voice_binding(tmp_path):
    repository = _AuthRepository()
    auth = AuthService(repository=repository, data_dir=tmp_path, logger=lambda _: None)
    actor = {"username": "doctor", "role": "user"}
    repository.assignments[("pt-1", "doctor")] = {
        "can_read": 1,
        "can_write": 1,
        "can_voice": 1,
        "revoked_at": None,
    }

    assert auth.authorize_patient(actor, "pt-1", "voice") == actor
    repository.assignments[("pt-1", "doctor")]["revoked_at"] = "2026-08-02T00:00:00"

    with pytest.raises(HTTPException) as denied:
        auth.authorize_patient(actor, "pt-1", "voice")
    assert denied.value.status_code == 403
    assert denied.value.detail == "PATIENT_ACCESS_DENIED"


def test_session_access_allows_owned_and_assigned_sessions(tmp_path):
    old_path = database.DB_PATH
    database.DB_PATH = str(tmp_path / "sessions.db")
    try:
        database.init_db()
        database.create_user("doctor", "hash", "salt")
        patient = database.create_patient({"name": "张阿姨"})
        database.assign_patient(patient["patient_id"], "doctor", can_read=True)
        database.create_session("owned", owner_username="doctor")
        database.create_session("assigned", owner_username="other", patient_id=patient["patient_id"])
        database.create_session("blocked", owner_username="other")
        for session_id in ("owned", "assigned", "blocked"):
            database.save_message(session_id, "user", "你好")

        assert database.can_user_access_session("owned", "doctor")
        assert database.can_user_access_session("assigned", "doctor")
        assert not database.can_user_access_session("blocked", "doctor")

        visible = database.list_sessions(actor_username="doctor", is_admin=False)
        assert {row["session_id"] for row in visible} == {"owned", "assigned"}
        assert {row["session_id"] for row in database.list_sessions(is_admin=True)} == {
            "owned",
            "assigned",
            "blocked",
        }
    finally:
        database.DB_PATH = old_path


def test_auth_service_require_session_access_uses_repository_checker(tmp_path):
    class Repository(_AuthRepository):
        def __init__(self, allowed):
            super().__init__()
            self.allowed = allowed

        def can_user_access_session(self, session_id, username):
            return self.allowed == (session_id, username)

    auth = AuthService(
        repository=Repository(("session-1", "doctor")),
        data_dir=tmp_path,
        logger=lambda _: None,
    )
    request = SimpleNamespace(
        cookies={auth.cookie_name: auth.sign_session_token("doctor")}
    )

    assert auth.require_session_access(request, "session-1")["username"] == "doctor"
    with pytest.raises(HTTPException) as denied:
        auth.require_session_access(request, "session-2")
    assert denied.value.status_code == 403


class _Memory:
    def __init__(self):
        self.calls = []

    def get_memory_for_user(self, patient_id):
        self.calls.append(("get", patient_id))
        return {"patient_id": patient_id}

    def get_session_reflection(self, patient_id, session_id):
        self.calls.append(("reflection", patient_id, session_id))
        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "status": "succeeded",
            "session_summary": "本次会话已完成整理。",
            "attempt_count": 1,
            "created_at": "2026-08-18T10:00:00",
            "updated_at": "2026-08-18T10:00:01",
            "completed_at": "2026-08-18T10:00:01",
            "lease_token": "must-not-leak",
            "last_error": "must-not-leak",
            "summary_evidence_json": "must-not-leak",
        }

    def list_memory_items(self, patient_id, *, include_deleted=False, mode=None):
        self.calls.append(("list", patient_id, include_deleted, mode))
        return [{"item_id": "item-1", "patient_id": patient_id}]

    def update_memory_by_user(self, patient_id, updates, *, expected_revision):
        self.calls.append(("update", patient_id, updates, expected_revision))
        return {"patient_id": patient_id, "revision": expected_revision + 1}

    def update_memory_item(self, patient_id, item_id, content, expected_version, *, updated_by="user"):
        self.calls.append(("patch_item", patient_id, item_id, content, expected_version, updated_by))
        return {"item_id": item_id, "content": content, "version": expected_version + 1}

    def delete_memory_item(self, patient_id, item_id, expected_version, deletion_token, *, deleted_by="user"):
        self.calls.append(("delete_item", patient_id, item_id, expected_version, deletion_token, deleted_by))
        return {"item_id": item_id, "status": "deleted", "version": expected_version + 1}

    def confirm_memory_item(self, patient_id, item_id, expected_version, *, confirmed_by="user"):
        self.calls.append(("confirm_item", patient_id, item_id, expected_version, confirmed_by))
        return {"item_id": item_id, "status": "active", "version": expected_version + 1}

    def reject_memory_item(self, patient_id, item_id, expected_version, *, rejected_by="user"):
        self.calls.append(("reject_item", patient_id, item_id, expected_version, rejected_by))
        return {"item_id": item_id, "status": "blocked", "version": expected_version + 1}

    def stop_memory_item(self, patient_id, item_id, expected_version, *, stopped_by="user"):
        self.calls.append(("stop_item", patient_id, item_id, expected_version, stopped_by))
        return {"item_id": item_id, "status": "blocked", "version": expected_version + 1}

    def get_memory_item(self, patient_id, item_id):
        self.calls.append(("get_item", patient_id, item_id))
        return {
            "item_id": item_id,
            "category": "facts",
            "content": "当前安全内容",
            "status": "active",
            "version": 9,
            "evidence_json": '[{"quote":"不应返回"}]',
            "metadata_json": '{"secret":"不应返回"}',
            "source_turn_id": "turn-secret",
        }


class _ApiAuth:
    def __init__(self, allow=True):
        self.allow = allow
        self.calls = []

    def require_patient_access(self, request, patient_id, action):
        self.calls.append((patient_id, action))
        if not self.allow:
            raise HTTPException(status_code=403, detail="PATIENT_ACCESS_DENIED")


def test_memory_api_authorizes_before_read_and_passes_expected_revision():
    memory = _Memory()
    auth = _ApiAuth()
    controller = MemoryApiController(memory, auth=auth)
    request = SimpleNamespace()

    async def scenario():
        read = await controller.get_patient_memory("pt-1", request)
        updated = await controller.update_patient_memory(
            "pt-1",
            MemoryUpdateRequest(updates={"facts": ["已确认"]}, expected_revision=4),
            request,
        )
        return read, updated

    read, updated = asyncio.run(scenario())
    assert read["data"]["patient_id"] == "pt-1"
    assert updated["data"]["revision"] == 5
    assert auth.calls == [("pt-1", "read"), ("pt-1", "write")]
    assert memory.calls[-1][-1] == 4


def test_session_reflection_api_authorizes_and_hides_internal_fields():
    memory = _Memory()
    auth = _ApiAuth()
    controller = MemoryApiController(memory, auth=auth)

    result = asyncio.run(
        controller.get_session_reflection("pt-1", SimpleNamespace(), "session-1")
    )

    assert auth.calls == [("pt-1", "read")]
    assert memory.calls == [("reflection", "pt-1", "session-1")]
    assert result["data"] == {
        "session_id": "session-1",
        "status": "succeeded",
        "session_summary": "本次会话已完成整理。",
        "attempt_count": 1,
        "created_at": "2026-08-18T10:00:00",
        "updated_at": "2026-08-18T10:00:01",
        "completed_at": "2026-08-18T10:00:01",
    }


def test_session_reflection_api_includes_previous_completed_summary():
    class PendingMemory(_Memory):
        def get_session_reflection(self, patient_id, session_id):
            self.calls.append(("reflection", patient_id, session_id))
            return {"patient_id": patient_id, "session_id": session_id, "status": "missing"}

        def get_latest_session_reflection(self, patient_id, *, exclude_session_id=None):
            self.calls.append(("latest", patient_id, exclude_session_id))
            return {
                "session_id": "previous-session",
                "status": "succeeded",
                "session_summary": "上一轮会话摘要。",
                "attempt_count": 1,
                "completed_at": "2026-08-18T09:00:00",
            }

    memory = PendingMemory()
    controller = MemoryApiController(memory, auth=_ApiAuth())

    result = asyncio.run(
        controller.get_session_reflection("pt-1", SimpleNamespace(), "session-2")
    )

    assert result["data"]["status"] == "missing"
    assert result["data"]["previous_reflection"] == {
        "session_id": "previous-session",
        "status": "succeeded",
        "session_summary": "上一轮会话摘要。",
        "attempt_count": 1,
        "created_at": None,
        "updated_at": None,
        "completed_at": "2026-08-18T09:00:00",
    }


def test_session_reflection_api_maps_failure_without_leaking_provider_error():
    class FailedMemory(_Memory):
        def get_session_reflection(self, patient_id, session_id):
            return {
                "session_id": session_id,
                "status": "failed",
                "attempt_count": 2,
                "next_retry_at": "2026-08-19T10:05:00",
                "last_error": "TimeoutError: qwen request timed out; api_key=secret",
            }

        def get_latest_session_reflection(self, patient_id, *, exclude_session_id=None):
            return {"status": "missing"}

    result = asyncio.run(
        MemoryApiController(FailedMemory(), auth=_ApiAuth()).get_session_reflection(
            "pt-1", SimpleNamespace(), "session-1"
        )
    )

    assert result["data"]["failure_reason"] == "模型服务响应超时"
    assert result["data"]["next_retry_at"] == "2026-08-19T10:05:00"
    assert "last_error" not in result["data"]
    assert "secret" not in str(result["data"])


def test_memory_api_exposes_item_list_patch_and_delete_routes():
    memory = _Memory()
    audit_events = []
    controller = MemoryApiController(
        memory,
        auth=_ApiAuth(),
        audit_event=lambda **event: audit_events.append(event),
    )
    request = SimpleNamespace()

    async def scenario():
        listed = await controller.list_memory_items("pt-1", request, include_deleted=True, mode="wellbeing")
        patched = await controller.update_memory_item(
            "pt-1",
            "item-1",
            MemoryItemUpdateRequest(content="已更新", expected_version=2),
            request,
        )
        deleted = await controller.delete_memory_item(
            "pt-1",
            "item-1",
            MemoryItemDeleteRequest(expected_version=3, deletion_token="delete-token-001"),
            request,
        )
        return listed, patched, deleted

    listed, patched, deleted = asyncio.run(scenario())
    assert listed["items"][0]["item_id"] == "item-1"
    assert patched["item"]["content"] == "已更新"
    assert deleted["item"]["status"] == "deleted"
    assert ("pt-1", "read") in controller.auth.calls
    assert ("pt-1", "write") in controller.auth.calls
    assert memory.calls[0] == ("list", "pt-1", True, "wellbeing")
    assert [event["action"] for event in audit_events] == [
        "memory_item_updated",
        "memory_item_deleted",
    ]
    assert [event["object_revision"] for event in audit_events] == [3, 4]
    assert all("已更新" not in str(event) for event in audit_events)


def test_memory_item_candidate_actions_are_versioned_and_audited():
    memory = _Memory()
    audit_events = []
    controller = MemoryApiController(
        memory,
        auth=_ApiAuth(),
        audit_event=lambda **event: audit_events.append(event),
    )

    async def scenario():
        confirmed = await controller.confirm_memory_item(
            "pt-1", "candidate-1", MemoryItemVersionRequest(expected_version=2), SimpleNamespace()
        )
        rejected = await controller.reject_memory_item(
            "pt-1", "candidate-2", MemoryItemVersionRequest(expected_version=3), SimpleNamespace()
        )
        stopped = await controller.stop_memory_item(
            "pt-1", "active-1", MemoryItemVersionRequest(expected_version=4), SimpleNamespace()
        )
        return confirmed, rejected, stopped

    confirmed, rejected, stopped = asyncio.run(scenario())

    assert confirmed["item"]["status"] == "active"
    assert rejected["item"]["status"] == "blocked"
    assert stopped["item"]["status"] == "blocked"
    assert [call[0] for call in memory.calls] == ["confirm_item", "reject_item", "stop_item"]
    assert [event["action"] for event in audit_events] == [
        "memory_item_confirmed",
        "memory_item_rejected",
        "memory_item_stopped",
    ]


def test_memory_item_api_returns_minimum_safe_fields_and_current_item_on_conflict():
    class ConflictMemory(_Memory):
        def update_memory_item(self, patient_id, item_id, content, expected_version, *, updated_by="user"):
            raise MemoryRevisionConflict(9)

        def list_memory_items(self, patient_id, *, include_deleted=False, mode=None):
            return [self.get_memory_item(patient_id, "item-1")]

    controller = MemoryApiController(ConflictMemory(), auth=_ApiAuth())

    async def scenario():
        listed = await controller.list_memory_items("pt-1", SimpleNamespace())
        try:
            await controller.update_memory_item(
                "pt-1", "item-1", MemoryItemUpdateRequest(content="保留草稿", expected_version=3), SimpleNamespace()
            )
        except HTTPException as exc:
            return listed, exc
        raise AssertionError("stale item write must conflict")

    listed, conflict = asyncio.run(scenario())

    assert set(listed["items"][0]) == {
        "item_id", "category", "content", "status", "version", "observed_at",
        "valid_until", "sensitivity", "confidence", "created_at", "updated_at", "source",
    }
    assert conflict.status_code == 409
    assert conflict.detail["current_revision"] == 9
    assert conflict.detail["current_item"]["content"] == "当前安全内容"
    assert "evidence_json" not in conflict.detail["current_item"]
    assert "source_turn_id" not in conflict.detail["current_item"]


def test_memory_item_audit_events_persist_without_content(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "audit.db"))
    database.init_db()

    class AuthWithUser(_ApiAuth):
        def current_user(self, _request):
            return {"username": "doctor"}

    controller = MemoryApiController(_Memory(), auth=AuthWithUser())

    async def scenario():
        await controller.update_memory_item(
            "pt-1",
            "item-1",
            MemoryItemUpdateRequest(content="敏感记忆原文", expected_version=2),
            SimpleNamespace(),
        )
        await controller.delete_memory_item(
            "pt-1",
            "item-1",
            MemoryItemDeleteRequest(
                expected_version=3,
                deletion_token="delete-token-002",
            ),
            SimpleNamespace(),
        )

    asyncio.run(scenario())

    events = database.list_patient_audit_events("pt-1")
    assert [event["action"] for event in events[:2]] == [
        "memory_item_deleted",
        "memory_item_updated",
    ]
    assert [event["actor_username"] for event in events[:2]] == [
        "doctor",
        "doctor",
    ]
    assert "敏感记忆原文" not in str(events)


def test_memory_delete_release_flag_blocks_delete_before_write():
    memory = _Memory()
    auth = _ApiAuth()
    controller = MemoryApiController(memory, auth=auth)

    with mock.patch.dict(
        "os.environ",
        {"ENABLE_PATIENT_MEMORY_DELETE": "false"},
    ):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                controller.delete_memory_item(
                    "pt-1",
                    "item-1",
                    MemoryItemDeleteRequest(
                        expected_version=3,
                        deletion_token="delete-token-003",
                    ),
                    SimpleNamespace(),
                )
            )

    assert caught.value.status_code == 403
    assert caught.value.detail == "MEMORY_DELETE_DISABLED"
    assert auth.calls == []
    assert memory.calls == []


def test_memory_mmse_release_flag_blocks_before_write():
    memory = _Memory()
    auth = _ApiAuth()
    controller = MemoryApiController(memory, auth=auth)

    with mock.patch.dict(
        "os.environ",
        {"ENABLE_COGNITIVE_SCREENING": "false"},
    ):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                controller.record_mmse(
                    "pt-1",
                    MMSERecordRequest(score=24, weak_dimensions=["recall"]),
                    SimpleNamespace(),
                )
            )

    assert caught.value.status_code == 403
    assert caught.value.detail == "COGNITIVE_SCREENING_DISABLED"
    assert auth.calls == []
    assert memory.calls == []


def test_memory_api_denies_without_calling_memory():
    memory = _Memory()
    controller = MemoryApiController(memory, auth=_ApiAuth(allow=False))

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            controller.get_patient_memory("pt-1", SimpleNamespace())
        )
    assert caught.value.status_code == 403
    assert memory.calls == []


def test_memory_api_maps_revision_conflict_to_409():
    class ConflictMemory(_Memory):
        def update_memory_by_user(self, patient_id, updates, *, expected_revision):
            raise MemoryRevisionConflict(8)

    controller = MemoryApiController(ConflictMemory(), auth=_ApiAuth())
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            controller.update_patient_memory(
                "pt-1",
                MemoryUpdateRequest(updates={"facts": ["x"]}, expected_revision=3),
                SimpleNamespace(),
            )
        )
    assert caught.value.status_code == 409
    assert caught.value.detail == {
        "error": "MEMORY_REVISION_CONFLICT",
        "current_revision": 8,
    }


def test_memory_api_maps_item_conflicts_and_missing_items():
    class ConflictMemory(_Memory):
        def update_memory_item(self, patient_id, item_id, content, expected_version, *, updated_by="user"):
            raise MemoryRevisionConflict(9)

        def delete_memory_item(self, patient_id, item_id, expected_version, deletion_token, *, deleted_by="user"):
            raise KeyError("MEMORY_ITEM_NOT_FOUND")

    controller = MemoryApiController(ConflictMemory(), auth=_ApiAuth())

    with pytest.raises(HTTPException) as caught_update:
        asyncio.run(
            controller.update_memory_item(
                "pt-1",
                "item-1",
                MemoryItemUpdateRequest(content="x", expected_version=3),
                SimpleNamespace(),
            )
        )
    assert caught_update.value.status_code == 409
    assert caught_update.value.detail["error"] == "MEMORY_REVISION_CONFLICT"
    assert caught_update.value.detail["current_revision"] == 9
    assert caught_update.value.detail["current_item"]["version"] == 9

    with pytest.raises(HTTPException) as caught_delete:
        asyncio.run(
            controller.delete_memory_item(
                "pt-1",
                "item-1",
                MemoryItemDeleteRequest(expected_version=3, deletion_token="delete-token-002"),
                SimpleNamespace(),
            )
        )
    assert caught_delete.value.status_code == 404
