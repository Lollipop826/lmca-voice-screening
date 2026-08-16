import ast
import inspect
import sqlite3
from pathlib import Path

from src.db import database
from src.voice.services import PatientMemoryService


ROOT = Path(__file__).resolve().parents[2]


def test_voice_server_wires_only_emotion_memobase_for_patient_memory():
    tree = ast.parse(
        (ROOT / "voice_server.py").read_text(encoding="utf-8"),
        filename=str(ROOT / "voice_server.py"),
    )
    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "patient_memory_manager" not in imported_modules

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PatientMemoryService"
    ]
    assert len(constructors) == 1
    assert all(
        keyword.arg != "memory_manager"
        for keyword in constructors[0].keywords
    )
    assert any(
        keyword.arg == "long_term_memory"
        and isinstance(keyword.value, ast.IfExp)
        and isinstance(keyword.value.test, ast.Name)
        and keyword.value.test.id == "_ENABLE_LONG_TERM_MEMORY"
        and isinstance(keyword.value.body, ast.Name)
        and keyword.value.body.id == "EMOTION_MEMORY"
        and isinstance(keyword.value.orelse, ast.Constant)
        and keyword.value.orelse.value is None
        for keyword in constructors[0].keywords
    )

    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    flag = assignments["_ENABLE_LONG_TERM_MEMORY"]
    assert isinstance(flag, ast.Call)
    assert isinstance(flag.args[1], ast.Constant)
    assert flag.args[1].value is True
    write_flag = assignments["_ENABLE_LONG_TERM_MEMORY_WRITES"]
    assert isinstance(write_flag, ast.Call)
    assert isinstance(write_flag.args[1], ast.Constant)
    assert write_flag.args[1].value is True


def test_patient_memory_service_has_no_legacy_manager_or_database_fallback():
    parameters = inspect.signature(PatientMemoryService).parameters
    assert "memory_manager" not in parameters
    services_source = (ROOT / "src" / "voice" / "services.py").read_text(encoding="utf-8")
    memory_source = (ROOT / "src" / "context_management" / "emotion_memobase.py").read_text(encoding="utf-8")
    assert "self._memory_manager" not in services_source
    assert "consolidate_session_async" not in services_source
    assert "patient_memory" not in memory_source


def test_database_init_does_not_recreate_retired_memory_tables(tmp_path, monkeypatch):
    database_path = tmp_path / "voice.db"
    monkeypatch.setattr(database, "DB_PATH", str(database_path))

    database.init_db()

    with sqlite3.connect(database_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "patient_memory" not in tables
    assert "patient_memory_sessions" not in tables
