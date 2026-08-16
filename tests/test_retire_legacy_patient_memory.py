import sqlite3

import pytest

from scripts.retire_legacy_patient_memory import (
    TARGET_USER_VERSION,
    inspect_database,
    retire,
)


def _create_legacy_database(path, *, rows: int = 0):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE patient_memory (id INTEGER PRIMARY KEY, content TEXT);
            CREATE TABLE patient_memory_sessions (id INTEGER PRIMARY KEY, content TEXT);
            CREATE TABLE patient_audit_events (
                id INTEGER PRIMARY KEY,
                actor_username TEXT NOT NULL,
                patient_id TEXT,
                action TEXT NOT NULL,
                object_revision INTEGER,
                outcome TEXT NOT NULL,
                trace_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE preserved_data (value TEXT NOT NULL);
            INSERT INTO preserved_data(value) VALUES ('keep');
            """
        )
        for index in range(rows):
            conn.execute(
                "INSERT INTO patient_memory(content) VALUES (?)",
                (f"legacy-{index}",),
            )


def test_retire_empty_legacy_tables_creates_restorable_backup(tmp_path):
    database_path = tmp_path / "voice.db"
    backup_dir = tmp_path / "backups"
    _create_legacy_database(database_path)

    result = retire(database_path, backup_dir=backup_dir)

    assert result["retired"] is True
    assert inspect_database(database_path)["legacy_rows"] == {
        "patient_memory_sessions": None,
        "patient_memory": None,
    }
    with sqlite3.connect(database_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == TARGET_USER_VERSION
    with sqlite3.connect(result["backup"]) as backup:
        assert backup.execute("SELECT value FROM preserved_data").fetchone()[0] == "keep"
        assert backup.execute("SELECT COUNT(*) FROM patient_memory").fetchone()[0] == 0


def test_retire_nonempty_legacy_tables_stops_without_modifying_database(tmp_path):
    database_path = tmp_path / "voice.db"
    backup_dir = tmp_path / "backups"
    _create_legacy_database(database_path, rows=1)

    with pytest.raises(RuntimeError, match="non-empty"):
        retire(database_path, backup_dir=backup_dir)

    assert inspect_database(database_path)["legacy_rows"]["patient_memory"] == 1
    assert not backup_dir.exists()
