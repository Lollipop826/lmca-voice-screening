"""One-time, guarded retirement of the former patient-memory tables."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


LEGACY_TABLES = ("patient_memory_sessions", "patient_memory")
TARGET_USER_VERSION = 20260802


def _inspect_connection(conn: sqlite3.Connection, path: Path) -> dict[str, object]:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    return {
        "path": str(path.resolve()),
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "legacy_rows": {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if table in tables
            else None
            for table in LEGACY_TABLES
        },
    }


def inspect_database(path: Path) -> dict[str, object]:
    with sqlite3.connect(path) as conn:
        return _inspect_connection(conn, path)


def backup_database(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / (
        f"{path.stem}.pre-unified-memory-{datetime.now():%Y%m%d-%H%M%S}.db"
    )
    with sqlite3.connect(path) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def retire(path: Path, *, backup_dir: Path, dry_run: bool = False) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = inspect_database(path)
    nonempty = {
        table: count
        for table, count in before["legacy_rows"].items()
        if count not in (None, 0)
    }
    if nonempty:
        raise RuntimeError(
            "legacy memory tables are non-empty; manual migration approval is required: "
            + json.dumps(nonempty, ensure_ascii=False)
        )
    if dry_run:
        return {**before, "dry_run": True, "retired": False}

    backup = backup_database(path, backup_dir)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        current = _inspect_connection(conn, path)
        nonempty = {
            table: count
            for table, count in current["legacy_rows"].items()
            if count not in (None, 0)
        }
        if nonempty:
            conn.rollback()
            raise RuntimeError("legacy table changed during migration")
        for table in LEGACY_TABLES:
            if current["legacy_rows"][table] is not None:
                conn.execute(f'DROP TABLE "{table}"')
        conn.execute(f"PRAGMA user_version={TARGET_USER_VERSION}")
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_audit_events'"
        ).fetchone():
            conn.execute(
                """INSERT INTO patient_audit_events
                   (actor_username, patient_id, action, outcome, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("migration", None, "legacy_memory_retired", "success", datetime.now().isoformat()),
            )
        conn.commit()
    return {
        **inspect_database(path),
        "backup": str(backup.resolve()),
        "dry_run": False,
        "retired": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backup_dir = args.backup_dir or args.db.resolve().parent / "backups"
    print(json.dumps(retire(args.db, backup_dir=backup_dir, dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
