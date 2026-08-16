"""
SQLite 数据库模块 - 持久化存储患者信息、对话记录、MMSE评分、录音文件
"""
import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.voice_modes import (
    COGNITIVE_SCREENING,
    LEGACY,
    WELLBEING,
    is_cognitive_screening,
    normalize_session_mode,
)

DB_PATH = os.getenv("DB_PATH", "data/voice_server.db")


class MemoryRevisionConflict(Exception):
    """Raised when a versioned patient-memory write uses a stale revision."""

    def __init__(self, current_revision: int):
        self.current_revision = int(current_revision)
        self.revision = self.current_revision
        super().__init__(f"memory revision conflict: {self.current_revision}")


def get_db_path() -> str:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _parse_notes(notes: Optional[str]) -> Dict[str, Any]:
    if not notes:
        return {}
    try:
        payload = json.loads(notes)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_notes_payload(profile: Optional[Dict[str, Any]], existing_notes: Optional[str] = None) -> Optional[str]:
    payload = _parse_notes(existing_notes)
    extra_profile = {}
    for key, value in (profile or {}).items():
        if key in {"name", "age", "gender", "education_years"}:
            continue
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        extra_profile[key] = value
    if extra_profile:
        payload["extended_profile"] = extra_profile
    else:
        payload.pop("extended_profile", None)
    return json.dumps(payload, ensure_ascii=False) if payload else None


def _build_profile_from_session_row(session: sqlite3.Row) -> Dict[str, Any]:
    profile = {
        "name": session["patient_name"],
        "age": session["patient_age"],
        "gender": session["patient_gender"],
        "education_years": session["education_years"],
    }
    extra_profile = _parse_notes(session["notes"]).get("extended_profile")
    if isinstance(extra_profile, dict):
        profile.update(extra_profile)
    return profile


@contextmanager
def get_conn():
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def init_db():
    """初始化数据库，创建所有表"""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                ended_at        TEXT,
                patient_name    TEXT,
                patient_age     INTEGER,
                patient_gender  TEXT,
                education_years INTEGER,
                owner_username  TEXT,
                total_mmse_score INTEGER,
                cognitive_status TEXT,
                mode            TEXT NOT NULL DEFAULT 'wellbeing',
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                emotion     TEXT,
                language    TEXT,
                turn_id     TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mmse_scores (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL REFERENCES sessions(session_id),
                dimension_id        TEXT NOT NULL,
                score               INTEGER NOT NULL,
                max_score           INTEGER NOT NULL,
                question            TEXT,
                answer              TEXT,
                evaluation_detail   TEXT,
                created_at          TEXT NOT NULL,
                UNIQUE(session_id, dimension_id)
            );

            CREATE TABLE IF NOT EXISTS audio_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                file_path   TEXT NOT NULL,
                duration_s  REAL,
                asr_text    TEXT,
                role        TEXT,
                content_text TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL UNIQUE,
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                display_name    TEXT,
                role            TEXT NOT NULL DEFAULT 'user',
                created_at      TEXT NOT NULL,
                last_login_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS public_api_keys (
                key_id          TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                key_prefix      TEXT NOT NULL,
                key_hash        TEXT NOT NULL UNIQUE,
                created_by      TEXT,
                created_at      TEXT NOT NULL,
                last_used_at    TEXT,
                revoked_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS patients (
                patient_id      TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                gender          TEXT,
                age             INTEGER,
                education_years INTEGER,
                extra_profile   TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                archived_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS patient_assignments (
                patient_id      TEXT NOT NULL REFERENCES patients(patient_id),
                username        TEXT NOT NULL REFERENCES users(username),
                can_read        INTEGER NOT NULL DEFAULT 0 CHECK (can_read IN (0, 1)),
                can_write       INTEGER NOT NULL DEFAULT 0 CHECK (can_write IN (0, 1)),
                can_voice       INTEGER NOT NULL DEFAULT 0 CHECK (can_voice IN (0, 1)),
                assigned_by     TEXT,
                assigned_at     TEXT NOT NULL,
                revoked_at      TEXT,
                PRIMARY KEY (patient_id, username)
            );

            CREATE TABLE IF NOT EXISTS patient_audit_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_username  TEXT,
                patient_id      TEXT,
                action          TEXT NOT NULL,
                object_revision INTEGER,
                outcome         TEXT NOT NULL,
                trace_id        TEXT,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS safety_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      TEXT REFERENCES patients(patient_id),
                session_id      TEXT NOT NULL,
                turn_id         TEXT NOT NULL,
                source          TEXT NOT NULL,
                level           TEXT NOT NULL,
                rule_version    TEXT NOT NULL,
                text_hmac       TEXT NOT NULL,
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                handled_by      TEXT,
                handled_at      TEXT,
                UNIQUE(session_id, turn_id, rule_version)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_mmse_session ON mmse_scores(session_id);
            CREATE INDEX IF NOT EXISTS idx_audio_session ON audio_files(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(patient_name);
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_public_api_keys_hash ON public_api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(name);
            CREATE INDEX IF NOT EXISTS idx_patient_assignments_username ON patient_assignments(username);
            CREATE INDEX IF NOT EXISTS idx_patient_assignments_patient_active ON patient_assignments(patient_id, revoked_at);
            CREATE INDEX IF NOT EXISTS idx_patient_audit_events_patient_created ON patient_audit_events(patient_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_safety_events_patient_created ON safety_events(patient_id, created_at);
        """)
        _ensure_column(conn, "sessions", "owner_username", "owner_username TEXT")
        _ensure_column(conn, "sessions", "patient_id", "patient_id TEXT")
        _ensure_column(
            conn,
            "sessions",
            "mode",
            "mode TEXT NOT NULL DEFAULT 'wellbeing'",
        )
        conn.execute(
            """UPDATE sessions
               SET mode=?
               WHERE (mode IS NULL OR TRIM(mode)='')
                 AND (total_mmse_score IS NOT NULL OR EXISTS (
                     SELECT 1 FROM mmse_scores m
                     WHERE m.session_id=sessions.session_id
                 ))""",
            (COGNITIVE_SCREENING,),
        )
        conn.execute(
            "UPDATE sessions SET mode=? WHERE mode IS NULL OR TRIM(mode)=''",
            (LEGACY,),
        )
        _ensure_column(conn, "audio_files", "role", "role TEXT DEFAULT 'user'")
        _ensure_column(conn, "audio_files", "content_text", "content_text TEXT")
        _ensure_column(conn, "messages", "turn_id", "turn_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id)")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_session_role_turn "
            "ON messages(session_id, role, turn_id) "
            "WHERE turn_id IS NOT NULL AND turn_id<>''"
        )
    print(f"[DB] ✅ 数据库初始化完成: {get_db_path()}")


# ────────────────────────────────────────────
# Public API keys
# ────────────────────────────────────────────

def create_public_api_key(
    key_id: str,
    name: str,
    key_prefix: str,
    key_hash: str,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """保存 API Key 元数据与哈希；明文 Key 不会写入数据库。"""
    now = datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO public_api_keys
               (key_id, name, key_prefix, key_hash, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key_id, name, key_prefix, key_hash, created_by, now),
        )
    return {
        "key_id": key_id,
        "name": name,
        "key_prefix": key_prefix,
        "created_by": created_by,
        "created_at": now,
        "last_used_at": None,
        "revoked_at": None,
    }


def list_public_api_keys() -> List[Dict[str, Any]]:
    """列出可公开展示的 Key 元数据，不返回哈希或明文。"""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT key_id, name, key_prefix, created_by, created_at, last_used_at, revoked_at
               FROM public_api_keys
               ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]


def get_active_public_api_key_by_hash(key_hash: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT key_id, name, key_prefix, created_by, created_at, last_used_at
               FROM public_api_keys
               WHERE key_hash=? AND revoked_at IS NULL
               LIMIT 1""",
            (key_hash,),
        ).fetchone()
        return dict(row) if row else None


def touch_public_api_key(key_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE public_api_keys SET last_used_at=? WHERE key_id=? AND revoked_at IS NULL",
            (datetime.now().isoformat(), key_id),
        )


def revoke_public_api_key(key_id: str) -> bool:
    with get_conn() as conn:
        result = conn.execute(
            """UPDATE public_api_keys
               SET revoked_at=?
               WHERE key_id=? AND revoked_at IS NULL""",
            (datetime.now().isoformat(), key_id),
        )
        return result.rowcount > 0


# ────────────────────────────────────────────
# Patients & 患者长期记忆
# ────────────────────────────────────────────

_PATIENT_CORE_FIELDS = {"name", "gender", "age", "education_years"}


def _split_patient_profile(profile: Dict[str, Any]) -> tuple:
    """把 profile 拆成核心列字段和 extra_profile JSON。"""
    core = {k: profile.get(k) for k in _PATIENT_CORE_FIELDS if profile.get(k) not in (None, "")}
    extra = {
        k: v for k, v in (profile or {}).items()
        if k not in _PATIENT_CORE_FIELDS and k != "patient_id" and v not in (None, "")
    }
    return core, (json.dumps(extra, ensure_ascii=False) if extra else None)


def _patient_row_to_dict(row) -> Dict[str, Any]:
    data = dict(row)
    try:
        data["extra_profile"] = json.loads(data.get("extra_profile") or "{}")
    except Exception:
        data["extra_profile"] = {}
    return data


def create_patient(profile: Dict[str, Any]) -> Dict[str, Any]:
    """创建患者档案，返回完整患者记录（patient_id 为 uuid hex）。"""
    import uuid as _uuid

    patient_id = f"pt_{_uuid.uuid4().hex[:16]}"
    now = datetime.now().isoformat()
    core, extra_json = _split_patient_profile(profile or {})
    name = str(core.get("name") or "").strip()
    if not name:
        raise ValueError("patient name is required")
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, gender, age, education_years, extra_profile, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                name,
                core.get("gender"),
                core.get("age"),
                core.get("education_years"),
                extra_json,
                now,
                now,
            ),
        )
    return get_patient(patient_id)


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id=? AND archived_at IS NULL",
            (patient_id,),
        ).fetchone()
        return _patient_row_to_dict(row) if row else None


def _write_patient_audit_event(
    conn: sqlite3.Connection,
    *,
    actor_username: Optional[str],
    patient_id: Optional[str],
    action: str,
    outcome: str,
    object_revision: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    action = str(action or "").strip()
    outcome = str(outcome or "").strip()
    if not action or not outcome:
        raise ValueError("patient audit action and outcome are required")
    created_at = datetime.now().isoformat()
    result = conn.execute(
        """INSERT INTO patient_audit_events
           (actor_username, patient_id, action, object_revision, outcome, trace_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            str(actor_username or "").strip() or None,
            str(patient_id or "").strip() or None,
            action,
            object_revision,
            outcome,
            str(trace_id or "").strip() or None,
            created_at,
        ),
    )
    return {
        "id": result.lastrowid,
        "actor_username": str(actor_username or "").strip() or None,
        "patient_id": str(patient_id or "").strip() or None,
        "action": action,
        "object_revision": object_revision,
        "outcome": outcome,
        "trace_id": str(trace_id or "").strip() or None,
        "created_at": created_at,
    }


def record_patient_audit_event(
    *,
    actor_username: Optional[str],
    patient_id: Optional[str],
    action: str,
    outcome: str,
    object_revision: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """记录不含患者内容的访问或管理事件。"""
    with get_conn() as conn:
        return _write_patient_audit_event(
            conn,
            actor_username=actor_username,
            patient_id=patient_id,
            action=action,
            outcome=outcome,
            object_revision=object_revision,
            trace_id=trace_id,
        )


def record_safety_event(
    *,
    patient_id: Optional[str],
    session_id: str,
    turn_id: str,
    source: str,
    level: str,
    rule_version: str,
    text_hmac: str,
    status: str,
    handled_by: str = "system",
) -> Dict[str, Any]:
    """Write one idempotent safety disposition without retaining patient text."""
    session_id = str(session_id or "").strip()
    turn_id = str(turn_id or "").strip()
    rule_version = str(rule_version or "").strip()
    if not session_id or not turn_id or not rule_version:
        raise ValueError("session_id, turn_id and rule_version are required")
    now = datetime.now().isoformat()
    with get_conn() as conn:
        inserted = conn.execute(
            """INSERT OR IGNORE INTO safety_events
               (patient_id, session_id, turn_id, source, level, rule_version,
                text_hmac, status, created_at, handled_by, handled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(patient_id or "").strip() or None,
                session_id,
                turn_id,
                str(source or "final").strip() or "final",
                str(level or "unknown").strip() or "unknown",
                rule_version,
                str(text_hmac or "").strip(),
                str(status or "handled").strip() or "handled",
                now,
                str(handled_by or "system").strip() or "system",
                now,
            ),
        ).rowcount > 0
        row = conn.execute(
            """SELECT id, patient_id, session_id, turn_id, source, level,
                      rule_version, text_hmac, status, created_at, handled_by, handled_at
               FROM safety_events
               WHERE session_id=? AND turn_id=? AND rule_version=?""",
            (session_id, turn_id, rule_version),
        ).fetchone()
        if inserted:
            _write_patient_audit_event(
                conn,
                actor_username=handled_by,
                patient_id=patient_id,
                action="safety_disposition",
                outcome=f"{level}:{status}",
            )
    return dict(row)


def list_patient_audit_events(
    patient_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, actor_username, patient_id, action, object_revision,
                      outcome, trace_id, created_at
               FROM patient_audit_events WHERE patient_id=?
               ORDER BY id DESC LIMIT ?""",
            (patient_id, max(1, min(int(limit), 500))),
        ).fetchall()
        return [dict(row) for row in rows]


def get_patient_assignment(
    patient_id: str,
    username: str,
) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT patient_id, username, can_read, can_write, can_voice,
                      assigned_by, assigned_at, revoked_at
               FROM patient_assignments WHERE patient_id=? AND username=?""",
            (patient_id, username),
        ).fetchone()
        return dict(row) if row else None


def assign_patient(
    patient_id: str,
    username: str,
    *,
    can_read: bool = False,
    can_write: bool = False,
    can_voice: bool = False,
    assigned_by: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """创建或恢复一条患者分配，并记录无内容审计事件。"""
    now = datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO patient_assignments
               (patient_id, username, can_read, can_write, can_voice,
                assigned_by, assigned_at, revoked_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
               ON CONFLICT(patient_id, username) DO UPDATE SET
                 can_read=excluded.can_read,
                 can_write=excluded.can_write,
                 can_voice=excluded.can_voice,
                 assigned_by=excluded.assigned_by,
                 assigned_at=excluded.assigned_at,
                 revoked_at=NULL""",
            (
                patient_id,
                username,
                int(bool(can_read)),
                int(bool(can_write)),
                int(bool(can_voice)),
                assigned_by,
                now,
            ),
        )
        _write_patient_audit_event(
            conn,
            actor_username=assigned_by,
            patient_id=patient_id,
            action="patient_assignment_granted",
            outcome="success",
            trace_id=trace_id,
        )
        row = conn.execute(
            """SELECT patient_id, username, can_read, can_write, can_voice,
                      assigned_by, assigned_at, revoked_at
               FROM patient_assignments WHERE patient_id=? AND username=?""",
            (patient_id, username),
        ).fetchone()
        return dict(row) if row else {}


def revoke_patient_assignment(
    patient_id: str,
    username: str,
    *,
    revoked_by: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> bool:
    """撤销一条患者分配；重复撤销不会重新激活记录。"""
    with get_conn() as conn:
        result = conn.execute(
            """UPDATE patient_assignments SET revoked_at=?
               WHERE patient_id=? AND username=? AND revoked_at IS NULL""",
            (datetime.now().isoformat(), patient_id, username),
        )
        _write_patient_audit_event(
            conn,
            actor_username=revoked_by,
            patient_id=patient_id,
            action="patient_assignment_revoked",
            outcome="success" if result.rowcount else "not_found",
            trace_id=trace_id,
        )
        return result.rowcount > 0


def search_patients(query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """按姓名模糊搜索患者，附带最近一次会话时间和得分。"""
    like = f"%{(query or '').strip()}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.*,
                      (SELECT MAX(s.created_at) FROM sessions s WHERE s.patient_id = p.patient_id) AS last_session_at,
                      (SELECT s.total_mmse_score FROM sessions s
                       WHERE s.patient_id = p.patient_id AND s.total_mmse_score IS NOT NULL
                       ORDER BY s.created_at DESC LIMIT 1) AS last_mmse_score
               FROM patients p
               WHERE p.archived_at IS NULL AND p.name LIKE ?
               ORDER BY last_session_at DESC NULLS LAST, p.updated_at DESC
               LIMIT ?""",
            (like, max(1, min(int(limit), 100))),
        ).fetchall()
        return [_patient_row_to_dict(row) for row in rows]


def list_accessible_patients(
    query: str = "",
    limit: int = 100,
    actor_username: str = "",
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """列出当前账号可进行语音绑定的患者。"""
    like = f"%{(query or '').strip()}%"
    params: list[Any] = [like]
    access_clause = "1=1"
    if not is_admin:
        access_clause = """
            EXISTS (
                SELECT 1
                FROM patient_assignments pa
                WHERE pa.patient_id = p.patient_id
                  AND pa.username = ?
                  AND pa.can_voice = 1
                  AND pa.revoked_at IS NULL
            )
        """
        params.append(str(actor_username or "").strip())
    params.append(max(1, min(int(limit), 100)))
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT p.*,
                      (SELECT MAX(s.created_at) FROM sessions s
                       WHERE s.patient_id = p.patient_id) AS last_session_at,
                      (SELECT s.total_mmse_score FROM sessions s
                       WHERE s.patient_id = p.patient_id
                         AND s.total_mmse_score IS NOT NULL
                       ORDER BY s.created_at DESC LIMIT 1) AS last_mmse_score
               FROM patients p
               WHERE p.archived_at IS NULL
                 AND p.name LIKE ?
                 AND {access_clause}
               ORDER BY last_session_at DESC NULLS LAST, p.updated_at DESC
               LIMIT ?""",
            params,
        ).fetchall()
        return [_patient_row_to_dict(row) for row in rows]


def update_patient_profile(patient_id: str, profile: Dict[str, Any]) -> None:
    """用最新会话录入的信息刷新患者档案（仅覆盖有值字段）。"""
    existing = get_patient(patient_id)
    if not existing:
        return
    core, _ = _split_patient_profile(profile or {})
    merged_extra = dict(existing.get("extra_profile") or {})
    _, extra_json_new = _split_patient_profile(profile or {})
    if extra_json_new:
        merged_extra.update(json.loads(extra_json_new))
    with get_conn() as conn:
        conn.execute(
            """UPDATE patients SET name=?, gender=?, age=?, education_years=?, extra_profile=?, updated_at=?
               WHERE patient_id=?""",
            (
                core.get("name") or existing.get("name"),
                core.get("gender") or existing.get("gender"),
                core.get("age") if core.get("age") is not None else existing.get("age"),
                core.get("education_years") if core.get("education_years") is not None else existing.get("education_years"),
                json.dumps(merged_extra, ensure_ascii=False) if merged_extra else None,
                datetime.now().isoformat(),
                patient_id,
            ),
        )


def link_session_patient(session_id: str, patient_id: str) -> None:
    with get_conn() as conn:
        _ensure_column(conn, "sessions", "patient_id", "patient_id TEXT")
        conn.execute(
            "UPDATE sessions SET patient_id=? WHERE session_id=?",
            (patient_id, session_id),
        )


def get_patient_score_history(patient_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    """按时间正序返回该患者历次筛查得分（可信数据，用于记忆卡趋势行）。"""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT session_id, created_at, ended_at, total_mmse_score, cognitive_status
               FROM sessions
               WHERE patient_id=? AND total_mmse_score IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (patient_id, max(1, min(int(limit), 50))),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


# ────────────────────────────────────────────
# Sessions
# ────────────────────────────────────────────

def create_session(session_id: str, profile: Optional[Dict[str, Any]] = None,
                   owner_username: Optional[str] = None, patient_id: Optional[str] = None,
                   mode: str = WELLBEING):
    now = datetime.now().isoformat()
    normalized_mode = normalize_session_mode(mode)
    with get_conn() as conn:
        _ensure_column(conn, "sessions", "owner_username", "owner_username TEXT")
        _ensure_column(conn, "sessions", "patient_id", "patient_id TEXT")
        _ensure_column(conn, "sessions", "mode", "mode TEXT NOT NULL DEFAULT 'wellbeing'")
        conn.execute(
            """INSERT INTO sessions
               (session_id, created_at, patient_name, patient_age, patient_gender,
                education_years, owner_username, patient_id, mode, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                profile.get("name") if profile else None,
                profile.get("age") if profile else None,
                profile.get("gender") if profile else None,
                profile.get("education_years") if profile else None,
                owner_username,
                patient_id,
                normalized_mode,
                _build_notes_payload(profile),
            ),
        )


def assign_session_owner(session_id: str, owner_username: Optional[str], overwrite: bool = False) -> None:
    if not owner_username:
        return
    with get_conn() as conn:
        _ensure_column(conn, "sessions", "owner_username", "owner_username TEXT")
        if overwrite:
            conn.execute(
                "UPDATE sessions SET owner_username=? WHERE session_id=?",
                (owner_username, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET owner_username=? WHERE session_id=? AND COALESCE(owner_username, '')=''",
                (owner_username, session_id),
            )


def end_session(session_id: str, total_mmse_score: Optional[int] = None,
                cognitive_status: Optional[str] = None):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT mode FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row or not is_cognitive_screening(row["mode"]):
            total_mmse_score = None
            cognitive_status = None
        conn.execute(
            """UPDATE sessions SET ended_at=?, total_mmse_score=?, cognitive_status=?
               WHERE session_id=?""",
            (datetime.now().isoformat(), total_mmse_score, cognitive_status, session_id),
        )


def update_session_profile(session_id: str, profile: Dict[str, Any]):
    with get_conn() as conn:
        existing_row = conn.execute(
            "SELECT notes FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        conn.execute(
            """UPDATE sessions SET patient_name=?, patient_age=?, patient_gender=?, education_years=?, notes=?
               WHERE session_id=?""",
            (
                profile.get("name"),
                profile.get("age"),
                profile.get("gender"),
                profile.get("education_years"),
                _build_notes_payload(profile, existing_row["notes"] if existing_row else None),
                session_id,
            ),
        )


# ────────────────────────────────────────────
# Messages
# ────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    emotion: Optional[str] = None,
    language: Optional[str] = None,
    turn_id: Optional[str] = None,
) -> bool:
    normalized_turn_id = str(turn_id or "").strip() or None
    with get_conn() as conn:
        if normalized_turn_id:
            result = conn.execute(
                """INSERT OR IGNORE INTO messages
                   (session_id, role, content, emotion, language, turn_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    content,
                    emotion,
                    language,
                    normalized_turn_id,
                    datetime.now().isoformat(),
                ),
            )
            return result.rowcount > 0
        conn.execute(
            """INSERT INTO messages
               (session_id, role, content, emotion, language, turn_id, created_at)
               VALUES (?, ?, ?, ?, ?, NULL, ?)""",
            (session_id, role, content, emotion, language, datetime.now().isoformat()),
        )
        return True


# ────────────────────────────────────────────
# MMSE Scores
# ────────────────────────────────────────────

def is_cognitive_screening_session(session_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT mode FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return bool(row and is_cognitive_screening(row["mode"]))


def save_mmse_score(session_id: str, dimension_id: str, score: int, max_score: int,
                    question: str = "", answer: str = "", evaluation_detail: str = ""):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT mode FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row or not is_cognitive_screening(row["mode"]):
            return False
        conn.execute(
            """INSERT INTO mmse_scores
               (session_id, dimension_id, score, max_score, question, answer, evaluation_detail, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, dimension_id) DO UPDATE SET
                 score=excluded.score, max_score=excluded.max_score,
                 question=excluded.question, answer=excluded.answer,
                 evaluation_detail=excluded.evaluation_detail,
                 created_at=excluded.created_at""",
            (session_id, dimension_id, score, max_score,
             question, answer, evaluation_detail, datetime.now().isoformat()),
        )
        return True


# ────────────────────────────────────────────
# Audio Files
# ────────────────────────────────────────────

def save_audio_record(session_id: str, file_path: str,
                      duration_s: Optional[float] = None, asr_text: Optional[str] = None,
                      role: str = "user", content_text: Optional[str] = None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO audio_files (session_id, file_path, duration_s, asr_text, role, content_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, file_path, duration_s, asr_text, role, content_text, datetime.now().isoformat()),
        )


# ────────────────────────────────────────────
# Query Helpers
# ────────────────────────────────────────────

def list_sessions(limit: int = 50, offset: int = 0,
                  name_filter: Optional[str] = None,
                  actor_username: Optional[str] = None,
                  is_admin: bool = True) -> List[Dict]:
    with get_conn() as conn:
        query = """
            SELECT
                s.*,
                COALESCE(msg.message_count, 0) AS message_count,
                COALESCE(aud.audio_count, 0) AS audio_count,
                COALESCE(sc.score_count, 0) AS score_count,
                COALESCE(sc.total_score, s.total_mmse_score) AS derived_mmse_score,
                COALESCE(sc.total_max_score, 35) AS derived_mmse_max_score,
                MAX(
                    COALESCE(msg.last_message_at, ''),
                    COALESCE(aud.last_audio_at, ''),
                    COALESCE(sc.last_score_at, ''),
                    COALESCE(s.created_at, '')
                ) AS last_activity_at
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS message_count, MAX(created_at) AS last_message_at
                FROM messages
                GROUP BY session_id
            ) msg ON msg.session_id = s.session_id
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS audio_count, MAX(created_at) AS last_audio_at
                FROM audio_files
                GROUP BY session_id
            ) aud ON aud.session_id = s.session_id
            LEFT JOIN (
                SELECT session_id,
                       COUNT(*) AS score_count,
                       SUM(score) AS total_score,
                       SUM(max_score) AS total_max_score,
                       MAX(created_at) AS last_score_at
                FROM mmse_scores
                GROUP BY session_id
            ) sc ON sc.session_id = s.session_id
            WHERE (
                COALESCE(msg.message_count, 0) > 0 OR
                COALESCE(aud.audio_count, 0) > 0 OR
                COALESCE(sc.score_count, 0) > 0
            )
        """
        params: List[Any] = []
        if name_filter:
            query += " AND COALESCE(s.patient_name, '') LIKE ?"
            params.append(f"%{name_filter}%")
        if not is_admin:
            username = str(actor_username or "").strip()
            if not username:
                return []
            query += """
                AND (
                    COALESCE(s.owner_username, '') = ?
                    OR EXISTS (
                        SELECT 1 FROM patient_assignments pa
                        WHERE pa.patient_id = s.patient_id
                          AND pa.username = ?
                          AND pa.can_read = 1
                          AND pa.revoked_at IS NULL
                    )
                )
            """
            params.extend([username, username])
        query += " ORDER BY last_activity_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def can_user_access_session(session_id: str, username: str) -> bool:
    username = str(username or "").strip()
    session_id = str(session_id or "").strip()
    if not username or not session_id:
        return False
    with get_conn() as conn:
        row = conn.execute(
            """SELECT session_id, owner_username, patient_id
               FROM sessions WHERE session_id=?""",
            (session_id,),
        ).fetchone()
        if not row:
            return False
        if str(row["owner_username"] or "").strip() == username:
            return True
        patient_id = str(row["patient_id"] or "").strip()
        if not patient_id:
            return False
        assignment = conn.execute(
            """SELECT can_read, revoked_at FROM patient_assignments
               WHERE patient_id=? AND username=?""",
            (patient_id, username),
        ).fetchone()
        return bool(
            assignment
            and assignment["can_read"] == 1
            and not assignment["revoked_at"]
        )


def get_session_for_resume(session_id: str) -> Optional[Dict]:
    """获取会话恢复所需的全部数据：患者信息 + 对话历史 + MMSE状态"""
    with get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not session:
            return None
        msgs = conn.execute(
            "SELECT role, content, emotion, language, turn_id FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        scores = conn.execute(
            """SELECT dimension_id, score, max_score, question, answer, evaluation_detail, created_at
               FROM mmse_scores WHERE session_id=?""",
            (session_id,),
        ).fetchall()
        return {
            "session_id": session["session_id"],
            "patient_id": session["patient_id"],
            "mode": normalize_session_mode(session["mode"], default=LEGACY),
            "profile": _build_profile_from_session_row(session),
            "chat_history": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    **({"turn_id": m["turn_id"]} if m["turn_id"] else {}),
                }
                for m in msgs
            ],
            "messages": [dict(m) for m in msgs],
            "mmse_scores": {
                s["dimension_id"]: {
                    "score": s["score"],
                    "max_score": s["max_score"],
                    "question": s["question"],
                    "answer": s["answer"],
                    "evaluation_detail": s["evaluation_detail"],
                    "created_at": s["created_at"],
                }
                for s in scores
            },
            "ended_at": session["ended_at"],
        }


def get_session_resume_binding(session_id: str) -> Optional[Dict[str, Any]]:
    """Return only the data needed to authorize a session-resume request."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT patient_id, ended_at, mode FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


def get_active_sessions(limit: int = 20) -> List[Dict]:
    """获取未结束的会话列表（可恢复的）"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.patient_name,
                s.patient_age,
                s.patient_gender,
                s.created_at,
                s.total_mmse_score,
                COALESCE(msg.message_count, 0) AS message_count,
                COALESCE(aud.audio_count, 0) AS audio_count,
                MAX(
                    COALESCE(msg.last_message_at, ''),
                    COALESCE(aud.last_audio_at, ''),
                    COALESCE(s.created_at, '')
                ) AS last_activity_at
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS message_count, MAX(created_at) AS last_message_at
                FROM messages
                GROUP BY session_id
            ) msg ON msg.session_id = s.session_id
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS audio_count, MAX(created_at) AS last_audio_at
                FROM audio_files
                GROUP BY session_id
            ) aud ON aud.session_id = s.session_id
            WHERE s.ended_at IS NULL
              AND (
                  COALESCE(msg.message_count, 0) > 0 OR
                  COALESCE(aud.audio_count, 0) > 0
              )
            ORDER BY last_activity_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_detail(session_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not session:
            return None
        msgs = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        scores = conn.execute(
            "SELECT * FROM mmse_scores WHERE session_id=?", (session_id,)
        ).fetchall()
        audios = conn.execute(
            "SELECT * FROM audio_files WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return {
            "session": dict(session),
            "profile": _build_profile_from_session_row(session),
            "messages": [dict(m) for m in msgs],
            "mmse_scores": [dict(s) for s in scores],
            "audio_files": [dict(a) for a in audios],
        }


def get_audio_record(session_id: str, audio_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM audio_files WHERE session_id=? AND id=?",
            (session_id, audio_id),
        ).fetchone()
        return dict(row) if row else None


def list_audio_records(session_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audio_files WHERE session_id=? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


# ────────────────────────────────────────────
# Users / Auth
# ────────────────────────────────────────────

def create_user(username: str, password_hash: str, salt: str,
                display_name: Optional[str] = None, role: str = "user") -> Dict[str, Any]:
    now = datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO users (username, password_hash, salt, display_name, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, password_hash, salt, display_name or username, role, now),
        )
        row = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users WHERE username=?",
            (username,),
        ).fetchone()
        return dict(row) if row else {}


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def list_users() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users ORDER BY created_at ASC"
        ).fetchall()
        return [dict(row) for row in rows]


def update_user_last_login(username: str) -> None:
    now = datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login_at=? WHERE username=?",
            (now, username),
        )


def count_admin_users() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM users WHERE role='admin'").fetchone()
        return int((row["count"] if row else 0) or 0)


def update_user_profile(username: str, display_name: Optional[str] = None, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    assignments: List[str] = []
    params: List[Any] = []
    if display_name is not None:
        assignments.append("display_name=?")
        params.append(display_name)
    if role is not None:
        assignments.append("role=?")
        params.append(role)
    if not assignments:
        return get_user_by_username(username)
    params.append(username)
    with get_conn() as conn:
        conn.execute(f"UPDATE users SET {', '.join(assignments)} WHERE username=?", tuple(params))
        row = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users WHERE username=?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def update_user_password(username: str, password_hash: str, salt: str) -> bool:
    with get_conn() as conn:
        result = conn.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE username=?",
            (password_hash, salt, username),
        )
        return result.rowcount > 0


def ensure_bootstrap_admin() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        existing_admin = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users WHERE role='admin' ORDER BY id LIMIT 1"
        ).fetchone()
        if existing_admin:
            return dict(existing_admin)
        first_user = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users ORDER BY id LIMIT 1"
        ).fetchone()
        if not first_user:
            return None
        conn.execute("UPDATE users SET role='admin' WHERE id=?", (first_user["id"],))
        updated = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users WHERE id=?",
            (first_user["id"],),
        ).fetchone()
        return dict(updated) if updated else None


def get_admin_usage_snapshot(days_active: int = 7, recent_limit: int = 12) -> Dict[str, Any]:
    with get_conn() as conn:
        _ensure_column(conn, "sessions", "owner_username", "owner_username TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_username)")
        user_rows = [dict(row) for row in conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM users ORDER BY created_at ASC"
        ).fetchall()]
        session_rows = [dict(row) for row in conn.execute(
            """
            SELECT
                s.session_id,
                s.owner_username,
                s.created_at,
                s.ended_at,
                s.patient_name,
                s.patient_age,
                s.patient_gender,
                s.total_mmse_score,
                s.cognitive_status,
                COALESCE(msg.message_count, 0) AS message_count,
                COALESCE(aud.audio_count, 0) AS audio_count,
                COALESCE(aud.total_audio_duration_s, 0) AS total_audio_duration_s,
                COALESCE(sc.score_count, 0) AS score_count,
                COALESCE(sc.total_score, s.total_mmse_score, 0) AS derived_mmse_score,
                COALESCE(sc.total_max_score, 35) AS derived_mmse_max_score,
                MAX(
                    COALESCE(msg.last_message_at, ''),
                    COALESCE(aud.last_audio_at, ''),
                    COALESCE(sc.last_score_at, ''),
                    COALESCE(s.created_at, '')
                ) AS last_activity_at
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS message_count, MAX(created_at) AS last_message_at
                FROM messages
                GROUP BY session_id
            ) msg ON msg.session_id = s.session_id
            LEFT JOIN (
                SELECT session_id,
                       COUNT(*) AS audio_count,
                       COALESCE(SUM(duration_s), 0) AS total_audio_duration_s,
                       MAX(created_at) AS last_audio_at
                FROM audio_files
                GROUP BY session_id
            ) aud ON aud.session_id = s.session_id
            LEFT JOIN (
                SELECT session_id,
                       COUNT(*) AS score_count,
                       COALESCE(SUM(score), 0) AS total_score,
                       COALESCE(SUM(max_score), 35) AS total_max_score,
                       MAX(created_at) AS last_score_at
                FROM mmse_scores
                GROUP BY session_id
            ) sc ON sc.session_id = s.session_id
            ORDER BY last_activity_at DESC, s.created_at DESC
            """
        ).fetchall()]

    def _parse_ts(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    active_cutoff = datetime.now() - timedelta(days=max(1, int(days_active or 7)))
    usage_by_user: Dict[str, Dict[str, Any]] = {}
    for user in user_rows:
        usage_by_user[user["username"]] = {
            "username": user["username"],
            "display_name": user.get("display_name") or user["username"],
            "role": user.get("role") or "user",
            "created_at": user.get("created_at"),
            "last_login_at": user.get("last_login_at"),
            "session_count": 0,
            "active_session_count": 0,
            "message_count": 0,
            "audio_count": 0,
            "audio_duration_s": 0.0,
            "score_count": 0,
            "last_activity_at": user.get("last_login_at"),
            "latest_patient_name": "",
            "latest_session_id": "",
        }

    overview = {
        "total_users": len(user_rows),
        "admin_users": sum(1 for user in user_rows if (user.get("role") or "user") == "admin"),
        "active_users_7d": 0,
        "total_sessions": len(session_rows),
        "assigned_sessions": 0,
        "unassigned_sessions": 0,
        "active_sessions": 0,
        "message_count": 0,
        "audio_count": 0,
        "audio_duration_s": 0.0,
    }

    daily_map: Dict[str, Dict[str, Any]] = {}
    recent_sessions: List[Dict[str, Any]] = []
    unassigned_summary = {
        "session_count": 0,
        "message_count": 0,
        "audio_count": 0,
        "audio_duration_s": 0.0,
    }

    for session in session_rows:
        owner_username = (session.get("owner_username") or "").strip()
        activity_at = session.get("last_activity_at") or session.get("created_at")
        created_at = session.get("created_at")
        message_count = int(session.get("message_count") or 0)
        audio_count = int(session.get("audio_count") or 0)
        score_count = int(session.get("score_count") or 0)
        audio_duration_s = float(session.get("total_audio_duration_s") or 0.0)
        is_active = not session.get("ended_at")

        overview["message_count"] += message_count
        overview["audio_count"] += audio_count
        overview["audio_duration_s"] += audio_duration_s
        if is_active:
            overview["active_sessions"] += 1

        date_key = str(created_at or "")[:10] if created_at else ""
        if date_key:
            bucket = daily_map.setdefault(date_key, {"date": date_key, "session_count": 0, "usernames": set()})
            bucket["session_count"] += 1
            if owner_username:
                bucket["usernames"].add(owner_username)

        if owner_username and owner_username in usage_by_user:
            overview["assigned_sessions"] += 1
            user_usage = usage_by_user[owner_username]
            user_usage["session_count"] += 1
            user_usage["active_session_count"] += 1 if is_active else 0
            user_usage["message_count"] += message_count
            user_usage["audio_count"] += audio_count
            user_usage["audio_duration_s"] += audio_duration_s
            user_usage["score_count"] += score_count
            latest_seen = _parse_ts(user_usage.get("last_activity_at"))
            current_seen = _parse_ts(activity_at)
            if current_seen and (latest_seen is None or current_seen >= latest_seen):
                user_usage["last_activity_at"] = activity_at
                user_usage["latest_patient_name"] = session.get("patient_name") or ""
                user_usage["latest_session_id"] = session.get("session_id") or ""
        else:
            overview["unassigned_sessions"] += 1
            unassigned_summary["session_count"] += 1
            unassigned_summary["message_count"] += message_count
            unassigned_summary["audio_count"] += audio_count
            unassigned_summary["audio_duration_s"] += audio_duration_s

        owner_user = usage_by_user.get(owner_username) if owner_username else None
        recent_sessions.append({
            "session_id": session.get("session_id"),
            "owner_username": owner_username or "",
            "owner_display_name": (owner_user or {}).get("display_name") or (owner_username or "未归属"),
            "patient_name": session.get("patient_name") or "未填写患者",
            "created_at": created_at,
            "last_activity_at": activity_at,
            "ended_at": session.get("ended_at"),
            "message_count": message_count,
            "audio_count": audio_count,
            "audio_duration_s": round(audio_duration_s, 2),
            "score_count": score_count,
            "derived_mmse_score": session.get("derived_mmse_score") or 0,
            "derived_mmse_max_score": session.get("derived_mmse_max_score") or 35,
            "cognitive_status": session.get("cognitive_status") or "",
        })

    for user_usage in usage_by_user.values():
        latest_seen = max(
            (_parse_ts(user_usage.get("last_login_at")) or datetime.min),
            (_parse_ts(user_usage.get("last_activity_at")) or datetime.min),
        )
        if latest_seen >= active_cutoff:
            overview["active_users_7d"] += 1

    users_sorted = sorted(
        usage_by_user.values(),
        key=lambda item: (
            _parse_ts(item.get("last_activity_at")) or datetime.min,
            _parse_ts(item.get("last_login_at")) or datetime.min,
            _parse_ts(item.get("created_at")) or datetime.min,
        ),
        reverse=True,
    )
    daily_activity = sorted(
        (
            {
                "date": item["date"],
                "session_count": item["session_count"],
                "active_user_count": len(item["usernames"]),
            }
            for item in daily_map.values()
        ),
        key=lambda item: item["date"],
        reverse=True,
    )[:14]

    return {
        "overview": {
            **overview,
            "audio_duration_minutes": round(float(overview["audio_duration_s"]) / 60.0, 1),
        },
        "users": [
            {
                **item,
                "audio_duration_minutes": round(float(item["audio_duration_s"]) / 60.0, 1),
            }
            for item in users_sorted
        ],
        "recent_sessions": recent_sessions[:max(1, int(recent_limit or 12))],
        "daily_activity": list(reversed(daily_activity)),
        "unassigned_summary": {
            **unassigned_summary,
            "audio_duration_minutes": round(float(unassigned_summary["audio_duration_s"]) / 60.0, 1),
        },
    }
