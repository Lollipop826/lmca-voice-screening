"""Local-first long-term memory for patient conversations.

SQLite is the source of truth. Local Memobase supplies semantic recall and is
kept as a recoverable mirror, so its failure never breaks patient dialogue.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from src.db import database as _database
from src.voice_modes import WELLBEING, normalize_session_mode


# Keep HTTP/API and memory implementations on one conflict type.
MemoryRevisionConflict = _database.MemoryRevisionConflict


_EMOTIONS = ("joy", "sadness", "anger", "fear", "anxiety", "calm", "confusion")
_EMOTION_LABELS = {
    "快乐": "joy",
    "开心": "joy",
    "喜悦": "joy",
    "悲伤": "sadness",
    "难过": "sadness",
    "生气": "anger",
    "愤怒": "anger",
    "害怕": "fear",
    "恐惧": "fear",
    "焦虑": "anxiety",
    "平静": "calm",
    "困惑": "confusion",
}
_EMOTION_LABELS.update(
    {
        "happy": "joy",
        "positive": "joy",
        "sad": "sadness",
        "negative": "sadness",
        "angry": "anger",
        "fearful": "fear",
        "anxious": "anxiety",
        "relaxed": "calm",
        "confused": "confusion",
        "neutral": "calm",
    }
)
_EMOTION_KEYWORDS = {
    "joy": ("开心", "高兴", "快乐", "喜悦", "幸福", "喜欢", "满意", "笑", "good", "happy"),
    "sadness": ("难过", "伤心", "悲伤", "失落", "孤独", "哭", "遗憾", "sad", "upset"),
    "anger": ("生气", "愤怒", "恼火", "烦死", "讨厌", "气死", "angry", "mad"),
    "fear": ("害怕", "恐惧", "担心", "不敢", "危险", "恐慌", "afraid", "scared"),
    "anxiety": ("焦虑", "紧张", "不安", "压力", "忧虑", "着急", "忐忑", "anxious", "nervous"),
    "calm": ("平静", "安心", "放松", "踏实", "calm", "relaxed"),
    "confusion": ("不明白", "弄不清", "困惑", "糊涂", "想不起来", "忘了", "不知道", "confused"),
}
_LIST_FIELDS = ("facts", "preferences", "events", "mmse_history", "comfort_strategies")
_SNAPSHOT_TABLE = "emotion_memobase_snapshots"
_TURN_TABLE = "emotion_memobase_turns"
_SYNC_TABLE = "emotion_memobase_sync_state"
_MEMORY_ITEM_TABLE = "emotion_memobase_memory_items"
_MEMORY_DELETION_TABLE = "emotion_memobase_memory_deletions"
_MAX_EMOTION_HISTORY = 30
_MAX_MMSE_HISTORY = 50
_MAX_TURNS_IN_SNAPSHOT = 50
_MAX_TURNS_IN_CONTEXT = 12
_MAX_CONTEXT_CHARS = 7000
_MEMOBASE_RETRY_SECONDS = 10.0
_EVENT_GIST_TOPK = 3
_EVENT_GIST_SIMILARITY_THRESHOLD = float(
    os.getenv("MEMOBASE_EVENT_GIST_SIMILARITY_THRESHOLD", "0.35")
)
_EVENT_GIST_TIME_RANGE_DAYS = os.getenv("MEMOBASE_EVENT_GIST_TIME_RANGE_DAYS", "").strip()
_LONG_TERM_EVENT_TOKEN_BUDGET = int(
    os.getenv("LONG_TERM_EVENT_TOKEN_BUDGET", "900")
)
_SYNC_LEASE_SECONDS = 60
_SYNC_RETRY_MAX_SECONDS = 300
_CONSOLIDATABLE_FIELDS = {
    "facts", "preferences", "events", "comfort_strategies", "narrative"
}

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _path_key(path: str) -> str:
    return str(Path(path).resolve())


def _lock_for(path: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(_path_key(path), threading.RLock())


def _now() -> str:
    return datetime.now().isoformat()


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _token_estimate(text: str) -> int:
    return max(1, (len(str(text or "")) + 1) // 2)


def _is_local_memobase_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
        "memobase-server-api",
    }


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("memory update must be JSON serializable") from exc


def _strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("expected a list of strings")
    result = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _default_emotion() -> dict[str, float]:
    return {name: 0.0 for name in _EMOTIONS}


def _dominant_emotion(scores: Mapping[str, Any]) -> str:
    values = {
        name: float(scores.get(name, 0.0) or 0.0)
        for name in _EMOTIONS
    }
    highest = max(values.values(), default=0.0)
    if highest <= 0 or max(values.values()) - min(values.values()) < 1e-9:
        return "calm"
    return max(values, key=values.get)


def _parse_classifier_result(result: Any) -> dict[str, float]:
    if isinstance(result, Mapping) and isinstance(result.get("scores"), Mapping):
        result = result["scores"]
    if isinstance(result, list):
        result = result[0] if result else {}
    if isinstance(result, Mapping) and "label" in result:
        result = {result.get("label"): result.get("score", 1.0)}
    if not isinstance(result, Mapping):
        return {}

    scores = _default_emotion()
    for raw_label, raw_score in result.items():
        label = _EMOTION_LABELS.get(str(raw_label), str(raw_label))
        if label not in scores:
            continue
        try:
            scores[label] = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            continue
    return scores if any(scores.values()) else {}


def _keyword_emotion(text: str) -> dict[str, float]:
    scores = _default_emotion()
    lowered = text.lower()
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        hits = sum(lowered.count(keyword.lower()) for keyword in keywords)
        scores[emotion] = min(1.0, hits * 0.35)
    return scores


def _normalise_emotion(scores: Mapping[str, Any]) -> dict[str, float]:
    nested = scores.get("scores") if isinstance(scores, Mapping) else None
    if isinstance(nested, Mapping):
        scores = nested
    values = {}
    for name in _EMOTIONS:
        try:
            values[name] = max(0.0, float(scores.get(name, 0.0) or 0.0))
        except (TypeError, ValueError):
            values[name] = 0.0
    total = sum(values.values())
    if total <= 0:
        return {name: 1.0 / len(_EMOTIONS) for name in _EMOTIONS}
    return {name: value / total for name, value in values.items()}


def _emotion_valence(scores: Mapping[str, float]) -> float:
    return round(
        float(scores.get("joy", 0.0))
        + float(scores.get("calm", 0.0))
        - float(scores.get("sadness", 0.0))
        - float(scores.get("anger", 0.0)),
        6,
    )


def _emotion_arousal(scores: Mapping[str, float]) -> float:
    return round(
        float(scores.get("anxiety", 0.0))
        + float(scores.get("anger", 0.0))
        + float(scores.get("fear", 0.0)),
        6,
    )


def _classify_emotion(
    text: str,
    classifier: Optional[Callable[[str], Any]] = None,
) -> dict[str, float]:
    if classifier is not None:
        try:
            parsed = _parse_classifier_result(classifier(text))
            if parsed:
                return _normalise_emotion(parsed)
        except Exception:
            pass
    return _normalise_emotion(_keyword_emotion(text))


def _extract_memory_items(text: str) -> dict[str, list[str]]:
    """Extract only explicit, low-risk facts for the local fallback."""
    import re

    facts: list[str] = []
    preferences: list[str] = []
    events: list[str] = []
    for pattern in (
        r"(?:我叫|我的名字是|名字叫)\s*([^，。！？；\s]{1,20})",
        r"(?:我今年|今年我)\s*(\d{1,3})\s*岁",
    ):
        for match in re.finditer(pattern, text):
            facts.append(match.group(0).strip())
    for match in re.finditer(
        r"(?:我喜欢|我爱|平时喜欢|平常喜欢)\s*([^。！？；\n]{1,80})",
        text,
    ):
        preferences.append(match.group(0).strip())
    if any(token in text for token in ("今天", "昨天", "最近", "上周", "孙子", "孙女", "儿子", "女儿")):
        events.append(_clip(text, 240))
    return {
        "facts": list(dict.fromkeys(facts)),
        "preferences": list(dict.fromkeys(preferences)),
        "events": list(dict.fromkeys(events)),
    }


def _default_snapshot(patient_id: str, profile: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "patient_id": patient_id,
        "revision": 0,
        "last_consolidated_turn_id": 0,
        "profile": dict(profile or {}),
        "facts": [],
        "preferences": [],
        "events": [],
        "emotion": {"latest": None, "history": []},
        "mmse_history": [],
        "comfort_strategies": [],
        "narrative": "",
        "updated_at": None,
    }


def _normalise_snapshot(
    patient_id: str,
    raw: Any,
    profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    data = _default_snapshot(patient_id, profile)
    if isinstance(raw, Mapping):
        data.update(_json_copy(dict(raw)))
    if not isinstance(data.get("profile"), Mapping):
        data["profile"] = {}
    merged_profile = dict(profile or {})
    merged_profile.update(data["profile"])
    data["profile"] = merged_profile
    for field in _LIST_FIELDS:
        if not isinstance(data.get(field), list):
            data[field] = []
    if not isinstance(data.get("emotion"), Mapping):
        data["emotion"] = {"latest": None, "history": []}
    data["emotion"] = dict(data["emotion"])
    if not isinstance(data["emotion"].get("history"), list):
        data["emotion"]["history"] = []
    if "mmse_scores" in data and not data["mmse_history"]:
        data["mmse_history"] = data["mmse_scores"]
    if "emotion_history" in data and not data["emotion"].get("history"):
        data["emotion"]["history"] = data["emotion_history"]
    data["patient_id"] = patient_id
    data["narrative"] = str(data.get("narrative") or "")
    try:
        data["revision"] = max(0, int(data.get("revision") or 0))
    except (TypeError, ValueError):
        data["revision"] = 0
    try:
        data["last_consolidated_turn_id"] = max(
            0, int(data.get("last_consolidated_turn_id") or 0)
        )
    except (TypeError, ValueError):
        data["last_consolidated_turn_id"] = 0
    return data


def _memory_item_text(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("text", "value", "fact", "preference", "event", "strategy"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""
    return str(item or "").strip()


def _event_text_key(item: Any) -> str:
    import re

    return re.sub(r"[\s，。！？；、,.!?;:：]+", "", _memory_item_text(item))


def _events_overlap(first: Any, second: Any) -> bool:
    first_key = _event_text_key(first)
    second_key = _event_text_key(second)
    if not first_key or not second_key:
        return False
    shorter, longer = sorted((first_key, second_key), key=len)
    return len(shorter) >= 7 and shorter in longer


def _event_is_user_managed(item: Any) -> bool:
    return isinstance(item, Mapping) and str(item.get("source") or "").lower() in {"manual", "user"}


def _merge_equivalent_events(first: Any, second: Any) -> dict[str, Any]:
    first_text = _memory_item_text(first)
    second_text = _memory_item_text(second)
    preferred = first if len(first_text) >= len(second_text) else second
    merged = dict(preferred) if isinstance(preferred, Mapping) else {"text": _memory_item_text(preferred)}
    merged["text"] = _memory_item_text(preferred)
    evidence: list[str] = []
    for item in (first, second):
        if not isinstance(item, Mapping):
            continue
        for turn_id in item.get("evidence_turn_ids") or []:
            value = str(turn_id or "").strip()
            if value and value not in evidence:
                evidence.append(value)
    if evidence:
        merged["evidence_turn_ids"] = evidence
    return merged


def _compact_event_entries(entries: list[Any]) -> list[Any]:
    compacted: list[Any] = []
    for entry in entries:
        if not _memory_item_text(entry):
            continue
        match_index = next(
            (index for index, current in enumerate(compacted) if _events_overlap(current, entry)),
            None,
        )
        if match_index is None:
            compacted.append(entry)
            continue
        current = compacted[match_index]
        if _event_is_user_managed(current):
            continue
        if _event_is_user_managed(entry):
            compacted[match_index] = entry
        else:
            compacted[match_index] = _merge_equivalent_events(current, entry)
    return compacted


def _snapshot_item_id(
    patient_id: str,
    category: str,
    item: Any,
    text: str,
) -> str:
    if isinstance(item, Mapping):
        raw = str(item.get("item_id") or item.get("id") or "").strip()
        if raw:
            return raw
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"lmca-memory:{patient_id}:{category}:{text}",
    ).hex


def _active_memory_items(
    values: Any,
    *,
    patient_id: str = "",
    category: str = "",
    deleted_item_ids: Optional[set[str]] = None,
    deleted_texts: Optional[set[str]] = None,
) -> list[tuple[str, bool]]:
    if not isinstance(values, list):
        return []
    deleted_item_ids = deleted_item_ids or set()
    deleted_texts = deleted_texts or set()
    result = []
    for item in values:
        if isinstance(item, Mapping) and str(item.get("status") or "active").lower() != "active":
            continue
        text = _memory_item_text(item)
        if not text or text in deleted_texts:
            continue
        if patient_id and category:
            item_id = _snapshot_item_id(patient_id, category, item, text)
            if item_id in deleted_item_ids:
                continue
        if text:
            result.append((text, isinstance(item, Mapping) and str(item.get("source") or "").lower() == "manual"))
    return result


def _superseded_turn_ids(snapshot: Mapping[str, Any]) -> set[str]:
    """Return local evidence excluded by an active manual correction."""
    result: set[str] = set()
    for field in ("facts", "preferences", "events", "comfort_strategies"):
        for item in snapshot.get(field) or []:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "active").lower()
            source = str(item.get("source") or "").lower()
            values = item.get("evidence_turn_ids") or []
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            if status in {"deleted", "superseded"} or (status == "active" and source == "manual"):
                result.update(str(value).strip() for value in values if str(value).strip())
                source_turn = str(item.get("source_turn_id") or "").strip()
                if source_turn:
                    result.add(source_turn)
    return result


class EmotionMemobase:
    """Thread-safe, synchronous SQLite memory facade.

    ``capture_turn`` and all update methods commit locally before returning.
    An explicitly supplied Memobase client is only used as a background mirror.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        memobase_client: Any = None,
        emotion_classifier: Optional[Callable[[str], Any]] = None,
        memobase: Any = None,
        logger: Callable[[str], Any] = print,
    ) -> None:
        configured_path = db_path or storage_path or _database.get_db_path()
        if str(configured_path) == ":memory:":
            self._db_path = ":memory:"
        else:
            path = Path(configured_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = _path_key(str(path))
        self._lock = _lock_for(self._db_path)
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self._db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        if emotion_classifier is None:
            try:
                from src.tools.emotion.emotion_classifier import (
                    classify_emotion,
                )

                emotion_classifier = classify_emotion
            except Exception:
                pass
        self._emotion_classifier = emotion_classifier
        self._memobase = memobase_client if memobase_client is not None else memobase
        self._memobase_auto = self._memobase is None and bool(
            os.getenv("MEMOBASE_PROJECT_URL") and os.getenv("MEMOBASE_API_KEY")
        )
        self._memobase_retry_at = 0.0
        self._memobase_executor: Optional[ThreadPoolExecutor] = None
        self._memobase_lock = threading.Lock()
        self._log = logger
        self._ensure_schema()

    def _load_memobase(self) -> Any:
        project_url = os.getenv("MEMOBASE_PROJECT_URL")
        api_key = os.getenv("MEMOBASE_API_KEY")
        if not project_url or not api_key:
            return None
        if not _is_local_memobase_url(project_url):
            self._log("[Memobase] 已拒绝非本机 MEMOBASE_PROJECT_URL")
            return None
        client = None
        try:
            import httpx
            from memobase import MemoBaseClient

            client = MemoBaseClient(project_url=project_url, api_key=api_key)
            client.client.close()
            client._client = httpx.Client(
                base_url=client.base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
                trust_env=False,
            )
            if not client.ping():
                client.client.close()
                return None
            return client
        except Exception as exc:
            close = getattr(getattr(client, "client", None), "close", None)
            if callable(close):
                close()
            self._log(f"[Memobase] 本地服务不可用: {type(exc).__name__}")
            return None

    @contextmanager
    def _connection(self):
        conn = self._memory_conn
        owns_connection = conn is None
        if owns_connection:
            conn = sqlite3.connect(
                self._db_path,
                timeout=5,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_SNAPSHOT_TABLE} (
                    patient_id TEXT PRIMARY KEY,
                    snapshot_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    last_consolidated_turn_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {_TURN_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    session_id TEXT,
                    turn_id TEXT NOT NULL DEFAULT '',
                    turn_state TEXT NOT NULL DEFAULT 'DURABLY_CAPTURED',
                    response_status TEXT NOT NULL DEFAULT 'responded',
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    emotion_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_{_TURN_TABLE}_patient_time
                    ON {_TURN_TABLE}(patient_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_{_TURN_TABLE}_patient_id
                    ON {_TURN_TABLE}(patient_id, id DESC);
                CREATE TABLE IF NOT EXISTS {_SYNC_TABLE} (
                    patient_id TEXT PRIMARY KEY,
                    last_synced_turn_id INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT,
                    lease_until TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {_MEMORY_ITEM_TABLE} (
                    item_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'wellbeing',
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'derived',
                    source_session_id TEXT,
                    source_turn_id TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    deleted_at TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {_MEMORY_DELETION_TABLE} (
                    item_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    deletion_token TEXT NOT NULL,
                    deleted_by TEXT,
                    deleted_at TEXT NOT NULL,
                    item_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_{_MEMORY_ITEM_TABLE}_patient_status
                    ON {_MEMORY_ITEM_TABLE}(patient_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_{_MEMORY_ITEM_TABLE}_patient_mode
                    ON {_MEMORY_ITEM_TABLE}(patient_id, mode, status);
                CREATE TABLE IF NOT EXISTS emotion_trajectory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    session_id TEXT,
                    turn_id TEXT NOT NULL DEFAULT '',
                    analysis_status TEXT NOT NULL DEFAULT 'final',
                    turn_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    joy REAL NOT NULL,
                    sadness REAL NOT NULL,
                    anger REAL NOT NULL,
                    fear REAL NOT NULL,
                    anxiety REAL NOT NULL,
                    calm REAL NOT NULL,
                    confusion REAL NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    dominant_emotion TEXT NOT NULL,
                    trigger_content TEXT NOT NULL DEFAULT '',
                    audio_path TEXT,
                    source TEXT NOT NULL DEFAULT 'text'
                );
                CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_patient_time
                    ON emotion_trajectory(patient_id, timestamp, id);
                """
            )
            self._ensure_column(conn, _SNAPSHOT_TABLE, "revision", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(
                conn, _SNAPSHOT_TABLE, "last_consolidated_turn_id", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(conn, _SNAPSHOT_TABLE, "updated_by", "TEXT NOT NULL DEFAULT 'system'")
            for column, definition in (
                ("turn_id", "TEXT NOT NULL DEFAULT ''"),
                ("turn_state", "TEXT NOT NULL DEFAULT 'DURABLY_CAPTURED'"),
                ("response_status", "TEXT NOT NULL DEFAULT 'responded'"),
            ):
                self._ensure_column(conn, _TURN_TABLE, column, definition)
            for column, definition in (
                ("turn_id", "TEXT NOT NULL DEFAULT ''"),
                ("analysis_status", "TEXT NOT NULL DEFAULT 'final'"),
            ):
                self._ensure_column(conn, "emotion_trajectory", column, definition)
            for column, definition in (
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("next_retry_at", "TEXT"),
                ("last_error", "TEXT"),
                ("lease_until", "TEXT"),
            ):
                self._ensure_column(conn, _SYNC_TABLE, column, definition)
            conn.execute(
                f"UPDATE {_TURN_TABLE} SET turn_id='legacy:' || id WHERE turn_id IS NULL OR turn_id=''"
            )
            conn.execute(
                "UPDATE emotion_trajectory SET turn_id='legacy:trajectory:' || id "
                "WHERE turn_id IS NULL OR turn_id=''"
            )
            conn.execute(f"DROP INDEX IF EXISTS uq_{_TURN_TABLE}_patient_turn")
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{_TURN_TABLE}_patient_turn "
                f"ON {_TURN_TABLE}(patient_id, COALESCE(session_id, ''), turn_id)"
            )
            conn.execute("DROP INDEX IF EXISTS uq_emotion_trajectory_patient_turn")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_emotion_trajectory_patient_turn "
                "ON emotion_trajectory(patient_id, COALESCE(session_id, ''), turn_id)"
            )

    @staticmethod
    def _memory_item_row(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["version"] = int(result.get("version") or 0)
        if str(result.get("status") or "").lower() == "deleted":
            result["content"] = ""
        return result

    def _deleted_item_ids(self, conn: sqlite3.Connection, patient_id: str) -> set[str]:
        rows = conn.execute(
            f"SELECT item_id FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND status='deleted'",
            (patient_id,),
        ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def _deleted_texts_by_category(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
    ) -> dict[str, set[str]]:
        rows = conn.execute(
            f"""SELECT category, content FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND status='deleted' AND content<>''""",
            (patient_id,),
        ).fetchall()
        result: dict[str, set[str]] = {}
        for row in rows:
            result.setdefault(str(row["category"]), set()).add(str(row["content"]))
        return result

    def _deleted_turn_ids(self, conn: sqlite3.Connection, patient_id: str) -> set[str]:
        rows = conn.execute(
            f"""SELECT source_turn_id FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND status='deleted'
                  AND COALESCE(source_turn_id, '')<>''""",
            (patient_id,),
        ).fetchall()
        return {str(row["source_turn_id"]) for row in rows}

    def _suppress_redundant_event_items(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        mode: str,
    ) -> None:
        rows = conn.execute(
            f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND mode=? AND category='events' AND status='active'
                ORDER BY created_at ASC, item_id ASC""",
            (patient_id, normalize_session_mode(mode)),
        ).fetchall()
        kept: list[dict[str, Any]] = []
        superseded: list[str] = []
        for row in rows:
            item = self._memory_item_row(row)
            match_index = next(
                (index for index, current in enumerate(kept) if _events_overlap(current, item)),
                None,
            )
            if match_index is None:
                kept.append(item)
                continue
            current = kept[match_index]
            if _event_is_user_managed(current):
                superseded.append(item["item_id"])
            elif _event_is_user_managed(item) or len(item["content"]) > len(current["content"]):
                superseded.append(current["item_id"])
                kept[match_index] = item
            else:
                superseded.append(item["item_id"])
        if superseded:
            conn.executemany(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status='superseded', updated_at=?
                    WHERE item_id=? AND status='active'""",
                [(_now(), item_id) for item_id in superseded],
            )

    def _sync_memory_items_from_snapshot(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        snapshot: Mapping[str, Any],
        *,
        mode: str = WELLBEING,
    ) -> None:
        now = _now()
        deleted = self._deleted_item_ids(conn, patient_id)
        for category in ("facts", "preferences", "events", "comfort_strategies"):
            for item in snapshot.get(category) or []:
                text = _memory_item_text(item)
                if not text:
                    continue
                item_id = _snapshot_item_id(patient_id, category, item, text)
                if item_id in deleted:
                    continue
                if isinstance(item, Mapping) and str(item.get("status") or "active").lower() != "active":
                    continue
                source = "derived"
                source_session_id = None
                source_turn_id = None
                created_at = now
                updated_at = now
                if isinstance(item, Mapping):
                    source = str(item.get("source") or source).strip() or source
                    source_session_id = str(item.get("source_session_id") or "").strip() or None
                    source_turn_id = str(item.get("source_turn_id") or "").strip() or None
                    if not source_turn_id:
                        evidence = item.get("evidence_turn_ids") or []
                        if isinstance(evidence, (list, tuple)) and evidence:
                            source_turn_id = str(evidence[0] or "").strip() or None
                    created_at = str(item.get("created_at") or now)
                    updated_at = str(item.get("updated_at") or now)
                conn.execute(
                    f"""INSERT OR IGNORE INTO {_MEMORY_ITEM_TABLE}
                        (item_id, patient_id, mode, category, content, source,
                         source_session_id, source_turn_id, version, status,
                         deleted_at, updated_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'active', NULL, ?, ?)""",
                    (
                        item_id,
                        patient_id,
                        normalize_session_mode(mode),
                        category,
                        text,
                        source,
                        source_session_id,
                        source_turn_id,
                        updated_at,
                        created_at,
                    ),
                )
                current = conn.execute(
                    f"SELECT content, source, source_session_id, source_turn_id, status, version "
                    f"FROM {_MEMORY_ITEM_TABLE} WHERE item_id=?",
                    (item_id,),
                ).fetchone()
                if current and str(current["status"] or "").lower() == "active" and str(current["content"] or "") != text:
                    conn.execute(
                        f"""UPDATE {_MEMORY_ITEM_TABLE}
                            SET content=?, source=?, source_session_id=?, source_turn_id=?,
                                version=?, updated_at=?
                            WHERE item_id=?""",
                        (
                            text,
                            source,
                            source_session_id,
                            source_turn_id,
                            int(current["version"] or 0) + 1,
                            updated_at,
                            item_id,
                        ),
                    )
        self._suppress_redundant_event_items(conn, patient_id, mode)

    def _snapshot_with_deleted_filtered(
        self,
        conn: sqlite3.Connection,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        patient_id = str(result["patient_id"])
        deleted_ids = self._deleted_item_ids(conn, patient_id)
        deleted_texts = self._deleted_texts_by_category(conn, patient_id)
        for category in ("facts", "preferences", "events", "comfort_strategies"):
            values = []
            for item in result.get(category) or []:
                text = _memory_item_text(item)
                if not text or text in deleted_texts.get(category, set()):
                    continue
                item_id = _snapshot_item_id(patient_id, category, item, text)
                if item_id in deleted_ids:
                    continue
                values.append(item)
            result[category] = values
        return result

    def _active_items_by_category(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        *,
        mode: Optional[str] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        query = f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                    WHERE patient_id=? AND status='active'"""
        params: list[Any] = [patient_id]
        if mode:
            query += " AND mode=?"
            params.append(normalize_session_mode(mode))
        query += " ORDER BY updated_at ASC, created_at ASC"
        result: dict[str, list[dict[str, Any]]] = {}
        for row in conn.execute(query, tuple(params)).fetchall():
            item = self._memory_item_row(row)
            result.setdefault(str(item["category"]), []).append(item)
        return result

    def _set_snapshot_item_content(
        self,
        snapshot: Mapping[str, Any],
        item_id: str,
        content: str,
    ) -> dict[str, Any]:
        updated = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        now = _now()
        for category in ("facts", "preferences", "events", "comfort_strategies"):
            values = list(updated.get(category) or [])
            for index, item in enumerate(values):
                text = _memory_item_text(item)
                if not text:
                    continue
                current_id = _snapshot_item_id(updated["patient_id"], category, item, text)
                if current_id != item_id:
                    continue
                entry = dict(item) if isinstance(item, Mapping) else {"text": text}
                entry["item_id"] = item_id
                entry["source"] = "manual"
                entry["status"] = "active"
                entry["updated_at"] = now
                if category == "comfort_strategies":
                    entry["strategy"] = content
                else:
                    entry["text"] = content
                values[index] = entry
                updated[category] = values
                return updated
        return updated

    def _mark_snapshot_item_deleted(
        self,
        snapshot: Mapping[str, Any],
        item_id: str,
        content: str,
        deleted_at: str,
    ) -> dict[str, Any]:
        updated = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        for category in ("facts", "preferences", "events", "comfort_strategies"):
            values = list(updated.get(category) or [])
            changed = False
            for index, item in enumerate(values):
                text = _memory_item_text(item)
                if not text:
                    continue
                current_id = _snapshot_item_id(updated["patient_id"], category, item, text)
                if current_id != item_id and text != content:
                    continue
                entry = dict(item) if isinstance(item, Mapping) else {"text": text}
                entry["item_id"] = item_id
                entry["status"] = "deleted"
                entry["deleted_at"] = deleted_at
                entry["updated_at"] = deleted_at
                values[index] = entry
                changed = True
            if changed:
                updated[category] = values
        return updated

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _patient_profile(self, conn: sqlite3.Connection, patient_id: str) -> dict[str, Any]:
        try:
            row = conn.execute(
                """SELECT name, gender, age, education_years, extra_profile
                   FROM patients WHERE patient_id=?""",
                (patient_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return {}
        if not row:
            return {}
        profile = {
            key: row[key]
            for key in ("name", "gender", "age", "education_years")
            if row[key] not in (None, "")
        }
        try:
            extra = json.loads(row["extra_profile"] or "{}")
            if isinstance(extra, dict):
                profile.update(extra)
        except (TypeError, json.JSONDecodeError):
            pass
        return profile

    def _update_patient_profile(self, conn: sqlite3.Connection, patient_id: str, updates: Mapping[str, Any]) -> None:
        try:
            row = conn.execute(
                """SELECT name, gender, age, education_years, extra_profile
                   FROM patients WHERE patient_id=?""",
                (patient_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return
        if not row:
            return
        current = self._patient_profile(conn, patient_id)
        for key, value in updates.items():
            if key != "patient_id" and value not in (None, ""):
                current[key] = value
        core = {key: current.get(key) for key in ("name", "gender", "age", "education_years")}
        extra = {
            key: value
            for key, value in current.items()
            if key not in {"name", "gender", "age", "education_years", "patient_id"}
            and value not in (None, "")
        }
        conn.execute(
            """UPDATE patients SET name=?, gender=?, age=?, education_years=?,
               extra_profile=?, updated_at=? WHERE patient_id=?""",
            (
                core["name"] or row["name"],
                core["gender"],
                core["age"],
                core["education_years"],
                json.dumps(extra, ensure_ascii=False) if extra else None,
                _now(),
                patient_id,
            ),
        )

    def _load_snapshot(self, conn: sqlite3.Connection, patient_id: str) -> dict[str, Any]:
        row = conn.execute(
            f"SELECT snapshot_json, revision, last_consolidated_turn_id, updated_at "
            f"FROM {_SNAPSHOT_TABLE} WHERE patient_id=?",
            (patient_id,),
        ).fetchone()
        raw: Any = {}
        stored_updated_at = None
        if row:
            stored_updated_at = row["updated_at"]
            try:
                raw = json.loads(row["snapshot_json"] or "{}")
            except json.JSONDecodeError:
                raw = {}
        data = _normalise_snapshot(patient_id, raw, self._patient_profile(conn, patient_id))
        data["updated_at"] = stored_updated_at or data.get("updated_at")
        if row:
            data["revision"] = max(int(row["revision"] or 0), int(data.get("revision") or 0))
            data["last_consolidated_turn_id"] = max(
                int(row["last_consolidated_turn_id"] or 0),
                int(data.get("last_consolidated_turn_id") or 0),
            )
        return data

    def _save_snapshot(self, conn: sqlite3.Connection, snapshot: Mapping[str, Any], updated_by: str = "system") -> str:
        patient_id = str(snapshot["patient_id"])
        updated_at = _now()
        stored = _normalise_snapshot(patient_id, snapshot)
        row = conn.execute(
            f"SELECT revision, last_consolidated_turn_id FROM {_SNAPSHOT_TABLE} WHERE patient_id=?",
            (patient_id,),
        ).fetchone()
        stored["revision"] = (int(row["revision"] or 0) if row else 0) + 1
        if row:
            stored["last_consolidated_turn_id"] = max(
                int(row["last_consolidated_turn_id"] or 0),
                int(stored.get("last_consolidated_turn_id") or 0),
            )
        stored["updated_at"] = updated_at
        stored["updated_by"] = updated_by or "system"
        stored.pop("mmse_scores", None)
        stored.pop("emotion_history", None)
        stored.pop("recent_turns", None)
        payload = json.dumps(stored, ensure_ascii=False, sort_keys=True)
        conn.execute(
            f"""INSERT INTO {_SNAPSHOT_TABLE}
                (patient_id, snapshot_json, revision, last_consolidated_turn_id, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    snapshot_json=excluded.snapshot_json,
                    revision=excluded.revision,
                    last_consolidated_turn_id=excluded.last_consolidated_turn_id,
                    updated_by=excluded.updated_by,
                    updated_at=excluded.updated_at""",
            (
                patient_id,
                payload,
                stored["revision"],
                stored["last_consolidated_turn_id"],
                updated_at,
                stored["updated_by"],
            ),
        )
        return updated_at

    def _read_turns(self, conn: sqlite3.Connection, patient_id: str, limit: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"""SELECT session_id, turn_id, turn_state, response_status,
                       user_message, assistant_message, emotion_json, created_at
                FROM {_TURN_TABLE} WHERE patient_id=?
                ORDER BY created_at DESC, id DESC LIMIT ?""",
            (patient_id, max(1, int(limit))),
        ).fetchall()
        turns = []
        for row in reversed(rows):
            try:
                emotion = json.loads(row["emotion_json"] or "{}")
            except json.JSONDecodeError:
                emotion = {}
            turns.append(
                {
                    "session_id": row["session_id"],
                    "turn_id": row["turn_id"],
                    "turn_state": row["turn_state"],
                    "response_status": row["response_status"],
                    "user_message": row["user_message"],
                    "assistant_message": row["assistant_message"],
                    "emotion": emotion,
                    "created_at": row["created_at"],
                }
            )
        return turns

    def _read_emotion_trajectory(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        limit: int,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT patient_id, session_id, turn_id, analysis_status, turn_index, timestamp,
                   joy, sadness, anger, fear, anxiety, calm, confusion,
                   valence, arousal, dominant_emotion, trigger_content,
                   audio_path, source
            FROM emotion_trajectory
            WHERE patient_id=?
        """
        params: list[Any] = [patient_id]
        if session_id:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def _external_mmse_history(self, conn: sqlite3.Connection, patient_id: str) -> list[dict[str, Any]]:
        try:
            rows = conn.execute(
                """SELECT session_id, created_at, total_mmse_score
                   FROM sessions
                   WHERE patient_id=? AND total_mmse_score IS NOT NULL
                   ORDER BY created_at ASC""",
                (patient_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "score": row["total_mmse_score"],
                "weak_dimensions": [],
                "recorded_at": row["created_at"],
                "session_id": row["session_id"],
                "source": "sessions",
            }
            for row in rows
        ]

    def _public_snapshot(
        self,
        conn: sqlite3.Connection,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._sync_memory_items_from_snapshot(
            conn,
            str(snapshot["patient_id"]),
            snapshot,
        )
        result = self._snapshot_with_deleted_filtered(conn, snapshot)
        result["recent_turns"] = self._read_turns(
            conn, result["patient_id"], _MAX_TURNS_IN_SNAPSHOT
        )
        own_scores = list(result.get("mmse_history") or [])
        external_scores = self._external_mmse_history(conn, result["patient_id"])
        known = {
            (item.get("session_id"), item.get("recorded_at"), item.get("score"))
            for item in own_scores
            if isinstance(item, Mapping)
        }
        result["mmse_history"] = own_scores + [
            item for item in external_scores
            if (item["session_id"], item["recorded_at"], item["score"]) not in known
        ]
        result["mmse_history"].sort(key=lambda item: str(item.get("recorded_at") or ""))
        result["mmse_scores"] = _json_copy(result["mmse_history"])
        result["emotion_history"] = _json_copy(result["emotion"].get("history") or [])
        result["emotion_summary"] = _json_copy(result["emotion"])
        result["emotion_trajectory"] = self._read_emotion_trajectory(
            conn,
            result["patient_id"],
            _MAX_EMOTION_HISTORY,
        )
        return result

    def get_snapshot(self, patient_id: str) -> dict[str, Any]:
        """Return the editable structured memory snapshot for one patient."""
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            return self._public_snapshot(conn, self._load_snapshot(conn, patient_id))

    get_memory_for_user = get_snapshot

    def get_memory_snapshot(self, patient_id: str) -> dict[str, Any]:
        """Explicit alias used by memory-facing APIs."""
        return self.get_snapshot(patient_id)

    def get_emotion_trajectory(
        self,
        patient_id: str,
        *,
        limit: int = 100,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            return self._read_emotion_trajectory(
                conn,
                patient_id,
                limit,
                str(session_id).strip() if session_id else None,
            )

    def update_emotion_analysis(
        self,
        patient_id: str,
        turn_id: str,
        emotions: Mapping[str, Any],
        *,
        session_id: Optional[str] = None,
        analysis_status: str = "final",
        audio_path: Optional[str] = None,
        emotion_source: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update the existing trajectory row for a turn without inserting a duplicate."""
        patient_id = self._patient_id(patient_id)
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            raise ValueError("turn_id is required")
        session_id = str(session_id).strip() if session_id else None
        emotion_scores = _normalise_emotion(emotions)
        valence = _emotion_valence(emotion_scores)
        arousal = _emotion_arousal(emotion_scores)
        dominant = _dominant_emotion(emotion_scores)
        status = str(analysis_status or "final").strip() or "final"
        source = str(emotion_source or "multimodal").strip() or "multimodal"
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            trajectory = conn.execute(
                """SELECT id, session_id, timestamp, trigger_content, audio_path, source
                   FROM emotion_trajectory
                   WHERE patient_id=? AND COALESCE(session_id, '')=COALESCE(?, '') AND turn_id=?""",
                (patient_id, session_id, turn_id),
            ).fetchone()
            if not trajectory:
                raise KeyError(f"turn not found: {turn_id}")
            conn.execute(
                """UPDATE emotion_trajectory
                   SET joy=?, sadness=?, anger=?, fear=?, anxiety=?, calm=?, confusion=?,
                       valence=?, arousal=?, dominant_emotion=?, analysis_status=?,
                       audio_path=COALESCE(?, audio_path), source=?
                   WHERE id=?""",
                (
                    emotion_scores["joy"], emotion_scores["sadness"],
                    emotion_scores["anger"], emotion_scores["fear"],
                    emotion_scores["anxiety"], emotion_scores["calm"],
                    emotion_scores["confusion"], valence, arousal, dominant,
                    status, audio_path, source, int(trajectory["id"]),
                ),
            )
            conn.execute(
                f"""UPDATE {_TURN_TABLE} SET emotion_json=?
                    WHERE patient_id=? AND COALESCE(session_id, '')=COALESCE(?, '') AND turn_id=?""",
                (json.dumps(emotion_scores, ensure_ascii=False, sort_keys=True), patient_id, session_id, turn_id),
            )
            snapshot = self._load_snapshot(conn, patient_id)
            emotion_state = dict(snapshot.get("emotion") or {})
            history = list(emotion_state.get("history") or [])
            entry = {
                "scores": emotion_scores,
                "dominant": dominant,
                "valence": valence,
                "arousal": arousal,
                "recorded_at": trajectory["timestamp"],
                "session_id": trajectory["session_id"],
                "turn_id": turn_id,
                "audio_path": audio_path or trajectory["audio_path"],
                "source": source,
                "analysis_status": status,
            }
            replaced = False
            for index, item in enumerate(history):
                if isinstance(item, Mapping) and str(item.get("turn_id") or "") == turn_id:
                    history[index] = entry
                    replaced = True
                    break
            if not replaced:
                history.append(entry)
            latest = emotion_state.get("latest")
            latest_at = str(latest.get("recorded_at") or "") if isinstance(latest, Mapping) else ""
            if not latest_at or trajectory["timestamp"] >= latest_at or (
                isinstance(latest, Mapping) and latest.get("turn_id") == turn_id
            ):
                emotion_state["latest"] = entry
            emotion_state["history"] = history[-_MAX_EMOTION_HISTORY:]
            snapshot["emotion"] = emotion_state
            self._save_snapshot(conn, snapshot, updated_by="system")
        return entry

    update_turn_emotion = update_emotion_analysis

    def update_turn_status(
        self,
        patient_id: str,
        turn_id: str,
        *,
        session_id: Optional[str] = None,
        assistant_message: Optional[str] = None,
        response_status: Optional[str] = None,
        turn_state: Optional[str] = None,
    ) -> bool:
        patient_id = self._patient_id(patient_id)
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            raise ValueError("turn_id is required")
        fields = []
        values: list[Any] = []
        if assistant_message is not None:
            fields.append("assistant_message=?")
            values.append(str(assistant_message or "").strip())
        if response_status is not None:
            fields.append("response_status=?")
            values.append(str(response_status).strip())
        if turn_state is not None:
            fields.append("turn_state=?")
            values.append(str(turn_state).strip())
        if not fields:
            return False
        values.extend((patient_id, session_id, turn_id))
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                f"""UPDATE {_TURN_TABLE} SET {', '.join(fields)}
                    WHERE patient_id=? AND COALESCE(session_id, '')=COALESCE(?, '') AND turn_id=?""",
                tuple(values),
            )
            return cursor.rowcount > 0

    def update_snapshot(
        self,
        patient_id: str,
        updates: Optional[Mapping[str, Any]] = None,
        *,
        updated_by: str = "user",
        expected_revision: Optional[int] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Merge user-editable fields and return the resulting snapshot."""
        patient_id = self._patient_id(patient_id)
        if updates is not None and not isinstance(updates, Mapping):
            raise ValueError("updates must be a mapping")
        payload = dict(updates or {})
        payload.update(fields)
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            if expected_revision is not None:
                try:
                    expected = int(expected_revision)
                except (TypeError, ValueError) as exc:
                    raise ValueError("expected_revision must be an integer") from exc
                if int(snapshot.get("revision") or 0) != expected:
                    raise MemoryRevisionConflict(int(snapshot.get("revision") or 0))
            profile_updates: Optional[Mapping[str, Any]] = None
            for key, value in payload.items():
                if key in {
                    "patient_id", "schema_version", "updated_at", "updated_by",
                    "revision", "last_consolidated_turn_id", "recent_turns",
                }:
                    continue
                if key == "profile":
                    if not isinstance(value, Mapping):
                        raise ValueError("profile must be a mapping")
                    profile_updates = dict(value)
                    merged = dict(snapshot.get("profile") or {})
                    merged.update(_json_copy(dict(value)))
                    snapshot["profile"] = merged
                elif key == "mmse_scores":
                    snapshot["mmse_history"] = _json_copy(value)
                elif key == "emotion_history":
                    emotion = dict(snapshot.get("emotion") or {})
                    emotion["history"] = _json_copy(value)
                    snapshot["emotion"] = emotion
                elif key in _LIST_FIELDS:
                    if not isinstance(value, list):
                        raise ValueError(f"{key} must be a list")
                    snapshot[key] = _json_copy(value)
                elif key == "narrative":
                    snapshot[key] = str(value or "")
                else:
                    snapshot[key] = _json_copy(value)
            if profile_updates:
                self._update_patient_profile(conn, patient_id, profile_updates)
            self._save_snapshot(conn, snapshot, updated_by=updated_by)
            stored = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, stored)
            return self._public_snapshot(conn, stored)

    def update_memory(
        self,
        patient_id: str,
        updates: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Compatibility alias for structured memory editing."""
        return self.update_snapshot(patient_id, updates, **fields)

    def update_memory_by_user(
        self,
        patient_id: str,
        updates: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        payload = dict(updates or {})
        payload.update(fields)
        expected_revision = payload.pop("expected_revision", None)
        allowed = {"profile", "facts", "preferences", "events", "narrative"}
        unsupported = sorted(set(payload) - allowed)
        if unsupported:
            raise ValueError(
                "user memory cannot update: " + ", ".join(unsupported)
            )
        for field in ("facts", "preferences", "events"):
            if field not in payload:
                continue
            if not isinstance(payload[field], list):
                raise ValueError(f"{field} must be a list")
            items = []
            for item in payload[field]:
                if isinstance(item, Mapping):
                    entry = _json_copy(dict(item))
                else:
                    entry = {"text": str(item or "").strip()}
                if not _memory_item_text(entry):
                    raise ValueError(f"{field} cannot contain blank items")
                entry.setdefault("id", uuid.uuid4().hex)
                entry["source"] = "manual"
                entry.setdefault("status", "active")
                entry.setdefault("evidence_turn_ids", [])
                items.append(entry)
            payload[field] = items
        return self.update_snapshot(
            patient_id,
            payload,
            updated_by="user",
            expected_revision=expected_revision,
        )

    def update_mmse_score(
        self,
        patient_id: str,
        score: int,
        weak_dimensions: Optional[list[str]] = None,
        date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append one MMSE result without replacing earlier scores."""
        patient_id = self._patient_id(patient_id)
        if isinstance(score, bool):
            raise ValueError("score must be an integer")
        try:
            score = int(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("score must be an integer") from exc
        if score < 0 or score > 35:
            raise ValueError("score must be between 0 and 35")
        entry = {
            "score": score,
            "weak_dimensions": _strings(weak_dimensions),
            "recorded_at": str(date or _now()).strip(),
        }
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            history = list(snapshot.get("mmse_history") or [])
            history.append(entry)
            snapshot["mmse_history"] = history[-_MAX_MMSE_HISTORY:]
            self._save_snapshot(conn, snapshot, updated_by="system")
        return entry

    def add_comfort_strategy(
        self,
        patient_id: str,
        strategy: str,
        effective: bool,
    ) -> dict[str, Any]:
        """Upsert a comfort strategy and keep its latest effectiveness."""
        patient_id = self._patient_id(patient_id)
        strategy = str(strategy or "").strip()
        if not strategy:
            raise ValueError("strategy is required")
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            strategies = list(snapshot.get("comfort_strategies") or [])
            entry = None
            for index, item in enumerate(strategies):
                if isinstance(item, Mapping) and str(item.get("strategy") or "").strip() == strategy:
                    item = dict(item)
                    item.setdefault("id", uuid.uuid4().hex)
                    item["effective"] = bool(effective)
                    item["updated_at"] = now
                    try:
                        count = int(item.get("use_count") or 0)
                    except (TypeError, ValueError):
                        count = 0
                    item["use_count"] = count + 1
                    entry = item
                    strategies[index] = item
                    break
            if entry is None:
                entry = {
                    "id": uuid.uuid4().hex,
                    "strategy": strategy,
                    "effective": bool(effective),
                    "use_count": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                strategies.append(entry)
            snapshot["comfort_strategies"] = strategies
            self._save_snapshot(conn, snapshot, updated_by="system")
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
        return entry

    def list_memory_items(
        self,
        patient_id: str,
        *,
        include_deleted: bool = False,
        mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            query = f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=?"
            params: list[Any] = [patient_id]
            if not include_deleted:
                query += " AND status='active'"
            if mode:
                query += " AND mode=?"
                params.append(normalize_session_mode(mode))
            query += " ORDER BY updated_at DESC, created_at DESC"
            return [
                self._memory_item_row(row)
                for row in conn.execute(query, tuple(params)).fetchall()
            ]

    def update_memory_item(
        self,
        patient_id: str,
        item_id: str,
        content: str,
        expected_version: int,
        *,
        updated_by: str = "user",
    ) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        item_id = str(item_id or "").strip()
        content = str(content or "").strip()
        if not item_id:
            raise ValueError("item_id is required")
        if not content:
            raise ValueError("content is required")
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            current = self._memory_item_row(row)
            if current["status"] == "deleted":
                raise KeyError("MEMORY_ITEM_DELETED")
            if int(current["version"]) != int(expected_version):
                raise MemoryRevisionConflict(int(current["version"]))
            now = _now()
            next_version = int(current["version"]) + 1
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET content=?, version=?, source=?, updated_at=?
                    WHERE patient_id=? AND item_id=?""",
                (
                    content,
                    next_version,
                    str(updated_by or "user"),
                    now,
                    patient_id,
                    item_id,
                ),
            )
            updated_snapshot = self._set_snapshot_item_content(snapshot, item_id, content)
            self._save_snapshot(conn, updated_snapshot, updated_by=updated_by)
            updated = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            return self._memory_item_row(updated)

    def delete_memory_item(
        self,
        patient_id: str,
        item_id: str,
        expected_version: int,
        deletion_token: str,
        *,
        deleted_by: str = "user",
    ) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        item_id = str(item_id or "").strip()
        token = str(deletion_token or "").strip()
        if not item_id:
            raise ValueError("item_id is required")
        if not token:
            raise ValueError("deletion_token is required")
        cleanup_content = ""
        cleanup_source_turn_id = ""
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            existing_delete = conn.execute(
                f"""SELECT deletion_token FROM {_MEMORY_DELETION_TABLE}
                    WHERE patient_id=? AND item_id=?""",
                (patient_id, item_id),
            ).fetchone()
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            current = dict(row)
            cleanup_content = str(current.get("content") or "")
            cleanup_source_turn_id = str(current.get("source_turn_id") or "")
            if str(current.get("status") or "").lower() == "deleted":
                deleted = self._memory_item_row(current)
                deleted["idempotent_replay"] = bool(
                    existing_delete and existing_delete["deletion_token"] == token
                )
                result = deleted
            else:
                if int(current["version"]) != int(expected_version):
                    raise MemoryRevisionConflict(int(current["version"]))
                now = _now()
                next_version = int(current["version"]) + 1
                conn.execute(
                    f"""UPDATE {_MEMORY_ITEM_TABLE}
                        SET version=?, status='deleted', deleted_at=?, updated_at=?
                        WHERE patient_id=? AND item_id=?""",
                    (next_version, now, now, patient_id, item_id),
                )
                conn.execute(
                    f"""INSERT OR IGNORE INTO {_MEMORY_DELETION_TABLE}
                        (item_id, patient_id, deletion_token, deleted_by, deleted_at, item_version)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                    (item_id, patient_id, token, deleted_by, now, next_version),
                )
                updated_snapshot = self._mark_snapshot_item_deleted(
                    snapshot,
                    item_id,
                    cleanup_content,
                    now,
                )
                self._save_snapshot(conn, updated_snapshot, updated_by=deleted_by)
                deleted = conn.execute(
                    f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                    (patient_id, item_id),
                ).fetchone()
                result = self._memory_item_row(deleted)
                result["idempotent_replay"] = False
        result["memobase_delete_enqueued"] = self._enqueue_memobase_deletion(
            patient_id,
            cleanup_content,
            cleanup_source_turn_id,
        )
        return result

    def capture_turn(
        self,
        patient_id: str,
        user_message: str,
        assistant_message: str,
        session_id: Optional[str] = None,
        emotion: Any = None,
        audio_path: Optional[str] = None,
        emotions: Any = None,
        emotion_source: Optional[str] = None,
        turn_id: Optional[str] = None,
        analysis_status: str = "final",
    ) -> dict[str, Any]:
        """Persist one turn and return its deterministic local emotion result."""
        patient_id = self._patient_id(patient_id)
        user_message = str(user_message or "").strip()
        if not user_message:
            raise ValueError("user_message is required")
        assistant_message = str(assistant_message or "").strip()
        session_id = str(session_id).strip() if session_id is not None and str(session_id).strip() else None
        turn_id = str(turn_id).strip() if turn_id is not None and str(turn_id).strip() else None
        if turn_id is None:
            turn_id = f"legacy:{uuid.uuid4()}"
        if emotions is not None and emotion is None:
            emotion = emotions
        if emotion is None and audio_path:
            try:
                from src.tools.emotion import classify_multimodal

                emotion = classify_multimodal(user_message, audio_path=audio_path)
            except Exception:
                emotion = None
        emotion_scores = _classify_emotion(
            user_message,
            self._emotion_classifier,
        )
        if isinstance(emotion, Mapping):
            emotion_scores = _normalise_emotion(emotion)
        else:
            override = _EMOTION_LABELS.get(str(emotion or "").strip().lower())
            if override:
                emotion_scores = dict(emotion_scores)
                emotion_scores[override] = max(
                    float(emotion_scores.get(override, 0.0)),
                    0.8,
                )
                emotion_scores = _normalise_emotion(emotion_scores)
        captured_at = _now()
        valence = _emotion_valence(emotion_scores)
        arousal = _emotion_arousal(emotion_scores)
        dominant = _dominant_emotion(emotion_scores)
        source = str(
            emotion_source or ("multimodal" if audio_path else "text")
        ).strip() or "text"
        analysis_status = str(analysis_status or "final").strip() or "final"
        emotion_entry = {
            "scores": emotion_scores,
            "dominant": dominant,
            "valence": valence,
            "arousal": arousal,
            "recorded_at": captured_at,
            "session_id": session_id,
            "audio_path": str(audio_path) if audio_path else None,
            "source": source,
            "analysis_status": analysis_status,
        }
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                f"""SELECT id, patient_id, session_id, turn_id, created_at,
                           emotion_json, user_message, assistant_message,
                           response_status, turn_state
                    FROM {_TURN_TABLE}
                    WHERE patient_id=? AND COALESCE(session_id, '')=COALESCE(?, '') AND turn_id=?""",
                (patient_id, session_id, turn_id),
            ).fetchone()
            if existing:
                try:
                    existing_emotion = json.loads(existing["emotion_json"] or "{}")
                except json.JSONDecodeError:
                    existing_emotion = {}
                existing_scores = _normalise_emotion(existing_emotion)
                existing_analysis_status = analysis_status
                existing_audio_path = audio_path
                trajectory = conn.execute(
                    """SELECT analysis_status, audio_path FROM emotion_trajectory
                       WHERE patient_id=? AND COALESCE(session_id, '')=COALESCE(?, '') AND turn_id=?""",
                    (patient_id, session_id, turn_id),
                ).fetchone()
                if trajectory:
                    existing_analysis_status = trajectory["analysis_status"] or existing_analysis_status
                    existing_audio_path = trajectory["audio_path"] or existing_audio_path
                return {
                    "patient_id": patient_id,
                    "session_id": existing["session_id"],
                    "turn_id": existing["turn_id"],
                    "local_turn_id": int(existing["id"]),
                    "captured_at": existing["created_at"],
                    "emotion": existing_scores,
                    "emotion_label": None,
                    "dominant_emotion": _dominant_emotion(existing_scores),
                    "valence": _emotion_valence(existing_scores),
                    "arousal": _emotion_arousal(existing_scores),
                    "audio_path": str(existing_audio_path) if existing_audio_path else None,
                    "emotion_source": source,
                    "response_status": existing["response_status"],
                    "turn_state": existing["turn_state"],
                    "analysis_status": existing_analysis_status,
                    "idempotent_replay": True,
                }
            snapshot = self._load_snapshot(conn, patient_id)
            history = list(snapshot.get("emotion", {}).get("history") or [])
            emotion_entry["turn_id"] = turn_id
            history.append(emotion_entry)
            emotion_state = dict(snapshot.get("emotion") or {})
            emotion_state["latest"] = emotion_entry
            emotion_state["history"] = history[-_MAX_EMOTION_HISTORY:]
            snapshot["emotion"] = emotion_state
            snapshot["last_session_id"] = session_id
            snapshot["last_turn_at"] = captured_at
            cursor = conn.execute(
                f"""INSERT INTO {_TURN_TABLE}
                   (patient_id, session_id, turn_id, turn_state, response_status,
                    user_message, assistant_message, emotion_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    session_id,
                    turn_id,
                    "DURABLY_CAPTURED",
                    "responded" if assistant_message else "pending",
                    user_message,
                    assistant_message,
                    json.dumps(emotion_scores, ensure_ascii=False, sort_keys=True),
                    captured_at,
                ),
            )
            local_turn_id = int(cursor.lastrowid)
            turn_index = conn.execute(
                "SELECT COUNT(*) + 1 FROM emotion_trajectory WHERE patient_id=?",
                (patient_id,),
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO emotion_trajectory (
                    patient_id, session_id, turn_id, analysis_status, turn_index, timestamp,
                    joy, sadness, anger, fear, anxiety, calm, confusion,
                    valence, arousal, dominant_emotion, trigger_content,
                    audio_path, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    session_id,
                    turn_id,
                    analysis_status,
                    turn_index,
                    captured_at,
                    emotion_scores["joy"],
                    emotion_scores["sadness"],
                    emotion_scores["anger"],
                    emotion_scores["fear"],
                    emotion_scores["anxiety"],
                    emotion_scores["calm"],
                    emotion_scores["confusion"],
                    valence,
                    arousal,
                    dominant,
                    _clip(user_message, 200),
                    str(audio_path) if audio_path else None,
                    source,
                ),
            )
            self._save_snapshot(conn, snapshot, updated_by="system")

        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "local_turn_id": local_turn_id,
            "captured_at": captured_at,
            "emotion": emotion_scores,
            "emotion_label": emotion if isinstance(emotion, str) else None,
            "dominant_emotion": dominant,
            "valence": valence,
            "arousal": arousal,
            "audio_path": str(audio_path) if audio_path else None,
            "emotion_source": source,
            "analysis_status": analysis_status,
        }

    def get_relevant_event_gists(
        self,
        patient_id: str,
        current_text: str,
        *,
        topk: int = _EVENT_GIST_TOPK,
    ) -> list[dict[str, Any]]:
        """Return structured semantic event gists from Memobase."""
        patient_id = self._patient_id(patient_id)
        current_text = str(current_text or "").strip()
        if not current_text:
            return []
        client = self._ensure_memobase()
        if client is None:
            self._log(f"[Memobase] gist_search degraded=unavailable patient={patient_id}")
            return []
        params: dict[str, Any] = {
            "topk": int(topk),
            "similarity_threshold": _EVENT_GIST_SIMILARITY_THRESHOLD,
        }
        if _EVENT_GIST_TIME_RANGE_DAYS:
            try:
                params["time_range_in_days"] = int(_EVENT_GIST_TIME_RANGE_DAYS)
            except ValueError:
                self._log("[Memobase] 忽略非法 MEMOBASE_EVENT_GIST_TIME_RANGE_DAYS")
        started_at = time.perf_counter()
        try:
            user = self._get_or_create_memobase_user(client, patient_id)
            raw_gists = self._search_event_gist(user, current_text, params)
        except Exception as exc:
            self._mark_memobase_failed(exc)
            self._log(
                f"[Memobase] gist_search degraded=error patient={patient_id} "
                f"error={type(exc).__name__}"
            )
            return []
        gists = self._normalise_event_gists(
            patient_id,
            raw_gists,
            max_items=int(topk),
        )
        with self._lock, self._connection() as conn:
            deleted_turn_ids = self._deleted_turn_ids(conn, patient_id)
            deleted_texts = {
                text
                for values in self._deleted_texts_by_category(conn, patient_id).values()
                for text in values
                if text
            }
        if deleted_turn_ids or deleted_texts:
            gists = [
                gist for gist in gists
                if str(gist.get("source_turn_id") or "") not in deleted_turn_ids
                and str(gist.get("turn_id") or "") not in deleted_turn_ids
                and not any(text in str(gist.get("content") or "") for text in deleted_texts)
            ]
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        gist_ids = ",".join(str(item.get("event_gist_id") or "-") for item in gists)
        source_turn_ids = ",".join(str(item.get("source_turn_id") or "-") for item in gists)
        self._log(
            f"[Latency] memobase_event_gist_search_ms={elapsed_ms:.1f} "
            f"patient={patient_id} topk={int(topk)} hits={len(gists)} "
            f"event_gist_ids={gist_ids or '-'} source_turn_ids={source_turn_ids or '-'} "
            f"degraded={'none' if gists else 'empty'}"
        )
        return gists

    @staticmethod
    def _search_event_gist(user: Any, query: str, params: Mapping[str, Any]) -> Any:
        project_client = getattr(user, "project_client", None)
        http_client = getattr(project_client, "client", None)
        user_id = getattr(user, "user_id", None)
        if http_client is not None and user_id:
            from memobase.network import unpack_response

            response = unpack_response(
                http_client.get(
                    f"/users/event_gist/search/{user_id}",
                    params={"query": query, **dict(params)},
                )
            )
            return response.data.get("gists", [])
        return user.search_event_gist(query, **dict(params))

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _normalise_event_gists(
        cls,
        patient_id: str,
        raw_gists: Any,
        *,
        max_items: int,
    ) -> list[dict[str, Any]]:
        result = []
        seen_events: set[str] = set()
        if not isinstance(raw_gists, list):
            raw_gists = list(raw_gists or [])
        for raw in raw_gists:
            gist_data = cls._field(raw, "gist_data")
            content = str(cls._field(gist_data, "content") or "").strip()
            if not content:
                continue
            event_id = str(cls._field(raw, "event_id") or "").strip()
            dedupe_key = event_id or content
            if dedupe_key in seen_events:
                continue
            seen_events.add(dedupe_key)
            try:
                similarity = float(cls._field(raw, "similarity"))
            except (TypeError, ValueError):
                similarity = 0.0
            result.append(
                {
                    "patient_id": patient_id,
                    "project_id": str(cls._field(raw, "project_id") or ""),
                    "session_id": str(cls._field(raw, "session_id") or ""),
                    "turn_id": str(cls._field(raw, "turn_id") or ""),
                    "event_gist_id": str(cls._field(raw, "id") or ""),
                    "event_id": event_id,
                    "source_turn_id": str(cls._field(raw, "source_turn_id") or ""),
                    "created_at": str(cls._field(raw, "created_at") or ""),
                    "similarity": similarity,
                    "content": content,
                }
            )
        result.sort(key=lambda item: float(item.get("similarity") or 0.0), reverse=True)
        return result[:max_items]

    @classmethod
    def _render_event_gists(
        cls,
        gists: list[Mapping[str, Any]],
        *,
        token_budget: int = _LONG_TERM_EVENT_TOKEN_BUDGET,
    ) -> str:
        lines = ["[retrieved_long_term_memory]"]
        used = _token_estimate(lines[0])
        added = 0
        for index, gist in enumerate(gists, 1):
            content = str(gist.get("content") or "").strip()
            if not content:
                continue
            created_at = str(gist.get("created_at") or "")[:19]
            similarity = gist.get("similarity")
            try:
                score = f"{float(similarity):.3f}"
            except (TypeError, ValueError):
                score = "unknown"
            block = f"{index}. 摘要：{content}\n   创建时间：{created_at or 'unknown'}；相似度：{score}"
            cost = _token_estimate(block)
            if used + cost > token_budget:
                continue
            lines.append(block)
            used += cost
            added += 1
        if not added:
            lines.append("无通过阈值的跨会话长期事件。")
        return "\n".join(lines)

    def get_relevant_context(self, patient_id: str, current_text: str) -> str:
        """Compatibility wrapper for semantic event gist evidence."""
        return self.get_relevant_evidence(patient_id, current_text)

    def get_authoritative_card(self, patient_id: str) -> str:
        """Build the active SQLite memory card without raw conversation turns."""
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            items_by_category = self._active_items_by_category(
                conn,
                patient_id,
                mode=WELLBEING,
            )
        lines = ["【患者权威记忆卡】"]
        profile = snapshot.get("profile") or {}
        if profile:
            labels = {"name": "姓名", "age": "年龄", "gender": "性别", "education_years": "受教育年限"}
            bits = []
            for key, value in profile.items():
                if value in (None, ""):
                    continue
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                bits.append(f"{labels.get(key, key)}：{value}")
            if bits:
                lines.append("【患者基本信息】" + "；".join(bits))
        for title, key in (("长期事实", "facts"), ("偏好与兴趣", "preferences"), ("近期事件", "events")):
            values = [
                (item["content"], str(item.get("source") or "").lower() == "manual")
                for item in items_by_category.get(key, [])
            ]
            if values:
                rendered = [
                    ("人工修订：" if manual else "") + _clip(text, 240)
                    for text, manual in values
                ]
                lines.append(f"【{title}】" + "；".join(rendered))
        narrative = str(snapshot.get("narrative") or "").strip()
        if narrative:
            lines.append("【权威摘要】" + _clip(narrative, 1000))
        strategies = [
            (item["content"], False)
            for item in items_by_category.get("comfort_strategies", [])
        ]
        if strategies:
            lines.append(
                "【有效安抚策略】" + "；".join(
                    _clip(text, 180) for text, _manual in strategies
                )
            )
        return "" if len(lines) == 1 else _clip("\n".join(lines), _MAX_CONTEXT_CHARS)

    def get_cross_session_context(
        self,
        patient_id: str,
        exclude_session_id: Optional[str] = None,
    ) -> str:
        """Return bounded SQLite evidence while excluding the working session."""
        patient_id = self._patient_id(patient_id)
        excluded_session = str(exclude_session_id or "").strip()
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            suppressed = _superseded_turn_ids(snapshot) | self._deleted_turn_ids(conn, patient_id)
            deleted_texts = {
                text
                for values in self._deleted_texts_by_category(conn, patient_id).values()
                for text in values
                if text
            }
            query = f"""SELECT id, session_id, turn_id, turn_state, response_status,
                               user_message, assistant_message, created_at
                        FROM {_TURN_TABLE} WHERE patient_id=?"""
            params: list[Any] = [patient_id]
            if excluded_session:
                query += " AND COALESCE(session_id, '') != ?"
                params.append(excluded_session)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(_MAX_TURNS_IN_CONTEXT)
            rows = list(reversed(conn.execute(query, tuple(params)).fetchall()))
        lines = []
        for row in rows:
            if str(row["turn_state"] or "").upper() in {"CANCELLED", "FAILED", "RESPONSE_CANCELLED"}:
                continue
            source_ids = {str(row["id"]), str(row["turn_id"] or "")}
            if source_ids & suppressed:
                continue
            if any(
                text in str(row["user_message"] or "")
                or text in str(row["assistant_message"] or "")
                for text in deleted_texts
            ):
                continue
            stamp = str(row["created_at"] or "")[:19]
            session_id = row["session_id"] or "未命名会话"
            lines.append(f"[{stamp}][会话:{session_id}] 患者：{_clip(row['user_message'], 500)}")
            if str(row["response_status"] or "").lower() not in {"cancelled", "failed"}:
                assistant = str(row["assistant_message"] or "").strip()
                if assistant:
                    lines.append(f"[{stamp}][会话:{session_id}] 助手：{_clip(assistant, 500)}")
        return (
            _clip("【跨会话长期证据（SQLite）】\n" + "\n".join(lines), _MAX_CONTEXT_CHARS)
            if lines else ""
        )

    def get_relevant_evidence(
        self,
        patient_id: str,
        final_text: str,
        exclude_session_id: Optional[str] = None,
    ) -> str:
        """Return semantic event evidence only; never fake it with recent SQLite turns."""
        patient_id = self._patient_id(patient_id)
        text = str(final_text or "").strip()
        if not text:
            return ""
        gists = self.get_relevant_event_gists(patient_id, text, topk=_EVENT_GIST_TOPK)
        return self._render_event_gists(gists) if gists else ""

    def is_memobase_available(self) -> bool:
        return self._ensure_memobase() is not None

    def replay_unsynced(self, patient_id: str, timeout: float = 30.0) -> bool:
        """Replay local turns in order without moving past a failed turn."""
        patient_id = self._patient_id(patient_id)
        future = self._submit_memobase(self._replay_unsynced_impl, patient_id, True)
        if future is None:
            return False
        try:
            return bool(future.result(timeout=timeout))
        except FutureTimeoutError:
            self._log(f"[Memobase] 补发超时: patient={patient_id}")
        except Exception as exc:
            self._log(f"[Memobase] 补发失败: {type(exc).__name__}")
        return False

    def prewarm_semantic_retrieval(self, patient_id: str) -> bool:
        patient_id = self._patient_id(patient_id)
        return self._submit_memobase(self._prewarm_semantic_retrieval_impl, patient_id) is not None

    def _prewarm_semantic_retrieval_impl(self, patient_id: str) -> bool:
        client = self._ensure_memobase()
        if client is None:
            return False
        try:
            user = self._get_or_create_memobase_user(client, patient_id)
            self._search_event_gist(
                user,
                "语义检索预热",
                {"topk": 1, "similarity_threshold": 1.01},
            )
            return True
        except Exception as exc:
            self._mark_memobase_failed(exc)
            self._log(f"[Memobase] 语义检索预热失败: {type(exc).__name__}")
            return False

    def flush(self, patient_id: str, timeout: float = 30.0) -> bool:
        """Finish pending inserts and process the patient's Memobase buffer."""
        patient_id = self._patient_id(patient_id)
        future = self._submit_memobase(self._flush_memobase_impl, patient_id)
        if future is None:
            return False
        try:
            return bool(future.result(timeout=timeout))
        except FutureTimeoutError:
            self._log(f"[Memobase] flush 超时: patient={patient_id}")
        except Exception as exc:
            self._log(f"[Memobase] flush 失败: {type(exc).__name__}")
        return False

    def get_context_for_llm(self, patient_id: str) -> str:
        """Build bounded prompt context using only local, non-blocking reads."""
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            self._sync_memory_items_from_snapshot(conn, patient_id, snapshot)
            snapshot = self._snapshot_with_deleted_filtered(conn, snapshot)
            items_by_category = self._active_items_by_category(
                conn,
                patient_id,
                mode=WELLBEING,
            )
            turns = self._read_turns(conn, patient_id, _MAX_TURNS_IN_SNAPSHOT)
            deleted_texts = {
                text
                for values in self._deleted_texts_by_category(conn, patient_id).values()
                for text in values
                if text
            }
        profile = snapshot.get("profile") or {}
        emotion = (snapshot.get("emotion") or {}).get("latest") or {}
        narrative = str(snapshot.get("narrative") or "").strip()
        if not any((profile, turns, items_by_category, emotion, narrative)):
            return ""

        lines = ["【患者长期记忆】"]
        if profile:
            profile_bits = []
            labels = {"name": "姓名", "age": "年龄", "gender": "性别", "education_years": "受教育年限"}
            for key, value in profile.items():
                if value in (None, ""):
                    continue
                label = labels.get(key, key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                profile_bits.append(f"{label}：{value}")
            if profile_bits:
                lines.append("【患者基本信息】" + "；".join(profile_bits))
        for title, key in (("长期事实", "facts"), ("偏好与兴趣", "preferences"), ("近期事件", "events")):
            values = [item["content"] for item in items_by_category.get(key, [])]
            if values:
                lines.append(f"【{title}】" + "；".join(_clip(value, 240) for value in values))
        if narrative:
            lines.append("【既有记忆叙事】" + _clip(narrative, 1000))
        if emotion:
            scores = emotion.get("scores") if isinstance(emotion, Mapping) else {}
            scores = scores if isinstance(scores, Mapping) else {}
            top = sorted(
                ((name, float(scores.get(name, 0.0) or 0.0)) for name in _EMOTIONS),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
            score_text = "，".join(f"{name}={score:.2f}" for name, score in top if score > 0)
            emotion_line = f"【近期情绪】{emotion.get('dominant') or 'calm'}"
            if score_text:
                emotion_line += f"（{score_text}）"
            trend = [
                item.get("dominant")
                for item in ((snapshot.get("emotion") or {}).get("history") or [])[-5:]
                if isinstance(item, Mapping) and item.get("dominant")
            ]
            if len(trend) > 1:
                emotion_line += f"；轨迹：{' → '.join(trend)}"
            lines.append(emotion_line)
        strategies = [item["content"] for item in items_by_category.get("comfort_strategies", [])]
        if strategies:
            lines.append("【有效安抚策略】" + "；".join(_clip(item, 180) for item in strategies))
        if turns:
            lines.append("【近期对话（按会话和时间区分）】")
            for turn in turns[-_MAX_TURNS_IN_CONTEXT:]:
                if any(
                    text in str(turn.get("user_message") or "")
                    or text in str(turn.get("assistant_message") or "")
                    for text in deleted_texts
                ):
                    continue
                sid = turn.get("session_id") or "未命名会话"
                stamp = str(turn.get("created_at") or "")[:19]
                lines.append(
                    f"[{stamp}][会话:{sid}] 患者：{_clip(turn.get('user_message'), 500)}"
                )
                lines.append(f"[{stamp}][会话:{sid}] 助手：{_clip(turn.get('assistant_message'), 500)}")
        return _clip("\n".join(lines), _MAX_CONTEXT_CHARS)

    @staticmethod
    def _patient_id(patient_id: str) -> str:
        patient_id = str(patient_id or "").strip()
        if not patient_id:
            raise ValueError("patient_id is required")
        return patient_id

    def _enqueue_memobase(
        self,
        patient_id: str,
    ) -> None:
        if self._memobase is None and not self._memobase_auto:
            return
        self._submit_memobase(self._replay_unsynced_impl, patient_id)

    def _enqueue_memobase_deletion(
        self,
        patient_id: str,
        content: str,
        source_turn_id: str,
    ) -> bool:
        if self._memobase is None and not self._memobase_auto:
            return False
        future = self._submit_memobase(
            self._delete_memobase_memory_item_impl,
            patient_id,
            str(content or ""),
            str(source_turn_id or ""),
        )
        return future is not None

    def _delete_memobase_memory_item_impl(
        self,
        patient_id: str,
        content: str,
        source_turn_id: str = "",
    ) -> bool:
        content = str(content or "").strip()
        source_turn_id = str(source_turn_id or "").strip()
        if not content and not source_turn_id:
            return False
        client = self._ensure_memobase()
        if client is None:
            return False
        try:
            user = self._get_or_create_memobase_user(client, patient_id)
        except Exception as exc:
            self._mark_memobase_failed(exc)
            return False

        deleted = 0
        if content:
            try:
                profiles = user.profile(max_token_size=5000)
            except TypeError:
                profiles = user.profile()
            except Exception as exc:
                self._log(f"[Memobase] 删除 profile 查询失败: {type(exc).__name__}")
                profiles = []
            for profile in list(profiles or []):
                profile_content = str(self._field(profile, "content") or "")
                profile_id = str(self._field(profile, "id") or "")
                if profile_id and profile_content and (
                    content in profile_content or profile_content in content
                ):
                    try:
                        if user.delete_profile(profile_id):
                            deleted += 1
                    except Exception as exc:
                        self._log(
                            f"[Memobase] 删除 profile 失败: {type(exc).__name__}"
                        )

        gists: list[dict[str, Any]] = []
        if content:
            try:
                raw_gists = self._search_event_gist(
                    user,
                    content,
                    {
                        "topk": max(10, int(_EVENT_GIST_TOPK)),
                        "similarity_threshold": 0.0,
                    },
                )
                gists = self._normalise_event_gists(
                    patient_id,
                    raw_gists,
                    max_items=max(10, int(_EVENT_GIST_TOPK)),
                )
            except Exception as exc:
                self._log(f"[Memobase] 删除 event 查询失败: {type(exc).__name__}")
        for gist in gists:
            gist_content = str(gist.get("content") or "")
            gist_source_turn = str(gist.get("source_turn_id") or "")
            gist_turn = str(gist.get("turn_id") or "")
            if source_turn_id and source_turn_id not in {gist_source_turn, gist_turn}:
                continue
            if content and content not in gist_content:
                continue
            event_id = str(gist.get("event_id") or "")
            if not event_id:
                continue
            try:
                if user.delete_event(event_id):
                    deleted += 1
            except Exception as exc:
                self._log(f"[Memobase] 删除 event 失败: {type(exc).__name__}")
        self._log(
            f"[Memobase] deletion_cleanup patient={patient_id} "
            f"source_turn_id={source_turn_id or '-'} deleted={deleted}"
        )
        return deleted > 0

    def _submit_memobase(self, callback: Callable[..., Any], *args: Any):
        if self._memobase is None and not self._memobase_auto:
            return None
        with self._memobase_lock:
            if self._memobase_executor is None:
                self._memobase_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="EmotionMemobase",
                )
            try:
                return self._memobase_executor.submit(callback, *args)
            except RuntimeError:
                return None

    def _ensure_memobase(self) -> Any:
        if self._memobase is not None:
            return self._memobase
        if not self._memobase_auto or time.monotonic() < self._memobase_retry_at:
            return None
        with self._memobase_lock:
            if self._memobase is not None:
                return self._memobase
            if time.monotonic() < self._memobase_retry_at:
                return None
            self._memobase = self._load_memobase()
            if self._memobase is None:
                self._memobase_retry_at = time.monotonic() + _MEMOBASE_RETRY_SECONDS
            return self._memobase

    def _mark_memobase_failed(self, exc: Exception) -> None:
        self._log(f"[Memobase] 同步失败，保留 SQLite 待补发: {type(exc).__name__}")
        if self._memobase_auto:
            with self._memobase_lock:
                client = self._memobase
                self._memobase = None
                self._memobase_retry_at = time.monotonic() + _MEMOBASE_RETRY_SECONDS
            close = getattr(getattr(client, "client", None), "close", None)
            if callable(close):
                close()

    @staticmethod
    def _memobase_user_id(patient_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lmca-memobase:{patient_id}"))

    @staticmethod
    def _get_or_create_memobase_user(client: Any, patient_id: str) -> Any:
        memobase_user_id = EmotionMemobase._memobase_user_id(patient_id)
        getter = getattr(client, "get_or_create_user", None)
        try:
            if callable(getter):
                return getter(memobase_user_id)
            return client.get_user(memobase_user_id)
        except Exception:
            client.add_user({"patient_id": patient_id}, id=memobase_user_id)
            return client.get_user(memobase_user_id)

    def _read_unsynced_turns(self, patient_id: str) -> list[sqlite3.Row]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT last_synced_turn_id FROM {_SYNC_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
            last_synced = int(row[0]) if row else 0
            return conn.execute(
                f"""SELECT id, session_id, turn_id, user_message, assistant_message,
                           emotion_json, created_at
                    FROM {_TURN_TABLE}
                    WHERE patient_id=? AND id>?
                    ORDER BY id ASC""",
                    (patient_id, last_synced),
            ).fetchall()

    def get_sync_state(self, patient_id: str) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT last_synced_turn_id, attempt_count, next_retry_at, last_error, "
                f"lease_until, updated_at FROM {_SYNC_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
        if not row:
            return {
                "patient_id": patient_id,
                "last_synced_turn_id": 0,
                "attempt_count": 0,
                "next_retry_at": None,
                "last_error": None,
                "lease_until": None,
                "updated_at": None,
            }
        result = dict(row)
        result["patient_id"] = patient_id
        return result

    def _set_sync_state(
        self,
        patient_id: str,
        *,
        last_synced_turn_id: Optional[int] = None,
        attempt_count: Optional[int] = None,
        next_retry_at: Optional[str] = None,
        last_error: Optional[str] = None,
        lease_until: Optional[str] = None,
    ) -> None:
        current = self.get_sync_state(patient_id)
        values = {
            "last_synced_turn_id": int(
                current["last_synced_turn_id"] if last_synced_turn_id is None else last_synced_turn_id
            ),
            "attempt_count": int(current["attempt_count"] if attempt_count is None else attempt_count),
            "next_retry_at": current["next_retry_at"] if next_retry_at is None else next_retry_at,
            "last_error": current["last_error"] if last_error is None else last_error,
            "lease_until": current["lease_until"] if lease_until is None else lease_until,
            "updated_at": _now(),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                f"""INSERT INTO {_SYNC_TABLE}
                    (patient_id, last_synced_turn_id, attempt_count, next_retry_at,
                     last_error, lease_until, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                      last_synced_turn_id=excluded.last_synced_turn_id,
                      attempt_count=excluded.attempt_count,
                      next_retry_at=excluded.next_retry_at,
                      last_error=excluded.last_error,
                      lease_until=excluded.lease_until,
                      updated_at=excluded.updated_at""",
                (
                    patient_id, values["last_synced_turn_id"], values["attempt_count"],
                    values["next_retry_at"], values["last_error"], values["lease_until"],
                    values["updated_at"],
                ),
            )

    def _claim_sync_lease(self, patient_id: str) -> bool:
        now = datetime.now()
        lease_until = (now + timedelta(seconds=_SYNC_LEASE_SECONDS)).isoformat()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT lease_until FROM {_SYNC_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
            if row and row["lease_until"] and str(row["lease_until"]) > now.isoformat():
                return False
            conn.execute(
                f"""INSERT INTO {_SYNC_TABLE}
                    (patient_id, lease_until, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                      lease_until=excluded.lease_until, updated_at=excluded.updated_at""",
                (patient_id, lease_until, _now()),
            )
        return True

    def _mark_turn_synced(self, patient_id: str, turn_id: int) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                f"""INSERT INTO {_SYNC_TABLE}
                    (patient_id, last_synced_turn_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        last_synced_turn_id=excluded.last_synced_turn_id,
                        attempt_count=0,
                        next_retry_at=NULL,
                        last_error=NULL,
                        lease_until=NULL,
                        updated_at=excluded.updated_at""",
                (patient_id, int(turn_id), _now()),
            )

    def _mark_sync_failed(self, patient_id: str, exc: Exception) -> None:
        state = self.get_sync_state(patient_id)
        attempts = int(state.get("attempt_count") or 0) + 1
        delay = min(_SYNC_RETRY_MAX_SECONDS, _MEMOBASE_RETRY_SECONDS * (2 ** min(attempts - 1, 5)))
        self._set_sync_state(
            patient_id,
            attempt_count=attempts,
            next_retry_at=(datetime.now() + timedelta(seconds=delay)).isoformat(),
            last_error=f"{type(exc).__name__}: {exc}"[:500],
            lease_until=None,
        )
        with self._lock, self._connection() as conn:
            conn.execute(
                f"UPDATE {_SYNC_TABLE} SET lease_until=NULL, updated_at=? WHERE patient_id=?",
                (_now(), patient_id),
            )

    def _clear_sync_lease(self, patient_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                f"UPDATE {_SYNC_TABLE} SET lease_until=NULL, updated_at=? WHERE patient_id=?",
                (_now(), patient_id),
            )

    def _pending_consolidation_turns(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        start_turn_id: int,
        cutoff_turn_id: int,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = f"""SELECT id, session_id, turn_id, user_message, assistant_message,
                           emotion_json, created_at, turn_state, response_status
                    FROM {_TURN_TABLE}
                    WHERE patient_id=? AND id>? AND id<=?"""
        params: list[Any] = [patient_id, int(start_turn_id), int(cutoff_turn_id)]
        if session_id:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY id ASC"
        rows = conn.execute(query, tuple(params)).fetchall()
        result = []
        for row in rows:
            if str(row["turn_state"] or "").upper() in {"CANCELLED", "FAILED"}:
                continue
            try:
                emotion = json.loads(row["emotion_json"] or "{}")
            except json.JSONDecodeError:
                emotion = {}
            result.append(
                {
                    "local_turn_id": int(row["id"]),
                    "turn_id": row["turn_id"],
                    "session_id": row["session_id"],
                    "user_message": row["user_message"],
                    "assistant_message": row["assistant_message"],
                    "emotion": emotion,
                    "created_at": row["created_at"],
                    "turn_state": row["turn_state"],
                    "response_status": row["response_status"],
                }
            )
        return result

    @staticmethod
    def _default_consolidation_patch(turns: list[Mapping[str, Any]]) -> dict[str, Any]:
        patch = {"facts": [], "preferences": [], "events": []}
        known = {field: set() for field in patch}
        for turn in turns:
            items = _extract_memory_items(str(turn.get("user_message") or ""))
            evidence_turn_id = str(
                turn.get("turn_id") or turn.get("local_turn_id") or ""
            ).strip()
            for field in patch:
                for item in items[field]:
                    if item in known[field]:
                        continue
                    known[field].add(item)
                    patch[field].append(
                        {
                            "id": f"derived:{evidence_turn_id}:{field}:{len(patch[field])}",
                            "text": item,
                            "source": "derived",
                            "status": "active",
                            "evidence_turn_ids": [evidence_turn_id] if evidence_turn_id else [],
                        }
                    )
        return patch

    @staticmethod
    def _validate_consolidation_patch(patch: Any) -> dict[str, Any]:
        if patch is None:
            return {}
        if not isinstance(patch, Mapping):
            raise ValueError("consolidation patch must be a mapping")
        unsupported = set(patch) - _CONSOLIDATABLE_FIELDS
        if unsupported:
            raise ValueError("unsupported consolidation fields: " + ", ".join(sorted(unsupported)))
        result: dict[str, Any] = {}
        for field in _CONSOLIDATABLE_FIELDS:
            if field not in patch:
                continue
            value = patch[field]
            if field == "narrative":
                result[field] = _clip(value, 4000) if value else ""
                continue
            if not isinstance(value, list):
                raise ValueError(f"{field} must be a list")
            result[field] = _json_copy(value)
        return result

    @staticmethod
    def _merge_consolidation_patch(
        snapshot: Mapping[str, Any], patch: Mapping[str, Any]
    ) -> dict[str, Any]:
        merged = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        for field in ("facts", "preferences", "events"):
            current = list(merged.get(field) or [])
            if field == "events":
                current = _compact_event_entries(current)
            superseded = _superseded_turn_ids({field: current})
            for item in patch.get(field) or []:
                if isinstance(item, Mapping):
                    entry = dict(item)
                    if str(entry.get("source") or "derived").lower() == "manual":
                        continue
                    if str(entry.get("status") or "active").lower() != "active":
                        continue
                    evidence = {
                        str(value).strip()
                        for value in entry.get("evidence_turn_ids") or []
                        if str(value).strip()
                    }
                    source_turn = str(entry.get("source_turn_id") or "").strip()
                    if source_turn:
                        evidence.add(source_turn)
                    if evidence & superseded:
                        continue
                    text = _memory_item_text(entry)
                    if not text or any(_memory_item_text(value) == text for value in current):
                        continue
                    entry.setdefault("id", uuid.uuid4().hex)
                    entry.setdefault("source", "derived")
                    entry.setdefault("status", "active")
                    entry.setdefault("evidence_turn_ids", [])
                    if field == "events":
                        current = _compact_event_entries([*current, entry])
                    else:
                        current.append(entry)
                elif str(item).strip() and not any(
                    _memory_item_text(value) == str(item).strip()
                    for value in current
                ):
                    if field == "events":
                        current = _compact_event_entries([*current, item])
                    else:
                        current.append(item)
            merged[field] = current[-40:]
        if "comfort_strategies" in patch:
            strategies = list(merged.get("comfort_strategies") or [])
            for item in patch["comfort_strategies"]:
                if not isinstance(item, Mapping):
                    continue
                strategy = str(item.get("strategy") or "").strip()
                if not strategy:
                    continue
                existing = next(
                    (index for index, current in enumerate(strategies)
                     if isinstance(current, Mapping) and str(current.get("strategy") or "").strip() == strategy),
                    None,
                )
                if existing is None:
                    strategies.append(dict(item))
                elif (
                    str(strategies[existing].get("source") or "derived") != "manual"
                    and str(strategies[existing].get("status") or "active") == "active"
                ):
                    strategies[existing] = dict(item)
            merged["comfort_strategies"] = strategies[-40:]
        if patch.get("narrative"):
            merged["narrative"] = str(patch["narrative"])
        return merged

    def consolidate_pending_turns(
        self,
        patient_id: str,
        cutoff_turn_id: Optional[int] = None,
        *,
        patch: Optional[Mapping[str, Any]] = None,
        patch_builder: Optional[Callable[..., Mapping[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply one CAS-protected incremental patch to turns after the snapshot waterline."""
        patient_id = self._patient_id(patient_id)
        session_id = str(session_id).strip() or None if session_id else None
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            upper = cutoff_turn_id
            if upper is None:
                query = f"SELECT MAX(id) FROM {_TURN_TABLE} WHERE patient_id=?"
                params: list[Any] = [patient_id]
                if session_id:
                    query += " AND session_id=?"
                    params.append(session_id)
                row = conn.execute(query, tuple(params)).fetchone()
                upper = int(row[0] or 0)
            upper = int(upper)
            start = int(snapshot.get("last_consolidated_turn_id") or 0)
            turns = self._pending_consolidation_turns(conn, patient_id, start, upper, session_id)
            base_revision = int(snapshot.get("revision") or 0)
        if not turns:
            if upper <= start:
                return {
                    "applied": True,
                    "patient_id": patient_id,
                    "revision": base_revision,
                    "last_consolidated_turn_id": start,
                    "turn_count": 0,
                    "patch": {},
                }
            validated = {}
            merged = dict(snapshot)
            merged["last_consolidated_turn_id"] = upper
            with self._lock, self._connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                current = self._load_snapshot(conn, patient_id)
                current_revision = int(current.get("revision") or 0)
                current_waterline = int(current.get("last_consolidated_turn_id") or 0)
                if current_revision != base_revision or current_waterline != start:
                    raise MemoryRevisionConflict(current_revision)
                updated_at = _now()
                stored = _normalise_snapshot(patient_id, merged)
                stored["revision"] = base_revision + 1
                stored["last_consolidated_turn_id"] = upper
                stored["updated_at"] = updated_at
                stored["updated_by"] = "system"
                payload = json.dumps(stored, ensure_ascii=False, sort_keys=True)
                cursor = conn.execute(
                    f"""UPDATE {_SNAPSHOT_TABLE}
                        SET snapshot_json=?, revision=?, last_consolidated_turn_id=?,
                            updated_at=?, updated_by=?
                        WHERE patient_id=? AND revision=? AND last_consolidated_turn_id=?""",
                    (payload, stored["revision"], upper, updated_at, "system",
                     patient_id, base_revision, start),
                )
                if cursor.rowcount != 1:
                    raise MemoryRevisionConflict(current_revision)
                self._sync_memory_items_from_snapshot(conn, patient_id, stored)
            return {
                "applied": True,
                "patient_id": patient_id,
                "revision": base_revision + 1,
                "last_consolidated_turn_id": upper,
                "turn_count": 0,
                "patch": {},
            }
        if patch_builder is not None:
            try:
                generated = patch_builder(turns, snapshot)
            except TypeError:
                generated = patch_builder(turns)
        else:
            generated = patch if patch is not None else self._default_consolidation_patch(turns)
        validated = self._validate_consolidation_patch(generated)
        merged = self._merge_consolidation_patch(snapshot, validated)
        merged["last_consolidated_turn_id"] = upper
        merged["revision"] = base_revision
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._load_snapshot(conn, patient_id)
            current_revision = int(current.get("revision") or 0)
            current_waterline = int(current.get("last_consolidated_turn_id") or 0)
            if current_revision != base_revision or current_waterline != start:
                raise MemoryRevisionConflict(current_revision)
            updated_at = _now()
            stored = _normalise_snapshot(patient_id, merged)
            stored["revision"] = base_revision + 1
            stored["last_consolidated_turn_id"] = upper
            stored["updated_at"] = updated_at
            stored["updated_by"] = "system"
            stored.pop("mmse_scores", None)
            stored.pop("emotion_history", None)
            stored.pop("recent_turns", None)
            payload = json.dumps(stored, ensure_ascii=False, sort_keys=True)
            cursor = conn.execute(
                f"""UPDATE {_SNAPSHOT_TABLE}
                    SET snapshot_json=?, revision=?, last_consolidated_turn_id=?,
                        updated_at=?, updated_by=?
                    WHERE patient_id=? AND revision=? AND last_consolidated_turn_id=?""",
                (
                    payload, stored["revision"], upper, updated_at, "system",
                    patient_id, base_revision, start,
                ),
            )
            if cursor.rowcount != 1:
                raise MemoryRevisionConflict(current_revision)
            self._sync_memory_items_from_snapshot(conn, patient_id, stored)
        return {
            "applied": True,
            "patient_id": patient_id,
            "revision": base_revision + 1,
            "last_consolidated_turn_id": upper,
            "turn_count": len(turns),
            "patch": validated,
        }

    def _replay_unsynced_impl(self, patient_id: str, force: bool = False) -> bool:
        if not force:
            state = self.get_sync_state(patient_id)
            retry_at = str(state.get("next_retry_at") or "")
            if retry_at and retry_at > datetime.now().isoformat():
                return False
        if not self._claim_sync_lease(patient_id):
            return False
        rows = self._read_unsynced_turns(patient_id)
        if not rows:
            self._set_sync_state(patient_id, lease_until=None)
            return True
        client = self._ensure_memobase()
        if client is None:
            self._mark_sync_failed(patient_id, RuntimeError("memobase unavailable"))
            return False
        try:
            user = self._get_or_create_memobase_user(client, patient_id)
            for row in rows:
                emotion = json.loads(row["emotion_json"] or "{}")
                from memobase import ChatBlob

                user.insert(
                    ChatBlob(
                        messages=[
                            {"role": "user", "content": row["user_message"]},
                            {"role": "assistant", "content": row["assistant_message"]},
                        ],
                        fields={
                            "session_id": row["session_id"],
                            "source_turn_id": int(row["id"]),
                            "turn_id": row["turn_id"],
                            "dominant_emotion": _dominant_emotion(emotion),
                            "created_at": row["created_at"],
                        },
                    )
                )
            user.flush(sync=True)
            if rows:
                self._mark_turn_synced(patient_id, int(rows[-1]["id"]))
            return True
        except Exception as exc:
            self._mark_sync_failed(patient_id, exc)
            self._mark_memobase_failed(exc)
            return False

    def _flush_memobase_impl(self, patient_id: str) -> bool:
        # Explicit flush is a recovery command and may run immediately after a failed worker.
        self._clear_sync_lease(patient_id)
        return self._replay_unsynced_impl(patient_id, True)

    def close(self) -> None:
        with self._memobase_lock:
            if self._memobase_executor is not None:
                self._memobase_executor.shutdown(wait=False, cancel_futures=True)
                self._memobase_executor = None
        close = getattr(getattr(self._memobase, "client", None), "close", None)
        if callable(close):
            close()
        if self._memory_conn is not None:
            with self._lock:
                self._memory_conn.close()
                self._memory_conn = None


__all__ = ["EmotionMemobase", "MemoryRevisionConflict"]
