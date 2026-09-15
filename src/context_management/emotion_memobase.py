"""Local-first long-term memory for patient conversations.

SQLite is the source of truth. Local Memobase supplies semantic recall and is
kept as a recoverable mirror, so its failure never breaks patient dialogue.
"""

from __future__ import annotations

import copy
import json
import os
import re
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

from .retrieval_cache import (
    RetrievalBusyError,
    RetrievalCache,
    RetrievalHttpClient,
    RetrievalUnavailableError,
)


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
_MIRROR_OP_TABLE = "emotion_memobase_mirror_ops"
_SESSION_REFLECTION_TABLE = "emotion_memobase_session_reflections"
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
_AUTHORITATIVE_CARD_TOKEN_BUDGET = 900
_LOCAL_FALLBACK_ITEM_LIMIT = 3
# Marks context produced by the query-independent local fallback rather than by
# semantic retrieval.  Callers that report retrieval quality must be able to
# tell the two apart, so the marker is part of this module's public contract.
LOCAL_ACTIVE_MEMORY_MARKER = "[local_active_memory]"
_SYNC_LEASE_SECONDS = 60
_SYNC_RETRY_MAX_SECONDS = 300
_REFLECTION_LEASE_SECONDS = 120
_REFLECTION_RETRY_SECONDS = 60
_REFLECTION_MAX_ATTEMPTS = 3
_MEMORY_WORKER_INTERVAL = 1.0
_VERIFICATION_PROMPT_VERSION = "v3"
_CONSOLIDATABLE_FIELDS = {
    "facts", "preferences", "events", "comfort_strategies", "narrative"
}
_REFLECTION_CATEGORIES = {
    "facts", "preferences", "events", "emotional_triggers", "boundaries",
    "comfort_strategies",
}


_REFLECTION_OPERATIONS = {"ADD", "SUPERSEDE", "CANDIDATE", "NOOP", "BLOCK"}
_TEMPORARY_SCOPE_MARKERS = (
    "现在", "今天", "今晚", "这次", "暂时", "先别", "这几天", "这两天",
    "本周", "近期", "这阵子", "过几天", "几天后", "明天", "下周", "下月",
    "下个月", "以后", "之后",
)
_TEMPORARY_REFUSAL_MARKERS = ("不想", "不愿", "不要", "先别", "暂时", "别聊")
_UNCERTAINTY_MARKERS = ("可能", "也许", "好像", "不太确定", "不确定", "记得", "似乎")
_CORRECTION_MARKERS = ("不是", "改成", "记错", "搬到", "改为")
_NEGATED_NON_MEMORY_TERMS = (
    "失眠", "睡不着", "害怕", "不怕", "恐惧", "吵架", "争吵",
    "伤害自己", "自伤", "自杀", "危机",
)
_STRONG_NEGATION_MARKERS = ("没有", "没", "未", "不怕", "不失眠", "从未")
_EPHEMERAL_CONTEXT_MARKERS = ("刚", "刚刚", "今天", "上午", "下午", "早饭", "喝水", "天气")
_LONG_TERM_SEMANTIC_MARKERS = (
    "喜欢", "不喜欢", "希望", "不想", "不要", "不能", "住", "姓", "叫",
    "女儿", "儿子", "家人", "害怕", "担心", "难过", "检查", "复查", "睡",
    "药", "参加", "照顾", "关系", "下结论", "出门",
)
_AFFECTIVE_MARKERS = (
    "担心", "担忧", "害怕", "恐惧", "难过", "难受", "伤心", "焦虑",
    "开心", "高兴", "平静", "安心", "生气", "烦", "发抖", "哭",
)


def _direct_non_memory_denial(text: str) -> bool:
    """Reject only an explicit denial scoped directly to a symptom/event/crisis."""
    for term in _NEGATED_NON_MEMORY_TERMS:
        if re.search(rf"(?:没有|没|未|从未|不曾|不是|并非)"
                     rf"(?!想到|想过|意识到|料到)[^，。！？；]{{0,6}}"
                     rf"{re.escape(term)}", text):
            return True
    if re.search(r"(?:不怕|不失眠|不自杀|不自伤)", text):
        return True
    if re.search(r"(?:不|不太)(?:会|曾|再)?(?:害怕|失眠|睡不着|吵架|争吵|危机)", text):
        return True
    return False

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


def _memory_item_is_current(item: Mapping[str, Any]) -> bool:
    if str(item.get("status") or "").lower() != "active":
        return False
    valid_until = str(item.get("valid_until") or "").strip()
    if not valid_until:
        return True
    try:
        if "T" not in valid_until and " " not in valid_until:
            return datetime.fromisoformat(valid_until).date() >= datetime.now().date()
        expires_at = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
        now = datetime.now(expires_at.tzinfo) if expires_at.tzinfo else datetime.now()
        return expires_at >= now
    except ValueError:
        return False


def _budgeted_values(values: list[str], token_budget: int, item_limit: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for value in values:
        text = _clip(value, item_limit)
        cost = _token_estimate(text)
        if not text or used + cost > token_budget:
            continue
        selected.append(text)
        used += cost
    return selected


def _emotion_trend(history: Any) -> str:
    entries = [
        item for item in history or []
        if isinstance(item, Mapping)
        and str(item.get("analysis_status") or "final") == "final"
        and str(item.get("dominant") or "unknown") not in {"unknown", "unavailable"}
    ][-5:]
    if len(entries) < 2:
        return ""
    try:
        valence_change = float(entries[-1].get("valence") or 0.0) - float(
            entries[0].get("valence") or 0.0
        )
        arousal_change = float(entries[-1].get("arousal") or 0.0) - float(
            entries[0].get("arousal") or 0.0
        )
    except (TypeError, ValueError):
        return ""
    if valence_change >= 0.2:
        return "好转"
    if valence_change <= -0.2:
        return "加重"
    if arousal_change <= -0.2:
        return "趋稳"
    if arousal_change >= 0.2:
        return "唤醒升高"
    return "平稳"


def _local_fallback_enabled() -> bool:
    return os.getenv("MEMORY_LOCAL_FALLBACK_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


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


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


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
        return "unknown"
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
        return {}
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
        "_narrative_manual": False,
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
    if profile is not None:
        # Registered patients use the patients table as the profile authority;
        # snapshot fields are projections and must not resurrect deleted data.
        data["profile"] = dict(profile)
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
    return isinstance(item, Mapping) and str(item.get("source") or "").lower() in {
        "manual", "user", "user_direct", "user_correction", "clinician_confirmed",
    }


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
            if status in {"deleted", "superseded", "blocked"} or (
                status == "active"
                and source in {"manual", "user", "user_direct", "user_correction", "clinician_confirmed"}
            ):
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
        session_reflector: Optional[Callable[[str], Any]] = None,
        session_verifier: Optional[Callable[[str], Any]] = None,
        memobase: Any = None,
        logger: Callable[[str], Any] = print,
        audit_event: Optional[Callable[..., Any]] = None,
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
        self._session_reflector = session_reflector
        self._session_verifier = session_verifier
        self._auto_verify_reflections = session_verifier is not None or session_reflector is None
        self._memobase = memobase_client if memobase_client is not None else memobase
        self._memobase_auto = self._memobase is None and bool(
            os.getenv("MEMOBASE_PROJECT_URL") and os.getenv("MEMOBASE_API_KEY")
        )
        self._memobase_retry_at = 0.0
        self._memobase_executor: Optional[ThreadPoolExecutor] = None
        self._memobase_lock = threading.Lock()
        self._retrieval_http_timeout_s = max(
            0.05, float(os.getenv("MEMORY_RETRIEVAL_HTTP_TIMEOUT_S", "1.5"))
        )
        self._retrieval_cache = RetrievalCache(
            ttl_s=float(os.getenv("MEMORY_RETRIEVAL_CACHE_TTL_S", "60")),
            max_entries=int(os.getenv("MEMORY_RETRIEVAL_CACHE_MAX_ENTRIES", "128")),
            max_inflight=int(os.getenv("MEMORY_RETRIEVAL_MAX_INFLIGHT", "2")),
            wait_timeout_s=float(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_S", "0.25")),
        )
        self._memory_worker_stop = threading.Event()
        self._memory_worker_wakeup = threading.Event()
        self._memory_worker_thread: Optional[threading.Thread] = None
        self._log = logger
        self._audit_event = audit_event
        self._ensure_schema()

    def _emit_audit(self, patient_id: str | None, action: str, outcome: str) -> None:
        callback = self._audit_event
        if not callable(callback):
            return
        try:
            callback(
                patient_id=str(patient_id or "").strip() or None,
                action=str(action),
                outcome=str(outcome),
            )
        except Exception:
            self._log(f"[EmotionMemory] audit event failed: {action}")

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
                timeout=self._retrieval_http_timeout_s,
                trust_env=False,
            )
            if not client.ping():
                client.client.close()
                return None
            client.client.timeout = httpx.Timeout(30.0)
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
                    consolidation_status TEXT NOT NULL DEFAULT 'pending',
                    consolidated_at TEXT,
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
                    created_at TEXT NOT NULL,
                    slot_key TEXT,
                    observed_at TEXT,
                    valid_until TEXT,
                    sensitivity TEXT NOT NULL DEFAULT 'normal',
                    confidence REAL,
                    confirmed_by TEXT,
                    evidence_json TEXT,
                    supersedes_item_id TEXT,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS {_MEMORY_DELETION_TABLE} (
                    item_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    deletion_token TEXT NOT NULL,
                    deleted_by TEXT,
                    deleted_at TEXT NOT NULL,
                    item_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {_MIRROR_OP_TABLE} (
                    op_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    item_version INTEGER NOT NULL,
                    op_type TEXT NOT NULL,
                    source_turn_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT,
                    error_code TEXT,
                    lease_until TEXT,
                    lease_token TEXT,
                    payload_json TEXT,
                    dead_letter_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {_SESSION_REFLECTION_TABLE} (
                    patient_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    session_summary TEXT,
                    summary_evidence_json TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT,
                    lease_until TEXT,
                    lease_token TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY (patient_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_{_MEMORY_ITEM_TABLE}_patient_status
                    ON {_MEMORY_ITEM_TABLE}(patient_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_{_MEMORY_ITEM_TABLE}_patient_mode
                    ON {_MEMORY_ITEM_TABLE}(patient_id, mode, status);
                CREATE INDEX IF NOT EXISTS idx_{_MIRROR_OP_TABLE}_pending
                    ON {_MIRROR_OP_TABLE}(status, next_retry_at, updated_at);
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
                ("error_code", "TEXT"),
                ("lease_until", "TEXT"),
                ("lease_token", "TEXT"),
                ("payload_json", "TEXT"),
                ("dead_letter_at", "TEXT"),
            ):
                self._ensure_column(conn, _MIRROR_OP_TABLE, column, definition)
            for column, definition in (
                ("turn_id", "TEXT NOT NULL DEFAULT ''"),
                ("turn_state", "TEXT NOT NULL DEFAULT 'DURABLY_CAPTURED'"),
                ("response_status", "TEXT NOT NULL DEFAULT 'responded'"),
                ("consolidation_status", "TEXT NOT NULL DEFAULT 'pending'"),
                ("consolidated_at", "TEXT"),
            ):
                self._ensure_column(conn, _TURN_TABLE, column, definition)
            for column, definition in (
                ("slot_key", "TEXT"),
                ("observed_at", "TEXT"),
                ("valid_until", "TEXT"),
                ("sensitivity", "TEXT NOT NULL DEFAULT 'normal'"),
                ("confidence", "REAL"),
                ("confirmed_by", "TEXT"),
                ("evidence_json", "TEXT"),
                ("supersedes_item_id", "TEXT"),
                ("metadata_json", "TEXT"),
            ):
                self._ensure_column(conn, _MEMORY_ITEM_TABLE, column, definition)
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
            for column, definition in (
                ("session_summary", "TEXT"),
                ("summary_evidence_json", "TEXT"),
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("next_retry_at", "TEXT"),
                ("last_error", "TEXT"),
                ("lease_until", "TEXT"),
                ("lease_token", "TEXT"),
                ("completed_at", "TEXT"),
                ("error_code", "TEXT"),
                ("dead_letter_at", "TEXT"),
            ):
                self._ensure_column(conn, _SESSION_REFLECTION_TABLE, column, definition)
            conn.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{_SESSION_REFLECTION_TABLE}_pending
                    ON {_SESSION_REFLECTION_TABLE}(status, next_retry_at, updated_at)"""
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_MIRROR_OP_TABLE}_due "
                f"ON {_MIRROR_OP_TABLE}(status, next_retry_at, updated_at)"
            )
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
            conn.execute(
                f"UPDATE {_MEMORY_ITEM_TABLE} SET source='legacy_migrated' "
                "WHERE source IS NULL OR TRIM(source)=''"
            )
            self._migrate_legacy_memory(conn)

    def _migrate_legacy_memory(self, conn: sqlite3.Connection) -> None:
        """Backfill missing items once while preserving existing item state."""
        rows = conn.execute(
            f"SELECT patient_id, snapshot_json FROM {_SNAPSHOT_TABLE}"
        ).fetchall()
        for row in rows:
            patient_id = str(row["patient_id"])
            try:
                raw = json.loads(row["snapshot_json"] or "{}")
            except json.JSONDecodeError:
                raw = {}
            self._sync_memory_items_from_snapshot(
                conn,
                patient_id,
                _normalise_snapshot(patient_id, raw),
            )
            self._migrate_legacy_rule_items(conn, patient_id)
            snapshot = self._snapshot_from_items(conn, self._load_snapshot(conn, patient_id))
            self._save_snapshot(conn, snapshot, updated_by="system")
        for row in conn.execute(
            f"SELECT DISTINCT patient_id FROM {_MEMORY_ITEM_TABLE}"
        ).fetchall():
            self._migrate_legacy_rule_items(conn, str(row["patient_id"]))
        conn.execute(
            f"""UPDATE {_TURN_TABLE} AS turns
                SET consolidation_status='consolidated',
                    consolidated_at=COALESCE(consolidated_at, ?)
                WHERE consolidation_status='pending'
                  AND EXISTS (
                    SELECT 1 FROM {_MEMORY_ITEM_TABLE} AS items
                    WHERE items.patient_id=turns.patient_id
                      AND (
                        items.source_turn_id=CAST(turns.id AS TEXT)
                        OR items.source_turn_id=turns.turn_id
                      )
                  )""",
            (_now(),),
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
            f"SELECT item_id FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND status<>'active'",
            (patient_id,),
        ).fetchall()
        result = {str(row["item_id"]) for row in rows}
        result.update(
            str(row["item_id"])
            for row in conn.execute(
                f"SELECT item_id FROM {_MEMORY_DELETION_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchall()
        )
        return result

    def _deleted_texts_by_category(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
    ) -> dict[str, set[str]]:
        rows = conn.execute(
            f"""SELECT category, content FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND status<>'active' AND content<>''""",
            (patient_id,),
        ).fetchall()
        result: dict[str, set[str]] = {}
        for row in rows:
            result.setdefault(str(row["category"]), set()).add(str(row["content"]))
        return result

    def _deleted_turn_ids(self, conn: sqlite3.Connection, patient_id: str) -> set[str]:
        rows = conn.execute(
            f"""SELECT source_turn_id FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND status<>'active'
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
        """Backfill items from a legacy snapshot without overwriting item state."""
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
                source = "legacy_migrated"
                original_source = source
                source_session_id = None
                source_turn_id = None
                slot_key = None
                observed_at = None
                valid_until = None
                sensitivity = "normal"
                confidence = None
                confirmed_by = None
                supersedes_item_id = None
                metadata: dict[str, Any] = {}
                evidence: list[Any] = []
                created_at = now
                updated_at = now
                if isinstance(item, Mapping):
                    source = str(item.get("source") or source).strip() or source
                    original_source = source
                    if source in {"derived", "manual", "user"}:
                        source = "legacy_migrated" if source == "derived" else "user_direct"
                    source_session_id = str(item.get("source_session_id") or "").strip() or None
                    source_turn_id = str(item.get("source_turn_id") or "").strip() or None
                    if not source_turn_id:
                        evidence = item.get("evidence_turn_ids") or []
                        if isinstance(evidence, (list, tuple)) and evidence:
                            source_turn_id = str(evidence[0] or "").strip() or None
                    else:
                        evidence = item.get("evidence_turn_ids") or []
                    slot_key = str(item.get("slot_key") or "").strip() or None
                    observed_at = str(item.get("observed_at") or "").strip() or None
                    valid_until = str(item.get("valid_until") or "").strip() or None
                    sensitivity = str(item.get("sensitivity") or "normal").strip().lower()
                    if sensitivity not in {"normal", "sensitive", "high"}:
                        sensitivity = "normal"
                    try:
                        confidence = float(item["confidence"]) if item.get("confidence") is not None else None
                    except (TypeError, ValueError):
                        confidence = None
                    confirmed_by = str(item.get("confirmed_by") or "").strip() or None
                    supersedes_item_id = str(item.get("supersedes_item_id") or "").strip() or None
                    metadata = _json_object(item.get("metadata_json") or item.get("metadata"))
                    if original_source == "derived":
                        metadata.setdefault("legacy_original_source", original_source)
                    if category == "comfort_strategies" and "effective" in item:
                        metadata.setdefault("effective", item.get("effective"))
                    if category == "comfort_strategies" and "use_count" in item:
                        metadata.setdefault("use_count", item.get("use_count"))
                    created_at = str(item.get("created_at") or now)
                    updated_at = str(item.get("updated_at") or now)
                conn.execute(
                    f"""INSERT OR IGNORE INTO {_MEMORY_ITEM_TABLE}
                        (item_id, patient_id, mode, category, content, source,
                         source_session_id, source_turn_id, version, status,
                         deleted_at, updated_at, created_at, slot_key, observed_at,
                         valid_until, sensitivity, confidence, confirmed_by,
                         evidence_json, supersedes_item_id, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'active', NULL, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        slot_key,
                        observed_at,
                        valid_until,
                        sensitivity,
                        confidence,
                        confirmed_by,
                        json.dumps(
                            [{"turn_id": str(value).strip()} for value in evidence if str(value).strip()],
                            ensure_ascii=False,
                        ) if evidence else None,
                        supersedes_item_id,
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else None,
                    ),
                )
                current = conn.execute(
                    f"SELECT metadata_json FROM {_MEMORY_ITEM_TABLE} WHERE item_id=?",
                    (item_id,),
                ).fetchone()
                if current and not current["metadata_json"] and metadata:
                    conn.execute(
                        f"UPDATE {_MEMORY_ITEM_TABLE} SET metadata_json=? WHERE item_id=?",
                        (json.dumps(metadata, ensure_ascii=False, sort_keys=True), item_id),
                    )

    def _migrate_legacy_rule_items(self, conn: sqlite3.Connection, patient_id: str) -> None:
        """隔离旧规则派生 active，保留审计记录且不再进入陪伴上下文。"""
        rows = conn.execute(
            f"""SELECT item_id, source, metadata_json, version FROM {_MEMORY_ITEM_TABLE}
                WHERE patient_id=? AND category IN ('facts', 'preferences', 'events')
                  AND status='active' AND source IN ('system_inferred', 'derived', 'legacy_migrated')""",
            (patient_id,),
        ).fetchall()
        now = _now()
        for row in rows:
            metadata = _json_object(row["metadata_json"])
            metadata.setdefault("legacy_migration", {})
            metadata["legacy_migration"].update(
                {
                    "from_source": str(metadata.get("legacy_original_source") or row["source"]),
                    "migrated_at": now,
                }
            )
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status='candidate', source='legacy_unverified', version=?,
                        metadata_json=?, updated_at=?
                    WHERE patient_id=? AND item_id=? AND status='active'""",
                (
                    int(row["version"] or 0) + 1,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    now,
                    patient_id,
                    row["item_id"],
                ),
            )

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
            if not _memory_item_is_current(item):
                continue
            result.setdefault(str(item["category"]), []).append(item)
        return result

    @staticmethod
    def _snapshot_item_projection(item: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(item)
        result["id"] = result.get("item_id")
        result["text"] = result.get("content")
        evidence = result.get("evidence_json")
        try:
            evidence = json.loads(evidence or "[]")
        except json.JSONDecodeError:
            evidence = []
        if isinstance(evidence, list):
            result["evidence_turn_ids"] = [
                str(item.get("turn_id") or "").strip()
                for item in evidence
                if isinstance(item, Mapping) and str(item.get("turn_id") or "").strip()
            ]
        if result.get("category") == "comfort_strategies":
            metadata = _json_object(result.get("metadata_json"))
            result["strategy"] = result.get("content")
            result["effective"] = metadata.get("effective")
            if "use_count" in metadata:
                result["use_count"] = metadata["use_count"]
        return result

    def _snapshot_from_items(
        self,
        conn: sqlite3.Connection,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        items = self._active_items_by_category(
            conn,
            str(result["patient_id"]),
            mode=WELLBEING,
        )
        for category in ("facts", "preferences", "events", "comfort_strategies"):
            result[category] = [
                self._snapshot_item_projection(item)
                for item in items.get(category, [])
            ]
        return result

    @staticmethod
    def _item_evidence_json(item: Mapping[str, Any]) -> Optional[str]:
        raw = item.get("evidence_json")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        values = item.get("evidence")
        if values is None:
            values = item.get("evidence_turn_ids") or []
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        evidence = []
        for value in values:
            if isinstance(value, Mapping):
                turn_id = str(value.get("turn_id") or value.get("source_turn_id") or "").strip()
                quote = str(value.get("quote") or "").strip()
                if turn_id:
                    entry = {"turn_id": turn_id}
                    if quote:
                        entry["quote"] = quote
                    evidence.append(entry)
            else:
                turn_id = str(value or "").strip()
                if turn_id:
                    evidence.append({"turn_id": turn_id})
        return json.dumps(evidence, ensure_ascii=False) if evidence else None

    def _insert_memory_item(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        category: str,
        content: str,
        *,
        item_id: Optional[str] = None,
        mode: str = WELLBEING,
        source: str = "system_inferred",
        source_session_id: Optional[str] = None,
        source_turn_id: Optional[str] = None,
        version: int = 1,
        status: str = "active",
        deleted_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        slot_key: Optional[str] = None,
        observed_at: Optional[str] = None,
        valid_until: Optional[str] = None,
        sensitivity: str = "normal",
        confidence: Optional[float] = None,
        confirmed_by: Optional[str] = None,
        evidence_json: Optional[str] = None,
        supersedes_item_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        now = _now()
        item_id = str(item_id or uuid.uuid4().hex)
        updated_at = str(updated_at or now)
        created_at = str(created_at or now)
        sensitivity = str(sensitivity or "normal").strip().lower()
        if sensitivity not in {"normal", "sensitive", "high"}:
            sensitivity = "normal"
        conn.execute(
            f"""INSERT INTO {_MEMORY_ITEM_TABLE} (
                    item_id, patient_id, mode, category, content, source,
                    source_session_id, source_turn_id, version, status, deleted_at,
                    updated_at, created_at, slot_key, observed_at, valid_until,
                    sensitivity, confidence, confirmed_by, evidence_json,
                    supersedes_item_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item_id,
                patient_id,
                normalize_session_mode(mode),
                category,
                content,
                source,
                source_session_id,
                source_turn_id,
                int(version),
                status,
                deleted_at,
                updated_at,
                created_at,
                slot_key,
                observed_at,
                valid_until,
                sensitivity,
                confidence,
                confirmed_by,
                evidence_json,
                supersedes_item_id,
                json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True) if metadata else None,
            ),
        )
        return item_id

    def _replace_user_memory_category(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        category: str,
        values: list[Any],
        *,
        updated_by: str,
    ) -> None:
        rows = [
            self._memory_item_row(row)
            for row in conn.execute(
                f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                    WHERE patient_id=? AND mode=? AND category=? AND status='active'
                    ORDER BY created_at ASC, item_id ASC""",
                (patient_id, normalize_session_mode(WELLBEING), category),
            ).fetchall()
        ]
        by_id = {str(row["item_id"]): row for row in rows}
        by_slot = {
            str(row["slot_key"]): row
            for row in rows
            if str(row.get("slot_key") or "").strip()
        }
        matched: set[str] = set()
        for index, raw in enumerate(values):
            entry = dict(raw) if isinstance(raw, Mapping) else {"text": raw}
            text = _memory_item_text(entry)
            if not text:
                raise ValueError(f"{category} cannot contain blank items")
            requested_id = str(entry.get("item_id") or entry.get("id") or "").strip()
            slot_key = str(entry.get("slot_key") or "").strip() or None
            current = by_id.get(requested_id) if requested_id else None
            if current is None and slot_key:
                current = by_slot.get(slot_key)
            if current is None and index < len(rows):
                candidate = rows[index]
                if str(candidate["item_id"]) not in matched:
                    current = candidate
            if current is not None:
                matched.add(str(current["item_id"]))
                if str(current["content"]) == text:
                    continue
                now = _now()
                old_version = int(current["version"] or 0) + 1
                conn.execute(
                    f"""UPDATE {_MEMORY_ITEM_TABLE}
                        SET status='superseded', version=?, updated_at=?
                        WHERE patient_id=? AND item_id=? AND status='active'""",
                    (old_version, now, patient_id, current["item_id"]),
                )
                self._enqueue_mirror_op(
                    conn,
                    patient_id,
                    str(current["item_id"]),
                    old_version,
                    "invalidate",
                    str(current.get("source_turn_id") or "") or None,
                )
                source = "user_correction"
                supersedes_item_id = str(current["item_id"])
                new_item_id = None
            else:
                source = "user_direct"
                supersedes_item_id = None
                new_item_id = requested_id or None
                if new_item_id and new_item_id in by_id:
                    new_item_id = None
            metadata = _json_object(entry.get("metadata_json") or entry.get("metadata"))
            if category == "comfort_strategies" and "effective" in entry:
                metadata["effective"] = entry["effective"]
            evidence_json = self._item_evidence_json(entry)
            source_turn_id = str(entry.get("source_turn_id") or "").strip() or None
            if not source_turn_id:
                try:
                    evidence = json.loads(evidence_json or "[]")
                except json.JSONDecodeError:
                    evidence = []
                if evidence:
                    source_turn_id = str(evidence[0].get("turn_id") or "").strip() or None
            inherited_slot = str(current.get("slot_key") or "").strip() if current else None
            try:
                confidence = float(entry["confidence"]) if entry.get("confidence") is not None else None
            except (TypeError, ValueError):
                confidence = None
            self._insert_memory_item(
                conn,
                patient_id,
                category,
                text,
                item_id=new_item_id,
                source=source,
                version=old_version if current is not None else 1,
                source_session_id=str(entry.get("source_session_id") or "").strip() or None,
                source_turn_id=source_turn_id,
                slot_key=slot_key or inherited_slot,
                observed_at=str(entry.get("observed_at") or "").strip() or None,
                valid_until=str(entry.get("valid_until") or "").strip() or None,
                sensitivity=str(entry.get("sensitivity") or "normal"),
                confidence=confidence,
                confirmed_by="user" if updated_by == "user" else str(entry.get("confirmed_by") or "") or None,
                evidence_json=evidence_json,
                supersedes_item_id=supersedes_item_id,
                metadata=metadata or None,
            )
        now = _now()
        for row in rows:
            item_id = str(row["item_id"])
            if item_id in matched:
                continue
            version = int(row["version"] or 0) + 1
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status='deleted', version=?, deleted_at=?, updated_at=?
                    WHERE patient_id=? AND item_id=? AND status='active'""",
                (version, now, now, patient_id, item_id),
            )
            self._enqueue_mirror_op(
                conn,
                patient_id,
                item_id,
                version,
                "delete",
                str(row.get("source_turn_id") or "") or None,
            )
        if category == "events":
            self._suppress_redundant_event_items(conn, patient_id, WELLBEING)

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
        try:
            patient_row = conn.execute(
                "SELECT 1 FROM patients WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            patient_row = None
        data = _normalise_snapshot(
            patient_id,
            raw,
            self._patient_profile(conn, patient_id) if patient_row else None,
        )
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

    @staticmethod
    def _mmse_entry_belongs_to_session(
        entry: Any,
        *,
        session_id: str,
        created_at: str,
        ended_at: str,
        total_score: Any,
    ) -> bool:
        if not isinstance(entry, Mapping):
            return False
        if str(entry.get("session_id") or "").strip():
            return str(entry.get("session_id") or "").strip() == session_id
        if total_score is None or str(entry.get("recorded_at") or "").strip() == "":
            return False
        try:
            if int(entry.get("score")) != int(total_score):
                return False
        except (TypeError, ValueError):
            return False
        recorded_at = str(entry.get("recorded_at") or "").strip()
        return bool(created_at and ended_at and created_at <= recorded_at <= ended_at)

    def _public_snapshot(
        self,
        conn: sqlite3.Connection,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = self._snapshot_from_items(conn, snapshot)
        result.pop("_narrative_manual", None)
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

    def delete_session_data(self, session_id: str) -> Optional[dict[str, Any]]:
        """Remove one ended session's memory evidence and rebuild its patient projection."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                session = conn.execute(
                    "SELECT session_id, patient_id, created_at, ended_at, total_mmse_score "
                    "FROM sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
            if not session:
                return None
            if not session["ended_at"]:
                raise ValueError("SESSION_ACTIVE")
            patient_id = str(session["patient_id"] or "").strip()
            if not patient_id:
                return {"session_id": session_id, "patient_id": None, "deleted_items": 0}

            turn_refs = {
                str(row["id"])
                for row in conn.execute(
                    f"SELECT id FROM {_TURN_TABLE} WHERE patient_id=? AND session_id=?",
                    (patient_id, session_id),
                ).fetchall()
            }
            turn_refs.update(
                str(row["turn_id"] or "")
                for row in conn.execute(
                    f"SELECT turn_id FROM {_TURN_TABLE} WHERE patient_id=? AND session_id=?",
                    (patient_id, session_id),
                ).fetchall()
                if str(row["turn_id"] or "").strip()
            )
            turn_sessions = {
                str(row["turn_id"]): str(row["session_id"] or "").strip()
                for row in conn.execute(
                    f"SELECT turn_id, session_id FROM {_TURN_TABLE} WHERE patient_id=?",
                    (patient_id,),
                ).fetchall()
                if str(row["turn_id"] or "").strip()
            }
            try:
                turn_refs.update(
                    str(row["turn_id"] or "")
                    for row in conn.execute(
                        "SELECT turn_id FROM emotion_trajectory WHERE patient_id=? AND session_id=?",
                        (patient_id, session_id),
                    ).fetchall()
                    if str(row["turn_id"] or "").strip()
                )
            except sqlite3.OperationalError:
                pass

            deleted_summary = conn.execute(
                f"SELECT session_summary FROM {_SESSION_REFLECTION_TABLE} "
                "WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            ).fetchone()
            deleted_markers = (
                [str(deleted_summary[0] or "").strip()]
                if deleted_summary and str(deleted_summary[0] or "").strip()
                else []
            )
            all_summaries = conn.execute(
                f"""SELECT session_id, session_summary FROM {_SESSION_REFLECTION_TABLE}
                    WHERE patient_id=? AND status='succeeded'
                      AND COALESCE(session_summary, '')<>''
                    ORDER BY COALESCE(completed_at, updated_at, created_at) ASC""",
                (patient_id,),
            ).fetchall()
            rows = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchall()
            deleted_items = 0
            now = _now()
            for row in rows:
                item = dict(row)
                try:
                    evidence = json.loads(item.get("evidence_json") or "[]")
                except json.JSONDecodeError:
                    evidence = []
                evidence = evidence if isinstance(evidence, list) else []
                def evidence_turn(entry: Any) -> str:
                    if isinstance(entry, Mapping):
                        return str(entry.get("turn_id") or entry.get("source_turn_id") or "").strip()
                    return str(entry or "").strip()

                remaining_evidence = [
                    entry for entry in evidence
                    if evidence_turn(entry) not in turn_refs
                ]
                source_session = str(item.get("source_session_id") or "").strip()
                source_turn = str(item.get("source_turn_id") or "").strip()
                affected = source_session == session_id or source_turn in turn_refs or (
                    len(remaining_evidence) != len(evidence)
                )
                if not affected:
                    continue
                content = str(item.get("content") or "").strip()
                user_managed = _event_is_user_managed(item)
                if remaining_evidence or user_managed:
                    next_source_turn = source_turn
                    if next_source_turn in turn_refs:
                        next_source_turn = ""
                    surviving_session = source_session if source_session != session_id else ""
                    if not surviving_session:
                        for evidence_entry in remaining_evidence:
                            evidence_id = evidence_turn(evidence_entry)
                            surviving_session = turn_sessions.get(evidence_id, "")
                            if surviving_session:
                                break
                    evidence_json = (
                        json.dumps(remaining_evidence, ensure_ascii=False)
                        if remaining_evidence else None
                    )
                    conn.execute(
                        f"""UPDATE {_MEMORY_ITEM_TABLE}
                            SET source_session_id=?, source_turn_id=?, evidence_json=?,
                                version=?, updated_at=?
                            WHERE patient_id=? AND item_id=?""",
                        (
                            surviving_session or None,
                            next_source_turn or None,
                            evidence_json,
                            int(item.get("version") or 0) + 1,
                            now,
                            patient_id,
                            item["item_id"],
                        ),
                    )
                    if source_turn in turn_refs:
                        self._enqueue_mirror_op(
                            conn,
                            patient_id,
                            str(item["item_id"]),
                            int(item.get("version") or 0) + 1,
                            "invalidate",
                            source_turn,
                        )
                    continue
                if content:
                    deleted_markers.append(content)
                next_version = int(item.get("version") or 0) + 1
                conn.execute(
                    f"""UPDATE {_MEMORY_ITEM_TABLE}
                        SET status='deleted', deleted_at=?, version=?, updated_at=?
                        WHERE patient_id=? AND item_id=?""",
                    (now, next_version, now, patient_id, item["item_id"]),
                )
                conn.execute(
                    f"""INSERT OR IGNORE INTO {_MEMORY_DELETION_TABLE}
                        (item_id, patient_id, deletion_token, deleted_by, deleted_at, item_version)
                        VALUES (?, ?, ?, 'session_delete', ?, ?)""",
                    (
                        item["item_id"],
                        patient_id,
                        f"session:{session_id}:{item['item_id']}:{next_version}",
                        now,
                        next_version,
                    ),
                )
                self._enqueue_mirror_op(
                    conn,
                    patient_id,
                    str(item["item_id"]),
                    next_version,
                    "delete",
                    source_turn or None,
                )
                deleted_items += 1

            conn.execute(
                f"DELETE FROM {_TURN_TABLE} WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            )
            conn.execute(
                "DELETE FROM emotion_trajectory WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            )
            conn.execute(
                f"DELETE FROM {_SESSION_REFLECTION_TABLE} WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            )

            snapshot = self._load_snapshot(conn, patient_id)
            emotion_history: list[dict[str, Any]] = []
            try:
                trajectory_rows = conn.execute(
                    """SELECT session_id, turn_id, analysis_status, timestamp,
                              joy, sadness, anger, fear, anxiety, calm, confusion,
                              valence, arousal, dominant_emotion, audio_path, source
                       FROM emotion_trajectory WHERE patient_id=?
                       ORDER BY timestamp ASC, id ASC""",
                    (patient_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                trajectory_rows = []
            for row in trajectory_rows[-_MAX_EMOTION_HISTORY:]:
                scores = {
                    name: float(row[name] or 0.0)
                    for name in _EMOTIONS
                }
                emotion_history.append(
                    {
                        "scores": scores,
                        "dominant": row["dominant_emotion"],
                        "valence": row["valence"],
                        "arousal": row["arousal"],
                        "recorded_at": row["timestamp"],
                        "session_id": row["session_id"],
                        "turn_id": row["turn_id"],
                        "audio_path": row["audio_path"],
                        "source": row["source"],
                        "analysis_status": row["analysis_status"],
                    }
                )
            snapshot["emotion"] = {
                "latest": emotion_history[-1] if emotion_history else None,
                "history": emotion_history,
            }
            if emotion_history:
                snapshot["last_session_id"] = emotion_history[-1].get("session_id")
                snapshot["last_turn_at"] = emotion_history[-1].get("recorded_at")
            else:
                snapshot.pop("last_session_id", None)
                snapshot.pop("last_turn_at", None)

            narrative = str(snapshot.get("narrative") or "").strip()
            all_summary_text = [
                str(row["session_summary"]).strip()
                for row in all_summaries
                if str(row["session_summary"] or "").strip()
            ]
            surviving_summaries = [
                str(row["session_summary"]).strip()
                for row in all_summaries
                if str(row["session_id"] or "") != session_id
                and str(row["session_summary"] or "").strip()
            ]
            narrative_markers = [
                marker for marker in deleted_markers
                if len(_event_text_key(marker)) >= 4
            ]
            narrative_key = _event_text_key(narrative)
            narrative_has_deleted_content = bool(
                narrative and any(
                    marker in narrative or _event_text_key(marker) in narrative_key
                    for marker in narrative_markers
                )
            )
            narrative_is_projection = bool(
                snapshot.get("_narrative_manual") is False
                or (
                    narrative
                    and all_summaries
                    and narrative_key == _event_text_key("\n".join(all_summary_text))
                )
            )
            if narrative_is_projection:
                snapshot["narrative"] = "\n".join(surviving_summaries)
            elif narrative_has_deleted_content and snapshot.get("_narrative_manual") is not True:
                fragments = [
                    fragment.strip()
                    for fragment in re.split(r"[\r\n。！？；;]+", narrative)
                    if fragment.strip()
                ]
                snapshot["narrative"] = "\n".join(
                    fragment for fragment in fragments
                    if not any(
                        marker in fragment or _event_text_key(marker) in _event_text_key(fragment)
                        for marker in narrative_markers
                    )
                )
            elif not narrative and surviving_summaries:
                snapshot["narrative"] = "\n".join(surviving_summaries)
            mmse_history = [
                entry for entry in (snapshot.get("mmse_history") or [])
                if not self._mmse_entry_belongs_to_session(
                    entry,
                    session_id=session_id,
                    created_at=str(session["created_at"] or ""),
                    ended_at=str(session["ended_at"] or ""),
                    total_score=session["total_mmse_score"],
                )
            ]
            snapshot["mmse_history"] = mmse_history[-_MAX_MMSE_HISTORY:]
            snapshot = self._snapshot_from_items(conn, snapshot)
            patient_row = conn.execute(
                "SELECT 1 FROM patients WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
            if patient_row:
                snapshot["profile"] = self._patient_profile(conn, patient_id)
            self._save_snapshot(conn, snapshot, updated_by="session_delete")
            result = {
                "session_id": session_id,
                "patient_id": patient_id,
                "deleted_items": deleted_items,
                "turn_count": len(turn_refs),
            }
        result["memobase_delete_enqueued"] = self._schedule_mirror_ops(patient_id)
        return result

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
        status = "unavailable" if not emotion_scores else str(analysis_status or "final").strip() or "final"
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
                    emotion_scores.get("joy", 0.0), emotion_scores.get("sadness", 0.0),
                    emotion_scores.get("anger", 0.0), emotion_scores.get("fear", 0.0),
                    emotion_scores.get("anxiety", 0.0), emotion_scores.get("calm", 0.0),
                    emotion_scores.get("confusion", 0.0), valence, arousal, dominant,
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
                elif key in {"facts", "preferences", "events", "comfort_strategies"}:
                    if not isinstance(value, list):
                        raise ValueError(f"{key} must be a list")
                    self._replace_user_memory_category(
                        conn,
                        patient_id,
                        key,
                        list(value),
                        updated_by=updated_by,
                    )
                elif key == "mmse_history":
                    if not isinstance(value, list):
                        raise ValueError("mmse_history must be a list")
                    snapshot[key] = _json_copy(value)[-_MAX_MMSE_HISTORY:]
                elif key == "narrative":
                    snapshot[key] = str(value or "")
                    snapshot["_narrative_manual"] = updated_by not in {"system", "session_delete"}
                else:
                    snapshot[key] = _json_copy(value)
            if profile_updates:
                self._update_patient_profile(conn, patient_id, profile_updates)
            snapshot = self._snapshot_from_items(conn, snapshot)
            self._save_snapshot(conn, snapshot, updated_by=updated_by)
            stored = self._load_snapshot(conn, patient_id)
            result = self._public_snapshot(conn, stored)
        self._schedule_mirror_ops(patient_id)
        return result

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
                entry["source"] = "user_direct"
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
        session_id: Optional[str] = None,
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
        if str(session_id or "").strip():
            entry["session_id"] = str(session_id).strip()
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
            row = conn.execute(
                f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                    WHERE patient_id=? AND mode=? AND category='comfort_strategies'
                      AND content=? AND status='active'
                    LIMIT 1""",
                (patient_id, normalize_session_mode(WELLBEING), strategy),
            ).fetchone()
            if row:
                current = self._memory_item_row(row)
                metadata = _json_object(current.get("metadata_json"))
                try:
                    use_count = int(metadata.get("use_count") or 0)
                except (TypeError, ValueError):
                    use_count = 0
                metadata.update({"effective": bool(effective), "use_count": use_count + 1})
                conn.execute(
                    f"""UPDATE {_MEMORY_ITEM_TABLE}
                        SET version=?, metadata_json=?, updated_at=?
                        WHERE patient_id=? AND item_id=? AND status='active'""",
                    (
                        int(current["version"]) + 1,
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        now,
                        patient_id,
                        current["item_id"],
                    ),
                )
                item_id = str(current["item_id"])
            else:
                item_id = self._insert_memory_item(
                    conn,
                    patient_id,
                    "comfort_strategies",
                    strategy,
                    source="system_inferred",
                    updated_at=now,
                    created_at=now,
                    metadata={"effective": bool(effective), "use_count": 1},
                )
            snapshot = self._snapshot_from_items(conn, self._load_snapshot(conn, patient_id))
            self._save_snapshot(conn, snapshot, updated_by="system")
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            return self._snapshot_item_projection(self._memory_item_row(row))

    def list_memory_items(
        self,
        patient_id: str,
        *,
        include_deleted: bool = False,
        mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            query = f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=?"
            params: list[Any] = [patient_id]
            if not include_deleted:
                query += " AND status IN ('active', 'candidate', 'blocked')"
            if mode:
                query += " AND mode=?"
                params.append(normalize_session_mode(mode))
            query += " ORDER BY updated_at DESC, created_at DESC"
            return [
                self._memory_item_row(row)
                for row in conn.execute(query, tuple(params)).fetchall()
            ]

    def get_memory_item(self, patient_id: str, item_id: str) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        item_id = str(item_id or "").strip()
        if not item_id:
            raise ValueError("item_id is required")
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            if str(row["status"] or "") == "superseded":
                replacement = conn.execute(
                    f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                        WHERE patient_id=? AND supersedes_item_id=? AND status='active'
                        ORDER BY version DESC LIMIT 1""",
                    (patient_id, item_id),
                ).fetchone()
                if replacement:
                    return self._memory_item_row(replacement)
            return self._memory_item_row(row)

    def _set_memory_item_status(
        self,
        patient_id: str,
        item_id: str,
        expected_version: int,
        *,
        from_status: str,
        to_status: str,
        actor: str,
    ) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        item_id = str(item_id or "").strip()
        if not item_id:
            raise ValueError("item_id is required")
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            current = self._memory_item_row(row)
            if current["status"] != from_status:
                raise KeyError(f"MEMORY_ITEM_NOT_{from_status.upper()}")
            if int(current["version"]) != int(expected_version):
                raise MemoryRevisionConflict(int(current["version"]))
            version = int(current["version"]) + 1
            now = _now()
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status=?, version=?, confirmed_by=?, updated_at=?
                    WHERE patient_id=? AND item_id=?""",
                (to_status, version, actor, now, patient_id, item_id),
            )
            if from_status == "active" and to_status != "active":
                self._enqueue_mirror_op(
                    conn,
                    patient_id,
                    item_id,
                    version,
                    "invalidate",
                    str(current.get("source_turn_id") or "") or None,
                )
            snapshot = self._snapshot_from_items(conn, snapshot)
            self._save_snapshot(conn, snapshot, updated_by=actor)
            updated = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            result = self._memory_item_row(updated)
        self._schedule_mirror_ops(patient_id)
        return result

    def confirm_memory_item(
        self,
        patient_id: str,
        item_id: str,
        expected_version: int,
        *,
        confirmed_by: str = "user",
    ) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        item_id = str(item_id or "").strip()
        if not item_id:
            raise ValueError("item_id is required")
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            snapshot = self._load_snapshot(conn, patient_id)
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            current = self._memory_item_row(row)
            if current["status"] != "candidate":
                raise KeyError("MEMORY_ITEM_NOT_CANDIDATE")
            if int(current["version"]) != int(expected_version):
                raise MemoryRevisionConflict(int(current["version"]))
            metadata = _json_object(current.get("metadata_json"))
            operation = str(metadata.get("suggested_operation") or "").upper()
            previous: Optional[dict[str, Any]] = None
            if operation == "SUPERSEDE":
                target_id = str(current.get("supersedes_item_id") or "").strip()
                candidates = []
                if target_id:
                    target = conn.execute(
                        f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                            WHERE patient_id=? AND item_id=? AND status='active'""",
                        (patient_id, target_id),
                    ).fetchone()
                    if target:
                        candidates = [target]
                elif str(current.get("slot_key") or "").strip():
                    candidates = conn.execute(
                        f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                            WHERE patient_id=? AND mode=? AND category=? AND slot_key=?
                              AND status='active'""",
                        (
                            patient_id,
                            current["mode"],
                            current["category"],
                            current["slot_key"],
                        ),
                    ).fetchall()
                if len(candidates) != 1:
                    raise MemoryRevisionConflict(int(current["version"]))
                previous = self._memory_item_row(candidates[0])
            now = _now()
            if previous:
                previous_version = int(previous["version"]) + 1
                conn.execute(
                    f"""UPDATE {_MEMORY_ITEM_TABLE}
                        SET status='superseded', version=?, updated_at=?
                        WHERE patient_id=? AND item_id=? AND status='active'""",
                    (previous_version, now, patient_id, previous["item_id"]),
                )
                self._enqueue_mirror_op(
                    conn, patient_id, previous["item_id"], previous_version, "invalidate",
                    str(previous.get("source_turn_id") or "") or None,
                )
                conn.execute(
                    f"UPDATE {_MEMORY_ITEM_TABLE} SET supersedes_item_id=? WHERE item_id=?",
                    (previous["item_id"], item_id),
                )
            if confirmed_by == "clinician":
                source = "clinician_confirmed"
            elif confirmed_by.startswith(("qwen:", "memory:")):
                source = "llm_verified"
            else:
                source = "user"
            metadata["confirmed_from_source"] = current["source"]
            version = int(current["version"]) + 1
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status='active', source=?, version=?, confirmed_by=?, metadata_json=?, updated_at=?
                    WHERE patient_id=? AND item_id=?""",
                (
                    source,
                    version,
                    confirmed_by,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    now,
                    patient_id,
                    item_id,
                ),
            )
            snapshot = self._snapshot_from_items(conn, snapshot)
            self._save_snapshot(conn, snapshot, updated_by=confirmed_by)
            updated = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            result = self._memory_item_row(updated)
        self._schedule_mirror_ops(patient_id)
        return result

    def reject_memory_item(
        self,
        patient_id: str,
        item_id: str,
        expected_version: int,
        *,
        rejected_by: str = "user",
    ) -> dict[str, Any]:
        return self._set_memory_item_status(
            patient_id,
            item_id,
            expected_version,
            from_status="candidate",
            to_status="blocked",
            actor=rejected_by,
        )

    def stop_memory_item(
        self,
        patient_id: str,
        item_id: str,
        expected_version: int,
        *,
        stopped_by: str = "user",
    ) -> dict[str, Any]:
        """Keep the record for audit while excluding it from companionship."""
        return self._set_memory_item_status(
            patient_id,
            item_id,
            expected_version,
            from_status="active",
            to_status="blocked",
            actor=stopped_by,
        )

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
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                raise KeyError("MEMORY_ITEM_NOT_FOUND")
            current = self._memory_item_row(row)
            if current["status"] == "deleted":
                raise KeyError("MEMORY_ITEM_DELETED")
            if current["status"] == "superseded":
                replacement = conn.execute(
                    f"""SELECT version FROM {_MEMORY_ITEM_TABLE}
                        WHERE patient_id=? AND supersedes_item_id=? AND status='active'
                        ORDER BY version DESC LIMIT 1""",
                    (patient_id, item_id),
                ).fetchone()
                if replacement:
                    raise MemoryRevisionConflict(int(replacement["version"]))
            if current["status"] != "active":
                raise KeyError("MEMORY_ITEM_NOT_ACTIVE")
            if int(current["version"]) != int(expected_version):
                raise MemoryRevisionConflict(int(current["version"]))
            now = _now()
            old_version = int(current["version"]) + 1
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET version=?, status='superseded', updated_at=?
                    WHERE patient_id=? AND item_id=?""",
                (
                    old_version,
                    now,
                    patient_id,
                    item_id,
                ),
            )
            new_item_id = self._insert_memory_item(
                conn,
                patient_id,
                str(current["category"]),
                content,
                mode=str(current["mode"] or WELLBEING),
                source="user_correction",
                version=old_version,
                source_session_id=current.get("source_session_id"),
                source_turn_id=current.get("source_turn_id"),
                slot_key=current.get("slot_key"),
                observed_at=current.get("observed_at"),
                valid_until=current.get("valid_until"),
                sensitivity=current.get("sensitivity") or "normal",
                confidence=current.get("confidence"),
                confirmed_by="user" if updated_by == "user" else updated_by,
                evidence_json=current.get("evidence_json"),
                supersedes_item_id=item_id,
                metadata=_json_object(current.get("metadata_json")) or None,
            )
            self._enqueue_mirror_op(
                conn,
                patient_id,
                item_id,
                old_version,
                "invalidate",
                str(current.get("source_turn_id") or "") or None,
            )
            snapshot = self._snapshot_from_items(conn, snapshot)
            self._save_snapshot(conn, snapshot, updated_by=updated_by)
            updated = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                (patient_id, new_item_id),
            ).fetchone()
            result = self._memory_item_row(updated)
        self._schedule_mirror_ops(patient_id)
        return result

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
            elif str(current.get("status") or "").lower() != "active":
                result = self._memory_item_row(current)
                result["idempotent_replay"] = False
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
                self._enqueue_mirror_op(
                    conn,
                    patient_id,
                    item_id,
                    next_version,
                    "delete",
                    cleanup_source_turn_id or None,
                )
                snapshot = self._snapshot_from_items(conn, snapshot)
                self._save_snapshot(conn, snapshot, updated_by=deleted_by)
                deleted = conn.execute(
                    f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                    (patient_id, item_id),
                ).fetchone()
                result = self._memory_item_row(deleted)
                result["idempotent_replay"] = False
        result["memobase_delete_enqueued"] = self._schedule_mirror_ops(patient_id)
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
        analysis_status = "unavailable" if not emotion_scores else str(analysis_status or "final").strip() or "final"
        stored_emotion_scores = {
            name: float(emotion_scores.get(name, 0.0))
            for name in _EMOTIONS
        }
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
                     json.dumps(stored_emotion_scores, ensure_ascii=False, sort_keys=True),
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
                     stored_emotion_scores["joy"],
                     stored_emotion_scores["sadness"],
                     stored_emotion_scores["anger"],
                     stored_emotion_scores["fear"],
                     stored_emotion_scores["anxiety"],
                     stored_emotion_scores["calm"],
                     stored_emotion_scores["confusion"],
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
        _include_degraded: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], bool]:
        """Return structured semantic event gists from Memobase."""
        patient_id = self._patient_id(patient_id)
        current_text = str(current_text or "").strip()
        if not current_text:
            return ([], False) if _include_degraded else []
        if self._memobase is None and not self._memobase_auto:
            self._log(f"[Memobase] gist_search degraded=unavailable patient={patient_id}")
            return ([], True) if _include_degraded else []
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
        with self._lock, self._connection() as conn:
            snapshot = conn.execute(
                f"SELECT revision FROM {_SNAPSHOT_TABLE} WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
        revision = int(snapshot["revision"] or 0) if snapshot else 0
        cache_key = (patient_id, revision, current_text, tuple(sorted(params.items())))
        try:
            gists, cache_source = self._retrieval_cache.get(
                cache_key,
                lambda: self._retrieve_event_gists(patient_id, current_text, params),
            )
        except (RetrievalBusyError, FutureTimeoutError, RetrievalUnavailableError) as exc:
            self._log(
                f"[Memobase] gist_search degraded=unavailable patient={patient_id} "
                f"reason={type(exc).__name__}"
            )
            return ([], True) if _include_degraded else []
        except Exception as exc:
            self._mark_memobase_failed(exc)
            self._log(
                f"[Memobase] gist_search degraded=error patient={patient_id} "
                f"error={type(exc).__name__}"
            )
            return ([], True) if _include_degraded else []
        with self._lock, self._connection() as conn:
            deleted_turn_ids = self._deleted_turn_ids(conn, patient_id)
            deleted_texts = {
                text
                for values in self._deleted_texts_by_category(conn, patient_id).values()
                for text in values
                if text
            }
            gists = [
                gist for gist in gists
                if self._gist_has_valid_local_provenance(conn, patient_id, gist)
            ]
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
            f"cache={cache_source} "
            f"event_gist_ids={gist_ids or '-'} source_turn_ids={source_turn_ids or '-'} "
            f"degraded={'none' if gists else 'empty'}"
        )
        return (gists, False) if _include_degraded else gists

    def _retrieve_event_gists(
        self, patient_id: str, query: str, params: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        client = self._retrieval_client()
        user = self._get_or_create_memobase_user(client, patient_id)
        raw_gists = self._search_event_gist(user, query, params)
        return self._normalise_event_gists(
            patient_id, raw_gists, max_items=int(params["topk"])
        )

    def _retrieval_client(self) -> Any:
        client = self._ensure_memobase()
        if client is None:
            raise RetrievalUnavailableError("semantic retrieval is unavailable")
        if getattr(client, "_client", None) is not None:
            client = copy.copy(client)
            client._client = RetrievalHttpClient(
                client.client, self._retrieval_http_timeout_s
            )
        return client

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
                    "remote_patient_id": str(cls._field(raw, "patient_id") or ""),
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

    def _gist_has_valid_local_provenance(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        gist: Mapping[str, Any],
    ) -> bool:
        source_turn_id = str(gist.get("source_turn_id") or "").strip()
        if not source_turn_id:
            return False
        remote_patient_id = str(gist.get("remote_patient_id") or "").strip()
        if remote_patient_id and remote_patient_id not in {
            patient_id,
            self._memobase_user_id(patient_id),
        }:
            return False
        turn = conn.execute(
            f"""SELECT id, turn_id, turn_state FROM {_TURN_TABLE}
                WHERE patient_id=? AND (CAST(id AS TEXT)=? OR turn_id=?)
                LIMIT 1""",
            (patient_id, source_turn_id, source_turn_id),
        ).fetchone()
        if not turn or str(turn["turn_state"] or "").upper() in {
            "CANCELLED", "FAILED", "RESPONSE_CANCELLED"
        }:
            return False
        source_ids = {
            source_turn_id,
            str(turn["id"]),
            str(turn["turn_id"] or ""),
        }
        if source_ids & self._deleted_turn_ids(conn, patient_id):
            return False
        referenced_items = []
        for row in conn.execute(
            f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=?",
            (patient_id,),
        ).fetchall():
            item = self._memory_item_row(row)
            evidence_ids = {str(item.get("source_turn_id") or "").strip()}
            try:
                evidence = json.loads(item.get("evidence_json") or "[]")
            except json.JSONDecodeError:
                evidence = []
            for evidence_item in evidence if isinstance(evidence, list) else []:
                if isinstance(evidence_item, Mapping):
                    evidence_ids.add(str(evidence_item.get("turn_id") or "").strip())
            if source_ids & evidence_ids:
                referenced_items.append(item)
        return not referenced_items or any(
            _memory_item_is_current(item) for item in referenced_items
        )

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
            # The gist id must survive into the rendered text: downstream
            # telemetry recovers which memories reached the prompt by parsing
            # this block, and without it every semantic hit reports zero used
            # items even though retrieval succeeded.
            gist_id = str(gist.get("event_gist_id") or "").strip() or "unknown"
            block = (
                f"{index}. 摘要：{content}\n"
                f"   创建时间：{created_at or 'unknown'}；相似度：{score}"
                f"；event_gist_id={gist_id}"
            )
            cost = _token_estimate(block)
            if used + cost > token_budget:
                continue
            lines.append(block)
            used += cost
            added += 1
        if not added:
            lines.append("无通过阈值的跨会话长期事件。")
        return "\n".join(lines)

    def _render_local_active_fallback(self, patient_id: str) -> str:
        """Dump the most recent active items, ignoring the current query.

        This is a degraded path used only when semantic search is unavailable.
        It performs no relevance matching, so its output must never be reported
        as a semantic retrieval hit.
        """
        with self._lock, self._connection() as conn:
            items_by_category = self._active_items_by_category(
                conn, patient_id, mode=WELLBEING
            )
        labels = {"facts": "长期事实", "preferences": "偏好", "events": "近期事件"}
        items = [
            item
            for category in ("events", "facts", "preferences")
            for item in items_by_category.get(category, [])
        ]
        items.sort(
            key=lambda item: (str(item.get("updated_at") or ""), str(item.get("item_id") or "")),
            reverse=True,
        )
        if not items:
            return ""
        lines = [LOCAL_ACTIVE_MEMORY_MARKER]
        for index, item in enumerate(items[:_LOCAL_FALLBACK_ITEM_LIMIT], 1):
            lines.append(
                f"{index}. 来源：SQLite active memory_items；"
                f"item_id={str(item.get('item_id') or 'unknown')}；"
                f"{labels.get(str(item.get('category') or ''), '记忆')}："
                f"{_clip(item.get('content'), 220)}"
            )
        return "\n".join(lines)

    def get_relevant_context(self, patient_id: str, current_text: str) -> str:
        """Compatibility wrapper for semantic event gist evidence."""
        return self.get_relevant_evidence(patient_id, current_text)

    def get_authoritative_card(self, patient_id: str) -> str:
        """Build the active SQLite memory card without raw conversation turns."""
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            snapshot = self._load_snapshot(conn, patient_id)
            items_by_category = self._active_items_by_category(
                conn,
                patient_id,
                mode=WELLBEING,
            )
        lines = ["【患者权威记忆卡】"]
        used = _token_estimate(lines[0])

        def append_section(
            title: str,
            values: list[str],
            *,
            item_limit: int,
            section_budget: int,
        ) -> None:
            nonlocal used
            available = min(section_budget, _AUTHORITATIVE_CARD_TOKEN_BUDGET - used)
            available -= _token_estimate(title)
            if available <= 0:
                return
            selected = _budgeted_values(values, available, item_limit)
            while selected and used + _token_estimate(title + "；".join(selected)) > _AUTHORITATIVE_CARD_TOKEN_BUDGET:
                selected.pop()
            if selected:
                line = title + "；".join(selected)
                lines.append(line)
                used += _token_estimate(line)

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
                append_section("【患者基本信息】", bits, item_limit=160, section_budget=160)
        for title, key in (("长期事实", "facts"), ("偏好与兴趣", "preferences"), ("近期事件", "events")):
            values = [
                (
                    item["content"],
                    str(item.get("source") or "").lower()
                    in {"manual", "user", "user_direct", "user_correction", "clinician_confirmed"},
                )
                for item in items_by_category.get(key, [])
            ]
            if values:
                append_section(
                    f"【{title}】",
                    [("人工修订：" if manual else "") + text for text, manual in values],
                    item_limit=240,
                    section_budget={"facts": 360, "preferences": 220, "events": 300}[key],
                )
        narrative = str(snapshot.get("narrative") or "").strip()
        if narrative:
            append_section("【权威摘要】", [narrative], item_limit=1000, section_budget=180)
        effective_strategies = []
        avoid_strategies = []
        for item in items_by_category.get("comfort_strategies", []):
            effective = _json_object(item.get("metadata_json")).get("effective")
            if effective is True:
                effective_strategies.append(item["content"])
            elif effective is False:
                avoid_strategies.append(item["content"])
        if effective_strategies:
            append_section("【有效安抚策略】", effective_strategies, item_limit=180, section_budget=180)
        if avoid_strategies:
            append_section("【应避免方式】", avoid_strategies, item_limit=180, section_budget=180)
        return "" if len(lines) == 1 else "\n".join(lines)

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
        gists, degraded = self.get_relevant_event_gists(
            patient_id,
            text,
            topk=_EVENT_GIST_TOPK,
            _include_degraded=True,
        )
        if gists:
            return self._render_event_gists(gists)
        if degraded and _local_fallback_enabled():
            return self._render_local_active_fallback(patient_id)
        return ""

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
        try:
            client = self._retrieval_client()
            self._get_or_create_memobase_user(client, patient_id)
            return True
        except RetrievalUnavailableError:
            return False
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
        if emotion and str(emotion.get("analysis_status") or "final") == "final" and str(emotion.get("dominant") or "unknown") not in {"unknown", "unavailable"}:
            scores = emotion.get("scores") if isinstance(emotion, Mapping) else {}
            scores = scores if isinstance(scores, Mapping) else {}
            top = sorted(
                ((name, float(scores.get(name, 0.0) or 0.0)) for name in _EMOTIONS),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
            score_text = "，".join(f"{name}={score:.2f}" for name, score in top if score > 0)
            emotion_line = f"【近期情绪】{emotion.get('dominant')}"
            if score_text:
                emotion_line += f"（{score_text}）"
            emotion_history = (snapshot.get("emotion") or {}).get("history") or []
            trajectory = [
                item.get("dominant")
                for item in emotion_history[-5:]
                if isinstance(item, Mapping) and item.get("dominant")
            ]
            if len(trajectory) > 1:
                emotion_line += f"；轨迹：{' → '.join(trajectory)}"
            trend = _emotion_trend(emotion_history)
            if trend:
                emotion_line += f"；趋势：{trend}"
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

    def _enqueue_mirror_op(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        item_id: str,
        item_version: int,
        op_type: str,
        source_turn_id: Optional[str],
    ) -> None:
        now = _now()
        op_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"lmca-mirror:{patient_id}:{item_id}:{int(item_version)}:{op_type}",
        ).hex
        item = conn.execute(
            f"SELECT content, source_turn_id FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
            (patient_id, item_id),
        ).fetchone()
        payload = json.dumps(
            {
                "item_id": item_id,
                "item_version": int(item_version),
                "op_type": op_type,
                "source_turn_id": source_turn_id,
                "content": str(item["content"] or "") if item else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        conn.execute(
            f"""INSERT OR IGNORE INTO {_MIRROR_OP_TABLE} (
                    op_id, patient_id, item_id, item_version, op_type,
                    source_turn_id, status, attempt_count, next_retry_at,
                    last_error, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, NULL, ?, ?, ?)""",
            (
                op_id,
                patient_id,
                item_id,
                int(item_version),
                op_type,
                source_turn_id,
                payload,
                now,
                now,
            ),
        )

    def _schedule_mirror_ops(self, patient_id: str) -> bool:
        if self._memobase is None and not self._memobase_auto:
            return False
        return self._submit_memobase(self._process_mirror_ops_impl, patient_id) is not None

    def _process_mirror_ops_impl(self, patient_id: str) -> bool:
        now = datetime.now()
        now_text = now.isoformat()
        lease_token = uuid.uuid4().hex
        lease_until = (now + timedelta(seconds=_REFLECTION_LEASE_SECONDS)).isoformat()
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM {_MIRROR_OP_TABLE}
                    WHERE patient_id=? AND (
                        (status IN ('pending','failed') AND (next_retry_at IS NULL OR next_retry_at<=?))
                        OR (status='running' AND lease_until<=?)
                    )
                    ORDER BY created_at ASC LIMIT 20""",
                (patient_id, now_text, now_text),
            ).fetchall()
            claimed_rows = []
            for row in rows:
                token = uuid.uuid4().hex
                until = (now + timedelta(seconds=_REFLECTION_LEASE_SECONDS)).isoformat()
                conn.execute(
                    f"""UPDATE {_MIRROR_OP_TABLE}
                        SET status='running', lease_until=?, lease_token=?, updated_at=?,
                            error_code=CASE WHEN status='running' THEN 'lease_lost' ELSE error_code END
                        WHERE op_id=? AND (
                            (status IN ('pending','failed') AND (next_retry_at IS NULL OR next_retry_at<=?))
                            OR (status='running' AND lease_until<=?)
                        )""",
                    (until, token, now_text, row["op_id"], now_text, now_text),
                )
                if conn.execute("SELECT changes() AS n").fetchone()["n"]:
                    claimed_rows.append({**dict(row), "lease_token": token, "lease_until": until})
            rows = claimed_rows
        if not rows:
            return True
        client = self._ensure_memobase()
        if client is None:
            return False
        all_succeeded = True
        for op in rows:
            try:
                payload = json.loads(op.get("payload_json") or "{}") if isinstance(op, dict) else {}
                content = str(payload.get("content") or "")
                with self._lock, self._connection() as conn:
                    current = conn.execute(
                        f"SELECT version, status FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                        (patient_id, str(payload.get("item_id") or op["item_id"])),
                    ).fetchone()
                    if current and int(current["version"] or 0) > int(payload.get("item_version") or op["item_version"]):
                        conn.execute(
                            f"""UPDATE {_MIRROR_OP_TABLE}
                                SET status='succeeded', error_code='stale_version',
                                    last_error=NULL, lease_until=NULL, lease_token=NULL, updated_at=?
                                WHERE op_id=? AND status='running' AND lease_token=?""",
                            (_now(), op["op_id"], op["lease_token"]),
                        )
                        continue
                    if (
                        str(op.get("op_type") or payload.get("op_type") or "") == "invalidate"
                        and current
                        and str(current["status"] or "") == "active"
                    ):
                        conn.execute(
                            f"""UPDATE {_MIRROR_OP_TABLE}
                                SET status='succeeded', error_code='shared_active_evidence',
                                    last_error=NULL, lease_until=NULL, lease_token=NULL, updated_at=?
                                WHERE op_id=? AND status='running' AND lease_token=?""",
                            (_now(), op["op_id"], op["lease_token"]),
                        )
                        continue
                ok = self._delete_memobase_memory_item_impl(
                    patient_id,
                    content,
                    str(op["source_turn_id"] or ""),
                )
                if not ok:
                    raise RuntimeError("remote invalidation did not complete")
                with self._lock, self._connection() as conn:
                    changed = conn.execute(
                        f"""UPDATE {_MIRROR_OP_TABLE}
                            SET status='succeeded', updated_at=?, last_error=NULL,
                                error_code=NULL, lease_until=NULL, lease_token=NULL
                            WHERE op_id=? AND status='running' AND lease_token=? AND lease_until>?""",
                        (_now(), op["op_id"], op["lease_token"], _now()),
                    ).rowcount
                if not changed:
                    all_succeeded = False
                else:
                    self._emit_audit(patient_id, "mirror_succeeded", "success")
            except Exception as exc:
                all_succeeded = False
                attempts = int(op["attempt_count"] or 0) + 1
                failed = attempts >= _REFLECTION_MAX_ATTEMPTS
                retry_at = None if failed else (
                    datetime.now() + timedelta(
                        seconds=min(300, 10 * (2 ** (attempts - 1)))
                    )
                ).isoformat()
                with self._lock, self._connection() as conn:
                    conn.execute(
                        f"""UPDATE {_MIRROR_OP_TABLE}
                            SET status=?, attempt_count=?, next_retry_at=?,
                                last_error=?, error_code=?, dead_letter_at=CASE WHEN ? THEN ? ELSE NULL END,
                                lease_until=NULL, lease_token=NULL, updated_at=?
                            WHERE op_id=? AND status='running' AND lease_token=?""",
                        (
                            "dead_letter" if failed else "pending",
                            attempts,
                            retry_at,
                            f"{type(exc).__name__}: {exc}"[:500],
                            "remote_error",
                            failed,
                            _now() if failed else None,
                            _now(),
                            op["op_id"],
                            op["lease_token"],
                        ),
                    )
                self._emit_audit(patient_id, "mirror_dead_letter" if failed else "mirror_retry", "remote_error")
        return all_succeeded

    def requeue_mirror_op(
        self,
        op_id: str,
        reason: str = "manual",
        *,
        patient_id: str | None = None,
    ) -> bool:
        patient = self._patient_id(patient_id) if patient_id else None
        with self._lock, self._connection() as conn:
            where = " AND patient_id=?" if patient else ""
            updated = conn.execute(
                f"""UPDATE {_MIRROR_OP_TABLE}
                    SET status='pending', next_retry_at=NULL, lease_until=NULL, lease_token=NULL,
                        last_error=?, error_code=NULL, dead_letter_at=NULL, updated_at=?
                    WHERE op_id=? AND status='dead_letter'{where}""",
                (_clip(reason, 200), _now(), str(op_id).strip(), *([patient] if patient else [])),
            ).rowcount
        self._memory_worker_wakeup.set()
        self._emit_audit(patient, "mirror_requeued", "success" if updated else "missing")
        return bool(updated)

    def get_memory_worker_status(self, patient_id: str | None = None) -> dict[str, Any]:
        """Return operational counters only; never expose patient text or evidence."""
        params: list[Any] = []
        where = ""
        if patient_id:
            where = " WHERE patient_id=?"
            params.append(self._patient_id(patient_id))
        now = datetime.now().isoformat()
        latest_reflection_where = (
            " WHERE r2.patient_id=?" if patient_id else " WHERE 1=1"
        )
        latest_mirror_where = " WHERE m2.patient_id=?" if patient_id else " WHERE 1=1"
        with self._lock, self._connection() as conn:
            reflection = conn.execute(
                f"""SELECT status, COUNT(*) AS count, MIN(created_at) AS oldest
                    FROM {_SESSION_REFLECTION_TABLE}{where} GROUP BY status""", tuple(params)
            ).fetchall()
            mirror = conn.execute(
                f"""SELECT status, COUNT(*) AS count, MIN(created_at) AS oldest
                    FROM {_MIRROR_OP_TABLE}{where} GROUP BY status""", tuple(params)
            ).fetchall()
            reflection_summary = conn.execute(
                f"""SELECT COUNT(*) AS total,
                    SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running,
                    SUM(CASE WHEN status='dead_letter' THEN 1 ELSE 0 END) AS dead_letter,
                    SUM(CASE WHEN status IN ('pending','failed')
                              AND (next_retry_at IS NULL OR next_retry_at<=?) THEN 1 ELSE 0 END) AS due,
                    (SELECT error_code FROM {_SESSION_REFLECTION_TABLE} r2
                     {latest_reflection_where}
                     AND r2.error_code IS NOT NULL ORDER BY r2.updated_at DESC LIMIT 1) AS latest_error_code
                    FROM {_SESSION_REFLECTION_TABLE}{where}""",
                (now, *params, *params) if where else (now,),
            ).fetchone()
            mirror_summary = conn.execute(
                f"""SELECT COUNT(*) AS total,
                    SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running,
                    SUM(CASE WHEN status='dead_letter' THEN 1 ELSE 0 END) AS dead_letter,
                    SUM(CASE WHEN status IN ('pending','failed')
                              AND (next_retry_at IS NULL OR next_retry_at<=?) THEN 1 ELSE 0 END) AS due,
                    (SELECT error_code FROM {_MIRROR_OP_TABLE} m2
                     {latest_mirror_where}
                     AND m2.error_code IS NOT NULL ORDER BY m2.updated_at DESC LIMIT 1) AS latest_error_code
                    FROM {_MIRROR_OP_TABLE}{where}""",
                (now, *params, *params) if where else (now,),
            ).fetchone()
        def compact(rows):
            return {str(row["status"]): {"count": int(row["count"]), "oldest": row["oldest"]} for row in rows}
        def summary(row):
            return {
                "total": int(row["total"] or 0),
                "due": int(row["due"] or 0),
                "running": int(row["running"] or 0),
                "dead_letter": int(row["dead_letter"] or 0),
                "latest_error_code": row["latest_error_code"],
            }
        return {
            "worker_running": bool(self._memory_worker_thread and self._memory_worker_thread.is_alive()),
            "as_of": now,
            "reflections": compact(reflection),
            "mirror_ops": compact(mirror),
            "reflection_summary": summary(reflection_summary),
            "mirror_summary": summary(mirror_summary),
        }

    def check_memobase_consistency(self, patient_id: str) -> dict[str, Any]:
        """Compare local lifecycle state with mirror operation coverage without exposing text."""
        patient_id = self._patient_id(patient_id)
        queued = 0
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                f"""SELECT item_id, version, status, source_turn_id
                    FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=?
                      AND status IN ('superseded','deleted','blocked')""",
                (patient_id,),
            ).fetchall()
            for row in rows:
                covered = conn.execute(
                    f"""SELECT 1 FROM {_MIRROR_OP_TABLE}
                        WHERE patient_id=? AND item_id=? AND item_version=?
                          AND status='succeeded' LIMIT 1""",
                    (patient_id, row["item_id"], int(row["version"] or 0)),
                ).fetchone()
                if not covered:
                    self._enqueue_mirror_op(
                        conn,
                        patient_id,
                        str(row["item_id"]),
                        int(row["version"] or 0),
                        "delete" if str(row["status"]) == "deleted" else "invalidate",
                        str(row["source_turn_id"] or "") or None,
                    )
                    queued += 1
            pending = conn.execute(
                f"""SELECT COUNT(*) FROM {_MIRROR_OP_TABLE}
                    WHERE patient_id=? AND status IN ('pending','running','failed','dead_letter')""",
                (patient_id,),
            ).fetchone()[0]
        remote_checked = False
        remote_profile_count = None
        if self._ensure_memobase() is not None:
            try:
                user = self._get_or_create_memobase_user(self._memobase, patient_id)
                profiles = user.profile(max_token_size=5000)
                remote_profile_count = len(list(profiles or []))
                remote_checked = True
            except Exception:
                remote_checked = False
        if queued:
            self._memory_worker_wakeup.set()
            self._emit_audit(patient_id, "mirror_consistency_compensation", "queued")
        return {
            "patient_id_hash": uuid.uuid5(uuid.NAMESPACE_URL, patient_id).hex,
            "queued_compensations": queued,
            "mirror_lag_count": int(pending) + queued,
            "remote_checked": remote_checked,
            "remote_profile_count": remote_profile_count,
        }

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
        with self._lock, self._connection() as conn:
            shared_local_item = conn.execute(
                f"""SELECT 1 FROM {_MEMORY_ITEM_TABLE}
                    WHERE patient_id=? AND status='active' AND content=? LIMIT 1""",
                (patient_id, content),
            ).fetchone() if content else None
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
                if profile_id and profile_content and not shared_local_item and (
                    re.sub(r"[^\w]+", "", profile_content)
                    == re.sub(r"[^\w]+", "", content)
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
        cutoff_turn_id: int,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = f"""SELECT id, session_id, turn_id, user_message, assistant_message,
                           emotion_json, created_at, turn_state, response_status
                    FROM {_TURN_TABLE}
                    WHERE patient_id=? AND id<=? AND consolidation_status='pending'"""
        params: list[Any] = [patient_id, int(cutoff_turn_id)]
        if session_id:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY id ASC"
        rows = conn.execute(query, tuple(params)).fetchall()
        result = []
        for row in rows:
            if str(row["turn_state"] or "").upper() in {
                "CANCELLED", "FAILED", "RESPONSE_CANCELLED"
            }:
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
    def _consolidation_waterline(
        conn: sqlite3.Connection,
        patient_id: str,
        cutoff_turn_id: int,
    ) -> int:
        """Return the highest turn in the consecutively completed prefix."""
        rows = conn.execute(
            f"""SELECT id, consolidation_status FROM {_TURN_TABLE}
                WHERE patient_id=? AND id<=?
                ORDER BY id ASC""",
            (patient_id, int(cutoff_turn_id)),
        ).fetchall()
        waterline = 0
        for row in rows:
            if str(row["consolidation_status"] or "").lower() not in {
                "consolidated", "skipped"
            }:
                break
            waterline = int(row["id"])
        return waterline

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
        snapshot: Mapping[str, Any],
        patch: Mapping[str, Any],
        suppressed_turn_ids: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        merged = _normalise_snapshot(str(snapshot["patient_id"]), snapshot)
        for field in ("facts", "preferences", "events"):
            current = list(merged.get(field) or [])
            if field == "events":
                current = _compact_event_entries(current)
            superseded = _superseded_turn_ids({field: current})
            superseded.update(suppressed_turn_ids or set())
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

    def _apply_consolidation_items(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        merged: Mapping[str, Any],
    ) -> None:
        """Persist a derived projection without treating snapshot JSON as authority."""
        rows = [
            self._memory_item_row(row)
            for row in conn.execute(
                f"""SELECT * FROM {_MEMORY_ITEM_TABLE}
                    WHERE patient_id=? AND mode=? AND status='active'""",
                (patient_id, normalize_session_mode(WELLBEING)),
            ).fetchall()
        ]
        by_id = {str(row["item_id"]): row for row in rows}
        by_content = {
            (str(row["category"]), str(row["content"])): row
            for row in rows
        }
        target_ids: set[str] = set()

        for category in ("facts", "preferences", "events", "comfort_strategies"):
            for raw in merged.get(category) or []:
                entry = dict(raw) if isinstance(raw, Mapping) else {"text": raw}
                text = _memory_item_text(entry)
                if not text:
                    continue
                requested_id = str(
                    entry.get("item_id") or entry.get("id") or ""
                ).strip() or None
                current = by_id.get(requested_id) if requested_id else None
                if current is None:
                    current = by_content.get((category, text))
                if current is not None and str(current["content"]) != text:
                    requested_id = None
                    current = None
                if current is not None:
                    current_id = str(current["item_id"])
                    target_ids.add(current_id)
                    evidence_json = self._item_evidence_json(entry)
                    metadata = _json_object(
                        entry.get("metadata_json") or entry.get("metadata")
                    )
                    if category == "comfort_strategies" and "effective" in entry:
                        metadata["effective"] = entry["effective"]
                    metadata_json = (
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                        if metadata
                        else None
                    )
                    updates: list[str] = []
                    values: list[Any] = []
                    if evidence_json is not None and evidence_json != current.get("evidence_json"):
                        updates.append("evidence_json=?")
                        values.append(evidence_json)
                    if metadata_json is not None and metadata_json != current.get("metadata_json"):
                        updates.append("metadata_json=?")
                        values.append(metadata_json)
                    if updates:
                        now = _now()
                        values.extend((int(current["version"]) + 1, now, patient_id, current_id))
                        conn.execute(
                            f"""UPDATE {_MEMORY_ITEM_TABLE}
                                SET {', '.join(updates)}, version=?, updated_at=?
                                WHERE patient_id=? AND item_id=? AND status='active'""",
                            tuple(values),
                        )
                    continue

                if requested_id:
                    existing = conn.execute(
                        f"SELECT status FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=?",
                        (patient_id, requested_id),
                    ).fetchone()
                    if existing:
                        # A deleted/superseded id is a tombstone, never a reusable primary key.
                        continue
                item_id = requested_id or uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"lmca-derived:{patient_id}:{category}:{text}",
                ).hex
                if item_id in by_id:
                    continue
                source = str(entry.get("source") or "system_inferred").strip() or "system_inferred"
                if source == "derived":
                    source = "system_inferred"
                evidence_json = self._item_evidence_json(entry)
                source_turn_id = str(entry.get("source_turn_id") or "").strip() or None
                if not source_turn_id:
                    try:
                        evidence = json.loads(evidence_json or "[]")
                    except json.JSONDecodeError:
                        evidence = []
                    if evidence and isinstance(evidence[0], Mapping):
                        source_turn_id = str(evidence[0].get("turn_id") or "").strip() or None
                metadata = _json_object(
                    entry.get("metadata_json") or entry.get("metadata")
                )
                if category == "comfort_strategies" and "effective" in entry:
                    metadata["effective"] = entry["effective"]
                try:
                    confidence = (
                        float(entry["confidence"])
                        if entry.get("confidence") is not None
                        else None
                    )
                except (TypeError, ValueError):
                    confidence = None
                self._insert_memory_item(
                    conn,
                    patient_id,
                    category,
                    text,
                    item_id=item_id,
                    mode=WELLBEING,
                    source=source,
                    source_session_id=str(entry.get("source_session_id") or "").strip() or None,
                    source_turn_id=source_turn_id,
                    slot_key=str(entry.get("slot_key") or "").strip() or None,
                    observed_at=str(entry.get("observed_at") or "").strip() or None,
                    valid_until=str(entry.get("valid_until") or "").strip() or None,
                    sensitivity=str(entry.get("sensitivity") or "normal"),
                    confidence=confidence,
                    confirmed_by=str(entry.get("confirmed_by") or "").strip() or None,
                    evidence_json=evidence_json,
                    supersedes_item_id=str(entry.get("supersedes_item_id") or "").strip() or None,
                    metadata=metadata or None,
                )
                target_ids.add(item_id)

        # Event compaction can replace a derived event with a longer equivalent.
        # Keep the old evidence for audit, but make it unavailable for prompts.
        for row in rows:
            item_id = str(row["item_id"])
            if item_id in target_ids or str(row["category"]) != "events":
                continue
            if _event_is_user_managed(row):
                continue
            version = int(row["version"]) + 1
            now = _now()
            conn.execute(
                f"""UPDATE {_MEMORY_ITEM_TABLE}
                    SET status='superseded', version=?, updated_at=?
                    WHERE patient_id=? AND item_id=? AND status='active'""",
                (version, now, patient_id, item_id),
            )
            self._enqueue_mirror_op(
                conn,
                patient_id,
                item_id,
                version,
                "invalidate",
                str(row.get("source_turn_id") or "") or None,
            )

    def consolidate_pending_turns(
        self,
        patient_id: str,
        cutoff_turn_id: Optional[int] = None,
        *,
        patch: Optional[Mapping[str, Any]] = None,
        patch_builder: Optional[Callable[..., Mapping[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Consolidate each pending turn exactly once, independent of session order."""
        patient_id = self._patient_id(patient_id)
        if patch is not None or patch_builder is not None:
            raise ValueError(
                "direct consolidation patches are disabled; use Qwen reflection candidates"
            )
        session_id = str(session_id).strip() or None if session_id else None
        with self._lock, self._connection() as conn:
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
            conn.execute(
                f"""UPDATE {_TURN_TABLE}
                    SET consolidation_status='skipped', consolidated_at=?
                    WHERE patient_id=? AND id<=? AND consolidation_status='pending'
                      AND UPPER(COALESCE(turn_state, '')) IN
                          ('CANCELLED', 'FAILED', 'RESPONSE_CANCELLED')""",
                (_now(), patient_id, upper),
            )
            snapshot = self._snapshot_from_items(
                conn, self._load_snapshot(conn, patient_id)
            )
            turns = self._pending_consolidation_turns(conn, patient_id, upper, session_id)

        if not turns:
            with self._lock, self._connection() as conn:
                current = self._load_snapshot(conn, patient_id)
                waterline = self._consolidation_waterline(conn, patient_id, upper)
                revision = int(current.get("revision") or 0)
            return {
                "applied": True,
                "patient_id": patient_id,
                "revision": revision,
                "last_consolidated_turn_id": waterline,
                "turn_count": 0,
                "patch": {},
            }
        if patch_builder is not None:
            try:
                generated = patch_builder(turns, snapshot)
            except TypeError:
                generated = patch_builder(turns)
        else:
            # Turn persistence remains available to callers that explicitly provide a patch;
            # session memory extraction is owned exclusively by reflect_session().
            generated = patch if patch is not None else {}
        validated = self._validate_consolidation_patch(generated)
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._load_snapshot(conn, patient_id)
            current_items_snapshot = self._snapshot_from_items(conn, current)
            selected_ids = [int(turn["local_turn_id"]) for turn in turns]
            placeholders = ", ".join("?" for _ in selected_ids)
            ready_rows = conn.execute(
                f"""SELECT id FROM {_TURN_TABLE}
                    WHERE patient_id=? AND consolidation_status='pending'
                      AND id IN ({placeholders})
                      AND UPPER(COALESCE(turn_state, '')) NOT IN
                          ('CANCELLED', 'FAILED', 'RESPONSE_CANCELLED')""",
                (patient_id, *selected_ids),
            ).fetchall()
            if not ready_rows:
                waterline = self._consolidation_waterline(conn, patient_id, upper)
                return {
                    "applied": True,
                    "patient_id": patient_id,
                    "revision": int(current.get("revision") or 0),
                    "last_consolidated_turn_id": waterline,
                    "turn_count": 0,
                    "patch": {},
                }
            ready_ids = [int(row[0]) for row in ready_rows]
            ready_placeholders = ", ".join("?" for _ in ready_ids)
            suppressed = _superseded_turn_ids(current_items_snapshot)
            suppressed.update(self._deleted_turn_ids(conn, patient_id))
            merged = self._merge_consolidation_patch(
                current_items_snapshot,
                validated,
                suppressed,
            )
            now = _now()
            conn.execute(
                f"""UPDATE {_TURN_TABLE}
                    SET consolidation_status='consolidated', consolidated_at=?
                    WHERE patient_id=? AND id IN ({ready_placeholders})
                      AND consolidation_status='pending'""",
                (now, patient_id, *ready_ids),
            )
            waterline = self._consolidation_waterline(conn, patient_id, upper)
            merged["last_consolidated_turn_id"] = waterline
            self._apply_consolidation_items(conn, patient_id, merged)
            stored = self._snapshot_from_items(conn, current)
            stored["narrative"] = str(merged.get("narrative") or "")
            stored["_narrative_manual"] = False
            stored["last_consolidated_turn_id"] = waterline
            self._save_snapshot(conn, stored, updated_by="system")
        return {
            "applied": True,
            "patient_id": patient_id,
            "revision": int(current.get("revision") or 0) + 1,
            "last_consolidated_turn_id": waterline,
            "turn_count": len(ready_rows),
            "patch": validated,
        }

    def _session_turns_for_reflection(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"""SELECT id, turn_id, user_message, assistant_message, created_at
                FROM {_TURN_TABLE}
                WHERE patient_id=? AND session_id=?
                  AND UPPER(COALESCE(turn_state, '')) NOT IN
                      ('CANCELLED', 'FAILED', 'RESPONSE_CANCELLED')
                ORDER BY id ASC""",
            (patient_id, session_id),
        ).fetchall()
        return [dict(row) for row in rows if str(row["user_message"] or "").strip()]

    @staticmethod
    def _mark_reflection_turns_complete(
        conn: sqlite3.Connection,
        patient_id: str,
        session_id: str,
    ) -> None:
        now = _now()
        conn.execute(
            f"""UPDATE {_TURN_TABLE}
                SET consolidation_status='skipped', consolidated_at=?
                WHERE patient_id=? AND session_id=? AND consolidation_status='pending'
                  AND UPPER(COALESCE(turn_state, '')) IN
                      ('CANCELLED', 'FAILED', 'RESPONSE_CANCELLED')""",
            (now, patient_id, session_id),
        )
        conn.execute(
            f"""UPDATE {_TURN_TABLE}
                SET consolidation_status='consolidated', consolidated_at=?
                WHERE patient_id=? AND session_id=? AND consolidation_status='pending'
                  AND UPPER(COALESCE(turn_state, '')) NOT IN
                      ('CANCELLED', 'FAILED', 'RESPONSE_CANCELLED')""",
            (now, patient_id, session_id),
        )

    @staticmethod
    def _reflection_prompt(
        turns: list[Mapping[str, Any]],
        active_memory: list[Mapping[str, Any]] | None = None,
    ) -> str:
        transcript = []
        for turn in turns:
            turn_id = str(turn["turn_id"])
            transcript.append(
                f"[user turn_id={turn_id} observed_at={turn.get('created_at', '')}] "
                f"{turn['user_message']}"
            )
            assistant = str(turn.get("assistant_message") or "").strip()
            if assistant:
                transcript.append(f"[assistant context] {assistant}")
        schema = {
            "session_summary": {"content": "string", "evidence_turn_ids": ["turn_id"]},
            "memory_candidates": [{
                "category": "facts|preferences|events|emotional_triggers|boundaries|comfort_strategies",
                "content": "string",
                "source_turn_id": "turn_id",
                "evidence_turn_ids": ["turn_id"],
                "evidence_quote": "verbatim user quote",
                "observed_at": "ISO-8601 date/time",
                "valid_until": "ISO-8601 date/time or null",
                "sensitivity": "normal|sensitive|high",
                "confidence": 0.0,
                "slot_key": "stable field key or null",
                "supersedes_item_id": "existing active item_id or null",
                "suggested_operation": "ADD|SUPERSEDE|CANDIDATE|NOOP|BLOCK",
            }],
        }
        active = [
            {
                "item_id": str(item.get("item_id") or ""),
                "category": str(item.get("category") or ""),
                "content": str(item.get("content") or ""),
                "slot_key": item.get("slot_key"),
                "valid_until": item.get("valid_until"),
            }
            for item in (active_memory or [])
            if str(item.get("item_id") or "").strip()
        ]
        return (
            "从会话提取对后续陪伴有用的记忆；只返回 JSON，不诊断、不臆测。\n"
            "证据：只相信 [user]；[assistant] 不能作事实。source_turn_id 必须属于 evidence_turn_ids。"
            "evidence_quote 必须逐字引用 user；多轮按 turn 顺序逐行引用。content 必须保留否定方向、时间和不确定性。\n"
            "负向偏好和长期边界可以保存，但 content 必须保留否定方向；对症状、争执、危机或自伤意图的否认通常不保存，"
            "绝不能反写为存在；带“现在、今天、这次、暂时”等短期范围的拒绝必须保留范围或 valid_until，不得升级为永久边界。\n"
            "明确纠正必须提取为一条 candidate：识别“不是A，是B”“改成B”“我记错了”“搬到B”等表达时，记录被纠正后的B，"
            "suggested_operation 使用 SUPERSEDE，并保留纠正所在 user turn；不能因为纠正只有一轮、内容简单或已有旧记忆而漏掉。\n"
            "多轮 user 原话合起来形成长期陪伴价值时，必须形成一条综合 candidate，并在 evidence_turn_ids 中列出所有支持该事实的 user turn；"
            "不要把一个跨轮事实拆成互不相干的单轮记忆。\n"
            "类别：facts 身份/关系/居住/用药；preferences 稳定偏好；events 重要经历/变化；"
            "emotional_triggers 情绪触发；boundaries 互动边界；comfort_strategies 用户确认有效的方法。"
            "寒暄、天气、一次性小事、助手猜测、明确否认均为 NOOP。\n"
            "时间：出现今天/明天/本周/下周/下个月等相对日期，就必须填晚于 observed_at 的 valid_until，不论类别；长期事实才填 null。"
            "content 保留用户的相对时间，不自行改写成绝对日期。\n"
            "操作：新增 ADD；明确更正用 SUPERSEDE，并填写 EXISTING ACTIVE MEMORY 中被替换项的真实 item_id；无法唯一匹配则填 null，不得伪造。"
            "同一事实只输出一条并合并证据。用户直接表达的高风险内容为 high；明确否认不保存。"
            "无内容时也必须返回完整 JSON：{\"session_summary\":{},\"memory_candidates\":[]}。\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "EXISTING ACTIVE MEMORY:\n" + json.dumps(active, ensure_ascii=False, separators=(',', ':'))
            + "\n\n会话记录：\n" + "\n".join(transcript)
        )

    @staticmethod
    def _invoke_memory_model(prompt: str, model: str, max_tokens: int) -> Any:
        kwargs = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "timeout": 30,
            "max_retries": 2,
            "streaming": False,
            "disable_thinking": model.lower().startswith("qwen"),
        }
        if os.getenv("DASHSCOPE_API_KEY"):
            from src.llm.http_client_pool import get_dashscope_chat_openai
            llm = get_dashscope_chat_openai(**kwargs)
        elif os.getenv("SILICONFLOW_API_KEY"):
            from src.llm.http_client_pool import get_siliconflow_chat_openai
            llm = get_siliconflow_chat_openai(**kwargs)
        else:
            raise RuntimeError("memory reflection provider is unavailable")
        response = llm.invoke([{"role": "user", "content": prompt}])
        return getattr(response, "content", response)

    def _invoke_session_reflector(self, prompt: str) -> Any:
        if self._session_reflector is not None:
            return self._session_reflector(prompt)
        model = str(os.getenv("MEMORY_REFLECTION_MODEL") or "qwen3.8-max").strip()
        return self._invoke_memory_model(prompt, model, 1600)

    def _invoke_session_verifier(self, prompt: str) -> Any:
        if self._session_verifier is not None:
            return self._session_verifier(prompt)
        model = str(os.getenv("MEMORY_VERIFICATION_MODEL") or "qwen-plus").strip()
        return self._invoke_memory_model(prompt, model, 800)

    @staticmethod
    def _reflection_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        raw = str(getattr(value, "content", value) or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("reflection response must be a JSON object")
        return dict(parsed)

    @staticmethod
    def _verification_prompt(items: list[Mapping[str, Any]]) -> str:
        return (
            "复核长期陪伴记忆，只返回 JSON；每个 item_id 输出一次 CONFIRM 或 REJECT，不改写 candidate。"
            "直接由 user_evidence 支持且字段完整就 CONFIRM，包括固定日程、过去经历、关系变化、偏好和互动边界。"
            "只有以下情况 REJECT：证据来自 assistant 或用户明确否认；content 丢失否定/时间/不确定性；"
            "出现相对日期却没有 valid_until；或 SUPERSEDE 在 active_conflicts 找不到唯一旧项。"
            "高敏感内容有用户直接证据即可 CONFIRM，不以‘是否长期’作为拒绝理由。"
            "格式：{\"decisions\":[{\"item_id\":\"...\",\"decision\":\"CONFIRM|REJECT\"}]}\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    def _verification_inputs(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        item_ids: list[str],
        turns: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        by_turn_id = {str(turn["turn_id"]): turn for turn in turns}
        result = []
        for item_id in item_ids:
            row = conn.execute(
                f"SELECT * FROM {_MEMORY_ITEM_TABLE} WHERE patient_id=? AND item_id=? AND status='candidate'",
                (patient_id, item_id),
            ).fetchone()
            if not row:
                continue
            item = self._memory_item_row(row)
            try:
                evidence = json.loads(item.get("evidence_json") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError("verification evidence is invalid") from exc
            if not isinstance(evidence, list):
                raise ValueError("verification evidence is invalid")
            evidence_turns = [
                by_turn_id.get(str(entry.get("turn_id") or ""))
                for entry in evidence if isinstance(entry, Mapping)
            ]
            if not evidence_turns or not all(evidence_turns):
                raise ValueError("verification evidence is missing")
            metadata = _json_object(item.get("metadata_json"))
            conflicts: list[dict[str, Any]] = []
            if str(metadata.get("suggested_operation") or "").upper() == "SUPERSEDE":
                target_id = str(item.get("supersedes_item_id") or "").strip()
                if target_id:
                    target = conn.execute(
                        f"""SELECT item_id, category, content, slot_key FROM {_MEMORY_ITEM_TABLE}
                            WHERE patient_id=? AND item_id=? AND status='active'""",
                        (patient_id, target_id),
                    ).fetchone()
                    conflicts = [dict(target)] if target else []
                slot_key = str(item.get("slot_key") or "").strip()
                if slot_key and not conflicts:
                    conflicts = [dict(conflict) for conflict in conn.execute(
                        f"""SELECT item_id, category, content, slot_key FROM {_MEMORY_ITEM_TABLE}
                            WHERE patient_id=? AND mode=? AND category=? AND slot_key=? AND status='active'""",
                        (patient_id, item["mode"], item["category"], slot_key),
                    ).fetchall()]
            result.append({
                "item_id": item_id,
                "candidate": {
                    "category": item["category"], "content": item["content"],
                    "valid_until": item.get("valid_until"),
                    "sensitivity": item.get("sensitivity"),
                    "suggested_operation": metadata.get("suggested_operation"),
                },
                "user_evidence": [
                    {"turn_id": turn["turn_id"], "text": turn["user_message"]}
                    for turn in evidence_turns
                ],
                "active_conflicts": conflicts,
            })
        return result

    def _verify_candidates(
        self,
        patient_id: str,
        item_ids: list[str],
        turns: list[Mapping[str, Any]],
    ) -> dict[str, int | str]:
        with self._lock, self._connection() as conn:
            items = self._verification_inputs(conn, patient_id, item_ids, turns)
        if not items:
            return {"status": "skipped", "confirmed_count": 0, "rejected_count": 0}
        payload = self._reflection_payload(
            self._invoke_session_verifier(self._verification_prompt(items))
        )
        decisions = payload.get("decisions")
        expected = {item["item_id"] for item in items}
        if not isinstance(decisions, list):
            raise ValueError("verification decisions must be a list")
        parsed: dict[str, str] = {}
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("verification decision is invalid")
            item_id = str(decision.get("item_id") or "").strip()
            value = str(decision.get("decision") or "").upper().strip()
            if item_id in parsed or item_id not in expected or value not in {"CONFIRM", "REJECT"}:
                raise ValueError("verification decision is invalid")
            parsed[item_id] = value
        if set(parsed) != expected:
            raise ValueError("verification decisions are incomplete")
        model = str(os.getenv("MEMORY_VERIFICATION_MODEL") or "qwen-plus").strip()
        actor = f"memory:{model}|prompt:{_VERIFICATION_PROMPT_VERSION}|run:{uuid.uuid4().hex[:12]}"
        confirmed = rejected = 0
        for item in items:
            current = self.get_memory_item(patient_id, item["item_id"])
            if parsed[current["item_id"]] == "CONFIRM":
                self.confirm_memory_item(patient_id, current["item_id"], current["version"], confirmed_by=actor)
                confirmed += 1
            else:
                self.reject_memory_item(patient_id, current["item_id"], current["version"], rejected_by=actor)
                rejected += 1
        return {"status": "succeeded", "confirmed_count": confirmed, "rejected_count": rejected}

    def _claim_session_reflection(
        self,
        patient_id: str,
        session_id: str,
        *,
        force: bool,
    ) -> dict[str, Any]:
        now = datetime.now()
        now_text = now.isoformat()
        key = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"lmca-session-reflection:{patient_id}:{session_id}",
        ).hex
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT * FROM {_SESSION_REFLECTION_TABLE} WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            ).fetchone()
            if row and row["status"] == "succeeded":
                return {"claimed": False, "row": dict(row), "idempotent_replay": True}
            if row and row["status"] == "dead_letter" and not force:
                return {"claimed": False, "row": dict(row), "deferred": True}
            if row and row["status"] == "running" and str(row["lease_until"] or "") > now_text:
                return {"claimed": False, "row": dict(row), "deferred": True}
            if row and row["status"] in {"failed", "pending"} and not force and str(row["next_retry_at"] or "") > now_text:
                return {"claimed": False, "row": dict(row), "deferred": True}
            attempts = int(row["attempt_count"] or 0) + 1 if row else 1
            lease_until = (now + timedelta(seconds=_REFLECTION_LEASE_SECONDS)).isoformat()
            lease_token = uuid.uuid4().hex
            if row:
                conn.execute(
                    f"""UPDATE {_SESSION_REFLECTION_TABLE}
                        SET status='running', attempt_count=?, next_retry_at=NULL,
                            last_error=NULL, error_code=NULL, dead_letter_at=NULL,
                            lease_until=?, lease_token=?, updated_at=?
                        WHERE patient_id=? AND session_id=?""",
                    (attempts, lease_until, lease_token, now_text, patient_id, session_id),
                )
            else:
                conn.execute(
                    f"""INSERT INTO {_SESSION_REFLECTION_TABLE}
                        (patient_id, session_id, idempotency_key, status, attempt_count,
                         lease_until, lease_token, created_at, updated_at)
                        VALUES (?, ?, ?, 'running', 1, ?, ?, ?, ?)""",
                    (patient_id, session_id, key, lease_until, lease_token, now_text, now_text),
                )
            return {
                "claimed": True,
                "idempotency_key": key,
                "lease_token": lease_token,
                "attempt_count": attempts,
            }

    def _reflection_claim_is_current(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        session_id: str,
        lease_token: str,
    ) -> bool:
        row = conn.execute(
            f"""SELECT 1 FROM {_SESSION_REFLECTION_TABLE}
                WHERE patient_id=? AND session_id=? AND status='running'
                  AND lease_token=? AND lease_until>?""",
            (patient_id, session_id, lease_token, datetime.now().isoformat()),
        ).fetchone()
        return row is not None

    def _fail_session_reflection(
        self,
        patient_id: str,
        session_id: str,
        exc: Exception,
        lease_token: str,
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"""SELECT attempt_count FROM {_SESSION_REFLECTION_TABLE}
                    WHERE patient_id=? AND session_id=? AND status='running'
                      AND lease_token=? AND lease_until>?""",
                (patient_id, session_id, lease_token, datetime.now().isoformat()),
            ).fetchone()
            if not row:
                return {
                    "status": "stale",
                    "patient_id": patient_id,
                    "session_id": session_id,
                }
            attempts = int(row["attempt_count"] or 1) if row else 1
            terminal = attempts >= _REFLECTION_MAX_ATTEMPTS
            retry_at = None if terminal else datetime.now() + timedelta(
                seconds=min(300, _REFLECTION_RETRY_SECONDS * (2 ** min(attempts - 1, 2)))
            )
            error_code = self._reflection_error_code(exc)
            conn.execute(
                f"""UPDATE {_SESSION_REFLECTION_TABLE}
                    SET status=?, next_retry_at=?, last_error=?, error_code=?,
                        dead_letter_at=CASE WHEN ? THEN ? ELSE NULL END,
                        lease_until=NULL, lease_token=NULL, updated_at=?
                    WHERE patient_id=? AND session_id=? AND status='running'
                      AND lease_token=?""",
                (
                    "dead_letter" if terminal else "failed",
                    retry_at.isoformat() if retry_at else None,
                    f"{type(exc).__name__}: {exc}"[:500],
                    error_code,
                    terminal,
                    _now() if terminal else None,
                    _now(),
                    patient_id,
                    session_id,
                    lease_token,
                ),
            )
        self._emit_audit(patient_id, "reflection_dead_letter" if terminal else "reflection_retry", error_code)
        return {"status": "dead_letter" if terminal else "failed", "patient_id": patient_id, "session_id": session_id, "error_code": error_code}

    @staticmethod
    def _reflection_error_code(exc: Exception) -> str:
        message = str(exc).lower()
        if isinstance(exc, (json.JSONDecodeError,)) or "json" in message or "payload" in message:
            return "invalid_payload"
        if "safety" in message or "evidence" in message:
            return "evidence_rejected"
        if "qwen" in message or "provider" in message or "timeout" in message:
            return "provider_error"
        return "configuration_error" if isinstance(exc, (ValueError, RuntimeError)) else "provider_error"

    def enqueue_session_reflection(self, patient_id: str, session_id: str) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        now = _now()
        key = uuid.uuid5(uuid.NAMESPACE_URL, f"lmca-session-reflection:{patient_id}:{session_id}").hex
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"""INSERT INTO {_SESSION_REFLECTION_TABLE}
                    (patient_id, session_id, idempotency_key, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    ON CONFLICT(patient_id, session_id) DO UPDATE SET
                    status=CASE WHEN {_SESSION_REFLECTION_TABLE}.status IN ('failed','dead_letter') THEN 'pending' ELSE {_SESSION_REFLECTION_TABLE}.status END,
                    next_retry_at=NULL, last_error=NULL, error_code=NULL,
                    dead_letter_at=NULL, updated_at=?""",
                (patient_id, session_id, key, now, now, now),
            )
            row = conn.execute(
                f"SELECT * FROM {_SESSION_REFLECTION_TABLE} WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            ).fetchone()
        self._memory_worker_wakeup.set()
        self._emit_audit(patient_id, "reflection_enqueued", "success")
        return dict(row) if row else {"patient_id": patient_id, "session_id": session_id, "status": "pending"}

    def claim_due_reflection_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        now = datetime.now()
        now_text = now.isoformat()
        claimed: list[dict[str, Any]] = []
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""SELECT * FROM {_SESSION_REFLECTION_TABLE}
                    WHERE (status IN ('pending','failed') AND (next_retry_at IS NULL OR next_retry_at<=?))
                       OR (status='running' AND lease_until<=?)
                    ORDER BY created_at LIMIT ?""",
                (now_text, now_text, max(1, min(int(limit), 100))),
            ).fetchall()
            for row in rows:
                attempts = int(row["attempt_count"] or 0) + 1
                token = uuid.uuid4().hex
                lease_until = (now + timedelta(seconds=_REFLECTION_LEASE_SECONDS)).isoformat()
                conn.execute(
                    f"""UPDATE {_SESSION_REFLECTION_TABLE}
                        SET status='running', attempt_count=?, next_retry_at=NULL,
                            lease_until=?, lease_token=?, updated_at=?
                        WHERE patient_id=? AND session_id=?
                          AND (status IN ('pending','failed') OR lease_until<=?)""",
                    (attempts, lease_until, token, now_text, row["patient_id"], row["session_id"], now_text),
                )
                if conn.execute("SELECT changes() AS n").fetchone()["n"]:
                    claimed.append({**dict(row), "attempt_count": attempts, "lease_token": token, "lease_until": lease_until, "status": "running"})
        for task in claimed:
            self._emit_audit(str(task["patient_id"]), "reflection_claimed", "success")
        return claimed

    def requeue_session_reflection(self, patient_id: str, session_id: str, reason: str = "manual") -> bool:
        patient_id = self._patient_id(patient_id)
        with self._lock, self._connection() as conn:
            updated = conn.execute(
                f"""UPDATE {_SESSION_REFLECTION_TABLE}
                    SET status='pending', next_retry_at=NULL, lease_until=NULL, lease_token=NULL,
                        last_error=?, error_code=NULL, dead_letter_at=NULL, updated_at=?
                    WHERE patient_id=? AND session_id=?""",
                (_clip(reason, 200), _now(), patient_id, str(session_id).strip()),
            ).rowcount
        self._memory_worker_wakeup.set()
        self._emit_audit(patient_id, "reflection_requeued", "success" if updated else "missing")
        return bool(updated)

    def get_session_reflection(self, patient_id: str, session_id: str) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {_SESSION_REFLECTION_TABLE} WHERE patient_id=? AND session_id=?",
                (patient_id, session_id),
            ).fetchone()
        if not row:
            return {"status": "missing", "patient_id": patient_id, "session_id": session_id}
        result = dict(row)
        result["session_summary"] = result.pop("session_summary") or ""
        return result

    def get_latest_session_reflection(
        self,
        patient_id: str,
        *,
        exclude_session_id: str | None = None,
    ) -> dict[str, Any]:
        patient_id = self._patient_id(patient_id)
        excluded = str(exclude_session_id or "").strip()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"""SELECT * FROM {_SESSION_REFLECTION_TABLE}
                    WHERE patient_id=? AND status='succeeded'
                      AND COALESCE(session_summary, '')<>''
                      AND (?='' OR session_id<>?)
                    ORDER BY COALESCE(completed_at, updated_at, created_at) DESC
                    LIMIT 1""",
                (patient_id, excluded, excluded),
            ).fetchone()
        if not row:
            return {"status": "missing", "patient_id": patient_id}
        result = dict(row)
        result["session_summary"] = result.pop("session_summary") or ""
        return result

    def reflect_session(
        self,
        patient_id: str,
        session_id: str,
        *,
        force: bool = False,
        _claimed_lease: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract session candidates after a session ends."""
        patient_id = self._patient_id(patient_id)
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock, self._connection() as conn:
            turns = self._session_turns_for_reflection(conn, patient_id, session_id)
            active_memory = [
                item
                for items in self._active_items_by_category(conn, patient_id, mode=WELLBEING).values()
                for item in items
            ]
        if not turns:
            return {"status": "skipped", "patient_id": patient_id, "session_id": session_id}
        claim = dict(_claimed_lease or self._claim_session_reflection(patient_id, session_id, force=force))
        if not claim.get("claimed"):
            row = claim.get("row") or {}
            return {
                "status": str(row.get("status") or "pending"),
                "patient_id": patient_id,
                "session_id": session_id,
                "idempotent_replay": bool(claim.get("idempotent_replay")),
                "deferred": bool(claim.get("deferred")),
            }
        lease_token = str(claim["lease_token"])
        candidate_item_ids: list[str] = []
        verification: dict[str, int | str] = {
            "status": "disabled", "confirmed_count": 0, "rejected_count": 0,
        }
        try:
            payload = self._reflection_payload(
                self._invoke_session_reflector(self._reflection_prompt(turns, active_memory))
            )
            if not payload.get("memory_candidates") and any(
                str(turn.get("user_message") or "").strip() for turn in turns
            ):
                payload = self._reflection_payload(
                    self._invoke_session_reflector(
                        self._reflection_prompt(turns, active_memory)
                        + "\n请再次逐条复核 user 原话：如果存在任何可作为长期陪伴候选的事实、偏好、临时边界、关系变化或高风险表达，请按 schema 输出 candidate；只有确实没有可保存内容时才返回空数组。"
                    )
                )
        except Exception as exc:
            return self._fail_session_reflection(patient_id, session_id, exc, lease_token)

        try:
            with self._lock, self._connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if not self._reflection_claim_is_current(
                    conn, patient_id, session_id, lease_token
                ):
                    return {
                        "status": "stale",
                        "patient_id": patient_id,
                        "session_id": session_id,
                    }
                current_turns = self._session_turns_for_reflection(conn, patient_id, session_id)
                by_reference = {
                    reference: turn
                    for turn in current_turns
                    for reference in (str(turn["id"]), str(turn["turn_id"]))
                }
                summary = payload.get("session_summary") or {}
                summary_text = ""
                summary_evidence: list[dict[str, str]] = []
                if isinstance(summary, Mapping):
                    content = str(summary.get("content") or "").strip()
                    references = summary.get("evidence_turn_ids") or []
                    if isinstance(references, (list, tuple)) and content and len(content) <= 2000:
                        resolved = [by_reference.get(str(value or "").strip()) for value in references]
                        if resolved and all(resolved):
                            summary_text = content
                            summary_evidence = [
                                {"turn_id": str(turn["turn_id"])} for turn in resolved
                            ]
                candidates = payload.get("memory_candidates") or []
                if not isinstance(candidates, list):
                    raise ValueError("memory_candidates must be a list")
                deduplicated_candidates: list[Mapping[str, Any]] = []
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        deduplicated_candidates.append(candidate)
                        continue
                    source_id = str(candidate.get("source_turn_id") or "").strip()
                    evidence_ids = {
                        str(value or "").strip()
                        for value in candidate.get("evidence_turn_ids") or []
                        if str(value or "").strip()
                    }
                    content = str(candidate.get("content") or "").strip()
                    duplicate_index = next(
                        (
                            index
                            for index, existing in enumerate(deduplicated_candidates)
                            if isinstance(existing, Mapping)
                            and source_id
                            and source_id == str(existing.get("source_turn_id") or "").strip()
                            and evidence_ids == {
                                str(value or "").strip()
                                for value in existing.get("evidence_turn_ids") or []
                                if str(value or "").strip()
                            }
                            and content
                            and (
                                content in str(existing.get("content") or "")
                                or str(existing.get("content") or "") in content
                            )
                        ),
                        None,
                    )
                    if duplicate_index is None:
                        deduplicated_candidates.append(candidate)
                    elif len(content) > len(
                        str(deduplicated_candidates[duplicate_index].get("content") or "")
                    ):
                        deduplicated_candidates[duplicate_index] = candidate
                candidates = deduplicated_candidates
                multi_evidence_categories = {
                    str(candidate.get("category") or "").strip()
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and len(candidate.get("evidence_turn_ids") or []) > 1
                }
                multi_evidence_turns = {
                    str(turn_id or "").strip()
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and len(candidate.get("evidence_turn_ids") or []) > 1
                    for turn_id in candidate.get("evidence_turn_ids") or []
                }
                candidate_count = 0
                rejected_count = 0
                safety_failure = False
                for index, candidate in enumerate(candidates):
                    if not isinstance(candidate, Mapping):
                        rejected_count += 1
                        safety_failure = True
                        continue
                    category = str(candidate.get("category") or "").strip()
                    content = str(candidate.get("content") or "").strip()
                    operation = str(candidate.get("suggested_operation") or "").upper().strip()
                    source = by_reference.get(str(candidate.get("source_turn_id") or "").strip())
                    references = candidate.get("evidence_turn_ids") or []
                    quote = str(candidate.get("evidence_quote") or "").strip()
                    if (
                        len(references) == 1
                        and (
                            category in multi_evidence_categories
                            or str(references[0] or "").strip() in multi_evidence_turns
                        )
                    ):
                        # A single-turn candidate that overlaps a multi-turn one
                        # usually restates part of it, so it is dropped to keep
                        # shallow evidence out of the store.  It is not a
                        # malformed payload, so it must not abort the session:
                        # doing so discarded the well-formed multi-turn
                        # candidate alongside it and left the whole reflection
                        # empty.
                        rejected_count += 1
                        continue
                    if (
                        category not in _REFLECTION_CATEGORIES
                        or not content or len(content) > 500
                        or operation not in _REFLECTION_OPERATIONS
                        or not isinstance(references, (list, tuple))
                        or not quote or len(quote) > 240
                    ):
                        rejected_count += 1
                        safety_failure = True
                        continue
                    if source is None:
                        # The cited turn does not belong to this session, so
                        # nothing available here can ground the claim.  Refusing
                        # the candidate is the intended outcome rather than a
                        # malformed payload, so the session must still complete.
                        rejected_count += 1
                        continue
                    evidence_turns = [
                        by_reference.get(str(value or "").strip()) for value in references
                    ]
                    if (
                        not evidence_turns
                        or not all(evidence_turns)
                        or str(source["turn_id"])
                        not in {str(turn["turn_id"]) for turn in evidence_turns}
                    ):
                        rejected_count += 1
                        continue
                    quote_bound = all(
                        any(
                            part.strip() in str(turn["user_message"] or "")
                            for turn in evidence_turns
                        )
                        for part in (quote.splitlines() or [quote])
                        if part.strip()
                    )
                    if not quote_bound:
                        source_text = str(source["user_message"] or "")
                        if operation == "SUPERSEDE" and any(
                            marker in source_text for marker in _CORRECTION_MARKERS
                        ):
                            quote = source_text
                        else:
                            # The quote is not verbatim from the patient, which is
                            # how assistant-induced claims are caught.  Refusing
                            # this candidate is the guard working as intended, so
                            # the session must still complete and keep whatever
                            # other candidates were properly grounded.
                            rejected_count += 1
                            continue
                    if operation in {"NOOP", "BLOCK"}:
                        continue
                    sensitivity = str(candidate.get("sensitivity") or "").lower().strip()
                    try:
                        confidence = float(candidate.get("confidence"))
                    except (TypeError, ValueError):
                        confidence = -1.0
                    observed_at = str(candidate.get("observed_at") or "").strip()
                    valid_until = candidate.get("valid_until")
                    try:
                        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                        if valid_until not in (None, ""):
                            datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
                    except ValueError:
                        observed_at = str(source["created_at"] or "").strip()
                        try:
                            datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                        except ValueError:
                            rejected_count += 1
                            safety_failure = True
                            continue
                    if sensitivity not in {"normal", "sensitive", "high"} or not 0 <= confidence <= 1:
                        rejected_count += 1
                        safety_failure = True
                        continue
                    source_text = "\n".join(
                        str(turn["user_message"] or "") for turn in evidence_turns
                    )
                    denies_non_memory_fact = _direct_non_memory_denial(source_text) and not any(
                        marker in source_text for marker in _UNCERTAINTY_MARKERS
                    )
                    if denies_non_memory_fact:
                        rejected_count += 1
                        continue
                    short_ephemeral_turn = (
                        len(source_text) <= 8
                        and any(marker in source_text for marker in _EPHEMERAL_CONTEXT_MARKERS)
                        and not any(marker in source_text for marker in _LONG_TERM_SEMANTIC_MARKERS)
                    )
                    if short_ephemeral_turn:
                        rejected_count += 1
                        continue
                    source_has_uncertainty = any(
                        marker in source_text for marker in _UNCERTAINTY_MARKERS
                    )
                    if source_has_uncertainty and not any(
                        marker in content for marker in _UNCERTAINTY_MARKERS
                    ):
                        content = source_text
                    source_negations = [
                        marker for marker in _NEGATED_NON_MEMORY_TERMS if marker in source_text
                    ]
                    if source_negations and any(marker not in content for marker in source_negations):
                        content = source_text
                    temporary_refusal = (
                        category == "boundaries"
                        and any(marker in source_text for marker in _TEMPORARY_SCOPE_MARKERS)
                        and any(marker in source_text for marker in _TEMPORARY_REFUSAL_MARKERS)
                    )
                    source_has_time_scope = any(marker in source_text for marker in _TEMPORARY_SCOPE_MARKERS) or any(
                        marker in source_text
                        for marker in ("前些年", "十年前", "年轻时", "小时候", "去年")
                    )
                    source_time_markers = [
                        marker for marker in (
                            *_TEMPORARY_SCOPE_MARKERS,
                            "前些年", "十年前", "年轻时", "小时候", "去年",
                        )
                        if marker in source_text
                    ]
                    if source_has_time_scope and any(
                        marker not in content for marker in source_time_markers
                    ):
                        content = source_text
                    source_has_affect = any(marker in source_text for marker in _AFFECTIVE_MARKERS)
                    if source_has_affect and not any(
                        marker in content for marker in _AFFECTIVE_MARKERS
                    ):
                        content = source_text
                    if "住" in source_text and not any(
                        marker in content for marker in ("住", "居住")
                    ):
                        content = source_text
                    if temporary_refusal:
                        has_scope = (
                            any(marker in content for marker in _TEMPORARY_SCOPE_MARKERS)
                            or "前" in content
                            or valid_until not in (None, "")
                        )
                        if not has_scope:
                            rejected_count += 1
                            continue
                    source_turn_id = str(source["turn_id"])
                    evidence_json = [
                        {"turn_id": str(turn["turn_id"])} for turn in evidence_turns
                    ]
                    evidence_json[0]["quote"] = quote
                    item_id = uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"lmca-reflection:{claim['idempotency_key']}:{index}:{content}",
                    ).hex
                    existing = conn.execute(
                        f"SELECT status FROM {_MEMORY_ITEM_TABLE} WHERE item_id=?", (item_id,)
                    ).fetchone()
                    if not existing:
                        self._insert_memory_item(
                            conn,
                            patient_id,
                            category,
                            content,
                            item_id=item_id,
                            source="llm_shadow",
                            source_session_id=session_id,
                            source_turn_id=source_turn_id,
                            status="candidate",
                            slot_key=str(candidate.get("slot_key") or "").strip()[:128] or None,
                            observed_at=observed_at,
                            valid_until=str(valid_until).strip() if valid_until not in (None, "") else None,
                            sensitivity=sensitivity,
                            confidence=confidence,
                            evidence_json=json.dumps(evidence_json, ensure_ascii=False),
                            metadata={
                                "shadow": True,
                                "suggested_operation": operation,
                                "evidence_quote": quote,
                            },
                            supersedes_item_id=(
                                str(candidate.get("supersedes_item_id") or "").strip() or None
                            ),
                        )
                        candidate_item_ids.append(item_id)
                    elif str(existing["status"]) == "candidate":
                        candidate_item_ids.append(item_id)
                    candidate_count += 1
                if safety_failure:
                    raise ValueError("reflection candidate failed safety validation")
            if self._auto_verify_reflections and candidate_item_ids:
                verification = self._verify_candidates(patient_id, candidate_item_ids, turns)
            with self._lock, self._connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if not self._reflection_claim_is_current(
                    conn, patient_id, session_id, lease_token
                ):
                    return {
                        "status": "stale",
                        "patient_id": patient_id,
                        "session_id": session_id,
                    }
                now = _now()
                self._mark_reflection_turns_complete(conn, patient_id, session_id)
                conn.execute(
                    f"""UPDATE {_SESSION_REFLECTION_TABLE}
                        SET status='succeeded', session_summary=?, summary_evidence_json=?,
                            next_retry_at=NULL, last_error=NULL, lease_until=NULL,
                            lease_token=NULL,
                            completed_at=?, updated_at=?
                        WHERE patient_id=? AND session_id=? AND status='running'
                          AND lease_token=?""",
                    (
                        summary_text or None,
                        json.dumps(summary_evidence, ensure_ascii=False) if summary_evidence else None,
                        now,
                        now,
                        patient_id,
                        session_id,
                        lease_token,
                    ),
                )
        except Exception as exc:
            return self._fail_session_reflection(patient_id, session_id, exc, lease_token)
        return {
            "status": "succeeded",
            "patient_id": patient_id,
            "session_id": session_id,
            "candidate_count": candidate_count,
            "rejected_count": rejected_count,
            "verification_status": verification["status"],
            "confirmed_count": verification["confirmed_count"],
            "verification_rejected_count": verification["rejected_count"],
        }

    def run_reflection_task(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a task already claimed by the persistent worker."""
        claimed = dict(task)
        claimed["claimed"] = True
        return self.reflect_session(
            str(claimed.get("patient_id") or ""),
            str(claimed.get("session_id") or ""),
            _claimed_lease=claimed,
        )

    def _memory_worker_loop(self) -> None:
        while not self._memory_worker_stop.is_set():
            try:
                tasks = self.claim_due_reflection_tasks()
                for task in tasks:
                    if self._memory_worker_stop.is_set():
                        break
                    try:
                        self.run_reflection_task(task)
                    except Exception as exc:
                        token = str(task.get("lease_token") or "")
                        self._fail_session_reflection(
                            str(task.get("patient_id") or ""),
                            str(task.get("session_id") or ""),
                            exc,
                            token,
                        )
                self._process_due_mirror_ops()
            except Exception as exc:
                self._log(f"[EmotionMemory] worker error: {type(exc).__name__}")
            self._memory_worker_wakeup.wait(_MEMORY_WORKER_INTERVAL)
            self._memory_worker_wakeup.clear()

    def _process_due_mirror_ops(self) -> None:
        with self._lock, self._connection() as conn:
            patients = conn.execute(
                f"""SELECT DISTINCT patient_id FROM {_MIRROR_OP_TABLE}
                    WHERE (status IN ('pending','failed')
                           AND (next_retry_at IS NULL OR next_retry_at<=?))
                       OR (status='running' AND lease_until<=?) LIMIT 20""",
                (_now(), _now()),
            ).fetchall()
        for row in patients:
            self._process_mirror_ops_impl(str(row["patient_id"]))

    def start_memory_worker(self) -> None:
        with self._memobase_lock:
            if self._memory_worker_thread and self._memory_worker_thread.is_alive():
                return
            self._memory_worker_stop.clear()
            self._memory_worker_thread = threading.Thread(
                target=self._memory_worker_loop, name="emotion-memory-worker", daemon=True
            )
            self._memory_worker_thread.start()

    def stop_memory_worker(self, timeout: float = 5.0) -> None:
        self._memory_worker_stop.set()
        self._memory_worker_wakeup.set()
        thread = self._memory_worker_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, float(timeout)))
        self._memory_worker_thread = None

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
        with self._lock, self._connection() as conn:
            invalidated = self._deleted_turn_ids(conn, patient_id)
        rows_to_mirror = [
            row
            for row in rows
            if not ({str(row["id"]), str(row["turn_id"] or "")} & invalidated)
        ]
        if not rows_to_mirror:
            self._mark_turn_synced(patient_id, int(rows[-1]["id"]))
            return True
        client = self._ensure_memobase()
        if client is None:
            self._mark_sync_failed(patient_id, RuntimeError("memobase unavailable"))
            return False
        try:
            user = self._get_or_create_memobase_user(client, patient_id)
            for row in rows_to_mirror:
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
        self.stop_memory_worker()
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
