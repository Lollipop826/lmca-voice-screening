"""Build and verify deterministic long-term-memory scale fixtures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path


SCALES = (0, 10, 100, 500, 1000, 5000)
PATIENT_ID = "pt_memory_latency_scale"
MEMOBASE_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, f"lmca-memobase:{PATIENT_ID}"))
PROJECT_ID = "__root__"
HIT_QUERY = "你好呀，我今天心情可能不太好。"
MISS_QUERY = "量子引力对火星玄武岩同位素的影响"

_MEMORIES = (
    ("events", "工作", "项目临近交付时连续加班，晚上仍反复想着未完成的任务"),
    ("facts", "睡眠", "压力增大时入睡会变慢，夜里容易醒来两次"),
    ("preferences", "陪伴", "心情不好时更愿意先被安静倾听，再讨论具体办法"),
    ("events", "家庭", "和家人通话后感到被支持，孤独感有所减轻"),
    ("facts", "运动", "傍晚散步二十分钟后身体会放松一些"),
    ("preferences", "饮食", "胃口差的时候更容易接受清淡温热的食物"),
    ("events", "就医", "按预约完成复诊，并记录了最近一周的睡眠变化"),
    ("facts", "社交", "人多嘈杂的聚会会带来紧张和疲惫"),
    ("preferences", "兴趣", "听舒缓音乐和整理阳台植物有助于稳定情绪"),
    ("events", "生活", "完成一件拖延已久的小事后感到轻松和踏实"),
)
_EMOTIONS = ("焦虑", "平静", "难过", "开心", "困惑", "担心", "放松")
_SCORES = {
    "焦虑": "anxiety", "平静": "calm", "难过": "sadness", "开心": "joy",
    "困惑": "confusion", "担心": "fear", "放松": "calm",
}


def _target(value: str | int) -> int:
    value = int(value)
    if value not in SCALES:
        raise argparse.ArgumentTypeError(f"target must be one of {SCALES}")
    return value


def memory_record(index: int) -> dict:
    category, topic, detail = _MEMORIES[index % len(_MEMORIES)]
    emotion = _EMOTIONS[index % len(_EMOTIONS)]
    day = datetime(2024, 1, 1) + timedelta(days=index % 730)
    session = index // 10 + 1
    text = (
        f"[基准记忆{index + 1:05d}] {day:%Y-%m-%d} 第{session:04d}次历史会话，"
        f"主题为{topic}：患者提到{detail}；当时主要感受是{emotion}。"
    )
    return {
        "id": f"memory-scale-{index + 1:05d}",
        "category": category,
        "topic": topic,
        "emotion": emotion,
        "text": text,
        "session_id": f"memory-history-{session:04d}",
        "turn_id": f"memory-turn-{index + 1:05d}",
        "created_at": day.replace(hour=10, minute=index % 60).isoformat(),
    }


def _configure_local(db_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.db import database

    database.DB_PATH = str(Path(db_path).resolve())
    os.environ["DB_PATH"] = database.DB_PATH


def reset_local(db_path: str) -> dict:
    path = Path(db_path).resolve()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    return {"db_path": str(path), "reset": True}


def init_local(db_path: str, username: str, password: str) -> dict:
    _configure_local(db_path)
    from src.context_management.emotion_memobase import EmotionMemobase
    from src.db import database

    database.init_db()
    EmotionMemobase(
        db_path=database.get_db_path(),
        memobase_client=False,
        emotion_classifier=lambda _text: {},
        logger=lambda _msg: None,
    )
    now = datetime.now().isoformat()
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    with database.get_conn() as conn:
        conn.execute(
            """INSERT INTO users (username,password_hash,salt,display_name,role,created_at)
               VALUES (?,?,?,?,?,?) ON CONFLICT(username) DO UPDATE SET
               password_hash=excluded.password_hash,salt=excluded.salt""",
            (username, digest.hex(), salt.hex(), "Memory latency benchmark", "user", now),
        )
        conn.execute(
            """INSERT INTO patients
               (patient_id,name,gender,age,education_years,extra_profile,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(patient_id) DO UPDATE SET
               name=excluded.name,updated_at=excluded.updated_at,archived_at=NULL""",
            (PATIENT_ID, "长期记忆规模基准患者", "未说明", 45, 12,
             json.dumps({"synthetic_benchmark": True}, ensure_ascii=False), now, now),
        )
    database.assign_patient(
        PATIENT_ID, username, can_read=True, can_write=True, can_voice=True,
        assigned_by="memory_scale_fixture",
    )
    return {"db_path": database.get_db_path(), "patient_id": PATIENT_ID, "username": username}


def _emotion_payload(label: str) -> tuple[dict, str, float, float]:
    dominant = _SCORES[label]
    scores = {key: 0.02 for key in ("joy", "sadness", "anger", "fear", "anxiety", "calm", "confusion")}
    scores[dominant] = 0.88
    total = sum(scores.values())
    scores = {key: round(value / total, 6) for key, value in scores.items()}
    valence = scores["joy"] + scores["calm"] * 0.35 - scores["sadness"] - scores["anger"] - scores["fear"] - scores["anxiety"]
    arousal = scores["anger"] + scores["fear"] + scores["anxiety"] + scores["joy"] * 0.4
    return scores, dominant, valence, arousal


def grow_local(db_path: str, target: int) -> dict:
    _configure_local(db_path)
    from src.context_management.emotion_memobase import EmotionMemobase

    memory = EmotionMemobase(db_path=db_path, emotion_classifier=lambda _text: {}, logger=lambda _msg: None)
    with memory._lock, memory._connection() as conn:
        current = int(conn.execute(
            "SELECT COUNT(*) FROM emotion_memobase_turns WHERE patient_id=?", (PATIENT_ID,),
        ).fetchone()[0])
        if current > target:
            raise RuntimeError(f"local fixture already has {current} turns, target is {target}")
        for index in range(current, target):
            item = memory_record(index)
            scores, dominant, valence, arousal = _emotion_payload(item["emotion"])
            conn.execute(
                """INSERT INTO emotion_memobase_turns
                   (patient_id,session_id,turn_id,turn_state,response_status,user_message,
                    assistant_message,emotion_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (PATIENT_ID, item["session_id"], item["turn_id"], "DURABLY_CAPTURED", "responded",
                 item["text"], "我记住了这段经历，之后可以继续从这里聊。",
                 json.dumps(scores, ensure_ascii=False), item["created_at"]),
            )
            conn.execute(
                """INSERT INTO emotion_trajectory
                   (patient_id,session_id,turn_id,analysis_status,turn_index,timestamp,
                    joy,sadness,anger,fear,anxiety,calm,confusion,valence,arousal,
                    dominant_emotion,trigger_content,audio_path,source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (PATIENT_ID, item["session_id"], item["turn_id"], "final", index + 1,
                 item["created_at"], scores["joy"], scores["sadness"], scores["anger"],
                 scores["fear"], scores["anxiety"], scores["calm"], scores["confusion"],
                 valence, arousal, dominant, item["text"][:200], None, "synthetic_benchmark"),
            )
    fields = {key: [] for key in ("facts", "preferences", "events")}
    for index in range(target):
        item = memory_record(index)
        fields[item["category"]].append({
            "id": item["id"], "text": item["text"], "source": "synthetic_benchmark",
            "status": "active", "source_session_id": item["session_id"],
            "source_turn_id": item["turn_id"], "created_at": item["created_at"],
        })
    memory.update_snapshot(
        PATIENT_ID,
        {**fields, "narrative": f"合成长时记忆规模基准，当前包含{target}条可检索历史记忆。"},
        updated_by="memory_scale_fixture",
    )
    return verify_local(db_path, target)


def verify_local(db_path: str, target: int) -> dict:
    con = sqlite3.connect(str(Path(db_path).resolve()))
    counts = {
        "turns": con.execute("SELECT COUNT(*) FROM emotion_memobase_turns WHERE patient_id=?", (PATIENT_ID,)).fetchone()[0],
        "memory_items": con.execute("SELECT COUNT(*) FROM emotion_memobase_memory_items WHERE patient_id=? AND status='active'", (PATIENT_ID,)).fetchone()[0],
        "emotion_trajectory": con.execute("SELECT COUNT(*) FROM emotion_trajectory WHERE patient_id=?", (PATIENT_ID,)).fetchone()[0],
    }
    con.close()
    if any(value != target for value in counts.values()):
        raise RuntimeError(f"local fixture count mismatch: expected={target}, actual={counts}")
    return {"patient_id": PATIENT_ID, "target": target, "local": counts}


def _remote_counts() -> dict:
    from memobase_server.connectors import Session
    from memobase_server.models.database import UserEvent, UserEventGist

    with Session() as session:
        events = session.query(UserEvent).filter_by(user_id=MEMOBASE_USER_ID, project_id=PROJECT_ID).all()
        gists = session.query(UserEventGist).filter_by(user_id=MEMOBASE_USER_ID, project_id=PROJECT_ID).all()
        dimensions = sorted({len(item.embedding) for item in gists if item.embedding is not None})
        return {
            "events": len(events),
            "event_embeddings": sum(item.embedding is not None for item in events),
            "gists": len(gists),
            "gist_embeddings": sum(item.embedding is not None for item in gists),
            "embedding_dimensions": dimensions,
        }


async def _ensure_remote_user() -> None:
    from memobase_server.controllers import user as user_controller
    from memobase_server.connectors import Session
    from memobase_server.models.database import User
    from memobase_server.models.response import UserData

    with Session() as session:
        exists = session.query(User).filter_by(id=MEMOBASE_USER_ID, project_id=PROJECT_ID).one_or_none()
    if exists is None:
        result = await user_controller.create_user(
            UserData(id=MEMOBASE_USER_ID, data={"patient_id": PATIENT_ID, "synthetic_benchmark": True}),
            PROJECT_ID,
        )
        if not result.ok():
            raise RuntimeError(result.msg())


async def reset_memobase() -> dict:
    from memobase_server.controllers import user as user_controller
    from memobase_server.connectors import Session
    from memobase_server.models.database import User

    with Session() as session:
        exists = session.query(User).filter_by(id=MEMOBASE_USER_ID, project_id=PROJECT_ID).one_or_none()
    if exists is not None:
        result = await user_controller.delete_user(MEMOBASE_USER_ID, PROJECT_ID)
        if not result.ok():
            raise RuntimeError(result.msg())
    await _ensure_remote_user()
    return {"patient_id": PATIENT_ID, "memobase_user_id": MEMOBASE_USER_ID, **_remote_counts()}


async def grow_memobase(target: int, batch_size: int, retries: int) -> dict:
    if not 1 <= batch_size <= 20:
        raise ValueError("batch_size must be between 1 and 20 for qwen3.7-text-embedding")
    from memobase_server.controllers import event as event_controller
    from memobase_server.connectors import Session
    from memobase_server.models.database import UserEvent, UserEventGist

    await _ensure_remote_user()
    counts = _remote_counts()
    if counts["gists"] > target:
        raise RuntimeError(f"Memobase already has {counts['gists']} gists, target is {target}")
    if counts["event_embeddings"] != counts["events"] or counts["gist_embeddings"] != counts["gists"]:
        raise RuntimeError(f"existing NULL embedding detected: {counts}")
    started = time.perf_counter()
    while counts["gists"] < target:
        start = counts["gists"]
        stop = min(target, start + batch_size)
        event_data = {
            "profile_delta": [],
            "event_tip": "\n".join(f"- {memory_record(index)['text']}" for index in range(start, stop)),
            "event_tags": [{"tag": "fixture", "value": "memory_scale"}],
            "session_id": f"seed-{start + 1:05d}-{stop:05d}",
            "turn_id": f"seed-turn-{stop:05d}",
            "source_turn_id": f"seed-source-{stop:05d}",
        }
        for attempt in range(1, retries + 1):
            result = await event_controller.append_user_event(MEMOBASE_USER_ID, PROJECT_ID, event_data)
            if not result.ok():
                if attempt == retries:
                    raise RuntimeError(result.msg())
                await asyncio.sleep(attempt)
                continue
            event_id = str(result.data())
            with Session() as session:
                event = session.query(UserEvent).filter_by(id=event_id, project_id=PROJECT_ID).one()
                gists = session.query(UserEventGist).filter_by(event_id=event_id, project_id=PROJECT_ID).all()
                valid = event.embedding is not None and len(gists) == stop - start and all(g.embedding is not None for g in gists)
            if valid:
                break
            deleted = await event_controller.delete_user_event(MEMOBASE_USER_ID, PROJECT_ID, event_id)
            if not deleted.ok() or attempt == retries:
                raise RuntimeError(f"batch {start}:{stop} produced NULL/missing embeddings")
            await asyncio.sleep(attempt)
        counts = _remote_counts()
        if counts["gists"] != stop:
            raise RuntimeError(f"unexpected gist count after batch: expected={stop}, actual={counts}")
        if stop % 100 == 0 or stop == target:
            print(f"seeded_gists={stop}/{target}", flush=True)
    return {"target": target, "build_ms": round((time.perf_counter() - started) * 1000, 2), **verify_memobase(target)}


def verify_memobase(target: int) -> dict:
    from memobase_server.connectors import Session

    counts = _remote_counts()
    if counts["gists"] != target or counts["gist_embeddings"] != target:
        raise RuntimeError(f"Memobase gist count mismatch: expected={target}, actual={counts}")
    if counts["event_embeddings"] != counts["events"]:
        raise RuntimeError(f"Memobase event embedding mismatch: {counts}")
    expected_dimensions = [] if target == 0 else [1024]
    if counts["embedding_dimensions"] != expected_dimensions:
        raise RuntimeError(f"embedding dimension mismatch: {counts}")
    with Session() as session:
        index = session.execute(__import__("sqlalchemy").text(
            "SELECT indexdef FROM pg_indexes WHERE tablename='user_event_gists' "
            "AND indexname='idx_user_event_gists_embedding_hnsw_cosine'"
        )).scalar()
    if not index or "USING hnsw" not in index:
        raise RuntimeError("event gist HNSW cosine index is missing")
    return {"memobase": counts, "hnsw_index": "idx_user_event_gists_embedding_hnsw_cosine"}


async def search_memobase(target: int) -> dict:
    from memobase_server.controllers import event_gist

    started = time.perf_counter()
    hit = await event_gist.search_user_event_gists(MEMOBASE_USER_ID, PROJECT_ID, HIT_QUERY, 3, 0.35, None)
    hit_ms = (time.perf_counter() - started) * 1000
    if not hit.ok():
        raise RuntimeError(hit.msg())
    started = time.perf_counter()
    miss = await event_gist.search_user_event_gists(MEMOBASE_USER_ID, PROJECT_ID, MISS_QUERY, 3, 0.99, None)
    miss_ms = (time.perf_counter() - started) * 1000
    if not miss.ok():
        raise RuntimeError(miss.msg())
    hits = hit.data().gists
    misses = miss.data().gists
    if target and not hits:
        raise RuntimeError("expected hit query returned no event gists")
    if not target and hits:
        raise RuntimeError("empty fixture unexpectedly returned event gists")
    if misses:
        raise RuntimeError("miss query unexpectedly passed 0.99 similarity threshold")
    return {
        "target": target,
        "hit": {"count": len(hits), "total_ms": round(hit_ms, 2), "top_similarity": round(float(hits[0].similarity), 6) if hits else None},
        "miss": {"count": 0, "total_ms": round(miss_ms, 2)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reset-local", "init-local", "grow-local", "verify-local"):
        command = sub.add_parser(name)
        command.add_argument("--db", required=True)
        if name in {"grow-local", "verify-local"}:
            command.add_argument("--target", required=True, type=_target)
        if name == "init-local":
            command.add_argument("--username", required=True)
            command.add_argument("--password", required=True)
    sub.add_parser("reset-memobase")
    grow = sub.add_parser("grow-memobase")
    grow.add_argument("--target", required=True, type=_target)
    grow.add_argument("--batch-size", type=int, default=20)
    grow.add_argument("--retries", type=int, default=4)
    verify = sub.add_parser("verify-memobase")
    verify.add_argument("--target", required=True, type=_target)
    search = sub.add_parser("search-memobase")
    search.add_argument("--target", required=True, type=_target)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "reset-local":
        result = reset_local(args.db)
    elif args.command == "init-local":
        result = init_local(args.db, args.username, args.password)
    elif args.command == "grow-local":
        result = grow_local(args.db, args.target)
    elif args.command == "verify-local":
        result = verify_local(args.db, args.target)
    elif args.command == "reset-memobase":
        result = asyncio.run(reset_memobase())
    elif args.command == "grow-memobase":
        result = asyncio.run(grow_memobase(args.target, args.batch_size, args.retries))
    elif args.command == "verify-memobase":
        result = verify_memobase(args.target)
    else:
        result = asyncio.run(search_memobase(args.target))
    print("MEMORY_SCALE_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
