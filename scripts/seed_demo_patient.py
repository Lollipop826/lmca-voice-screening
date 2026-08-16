from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_management.emotion_memobase import EmotionMemobase
from src.db import database


DEMO_PATIENT_ID = "DEMO-work-stress-sleep"
DEMO_PROFILE = {
    "name": "DEMO-工作压力与睡眠困扰",
    "gender": "未说明",
    "age": 34,
    "education_years": 16,
    "demo": True,
    "scenario": "工作压力与睡眠困扰",
}
DEMO_MEMORY = {
    "events": [
        {
            "id": "demo-event-work-deadline",
            "text": "最近项目赶工，常在午夜后才停止工作。",
            "source": "demo",
            "status": "active",
            "evidence_turn_ids": [],
        }
    ],
    "preferences": [
        {
            "id": "demo-pref-phone-away",
            "text": "睡前反复看工作消息会让入睡更困难；愿意尝试把手机放远一点。",
            "source": "demo",
            "status": "active",
            "evidence_turn_ids": [],
        }
    ],
    "narrative": "DEMO 虚构患者：工作压力与睡眠困扰；仅用于演示长期记忆读取、引用和删除。",
}


def _configure_db(path: str) -> str:
    database.DB_PATH = str(Path(path))
    os.environ["DB_PATH"] = database.DB_PATH
    return database.get_db_path()


def _upsert_demo_patient() -> None:
    now = datetime.now().isoformat()
    extra = {
        "demo": True,
        "scenario": DEMO_PROFILE["scenario"],
        "seed_source": "scripts/seed_demo_patient.py",
    }
    with database.get_conn() as conn:
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, gender, age, education_years,
                extra_profile, created_at, updated_at, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
               ON CONFLICT(patient_id) DO UPDATE SET
                 name=excluded.name,
                 gender=excluded.gender,
                 age=excluded.age,
                 education_years=excluded.education_years,
                 extra_profile=excluded.extra_profile,
                 updated_at=excluded.updated_at,
                 archived_at=NULL""",
            (
                DEMO_PATIENT_ID,
                DEMO_PROFILE["name"],
                DEMO_PROFILE["gender"],
                DEMO_PROFILE["age"],
                DEMO_PROFILE["education_years"],
                json.dumps(extra, ensure_ascii=False),
                now,
                now,
            ),
        )


def _assign_owner(username: str) -> bool:
    username = str(username or "").strip()
    if not username or not database.get_user_by_username(username):
        return False
    database.assign_patient(
        DEMO_PATIENT_ID,
        username,
        can_read=True,
        can_write=True,
        can_voice=True,
        assigned_by="seed_demo_patient",
    )
    return True


def seed_demo_patient(*, db_path: str, owner: str = "") -> dict:
    db_path = _configure_db(db_path)
    database.init_db()
    _upsert_demo_patient()
    assigned = _assign_owner(owner)

    memory = EmotionMemobase(
        db_path=db_path,
        emotion_classifier=lambda _text: {},
        logger=lambda _message: None,
    )
    current = memory.get_memory_for_user(DEMO_PATIENT_ID)
    memory.update_memory_by_user(
        DEMO_PATIENT_ID,
        {
            "profile": dict(DEMO_PROFILE),
            **DEMO_MEMORY,
            "expected_revision": int(current.get("revision") or 0),
        },
    )
    items = memory.list_memory_items(DEMO_PATIENT_ID, mode="wellbeing")
    return {
        "db_path": db_path,
        "patient_id": DEMO_PATIENT_ID,
        "owner_assigned": assigned,
        "memory_item_count": len(items),
        "memory_item_ids": [item["item_id"] for item in items],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed the isolated DEMO wellbeing patient through the real SQLite memory layer."
    )
    parser.add_argument("--db", default="data/voice_server.db")
    parser.add_argument("--owner", default="", help="Optional existing username to assign the DEMO patient to.")
    parser.add_argument("--apply", action="store_true", help="Write the DEMO patient and memories.")
    args = parser.parse_args(argv)

    planned = {
        "db_path": str(Path(args.db)),
        "patient_id": DEMO_PATIENT_ID,
        "profile": DEMO_PROFILE,
        "memory": DEMO_MEMORY,
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps({"dry_run": True, **planned}, ensure_ascii=False, indent=2))
        return 0

    result = seed_demo_patient(db_path=args.db, owner=args.owner)
    print(json.dumps({"dry_run": False, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
