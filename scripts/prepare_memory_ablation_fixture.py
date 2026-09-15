#!/usr/bin/env python3
"""Prepare a controlled long-term-memory fixture for cross-session ablations.

The voice benchmark itself must remain read-only.  This helper performs the
one-time setup phase: it creates an isolated patient and writes explicit,
active wellbeing memories through the real ``EmotionMemobase`` layer.  The
subsequent A/B benchmark starts *new* sessions for that patient, with memory
writes disabled in both arms.

Example::

    python scripts/prepare_memory_ablation_fixture.py \
      --fixture tests/fixtures/memory_ablation_cases.json \
      --db data/voice_server.db --apply \
      --output output/memory_ablation_fixture.json

The fixture is intentionally separate from the benchmark output.  It records
the ground-truth facts that the LLM judge may see, but never exposes the
memory_on/off condition to the judge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_management.emotion_memobase import EmotionMemobase
from src.db import database


_MEMORY_FIELDS = ("facts", "preferences", "events")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 fixture {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("fixture 根节点必须是 JSON 对象")
    return value


def _text(value: Any, *, field: str, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{field} 不能为空")
    return result


def _normalise_memory_item(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        content = value.strip()
        item: dict[str, Any] = {"text": content}
    elif isinstance(value, Mapping):
        item = dict(value)
        content = str(item.get("text") or item.get("content") or "").strip()
        item["text"] = content
    else:
        raise ValueError(f"{field} 必须是字符串或对象")
    if not content:
        raise ValueError(f"{field} 的记忆内容不能为空")
    item.setdefault("status", "active")
    item.setdefault("source", "memory_ablation_fixture")
    item.setdefault("evidence_turn_ids", [])
    return item


def validate_fixture(raw: Mapping[str, Any]) -> dict[str, Any]:
    patient_id = _text(raw.get("patient_id"), field="patient_id")
    profile = raw.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("profile 必须是对象")
    profile = dict(profile)
    profile["name"] = _text(profile.get("name"), field="profile.name")

    memory = raw.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("memory 必须是对象")
    normalised_memory: dict[str, Any] = {}
    for field in _MEMORY_FIELDS:
        values = memory.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"memory.{field} 必须是数组")
        normalised_memory[field] = [
            _normalise_memory_item(value, field=f"memory.{field}[{index}]")
            for index, value in enumerate(values)
        ]
    normalised_memory["narrative"] = str(memory.get("narrative") or "").strip()
    if not any(normalised_memory[field] for field in _MEMORY_FIELDS) and not normalised_memory["narrative"]:
        raise ValueError("memory 至少要包含一条事实、事件、偏好或 narrative")

    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空数组")
    normalised_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"cases[{index}] 必须是对象")
        sample_id = _text(case.get("sample_id"), field=f"cases[{index}].sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"sample_id 重复: {sample_id}")
        seen_ids.add(sample_id)
        user_text = _text(case.get("user_text"), field=f"cases[{index}].user_text")
        memory_facts = case.get("memory_facts", [])
        if not isinstance(memory_facts, list) or not memory_facts:
            raise ValueError(f"cases[{index}].memory_facts 必须是非空数组")
        normalised_cases.append(
            {
                "sample_id": sample_id,
                "user_text": user_text,
                "memory_facts": [
                    _text(item, field=f"cases[{index}].memory_facts[{j}]")
                    for j, item in enumerate(memory_facts)
                ],
                "scenario": str(case.get("scenario") or "").strip(),
            }
        )

    return {
        "fixture_id": str(raw.get("fixture_id") or patient_id),
        "patient_id": patient_id,
        "profile": profile,
        "memory": normalised_memory,
        "cases": normalised_cases,
    }


def _configure_db(path: str) -> str:
    database.DB_PATH = str(Path(path))
    os.environ["DB_PATH"] = database.DB_PATH
    database.init_db()
    return database.get_db_path()


def _create_patient(patient_id: str, profile: Mapping[str, Any], fixture_id: str) -> None:
    if database.get_patient(patient_id):
        raise ValueError(
            f"患者 {patient_id} 已存在；为避免污染已有数据，请换一个 patient_id"
        )
    now = datetime.now(timezone.utc).isoformat()
    core = {
        key: profile.get(key)
        for key in ("name", "gender", "age", "education_years")
    }
    extra = {
        key: value
        for key, value in profile.items()
        if key not in core and value not in (None, "")
    }
    extra.update(
        {
            "memory_ablation_fixture": fixture_id,
            "memory_ablation_setup": "direct_fixture_seed",
        }
    )
    with database.get_conn() as conn:
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, gender, age, education_years,
                extra_profile, created_at, updated_at, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                patient_id,
                core["name"],
                core.get("gender"),
                core.get("age"),
                core.get("education_years"),
                json.dumps(extra, ensure_ascii=False),
                now,
                now,
            ),
        )


def prepare_fixture(
    fixture: Mapping[str, Any],
    *,
    db_path: str,
    apply: bool,
) -> dict[str, Any]:
    config = validate_fixture(fixture)
    resolved_db = _configure_db(db_path)
    result: dict[str, Any] = {
        "fixture_id": config["fixture_id"],
        "patient_id": config["patient_id"],
        "db_path": resolved_db,
        "applied": bool(apply),
        "setup": "direct_fixture_seed",
        "cross_session_required": True,
        "memory_writes_during_ablation": False,
        "profile": config["profile"],
        "cases": config["cases"],
    }
    if not apply:
        result["memory"] = config["memory"]
        return result

    _create_patient(config["patient_id"], config["profile"], config["fixture_id"])
    memory = EmotionMemobase(
        db_path=resolved_db,
        emotion_classifier=lambda _text: {},
        logger=lambda _message: None,
    )
    current = memory.get_memory_for_user(config["patient_id"])
    updates = dict(config["memory"])
    updates["profile"] = dict(config["profile"])
    updates["expected_revision"] = int(current.get("revision") or 0)
    memory.update_memory_by_user(config["patient_id"], updates)
    items = memory.list_memory_items(config["patient_id"], mode="wellbeing")
    result["memory_item_count"] = len(items)
    result["memory_item_ids"] = [str(item["item_id"]) for item in items]
    result["memory_items"] = [
        {
            "item_id": str(item["item_id"]),
            "category": item.get("category"),
            "content": item.get("content"),
            "status": item.get("status"),
        }
        for item in items
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="为跨会话记忆消融写入隔离患者和可审计的长期记忆 fixture"
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--db", default=os.getenv("DB_PATH", "data/voice_server.db"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际写入数据库；默认只校验并输出 dry-run 计划",
    )
    args = parser.parse_args(argv)
    try:
        result = prepare_fixture(
            _read_json(args.fixture),
            db_path=args.db,
            apply=args.apply,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
