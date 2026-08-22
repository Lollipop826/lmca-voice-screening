"""Validate real Qwen reflection through the voice session end/disconnect paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.context_management.emotion_memobase import EmotionMemobase
from src.voice.services import PatientMemoryService, VoiceConnectionCleanup
from src.voice.session import VoiceSession


def _service(memory: EmotionMemobase) -> PatientMemoryService:
    return PatientMemoryService(
        get_patient=lambda _patient_id: None,
        create_patient=lambda _profile: {"patient_id": "pt-simulated"},
        update_patient_profile=lambda *_args: None,
        link_session_patient=lambda *_args: None,
        long_term_memory=memory,
        logger=lambda _message: None,
    )


def _wait_for_reflection(memory: EmotionMemobase, session_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = memory.get_session_reflection("pt-simulated", session_id)
        if state["status"] in {"succeeded", "failed"}:
            return state
        time.sleep(0.1)
    return memory.get_session_reflection("pt-simulated", session_id)


def _shadow_counts(memory: EmotionMemobase) -> dict[str, int]:
    items = memory.list_memory_items("pt-simulated", include_deleted=True)
    shadow = [item for item in items if item.get("source") == "llm_shadow"]
    return {
        "shadow_candidates": sum(item.get("status") == "candidate" for item in shadow),
        "shadow_active": sum(item.get("status") == "active" for item in shadow),
    }


def run(model: str) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    os.environ["MEMORY_REFLECTION_MODEL"] = model
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("SILICONFLOW_API_KEY")):
        raise RuntimeError("未找到 Qwen API key")

    with tempfile.TemporaryDirectory(
        prefix="q4-voice-",
        dir=ROOT / ".planning" / "2026-08-17-memory-llm-validation",
    ) as temp_dir:
        explicit_id = "q4-explicit-end"
        explicit = EmotionMemobase(
            str(Path(temp_dir) / "explicit.db"),
            emotion_classifier=lambda _text: {"neutral": 1.0},
            logger=lambda _message: None,
        )
        explicit.capture_turn(
            "pt-simulated",
            "我喜欢在阳台晒太阳，心里会安静一些。",
            "我听到了。",
            session_id=explicit_id,
            turn_id="q4-explicit-turn-0",
        )
        explicit_session = VoiceSession(
            connection=object(), agent=object(), owner_username="synthetic"
        )
        explicit_session.bind_session(explicit_id)
        explicit_session.lifecycle.current_patient_id = "pt-simulated"
        explicit_returned = _service(explicit).consolidate(explicit_session)
        explicit_state = _wait_for_reflection(explicit, explicit_id)
        explicit_replay = explicit.reflect_session("pt-simulated", explicit_id)

        disconnect_id = "q4-disconnect"
        disconnect = EmotionMemobase(
            str(Path(temp_dir) / "disconnect.db"),
            emotion_classifier=lambda _text: {"neutral": 1.0},
            logger=lambda _message: None,
        )
        disconnect.capture_turn(
            "pt-simulated",
            "女儿周末陪我下棋，我很开心。",
            "听起来很温暖。",
            session_id=disconnect_id,
            turn_id="q4-disconnect-turn-0",
        )
        disconnect_session = VoiceSession(
            connection=object(), agent=object(), owner_username="synthetic"
        )
        disconnect_session.bind_session(disconnect_id)
        disconnect_session.lifecycle.current_patient_id = "pt-simulated"
        cleanup = VoiceConnectionCleanup(
            disconnect_session,
            answer_completion=SimpleNamespace(cancel_window=lambda: None),
            patient_memory_service=_service(disconnect),
            logger=lambda _message: None,
        )
        started = time.monotonic()
        asyncio.run(cleanup.close("synthetic-disconnect"))
        close_elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        disconnect_state = _wait_for_reflection(disconnect, disconnect_id)

        return {
            "model": model,
            "real_patient_data_used": False,
            "auto_active": False,
            "local_fallback": False,
            "explicit_end": {
                "submit_returned": explicit_returned,
                "status": explicit_state.get("status"),
                "attempt_count": explicit_state.get("attempt_count"),
                "replay_idempotent": explicit_replay.get("idempotent_replay", False),
                **_shadow_counts(explicit),
            },
            "disconnect": {
                "close_returned_ms": close_elapsed_ms,
                "status": disconnect_state.get("status"),
                "attempt_count": disconnect_state.get("attempt_count"),
                **_shadow_counts(disconnect),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 Qwen 语音入口 shadow 验收")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args.model)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(
        report[path]["status"] == "succeeded"
        and report[path]["attempt_count"] == 1
        and report[path]["shadow_active"] == 0
        for path in ("explicit_end", "disconnect")
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
