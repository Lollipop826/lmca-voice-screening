#!/usr/bin/env python3
"""Regenerate the frozen 112-response 2x2 quality study with oracle labels.

This is a quality-only counterfactual rerun.  It reuses the original ASR text,
memory context, block randomisation and trial metadata, but regenerates every
assistant response with the current downstream prompt.  Emotion-on arms receive
the scenario-prespecified label; emotion-off arms receive ``calm``.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_oracle_emotion_label_pilot import api_call
from src.agents.wellbeing_companion_agent import (
    WellbeingCompanionAgent,
    _emotion_control_block,
)


SOURCE = ROOT / "output/real_voice_ablation_20260914_quality_sensitive"
OUT = ROOT / "output/real_voice_ablation_20260914_quality_oracle"
OLD_LABEL = re.compile(
    r"【内部情绪推测·仅供参考·不得在回复中提及·与对方原话冲突时以原话为准】[^\n]*"
)
LABEL_MAP = {
    "happy": "joy",
    "sad": "sadness",
    "anger": "anger",
    "fear": "fear",
    "anxiety": "anxiety",
    "confusion": "confusion",
    "neutral": "calm",
    "calm": "calm",
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(value: Any) -> str:
    return "".join(char.casefold() for char in str(value or "") if char.isalnum())


def transformed_messages(event: dict[str, Any], label: str) -> list[dict[str, str]]:
    messages = []
    replaced = 0
    for raw in event["messages"]:
        role = "user" if raw.get("role") == "human" else str(raw.get("role"))
        content = str(raw.get("content") or "")
        if role == "system":
            content = WellbeingCompanionAgent._SYSTEM_PROMPT
        elif role == "user":
            content, count = OLD_LABEL.subn(_emotion_control_block(label), content, count=1)
            replaced += count
        messages.append({"role": role, "content": content})
    if replaced != 1:
        raise ValueError(f"expected one emotion block, replaced {replaced}")
    return messages


def generate_one(
    task: dict[str, Any], endpoint: str, api_key: str, model: str
) -> dict[str, Any]:
    content, elapsed_ms = api_call(
        endpoint=endpoint,
        api_key=api_key,
        body={
            "model": model,
            "messages": task["messages"],
            "temperature": 0.45,
            "max_tokens": 240,
        },
        timeout=50,
    )
    return {
        "trial_id": task["trial_id"],
        "sample_id": task["sample_id"],
        "arm": task["arm"],
        "emotion_enabled": task["emotion_enabled"],
        "injected_label": task["injected_label"],
        "target_label": task["target_label"],
        "response": content.strip(),
        "elapsed_ms": elapsed_ms,
        "model": model,
        "completed_at": now(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / "realtime_protocol.json").read_text(encoding="utf-8"))
    samples = {row["sample_id"]: row for row in protocol["samples"]}

    trial_rows = []
    for path in (SOURCE / "realtime_trials").glob("*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        trial_rows.append((row, path))
    trial_rows.sort(key=lambda item: item[0]["started_at"])
    events = sorted(
        [json.loads(line) for line in (SOURCE / "agent_prompt_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()],
        key=lambda row: row["perf_counter"],
    )
    if len(trial_rows) != len(events):
        raise ValueError(f"trial/event count mismatch: {len(trial_rows)} != {len(events)}")

    formal = []
    for (trial, path), event in zip(trial_rows, events):
        if normalize(trial.get("asr_text")) != normalize(event.get("inputs", {}).get("text")):
            raise ValueError(f"trial/event text mismatch for {trial['trial_id']}")
        if trial.get("stage") != "formal":
            continue
        requested = str(samples[trial["sample_id"]].get("synthesis_emotion_requested") or "neutral")
        target = LABEL_MAP.get(requested, "calm")
        enabled = bool(trial["session"]["effective_flags"]["emotion_enabled"])
        injected = target if enabled else "calm"
        formal.append(
            {
                "trial_id": trial["trial_id"],
                "sample_id": trial["sample_id"],
                "arm": trial["arm"],
                "emotion_enabled": enabled,
                "injected_label": injected,
                "target_label": target,
                "source_result": str(path),
                "messages": transformed_messages(event, injected),
            }
        )
    if len(formal) != 112:
        raise ValueError(f"expected 112 formal tasks, found {len(formal)}")

    env = dotenv_values(ROOT / ".env")
    api_key = str(env.get("SILICONFLOW_API_KEY") or "")
    base_url = str(env.get("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1")
    model = str(env.get("SILICONFLOW_MODEL") or "qwen-flash")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY is not configured")
    endpoint = base_url.rstrip("/") + "/chat/completions"

    progress = OUT / "oracle_generations.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["trial_id"]] = row
    pending = [task for task in formal if task["trial_id"] not in completed]
    print(json.dumps({"total": 112, "done": len(completed), "pending": len(pending), "model": model}, ensure_ascii=False), flush=True)
    with progress.open("a", encoding="utf-8", buffering=1) as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(generate_one, task, endpoint, api_key, model): task
                for task in pending
            }
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed[row["trial_id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[generate] {len(completed)}/112 {row['trial_id']} {row['arm']} label={row['injected_label']}", flush=True)

    ordered = [completed[task["trial_id"]] for task in formal]
    write_json(OUT / "oracle_generations.json", ordered)

    for name in ("authoritative_memory_card.txt", "realtime_summary.json"):
        shutil.copy2(SOURCE / name, OUT / name)
    copied_protocol = json.loads(json.dumps(protocol, ensure_ascii=False))
    copied_protocol["quality_response_override"] = {
        "created_at": now(),
        "design": "oracle emotion-label quality-only counterfactual",
        "emotion_on": "scenario-prespecified label injected directly",
        "emotion_off": "calm injected",
        "all_112_responses_regenerated": True,
        "timing_fields": "inherited from source and invalid for regenerated responses; use source study for latency",
        "recognition_claim": "does not validate Emotion2Vec or audio-label accuracy",
    }
    write_json(OUT / "realtime_protocol.json", copied_protocol)

    generated = {row["trial_id"]: row for row in ordered}
    for task in formal:
        source_path = Path(task["source_result"])
        trial = json.loads(source_path.read_text(encoding="utf-8"))
        row = generated[trial["trial_id"]]
        response = row["response"]
        trial["ai_text"] = response
        trial["final_ai_text"] = response
        trial["ai_chunks"] = [response]
        trial["output_text_chars"] = len(response)
        trial["quality_response_provenance"] = {
            "kind": "offline_regeneration_with_oracle_emotion_label",
            "source_trial_id": trial["trial_id"],
            "emotion_enabled": row["emotion_enabled"],
            "injected_label": row["injected_label"],
            "target_label": row["target_label"],
            "model": row["model"],
            "temperature": 0.45,
            "latency_not_recomputed": True,
        }
        write_json(OUT / "realtime_trials" / trial["trial_id"] / "result.json", trial)

    write_json(
        OUT / "oracle_quality_manifest.json",
        {
            "created_at": now(),
            "source": str(SOURCE),
            "output": str(OUT),
            "formal_responses": len(ordered),
            "emotion_on_responses": sum(row["emotion_enabled"] for row in ordered),
            "emotion_off_responses": sum(not row["emotion_enabled"] for row in ordered),
            "labels_are": "scenario-prespecified oracle labels, not human-validated audio annotations",
            "latency": "not measured in this regenerated quality dataset",
        },
    )
    print(json.dumps({"status": "generated", "output": str(OUT), "responses": len(ordered)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
