"""Run the simulated memory oracle through the real Qwen provider."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context_management.emotion_memobase import EmotionMemobase


DEFAULT_ORACLE = ROOT / "tests" / "fixtures" / "memory_reflection_oracle.json"
_TIME_SCOPE_WORDS = {
    "现在", "今天", "今晚", "这次", "暂时", "这几天", "这两天", "本周",
    "近期", "这阵子", "过几天", "几天后", "明天", "下周", "下月", "下个月",
    "以后", "之后", "前些年", "十年前", "年轻时", "小时候", "去年", "目前",
}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def candidate_matches(case: dict[str, Any], item: dict[str, Any]) -> bool:
    categories = case.get("allowed_categories") or [case["category"]]
    if item.get("category") not in categories:
        return False
    operations = case.get("allowed_operations")
    if operations and item.get("suggested_operation") not in operations:
        return False
    content = str(item.get("content") or "")
    for requirement in case.get("content_all_of") or []:
        choices = requirement if isinstance(requirement, list) else [requirement]
        if not any(str(choice) in content for choice in choices) and not (
            item.get("valid_until")
            and all(str(choice) in _TIME_SCOPE_WORDS for choice in choices)
        ):
            return False
    for requirement in case.get("content_any_of") or []:
        choices = requirement if isinstance(requirement, list) else [requirement]
        if not any(str(choice) in content for choice in choices):
            return False
    for forbidden in case.get("content_none_of") or []:
        choices = forbidden if isinstance(forbidden, list) else [forbidden]
        if any(str(choice) in content for choice in choices):
            return False
    return True


def evidence_is_bound(case: dict[str, Any], item: dict[str, Any]) -> bool:
    turn_ids = {
        f"{case['id']}-turn-{index}" for index in range(len(case.get("turns") or []))
    }
    source_turn_id = str(item.get("source_turn_id") or "").strip()
    evidence_turn_ids = {
        str(value or "").strip() for value in item.get("evidence_turn_ids") or []
    }
    quote = str(item.get("evidence_quote") or "").strip()
    user_messages = [str(turn[0] or "") for turn in case.get("turns") or []]
    return bool(
        source_turn_id
        and source_turn_id in turn_ids
        and evidence_turn_ids
        and evidence_turn_ids <= turn_ids
        and source_turn_id in evidence_turn_ids
        and quote
        and all(
            any(part.strip() in message for message in user_messages)
            for part in (quote.splitlines() or [quote])
            if part.strip()
        )
    )


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    failures = []
    if result.get("status") != "succeeded":
        failures.append("provider_or_session_failed")
    if any(item.get("status") == "active" for item in items):
        failures.append("shadow_item_became_active")
    if any(not evidence_is_bound(case, item) for item in items):
        failures.append("evidence_not_bound_to_current_user_turn")
    expected = case["expect"]
    if expected in {"none", "rejected"}:
        if items:
            failures.append("unexpected_candidate")
    elif expected == "candidate_or_none":
        if items:
            matches = [
                item for item in items
                if evidence_is_bound(case, item) and candidate_matches(case, item)
            ]
            if not matches:
                failures.append("unsafe_temporary_candidate")
            elif case.get("require_valid_until") and any(
                not str(item.get("valid_until") or "").strip()
                and not any(
                    marker in str(item.get("content") or "")
                    for marker in ("现在", "今天", "今晚", "这周", "这两天", "这阵子", "近期", "暂时", "下周", "下月", "待会")
                )
                for item in matches
            ):
                failures.append("temporary_scope_missing_valid_until")
    else:
        matches = [
            item for item in items
            if evidence_is_bound(case, item) and candidate_matches(case, item)
        ]
        required_evidence_count = len(set(case.get("evidence_indexes") or [0]))
        if not matches:
            failures.append("expected_candidate_not_found")
        elif not any(
            len(set(item.get("evidence_turn_ids") or [])) >= required_evidence_count
            for item in matches
        ):
            failures.append("multi_turn_evidence_incomplete")
    return ("pass" if not failures else "fail", failures)


def _safe_item(item: dict[str, Any]) -> dict[str, Any]:
    metadata = _json_object(item.get("metadata_json"))
    evidence = _json_list(item.get("evidence_json"))
    return {
        "category": item.get("category"),
        "content": item.get("content"),
        "status": item.get("status"),
        "source_turn_id": item.get("source_turn_id"),
        "suggested_operation": metadata.get("suggested_operation"),
        "evidence_quote": metadata.get("evidence_quote"),
        "evidence_turn_ids": [entry.get("turn_id") for entry in evidence if isinstance(entry, dict)],
        "observed_at": item.get("observed_at"),
        "valid_until": item.get("valid_until"),
        "sensitivity": item.get("sensitivity"),
    }


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    os.environ.pop("ARK_API_KEY", None)
    os.environ["MEMORY_REFLECTION_MODEL"] = args.model
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("SILICONFLOW_API_KEY")):
        raise RuntimeError("未找到 Qwen API key")

    cases = json.loads(args.oracle.read_text(encoding="utf-8"))
    if args.ids:
        requested = set(args.ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = sorted(requested - {case["id"] for case in cases})
        if missing:
            raise ValueError("未知 oracle id: " + ", ".join(missing))

    case_results = []
    latencies = []
    with tempfile.TemporaryDirectory(prefix="memory-qwen-", dir=args.output.parent) as temp_dir:
        for repeat_index in range(args.repeat):
            for case in cases:
                session_id = f"oracle-{case['id']}-repeat-{repeat_index + 1}"
                memory = EmotionMemobase(
                    str(Path(temp_dir) / f"{case['id']}-repeat-{repeat_index + 1}.db"),
                    emotion_classifier=lambda _text: {"neutral": 1.0},
                    logger=lambda _message: None,
                )
                for index, (user_message, assistant_message) in enumerate(case["turns"]):
                    memory.capture_turn(
                        "pt-simulated",
                        user_message,
                        assistant_message,
                        session_id=session_id,
                        turn_id=f"{case['id']}-turn-{index}",
                    )
                if case.get("existing"):
                    field = case["category"] if case["category"] in {"facts", "preferences", "events"} else "facts"
                    memory.update_memory_by_user("pt-simulated", {field: [case["existing"]]})

                started = time.perf_counter()
                result = memory.reflect_session("pt-simulated", session_id)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                latencies.append(elapsed_ms)
                items = [
                    _safe_item(item)
                    for item in memory.list_memory_items("pt-simulated", include_deleted=True)
                    if item.get("source") == "llm_shadow"
                ]
                verdict, failures = evaluate_case(case, result, items)
                case_result = {
                    "id": case["id"],
                    "repeat": repeat_index + 1,
                    "kind": case["kind"],
                    "expected": case["expect"],
                    "status": result.get("status"),
                    "candidate_count": result.get("candidate_count", 0),
                    "rejected_count": result.get("rejected_count", 0),
                    "active_shadow": sum(item["status"] == "active" for item in items),
                    "elapsed_ms": elapsed_ms,
                    "verdict": verdict,
                    "failures": failures,
                    "items": items,
                }
                case_results.append(case_result)
                print(json.dumps(case_result, ensure_ascii=False), flush=True)

    failed = [case for case in case_results if case["verdict"] == "fail"]
    return {
        "model": args.model,
        "repeat_count": args.repeat,
        "provider": "dashscope" if os.getenv("DASHSCOPE_API_KEY") else "qwen_openai_compatible",
        "provider_success": sum(case["status"] == "succeeded" for case in case_results),
        "provider_total": len(case_results),
        "passed": len(case_results) - len(failed),
        "failed": len(failed),
        "active_shadow": sum(case["active_shadow"] for case in case_results),
        "elapsed_ms": {
            "min": round(min(latencies), 2) if latencies else None,
            "p50": round(_percentile(latencies, 0.50), 2) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "overall_verdict": "pass" if not failed else "fail",
        "api_key_recorded": False,
        "real_patient_data_used": False,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="使用真实 Qwen 验收长期记忆抽取")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0 if report["overall_verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
