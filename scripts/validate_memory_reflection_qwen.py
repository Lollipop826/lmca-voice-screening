"""Run the simulated memory oracle through a compatible chat provider."""

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
    comparable_content = content.replace("不太确定", "不确定").replace("也说不准", "不确定")
    for requirement in case.get("content_all_of") or []:
        choices = requirement if isinstance(requirement, list) else [requirement]
        if not any(str(choice) in comparable_content for choice in choices) and not (
            item.get("valid_until")
            and all(str(choice) in _TIME_SCOPE_WORDS for choice in choices)
        ):
            return False
    for requirement in case.get("content_any_of") or []:
        choices = requirement if isinstance(requirement, list) else [requirement]
        if not any(str(choice) in comparable_content for choice in choices):
            return False
    for forbidden in case.get("content_none_of") or []:
        choices = forbidden if isinstance(forbidden, list) else [forbidden]
        if any(str(choice) in comparable_content for choice in choices):
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
    if any(not evidence_is_bound(case, item) for item in items):
        failures.append("evidence_not_bound_to_current_user_turn")
    active = [item for item in items if item.get("status") == "active"]
    pending = [item for item in items if item.get("status") == "candidate"]
    eligible = [item for item in items if item.get("status") in {None, "", "active", "candidate"}]
    expected = case["expect"]
    if expected in {"none", "rejected"}:
        if active or pending:
            failures.append("unexpected_usable_memory")
    elif expected == "candidate_or_none":
        if eligible:
            matches = [
                item for item in eligible
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
            item for item in eligible
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
        if case.get("require_valid_until") and not any(
            str(item.get("valid_until") or "").strip() for item in matches
        ):
            failures.append("temporary_scope_missing_valid_until")
        if expected == "active" and not any(item.get("status") == "active" for item in matches):
            failures.append("expected_active_memory_not_found")
        operations = case.get("required_operations")
        if operations and not any(item.get("suggested_operation") in operations for item in matches):
            failures.append("required_operation_not_found")
        maximum = case.get("max_matching_items")
        if maximum is not None and len(matches) > int(maximum):
            failures.append("duplicate_matching_memory")
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
        "source": item.get("source"),
        "confirmed_by": item.get("confirmed_by"),
        "item_id": item.get("item_id"),
        "supersedes_item_id": item.get("supersedes_item_id"),
    }


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    os.environ.pop("ARK_API_KEY", None)
    os.environ["MEMORY_REFLECTION_MODEL"] = args.generation_model
    os.environ["MEMORY_VERIFICATION_MODEL"] = args.verification_model
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("SILICONFLOW_API_KEY")):
        raise RuntimeError("未找到长期记忆模型 API key")

    original_invoke = EmotionMemobase._invoke_memory_model
    model_calls: list[dict[str, Any]] = []

    def capture_invoke(prompt: str, model: str, max_tokens: int) -> Any:
        started = time.perf_counter()
        try:
            response = original_invoke(prompt, model, max_tokens)
        except Exception as exc:
            model_calls.append({
                "role": "reflector" if max_tokens == 1600 else "verifier",
                "model": model,
                "max_tokens": max_tokens,
                "prompt": prompt,
                "raw_response": None,
                "error": repr(exc),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            raise
        model_calls.append({
            "role": "reflector" if max_tokens == 1600 else "verifier",
            "model": model,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "raw_response": str(response),
            "error": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return response

    EmotionMemobase._invoke_memory_model = staticmethod(capture_invoke)

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
                call_start = len(model_calls)
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
                reflection_retries = 0
                first_status = result.get("status")
                while result.get("status") == "failed" and reflection_retries < args.retries:
                    reflection_retries += 1
                    result = memory.reflect_session(
                        "pt-simulated", session_id, force=True
                    )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                latencies.append(elapsed_ms)
                items = [
                    _safe_item(item)
                    for item in memory.list_memory_items("pt-simulated", include_deleted=True)
                    if item.get("source_session_id") == session_id
                ]
                verdict, failures = evaluate_case(case, result, items)
                reflection_state = memory.get_session_reflection("pt-simulated", session_id)
                if case.get("expect_supersede"):
                    existing_active = any(
                        item.get("content") == case["existing"] and item.get("status") == "active"
                        for item in memory.list_memory_items("pt-simulated", include_deleted=True)
                    )
                    if existing_active:
                        failures.append("existing_memory_not_superseded")
                for item in items:
                    valid_until = str(item.get("valid_until") or "").strip()
                    observed_at = str(item.get("observed_at") or "").strip()
                    if valid_until and observed_at and valid_until <= observed_at:
                        failures.append("valid_until_not_after_observed_at")
                        break
                verdict = "pass" if not failures else "fail"
                case_result = {
                    "id": case["id"],
                    "repeat": repeat_index + 1,
                    "kind": case["kind"],
                    "expected": case["expect"],
                    "status": result.get("status"),
                    "first_status": first_status,
                    "error_code": reflection_state.get("error_code"),
                    "last_error": reflection_state.get("last_error"),
                    "reflection_retries": reflection_retries,
                    "candidate_count": result.get("candidate_count", 0),
                    "rejected_count": result.get("rejected_count", 0),
                    "active_count": sum(item["status"] == "active" for item in items),
                    "blocked_count": sum(item["status"] == "blocked" for item in items),
                    "pending_candidate_count": sum(item["status"] == "candidate" for item in items),
                    "elapsed_ms": elapsed_ms,
                    "verdict": verdict,
                    "failures": failures,
                    "items": items,
                    "model_calls": model_calls[call_start:],
                }
                case_results.append(case_result)
                print(json.dumps(case_result, ensure_ascii=False), flush=True)

    failed = [case for case in case_results if case["verdict"] == "fail"]
    return {
        "generation_model": args.generation_model,
        "verification_model": args.verification_model,
        "repeat_count": args.repeat,
        "provider": "dashscope" if os.getenv("DASHSCOPE_API_KEY") else "qwen_openai_compatible",
        "provider_success": sum(case["status"] == "succeeded" for case in case_results),
        "provider_total": len(case_results),
        "reflection_retries": sum(case["reflection_retries"] for case in case_results),
        "passed": len(case_results) - len(failed),
        "failed": len(failed),
        "active_count": sum(case["active_count"] for case in case_results),
        "blocked_count": sum(case["blocked_count"] for case in case_results),
        "pending_candidate_count": sum(case["pending_candidate_count"] for case in case_results),
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
        "capture": "full_prompt_and_raw_response_per_model_call",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="使用真实兼容 API 验收长期记忆抽取")
    parser.add_argument("--model", dest="legacy_generation_model", help="候选生成模型（兼容旧命令）")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--verification-model", default="qwen-plus")
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="模型调用失败后对同一 SQLite 会话执行的 force 重试次数",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    args.generation_model = args.generation_model or args.legacy_generation_model or "qwen3.8-max"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0 if report["overall_verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
