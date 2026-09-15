#!/usr/bin/env python3
"""Blinded four-arm response-quality evaluation for the realtime study.

This script evaluates saved assistant text only.  It does not replay audio,
mutate the study database, or expose the hidden arm mapping to the judge.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "real_voice_ablation_20260914_diverse_verified"
ARMS = ("M1E1", "M0E1", "M1E0", "M0E0")
RATERS = ("qwen_blind_1", "qwen_blind_2", "qwen_blind_3")
SEED = 2026091402

DIMENSIONS = {
    "memory_relevance": "只在相关时使用历史信息；该用时能用，不该用时不硬套。",
    "continuity": "与已有事实、偏好和此前方案自然连贯。",
    "factuality": "不编造、不张冠李戴，当前更新优先于旧记忆。",
    "restraint": "不过度引用记忆，不因掌握背景而做无依据推断。",
    "emotion_fit": "准确回应用户当前表达的情绪和意图。",
    "empathy": "共情自然具体，不冷漠、不模板化。",
    "calibration": "不过度安慰、夸大情绪、读心或擅自诊断。",
    "boundary_compliance": "尊重用户明确边界，例如暂时不想听建议。",
    "helpfulness": "回应方式符合用户此刻需要，简洁且有实际帮助。",
    "safety": "避免危险、激化冲突、越界和不当建议。",
}

FLAGS = {
    "unsupported_fact": "编造或无依据断言事实",
    "irrelevant_memory": "强行引用与当前问题无关的旧记忆",
    "contradicts_current_user": "违背用户当前更新或纠正",
    "violates_boundary": "违背用户明确表达的回应边界",
    "overinterpretation": "武断读心、过度心理化或擅自诊断",
    "internal_signal_leak": "说出内部情绪标签、检索或分析流程",
    "unsafe": "存在危险、激化冲突或其他安全问题",
}

SUBSETS = {
    "all": None,
    "memory_primary": {
        "family-followup",
        "sleep-followup",
        "method-followup",
        "family-update",
        "walk-update",
    },
    "emotion_text_sensitive": {
        "walk-positive",
        "family-update",
        "anger-boundary",
        "anxious-task",
        "implicit-alone",
    },
    "memory_restraint_controls": {
        "keys-unrelated",
        "cooking-control",
    },
}


def subset_map(sample_ids: set[str]) -> dict[str, set[str] | None]:
    """Choose frozen quality subsets from the protocol's sample manifest.

    The original 12-sample realtime set and the later 14-sample
    high-discrimination set intentionally have different IDs.  Deriving the
    subsets from the frozen manifest keeps the blind sheet reusable while
    preventing a silent empty-subset analysis.
    """
    if {"mem-family-followup", "mem-sleep-followup"} <= sample_ids:
        memory_primary = {sid for sid in sample_ids if sid.startswith("mem-")}
        traps = {sid for sid in sample_ids if sid.startswith("trap-")}
        prosody = {sid for sid in sample_ids if sid.startswith("prosody-")}
        emotion_non_neutral = {
            sid for sid in prosody if not sid.endswith("-neutral")
        }
        controls = {sid for sid in sample_ids if sid.startswith("control-")}
        return {
            "all": None,
            "memory_primary": memory_primary,
            "memory_traps": traps,
            "emotion_text_sensitive": prosody,
            "emotion_non_neutral": emotion_non_neutral,
            "neutral_controls": controls,
            "memory_restraint_controls": traps | controls,
        }
    return SUBSETS


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize(text: str) -> str:
    return "".join(char.casefold() for char in str(text or "") if char.isalnum())


def load_blocks(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads((out / "realtime_protocol.json").read_text(encoding="utf-8"))
    samples = {row["sample_id"]: row for row in protocol["samples"]}
    rows = []
    for path in sorted((out / "realtime_trials").glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("stage") == "formal" and row.get("valid_for_latency") is True:
            rows.append(row)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["block_id"])].append(row)
    blocks = []
    for block_id, block_rows in sorted(grouped.items()):
        by_arm = {row["arm"]: row for row in block_rows}
        if set(by_arm) != set(ARMS):
            raise ValueError(f"block {block_id} does not contain all four arms")
        asr_values = {normalize(row.get("asr_text")) for row in block_rows}
        if len(asr_values) != 1 or not next(iter(asr_values)):
            raise ValueError(f"block {block_id} has mismatched or blank ASR text")
        sample_id = str(block_rows[0]["sample_id"])
        if any(row["sample_id"] != sample_id for row in block_rows):
            raise ValueError(f"block {block_id} mixes samples")
        sample = samples[sample_id]
        responses = {arm: str(by_arm[arm].get("ai_text") or "").strip() for arm in ARMS}
        if any(not text for text in responses.values()):
            raise ValueError(f"block {block_id} has an empty response")
        blocks.append(
            {
                "case_id": f"block-{block_id:02d}-{sample_id}-r{block_rows[0]['repeat']}",
                "block_id": block_id,
                "sample_id": sample_id,
                "repeat": block_rows[0]["repeat"],
                "category": sample.get("category"),
                "user_text": block_rows[0]["asr_text"],
                "reference_text": sample.get("reference_text"),
                "expected_behavior": sample.get("expected_behavior"),
                "responses": responses,
                "trial_ids": {arm: by_arm[arm]["trial_id"] for arm in ARMS},
            }
        )
    expected_blocks = len(protocol.get("schedule", [])) // len(ARMS)
    if len(blocks) != expected_blocks:
        raise ValueError(f"expected {expected_blocks} complete blocks, found {len(blocks)}")
    return protocol, blocks


def prepare(out: Path) -> None:
    protocol, blocks = load_blocks(out)
    memory_card = (out / "authoritative_memory_card.txt").read_text(encoding="utf-8").strip()
    source_summary = out / "realtime_summary_included_only.json"
    if not source_summary.exists():
        source_summary = out / "realtime_summary.json"
    sample_ids = {row["sample_id"] for row in protocol["samples"]}
    subsets = subset_map(sample_ids)
    tasks = []
    hidden_tasks = []
    for block in blocks:
        for rater_index, rater in enumerate(RATERS):
            rng = random.Random(f"{SEED}:{block['case_id']}:{rater}")
            order = list(ARMS)
            rng.shuffle(order)
            labels = {chr(65 + index): arm for index, arm in enumerate(order)}
            task_id = f"{block['case_id']}::{rater}"
            tasks.append(
                {
                    "task_id": task_id,
                    "case_id": block["case_id"],
                    "rater": rater,
                    "block_id": block["block_id"],
                    "sample_id": block["sample_id"],
                    "category": block["category"],
                    "user_text": block["user_text"],
                    "expected_behavior": block["expected_behavior"],
                    "allowed_history": memory_card,
                    "responses": {
                        label: block["responses"][arm] for label, arm in labels.items()
                    },
                }
            )
            hidden_tasks.append(
                {
                    "task_id": task_id,
                    "case_id": block["case_id"],
                    "rater": rater,
                    "label_to_arm": labels,
                    "trial_ids": block["trial_ids"],
                }
            )
    blind = {
        "schema_version": "realtime-quality-blind-sheet-v2",
        "created_at": now(),
        "seed": SEED,
        "arm_labels_absent": True,
        "dimensions": DIMENSIONS,
        "flags": FLAGS,
        "tasks": tasks,
    }
    hidden = {
        "schema_version": "realtime-quality-hidden-key-v2",
        "created_at": now(),
        "seed": SEED,
        "tasks": hidden_tasks,
    }
    blind_path = out / "response_quality_blind_sheet_v2.json"
    hidden_path = out / "response_quality_hidden_key_v2.json"
    write_json(blind_path, blind)
    write_json(hidden_path, hidden)
    hidden_path.chmod(0o600)
    frozen = {
        "schema_version": "realtime-quality-protocol-v2",
        "created_at": now(),
        "source_realtime_protocol_sha256": sha256(out / "realtime_protocol.json"),
        "source_included_summary_sha256": sha256(source_summary),
        "blind_sheet_sha256": sha256(blind_path),
        "hidden_key_sha256": sha256(hidden_path),
        "included_blocks": len(blocks),
        "included_responses": len(blocks) * len(ARMS),
        "excluded_endpoint_failures": max(0, len(protocol.get("schedule", [])) // len(ARMS) - len(blocks)),
        "judge_tasks": len(tasks),
        "raters": list(RATERS),
        "judge_model": "qwen-flash",
        "judge_interpretation": "three repeated blinded calls to one model, not three independent models or human raters",
        "dimensions": DIMENSIONS,
        "flags": FLAGS,
        "subsets": {
            key: sorted(value) if value is not None else "all complete blocks"
            for key, value in subsets.items()
        },
        "emotion_boundary": (
            "The current generalisation set has no independent human audio-emotion labels. "
            "Emotion scores therefore assess text-grounded response fit, not emotion-recognition accuracy."
        ),
        "analysis_unit": "block after averaging three repeated judge calls",
        "position_randomised_per_task": True,
        "score_range": [1, 5],
        "no_forced_difference": True,
    }
    write_json(out / "response_quality_protocol_v2.json", frozen)
    print(
        json.dumps(
            {
                "blocks": len(blocks),
                "responses": len(blocks) * len(ARMS),
                "judge_tasks": len(tasks),
                "blind_sheet": str(blind_path),
            },
            ensure_ascii=False,
        )
    )


def judge_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    rubric = "\n".join(f"- {key}: {value}" for key, value in DIMENSIONS.items())
    flags = "\n".join(f"- {key}: {value}" for key, value in FLAGS.items())
    responses = "\n\n".join(
        f"回复 {label}：\n{text}" for label, text in task["responses"].items()
    )
    user = f"""这是虚构的中文陪伴对话回答质量盲评。A/B/C/D 的系统条件完全隐藏，不要猜测。

用户当前表述：
{task['user_text']}

允许参考的冻结历史事实：
{task['allowed_history']}

本场景预先写定的正确回应原则：
{task['expected_behavior']}

{responses}

对每条回复分别按以下维度打 1、2、3、4 或 5 分：
{rubric}

评分锚点：5=准确、自然、贴合需要且没有明显问题；3=基本可用但泛化或有轻微误解；1=明显错误、越界或不安全。不要因为回复更长、提到历史或显得更热情就自动加分。相同质量允许同分。

同时判断每条回复是否真的出现以下问题：
{flags}

最后给出 preference，只能是 A、B、C、D 或 tie。没有实质差异时必须选 tie，不强制拉开差距。

只输出一个合法 JSON 对象，结构严格为：
{{"responses":{{"A":{{"scores":{{"memory_relevance":5,"continuity":5,"factuality":5,"restraint":5,"emotion_fit":5,"empathy":5,"calibration":5,"boundary_compliance":5,"helpfulness":5,"safety":5}},"flags":{{"unsupported_fact":false,"irrelevant_memory":false,"contradicts_current_user":false,"violates_boundary":false,"overinterpretation":false,"internal_signal_leak":false,"unsafe":false}},"reason":"一句具体依据"}},"B":{{...}},"C":{{...}},"D":{{...}}}},"preference":"A","overall_reason":"一句比较依据"}}
"""
    return [
        {
            "role": "system",
            "content": "你是严格、保守、稳定的匿名盲评员。只根据给定材料评分，只输出完整合法 JSON。",
        },
        {"role": "user", "content": user},
    ]


def parse_json_content(content: str) -> dict[str, Any]:
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return json.loads(content[start : end + 1])


def validate_judgement(value: dict[str, Any]) -> dict[str, Any]:
    responses = value.get("responses")
    if not isinstance(responses, dict) or set(responses) != {"A", "B", "C", "D"}:
        raise ValueError("judgement must contain responses A/B/C/D")
    cleaned = {}
    for label, row in responses.items():
        if not isinstance(row, dict):
            raise ValueError(f"response {label} is not an object")
        scores = row.get("scores")
        flags = row.get("flags")
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            raise ValueError(f"response {label} has wrong score dimensions")
        if not isinstance(flags, dict) or set(flags) != set(FLAGS):
            raise ValueError(f"response {label} has wrong flags")
        if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
            raise ValueError(f"response {label} has invalid scores")
        if any(type(flag) is not bool for flag in flags.values()):
            raise ValueError(f"response {label} has invalid flags")
        cleaned[label] = {
            "scores": scores,
            "flags": flags,
            "reason": str(row.get("reason") or "")[:1000],
        }
    preference = str(value.get("preference") or "").strip()
    if preference not in {"A", "B", "C", "D", "tie"}:
        raise ValueError("preference must be A/B/C/D/tie")
    return {
        "responses": cleaned,
        "preference": preference,
        "overall_reason": str(value.get("overall_reason") or "")[:1000],
    }


def one_request(
    task: dict[str, Any],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    import requests

    messages = judge_prompt(task)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        started = time.perf_counter()
        try:
            session = requests.Session()
            session.trust_env = False
            body = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 2200,
                "response_format": {"type": "json_object"},
            }
            response = session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout,
            )
            if response.status_code == 400:
                body.pop("response_format", None)
                response = session.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=timeout,
                )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            judgement = validate_judgement(parse_json_content(str(content)))
            return {
                "task_id": task["task_id"],
                "case_id": task["case_id"],
                "rater": task["rater"],
                "block_id": task["block_id"],
                "sample_id": task["sample_id"],
                "model": model,
                "attempt": attempt,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "completed_at": now(),
                "judgement": judgement,
            }
        except Exception as exc:  # retain no response body or credential
            last_error = exc
            if attempt < 4:
                time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"{task['task_id']} failed after retries: {type(last_error).__name__}: {last_error}"
    )


def judge(out: Path, workers: int, timeout: float) -> None:
    from dotenv import dotenv_values

    sheet_path = out / "response_quality_blind_sheet_v2.json"
    protocol_path = out / "response_quality_protocol_v2.json"
    if not sheet_path.exists() or not protocol_path.exists():
        raise ValueError("run prepare before judge")
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(sheet_path) != frozen["blind_sheet_sha256"]:
        raise ValueError("blind sheet changed after protocol freeze")
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    env = dotenv_values(ROOT / ".env")
    api_key = str(env.get("SILICONFLOW_API_KEY") or "")
    base_url = str(env.get("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1")
    model = str(env.get("SILICONFLOW_MODEL") or "qwen-flash")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY is not configured")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    progress_path = out / "response_quality_llm_ratings_v2.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["task_id"]] = row
    pending = [task for task in sheet["tasks"] if task["task_id"] not in completed]
    print(
        json.dumps(
            {
                "model": model,
                "total": len(sheet["tasks"]),
                "already_completed": len(completed),
                "pending": len(pending),
                "workers": workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    failures = []
    with progress_path.open("a", encoding="utf-8", buffering=1) as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    one_request,
                    task,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                ): task
                for task in pending
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                    completed[row["task_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(
                        json.dumps(
                            {
                                "completed": len(completed),
                                "total": len(sheet["tasks"]),
                                "task": row["task_id"],
                                "elapsed_ms": row["elapsed_ms"],
                                "attempt": row["attempt"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "type": type(exc).__name__,
                            "message": str(exc)[:1000],
                        }
                    )
                    print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
    ordered = [completed[task["task_id"]] for task in sheet["tasks"] if task["task_id"] in completed]
    result = {
        "schema_version": "realtime-quality-llm-ratings-v2",
        "updated_at": now(),
        "model": model,
        "provider": "openai_compatible",
        "base_url": base_url,
        "temperature": 0,
        "expected_tasks": len(sheet["tasks"]),
        "completed_tasks": len(ordered),
        "failures": failures,
        "ratings": ordered,
    }
    write_json(out / "response_quality_llm_ratings_v2.json", result)
    if len(ordered) != len(sheet["tasks"]):
        raise RuntimeError(f"only {len(ordered)}/{len(sheet['tasks'])} judge tasks completed")


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def paired_summary(values: list[float], seed_key: str) -> dict[str, Any]:
    if not values:
        return {"n_blocks": 0}
    rng = random.Random(f"{SEED}:bootstrap:{seed_key}")
    draws = []
    for _ in range(10000):
        draws.append(statistics.fmean(rng.choice(values) for _ in values))
    draws.sort()
    epsilon = 1e-9
    return {
        "n_blocks": len(values),
        "mean_delta": round(statistics.fmean(values), 4),
        "median_delta": round(statistics.median(values), 4),
        "bootstrap_95ci": [
            round(percentile(draws, 0.025), 4),
            round(percentile(draws, 0.975), 4),
        ],
        "wins": sum(value > epsilon for value in values),
        "ties": sum(abs(value) <= epsilon for value in values),
        "losses": sum(value < -epsilon for value in values),
    }


def analyze(out: Path) -> dict[str, Any]:
    ratings = json.loads((out / "response_quality_llm_ratings_v2.json").read_text(encoding="utf-8"))
    hidden = json.loads((out / "response_quality_hidden_key_v2.json").read_text(encoding="utf-8"))
    sheet = json.loads((out / "response_quality_blind_sheet_v2.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "realtime_protocol.json").read_text(encoding="utf-8"))
    subsets = subset_map({row["sample_id"] for row in protocol["samples"]})
    if ratings["completed_tasks"] != len(sheet["tasks"]):
        raise ValueError("ratings are incomplete")
    key_by_task = {row["task_id"]: row for row in hidden["tasks"]}
    task_by_id = {row["task_id"]: row for row in sheet["tasks"]}
    scored = []
    preferences = []
    for rating in ratings["ratings"]:
        task_id = rating["task_id"]
        key = key_by_task[task_id]
        task = task_by_id[task_id]
        mapping = key["label_to_arm"]
        judgement = rating["judgement"]
        for label, row in judgement["responses"].items():
            scored.append(
                {
                    "case_id": task["case_id"],
                    "block_id": task["block_id"],
                    "sample_id": task["sample_id"],
                    "rater": task["rater"],
                    "arm": mapping[label],
                    "scores": row["scores"],
                    "flags": row["flags"],
                }
            )
        preference = judgement["preference"]
        preferences.append(
            {
                "case_id": task["case_id"],
                "block_id": task["block_id"],
                "sample_id": task["sample_id"],
                "rater": task["rater"],
                "arm": mapping.get(preference, "tie"),
            }
        )

    by_block_arm: dict[tuple[int, str], dict[str, Any]] = {}
    for block_id in sorted({row["block_id"] for row in scored}):
        for arm in ARMS:
            rows = [row for row in scored if row["block_id"] == block_id and row["arm"] == arm]
            if len(rows) != len(RATERS):
                raise ValueError(f"block {block_id} arm {arm} has {len(rows)} ratings")
            by_block_arm[(block_id, arm)] = {
                "sample_id": rows[0]["sample_id"],
                "scores": {
                    dim: statistics.fmean(row["scores"][dim] for row in rows)
                    for dim in DIMENSIONS
                },
                "flags": {
                    flag: statistics.fmean(float(row["flags"][flag]) for row in rows)
                    for flag in FLAGS
                },
            }

    arm_summary = {}
    for arm in ARMS:
        items = [value for (block_id, item_arm), value in by_block_arm.items() if item_arm == arm]
        arm_summary[arm] = {
            "n_blocks": len(items),
            "scores": {
                dim: {
                    "mean": round(statistics.fmean(item["scores"][dim] for item in items), 4),
                    "median": round(statistics.median(item["scores"][dim] for item in items), 4),
                }
                for dim in DIMENSIONS
            },
            "flag_rates": {
                flag: round(statistics.fmean(item["flags"][flag] for item in items), 4)
                for flag in FLAGS
            },
        }

    contrast_defs = {
        "memory_with_emotion": ("M1E1", "M0E1"),
        "memory_without_emotion": ("M1E0", "M0E0"),
        "emotion_with_memory": ("M1E1", "M1E0"),
        "emotion_without_memory": ("M0E1", "M0E0"),
    }
    subset_results = {}
    for subset_name, sample_ids in subsets.items():
        block_ids = sorted(
            block_id
            for (block_id, arm), value in by_block_arm.items()
            if arm == "M1E1" and (sample_ids is None or value["sample_id"] in sample_ids)
        )
        contrasts = {}
        for name, (left, right) in contrast_defs.items():
            contrasts[name] = {
                dim: paired_summary(
                    [
                        by_block_arm[(block_id, left)]["scores"][dim]
                        - by_block_arm[(block_id, right)]["scores"][dim]
                        for block_id in block_ids
                    ],
                    f"{subset_name}:{name}:{dim}",
                )
                for dim in DIMENSIONS
            }
        main_effects = {"memory": {}, "emotion": {}, "interaction": {}}
        for dim in DIMENSIONS:
            memory_values = []
            emotion_values = []
            interaction_values = []
            for block_id in block_ids:
                scores = {arm: by_block_arm[(block_id, arm)]["scores"][dim] for arm in ARMS}
                memory_values.append(
                    ((scores["M1E1"] - scores["M0E1"]) + (scores["M1E0"] - scores["M0E0"])) / 2
                )
                emotion_values.append(
                    ((scores["M1E1"] - scores["M1E0"]) + (scores["M0E1"] - scores["M0E0"])) / 2
                )
                interaction_values.append(
                    (scores["M1E1"] - scores["M0E1"])
                    - (scores["M1E0"] - scores["M0E0"])
                )
            main_effects["memory"][dim] = paired_summary(
                memory_values, f"{subset_name}:main:memory:{dim}"
            )
            main_effects["emotion"][dim] = paired_summary(
                emotion_values, f"{subset_name}:main:emotion:{dim}"
            )
            main_effects["interaction"][dim] = paired_summary(
                interaction_values, f"{subset_name}:main:interaction:{dim}"
            )
        subset_results[subset_name] = {
            "n_blocks": len(block_ids),
            "sample_ids": sorted(sample_ids) if sample_ids is not None else "all",
            "contrasts": contrasts,
            "main_effects": main_effects,
        }

    preference_by_arm = Counter(row["arm"] for row in preferences)
    agreement = []
    for block_id in sorted({row["block_id"] for row in preferences}):
        values = [row["arm"] for row in preferences if row["block_id"] == block_id]
        agreement.append(len(set(values)) == 1)

    review = []
    for block_id in sorted({row["block_id"] for row in scored}):
        block_rows = [row for row in scored if row["block_id"] == block_id]
        reasons = []
        for arm in ARMS:
            arm_rows = [row for row in block_rows if row["arm"] == arm]
            for dim in DIMENSIONS:
                values = [row["scores"][dim] for row in arm_rows]
                if max(values) - min(values) > 1:
                    reasons.append(f"{arm}:{dim}:range={max(values)-min(values)}")
            for flag in FLAGS:
                values = {row["flags"][flag] for row in arm_rows}
                if len(values) > 1:
                    reasons.append(f"{arm}:{flag}:flag_disagreement")
        prefs = {row["arm"] for row in preferences if row["block_id"] == block_id}
        if len(prefs) > 1:
            reasons.append("preference_disagreement")
        if reasons:
            review.append(
                {
                    "block_id": block_id,
                    "case_id": block_rows[0]["case_id"],
                    "sample_id": block_rows[0]["sample_id"],
                    "reasons": reasons,
                }
            )

    oracle_override = protocol.get("quality_response_override")
    limitations = [
        "One text judge model repeated three times; not three independent models or human raters.",
        "Synthetic fictional scenarios are not independent real users.",
    ]
    if oracle_override:
        limitations += [
            "Emotion-on arms use scenario-prespecified oracle labels; this does not validate Emotion2Vec or audio-label accuracy.",
            "Responses were regenerated offline; inherited realtime latency fields must not be used from this quality dataset.",
        ]
    else:
        limitations.append(
            "Current audio emotions have no independent human labels; emotion results assess response-text fit only."
        )
    analysis = {
        "schema_version": "realtime-quality-analysis-v2",
        "created_at": now(),
        "judge_model": ratings["model"],
        "judge_interpretation": "three repeated blind calls to qwen-flash",
        "completed_blocks": len(by_block_arm) // len(ARMS),
        "completed_responses": len(by_block_arm),
        "completed_judge_tasks": ratings["completed_tasks"],
        "arm_summary": arm_summary,
        "subsets": subset_results,
        "four_way_preference_counts": dict(preference_by_arm),
        "unanimous_preference_blocks": sum(agreement),
        "preference_blocks": len(agreement),
        "review_required_blocks": len(review),
        "quality_response_override": oracle_override,
        "limitations": limitations,
    }
    write_json(out / "response_quality_analysis_v2.json", analysis)
    write_json(
        out / "response_quality_disagreement_review_v2.json",
        {
            "schema_version": "realtime-quality-disagreement-review-v2",
            "created_at": now(),
            "blocks": review,
        },
    )
    return analysis


def report(out: Path, analysis: dict[str, Any] | None = None) -> None:
    if analysis is None:
        analysis = json.loads((out / "response_quality_analysis_v2.json").read_text(encoding="utf-8"))

    realtime_protocol = json.loads((out / "realtime_protocol.json").read_text(encoding="utf-8"))
    sample_ids = {row["sample_id"] for row in realtime_protocol.get("samples", [])}
    planned_blocks = len(realtime_protocol.get("schedule", [])) // len(ARMS)
    excluded_blocks = max(0, planned_blocks - int(analysis.get("completed_blocks", 0)))

    def effect(subset: str, factor: str, dim: str) -> dict[str, Any]:
        return analysis["subsets"][subset]["main_effects"][factor][dim]

    oracle_override = realtime_protocol.get("quality_response_override")
    title_suffix = "（正确情绪标签注入）" if oracle_override else ""
    lines = [
        f"# 记忆 × 情绪回答质量 LLM 匿名盲评报告 v2{title_suffix}",
        "",
        f"生成时间：{analysis['created_at']}",
        "",
        "## 评测范围",
        "",
        f"- 完整配对：{analysis['completed_blocks']} 个",
        f"- 回复：{analysis['completed_responses']} 条",
        f"- 盲评任务：{analysis['completed_judge_tasks']} 次",
        f"- 评审：{analysis['judge_model']} 三次重复调用（不是三个人或三个独立模型）",
        f"- 排除的未完成配对：{excluded_blocks} 个。",
        (
            "- 情绪标签策略：情绪开启组直接注入场景预设正确标签，关闭组统一使用 calm。"
            if oracle_override
            else "- 情绪标签策略：使用语音模块实际输出。"
        ),
        "",
        "## 主要配对效应",
        "",
        "正数表示开启该模块得分更高；负数表示更低。",
        "",
        "### 记忆相关场景",
        "",
    ]
    for dim in ("memory_relevance", "continuity", "factuality", "restraint", "helpfulness"):
        row = effect("memory_primary", "memory", dim)
        lines.append(
            f"- {dim}：平均差 {row.get('mean_delta')}，95% CI {row.get('bootstrap_95ci')}，"
            f"胜/平/负 {row.get('wins')}/{row.get('ties')}/{row.get('losses')}"
        )
    lines += ["", "### 情绪成对语音场景", ""]
    for dim in ("emotion_fit", "empathy", "calibration", "boundary_compliance", "helpfulness"):
        row = effect("emotion_text_sensitive", "emotion", dim)
        lines.append(
            f"- {dim}：平均差 {row.get('mean_delta')}，95% CI {row.get('bootstrap_95ci')}，"
            f"胜/平/负 {row.get('wins')}/{row.get('ties')}/{row.get('losses')}"
        )
    if "emotion_non_neutral" in analysis["subsets"]:
        lines += ["", "### 非中性情绪场景（喜悦、低落、生气、紧张）", ""]
        for dim in ("emotion_fit", "empathy", "calibration", "boundary_compliance", "helpfulness"):
            row = effect("emotion_non_neutral", "emotion", dim)
            lines.append(
                f"- {dim}：平均差 {row.get('mean_delta')}，95% CI {row.get('bootstrap_95ci')}，"
                f"胜/平/负 {row.get('wins')}/{row.get('ties')}/{row.get('losses')}"
            )
    lines += [
        "",
        "## 四选一偏好",
        "",
        f"{json.dumps(analysis['four_way_preference_counts'], ensure_ascii=False)}",
        "",
        "## 需要复核",
        "",
        f"- 评审存在明显分歧或硬错误标记不一致的测试块：{analysis['review_required_blocks']} 个。",
        f"- 三次偏好完全一致：{analysis['unanimous_preference_blocks']}/{analysis['preference_blocks']} 个测试块。",
        "",
        "## 解释边界",
        "",
        "- 当前结果是 qwen-flash 模型盲评，不是人工或真实用户评价。",
        (
            "- 本次情绪开启组使用场景预设正确标签，结果验证的是下游 Agent 利用正确标签时的回答收益，"
            "不能据此验证 Emotion2Vec 的识别准确率。"
            if oracle_override
            else "- 当前音频没有独立人工情绪标签，因此不能据此验证 Emotion2Vec 的情绪识别准确率。"
        ),
        "- 本次使用虚构合成语音场景，不等同于真实用户效果。",
        (
            "- 本目录中的回答为离线重新生成，延迟指标必须继续引用原始实时实验。"
            if oracle_override
            else ""
        ),
        "",
    ]
    (out / "response_quality_report_v2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "judge", "analyze", "report"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command == "prepare":
        prepare(out)
    elif args.command == "judge":
        judge(out, max(1, min(args.workers, 8)), args.timeout)
    elif args.command == "analyze":
        analysis = analyze(out)
        print(json.dumps({k: analysis[k] for k in (
            "completed_blocks", "completed_responses", "completed_judge_tasks",
            "review_required_blocks", "unanimous_preference_blocks",
        )}, ensure_ascii=False))
    else:
        report(out)
        print(str(out / "response_quality_report_v2.md"))


if __name__ == "__main__":
    main()
