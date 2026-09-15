#!/usr/bin/env python3
"""Prepare, LLM-judge, and analyse blinded paired response-quality evaluations.

The latency benchmark already produces paired ``memory_on/memory_off`` (or
``emotion_on/emotion_off``) responses.  This script turns those responses into
a blind rating sheet and then analyses completed 1--5 ratings.  The condition
key is deliberately written to a separate file so the evaluator cannot infer
which response contains the ablated component.

Examples
--------
Create a blind sheet and keep its key private::

    python scripts/evaluate_response_quality.py prepare \
      --input output/ablation_memory_strict_5pairs.json \
      --output output/memory_quality_ratings.json \
      --key-output output/memory_quality_key.json \
      --seed 20260913

Run an LLM judge over the blind sheet and analyse it::

    python scripts/evaluate_response_quality.py judge \
      --ratings output/memory_quality_ratings.json \
      --output output/memory_quality_llm_ratings.json \
      --judge-raters llm_judge_1,llm_judge_2,llm_judge_3

    python scripts/evaluate_response_quality.py analyze \
      --ratings output/memory_quality_llm_ratings.json \
      --key output/memory_quality_key.json \
      --output output/memory_quality_result.json

The judge never receives the condition key.  Multiple independent passes can
be used to reduce position and sampling variance; the final analysis remains
paired by case and maps A/B back to the hidden on/off condition only after
scoring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
SCORE_MIN = 1
SCORE_MAX = 5

# Which finalization sources count as a real streaming turn. ``soulx_final``
# came first because the early benchmarks always ran with SoulX turn-taking.
# The isolated 2x2 ablation deliberately runs on local Silero VAD, where the
# stream finalizes as ``final``; rejecting that would discard every trial from
# a protocol-conformant run, so both are accepted.
ACCEPTED_ASR_SOURCES = frozenset({"soulx_final", "final"})

# Evidence that the memory-on arm actually had memory available.
# ``memory_retrieval_hit`` only becomes true when Memobase semantic retrieval
# returns something. With Memobase down, the authoritative SQLite card is still
# injected and the arms remain cleanly separated by ``memory_retrieval_source``
# (``realtime`` vs ``disabled``), which is the manipulation this ablation
# actually varies.
MEMORY_ON_RETRIEVAL_SOURCES = frozenset({"realtime", "memobase"})

COMMON_RUBRIC: dict[str, str] = {
    "helpfulness": "这条回复是否真正帮助了患者处理当前问题？",
    "appropriateness": "语气、长度和回应方式是否适合当前场景？",
    "safety": "是否避免了危险、武断、越界或不当的建议？",
}

MEMORY_RUBRIC: dict[str, str] = {
    "relevance": "是否使用了与当前问题相关的历史信息？",
    "personalization": "是否体现了患者自身背景，而不是泛泛回答？",
    "continuity": "是否与此前对话自然连贯？",
    "factuality": "是否准确使用记忆，没有编造、张冠李戴或使用过期信息？",
    "restraint": "在记忆无关时，是否避免强行提及记忆？",
    **COMMON_RUBRIC,
}

EMOTION_RUBRIC: dict[str, str] = {
    "emotion_fit": "是否正确理解并回应了说话人的情绪？",
    "empathy": "是否体现出合适的共情，而不是冷漠或敷衍？",
    "calibration": "是否避免过度安慰、误判或夸大情绪？",
    **COMMON_RUBRIC,
}


class QualityEvaluationError(ValueError):
    """Input data is not safe to use for a paired quality evaluation."""


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityEvaluationError(f"无法读取 JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityEvaluationError(f"JSON 根节点必须是对象: {source}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rubric_for(evaluation_type: str) -> dict[str, str]:
    if evaluation_type == "memory":
        return dict(MEMORY_RUBRIC)
    if evaluation_type == "emotion":
        return dict(EMOTION_RUBRIC)
    raise QualityEvaluationError(
        f"不支持的消融类型 {evaluation_type!r}，只能是 memory 或 emotion"
    )


def _infer_evaluation_type(data: Mapping[str, Any]) -> str:
    configured = data.get("configuration")
    if isinstance(configured, Mapping) and configured.get("ablation") in {
        "memory",
        "emotion",
    }:
        return str(configured["ablation"])
    conditions = {
        str(record.get("condition"))
        for record in data.get("records", [])
        if isinstance(record, Mapping)
    }
    if {"memory_on", "memory_off"}.issubset(conditions):
        return "memory"
    if {"emotion_on", "emotion_off"}.issubset(conditions):
        return "emotion"
    raise QualityEvaluationError("无法从 benchmark 结果判断消融类型")


def _records_by_pair(
    data: Mapping[str, Any],
    evaluation_type: str,
) -> list[dict[str, Any]]:
    records = data.get("records")
    if not isinstance(records, list):
        raise QualityEvaluationError("benchmark JSON 缺少 records 数组")

    expected = {f"{evaluation_type}_on", f"{evaluation_type}_off"}
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        condition = str(record.get("condition") or "")
        if condition not in expected:
            continue
        pair_index = record.get("pair_index")
        if pair_index is None:
            raise QualityEvaluationError("质量评测要求每条记录有 pair_index")
        pair = grouped.setdefault(str(pair_index), {})
        if condition in pair:
            raise QualityEvaluationError(
                f"pair_index={pair_index} 出现重复条件 {condition}"
            )
        pair[condition] = record

    complete: list[dict[str, Any]] = []
    for pair_index, pair in sorted(grouped.items(), key=lambda item: int(item[0])):
        if set(pair) != expected:
            continue
        on = pair[f"{evaluation_type}_on"]
        off = pair[f"{evaluation_type}_off"]
        invalid_reason = ""
        for record in (on, off):
            if not record.get("valid_for_latency"):
                invalid_reason = "不是有效单轮"
                break
            if int(record.get("turn_count") or 0) != 1:
                invalid_reason = "被切成多轮"
                break
            if not str(record.get("assistant_text") or "").strip():
                invalid_reason = "缺少 assistant_text"
                break
            if str(record.get("asr_source") or "") not in ACCEPTED_ASR_SOURCES:
                invalid_reason = "ASR 终局来源不在白名单内"
                break
        if invalid_reason:
            # Voice benchmarks may contain otherwise useful pairs whose VAD
            # split the same audio into multiple turns.  They must not abort
            # the whole quality evaluation; simply exclude the pair and let
            # the final count expose how many strict pairs survived.
            continue
        if on.get("sample_id") != off.get("sample_id"):
            continue
        if evaluation_type == "memory":
            memory_available = on.get("memory_retrieval_hit") is True or str(
                on.get("memory_retrieval_source") or ""
            ) in MEMORY_ON_RETRIEVAL_SOURCES
            if not memory_available:
                continue
            if str(off.get("memory_retrieval_source") or "disabled") != "disabled":
                # The off arm must genuinely have had memory withheld, otherwise
                # the pair measures nothing.
                continue
            if on.get("memory_writes_enabled") or off.get("memory_writes_enabled"):
                continue
        elif evaluation_type == "emotion":
            if (
                on.get("emotion_source") != "emotion2vec_audio+text"
                or on.get("emotion_audio_model_used") is not True
            ):
                continue
            if off.get("emotion_source") != "disabled":
                continue
        complete.append(
            {
                "pair_index": pair_index,
                "sample_id": str(on.get("sample_id") or ""),
                "prompt": str(on.get("recognized_text") or ""),
                "on": dict(on),
                "off": dict(off),
            }
        )
    if not complete:
        raise QualityEvaluationError("没有可用于盲评的完整配对")
    return complete


def _rating_template(raters: Iterable[str], dimensions: Iterable[str]) -> dict[str, Any]:
    return {
        rater: {
            "a": {dimension: None for dimension in dimensions},
            "b": {dimension: None for dimension in dimensions},
            "preference": None,
            "notes": "",
        }
        for rater in raters
    }


def prepare_blind_sheet(
    benchmark: Mapping[str, Any],
    *,
    seed: int,
    raters: list[str],
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_type = _infer_evaluation_type(benchmark)
    rubric = rubric_for(evaluation_type)
    pairs = _records_by_pair(benchmark, evaluation_type)
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    key_cases: list[dict[str, Any]] = []

    context_cases_raw = context.get("cases", {}) if isinstance(context, Mapping) else {}
    if isinstance(context_cases_raw, Mapping):
        context_cases = context_cases_raw
    elif isinstance(context_cases_raw, list):
        # The memory-ablation fixture uses a list so it can be fed directly
        # to this command.  Convert it to the sample_id-keyed form used by
        # the blind sheet without copying the hidden A/B condition.
        context_cases = {}
        for index, item in enumerate(context_cases_raw):
            if not isinstance(item, Mapping):
                raise QualityEvaluationError(
                    f"context JSON 的 cases[{index}] 必须是对象"
                )
            sample_id = str(item.get("sample_id") or "").strip()
            if not sample_id:
                raise QualityEvaluationError(
                    f"context JSON 的 cases[{index}] 缺少 sample_id"
                )
            context_cases[sample_id] = {
                key: item[key]
                for key in ("memory_facts", "scenario", "user_text")
                if key in item
            }
    else:
        raise QualityEvaluationError("context JSON 的 cases 必须是对象或数组")

    for pair in pairs:
        case_id = f"{evaluation_type}-{pair['pair_index']}-{pair['sample_id']}"
        labels = [("on", pair["on"]), ("off", pair["off"])]
        rng.shuffle(labels)
        label_a, record_a = labels[0]
        label_b, record_b = labels[1]
        case_context = context_cases.get(case_id) or context_cases.get(
            pair["sample_id"], {}
        )
        if case_context is None:
            case_context = {}
        if not isinstance(case_context, Mapping):
            raise QualityEvaluationError(f"context {case_id} 必须是对象")

        cases.append(
            {
                "case_id": case_id,
                "prompt": pair["prompt"],
                "context": dict(case_context),
                "response_a": str(record_a["assistant_text"]),
                "response_b": str(record_b["assistant_text"]),
                "ratings": _rating_template(raters, rubric),
            }
        )
        key_cases.append(
            {
                "case_id": case_id,
                "pair_index": pair["pair_index"],
                "sample_id": pair["sample_id"],
                "label_a": f"{evaluation_type}_{label_a}",
                "label_b": f"{evaluation_type}_{label_b}",
                "audio_sha256": record_a.get("audio_sha256"),
                "asr_cer": {
                    "a": record_a.get("asr_cer"),
                    "b": record_b.get("asr_cer"),
                },
            }
        )

    sheet = {
        "schema_version": SCHEMA_VERSION,
        "kind": "blinded_response_quality_ratings",
        "evaluation_type": evaluation_type,
        "seed": seed,
        "instructions": [
            "不要猜测 A/B 对应哪个系统条件。",
            "评分模型应先独立阅读问题和两条回复，再按 1（很差）到 5（很好）评分。",
            "如果无法判断，使用 preference=unclear 并在 notes 中说明。",
        ],
        "rubric": [
            {"id": dimension, "question": question}
            for dimension, question in rubric.items()
        ],
        "raters": raters,
        "cases": cases,
    }
    key = {
        "schema_version": SCHEMA_VERSION,
        "kind": "blinded_response_quality_key",
        "evaluation_type": evaluation_type,
        "seed": seed,
        "cases": key_cases,
    }
    return sheet, key


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityEvaluationError(f"{label} 必须是 1 到 5 的数字")
    number = float(value)
    if not math.isfinite(number) or not SCORE_MIN <= number <= SCORE_MAX:
        raise QualityEvaluationError(f"{label} 必须在 1 到 5 之间")
    return number


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _exact_sign_flip_p(values: list[float]) -> float | None:
    values = [value for value in values if value != 0]
    if not values:
        return None
    observed = abs(sum(values))
    if len(values) <= 20:
        extreme = 0
        total = 1 << len(values)
        for mask in range(total):
            candidate = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            )
            if abs(candidate) >= observed - 1e-12:
                extreme += 1
        return round(extreme / total, 6)
    rng = random.Random(0)
    samples = 20000
    extreme = 0
    for _ in range(samples):
        candidate = sum(value if rng.getrandbits(1) else -value for value in values)
        extreme += abs(candidate) >= observed
    return round(extreme / samples, 6)


def _validate_key(key: Mapping[str, Any], sheet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if key.get("kind") != "blinded_response_quality_key":
        raise QualityEvaluationError("key 文件类型不正确")
    if key.get("evaluation_type") != sheet.get("evaluation_type"):
        raise QualityEvaluationError("ratings 与 key 的 evaluation_type 不一致")
    cases = key.get("cases")
    if not isinstance(cases, list):
        raise QualityEvaluationError("key 缺少 cases 数组")
    result: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping) or not case.get("case_id"):
            raise QualityEvaluationError("key 中存在无效 case")
        case_id = str(case["case_id"])
        if case_id in result:
            raise QualityEvaluationError(f"key 重复 case_id={case_id}")
        if {case.get("label_a"), case.get("label_b")} != {
            f"{sheet['evaluation_type']}_on",
            f"{sheet['evaluation_type']}_off",
        }:
            raise QualityEvaluationError(f"key case={case_id} 条件映射不完整")
        result[case_id] = case
    return result


def analyse_ratings(
    sheet: Mapping[str, Any],
    key: Mapping[str, Any],
) -> dict[str, Any]:
    if sheet.get("kind") != "blinded_response_quality_ratings":
        raise QualityEvaluationError("ratings 文件类型不正确")
    rubric_rows = sheet.get("rubric")
    raters = sheet.get("raters")
    cases = sheet.get("cases")
    if not isinstance(rubric_rows, list) or not isinstance(raters, list) or not isinstance(cases, list):
        raise QualityEvaluationError("ratings 缺少 rubric、raters 或 cases")
    dimensions = [str(row.get("id")) for row in rubric_rows if isinstance(row, Mapping)]
    if not dimensions or any(not dimension for dimension in dimensions):
        raise QualityEvaluationError("ratings rubric 无有效维度")
    key_by_case = _validate_key(key, sheet)

    per_dimension: dict[str, dict[str, Any]] = {}
    all_deltas: dict[str, list[float]] = {dimension: [] for dimension in dimensions}
    preference_counts = {"on": 0, "off": 0, "tie": 0, "unclear": 0}
    completed_cases = 0
    completed_ratings = 0
    # Scores remain paired by case so we can report a lightweight inter-rater
    # agreement check without adding a statistics dependency.
    rater_scores: dict[str, dict[str, list[tuple[float, float]]]] = {}
    # Deltas grouped by dimension and then by case. Significance testing needs
    # the case as its randomization unit: several raters scoring the same case
    # are repeated measurements of one sample, not independent observations, so
    # feeding all of them to the sign flip test inflates the apparent sample
    # size and understates the p value.
    case_deltas: dict[str, dict[str, list[float]]] = {
        dimension: {} for dimension in dimensions
    }
    # Deltas grouped by the originating audio sample rather than by case.
    # Re-running the same audio produces several cases that share a sample_id,
    # and those repeats are still repeated measurements of one stimulus. The
    # case-level unit already removes rater duplication but would count a
    # 3-scenario run repeated twice as 6 independent samples, so the sample is
    # the conservative randomization unit for the headline test.
    sample_deltas: dict[str, dict[str, list[float]]] = {
        dimension: {} for dimension in dimensions
    }
    # Sample ids behind the cases that actually produced a complete rating, so
    # the report can contrast independent samples against the larger case count.
    distinct_sample_ids: set[str] = set()

    for case in cases:
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("case_id") or "")
        mapping = key_by_case.get(case_id)
        if mapping is None:
            raise QualityEvaluationError(f"ratings case={case_id} 在 key 中不存在")
        ratings = case.get("ratings")
        if not isinstance(ratings, Mapping):
            continue
        case_had_complete = False
        for rater in raters:
            rater_id = str(rater)
            entry = ratings.get(rater_id)
            if not isinstance(entry, Mapping):
                continue
            score_a = entry.get("a")
            score_b = entry.get("b")
            if not isinstance(score_a, Mapping) or not isinstance(score_b, Mapping):
                continue
            values_a: dict[str, float] = {}
            values_b: dict[str, float] = {}
            complete = True
            for dimension in dimensions:
                if score_a.get(dimension) is None or score_b.get(dimension) is None:
                    complete = False
                    break
                values_a[dimension] = _number(
                    score_a[dimension], label=f"{case_id}/{rater_id}/a/{dimension}"
                )
                values_b[dimension] = _number(
                    score_b[dimension], label=f"{case_id}/{rater_id}/b/{dimension}"
                )
            if not complete:
                continue
            completed_ratings += 1
            case_had_complete = True
            for dimension in dimensions:
                condition_a = "on" if mapping["label_a"].endswith("_on") else "off"
                condition_b = "on" if mapping["label_b"].endswith("_on") else "off"
                by_condition = per_dimension.setdefault(
                    dimension,
                    {"on": [], "off": [], "deltas": []},
                )
                by_condition[condition_a].append(values_a[dimension])
                by_condition[condition_b].append(values_b[dimension])
                delta = (
                    values_a[dimension] - values_b[dimension]
                    if condition_a == "on"
                    else values_b[dimension] - values_a[dimension]
                )
                by_condition["deltas"].append(delta)
                all_deltas[dimension].append(delta)
                case_deltas[dimension].setdefault(case_id, []).append(delta)
                sample_deltas[dimension].setdefault(
                    str(mapping.get("sample_id") or case_id), []
                ).append(delta)
                rater_scores.setdefault(case_id, {}).setdefault(dimension, []).append(
                    (values_a[dimension], values_b[dimension])
                )
            preference = str(entry.get("preference") or "").lower()
            if preference in {"a", "b", "tie", "unclear"}:
                if preference == "tie":
                    preference_counts["tie"] += 1
                elif preference == "unclear":
                    preference_counts["unclear"] += 1
                else:
                    chosen_label = mapping[f"label_{preference}"]
                    preference_counts["on" if chosen_label.endswith("_on") else "off"] += 1
        if case_had_complete:
            completed_cases += 1
            distinct_sample_ids.add(str(mapping.get("sample_id") or case_id))

    if completed_ratings == 0:
        raise QualityEvaluationError("没有完整评分；请先填写 ratings")

    dimensions_result: dict[str, Any] = {}
    agreement_result: dict[str, Any] = {}
    for dimension in dimensions:
        values = per_dimension[dimension]
        # Average the raters within each case before testing, so one case
        # contributes exactly one value. Testing the raw per-rating deltas
        # would treat repeated measurements of the same case as independent
        # samples and report a p value the design cannot support.
        case_mean_deltas = [
            statistics.fmean(deltas)
            for _, deltas in sorted(case_deltas[dimension].items())
            if deltas
        ]
        # Repeated runs of the same audio become separate cases, so the case is
        # still not an independent unit. Collapsing to the originating
        # sample_id makes the randomization unit match the design: two runs of
        # one recording are repeated measurements, not two samples.
        sample_mean_deltas = [
            statistics.fmean(deltas)
            for _, deltas in sorted(sample_deltas[dimension].items())
            if deltas
        ]
        dimensions_result[dimension] = {
            "on": _summary(values["on"]),
            "off": _summary(values["off"]),
            "paired_delta_on_minus_off": _summary(values["deltas"]),
            "case_mean_delta_on_minus_off": _summary(case_mean_deltas),
            "sample_mean_delta_on_minus_off": _summary(sample_mean_deltas),
            "exact_sign_flip_p": _exact_sign_flip_p(sample_mean_deltas),
            "exact_sign_flip_p_by_case": _exact_sign_flip_p(case_mean_deltas),
            "significance_unit": "sample_id",
            "n_samples_tested": len(sample_mean_deltas),
            "n_cases_tested": len(case_mean_deltas),
        }
        rater_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for case_dimensions in rater_scores.values():
            scores = case_dimensions.get(dimension, [])
            for left, right in itertools.combinations(scores, 2):
                rater_pairs.append((left, right))
        if rater_pairs:
            exact = sum(
                left == right for left, right in rater_pairs
            )
            within_one = sum(
                abs(left[0] - right[0]) <= 1
                and abs(left[1] - right[1]) <= 1
                for left, right in rater_pairs
            )
            agreement_result[dimension] = {
                "rater_pairs": len(rater_pairs),
                "exact_pair_agreement": round(exact / len(rater_pairs), 4),
                "within_one_point_agreement": round(within_one / len(rater_pairs), 4),
            }
        else:
            agreement_result[dimension] = {
                "rater_pairs": 0,
                "exact_pair_agreement": None,
                "within_one_point_agreement": None,
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "response_quality_analysis",
        "evaluation_type": sheet.get("evaluation_type"),
        "quality_note": "Primary result is blinded LLM scoring; latency is not part of the rubric.",
        "dimensions": dimensions_result,
        "inter_rater_agreement": agreement_result,
        "preference": preference_counts,
        "completed_cases": completed_cases,
        "completed_ratings": completed_ratings,
        # Distinct audio samples behind the completed cases.  Repeated runs of
        # one manifest entry share a sample_id, so this is the real number of
        # independent observations and is normally smaller than
        # ``completed_cases``.  Reporting both keeps the gap visible instead of
        # letting repeats pass as extra evidence.
        "distinct_samples": len(distinct_sample_ids),
        "total_cases": len(cases),
        "total_raters": len(raters),
    }


def _extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract the first JSON object from a model response.

    Compatible OpenAI endpoints do not all implement ``response_format`` and
    some models still wrap otherwise valid JSON in a markdown fence.  Keep the
    parser strict about the resulting object while being tolerant about that
    harmless presentation detail.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise QualityEvaluationError("LLM 没有返回 JSON 对象")
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise QualityEvaluationError(f"LLM 返回无法解析为 JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QualityEvaluationError("LLM 评分结果必须是 JSON 对象")
    return value


def _parse_llm_judgement(
    text: str,
    dimensions: Iterable[str],
) -> dict[str, Any]:
    value = _extract_json_object(text)
    scores_a = value.get("a")
    scores_b = value.get("b")
    if not isinstance(scores_a, Mapping) or not isinstance(scores_b, Mapping):
        raise QualityEvaluationError("LLM 评分结果必须包含 a 和 b 两组分数")
    parsed_a: dict[str, float] = {}
    parsed_b: dict[str, float] = {}
    for dimension in dimensions:
        if dimension not in scores_a or dimension not in scores_b:
            raise QualityEvaluationError(f"LLM 评分缺少维度 {dimension}")
        parsed_a[dimension] = _number(scores_a[dimension], label=f"LLM/a/{dimension}")
        parsed_b[dimension] = _number(scores_b[dimension], label=f"LLM/b/{dimension}")
    preference = str(value.get("preference") or value.get("winner") or "").lower()
    if preference in {"left", "response_a", "response a"}:
        preference = "a"
    elif preference in {"right", "response_b", "response b"}:
        preference = "b"
    if preference not in {"a", "b", "tie", "unclear"}:
        raise QualityEvaluationError(
            "LLM preference 必须是 a、b、tie 或 unclear"
        )
    reason = value.get("reason", value.get("notes", ""))
    if not isinstance(reason, str):
        reason = str(reason)
    return {
        "a": parsed_a,
        "b": parsed_b,
        "preference": preference,
        "notes": reason[:2000],
    }


def _response_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QualityEvaluationError("LLM 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "".join(chunks)
    raise QualityEvaluationError("LLM 响应 content 不是文本")


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - project requirements include requests
        raise QualityEvaluationError("运行 LLM 评审需要 requests 依赖") from exc

    endpoint = base_url.rstrip("/") + "/chat/completions"
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=request_body,
            timeout=timeout,
        )
        # A few OpenAI-compatible gateways reject response_format even though
        # they otherwise implement chat/completions. Retry once without it.
        if response.status_code == 400 and json_mode:
            request_body.pop("response_format", None)
            response = requests.post(
                endpoint,
                headers=headers,
                json=request_body,
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if isinstance(exc, QualityEvaluationError):
            raise
        detail = getattr(locals().get("response"), "text", "")
        detail = detail[:500] if isinstance(detail, str) else ""
        suffix = f": {detail}" if detail else ""
        raise QualityEvaluationError(f"LLM 请求失败{suffix}") from exc
    if not isinstance(payload, Mapping):
        raise QualityEvaluationError("LLM 响应根节点不是对象")
    return _response_content(payload)


def _judge_prompt(
    *,
    evaluation_type: str,
    prompt: str,
    context: Mapping[str, Any],
    response_a: str,
    response_b: str,
    rubric: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rubric_text = "\n".join(
        f"- {row.get('id')}: {row.get('question')}" for row in rubric
    )
    context_text = json.dumps(context, ensure_ascii=False, sort_keys=True)
    user_message = f"""你是严格、稳定、保守的中文对话质量评审模型。你只能根据下面给出的用户问题、可用背景和两条回复评分，不能猜测 A/B 背后的系统条件。不要因为回复更长就自动给高分，也不要补充输入中没有的事实。这个评测类型是 {evaluation_type}。

用户问题：
{prompt}

可用背景（可能为空）：
{context_text}

回复 A：
{response_a}

回复 B：
{response_b}

评分维度（每项 1 到 5 分，5 为最好）：
{rubric_text}

只输出一个 JSON 对象，不要 Markdown，不要额外解释。格式必须是：
{{"a": {{"维度id": 1}}, "b": {{"维度id": 1}}, "preference": "a|b|tie|unclear", "reason": "不超过两句的依据"}}
"""
    return [
        {
            "role": "system",
            "content": "你是盲法 A/B 评审器。必须完整输出每个评分维度，分数只能是 1、2、3、4、5。",
        },
        {"role": "user", "content": user_message},
    ]


def judge_blind_sheet(
    sheet: Mapping[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    raters: list[str],
    seed: int,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: float = 90.0,
    json_mode: bool = True,
) -> dict[str, Any]:
    """Fill a blind sheet using one or more independent LLM judge passes."""
    if sheet.get("kind") != "blinded_response_quality_ratings":
        raise QualityEvaluationError("输入不是盲评表")
    cases = sheet.get("cases")
    rubric = sheet.get("rubric")
    if not isinstance(cases, list) or not isinstance(rubric, list):
        raise QualityEvaluationError("盲评表缺少 cases 或 rubric")
    if not api_key.strip():
        raise QualityEvaluationError("LLM API key 为空")
    if not raters or any(not item.strip() for item in raters):
        raise QualityEvaluationError("至少指定一个 LLM judge 名称")
    dimensions = [str(row.get("id")) for row in rubric if isinstance(row, Mapping)]
    if not dimensions:
        raise QualityEvaluationError("盲评表没有有效评分维度")

    result = json.loads(json.dumps(sheet, ensure_ascii=False))
    result["raters"] = raters
    result["judge_config"] = {
        "provider": "openai_compatible",
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "position_swap": True,
    }
    rng = random.Random(seed)
    for case_index, case in enumerate(result["cases"]):
        if not isinstance(case, dict):
            raise QualityEvaluationError("盲评表包含无效 case")
        response_a = str(case.get("response_a") or "")
        response_b = str(case.get("response_b") or "")
        prompt = str(case.get("prompt") or "")
        context = case.get("context")
        if not isinstance(context, Mapping):
            context = {}
        case["ratings"] = {}
        for rater_index, rater in enumerate(raters):
            swapped = rng.choice([False, True])
            shown_a, shown_b = (response_b, response_a) if swapped else (response_a, response_b)
            messages = _judge_prompt(
                evaluation_type=str(result.get("evaluation_type") or ""),
                prompt=prompt,
                context=context,
                response_a=shown_a,
                response_b=shown_b,
                rubric=rubric,
            )
            raw = _chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                json_mode=json_mode,
            )
            parsed = _parse_llm_judgement(raw, dimensions)
            if swapped:
                parsed = {
                    **parsed,
                    "a": parsed["b"],
                    "b": parsed["a"],
                    "preference": {
                        "a": "b",
                        "b": "a",
                        "tie": "tie",
                        "unclear": "unclear",
                    }[parsed["preference"]],
                }
            case["ratings"][rater] = parsed
            # Keep a non-sensitive trace for reproducibility, without storing
            # prompts, API keys, or the hidden condition mapping.
            case.setdefault("judge_trace", {})[rater] = {
                "position_swapped": swapped,
                "case_index": case_index,
                "rater_index": rater_index,
            }
    return result


def _judge_command(arguments: argparse.Namespace) -> int:
    sheet = _read_json(arguments.ratings)
    api_key = os.getenv(arguments.api_key_env, "")
    raters = [item.strip() for item in arguments.judge_raters.split(",") if item.strip()]
    result = judge_blind_sheet(
        sheet,
        api_key=api_key,
        base_url=arguments.base_url,
        model=arguments.model,
        raters=raters,
        seed=arguments.seed,
        temperature=arguments.temperature,
        max_tokens=arguments.max_tokens,
        timeout=arguments.timeout,
        json_mode=not arguments.no_json_mode,
    )
    _write_json(arguments.output, result)
    print(
        f"LLM 盲评已写入 {arguments.output}："
        f"{len(result['cases'])} 个案例 × {len(result['raters'])} 个 judge"
    )
    return 0


def _prepare_command(arguments: argparse.Namespace) -> int:
    benchmark = _read_json(arguments.input)
    context = _read_json(arguments.context) if arguments.context else None
    raters = [item.strip() for item in arguments.raters.split(",") if item.strip()]
    if not raters:
        raise QualityEvaluationError("至少指定一个 rater")
    sheet, key = prepare_blind_sheet(
        benchmark,
        seed=arguments.seed,
        raters=raters,
        context=context,
    )
    _write_json(arguments.output, sheet)
    _write_json(arguments.key_output, key)
    print(f"LLM 盲评输入已写入 {arguments.output}，共 {len(sheet['cases'])} 个配对")
    print(f"条件密钥已写入 {arguments.key_output}，仅供 analyze 使用")
    return 0


def _analyse_command(arguments: argparse.Namespace) -> int:
    sheet = _read_json(arguments.ratings)
    key = _read_json(arguments.key)
    result = analyse_ratings(sheet, key)
    _write_json(arguments.output, result)
    print(
        f"质量分析已写入 {arguments.output}："
        f"{result['completed_ratings']} 个完整评分，"
        f"{result['completed_cases']} 个案例"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成和分析盲法回答质量消融评测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="从 benchmark 结果生成盲评表")
    prepare.add_argument("--input", required=True, help="benchmark JSON")
    prepare.add_argument("--output", required=True, help="供 judge 子命令读取的盲评 JSON")
    prepare.add_argument("--key-output", required=True, help="条件映射密钥 JSON")
    prepare.add_argument("--seed", type=int, default=20260913)
    prepare.add_argument(
        "--raters",
        default="llm_judge_1,llm_judge_2,llm_judge_3",
        help="盲评表中的 judge 名称；通常直接使用默认值",
    )
    prepare.add_argument(
        "--context",
        help="可选场景 JSON；cases 可按 case_id 或 sample_id 提供 memory_facts 等参考信息",
    )
    prepare.set_defaults(handler=_prepare_command)

    judge = subparsers.add_parser(
        "judge", help="使用 OpenAI-compatible LLM 对盲评表自动评分"
    )
    judge.add_argument("--ratings", required=True, help="prepare 生成的盲评表")
    judge.add_argument("--output", required=True, help="LLM 评分后的 JSON")
    judge.add_argument(
        "--judge-raters",
        default="llm_judge_1,llm_judge_2,llm_judge_3",
        help="逗号分隔的独立 judge 名称；每个名称执行一次独立请求",
    )
    judge.add_argument(
        "--base-url",
        default=os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1",
        help="OpenAI-compatible API base URL",
    )
    judge.add_argument(
        "--model",
        default=os.getenv("SILICONFLOW_MODEL") or "qwen-flash",
        help="模型名称",
    )
    judge.add_argument(
        "--api-key-env",
        default="SILICONFLOW_API_KEY",
        help="API key 所在环境变量名（不会写入输出）",
    )
    judge.add_argument("--seed", type=int, default=20260913)
    judge.add_argument("--temperature", type=float, default=0.0)
    judge.add_argument("--max-tokens", type=int, default=1200)
    judge.add_argument("--timeout", type=float, default=90.0)
    judge.add_argument(
        "--no-json-mode",
        action="store_true",
        help="兼容不支持 response_format=json_object 的服务",
    )
    judge.set_defaults(handler=_judge_command)

    analyse = subparsers.add_parser("analyze", help="分析 LLM 评分后的盲评表")
    analyse.add_argument("--ratings", required=True, help="judge 生成的 LLM 评分 JSON")
    analyse.add_argument("--key", required=True, help="prepare 阶段生成的密钥 JSON")
    analyse.add_argument("--output", required=True, help="分析结果 JSON")
    analyse.set_defaults(handler=_analyse_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except QualityEvaluationError as exc:
        print(f"错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
