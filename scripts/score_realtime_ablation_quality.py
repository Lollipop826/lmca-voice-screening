#!/usr/bin/env python3
"""Score response quality for the 2x2 real-time voice ablation.

The latency benchmark answers "how fast", not "how good". This script adds the
quality half using the trials that already exist on disk: it never contacts the
service, never calls an LLM, and never needs API quota, so the numbers are
reproducible from a finished run directory alone.

Two ideas make deterministic scoring defensible here.

First, the seeded fixture card holds four facts that are *not* present in any
stimulus transcript. A reply can only contain them by reading long-term memory,
so their appearance is machine-checkable evidence that memory was used.

Second, any anchor phrase that already occurs in what the user said proves
nothing: the model can echo it straight from the transcript. Measured case in
point, "周三" appears in the walk-update transcript, and counting it scored all
four arms an identical 2/22 on the Wednesday-walk anchor. Contaminated phrases
are therefore dropped per stimulus before scoring.

Paired contrasts use the block as the randomization unit, matching the latency
summary, and significance comes from an exact McNemar test on discordant pairs.

Usage
-----
Score a finished run and write the report next to the trials::

    python scripts/score_realtime_ablation_quality.py score \
      --run output/real_voice_ablation_20260914_diverse_verified

Export the same pairs as a blind-judge input for the LLM rubric path::

    python scripts/score_realtime_ablation_quality.py export-pairs \
      --run output/real_voice_ablation_20260914_diverse_verified \
      --ablation memory \
      --output output/memory_quality_pairs.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ARM_IDS = ("M1E1", "M0E1", "M1E0", "M0E0")

# Contrasts mirror the latency summary so both halves of the ablation can be
# read side by side. Each pair holds exactly one factor constant.
PAIRED_CONTRASTS = (
    ("M1E1", "M0E1", "记忆效应（情绪开）"),
    ("M1E0", "M0E0", "记忆效应（情绪关）"),
    ("M1E1", "M1E0", "情绪效应（记忆开）"),
    ("M0E1", "M0E0", "情绪效应（记忆关）"),
)

# The four facts seeded into the authoritative card. Phrases are the surface
# forms observed in real replies during calibration, not guesses: scoring a
# phrase list that never fires would silently report a null effect.
MEMORY_CARD_ANCHORS: dict[str, dict[str, Any]] = {
    "work_crunch": {
        "fact": "最近项目赶工，常常午夜后才停止工作，第二天容易睡不好。",
        "phrases": ("赶工", "午夜", "熬夜", "加班", "项目"),
    },
    "phone_away_at_night": {
        "fact": "睡前反复看工作消息会让入睡更困难，愿意尝试把手机放远一点。",
        "phrases": ("放远", "客厅", "离床", "拿到卧室", "拿出卧室", "远一点", "放到别"),
    },
    "daughter_contact_argument": {
        "fact": "上周和女儿因为联系频率发生争执，女儿后来暂时没有回复消息。",
        "phrases": ("争执", "联系频率", "闹别扭", "上周", "吵"),
    },
    "wednesday_walk_with_neighbour": {
        "fact": "最近每周三上午会和邻居在小区里散步。",
        "phrases": ("周三", "邻居", "小区"),
    },
}

# Derived from the ``expected_behavior`` string the protocol froze for every
# stimulus. Only the machine-checkable part of each expectation is encoded; the
# subjective remainder is what the blinded LLM rubric is for.
SAMPLE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "sleep-followup": {
        "should_cite": ("phone_away_at_night",),
        "expectation": "可衔接已同意的睡前手机放远习惯",
    },
    "family-followup": {
        "should_cite": ("daughter_contact_argument",),
        "expectation": "允许自然衔接联系频率争执",
    },
    "method-followup": {
        "should_cite": ("phone_away_at_night",),
        "expectation": "结合睡前手机放远的既有偏好进行核对",
        # ASR heard 睡前 as 税前 in every trial, so the reference the model was
        # asked to resolve never reached it. Kept visible, excluded from the
        # citation rate: a corrupted input cannot test reference resolution.
        "input_corrupted_note": "ASR 把「睡前」识别为「税前」，指代目标未送达模型",
    },
    "walk-positive": {
        "should_withhold": ("wednesday_walk_with_neighbour",),
        "expectation": "接住轻松感；不擅自断言今天是周三或一定与邻居同行",
    },
    "keys-unrelated": {
        "should_withhold": (
            "work_crunch",
            "phone_away_at_night",
            "daughter_contact_argument",
            "wednesday_walk_with_neighbour",
        ),
        "expectation": "回应眼前经历；不强扯家庭矛盾、睡眠或散步史",
    },
    "cooking-control": {
        "should_withhold": (
            "work_crunch",
            "phone_away_at_night",
            "daughter_contact_argument",
            "wednesday_walk_with_neighbour",
        ),
        "expectation": "给与现有食材匹配的简短建议；不强行心理分析或引用旧事",
    },
    "walk-update": {
        "must_acknowledge": ("周五",),
        "expectation": "认可新时间；不能继续强调周三散步",
    },
    "family-update": {
        "must_acknowledge": ("说开", "和好", "误会", "好消息", "高兴", "替你"),
        "expectation": "以已经和好为准；接住高兴，不再把未回消息当作现状",
    },
    "anger-boundary": {
        "must_not_say": ("忍着", "忍一忍", "让一让", "消消气", "别生气", "大度", "冷静一下"),
        "expectation": "认可愤怒和不想忍让的边界；不劝用户忍着",
    },
    "sad-no-advice": {
        "must_not_say": ("建议", "试试", "要不要", "办法是", "可以先"),
        "expectation": "倾听并尊重不想听办法；不列解决方案",
    },
    "anxious-task": {
        "should_withhold": ("work_crunch",),
        "expectation": "回应办手续引起的紧张；不把全部原因归于历史赶工",
    },
    "implicit-alone": {
        "expectation": "温和留出空间；可试探但不武断断言被抛弃",
    },
}


class AblationQualityError(RuntimeError):
    """The run directory cannot support a paired quality comparison."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationQualityError(f"无法读取 {path}: {exc}") from exc


def load_trials(run_directory: Path) -> list[dict[str, Any]]:
    trials_directory = run_directory / "realtime_trials"
    if not trials_directory.is_dir():
        raise AblationQualityError(f"找不到试次目录 {trials_directory}")
    trials = [
        read_json(path)
        for path in sorted(trials_directory.glob("formal-*/result.json"))
    ]
    if not trials:
        raise AblationQualityError(f"{trials_directory} 下没有 formal 试次")
    return trials


def is_scorable(trial: Mapping[str, Any]) -> bool:
    """Whether one trial produced a reply that can be judged at all.

    Deliberately narrower than the latency validity flag: a trial with perfect
    timing marks but an empty reply carries no quality signal.
    """
    return bool(
        trial.get("status") == "ok"
        and trial.get("valid_for_latency")
        and str(trial.get("final_ai_text") or "").strip()
        and len(trial.get("observed_turn_ids") or []) == 1
        and len(trial.get("observed_asr_results") or []) == 1
    )


def build_paired_blocks(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group scorable trials into blocks that contain all four arms.

    A partial block cannot enter a paired contrast, so incomplete groups are
    returned separately rather than silently averaged into the arm totals.
    """
    grouped: dict[tuple[Any, str], dict[str, Mapping[str, Any]]] = {}
    for trial in trials:
        if not is_scorable(trial):
            continue
        key = (trial.get("block_id"), str(trial.get("sample_id") or ""))
        arm = str(trial.get("arm") or "")
        if arm not in ARM_IDS:
            continue
        grouped.setdefault(key, {})[arm] = trial

    blocks: list[dict[str, Any]] = []
    for (block_id, sample_id), arms in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        if set(arms) != set(ARM_IDS):
            continue
        recognized_texts = {str(trial.get("asr_text") or "") for trial in arms.values()}
        stimulus_hashes = {str(trial.get("sample_sha256") or "") for trial in arms.values()}
        if len(recognized_texts) != 1 or len(stimulus_hashes) != 1:
            # Different input across arms breaks the only thing pairing buys us.
            continue
        blocks.append(
            {
                "block_id": block_id,
                "sample_id": sample_id,
                "recognized_text": recognized_texts.pop(),
                "stimulus_sha256": stimulus_hashes.pop(),
                "arms": dict(arms),
            }
        )
    return blocks


def verify_arm_manipulations(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Confirm each arm really ran with the flags its label claims.

    Without this the whole comparison could be measuring nothing: an arm
    labelled memory-off that still read the card would produce a null effect
    that looks like a finding.
    """
    observations: dict[str, dict[str, set[str]]] = {
        arm: {"memory_source": set(), "emotion_source": set(), "audio_model": set()}
        for arm in ARM_IDS
    }
    for block in blocks:
        for arm, trial in block["arms"].items():
            insight = trial.get("final_insight") or {}
            memory = insight.get("memory") or {}
            emotion = insight.get("emotion") or {}
            observations[arm]["memory_source"].add(str(memory.get("retrieval_source")))
            observations[arm]["emotion_source"].add(str(emotion.get("source")))
            observations[arm]["audio_model"].add(str(emotion.get("audio_model_used")))

    findings: dict[str, Any] = {}
    consistent = True
    for arm in ARM_IDS:
        memory_expected_on = arm.startswith("M1")
        emotion_expected_on = arm.endswith("E1")
        memory_sources = observations[arm]["memory_source"]
        emotion_sources = observations[arm]["emotion_source"]
        memory_ok = memory_sources == ({"realtime"} if memory_expected_on else {"disabled"})
        emotion_ok = emotion_sources == (
            {"emotion2vec_audio+text"} if emotion_expected_on else {"disabled"}
        )
        consistent = consistent and memory_ok and emotion_ok
        findings[arm] = {
            "memory_retrieval_source": sorted(memory_sources),
            "emotion_source": sorted(emotion_sources),
            "emotion_audio_model_used": sorted(observations[arm]["audio_model"]),
            "memory_manipulation_verified": memory_ok,
            "emotion_manipulation_verified": emotion_ok,
        }
    findings["all_arms_verified"] = consistent
    return findings


def usable_anchor_phrases(anchor_id: str, recognized_text: str) -> tuple[str, ...]:
    """Anchor phrases that can serve as memory evidence for one stimulus.

    A phrase the user already said is not evidence of recall, so it is removed
    before matching. This is what separates a real memory citation from the
    model echoing the transcript back.
    """
    phrases = MEMORY_CARD_ANCHORS[anchor_id]["phrases"]
    return tuple(phrase for phrase in phrases if phrase not in recognized_text)


def cited_anchor(reply_text: str, anchor_id: str, recognized_text: str) -> bool:
    return any(
        phrase in reply_text
        for phrase in usable_anchor_phrases(anchor_id, recognized_text)
    )


def score_reply(
    reply_text: str,
    *,
    sample_id: str,
    recognized_text: str,
) -> dict[str, bool]:
    """Binary outcomes for one reply, keyed by metric name.

    Only metrics that apply to this stimulus are present; an absent key means
    "not applicable here" rather than "scored zero", which keeps inapplicable
    stimuli out of the denominators.
    """
    expectation = SAMPLE_EXPECTATIONS.get(sample_id, {})
    outcomes: dict[str, bool] = {}

    should_cite = expectation.get("should_cite", ())
    if should_cite and not expectation.get("input_corrupted_note"):
        outcomes["cited_expected_memory"] = any(
            cited_anchor(reply_text, anchor_id, recognized_text)
            for anchor_id in should_cite
        )

    should_withhold = expectation.get("should_withhold", ())
    if should_withhold:
        outcomes["forced_irrelevant_memory"] = any(
            cited_anchor(reply_text, anchor_id, recognized_text)
            for anchor_id in should_withhold
        )

    must_acknowledge = expectation.get("must_acknowledge", ())
    if must_acknowledge:
        outcomes["acknowledged_correction"] = any(
            phrase in reply_text for phrase in must_acknowledge
        )

    must_not_say = expectation.get("must_not_say", ())
    if must_not_say:
        outcomes["violated_emotional_boundary"] = any(
            phrase in reply_text for phrase in must_not_say
        )

    return outcomes


def exact_mcnemar_two_sided(left_only: int, right_only: int) -> float | None:
    """Two-sided exact McNemar p value from discordant pair counts.

    Concordant pairs carry no information about direction, so the test reduces
    to asking whether the discordant split looks like a fair coin.
    """
    discordant_total = left_only + right_only
    if discordant_total == 0:
        return None
    smaller_side = min(left_only, right_only)
    tail = sum(math.comb(discordant_total, k) for k in range(smaller_side + 1))
    probability = 2.0 * tail / (2.0**discordant_total)
    return round(min(1.0, probability), 6)


def summarize_metric(
    blocks: Sequence[Mapping[str, Any]],
    scores: Mapping[tuple[Any, str, str], Mapping[str, bool]],
    metric_name: str,
) -> dict[str, Any]:
    """Per-arm rates plus paired contrasts for one metric."""
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        applicable = [
            scores[(block["block_id"], block["sample_id"], arm)][metric_name]
            for block in blocks
            if metric_name in scores[(block["block_id"], block["sample_id"], arm)]
        ]
        per_arm[arm] = {
            "applicable_blocks": len(applicable),
            "positive": sum(1 for value in applicable if value),
            "rate": round(sum(1 for value in applicable if value) / len(applicable), 4)
            if applicable
            else None,
        }

    contrasts: dict[str, Any] = {}
    for left_arm, right_arm, label in PAIRED_CONTRASTS:
        left_only = 0
        right_only = 0
        both = 0
        neither = 0
        for block in blocks:
            left_key = (block["block_id"], block["sample_id"], left_arm)
            right_key = (block["block_id"], block["sample_id"], right_arm)
            if metric_name not in scores[left_key] or metric_name not in scores[right_key]:
                continue
            left_value = scores[left_key][metric_name]
            right_value = scores[right_key][metric_name]
            if left_value and right_value:
                both += 1
            elif left_value:
                left_only += 1
            elif right_value:
                right_only += 1
            else:
                neither += 1
        paired_total = left_only + right_only + both + neither
        contrasts[f"{left_arm} - {right_arm}"] = {
            "label": label,
            "paired_blocks": paired_total,
            "both_positive": both,
            "neither_positive": neither,
            "left_only": left_only,
            "right_only": right_only,
            "rate_difference": round((left_only - right_only) / paired_total, 4)
            if paired_total
            else None,
            "exact_mcnemar_p": exact_mcnemar_two_sided(left_only, right_only),
        }
    return {"per_arm": per_arm, "paired_contrasts": contrasts}


def summarize_reply_length(
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reply length per arm, as a guard against confusing verbosity with quality."""
    per_arm: dict[str, Any] = {}
    for arm in ARM_IDS:
        lengths = [len(str(block["arms"][arm].get("final_ai_text") or "")) for block in blocks]
        per_arm[arm] = {
            "blocks": len(lengths),
            "mean_chars": round(statistics.fmean(lengths), 2) if lengths else None,
            "median_chars": round(statistics.median(lengths), 2) if lengths else None,
        }
    return per_arm


def collect_data_quality_flags(
    trials: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Problems a reader must see before trusting any rate in this report."""
    flags: list[dict[str, Any]] = []

    scorable_samples = {block["sample_id"] for block in blocks}
    all_samples = {str(trial.get("sample_id") or "") for trial in trials}
    for sample_id in sorted(all_samples - scorable_samples):
        related = [t for t in trials if t.get("sample_id") == sample_id]
        flags.append(
            {
                "kind": "sample_without_any_scorable_trial",
                "sample_id": sample_id,
                "attempted_trials": len(related),
                "detail": "该素材全部试次不可评分，其测试点在本轮完全缺失",
            }
        )

    for sample_id, expectation in SAMPLE_EXPECTATIONS.items():
        note = expectation.get("input_corrupted_note")
        if note and sample_id in scorable_samples:
            flags.append(
                {
                    "kind": "stimulus_input_corrupted",
                    "sample_id": sample_id,
                    "detail": note,
                    "consequence": "已从记忆引用率中排除，回复仍保留供人工与盲评阅读",
                }
            )

    contaminated: list[dict[str, Any]] = []
    for block in blocks:
        for anchor_id in MEMORY_CARD_ANCHORS:
            dropped = [
                phrase
                for phrase in MEMORY_CARD_ANCHORS[anchor_id]["phrases"]
                if phrase in block["recognized_text"]
            ]
            if dropped:
                contaminated.append(
                    {
                        "sample_id": block["sample_id"],
                        "anchor": anchor_id,
                        "phrases_dropped": sorted(set(dropped)),
                    }
                )
    if contaminated:
        deduplicated = {
            (item["sample_id"], item["anchor"], tuple(item["phrases_dropped"])): item
            for item in contaminated
        }
        flags.append(
            {
                "kind": "anchor_phrases_present_in_user_input",
                "detail": "这些词出现在用户话里，复述不能算记忆证据，已在该素材上剔除",
                "occurrences": sorted(
                    deduplicated.values(),
                    key=lambda item: (item["sample_id"], item["anchor"]),
                ),
            }
        )
    return flags


def score_run(run_directory: Path) -> dict[str, Any]:
    trials = load_trials(run_directory)
    blocks = build_paired_blocks(trials)
    if not blocks:
        raise AblationQualityError("没有四臂齐全且输入一致的配对块，无法做质量对比")

    scores: dict[tuple[Any, str, str], dict[str, bool]] = {}
    for block in blocks:
        for arm, trial in block["arms"].items():
            scores[(block["block_id"], block["sample_id"], arm)] = score_reply(
                str(trial.get("final_ai_text") or ""),
                sample_id=block["sample_id"],
                recognized_text=block["recognized_text"],
            )

    measured_metrics = sorted({name for outcome in scores.values() for name in outcome})
    return {
        "schema_version": "realtime-ablation-quality-v1",
        "run_directory": str(run_directory),
        "trials_on_disk": len(trials),
        "scorable_trials": sum(1 for trial in trials if is_scorable(trial)),
        "complete_paired_blocks": len(blocks),
        "distinct_stimuli": len({block["sample_id"] for block in blocks}),
        "manipulation_checks": verify_arm_manipulations(blocks),
        "data_quality_flags": collect_data_quality_flags(trials, blocks),
        "metrics": {
            metric_name: summarize_metric(blocks, scores, metric_name)
            for metric_name in measured_metrics
        },
        "reply_length": summarize_reply_length(blocks),
        "method": {
            "unit_of_randomization": "配对块（同一素材、同一音频、同一 ASR 文本下的四臂）",
            "significance_test": "双侧精确 McNemar，仅用不一致配对",
            "memory_evidence": "回复命中记忆卡锚点短语，且该短语未出现在用户输入中",
            "scope": "确定性可判定部分；语气与共情等主观维度交由盲评 rubric",
        },
    }


def build_judge_records(
    run_directory: Path,
    ablation: str,
) -> dict[str, Any]:
    """Emit the paired-record contract consumed by evaluate_response_quality.py.

    The 2x2 design is collapsed to one factor by holding the other factor fixed,
    which keeps every emitted pair a genuine single-factor comparison instead of
    averaging two different conditions together.
    """
    if ablation not in {"memory", "emotion"}:
        raise AblationQualityError(f"--ablation 只能是 memory 或 emotion，收到 {ablation!r}")

    blocks = build_paired_blocks(load_trials(run_directory))
    if ablation == "memory":
        held_constant = [("M1E1", "M0E1", "情绪开"), ("M1E0", "M0E0", "情绪关")]
    else:
        held_constant = [("M1E1", "M1E0", "记忆开"), ("M0E1", "M0E0", "记忆关")]

    records: list[dict[str, Any]] = []
    pair_index = 0
    for block in blocks:
        for on_arm, off_arm, held_label in held_constant:
            pair_index += 1
            for condition, arm in ((f"{ablation}_on", on_arm), (f"{ablation}_off", off_arm)):
                trial = block["arms"][arm]
                insight = trial.get("final_insight") or {}
                emotion = insight.get("emotion") or {}
                memory = insight.get("memory") or {}
                records.append(
                    {
                        "pair_index": pair_index,
                        "condition": condition,
                        "arm": arm,
                        "held_constant": held_label,
                        "sample_id": block["sample_id"],
                        "block_id": block["block_id"],
                        "recognized_text": block["recognized_text"],
                        "assistant_text": str(trial.get("final_ai_text") or ""),
                        "turn_count": len(trial.get("observed_turn_ids") or []),
                        "valid_for_latency": bool(trial.get("valid_for_latency")),
                        "asr_source": trial.get("asr_source"),
                        "asr_cer": trial.get("asr_cer"),
                        "audio_sha256": trial.get("audio_sha256"),
                        "memory_retrieval_source": memory.get("retrieval_source"),
                        "memory_retrieval_hit": memory.get("retrieval_hit"),
                        "memory_writes_enabled": memory.get("writes_enabled"),
                        "emotion_source": emotion.get("source"),
                        "emotion_audio_model_used": emotion.get("audio_model_used"),
                    }
                )
    return {
        "kind": "realtime_ablation_paired_records",
        "configuration": {"ablation": ablation, "run_directory": str(run_directory)},
        "pairs": pair_index,
        "records": records,
    }


def print_score_report(report: Mapping[str, Any]) -> None:
    print(f"运行目录: {report['run_directory']}")
    print(
        f"试次 {report['trials_on_disk']} 个，可评分 {report['scorable_trials']} 个，"
        f"四臂齐全的配对块 {report['complete_paired_blocks']} 个，"
        f"覆盖 {report['distinct_stimuli']} 个素材"
    )

    checks = report["manipulation_checks"]
    verdict = "通过" if checks["all_arms_verified"] else "未通过"
    print(f"\n操纵检查: {verdict}")
    for arm in ARM_IDS:
        detail = checks[arm]
        print(
            f"  {arm}: 记忆={','.join(detail['memory_retrieval_source'])}"
            f" 情绪={','.join(detail['emotion_source'])}"
            f" 音频模型={','.join(detail['emotion_audio_model_used'])}"
        )

    if report["data_quality_flags"]:
        print("\n数据质量提示:")
        for flag in report["data_quality_flags"]:
            if flag["kind"] == "anchor_phrases_present_in_user_input":
                print(f"  [{flag['kind']}] {flag['detail']}")
                for occurrence in flag["occurrences"]:
                    print(
                        f"      {occurrence['sample_id']} / {occurrence['anchor']}:"
                        f" 剔除 {occurrence['phrases_dropped']}"
                    )
            else:
                sample = flag.get("sample_id", "")
                print(f"  [{flag['kind']}] {sample} {flag['detail']}")

    for metric_name, summary in report["metrics"].items():
        print(f"\n指标 {metric_name}")
        header = "  " + "".join(f"{arm:>12}" for arm in ARM_IDS)
        print(header)
        rates = "  "
        for arm in ARM_IDS:
            cell = summary["per_arm"][arm]
            rates += f"{cell['positive']:>5}/{cell['applicable_blocks']:<6}"
        print(rates)
        for contrast_name, contrast in summary["paired_contrasts"].items():
            if not contrast["paired_blocks"]:
                continue
            probability = contrast["exact_mcnemar_p"]
            probability_text = "n/a（无不一致配对）" if probability is None else f"p={probability}"
            print(
                f"    {contrast_name} [{contrast['label']}]"
                f" 差异={contrast['rate_difference']:+.4f}"
                f" 仅左={contrast['left_only']} 仅右={contrast['right_only']}"
                f" {probability_text}"
            )

    print("\n回复长度（防止把啰嗦当成质量）")
    for arm in ARM_IDS:
        cell = report["reply_length"][arm]
        print(f"  {arm}: 均值 {cell['mean_chars']} 字，中位 {cell['median_chars']} 字")


def run_score_command(arguments: argparse.Namespace) -> int:
    run_directory = Path(arguments.run).resolve()
    report = score_run(run_directory)
    destination = (
        Path(arguments.output).resolve()
        if arguments.output
        else run_directory / "response_quality_report.json"
    )
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_score_report(report)
    print(f"\n报告已写入 {destination}")
    return 0


def run_export_command(arguments: argparse.Namespace) -> int:
    run_directory = Path(arguments.run).resolve()
    payload = build_judge_records(run_directory, arguments.ablation)
    destination = Path(arguments.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"导出 {payload['pairs']} 个 {arguments.ablation} 配对"
        f"（{len(payload['records'])} 条记录）到 {destination}"
    )
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    score = subcommands.add_parser("score", help="确定性质量评分，不调用任何外部服务")
    score.add_argument("--run", required=True, help="消融运行目录")
    score.add_argument("--output", help="报告输出路径，默认写入运行目录")
    score.set_defaults(handler=run_score_command)

    export = subcommands.add_parser("export-pairs", help="导出盲评所需的配对记录")
    export.add_argument("--run", required=True, help="消融运行目录")
    export.add_argument("--ablation", required=True, choices=("memory", "emotion"))
    export.add_argument("--output", required=True, help="配对记录输出路径")
    export.set_defaults(handler=run_export_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        return arguments.handler(arguments)
    except AblationQualityError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
