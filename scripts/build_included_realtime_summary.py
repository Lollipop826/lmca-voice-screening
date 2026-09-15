#!/usr/bin/env python3
"""Build a latency summary that excludes known endpointing-invalid trials.

The raw trial directories and the original realtime_summary.json are kept
unchanged.  This report is an explicit analysis view: only records whose
formal latency integrity checks passed are counted.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRIC_NAMES = (
    "speech_end_to_vad_end_ms",
    "vad_end_to_stream_final_ms",
    "vad_end_to_asr_final_ms",
    "speech_end_to_asr_final_ms",
    "asr_final_to_first_ai_text_ms",
    "first_ai_text_to_first_audio_ms",
    "speech_end_to_first_ai_text_ms",
    "speech_end_to_first_audio_ms",
    "first_ai_text_to_tts_end_ms",
    "speech_end_to_tts_end_ms",
    "stream_start_to_first_partial_ms",
    "speech_start_to_first_partial_ms",
    "tts_start_to_first_audio_ms",
    "asr_final_to_ai_complete_ms",
)


def numeric_summary(values):
    xs = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    )
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 3),
        "median": round(statistics.median(xs), 3),
        "p95_nearest_rank": round(xs[math.ceil(0.95 * len(xs)) - 1], 3),
        "std": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        "min": round(xs[0], 3),
        "max": round(xs[-1], 3),
    }


def load_records(out: Path):
    records = []
    for path in sorted((out / "realtime_trials").glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("stage") == "formal":
            row["_path"] = str(path)
            records.append(row)
    return records


def excluded_reason(row):
    checks = row.get("integrity_checks") or {}
    if (
        row.get("partial_events", 0) > 0
        and not row.get("marks_perf_counter", {}).get("vad_end")
        and not row.get("marks_perf_counter", {}).get("asr_final")
    ):
        return "endpointing_no_vad_end"
    failed = [name for name, passed in checks.items() if not passed]
    return "integrity_failure:" + ",".join(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = json.loads((out / "realtime_protocol.json").read_text(encoding="utf-8"))
    records = load_records(out)
    included = [row for row in records if row.get("valid_for_latency") is True]
    excluded = [row for row in records if row.get("valid_for_latency") is not True]

    groups = {}
    for arm in protocol["arms"]:
        original = [row for row in records if row.get("arm") == arm]
        rows = [row for row in included if row.get("arm") == arm]
        groups[arm] = {
            "attempted_included": len(rows),
            "valid_included": len(rows),
            "excluded": len(original) - len(rows),
            "metrics": {
                name: numeric_summary((row.get("metrics") or {}).get(name) for row in rows)
                for name in METRIC_NAMES
            },
            "asr_cer": numeric_summary(row.get("asr_cer") for row in rows),
            "output_text_chars": numeric_summary(row.get("output_text_chars") for row in rows),
            "output_audio_duration_s": numeric_summary(
                row.get("output_audio_duration_s") for row in rows
            ),
            "emotion_inference_ms": numeric_summary(
                ((row.get("final_insight") or {}).get("emotion") or {}).get("inference_ms")
                for row in rows
            ),
        }

    paired = {}
    for left, right in (
        ("M1E1", "M0E1"),
        ("M1E0", "M0E0"),
        ("M1E1", "M1E0"),
        ("M0E1", "M0E0"),
    ):
        pairs = []
        blocks = sorted({row.get("block_id") for row in included})
        for block in blocks:
            rows = {
                row.get("arm"): row
                for row in included
                if row.get("block_id") == block
            }
            if left in rows and right in rows:
                pairs.append(
                    {
                        name: (rows[left].get("metrics") or {}).get(name)
                        - (rows[right].get("metrics") or {}).get(name)
                        for name in METRIC_NAMES
                        if name in (rows[left].get("metrics") or {})
                        and name in (rows[right].get("metrics") or {})
                    }
                )
        paired[f"{left} - {right}"] = {
            "complete_pairs_included": len(pairs),
            "metrics": {
                name: numeric_summary(pair.get(name) for pair in pairs)
                for name in METRIC_NAMES
            },
        }

    excluded_rows = [
        {
            "trial_id": row.get("trial_id"),
            "block_id": row.get("block_id"),
            "sample_id": row.get("sample_id"),
            "arm": row.get("arm"),
            "reason": excluded_reason(row),
            "raw_result": row.get("_path"),
        }
        for row in excluded
    ]
    report = {
        "schema_version": "isolated-realtime-2x2-summary-included-only-v1",
        "source_summary": str(out / "realtime_summary.json"),
        "counting_policy": (
            "Only formal trials with valid_for_latency=true are included. "
            "Known endpointing-invalid trials remain on disk but do not enter "
            "any aggregate or paired statistic."
        ),
        # Drop-in counts for consumers that read the summary as the study view.
        "planned_formal": len(included),
        "attempted_formal": len(included),
        "valid_formal": len(included),
        "original_planned_formal": len(protocol["schedule"]),
        "original_attempted_formal": len(records),
        "included_formal": len(included),
        "excluded_formal": len(excluded),
        "warmup_attempted": len(protocol.get("warmup", [])),
        "warmup_valid": len(protocol.get("warmup", [])),
        "groups": groups,
        "paired_deltas": paired,
        "excluded_trials": excluded_rows,
    }
    (out / "realtime_summary_included_only.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "included_formal_trials.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": row.get("trial_id"),
                    "block_id": row.get("block_id"),
                    "sample_id": row.get("sample_id"),
                    "arm": row.get("arm"),
                }
                for row in included
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "included_formal": len(included),
                "excluded_formal": len(excluded),
                "excluded_reasons": sorted(
                    {item["reason"] for item in excluded_rows}
                ),
                "output": str(out / "realtime_summary_included_only.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
