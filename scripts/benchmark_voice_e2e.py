"""
Benchmark the project's REAL streaming ASR path.

Actual project path:

    audio chunks
        ->
    RealtimeCompanion
        ->
    ArkASRStreamingSession
        ->
    BigASR streaming WebSocket
        ->
    partial / final ASR results

IMPORTANT:
This benchmark intentionally does NOT use `manual_audio`.
`manual_audio` is a complete-audio upload path and is not the
project's realtime streaming ASR path.

The benchmark reuses the exact ArkASRStreamingSession implementation
used by RealtimeCompanion.

Default streaming chunk size:
    200 ms

This matches:
    RealtimeCompanionConfig.asr_chunk_s = 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import wave
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000

# Project default:
# RealtimeCompanionConfig.asr_chunk_s = 0.2
DEFAULT_CHUNK_MS = 200


# ============================================================
# Metrics
# ============================================================

_METRICS = (
    "stream_start_ms",
    "first_partial_ms",
    "first_final_ms",
    "finish_to_final_ms",
    "stream_wall_ms",
    "rtf",
)


# ============================================================
# Utilities
# ============================================================

def _positive_int(value: str) -> int:
    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError(
            "must be greater than zero"
        )

    return number


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    if not ordered:
        return {
            "count": 0,
        }

    def percentile(ratio: float) -> float:
        return ordered[
            min(
                len(ordered) - 1,
                math.ceil(len(ordered) * ratio) - 1,
            )
        ]

    return {
        "count": len(ordered),
        "mean_ms": round(
            sum(ordered) / len(ordered),
            2,
        ),
        "p50_ms": round(
            percentile(0.50),
            2,
        ),
        "p95_ms": round(
            percentile(0.95),
            2,
        ),
        "max_ms": round(
            ordered[-1],
            2,
        ),
    }


def _summary_rtf(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    if not ordered:
        return {
            "count": 0,
        }

    def percentile(ratio: float) -> float:
        return ordered[
            min(
                len(ordered) - 1,
                math.ceil(len(ordered) * ratio) - 1,
            )
        ]

    return {
        "count": len(ordered),
        "mean": round(
            sum(ordered) / len(ordered),
            4,
        ),
        "p50": round(
            percentile(0.50),
            4,
        ),
        "p95": round(
            percentile(0.95),
            4,
        ),
        "max": round(
            ordered[-1],
            4,
        ),
    }


def _read_pcm16(
    path: str | Path,
) -> bytes:

    with wave.open(
        str(path),
        "rb",
    ) as source:

        params = source.getparams()

        if (
            params.nchannels,
            params.sampwidth,
            params.framerate,
            params.comptype,
        ) != (
            1,
            2,
            SAMPLE_RATE,
            "NONE",
        ):
            raise ValueError(
                "audio must be mono, PCM16, 16000 Hz WAV; "
                f"got channels={params.nchannels}, "
                f"width={params.sampwidth}, "
                f"rate={params.framerate}, "
                f"compression={params.comptype}"
            )

        data = source.readframes(
            params.nframes
        )

    if not data:
        raise ValueError(
            "audio file is empty"
        )

    return data


def _audio_duration_ms(
    pcm16: bytes,
) -> float:

    sample_count = len(pcm16) // 2

    return (
        sample_count
        / SAMPLE_RATE
        * 1000.0
    )


def _split_pcm16(
    pcm16: bytes,
    chunk_ms: int,
) -> list[bytes]:

    samples_per_chunk = int(
        SAMPLE_RATE
        * chunk_ms
        / 1000
    )

    if samples_per_chunk <= 0:
        raise ValueError(
            "chunk_ms must produce at least one sample"
        )

    bytes_per_chunk = (
        samples_per_chunk * 2
    )

    return [
        pcm16[offset : offset + bytes_per_chunk]
        for offset in range(
            0,
            len(pcm16),
            bytes_per_chunk,
        )
    ]


# ============================================================
# Streaming ASR benchmark
# ============================================================

async def _run_once(
    *,
    pcm16: bytes,
    chunk_ms: int,
    realtime: bool,
    timeout_s: float,
) -> dict[str, Any]:

    # Import the project's REAL implementation.
    #
    # This is the same class used by:
    #
    # RealtimeCompanion
    #     ->
    # ArkASRStreamingSession
    #
    from src.tools.voice.ark_asr import (
        ArkASRError,
        ArkASRStreamingSession,
        ark_asr_streaming_supported,
    )

    # --------------------------------------------------------
    # Verify actual project streaming mode
    # --------------------------------------------------------

    if not ark_asr_streaming_supported():
        raise RuntimeError(
            "Project streaming ASR is not enabled. "
            "ARK_ASR_MODE must be 'bigmodel' or 'bigmodel_async'."
        )

    # --------------------------------------------------------
    # Prepare chunks
    # --------------------------------------------------------

    chunks = _split_pcm16(
        pcm16,
        chunk_ms,
    )

    audio_duration_ms = _audio_duration_ms(
        pcm16
    )

    # --------------------------------------------------------
    # Timing state
    # --------------------------------------------------------

    marks: dict[str, float] = {}

    partial_texts: list[str] = []

    final_text = ""

    partial_count = 0

    final_count = 0

    first_feed_at = 0.0

    last_feed_end_at = 0.0

    final_received = asyncio.Event()

    callback_error: Exception | None = None

    # --------------------------------------------------------
    # Result callback
    #
    # This is exactly the callback used by
    # RealtimeCompanion._on_result().
    # --------------------------------------------------------

    async def on_result(
        text: str,
        is_final: bool,
    ) -> None:

        nonlocal \
            partial_count, \
            final_count, \
            final_text, \
            callback_error

        now = time.perf_counter()

        text = str(text or "").strip()

        if not text:
            return

        try:

            if is_final:

                final_count += 1

                final_text = text

                marks.setdefault(
                    "first_final",
                    now,
                )

                final_received.set()

            else:

                partial_count += 1

                partial_texts.append(
                    text
                )

                marks.setdefault(
                    "first_partial",
                    now,
                )

        except Exception as exc:

            callback_error = exc

            final_received.set()

    # --------------------------------------------------------
    # Create EXACT project streaming session
    # --------------------------------------------------------

    stream = ArkASRStreamingSession(
        on_result=on_result
    )

    stream_start_at = time.perf_counter()

    try:

        # ----------------------------------------------------
        # BigASR streaming WebSocket start
        # ----------------------------------------------------

        await stream.start()

        stream_started_at = time.perf_counter()

        marks[
            "stream_start"
        ] = stream_started_at

        # ----------------------------------------------------
        # Feed realtime audio chunks
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            if not chunk:
                continue

            # Convert bytes to the same numeric representation
            # expected by RealtimeCompanion.
            #
            # RealtimeCompanion feeds numpy audio arrays.
            import numpy as np

            audio = np.frombuffer(
                chunk,
                dtype=np.int16,
            ).astype(
                np.float32
            )

            # Project's realtime path works with normalized
            # float audio.
            audio /= 32768.0

            if first_feed_at == 0.0:

                first_feed_at = (
                    time.perf_counter()
                )

                marks[
                    "first_feed"
                ] = first_feed_at

            # ------------------------------------------------
            # Actual streaming ASR call
            # ------------------------------------------------

            await stream.feed(
                audio
            )

            last_feed_end_at = (
                time.perf_counter()
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # In real usage audio arrives in realtime.
            #
            # If we send all chunks immediately, we measure
            # upload/engine throughput rather than realtime
            # ASR responsiveness.
            # ------------------------------------------------

            if realtime:

                expected_elapsed = (
                    (index + 1)
                    * chunk_ms
                    / 1000.0
                )

                actual_elapsed = (
                    time.perf_counter()
                    - first_feed_at
                )

                remaining = (
                    expected_elapsed
                    - actual_elapsed
                )

                if remaining > 0:

                    await asyncio.sleep(
                        remaining
                    )

        # ----------------------------------------------------
        # All audio has been sent.
        #
        # This is important:
        #
        # finish() is the actual finalization of the
        # VAD-delimited utterance on the BigASR stream.
        # ----------------------------------------------------

        finish_started_at = (
            time.perf_counter()
        )

        marks[
            "finish_start"
        ] = finish_started_at

        # Give the ASR stream a little time to return its
        # final result.
        try:

            await asyncio.wait_for(
                stream.finish(),
                timeout=timeout_s,
            )

        except asyncio.TimeoutError:

            raise TimeoutError(
                "stream.finish() timed out"
            )

        # ----------------------------------------------------
        # The callback normally receives final result during
        # finish(). Give the callback a short scheduling window.
        # ----------------------------------------------------

        try:

            await asyncio.wait_for(
                final_received.wait(),
                timeout=min(
                    5.0,
                    timeout_s,
                ),
            )

        except asyncio.TimeoutError:

            # Some implementations return the final text
            # directly from finish() without calling the
            # callback in the expected order.
            pass

        finished_at = time.perf_counter()

        marks[
            "finished"
        ] = finished_at

        if callback_error:

            raise callback_error

    except ArkASRError:

        raise

    finally:

        try:

            await stream.aclose()

        except Exception:

            pass

    # --------------------------------------------------------
    # Build metrics
    # --------------------------------------------------------

    record: dict[str, Any] = {

        "audio_duration_ms":
            round(
                audio_duration_ms,
                2,
            ),

        "chunk_ms":
            chunk_ms,

        "chunk_count":
            len(chunks),

        "partial_count":
            partial_count,

        "final_count":
            final_count,

        "partial_texts":
            partial_texts,

        "final_text":
            final_text,

        "realtime":
            realtime,
    }

    # --------------------------------------------------------
    # Stream startup
    # --------------------------------------------------------

    if "stream_start" in marks:

        record[
            "stream_start_ms"
        ] = round(
            (
                marks[
                    "stream_start"
                ]
                - stream_start_at
            )
            * 1000,
            2,
        )

    # --------------------------------------------------------
    # First partial
    #
    # First audio chunk -> first non-final result
    # --------------------------------------------------------

    if (
        "first_feed" in marks
        and "first_partial" in marks
    ):

        record[
            "first_partial_ms"
        ] = round(
            (
                marks[
                    "first_partial"
                ]
                - marks[
                    "first_feed"
                ]
            )
            * 1000,
            2,
        )

    # --------------------------------------------------------
    # First final
    #
    # First audio chunk -> first final result
    # --------------------------------------------------------

    if (
        "first_feed" in marks
        and "first_final" in marks
    ):

        record[
            "first_final_ms"
        ] = round(
            (
                marks[
                    "first_final"
                ]
                - marks[
                    "first_feed"
                ]
            )
            * 1000,
            2,
        )

    # --------------------------------------------------------
    # Finalization latency
    #
    # Last audio chunk -> final result
    # --------------------------------------------------------

    if (
        "finish_start" in marks
        and "first_final" in marks
    ):

        record[
            "finish_to_final_ms"
        ] = round(
            (
                marks[
                    "first_final"
                ]
                - marks[
                    "finish_start"
                ]
            )
            * 1000,
            2,
        )

    # --------------------------------------------------------
    # Overall realtime streaming wall time
    #
    # First audio chunk -> final result
    # --------------------------------------------------------

    if (
        "first_feed" in marks
        and "first_final" in marks
    ):

        wall_ms = (
            marks[
                "first_final"
            ]
            - marks[
                "first_feed"
            ]
        ) * 1000

        record[
            "stream_wall_ms"
        ] = round(
            wall_ms,
            2,
        )

        record[
            "rtf"
        ] = round(
            wall_ms
            / max(
                audio_duration_ms,
                1e-6,
            ),
            4,
        )

    # --------------------------------------------------------
    # Additional useful metrics
    # --------------------------------------------------------

    if (
        "first_feed" in marks
        and "stream_start" in marks
    ):

        record[
            "stream_start_from_feed_ms"
        ] = round(
            (
                marks[
                    "first_feed"
                ]
                - stream_start_at
            )
            * 1000,
            2,
        )

    if (
        "first_partial" in marks
        and "first_final" in marks
    ):

        record[
            "partial_to_final_ms"
        ] = round(
            (
                marks[
                    "first_final"
                ]
                - marks[
                    "first_partial"
                ]
            )
            * 1000,
            2,
        )

    return record


# ============================================================
# Benchmark loop
# ============================================================

async def _run_benchmark(
    args,
    pcm16: bytes,
) -> dict[str, Any]:

    records: list[dict[str, Any]] = []

    failures: list[dict[str, Any]] = []

    total_runs = (
        args.warmup_runs
        + args.runs
    )

    for index in range(
        total_runs
    ):

        phase = (
            "warmup"
            if index < args.warmup_runs
            else "measured"
        )

        logical_run = (
            index + 1
            if phase == "warmup"
            else (
                index
                - args.warmup_runs
                + 1
            )
        )

        record = None

        for attempt in range(
            1,
            args.max_attempts_per_run + 1,
        ):

            try:

                record = await _run_once(
                    pcm16=pcm16,
                    chunk_ms=args.chunk_ms,
                    realtime=args.realtime,
                    timeout_s=args.timeout,
                )

                break

            except Exception as exc:

                failure = {
                    "phase": phase,
                    "logical_run": logical_run,
                    "attempt": attempt,
                    "error_type": type(
                        exc
                    ).__name__,
                    "detail": str(exc),
                }

                failures.append(
                    failure
                )

                print(
                    f"[{phase} run {logical_run}] "
                    f"attempt {attempt} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if (
                    attempt
                    == args.max_attempts_per_run
                ):

                    record = None

                    break

                await asyncio.sleep(
                    args.retry_delay
                )

        if (
            index >= args.warmup_runs
            and record is not None
        ):

            record[
                "run"
            ] = logical_run

            records.append(
                record
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary: dict[str, Any] = {}

    for metric in _METRICS:

        values = [
            float(
                record[metric]
            )
            for record in records
            if metric in record
        ]

        if metric == "rtf":

            summary[
                metric
            ] = _summary_rtf(
                values
            )

        else:

            summary[
                metric
            ] = _summary(
                values
            )

    summary[
        "partial_count"
    ] = _summary(
        [
            float(
                record[
                    "partial_count"
                ]
            )
            for record in records
            if "partial_count" in record
        ]
    )

    summary[
        "chunk_count"
    ] = _summary(
        [
            float(
                record[
                    "chunk_count"
                ]
            )
            for record in records
            if "chunk_count" in record
        ]
    )

    return {
        "runs": records,
        "failures": failures,
        "summary": summary,
    }


# ============================================================
# CLI
# ============================================================

def _parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the project's real "
            "ArkASR streaming path"
        )
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--chunk-ms",
        type=_positive_int,
        default=DEFAULT_CHUNK_MS,
        help=(
            "streaming audio chunk size in ms; "
            "project default is 200 ms"
        ),
    )

    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pace audio in realtime; "
            "disable only for throughput testing"
        ),
    )

    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=10,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--max-attempts-per-run",
        type=_positive_int,
        default=1,
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    return parser


# ============================================================
# Main
# ============================================================

def main(
    argv: list[str] | None = None,
) -> int:

    args = _parser().parse_args(
        argv
    )

    if args.warmup_runs < 0:

        _parser().error(
            "--warmup-runs must be "
            "zero or greater"
        )

    if args.retry_delay < 0:

        _parser().error(
            "--retry-delay must be "
            "zero or greater"
        )

    # --------------------------------------------------------
    # Read audio
    # --------------------------------------------------------

    pcm16 = _read_pcm16(
        args.audio
    )

    # --------------------------------------------------------
    # Record configuration
    # --------------------------------------------------------

    requested = {
        "audio": str(
            args.audio
        ),
        "chunk_ms": args.chunk_ms,
        "realtime": args.realtime,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "timeout": args.timeout,
        "max_attempts_per_run":
            args.max_attempts_per_run,
        "retry_delay":
            args.retry_delay,
        "audio_duration_ms":
            round(
                _audio_duration_ms(
                    pcm16
                ),
                2,
            ),
    }

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    result = asyncio.run(
        _run_benchmark(
            args,
            pcm16,
        )
    )

    output = {

        "benchmark":
            "real_streaming_asr",

        "requested":
            requested,

        "actual_path":
            (
                "audio chunks -> "
                "ArkASRStreamingSession -> "
                "BigASR streaming WebSocket -> "
                "partial/final callbacks"
            ),

        "metric_definitions": {

            "stream_start_ms":
                "ArkASRStreamingSession.start() duration",

            "first_partial_ms":
                "first audio chunk -> first non-final ASR result",

            "first_final_ms":
                "first audio chunk -> first final ASR result",

            "finish_to_final_ms":
                "finish() start -> first final ASR result",

            "stream_wall_ms":
                "first audio chunk -> first final ASR result",

            "rtf":
                "stream_wall_ms / audio_duration_ms",

            "partial_count":
                "number of non-final ASR results",

            "chunk_count":
                "number of audio chunks fed into the streaming ASR",
        },

        "notes": [
            (
                "This benchmark intentionally does not use "
                "manual_audio."
            ),
            (
                "The project RealtimeCompanion uses "
                "asr_chunk_s=0.2, therefore the default "
                "benchmark chunk size is 200 ms."
            ),
            (
                "realtime=true is required for user-facing "
                "streaming latency. Sending all chunks "
                "immediately measures throughput rather "
                "than live responsiveness."
            ),
            (
                "first_partial_ms measures streaming ASR "
                "responsiveness while audio is still being "
                "fed."
            ),
            (
                "first_final_ms measures complete recognition "
                "latency from the beginning of the streamed "
                "utterance."
            ),
        ],

        **result,
    }

    encoded = json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    )

    if args.output:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            encoded + "\n",
            encoding="utf-8",
        )

    print(encoded)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
