#!/usr/bin/env python3
"""
Measure true streaming BigASR latency.

Pipeline:
    WAV -> realtime PCM chunks -> ArkASRStreamingSession.feed()
        -> first partial result
        -> final result

This benchmark measures ASR itself and does not involve:
    VAD / LLM / TTS / memory / emotion / voice_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.voice.ark_asr import ArkASRStreamingSession


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        raise ValueError(
            f"要求单声道 WAV，当前 channels={channels}"
        )

    if sample_width != 2:
        raise ValueError(
            f"要求 16-bit PCM WAV，当前 sample_width={sample_width}"
        )

    if sample_rate != 16000:
        raise ValueError(
            f"要求 16kHz WAV，当前 sample_rate={sample_rate}"
        )

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0

    return audio, sample_rate


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
            "std": None,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


async def run_once(
    audio: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
    realtime: bool,
    run_id: int,
) -> dict[str, Any]:

    chunk_samples = int(sample_rate * chunk_ms / 1000)

    if chunk_samples <= 0:
        raise ValueError("chunk_ms 太小")

    timestamps: dict[str, float | None] = {
        "connect_start": None,
        "connected": None,
        "first_audio_send": None,
        "last_audio_send": None,
        "first_partial": None,
        "final_result": None,
    }

    partial_results: list[dict[str, Any]] = []
    final_text = ""

    loop = asyncio.get_running_loop()

    def on_result(text: str, is_last: bool):
        nonlocal final_text

        now = loop.time()
        text = str(text or "").strip()

        if not text:
            return

        if timestamps["first_partial"] is None and not is_last:
            timestamps["first_partial"] = now

            partial_results.append(
                {
                    "t_ms": (
                        now - timestamps["first_audio_send"]
                    ) * 1000.0,
                    "text": text,
                }
            )

            print(
                f"[run {run_id:02d}] "
                f"[PARTIAL] "
                f"{text}"
            )

        if is_last:
            final_text = text
            timestamps["final_result"] = now

            print(
                f"[run {run_id:02d}] "
                f"[FINAL] "
                f"{text}"
            )

    session = ArkASRStreamingSession(
        sample_rate=sample_rate,
        on_result=on_result,
    )

    try:
        timestamps["connect_start"] = loop.time()

        await session.start()

        timestamps["connected"] = loop.time()

        print(
            f"[run {run_id:02d}] "
            f"connected"
        )

        total_samples = len(audio)
        offset = 0

        while offset < total_samples:
            end = min(
                offset + chunk_samples,
                total_samples,
            )

            chunk = audio[offset:end]

            now = loop.time()

            if timestamps["first_audio_send"] is None:
                timestamps["first_audio_send"] = now

            await session.feed(chunk)

            timestamps["last_audio_send"] = loop.time()

            offset = end

            if realtime:
                actual_chunk_duration = len(chunk) / sample_rate
                await asyncio.sleep(actual_chunk_duration)

        # Mark the end of user speech and wait for final ASR.
        await session.finish()

        result: dict[str, Any] = {
            "run": run_id,
            "success": True,
            "audio_duration_ms": len(audio) / sample_rate * 1000.0,
            "chunk_ms": chunk_ms,
            "chunk_count": (
                total_samples + chunk_samples - 1
            ) // chunk_samples,
            "partial_count": len(partial_results),
            "partial_results": partial_results,
            "final_text": final_text or session.text,
        }

        connect_start = timestamps["connect_start"]
        connected = timestamps["connected"]
        first_audio = timestamps["first_audio_send"]
        last_audio = timestamps["last_audio_send"]
        first_partial = timestamps["first_partial"]
        final_result = timestamps["final_result"]

        if connect_start is not None and connected is not None:
            result["connect_latency_ms"] = (
                connected - connect_start
            ) * 1000.0

        # From the beginning of audio transmission to the first ASR partial.
        if first_audio is not None and first_partial is not None:
            result["first_partial_latency_ms"] = (
                first_partial - first_audio
            ) * 1000.0
        else:
            result["first_partial_latency_ms"] = None

        # From WebSocket connection establishment to the first ASR partial.
        # This removes connection setup time from the ASR streaming response latency.
        if connected is not None and first_partial is not None:
            result["first_partial_after_connect_ms"] = (
                first_partial - connected
            ) * 1000.0
        else:
            result["first_partial_after_connect_ms"] = None

        # From the beginning of audio transmission to completion of all audio sends.
        if first_audio is not None and last_audio is not None:
            result["audio_send_complete_latency_ms"] = (
                last_audio - first_audio
            ) * 1000.0
        else:
            result["audio_send_complete_latency_ms"] = None

        if first_audio is not None and final_result is not None:
            result["final_e2e_latency_ms"] = (
                final_result - first_audio
            ) * 1000.0
        else:
            result["final_e2e_latency_ms"] = None

        if last_audio is not None and final_result is not None:
            result["final_processing_latency_ms"] = (
                final_result - last_audio
            ) * 1000.0
        else:
            result["final_processing_latency_ms"] = None

        if first_partial is not None and final_result is not None:
            result["partial_to_final_ms"] = (
                final_result - first_partial
            ) * 1000.0
        else:
            result["partial_to_final_ms"] = None

        print(
            f"[run {run_id:02d}] "
            f"first_partial="
            f"{result['first_partial_latency_ms']}"
            f" ms | "
            f"final_e2e="
            f"{result['final_e2e_latency_ms']}"
            f" ms | "
            f"final_processing="
            f"{result['final_processing_latency_ms']}"
            f" ms"
        )

        return result

    except Exception as exc:
        print(
            f"[run {run_id:02d}] ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        try:
            await session.aclose()
        except Exception:
            pass

        return {
            "run": run_id,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:

    success = [
        r for r in results
        if r.get("success")
    ]

    def collect(name: str) -> list[float]:
        values = []

        for r in success:
            value = r.get(name)

            if isinstance(value, (int, float)):
                values.append(float(value))

        return values

    return {
        "total_runs": len(results),
        "successful_runs": len(success),
        "failed_runs": len(results) - len(success),

        "connect_latency_ms": stats(
            collect("connect_latency_ms")
        ),

        "first_partial_latency_ms": stats(
            collect("first_partial_latency_ms")
        ),

        "first_partial_after_connect_ms": stats(
            collect("first_partial_after_connect_ms")
        ),

        "audio_send_complete_latency_ms": stats(
            collect("audio_send_complete_latency_ms")
        ),

        "final_e2e_latency_ms": stats(
            collect("final_e2e_latency_ms")
        ),

        "final_processing_latency_ms": stats(
            collect("final_processing_latency_ms")
        ),

        "partial_to_final_ms": stats(
            collect("partial_to_final_ms")
        ),

        "partial_available_runs": sum(
            1
            for r in success
            if r.get("first_partial_latency_ms") is not None
        ),
    }


async def main_async(args):
    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"音频不存在: {audio_path}"
        )

    audio, sample_rate = read_wav(audio_path)

    print("=" * 70)
    print("True Streaming BigASR Benchmark")
    print("=" * 70)
    print(f"audio       : {audio_path}")
    print(f"sample_rate : {sample_rate}")
    print(
        f"duration    : "
        f"{len(audio) / sample_rate:.3f}s"
    )
    print(f"chunk_ms    : {args.chunk_ms}")
    print(f"realtime    : {args.realtime}")
    print(f"runs        : {args.runs}")
    print("=" * 70)

    results = []

    for run_id in range(1, args.runs + 1):

        if run_id > 1:
            await asyncio.sleep(args.interval)

        result = await run_once(
            audio=audio,
            sample_rate=sample_rate,
            chunk_ms=args.chunk_ms,
            realtime=args.realtime,
            run_id=run_id,
        )

        results.append(result)

    summary = build_summary(results)

    output = {
        "benchmark": "ark_asr_true_streaming",
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "chunk_ms": args.chunk_ms,
        "realtime": args.realtime,
        "runs": args.runs,
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for metric in (
        "connect_latency_ms",
        "first_partial_latency_ms",
        "first_partial_after_connect_ms",
        "audio_send_complete_latency_ms",
        "final_e2e_latency_ms",
        "final_processing_latency_ms",
        "partial_to_final_ms",
    ):
        print(
            f"{metric}: "
            f"{json.dumps(summary[metric], ensure_ascii=False)}"
        )

    print(
        "partial_available_runs: "
        f"{summary['partial_available_runs']}"
        f"/{summary['successful_runs']}"
    )

    print()
    print(f"结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        required=True,
    )

    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="实时音频分片大小，默认100ms",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按照真实音频时钟发送chunk",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="两次测试之间的间隔秒数",
    )

    parser.add_argument(
        "--output",
        default="output/benchmark_ark_asr_streaming.json",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
