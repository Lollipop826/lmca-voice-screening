"""Send fixture PCM through SoulX without creating application/patient records."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.voice.soulx_turn_taking import SoulXTurnTakingClient
from src.voice.health import SoulXHealthProbe


async def check(args) -> dict:
    health = await SoulXHealthProbe(args.url).check()
    if not health["available"]:
        raise RuntimeError(f"SoulX is not ready: {health['error']}")
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError("The fixture must use a 16000 Hz sample rate")
    audio = audio.mean(axis=1)
    audio = np.concatenate((np.zeros(10240), audio, np.zeros(40960))).astype(np.float32)
    chunk_samples = 2560
    audio = np.pad(audio, (0, (-len(audio)) % chunk_samples))
    client = SoulXTurnTakingClient(args.url, timeout=args.timeout)
    silent_client = SoulXTurnTakingClient(args.url, timeout=args.timeout)
    durations = []
    states = Counter()
    finals = []
    silence_leaks = []
    try:
        for offset in range(0, len(audio), chunk_samples):
            started = time.perf_counter()
            result = await client.process(audio[offset:offset + chunk_samples])
            durations.append((time.perf_counter() - started) * 1000)
            states[result.state] += 1
            if result.state == "speak":
                finals.append({
                    "audio_end_s": (offset + chunk_samples) / sample_rate,
                    "text": result.text,
                    "detail_state": result.detail_state,
                })
            if args.check_isolation:
                silent = await silent_client.process(np.zeros(chunk_samples, dtype=np.float32))
                if silent.state in {"nonidle", "speak"} or silent.text or silent.asr_buffer:
                    silence_leaks.append({"state": silent.state, "text": silent.text or silent.asr_buffer})
    finally:
        await client.close()
        await silent_client.close()
    report = {
        "url": args.url,
        "audio_s": len(audio) / sample_rate,
        "frames": len(durations),
        "states": dict(states),
        "finals": finals,
        "roundtrip_ms": {
            "median": round(float(np.median(durations)), 1),
            "p95": round(float(np.percentile(durations, 95)), 1),
            "max": round(max(durations), 1),
        },
        "processing_to_audio_ratio": round(sum(durations) / (len(audio) / sample_rate * 1000), 3),
        "isolation_checked": args.check_isolation,
        "silence_leaks": silence_leaks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not states["nonidle"] or not any(item["text"] for item in finals):
        raise RuntimeError("Fixture did not produce speech detection and a nonempty completed turn")
    if silence_leaks:
        raise RuntimeError("Speech appeared in an independent silence-only session")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8001/turn")
    parser.add_argument("--wav", type=Path, default=ROOT / "tests/fixtures/bench_speech_zh_8s.wav")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--check-isolation", action="store_true")
    asyncio.run(check(parser.parse_args()))
