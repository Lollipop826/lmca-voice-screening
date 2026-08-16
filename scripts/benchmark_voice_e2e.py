"""Measure one complete authenticated voice turn over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import uuid
import wave
from array import array
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


_METRICS = (
    "session_start_ms",
    "input_to_vad_end_ms",
    "input_to_asr_result_ms",
    "input_to_first_ai_text_ms",
    "input_to_tts_start_ms",
    "input_to_first_tts_byte_ms",
    "input_to_tts_end_ms",
    "asr_to_first_ai_text_ms",
    "asr_to_first_tts_byte_ms",
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def percentile(ratio: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1)]
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "max_ms": round(ordered[-1], 2),
    }


def _read_pcm16(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        if (
            params.nchannels,
            params.sampwidth,
            params.framerate,
            params.comptype,
        ) != (1, 2, 16000, "NONE"):
            raise ValueError(
                "audio must be mono, PCM16, 16000 Hz WAV; "
                f"got channels={params.nchannels}, width={params.sampwidth}, "
                f"rate={params.framerate}, compression={params.comptype}"
            )
        data = source.readframes(params.nframes)
    if not data:
        raise ValueError("audio file is empty")
    return data


def _websocket_url(server: str) -> str:
    parsed = urlsplit(server.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("server must be an http(s) URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws"


def _connect_ws(websocket_url: str, cookie: str):
    import websockets

    headers = {"Cookie": cookie}
    try:
        return websockets.connect(
            websocket_url,
            additional_headers=headers,
            proxy=None,
            open_timeout=10,
        )
    except TypeError:
        return websockets.connect(
            websocket_url,
            extra_headers=headers,
            open_timeout=10,
        )


async def _recv_json(websocket, deadline: float) -> dict[str, Any]:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("WebSocket event timeout")
        raw = await asyncio.wait_for(websocket.recv(), remaining)
        if isinstance(raw, bytes):
            continue
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            return event


async def _wait_for_type(websocket, event_type: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while True:
        event = await _recv_json(websocket, deadline)
        if event.get("type") == event_type:
            return event


async def _prepare_session(
    websocket,
    *,
    patient_id: str | None,
    new_patient: bool,
    profile_name: str,
    long_term_memory_enabled: bool,
    timeout_s: float,
) -> tuple[str | None, float]:
    await _wait_for_type(websocket, "waiting_for_info", timeout_s)
    payload: dict[str, Any] = {
        "type": "start_session",
        "force_new_session": True,
        "long_term_memory_enabled": long_term_memory_enabled,
    }
    if patient_id:
        payload["patient_id"] = patient_id
    if new_patient:
        payload["profile"] = {
            "name": profile_name,
            "age": 70,
            "gender": "female",
            "education_years": 6,
        }
    started_at = time.perf_counter()
    await websocket.send(json.dumps(payload, ensure_ascii=False))

    deadline = time.perf_counter() + timeout_s
    started = False
    resolved_patient_id = patient_id
    session_started_at = 0.0
    greeting_expected = new_patient
    greeting_finished = not greeting_expected
    while not started or not greeting_finished:
        event = await _recv_json(websocket, deadline)
        event_type = event.get("type")
        if event_type == "patient_memory":
            resolved_patient_id = str(event.get("patient_id") or "") or resolved_patient_id
        elif event_type == "session_started":
            started = True
            session_started_at = time.perf_counter()
        elif greeting_expected and event_type == "tts_end":
            greeting_finished = True
    return resolved_patient_id, (session_started_at - started_at) * 1000


async def _run_once(
    *,
    websocket_url: str,
    cookie: str,
    pcm16: bytes,
    timeout_s: float,
    patient_id: str | None,
    new_patient: bool,
    profile_name: str,
    long_term_memory_enabled: bool,
    audio_output: Path | None,
) -> dict[str, Any]:
    async with _connect_ws(websocket_url, cookie) as websocket:
        resolved_patient_id, session_start_ms = await _prepare_session(
            websocket,
            patient_id=patient_id,
            new_patient=new_patient,
            profile_name=profile_name,
            long_term_memory_enabled=long_term_memory_enabled,
            timeout_s=timeout_s,
        )
        started_at = time.perf_counter()
        await websocket.send(
            json.dumps(
                {
                    "type": "manual_audio",
                    "audio": base64.b64encode(pcm16).decode("ascii"),
                    "sample_rate": 16000,
                }
            )
        )
        deadline = started_at + timeout_s
        marks: dict[str, float] = {}
        turn_id = ""
        final_insight = None
        ai_text = ""
        tts_audio = bytearray()
        while "input_to_tts_end" not in marks or final_insight is None:
            event = await _recv_json(websocket, deadline)
            now = time.perf_counter()
            event_type = event.get("type")
            if event_type == "turn_insight" and event.get("state") == "final":
                final_insight = event
                continue
            if event_type == "vad_end":
                marks.setdefault("input_to_vad_end", now)
                continue
            if event_type == "asr_error":
                raise RuntimeError(str(event.get("detail") or "ASR failed"))
            if event_type == "asr_result":
                turn_id = str(event.get("turn_id") or "")
                if turn_id:
                    marks.setdefault("input_to_asr_result", now)
                continue
            if not turn_id or str(event.get("turn_id") or "") != turn_id:
                continue
            if event_type == "ai_response_chunk":
                if event.get("is_first") or int(event.get("sentence_index") or 0) == 1:
                    marks.setdefault("input_to_first_ai_text", now)
                ai_text += str(event.get("text") or "")
            elif event_type == "ai_response":
                if event.get("error"):
                    raise RuntimeError(
                        str(event.get("text") or "AI response failed")
                    )
                marks.setdefault("input_to_first_ai_text", now)
                if not ai_text:
                    ai_text = str(event.get("text") or "")
            elif event_type == "tts_start":
                marks.setdefault("input_to_tts_start", now)
            elif event_type in {"tts_chunk", "tts_audio"}:
                marks.setdefault("input_to_first_tts_byte", now)
                chunk = event.get("chunk") or event.get("audio") or event.get("data")
                if chunk:
                    tts_audio.extend(base64.b64decode(chunk))
            elif event_type == "tts_end":
                marks.setdefault("input_to_tts_end", now)

    if audio_output and tts_audio:
        samples = array("f")
        samples.frombytes(tts_audio)
        pcm16 = array(
            "h",
            (round(max(-1.0, min(1.0, sample)) * 32767) for sample in samples),
        )
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_output), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(24000)
            target.writeframes(pcm16.tobytes())
    record = {
        "patient_id": resolved_patient_id,
        "turn_id": turn_id,
        "session_start_ms": round(session_start_ms, 2),
        "emotion": final_insight.get("emotion", {}),
        "ai_text": ai_text,
        "tts_audio": str(audio_output) if audio_output and tts_audio else None,
        **{
            metric: round((marks[key] - started_at) * 1000, 2)
            for metric, key in (
                ("input_to_vad_end_ms", "input_to_vad_end"),
                ("input_to_asr_result_ms", "input_to_asr_result"),
                ("input_to_first_ai_text_ms", "input_to_first_ai_text"),
                ("input_to_tts_start_ms", "input_to_tts_start"),
                ("input_to_first_tts_byte_ms", "input_to_first_tts_byte"),
                ("input_to_tts_end_ms", "input_to_tts_end"),
            )
            if key in marks
        },
    }
    if "input_to_asr_result_ms" in record:
        for target, source in (
            ("asr_to_first_ai_text_ms", "input_to_first_ai_text_ms"),
            ("asr_to_first_tts_byte_ms", "input_to_first_tts_byte_ms"),
        ):
            if source in record:
                record[target] = round(record[source] - record["input_to_asr_result_ms"], 2)
    return record


def _login(server: str, username: str, password: str) -> str:
    import requests

    client = requests.Session()
    client.trust_env = False
    response = client.post(
        f"{server.rstrip('/')}/api/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError("login failed")
    cookie = client.cookies.get("aa_session")
    if not cookie:
        raise RuntimeError("login response did not set aa_session")
    return f"aa_session={cookie}"


def _server_flags(server: str) -> dict[str, Any]:
    import requests

    response = requests.get(f"{server.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return dict(response.json().get("release_flags") or {})


async def _run_benchmark(args, pcm16: bytes, cookie: str) -> dict[str, Any]:
    records = []
    failures = []
    websocket_url = _websocket_url(args.server)
    total_runs = args.warmup_runs + args.runs
    for index in range(total_runs):
        phase = "warmup" if index < args.warmup_runs else "measured"
        logical_run = index + 1 if phase == "warmup" else index - args.warmup_runs + 1
        for attempt in range(1, args.max_attempts_per_run + 1):
            try:
                record = await _run_once(
                    websocket_url=websocket_url,
                    cookie=cookie,
                    pcm16=pcm16,
                    timeout_s=args.timeout,
                    patient_id=args.patient_id,
                    new_patient=args.new_patient,
                    profile_name=f"{args.profile_name}-{index + 1:03d}-{uuid.uuid4().hex[:6]}",
                    long_term_memory_enabled=args.long_term_memory == "on",
                    audio_output=(
                        args.audio_output_dir / f"{args.long_term_memory}-{phase}-{logical_run}.wav"
                        if args.audio_output_dir else None
                    ),
                )
                emotion = record.get("emotion", {})
                if args.require_audio_emotion and (
                    emotion.get("source") != "emotion2vec_audio+text"
                    or not emotion.get("audio_model_used")
                ):
                    raise RuntimeError("audio emotion verification failed")
                break
            except Exception as exc:
                failures.append({
                    "phase": phase,
                    "logical_run": logical_run,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                })
                if attempt == args.max_attempts_per_run:
                    raise
                await asyncio.sleep(args.retry_delay)
        if index >= args.warmup_runs:
            records.append(record)
    summary = {
        metric: _summary(
            [float(record[metric]) for record in records if metric in record]
        )
        for metric in _METRICS
    }
    return {"runs": records, "failures": failures, "summary": summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authenticated voice WebSocket latency benchmark")
    parser.add_argument("--live", action="store_true", help="perform network calls")
    parser.add_argument("--server", default="http://127.0.0.1:8502")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    patient = parser.add_mutually_exclusive_group(required=True)
    patient.add_argument("--patient-id")
    patient.add_argument(
        "--new-patient",
        action="store_true",
        help="create synthetic patients; this writes benchmark data to the server DB",
    )
    parser.add_argument("--profile-name", default="voice-benchmark")
    parser.add_argument("--long-term-memory", choices=("on", "off"), default="on")
    parser.add_argument("--audio-output-dir", type=Path)
    parser.add_argument("--runs", type=_positive_int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-attempts-per-run", type=_positive_int, default=1)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--expect-long-term-memory",
        choices=("on", "off"),
        help="fail if the server long-term-memory flag does not match",
    )
    parser.add_argument(
        "--expect-long-term-memory-writes",
        choices=("on", "off"),
        help="fail if the server long-term-memory write flag does not match",
    )
    parser.add_argument(
        "--require-audio-emotion",
        action="store_true",
        help="fail unless every run completes Emotion2Vec audio inference",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup_runs < 0:
        _parser().error("--warmup-runs must be zero or greater")
    if args.retry_delay < 0:
        _parser().error("--retry-delay must be zero or greater")
    requested = {
        "server": args.server,
        "audio": str(args.audio),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "max_attempts_per_run": args.max_attempts_per_run,
        "retry_delay": args.retry_delay,
        "patient_id": args.patient_id,
        "new_patient": args.new_patient,
        "expect_long_term_memory": args.expect_long_term_memory,
        "expect_long_term_memory_writes": args.expect_long_term_memory_writes,
        "require_audio_emotion": args.require_audio_emotion,
        "long_term_memory": args.long_term_memory,
    }
    if not args.live:
        print(json.dumps({"live": False, "requested": requested}, ensure_ascii=False))
        return 0
    pcm16 = _read_pcm16(args.audio)
    server_flags = _server_flags(args.server)
    if not server_flags.get("turn_insight"):
        raise RuntimeError("server must enable ENABLE_TURN_INSIGHT for emotion verification")
    expected_memory = args.expect_long_term_memory
    if expected_memory and bool(server_flags.get("long_term_memory")) != (expected_memory == "on"):
        raise RuntimeError(
            "server long-term-memory state does not match "
            f"--expect-long-term-memory={expected_memory}"
        )
    expected_writes = args.expect_long_term_memory_writes
    if expected_writes and bool(server_flags.get("long_term_memory_writes")) != (expected_writes == "on"):
        raise RuntimeError(
            "server long-term-memory write state does not match "
            f"--expect-long-term-memory-writes={expected_writes}"
        )
    cookie = _login(args.server, args.username, args.password)
    result = {"live": True, "requested": requested, "server_flags": server_flags}
    result.update(asyncio.run(_run_benchmark(args, pcm16, cookie)))
    if args.require_audio_emotion:
        invalid = [
            record.get("turn_id") or "unknown"
            for record in result["runs"]
            if record.get("emotion", {}).get("source") != "emotion2vec_audio+text"
            or not record.get("emotion", {}).get("audio_model_used")
        ]
        if invalid:
            raise RuntimeError(
                "audio emotion verification failed for turns: " + ", ".join(invalid)
            )
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
