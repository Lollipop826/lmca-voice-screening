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


# ============================================================
# Official four core latency metrics
# ============================================================
#
# 1. ASR latency
#    VAD_END -> ASR_RESULT
#
# 2. LLM first response latency
#    ASR_RESULT -> first AI response event
#
# 3. TTS first audio latency
#    first AI response event -> first TTS audio byte
#
# 4. Complete voice response latency
#    first AI response event -> TTS_END
#
# IMPORTANT:
# "LLM first response" is NOT an internal LLM token-generation
# timestamp. It is the first AI response event observed by the
# benchmark through WebSocket.
# ============================================================

_METRICS = (
    "asr_first_partial_latency_ms",
    "asr_final_latency_ms",
    "asr_partial_to_final_ms",
    "llm_first_response_latency_ms",
    "tts_first_audio_latency_ms",
    "llm_first_text_to_tts_end_ms",
)
# ============================================================
# Basic utilities
# ============================================================

def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _summary(values: list[float]) -> dict[str, float | int]:
    """
    Calculate mean / p50 / p95 / max.

    The percentile implementation keeps the same behavior as
    the original benchmark script so that the existing result
    processing remains comparable.
    """

    ordered = sorted(values)

    if not ordered:
        return {"count": 0}

    def percentile(ratio: float) -> float:
        return ordered[
            min(
                len(ordered) - 1,
                math.ceil(len(ordered) * ratio) - 1,
            )
        ]

    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "max_ms": round(ordered[-1], 2),
    }


def _read_pcm16(path: str | Path) -> bytes:
    """
    Read and validate benchmark audio.

    Required format:
        mono
        PCM16
        16000 Hz
        uncompressed WAV
    """

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
                f"got channels={params.nchannels}, "
                f"width={params.sampwidth}, "
                f"rate={params.framerate}, "
                f"compression={params.comptype}"
            )

        data = source.readframes(params.nframes)

    if not data:
        raise ValueError("audio file is empty")

    return data


def _websocket_url(server: str) -> str:
    """
    Convert HTTP(S) server URL to the project's WebSocket URL.
    """

    parsed = urlsplit(server.rstrip("/"))

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("server must be an http(s) URL")

    scheme = "wss" if parsed.scheme == "https" else "ws"

    return f"{scheme}://{parsed.netloc}/ws"


# ============================================================
# WebSocket connection
# ============================================================

def _connect_ws(websocket_url: str, cookie: str):
    import websockets
    import ssl

    headers = {
        "Cookie": cookie,
    }

    ssl_context = ssl._create_unverified_context()

    # websockets >= 14
    try:
        return websockets.connect(
            websocket_url,
            additional_headers=headers,
            proxy=None,
            open_timeout=10,
            ssl=ssl_context,
        )

    # Older websockets versions
    except TypeError:
        return websockets.connect(
            websocket_url,
            extra_headers=headers,
            open_timeout=10,
            ssl=ssl_context,
        )


async def _recv_json(
    websocket,
    deadline: float,
) -> dict[str, Any]:
    """
    Receive the next JSON object from WebSocket.

    Binary frames are ignored here because TTS audio is delivered
    as JSON events by the current server protocol.
    """

    while True:

        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            raise TimeoutError(
                "WebSocket event timeout"
            )

        raw = await asyncio.wait_for(
            websocket.recv(),
            remaining,
        )

        if isinstance(raw, bytes):
            continue

        try:
            event = json.loads(raw)

        except (TypeError, json.JSONDecodeError):
            continue

        if isinstance(event, dict):
            return event


async def _wait_for_type(
    websocket,
    event_type: str,
    timeout_s: float,
) -> dict[str, Any]:

    deadline = time.perf_counter() + timeout_s

    while True:

        event = await _recv_json(
            websocket,
            deadline,
        )

        if event.get("type") == event_type:
            return event


# ============================================================
# Session preparation
# ============================================================

async def _prepare_session(
    websocket,
    *,
    patient_id: str | None,
    new_patient: bool,
    profile_name: str,
    long_term_memory_enabled: bool,
    long_term_memory_writes_enabled: bool | None,
    emotion_enabled: bool,
    timeout_s: float,
) -> tuple[str | None, float]:

    await _wait_for_type(
        websocket,
        "waiting_for_info",
        timeout_s,
    )

    payload: dict[str, Any] = {
        "type": "start_session",
        "force_new_session": True,
        "long_term_memory_enabled":
            long_term_memory_enabled,
        "emotion_enabled": emotion_enabled,
    }
    if long_term_memory_writes_enabled is not None:
        payload["long_term_memory_writes_enabled"] = (
            long_term_memory_writes_enabled
        )

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

    await websocket.send(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    deadline = (
        time.perf_counter()
        + timeout_s
    )

    started = False

    resolved_patient_id = patient_id

    session_started_at = 0.0
    session_started_event: dict[str, Any] = {}

    greeting_expected = new_patient

    greeting_finished = not greeting_expected

    while not started or not greeting_finished:

        event = await _recv_json(
            websocket,
            deadline,
        )

        event_type = event.get("type")

        if event_type == "patient_memory":

            resolved_patient_id = (
                str(
                    event.get("patient_id")
                    or ""
                )
                or resolved_patient_id
            )

        elif event_type == "session_started":

            started = True

            session_started_at = (
                time.perf_counter()
            )
            session_started_event = event

        elif (
            greeting_expected
            and event_type == "tts_end"
        ):

            greeting_finished = True

    requested_flags: dict[str, bool] = {
        "long_term_memory_enabled": long_term_memory_enabled,
        "emotion_enabled": emotion_enabled,
    }
    if long_term_memory_writes_enabled is not None:
        requested_flags["long_term_memory_writes_enabled"] = (
            long_term_memory_writes_enabled
        )
    missing = [name for name in requested_flags if name not in session_started_event]
    mismatched = [
        name
        for name, expected in requested_flags.items()
        if name in session_started_event
        and bool(session_started_event[name]) is not bool(expected)
    ]
    if missing or mismatched:
        raise RuntimeError(
            "session ablation flags were not applied: "
            f"missing={missing}, mismatched={mismatched}"
        )

    return (
        resolved_patient_id,
        (session_started_at - started_at)
        * 1000,
    )


# ============================================================
# One benchmark run
# ============================================================

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
    long_term_memory_writes_enabled: bool | None,
    emotion_enabled: bool,
    require_final_insight: bool,
    audio_output: Path | None,
) -> dict[str, Any]:

    async with _connect_ws(
        websocket_url,
        cookie,
    ) as websocket:

        # ----------------------------------------------------
        # Prepare authenticated session
        # ----------------------------------------------------

        (
            resolved_patient_id,
            session_start_ms,
        ) = await _prepare_session(
            websocket,
            patient_id=patient_id,
            new_patient=new_patient,
            profile_name=profile_name,
            long_term_memory_enabled=
                long_term_memory_enabled,
            long_term_memory_writes_enabled=
                long_term_memory_writes_enabled,
            emotion_enabled=emotion_enabled,
            timeout_s=timeout_s,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # This is the actual start point of the measured
        # user voice turn.
        # ----------------------------------------------------

        started_at = time.perf_counter()

        await websocket.send(
            json.dumps(
                {
                    "type": "manual_audio",
                    "audio": base64.b64encode(
                        pcm16
                    ).decode("ascii"),
                    "sample_rate": 16000,
                }
            )
        )

        deadline = (
            started_at
            + timeout_s
        )

        # ----------------------------------------------------
        # Raw event timestamps
        # ----------------------------------------------------

        marks: dict[str, float] = {}

        turn_id = ""

        final_insight = None

        ai_text = ""

        tts_audio = bytearray()

        # ----------------------------------------------------
        # Receive events until:
        #
        #   TTS_END
        #
        # and
        #
        #   final turn_insight
        #
        # have both arrived.
        # ----------------------------------------------------

        while (
            "input_to_tts_end" not in marks
            or (require_final_insight and final_insight is None)
        ):

            event = await _recv_json(
                websocket,
                deadline,
            )

            now = time.perf_counter()

            event_type = event.get("type")

            # ------------------------------------------------
            # Final turn insight
            # ------------------------------------------------

            if (
                event_type == "turn_insight"
                and event.get("state") == "final"
            ):

                final_insight = event

                continue

            # ------------------------------------------------
            # VAD END
            #
            # This is the end point of speech activity detection.
            # ------------------------------------------------

            if event_type == "vad_end":

                marks.setdefault(
                    "input_to_vad_end",
                    now,
                )

                continue

            # ------------------------------------------------
            # ASR error
            # ------------------------------------------------

            if event_type == "asr_error":

                raise RuntimeError(
                    str(
                        event.get("detail")
                        or "ASR failed"
                    )
                )

            if event_type == "asr_partial":

                partial_text = str(
                    event.get("text")
                    or ""
                ).strip()

                partial_is_final = bool(
                    event.get("final", False)
                )

                if (
                    partial_text
                    and not partial_is_final
                ):
                    marks.setdefault(
                        "input_to_asr_first_partial",
                        now,
                    )

                continue
            # ------------------------------------------------
            # ASR RESULT
            #
            # We only accept the ASR event that contains a
            # valid turn_id, because later AI/TTS events are
            # matched against this turn.
            # ------------------------------------------------

            if event_type == "asr_result":

                current_turn_id = str(
                    event.get("turn_id")
                    or ""
                )

                if current_turn_id:

                    turn_id = current_turn_id

                    marks.setdefault(
                        "input_to_asr_result",
                        now,
                    )

                continue

            # ------------------------------------------------
            # Ignore events belonging to another turn.
            # ------------------------------------------------

            if (
                not turn_id
                or str(
                    event.get("turn_id")
                    or ""
                ) != turn_id
            ):
                continue

            # =================================================
            # AI RESPONSE
            # =================================================

            if event_type == "ai_response_chunk":

                # ---------------------------------------------
                # First AI response event
                #
                # Preferred:
                #   is_first == true
                #
                # Fallback:
                #   sentence_index == 1
                # ---------------------------------------------

                is_first = bool(
                    event.get("is_first")
                )

                sentence_index = (
                    event.get("sentence_index")
                )

                sentence_is_first = False

                if sentence_index is not None:

                    try:

                        sentence_is_first = (
                            int(sentence_index)
                            == 1
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        sentence_is_first = False

                if (
                    is_first
                    or sentence_is_first
                ):

                    marks.setdefault(
                        "input_to_first_ai_text",
                        now,
                    )

                ai_text += str(
                    event.get("text")
                    or ""
                )

            # ------------------------------------------------
            # Non-streaming AI response
            # ------------------------------------------------

            elif event_type == "ai_response":

                if event.get("error"):

                    raise RuntimeError(
                        str(
                            event.get("text")
                            or "AI response failed"
                        )
                    )

                marks.setdefault(
                    "input_to_first_ai_text",
                    now,
                )

                if not ai_text:

                    ai_text = str(
                        event.get("text")
                        or ""
                    )

            # =================================================
            # TTS START
            # =================================================

            elif event_type == "tts_start":

                marks.setdefault(
                    "input_to_tts_start",
                    now,
                )

            # =================================================
            # TTS FIRST AUDIO
            # =================================================

            elif event_type in {
                "tts_chunk",
                "tts_audio",
            }:

                # First TTS audio event.
                #
                # This is the timestamp used for:
                #
                # AI first response
                #       ->
                # first TTS audio
                #
                marks.setdefault(
                    "input_to_first_tts_byte",
                    now,
                )

                chunk = (
                    event.get("chunk")
                    or event.get("audio")
                    or event.get("data")
                )

                if chunk:

                    try:

                        tts_audio.extend(
                            base64.b64decode(
                                chunk
                            )
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):

                        # Keep latency measurement valid
                        # even if an individual audio payload
                        # cannot be decoded.
                        pass

            # =================================================
            # TTS END
            # =================================================

            elif event_type == "tts_end":

                marks.setdefault(
                    "input_to_tts_end",
                    now,
                )

        # End WebSocket receive loop.

    # ========================================================
    # Optional TTS audio export
    # ========================================================

    if audio_output and tts_audio:

        samples = array("f")

        samples.frombytes(
            tts_audio
        )

        pcm16_output = array(
            "h",
            (
                round(
                    max(
                        -1.0,
                        min(
                            1.0,
                            sample,
                        ),
                    )
                    * 32767
                )
                for sample in samples
            ),
        )

        audio_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with wave.open(
            str(audio_output),
            "wb",
        ) as target:

            target.setnchannels(1)

            target.setsampwidth(2)

            target.setframerate(24000)

            target.writeframes(
                pcm16_output.tobytes()
            )

    # ========================================================
    # Build raw record
    # ========================================================

    record: dict[str, Any] = {

        "patient_id":
            resolved_patient_id,

        "turn_id":
            turn_id,

        "session_start_ms":
            round(
                session_start_ms,
                2,
            ),

        "emotion":
            (final_insight or {}).get(
                "emotion",
                {},
            ),

        "ai_text":
            ai_text,

        "tts_audio":
            (
                str(audio_output)
                if audio_output
                and tts_audio
                else None
            ),

        # ----------------------------------------------------
        # Raw input-relative timestamps.
        #
        # These are deliberately retained for auditing,
        # debugging, and later analysis.
        # ----------------------------------------------------

        **{
            metric: round(
                (
                    marks[key]
                    - started_at
                )
                * 1000,
                2,
            )

            for metric, key in (

                (
                    "input_to_vad_end_ms",
                    "input_to_vad_end",
                ),

                (
                    "input_to_asr_first_partial_ms",
                    "input_to_asr_first_partial",
                ),

                (
                    "input_to_asr_result_ms",
                    "input_to_asr_result",
                ),

                (
                    "input_to_first_ai_text_ms",
                    "input_to_first_ai_text",
                ),

                (
                    "input_to_tts_start_ms",
                    "input_to_tts_start",
                ),

                (
                    "input_to_first_tts_byte_ms",
                    "input_to_first_tts_byte",
                ),

                (
                    "input_to_tts_end_ms",
                    "input_to_tts_end",
                ),
            )

            if key in marks
        },
    }

    # ========================================================
    # Official four core metrics
    # ========================================================

    # --------------------------------------------------------
    # 1. ASR latency
    #
    # VAD_END -> ASR_RESULT
    # --------------------------------------------------------

# --------------------------------------------------------
# 1. ASR first partial latency
#
# benchmark input start
# ->
# first non-empty asr_partial(final=false)
#
# IMPORTANT:
# This is NOT microphone-capture-to-first-character
# latency. The benchmark sends the complete WAV through
# manual_audio, so this measures:
#
# benchmark input submission start
# ->
# first interim ASR event observed over WebSocket.
# --------------------------------------------------------

    if "input_to_asr_first_partial_ms" in record:

        record[
            "asr_first_partial_latency_ms"
        ] = round(
            record[
                "input_to_asr_first_partial_ms"
            ],
            2,
        )


# --------------------------------------------------------
# 2. ASR final latency
#
# VAD_END -> ASR_RESULT
#
# This is the original ASR latency definition.
# --------------------------------------------------------

    if (
        "input_to_vad_end_ms" in record
        and "input_to_asr_result_ms" in record
    ):

        record[
            "asr_final_latency_ms"
        ] = round(
            record[
                "input_to_asr_result_ms"
            ]
            - record[
                "input_to_vad_end_ms"
            ],
            2,
        )

    # Backward compatibility:
    #
    # Existing analysis scripts may still expect
    # asr_latency_ms.
    #
    # Keep it as an alias of the original final-ASR
    # latency definition.

        record["asr_latency_ms"] = record[
            "asr_final_latency_ms"
        ]


# --------------------------------------------------------
# 3. ASR partial -> final latency
#
# first ASR partial
# ->
# ASR_RESULT
# --------------------------------------------------------

    if (
        "input_to_asr_first_partial_ms" in record
        and "input_to_asr_result_ms" in record
    ):

        record[
            "asr_partial_to_final_ms"
        ] = round(
            record[
                "input_to_asr_result_ms"
            ]
            - record[
                "input_to_asr_first_partial_ms"
            ],
            2,
        )
    # --------------------------------------------------------
    # 2. LLM first response latency
    #
    # ASR_RESULT -> first AI response event
    # --------------------------------------------------------

    if (
        "input_to_asr_result_ms" in record
        and "input_to_first_ai_text_ms" in record
    ):

        record[
            "llm_first_response_latency_ms"
        ] = round(
            record[
                "input_to_first_ai_text_ms"
            ]
            - record[
                "input_to_asr_result_ms"
            ],
            2,
        )

    # --------------------------------------------------------
    # 3. TTS first audio latency
    #
    # first AI response event
    # ->
    # first TTS audio byte/event
    # --------------------------------------------------------

    if (
        "input_to_first_ai_text_ms" in record
        and "input_to_first_tts_byte_ms" in record
    ):

        record[
            "tts_first_audio_latency_ms"
        ] = round(
            record[
                "input_to_first_tts_byte_ms"
            ]
            - record[
                "input_to_first_ai_text_ms"
            ],
            2,
        )

    # --------------------------------------------------------
    # 4. Complete voice response latency
    #
    # first AI response event
    # ->
    # TTS_END
    # --------------------------------------------------------

    if (
        "input_to_first_ai_text_ms" in record
        and "input_to_tts_end_ms" in record
    ):

        record[
            "llm_first_text_to_tts_end_ms"
        ] = round(
            record[
                "input_to_tts_end_ms"
            ]
            - record[
                "input_to_first_ai_text_ms"
            ],
            2,
        )

    # ========================================================
    # Keep the old derived metrics for backward compatibility.
    #
    # They are NOT part of the official four-metric summary.
    # ========================================================

    if "input_to_asr_result_ms" in record:

        if "input_to_first_ai_text_ms" in record:

            record[
                "asr_to_first_ai_text_ms"
            ] = round(
                record[
                    "input_to_first_ai_text_ms"
                ]
                - record[
                    "input_to_asr_result_ms"
                ],
                2,
            )

        if "input_to_first_tts_byte_ms" in record:

            record[
                "asr_to_first_tts_byte_ms"
            ] = round(
                record[
                    "input_to_first_tts_byte_ms"
                ]
                - record[
                    "input_to_asr_result_ms"
                ],
                2,
            )

    return record


# ============================================================
# Login
# ============================================================

def _login(
    server: str,
    username: str,
    password: str,
) -> str:

    import requests

    client = requests.Session()

    client.verify = False

    client.trust_env = False

    response = client.post(
        f"{server.rstrip('/')}/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
        timeout=10,
        verify=False,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get("success"):
        raise RuntimeError(
            "login failed"
        )

    cookie = client.cookies.get(
        "aa_session"
    )

    if not cookie:
        raise RuntimeError(
            "login response did not set aa_session"
        )

    return f"aa_session={cookie}"


# ============================================================
# Server flags
# ============================================================

def _server_flags(
    server: str,
) -> dict[str, Any]:

    import requests

    response = requests.get(
        f"{server.rstrip('/')}/health",
        timeout=10,
        verify=False,
    )

    response.raise_for_status()

    return dict(
        response.json().get(
            "release_flags"
        )
        or {}
    )


# ============================================================
# Benchmark
# ============================================================

async def _run_benchmark(
    args,
    pcm16: bytes,
    cookie: str,
) -> dict[str, Any]:

    records: list[dict[str, Any]] = []

    failures: list[dict[str, Any]] = []

    websocket_url = _websocket_url(
        args.server
    )

    total_runs = (
        args.warmup_runs
        + args.runs
    )

    for index in range(total_runs):

        phase = (
            "warmup"
            if index < args.warmup_runs
            else "measured"
        )

        logical_run = (
            index + 1
            if phase == "warmup"
            else index
            - args.warmup_runs
            + 1
        )

        record = None

        for attempt in range(
            1,
            args.max_attempts_per_run + 1,
        ):

            try:

                record = await _run_once(
                    websocket_url=websocket_url,
                    cookie=cookie,
                    pcm16=pcm16,
                    timeout_s=args.timeout,
                    patient_id=args.patient_id,
                    new_patient=args.new_patient,
                    profile_name=(
                        f"{args.profile_name}-"
                        f"{index + 1:03d}-"
                        f"{uuid.uuid4().hex[:6]}"
                    ),
                    long_term_memory_enabled=(
                        args.long_term_memory
                        == "on"
                    ),
                    long_term_memory_writes_enabled=(
                        False if args.freeze_memory_writes else None
                    ),
                    emotion_enabled=(args.emotion == "on"),
                    require_final_insight=args.require_audio_emotion,
                    audio_output=(
                        args.audio_output_dir
                        / (
                            f"{args.long_term_memory}-"
                            f"{phase}-"
                            f"{logical_run}.wav"
                        )
                        if args.audio_output_dir
                        else None
                    ),
                )

                # ------------------------------------------------
                # Emotion verification
                #
                # This is only enabled when the CLI explicitly
                # asks for it.
                #
                # Therefore:
                #
                # --emotion changes the session behavior.
                # --require-audio-emotion only validates that an enabled
                # session actually used Emotion2Vec; it is not a toggle.
                # ------------------------------------------------

                emotion = record.get(
                    "emotion",
                    {},
                )

                if args.require_audio_emotion and (
                    emotion.get("source")
                    != "emotion2vec_audio+text"
                    or not emotion.get(
                        "audio_model_used"
                    )
                ):

                    raise RuntimeError(
                        "audio emotion verification failed"
                    )

                break

            except Exception as exc:

                failure = {
                    "phase": phase,
                    "logical_run": logical_run,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
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

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Do not append a partial record.
                    #
                    # Failed runs are excluded from the statistics.
                    # ------------------------------------------------

                    record = None

                    break

                await asyncio.sleep(
                    args.retry_delay
                )

        # ----------------------------------------------------
        # Only measured runs enter records.
        # Warmup runs are deliberately excluded.
        # ----------------------------------------------------

        if (
            index >= args.warmup_runs
            and record is not None
        ):

            records.append(
                record
            )

    # ========================================================
    # Summary
    #
    # ONLY the four official metrics are summarized.
    # ========================================================

    summary = {

        metric: _summary(
            [
                float(record[metric])
                for record in records
                if metric in record
            ]
        )

        for metric in _METRICS
    }

    return {
        "runs": records,
        "failures": failures,
        "summary": summary,
    }


# ============================================================
# Argument parser
# ============================================================

def _parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Authenticated voice WebSocket "
            "latency benchmark"
        )
    )

    # --------------------------------------------------------
    # Live execution
    # --------------------------------------------------------

    parser.add_argument(
        "--live",
        action="store_true",
        help="perform network calls",
    )

    # --------------------------------------------------------
    # Server
    # --------------------------------------------------------

    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8502",
    )

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    parser.add_argument(
        "--username",
        required=True,
    )

    parser.add_argument(
        "--password",
        required=True,
    )

    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------

    patient = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    patient.add_argument(
        "--patient-id"
    )

    patient.add_argument(
        "--new-patient",
        action="store_true",
        help=(
            "create synthetic patients; "
            "this writes benchmark data "
            "to the server DB"
        ),
    )

    # --------------------------------------------------------
    # Session profile
    # --------------------------------------------------------

    parser.add_argument(
        "--profile-name",
        default="voice-benchmark",
    )

    # --------------------------------------------------------
    # Long-term memory
    # --------------------------------------------------------

    parser.add_argument(
        "--long-term-memory",
        choices=("on", "off"),
        default="on",
        help=(
            "long-term memory mode used by "
            "the benchmark session"
        ),
    )

    parser.add_argument(
        "--freeze-memory-writes",
        action="store_true",
        help=(
            "disable long-term-memory writes for this session while keeping "
            "reads controlled by --long-term-memory"
        ),
    )

    parser.add_argument(
        "--emotion",
        choices=("on", "off"),
        default="on",
        help="session-level emotion inference toggle",
    )

    # --------------------------------------------------------
    # Optional audio output
    # --------------------------------------------------------

    parser.add_argument(
        "--audio-output-dir",
        type=Path,
    )

    # --------------------------------------------------------
    # Runs
    # --------------------------------------------------------

    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
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

    # --------------------------------------------------------
    # Long-term memory verification
    # --------------------------------------------------------

    parser.add_argument(
        "--expect-long-term-memory",
        choices=("on", "off"),
        help=(
            "deprecated capability check; session behavior is verified "
            "directly from session_started"
        ),
    )

    parser.add_argument(
        "--expect-long-term-memory-writes",
        choices=("on", "off"),
        help=(
            "fail if the server long-term-memory "
            "write flag does not match"
        ),
    )

    # --------------------------------------------------------
    # Emotion verification
    # --------------------------------------------------------

    parser.add_argument(
        "--require-audio-emotion",
        action="store_true",
        help=(
            "validation only: fail unless every run completes Emotion2Vec "
            "audio inference; use --emotion on/off to change behavior"
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

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

    if args.require_audio_emotion and args.emotion != "on":
        _parser().error(
            "--require-audio-emotion requires --emotion on"
        )

    # --------------------------------------------------------
    # Record requested configuration
    # --------------------------------------------------------

    requested = {

        "server":
            args.server,

        "audio":
            str(args.audio),

        "runs":
            args.runs,

        "warmup_runs":
            args.warmup_runs,

        "max_attempts_per_run":
            args.max_attempts_per_run,

        "retry_delay":
            args.retry_delay,

        "patient_id":
            args.patient_id,

        "new_patient":
            args.new_patient,

        "expect_long_term_memory":
            args.expect_long_term_memory,

        "expect_long_term_memory_writes":
            args.expect_long_term_memory_writes,

        "require_audio_emotion":
            args.require_audio_emotion,

        "long_term_memory":
            args.long_term_memory,

        "freeze_memory_writes":
            args.freeze_memory_writes,

        "emotion":
            args.emotion,
    }

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if not args.live:

        print(
            json.dumps(
                {
                    "live": False,
                    "requested": requested,
                },
                ensure_ascii=False,
            )
        )

        return 0

    # ========================================================
    # Load audio
    # ========================================================

    pcm16 = _read_pcm16(
        args.audio
    )

    # ========================================================
    # Read server flags
    # ========================================================

    server_flags = _server_flags(
        args.server
    )

    # --------------------------------------------------------
    # Final turn insight is only required for explicit emotion validation.
    # --------------------------------------------------------

    if args.require_audio_emotion and not server_flags.get("turn_insight"):

        raise RuntimeError(
            "server must enable "
            "ENABLE_TURN_INSIGHT "
            "when --require-audio-emotion is used"
        )

    # ========================================================
    # Verify long-term-memory state
    # ========================================================

    expected_memory = args.expect_long_term_memory
    if expected_memory and expected_memory != args.long_term_memory:
        raise RuntimeError(
            "--expect-long-term-memory must match the requested session "
            "--long-term-memory value"
        )
    if args.long_term_memory == "on" and not server_flags.get("long_term_memory"):
        raise RuntimeError("server long-term-memory capability is disabled")

    # ========================================================
    # Verify long-term-memory write state
    # ========================================================

    expected_writes = (
        args.expect_long_term_memory_writes
    )

    if expected_writes:
        if args.freeze_memory_writes:
            if expected_writes != "off":
                raise RuntimeError(
                    "--freeze-memory-writes makes the effective session "
                    "write state off"
                )
        else:
            actual_writes = bool(server_flags.get("long_term_memory_writes"))
            expected_writes_bool = expected_writes == "on"
            if actual_writes != expected_writes_bool:
                raise RuntimeError(
                    "server long-term-memory write capability does not match "
                    f"--expect-long-term-memory-writes={expected_writes}"
                )

    # ========================================================
    # Login
    # ========================================================

    cookie = _login(
        args.server,
        args.username,
        args.password,
    )

    # ========================================================
    # Run benchmark
    # ========================================================

    result = {

        "live":
            True,

        "requested":
            requested,

        "server_flags":
            server_flags,

        # ----------------------------------------------------
        # Explicit description of the official metrics.
        # ----------------------------------------------------

        "official_metrics": {
            "asr_first_partial_latency_ms":
                "benchmark input start -> "
                "first non-empty asr_partial(final=false)",

            "asr_final_latency_ms":
                "VAD_END -> ASR_RESULT",

            "asr_partial_to_final_ms":
                "first ASR partial -> ASR_RESULT",

            "llm_first_response_latency_ms":
                "ASR_RESULT -> first AI response event",

            "tts_first_audio_latency_ms":
                "first AI response event -> "
                "first TTS audio event",

            "llm_first_text_to_tts_end_ms":
                "first AI response event -> TTS_END",

    # Backward-compatible alias:
    # asr_latency_ms == asr_final_latency_ms
        },

        "metric_note": (
            "asr_first_partial_latency_ms measures "
            "benchmark input submission start to the "
            "first non-empty interim asr_partial event "
            "with final=false observed through WebSocket; "
            "because the benchmark submits the complete WAV "
            "through manual_audio, it is not strict "
            "microphone-capture-to-first-character latency. "
            "asr_final_latency_ms preserves the original "
            "VAD_END -> ASR_RESULT definition. "
            "llm_first_response_latency_ms uses the first "
            "AI response event observed through WebSocket; "
            "it is not an internal LLM token-generation "
            "timestamp."
        ),
    }

    result.update(
        asyncio.run(
            _run_benchmark(
                args,
                pcm16,
                cookie,
            )
        )
    )

    # ========================================================
    # Final emotion verification
    #
    # Only applied when --require-audio-emotion
    # was explicitly supplied.
    # ========================================================

    if args.require_audio_emotion:

        invalid = [

            record.get(
                "turn_id"
            )
            or "unknown"

            for record in result["runs"]

            if (
                record.get(
                    "emotion",
                    {},
                ).get("source")
                != "emotion2vec_audio+text"

                or not record.get(
                    "emotion",
                    {},
                ).get(
                    "audio_model_used"
                )
            )
        ]

        if invalid:

            raise RuntimeError(
                "audio emotion verification "
                "failed for turns: "
                + ", ".join(invalid)
            )

    # ========================================================
    # Save JSON
    # ========================================================

    encoded = json.dumps(
        result,
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

    # ========================================================
    # Console output
    # ========================================================

    print(encoded)

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
