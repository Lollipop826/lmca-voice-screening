"""Run opt-in Emotion2Vec, DashScope, local Memobase, and BigASR measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须大于等于 0")
    return number


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("没有可统计的延迟样本")
    ordered = sorted(samples)

    def percentile(ratio: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1)]

    return {
        "count": len(samples),
        "mean_ms": round(sum(samples) / len(samples), 2),
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
    }


def measure_audio_emotion(
    audio_path: str | Path,
    *,
    runs: int,
    classifier: Any = None,
) -> dict[str, float | int]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    if classifier is None:
        from src.tools.emotion import get_audio_emotion_classifier

        classifier = get_audio_emotion_classifier()

    started_at = time.perf_counter()
    classifier.classify_audio(str(path))
    warmup_ms = (time.perf_counter() - started_at) * 1000
    if not classifier.model_available:
        raise RuntimeError(
            "Emotion2Vec+ 模型不可用: "
            f"{classifier.last_error or '请检查 USE_MODELSCOPE 和模型依赖'}"
        )

    samples = []
    for _ in range(runs):
        started_at = time.perf_counter()
        scores = classifier.classify_audio(str(path))
        samples.append((time.perf_counter() - started_at) * 1000)
        if not classifier.model_available or not scores:
            raise RuntimeError(classifier.last_error or "Emotion2Vec+ 推理失败")

    return {"warmup_ms": round(warmup_ms, 2), **_summary(samples)}


def measure_memobase_retrieval(
    memory: Any,
    *,
    patient_id: str,
    current_text: str,
    runs: int,
) -> dict[str, float | int]:
    patient_id = str(patient_id or "").strip()
    current_text = str(current_text or "").strip()
    if not patient_id or not current_text:
        raise ValueError("Memobase 检索需要患者 ID 和当前文本")
    if not memory.is_memobase_available():
        raise RuntimeError("本地 Memobase 不可用")

    started_at = time.perf_counter()
    context = memory.get_relevant_context(patient_id, current_text)
    warmup_ms = (time.perf_counter() - started_at) * 1000
    samples = []
    for _ in range(runs):
        started_at = time.perf_counter()
        context = memory.get_relevant_context(patient_id, current_text)
        samples.append((time.perf_counter() - started_at) * 1000)

    return {
        "warmup_ms": round(warmup_ms, 2),
        "context_chars": len(str(context or "")),
        **_summary(samples),
    }


def measure_local_memobase(
    *,
    patient_id: str,
    current_text: str,
    runs: int,
) -> dict[str, float | int]:
    from src.context_management.emotion_memobase import EmotionMemobase

    memory = EmotionMemobase(emotion_classifier=lambda _text: {})
    try:
        return measure_memobase_retrieval(
            memory,
            patient_id=patient_id,
            current_text=current_text,
            runs=runs,
        )
    finally:
        memory.close()


def _measure_call(callback, runs: int) -> tuple[Any, dict[str, float | int]]:
    started_at = time.perf_counter()
    value = callback()
    warmup_ms = (time.perf_counter() - started_at) * 1000
    samples = []
    for _ in range(runs):
        started_at = time.perf_counter()
        value = callback()
        samples.append((time.perf_counter() - started_at) * 1000)
    return value, {"warmup_ms": round(warmup_ms, 2), **_summary(samples)}


def measure_sqlite_memory_scale(
    history_sizes: list[int],
    *,
    runs: int,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure the same local-memory path used by a voice turn in a disposable DB."""
    from src.context_management.emotion_memobase import EmotionMemobase
    from src.voice.services import PatientMemoryService

    if not history_sizes:
        raise ValueError("至少需要一个历史条数")
    temporary_dir = None
    if db_path is None:
        temp_root = Path("tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = tempfile.TemporaryDirectory(
            prefix="memory-scale-",
            dir=temp_root,
        )
        db_path = Path(temporary_dir.name) / "memory.db"
    db_path = Path(db_path)
    saved_env = {
        name: os.environ.pop(name, None)
        for name in ("MEMOBASE_PROJECT_URL", "MEMOBASE_API_KEY")
    }
    memory = None
    try:
        memory = EmotionMemobase(
            str(db_path),
            emotion_classifier=lambda _text: {},
            logger=lambda _message: None,
        )
        service = PatientMemoryService(
            get_patient=lambda _patient_id: None,
            create_patient=lambda _profile: {},
            update_patient_profile=lambda _patient_id, _profile: None,
            link_session_patient=lambda _session_id, _patient_id: None,
            long_term_memory=memory,
            logger=lambda _message: None,
        )
        scenarios = []
        for history_size in history_sizes:
            patient_id = f"memory-scale-{history_size}"
            for index in range(history_size):
                memory.capture_turn(
                    patient_id,
                    f"第{index}条历史：我今天想起以前的事情。",
                    "我在听，我们慢慢说。",
                    session_id=f"history-{index // 3}",
                    turn_id=f"history-{index}",
                    emotion={},
                )
            active_session = f"active-{history_size}"
            turn_index = 0

            def capture_turn():
                nonlocal turn_index
                turn_index += 1
                return memory.capture_turn(
                    patient_id,
                    "我现在有点担心。",
                    "",
                    session_id=active_session,
                    turn_id=f"active-{turn_index}",
                    emotion={},
                )

            _captured, capture = _measure_call(capture_turn, runs)
            session = SimpleNamespace(
                session_id=active_session,
                lifecycle=SimpleNamespace(current_patient_id=patient_id),
            )
            cross_context, cross_session = _measure_call(
                lambda: memory.get_cross_session_context(
                    patient_id,
                    exclude_session_id=active_session,
                ),
                runs,
            )
            evidence, evidence_lookup = _measure_call(
                lambda: memory.get_relevant_evidence(
                    patient_id,
                    "我现在有点担心。",
                    exclude_session_id=active_session,
                ),
                runs,
            )
            turn_context, turn_lookup = _measure_call(
                lambda: service.get_turn_context(session, "我现在有点担心。"),
                runs,
            )
            _snapshot, snapshot_lookup = _measure_call(
                lambda: memory.get_snapshot(patient_id),
                runs,
            )
            card, card_lookup = _measure_call(
                lambda: memory.get_authoritative_card(patient_id),
                runs,
            )
            conn = sqlite3.connect(str(db_path))
            try:
                plan = [
                    row[-1]
                    for row in conn.execute(
                        "EXPLAIN QUERY PLAN "
                        "SELECT id FROM emotion_memobase_turns "
                        "WHERE patient_id=? AND COALESCE(session_id, '') != ? "
                        "ORDER BY id DESC LIMIT ?",
                        (patient_id, active_session, 12),
                    )
                ]
            finally:
                conn.close()
            scenarios.append(
                {
                    "history_turns": history_size,
                    "capture_turn": capture,
                    "cross_session": cross_session,
                    "evidence_lookup": evidence_lookup,
                    "voice_turn_lookup": turn_lookup,
                    "snapshot": snapshot_lookup,
                    "authoritative_card": card_lookup,
                    "cross_session_chars": len(cross_context),
                    "evidence_chars": len(evidence),
                    "voice_turn_context_chars": len(turn_context),
                    "authoritative_card_chars": len(card),
                    "query_plan": plan,
                }
            )
        return {"runs": runs, "scenarios": scenarios}
    finally:
        if memory is not None:
            memory.close()
        for name, value in saved_env.items():
            if value is not None:
                os.environ[name] = value
        if temporary_dir is not None:
            temporary_dir.cleanup()


async def measure_ark_asr_stream(
    audio: Any,
    sample_rate: int,
    *,
    stream_factory: Any,
    chunk_seconds: float = 0.2,
    sleeper=None,
) -> dict[str, float | int | None]:
    import asyncio

    if sample_rate <= 0 or len(audio) == 0:
        raise ValueError("BigASR 测量需要非空音频")
    chunk_samples = max(1, int(sample_rate * chunk_seconds))
    sleeper = sleeper or asyncio.sleep
    first_partial_ms = None
    final_text_ms = None
    final_text = ""
    started_at = time.perf_counter()

    async def on_result(text: str, is_final: bool) -> None:
        nonlocal first_partial_ms, final_text_ms, final_text
        text = str(text or "").strip()
        if not text:
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if is_final:
            final_text = text
            final_text_ms = elapsed_ms
        elif first_partial_ms is None:
            first_partial_ms = elapsed_ms

    stream = stream_factory(sample_rate=sample_rate, on_result=on_result)
    try:
        await stream.start()
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            await sleeper(len(chunk) / sample_rate)
            await stream.feed(chunk)
        final_text = str(await stream.finish() or final_text).strip()
        if not final_text:
            raise RuntimeError("BigASR 流式未返回最终文字")
        if final_text_ms is None:
            final_text_ms = (time.perf_counter() - started_at) * 1000
        return {
            "audio_seconds": round(len(audio) / sample_rate, 3),
            "chunks": math.ceil(len(audio) / chunk_samples),
            "first_partial_ms": (
                round(first_partial_ms, 2)
                if first_partial_ms is not None
                else None
            ),
            "final_text_ms": round(final_text_ms, 2),
        }
    finally:
        await stream.aclose()


def measure_ark_asr_streaming(
    audio_path: str | Path,
    *,
    runs: int,
    stream_factory: Any = None,
) -> dict[str, Any]:
    import asyncio
    import soundfile as sf

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if stream_factory is None:
        from src.tools.voice.ark_asr import ArkASRStreamingSession

        stream_factory = ArkASRStreamingSession

    records = [
        asyncio.run(
            measure_ark_asr_stream(
                audio,
                int(sample_rate),
                stream_factory=stream_factory,
            )
        )
        for _ in range(runs)
    ]
    partials = [
        item["first_partial_ms"]
        for item in records
        if item["first_partial_ms"] is not None
    ]
    return {
        "audio_seconds": records[0]["audio_seconds"],
        "runs": len(records),
        "first_partial": _summary(partials) if partials else None,
        "final_text": _summary([item["final_text_ms"] for item in records]),
        "runs_without_partial": len(records) - len(partials),
    }


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def measure_streaming_llm(
    llm: Any,
    *,
    prompt: str,
    runs: int,
) -> dict[str, dict[str, float | int]]:
    first_token_samples = []
    complete_samples = []
    messages = [{"role": "user", "content": prompt}]

    for _ in range(runs):
        started_at = time.perf_counter()
        first_token_ms = None
        received_text = []
        for chunk in llm.stream(messages):
            text = _chunk_text(chunk)
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started_at) * 1000
            received_text.append(text)
        if first_token_ms is None or not "".join(received_text).strip():
            raise RuntimeError("流式响应未返回文本内容")
        first_token_samples.append(first_token_ms)
        complete_samples.append((time.perf_counter() - started_at) * 1000)

    return {
        "first_token": _summary(first_token_samples),
        "complete": _summary(complete_samples),
    }


def measure_dashscope_model(
    model: str,
    *,
    prompt: str,
    runs: int,
) -> dict[str, Any]:
    from src.llm.http_client_pool import get_dashscope_chat_openai

    llm = get_dashscope_chat_openai(
        model=model,
        temperature=0,
        max_tokens=96,
        timeout=20,
        max_retries=0,
        streaming=True,
        disable_thinking=True,
    )
    return {"model": model, **measure_streaming_llm(llm, prompt=prompt, runs=runs)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可选的本地、DashScope、Memobase 与 BigASR 延迟实测")
    parser.add_argument("--live", action="store_true", help="实际执行推理或 API 调用")
    parser.add_argument("--audio", type=Path, help="用于 Emotion2Vec+ 实测的音频文件")
    parser.add_argument(
        "--llm-model",
        action="append",
        default=[],
        metavar="MODEL",
        help="DashScope Flash 模型；可重复指定",
    )
    parser.add_argument("--emotion-runs", type=_positive_int, default=5)
    parser.add_argument("--llm-runs", type=_positive_int, default=3)
    parser.add_argument("--memobase-patient-id", help="Memobase 检索患者 ID")
    parser.add_argument("--memory-text", help="Memobase 当前文本检索内容")
    parser.add_argument("--memobase-runs", type=_positive_int, default=5)
    parser.add_argument(
        "--memory-scale",
        nargs="+",
        type=_nonnegative_int,
        metavar="TURNS",
        help="临时 SQLite 中的历史轮次数，可指定多个值",
    )
    parser.add_argument("--memory-scale-runs", type=_positive_int, default=10)
    parser.add_argument("--memory-scale-db", type=Path)
    parser.add_argument("--ark-asr-audio", type=Path, help="用于 BigASR 流式实测的音频文件")
    parser.add_argument("--ark-asr-runs", type=_positive_int, default=1)
    parser.add_argument(
        "--prompt",
        default="请用一句中文简短回应：我今天有点焦虑。",
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 结果文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.memobase_patient_id) != bool(args.memory_text):
        _parser().error("--memobase-patient-id 与 --memory-text 必须同时指定")
    if not any((
        args.audio,
        args.llm_model,
        args.memobase_patient_id,
        args.ark_asr_audio,
        args.memory_scale,
    )):
        _parser().error("至少指定一个待测项目")

    requested = {
        "audio": str(args.audio) if args.audio else None,
        "llm_models": args.llm_model,
        "memobase": bool(args.memobase_patient_id),
        "memory_text_chars": len(args.memory_text or ""),
        "ark_asr_audio": str(args.ark_asr_audio) if args.ark_asr_audio else None,
        "memory_scale": args.memory_scale,
    }
    if not args.live:
        print(json.dumps({"live": False, "requested": requested}, ensure_ascii=False))
        return 0

    results: dict[str, Any] = {"live": True, "requested": requested}
    failures = 0
    if args.audio:
        try:
            results["emotion2vec"] = measure_audio_emotion(
                args.audio,
                runs=args.emotion_runs,
            )
        except Exception as exc:
            failures += 1
            results["emotion2vec"] = {"error": str(exc)}

    if args.memobase_patient_id:
        try:
            results["memobase"] = measure_local_memobase(
                patient_id=args.memobase_patient_id,
                current_text=args.memory_text,
                runs=args.memobase_runs,
            )
        except Exception as exc:
            failures += 1
            results["memobase"] = {"error": str(exc)}

    if args.ark_asr_audio:
        try:
            results["ark_asr"] = measure_ark_asr_streaming(
                args.ark_asr_audio,
                runs=args.ark_asr_runs,
            )
        except Exception as exc:
            failures += 1
            results["ark_asr"] = {"error": str(exc)}

    if args.memory_scale:
        try:
            results["sqlite_memory_scale"] = measure_sqlite_memory_scale(
                args.memory_scale,
                runs=args.memory_scale_runs,
                db_path=args.memory_scale_db,
            )
        except Exception as exc:
            failures += 1
            results["sqlite_memory_scale"] = {"error": str(exc)}

    llm_results = []
    for model in args.llm_model:
        try:
            llm_results.append(
                measure_dashscope_model(
                    model,
                    prompt=args.prompt,
                    runs=args.llm_runs,
                )
            )
        except Exception as exc:
            failures += 1
            llm_results.append({"model": model, "error": str(exc)})
    if llm_results:
        results["dashscope"] = llm_results

    encoded = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
