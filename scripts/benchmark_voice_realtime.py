"""按真实说话节奏测量完整语音链路的延迟。

与 ``benchmark_voice_e2e.py`` 的区别在于送音频的方式，这直接决定了测出来的
数字有没有意义：

  旧做法  一条 ``manual_audio`` 消息把整段音频灌进去，然后从这一刻开始计时。
          服务端收到就立刻发 ``vad_end``，所以"ASR 延迟"实际是"批量转写整段
          音频的耗时"。8 秒素材测出 4.7 秒，这个数字既不反映用户等待，也不
          随链路优化而变化。另外 ``manual_audio`` 走 ManualAudioInputHandler，
          绕开了 VAD 与 SoulX，轮次判定这一环根本没被测到。

  本脚本  按 32ms 一帧（512 样本，与 WebRTC 上游帧长一致）用二进制帧推送，走
          和浏览器完全相同的 ``audio`` 通道，因此会真实经过 VAD、SoulX 轮次
          判定、ASR、Agent、TTS。说完之后继续推静音帧，让 VAD 有静音可判。

计时基准是"最后一帧语音送出的时刻"，对应用户说完最后一个字。这是唯一能代表
用户感知等待的起点：真实通话中音频边说边处理，说完时前面的部分早已消化完。

因为走真实链路，一次输入可能被 SoulX 判成多个轮次（说话中略有停顿就切分），
这正是需要被观测的行为，所以脚本会统计每次输入实际产生了几个轮次。

用法示例：

    python scripts/benchmark_voice_realtime.py \\
        --server http://127.0.0.1:8426 \\
        --username <user> --password <pass> \\
        --runs 20 --warmup-runs 3

记忆配对消融（每个 pair 的 on/off 使用同一患者、同一音频，默认禁止写记忆）：

    python scripts/benchmark_voice_realtime.py \\
        --server https://127.0.0.1:8427 --insecure \\
        --username <user> --password <pass> \\
        --patient-id <existing-patient> --ablation memory \\
        --runs 20 --warmup-runs 3 --output output/memory_ablation.json

多素材时传 ``--audio-manifest``。清单每项包含 ``sample_id``、``audio_path``
和可选 ``reference_text``；同一 pair 始终使用同一条素材。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import statistics
import ssl
import subprocess
import time
import uuid
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# ============================================================
# 帧参数
# ============================================================

# 512 样本 = 32ms@16k。与项目上游帧长保持一致：SoulX 每 2560 样本(160ms)推理
# 一次，正好由 5 个这样的帧组成。用别的帧长会让攒批边界和真实情况错位。
FRAME_SAMPLES = 512
SAMPLE_RATE = 16000
FRAME_INTERVAL_S = FRAME_SAMPLES / SAMPLE_RATE


# ============================================================
# 首尾静音裁剪参数
# ============================================================
#
# 服务落盘的通话录音首尾都带着 VAD 判定前后的静音。实测本机 642 个真实用户
# 录音：开头静音中位数 0.80s（p90 1.73s），末尾静音 47.4% 超过 1 秒、最长
# 4.26s。这段静音必须裁掉，否则"最后一帧语音"这个零点会落在静音之后，
# turn_decision_ms 被系统性低估最多 4 秒，恰好把最该测的指标测废。

# 静音门限取整段峰值 RMS 的比例。真实录音底噪 RMS 通常在峰值的 1~3%，取 8%
# 能滤掉底噪，又不至于切掉气声、擦音这类低能量音节。
_SILENCE_THRESHOLD_RATIO = 0.08

# 裁剪后在语音末尾保留的余量。计时零点落在这段余量之后，所以它会让
# turn_decision_ms 系统性偏小这么多。用两帧换一个不切掉尾音的安全边际，
# 相比原始录音那 4 秒的偏差可以忽略。
_TRAILING_GUARD_S = 0.064

# 裁剪后在语音开头保留的静音。VAD 与 SoulX 需要一点前导静音建立噪声基准，
# 但没必要留满真实录音那 0.8s——那只是白等。
_LEADING_PAD_S = 0.192


# ============================================================
# 默认素材
# ============================================================

# 从 data/voice_calls 里固化出来的一段真实用户录音，内容是
# "你好呀，我今天心情可能不太好。"。选它的理由是它正是旧 benchmark 反复使用的
# 素材（在落盘录音里出现 266 次），拿同一段音频对比新旧脚本才有意义。
#
# 原始文件 8.00s，其中真正的语音只有约 3.2s，首尾共 4.8s 是静音——这就是旧脚本
# "8 秒素材测出 4.7 秒 ASR"的直接来源：转写引擎在啃一段一半是空白的音频。
_DEFAULT_AUDIO_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "bench_speech_zh_8s.wav"
)


# ============================================================
# 指标定义
# ============================================================
#
# 全部以"最后一帧语音送出"为零点，除了明确标注的两项。
#
#   turn_decision_ms   最后一帧语音 → vad_end
#                      轮次判定耗时。SoulX 判"说完了"要多久。这一项在旧脚本
#                      里测不到，因为 manual_audio 收到音频就无条件发 vad_end。
#
#   soulx_first_text_ms 音频开始推送 → 首个 SoulX 增量文字
#                       这是 SoulX/Paraformer 的真实流式首字延迟。
#
#   post_vad_pipeline_ms vad_end → asr_result
#                        这是轮次提交、排队和结果发布耗时；SoulX 路径已经有文字，
#                        因此绝不能把它称为“纯 ASR”。
#
#   agent_pipeline_ms  asr_result → 首个 AI 文本事件
#                      包含记忆检索、Agent 准备和 LLM 首句，不是纯 LLM 推理。
#
#   tts_queue_ms       首个 AI 文本事件 → tts_start
#   tts_provider_ms    tts_start → 首个 TTS 音频帧
#
#   perceived_ms       最后一帧语音 → 首个 TTS 音频帧
#                      用户实际感知的等待。这是最该被优化的数字。
#
#   full_response_ms   最后一帧语音 → tts_end
#
_METRICS = (
    "soulx_first_text_ms",
    "turn_decision_ms",
    "post_vad_pipeline_ms",
    "agent_pipeline_ms",
    "tts_queue_ms",
    "tts_provider_first_byte_ms",
    "tts_first_byte_ms",
    "perceived_ms",
    "full_response_ms",
)

# 依赖 TTS 才能算出的指标。TTS 不可用时（例如火山返回 403 未授权）这几项没有
# 数据，但前三项仍然成立，所以要能单独跳过而不是让整轮失败。
_TTS_DEPENDENT_METRICS = frozenset(
    {
        "tts_queue_ms",
        "tts_provider_first_byte_ms",
        "tts_first_byte_ms",
        "perceived_ms",
        "full_response_ms",
    }
)

# 拿到首个 AI 文本之后，再等多久还没有 TTS 音频就判定 TTS 这一环不可用。
# 服务端 TTS 失败时只在自己日志里记一行，不给客户端任何事件，所以只能靠超时
# 识别。取 12 秒的理由：实测正常情况下首个音频帧在首句文本后 1~2 秒到达，
# 冷启动建连最慢约 2 秒，12 秒留足余量又不会把每轮拖成 --timeout 那么长。
_TTS_GRACE_S = 12.0

# 最后一个轮次的应答收尾之后，再观察这么久确认没有新轮次冒出来。
# SoulX 在句中停顿处切轮次时，下一个轮次的 asr_result 紧跟着前一轮的应答到达，
# 实测间隔在 1 秒以内；取 2 秒留一倍余量。这段等待每轮只付一次。
_TURN_SETTLE_S = 2.0

# 服务端的 ``tts_end`` 表示开场语音已完成合成和下发，不代表客户端已经播放完。
# 它随后会按 ``duration`` 继续把 AI 标记为正在说话；若此时立即送测试音频，会
# 误入“用户打断开场”的快速 ASR 路径，把轮次判定虚增数秒。
_GREETING_PLAYBACK_GUARD_S = 0.25

# 小于该样本量时 p95 退化为最大值，报出来会误导读者，所以直接省略该列。
_MIN_SAMPLES_FOR_P95 = 20


# ============================================================
# 基础工具
# ============================================================

def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _summarize(values: list[float]) -> dict[str, float | int]:
    """给出样本量、均值、中位数、p95（样本足够时）和最大值。"""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    summary: dict[str, float | int] = {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": round(statistics.median(ordered), 1),
        "max_ms": round(ordered[-1], 1),
    }
    if len(ordered) >= _MIN_SAMPLES_FOR_P95:
        index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        summary["p95_ms"] = round(ordered[index], 1)
    return summary


def _git_state() -> dict[str, Any]:
    """Record the exact source revision without making Git a dependency."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _server_health(server: str, *, verify_tls: bool) -> dict[str, Any]:
    import requests

    response = requests.get(
        f"{server.rstrip('/')}/health",
        timeout=15,
        verify=verify_tls,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _normalize_text(value: str) -> str:
    return "".join(
        character
        for character in str(value or "").strip().lower()
        if not character.isspace()
        and character not in "，。！？、,.!?；;：:\"'‘’“”（）()【】[]…·"
    )


def _character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return Unicode character error rate using Levenshtein distance."""
    expected = _normalize_text(reference)
    actual = _normalize_text(hypothesis)
    if not expected:
        return None
    previous = list(range(len(actual) + 1))
    for row, expected_character in enumerate(expected, start=1):
        current = [row]
        for column, actual_character in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (expected_character != actual_character),
                )
            )
        previous = current
    return round(previous[-1] / len(expected), 4)


def _paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        pair_index = record.get("pair_index")
        condition = str(record.get("condition") or "")
        if not isinstance(pair_index, int) or not condition:
            continue
        grouped.setdefault(pair_index, {})[condition] = record

    pairs = [pair for pair in grouped.values() if len(pair) == 2]
    conditions = sorted({name for pair in pairs for name in pair})
    if len(conditions) != 2:
        return {"complete_pairs": 0, "metrics": {}}
    baseline, treatment = conditions
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        deltas = []
        for pair in pairs:
            left = pair[baseline].get(metric)
            right = pair[treatment].get(metric)
            if isinstance(left, (int, float)) and isinstance(
                right, (int, float)
            ):
                deltas.append(float(right) - float(left))
        if deltas:
            metrics[metric] = {
                "definition": f"{treatment} - {baseline}",
                **_summarize(deltas),
            }
    return {
        "complete_pairs": len(pairs),
        "baseline": baseline,
        "treatment": treatment,
        "metrics": metrics,
    }


def _read_wav_pcm16(path: str | Path) -> bytes:
    """读取单声道 16kHz PCM16 WAV，返回原始字节。"""
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        actual = (
            params.nchannels,
            params.sampwidth,
            params.framerate,
            params.comptype,
        )
        if actual != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError(
                f"音频需为单声道 PCM16 {SAMPLE_RATE}Hz 未压缩 WAV，"
                f"实际 channels={params.nchannels} width={params.sampwidth} "
                f"rate={params.framerate} compression={params.comptype}"
            )
        data = source.readframes(params.nframes)

    if not data:
        raise ValueError("音频文件为空")
    return data


def _load_audio_cases(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    """Load one WAV or a manifest of independently labelled WAV samples."""
    if not arguments.audio_manifest:
        return [
            {
                "sample_id": Path(arguments.audio).stem,
                "audio_path": str(Path(arguments.audio).resolve()),
                "reference_text": arguments.reference_text,
            }
        ]

    manifest_path = Path(arguments.audio_manifest).resolve()
    with manifest_path.open(encoding="utf-8") as source:
        payload = json.load(source)
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not samples:
        raise ValueError("--audio-manifest 必须是非空 JSON 数组或含 samples 的对象")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"音频清单第 {index} 项不是对象")
        raw_path = str(sample.get("audio_path") or sample.get("audio") or "")
        path = Path(raw_path.replace("\\", "/"))
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"音频清单第 {index} 项不存在: {path}")
        sample_id = str(sample.get("sample_id") or path.stem).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"音频清单 sample_id 为空或重复: {sample_id!r}")
        seen_ids.add(sample_id)
        cases.append(
            {
                "sample_id": sample_id,
                "audio_path": str(path),
                "reference_text": sample.get("reference_text")
                or sample.get("text")
                or arguments.reference_text,
            }
        )
    return cases


def _prepare_audio_cases(
    cases: list[dict[str, Any]], *, no_trim: bool
) -> list[dict[str, Any]]:
    prepared = []
    for case in cases:
        raw_pcm16 = _read_wav_pcm16(case["audio_path"])
        raw_seconds = len(raw_pcm16) / 2 / SAMPLE_RATE
        if no_trim:
            pcm16 = raw_pcm16
            trim_report = {
                "leading_trimmed_s": 0.0,
                "trailing_trimmed_s": 0.0,
            }
        else:
            pcm16, trim_report = _trim_surrounding_silence(raw_pcm16)
        if not pcm16:
            raise ValueError(f"裁剪后音频为空: {case['audio_path']}")
        frames = _split_into_frames(pcm16)
        prepared.append(
            {
                **case,
                "frames": frames,
                "raw_seconds": raw_seconds,
                "speech_seconds": len(frames) * FRAME_INTERVAL_S,
                "trim_report": trim_report,
                "sha256": hashlib.sha256(raw_pcm16).hexdigest(),
            }
        )
    return prepared


def _split_into_frames(pcm16: bytes) -> list[bytes]:
    """切成固定长度的帧，最后一帧不足时补零对齐。"""
    frame_bytes = FRAME_SAMPLES * 2
    frames = [
        pcm16[offset:offset + frame_bytes]
        for offset in range(0, len(pcm16), frame_bytes)
    ]
    if frames and len(frames[-1]) < frame_bytes:
        frames[-1] = frames[-1].ljust(frame_bytes, b"\x00")
    return frames


def _frame_rms_envelope(samples: array) -> list[float]:
    """逐帧算 RMS。帧长与推送帧长一致，让裁剪边界落在帧边界上。"""
    envelope: list[float] = []
    for offset in range(0, len(samples) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        window = samples[offset:offset + FRAME_SAMPLES]
        total = 0.0
        for value in window:
            normalized = value / 32768.0
            total += normalized * normalized
        envelope.append(math.sqrt(total / FRAME_SAMPLES))
    return envelope


def _trim_surrounding_silence(pcm16: bytes) -> tuple[bytes, dict[str, float]]:
    """裁掉首尾静音，返回裁剪后的音频与裁掉的时长。

    必须裁：计时零点是"最后一帧语音送出的时刻"，如果末尾还挂着录音自带的静音，
    零点就落在静音之后，而 VAD/SoulX 早在真正的语音结束处就开始计时了，
    turn_decision_ms 会被系统性低估。本机真实录音末尾静音最长 4.26s，
    足以让这个指标彻底失真。

    裁剪按帧边界做，保证裁完仍是整数帧，不会在中间引入错位。
    """
    samples = array("h")
    samples.frombytes(pcm16[: len(pcm16) - len(pcm16) % 2])

    envelope = _frame_rms_envelope(samples)
    if not envelope:
        return pcm16, {"leading_trimmed_s": 0.0, "trailing_trimmed_s": 0.0}

    peak_rms = max(envelope)
    threshold = peak_rms * _SILENCE_THRESHOLD_RATIO
    voiced_indexes = [
        index for index, value in enumerate(envelope) if value > threshold
    ]
    # 全程低于门限（纯静音或纯底噪）时不裁，交给调用方的校验去报错。
    if not voiced_indexes:
        return pcm16, {"leading_trimmed_s": 0.0, "trailing_trimmed_s": 0.0}

    leading_pad_frames = round(_LEADING_PAD_S / FRAME_INTERVAL_S)
    trailing_guard_frames = round(_TRAILING_GUARD_S / FRAME_INTERVAL_S)

    start_frame = max(0, voiced_indexes[0] - leading_pad_frames)
    end_frame = min(
        len(envelope),
        voiced_indexes[-1] + 1 + trailing_guard_frames,
    )

    frame_bytes = FRAME_SAMPLES * 2
    trimmed = pcm16[start_frame * frame_bytes:end_frame * frame_bytes]

    return trimmed, {
        "leading_trimmed_s": round(start_frame * FRAME_INTERVAL_S, 3),
        "trailing_trimmed_s": round(
            (len(envelope) - end_frame) * FRAME_INTERVAL_S,
            3,
        ),
    }


def _websocket_url(server: str) -> str:
    parsed = urlsplit(server.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("server 必须是 http(s) URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws"


# ============================================================
# 登录与连接
# ============================================================

def _login(server: str, username: str, password: str, *, verify_tls: bool) -> str:
    """登录取 Cookie。沿用项目现有的 /api/auth/login 接口。"""
    import requests

    response = requests.post(
        f"{server.rstrip('/')}/api/auth/login",
        json={"username": username, "password": password},
        timeout=15,
        verify=verify_tls,
    )
    response.raise_for_status()

    cookie = "; ".join(
        f"{name}={value}" for name, value in response.cookies.items()
    )
    if not cookie:
        raise RuntimeError("登录成功但未返回 Cookie，无法建立会话")
    return cookie


def _connect_websocket(websocket_url: str, cookie: str, *, verify_tls: bool):
    import websockets

    ssl_context = None
    if websocket_url.startswith("wss://") and not verify_tls:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    return websockets.connect(
        websocket_url,
        additional_headers={"Cookie": cookie},
        # 语音服务在本机或局域网，不能走系统 HTTP 代理
        proxy=None,
        open_timeout=10,
        max_size=8 * 1024 * 1024,
        ssl=ssl_context,
    )


async def _receive_json(websocket, deadline: float) -> dict[str, Any]:
    """取下一个 JSON 事件，忽略二进制帧与非字典载荷。"""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("等待 WebSocket 事件超时")

        raw = await asyncio.wait_for(websocket.recv(), remaining)
        if isinstance(raw, (bytes, bytearray)):
            continue
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            return event


async def _wait_for_event(
    websocket,
    event_type: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while True:
        event = await _receive_json(websocket, deadline)
        if event.get("type") == event_type:
            return event


# ============================================================
# 会话准备
# ============================================================

async def _start_session(
    websocket,
    *,
    patient_id: str | None,
    create_patient: bool,
    profile_name: str,
    long_term_memory_enabled: bool,
    long_term_memory_writes_enabled: bool,
    emotion_enabled: bool,
    verify_effective_flags: bool,
    timeout_s: float,
) -> dict[str, Any]:
    """建会话并等开场问候播完，避免问候的 TTS 混进被测轮次。"""
    await _wait_for_event(websocket, "waiting_for_info", timeout_s)

    payload: dict[str, Any] = {
        "type": "start_session",
        "force_new_session": True,
        "long_term_memory_enabled": long_term_memory_enabled,
        "long_term_memory_writes_enabled": long_term_memory_writes_enabled,
        "emotion_enabled": emotion_enabled,
    }
    if patient_id:
        payload["patient_id"] = patient_id
    if create_patient:
        payload["profile"] = {
            "name": profile_name,
            "age": 70,
            "gender": "female",
            "education_years": 6,
        }

    await websocket.send(json.dumps(payload, ensure_ascii=False))

    deadline = time.perf_counter() + timeout_s
    resolved_patient_id = patient_id
    session_started = False
    session_started_event: dict[str, Any] = {}
    patient_has_history = False
    # 当前服务会为每条新 WebSocket 会话播放问候，包括复用已有患者的情况。
    # tts_end 只代表合成/下发结束；还要按事件里的 duration 等待客户端播放完。
    greeting_pending = True
    greeting_playback_s = 0.0

    while not session_started or greeting_pending:
        event = await _receive_json(websocket, deadline)
        event_type = event.get("type")

        if event_type == "patient_memory":
            resolved_patient_id = (
                str(event.get("patient_id") or "") or resolved_patient_id
            )
            patient_has_history = bool(event.get("has_history"))
        elif event_type == "session_started":
            session_started = True
            session_started_event = event
        elif event_type == "tts_end" and greeting_pending:
            greeting_playback_s = max(
                0.0,
                float(event.get("duration") or 0.0),
            )
            greeting_pending = False

    if greeting_playback_s > 0:
        await asyncio.sleep(
            greeting_playback_s + _GREETING_PLAYBACK_GUARD_S
        )

    requested_flags = {
        "long_term_memory_enabled": long_term_memory_enabled,
        "long_term_memory_writes_enabled": long_term_memory_writes_enabled,
        "emotion_enabled": emotion_enabled,
    }
    effective_flags = {
        name: session_started_event.get(name)
        for name in requested_flags
        if name in session_started_event
    }
    if verify_effective_flags:
        missing = [
            name for name in requested_flags if name not in session_started_event
        ]
        mismatched = [
            name
            for name, expected in requested_flags.items()
            if name in session_started_event
            and bool(session_started_event[name]) is not bool(expected)
        ]
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"未回显: {', '.join(missing)}")
            if mismatched:
                details.append(f"不匹配: {', '.join(mismatched)}")
            raise RuntimeError(
                "服务端没有落实消融会话参数（" + "；".join(details) + "）"
            )
    return {
        "patient_id": resolved_patient_id,
        "patient_has_history": patient_has_history,
        "requested_flags": requested_flags,
        "effective_flags": effective_flags,
        "session_id": session_started_event.get("session_id"),
    }


# ============================================================
# 按实时节奏推送音频
# ============================================================

async def _stream_frames_in_realtime(
    websocket,
    frames: list[bytes],
    *,
    stream_started_at: float,
    frame_offset: int = 0,
) -> float:
    """按墙钟节奏推送二进制帧，返回最后一帧送出的时刻。

    第 n 帧的目标时刻固定为 ``stream_started_at + n × 32ms``，而不是每轮
    ``sleep(0.032)``。后者会把 send 自身的耗时累加进去，8 秒素材可能送成 9 秒，
    让所有以时间为基准的指标一起漂移。
    """
    sent_at = stream_started_at

    for index, frame in enumerate(frames):
        target_at = (
            stream_started_at
            + (frame_offset + index) * FRAME_INTERVAL_S
        )
        delay = target_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        # 二进制帧走 Connection._decode_binary_audio，与浏览器一致。
        await websocket.send(frame)
        sent_at = time.perf_counter()

    return sent_at


async def _stream_silence_until_cancelled(
    websocket,
    *,
    stream_started_at: float,
    frame_offset: int,
) -> None:
    """持续推静音帧，直到被取消。

    不能只推固定长度就停：真实浏览器的麦克风一直开着，音频流不会中断。
    SoulX 每 2560 样本（5 帧）推理一次，断流会让它卡在半个批次上，永远等不到
    下一次判定；VAD 侧同理，需要连续的静音才能累计到 VAD_END_SILENCE_S。
    所以这里一直推，由主协程收齐事件后取消。
    """
    silence_frame = b"\x00" * (FRAME_SAMPLES * 2)
    index = 0

    while True:
        target_at = (
            stream_started_at
            + (frame_offset + index) * FRAME_INTERVAL_S
        )
        delay = target_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        await websocket.send(silence_frame)
        index += 1


# ============================================================
# 单轮测量
# ============================================================

async def _collect_turn_events(
    websocket,
    *,
    deadline: float,
) -> list[tuple[float, dict[str, Any]]]:
    """收事件直到本次输入的应答彻底结束，返回 (到达时刻, 事件) 序列。

    收集期间不做任何归属判断。一次输入会被切成几个轮次、哪个轮次才是该测的
    那个，只有等序列收完才知道；边收边判断是上一版测出全 0ms 的根因。

    何时停：最后一个轮次的应答收尾（拿到 tts_end，或等不到 TTS 而放弃）之后，
    再观察 _TURN_SETTLE_S。这段观察期内若又冒出新轮次，说明序列还没稳定，
    重新开始等。
    """
    timeline: list[tuple[float, dict[str, Any]]] = []

    turn_ids: list[str] = []
    first_ai_text_at: dict[str, float] = {}
    first_tts_byte_at: dict[str, float] = {}
    completed_turns: set[str] = set()
    # 拿到了 AI 文本却等不到 TTS 音频的轮次。TTS 不可用时（火山 403 等）服务端
    # 只在自己日志里记一行，不给客户端任何事件，只能靠超时识别。
    abandoned_turns: set[str] = set()
    settle_started_at: float | None = None

    while True:
        latest_turn_id = turn_ids[-1] if turn_ids else ""
        latest_turn_settled = bool(latest_turn_id) and (
            latest_turn_id in completed_turns
            or latest_turn_id in abandoned_turns
        )

        candidate_deadlines = [deadline]
        if latest_turn_settled:
            if settle_started_at is None:
                settle_started_at = time.perf_counter()
            candidate_deadlines.append(settle_started_at + _TURN_SETTLE_S)
        elif (
            latest_turn_id in first_ai_text_at
            and latest_turn_id not in first_tts_byte_at
        ):
            candidate_deadlines.append(
                first_ai_text_at[latest_turn_id] + _TTS_GRACE_S
            )

        try:
            event = await _receive_json(websocket, min(candidate_deadlines))
        except TimeoutError:
            if latest_turn_settled:
                return timeline
            if latest_turn_id and latest_turn_id in first_ai_text_at:
                abandoned_turns.add(latest_turn_id)
                settle_started_at = None
                continue
            # 连 AI 文本都没等到，链路是真卡住了，交给上层记为失败。
            raise

        arrived_at = time.perf_counter()
        timeline.append((arrived_at, event))

        event_type = event.get("type")
        if event_type == "asr_error":
            raise RuntimeError(str(event.get("detail") or "ASR 失败"))

        turn_id = str(event.get("turn_id") or "")
        if not turn_id:
            continue

        if event_type == "asr_result":
            if turn_id not in turn_ids:
                turn_ids.append(turn_id)
                # 新轮次出现，之前的观察期作废。
                settle_started_at = None
        elif event_type in {"ai_response_chunk", "ai_response"}:
            if event.get("error"):
                raise RuntimeError(
                    str(event.get("text") or "Agent 返回错误")
                )
            first_ai_text_at.setdefault(turn_id, arrived_at)
        elif event_type in {"tts_audio", "tts_chunk"}:
            first_tts_byte_at.setdefault(turn_id, arrived_at)
        elif event_type == "tts_end":
            completed_turns.add(turn_id)


def _select_measured_turn(
    timeline: list[tuple[float, dict[str, Any]]],
) -> dict[str, Any]:
    """从事件序列里挑出该测的轮次，归集它的各时刻与文本。

    取最后一个轮次。SoulX 常在句中停顿（例如逗号）处就判 semantic_complete，
    于是 AI 抢答前半句、后半句语音进来又把它打断；服务端把被打断轮次的
    tts_end 当作 stale 丢弃，客户端永远等不到。只有最后一个轮次的应答完整走
    完，也只有它的起点对得上"用户说完最后一个字"这个零点。

    vad_end 事件不带 turn_id，按时间就近归属：取该轮 asr_result 之前最近的
    那一个。
    """
    observed_turn_ids: list[str] = []
    for _, event in timeline:
        if event.get("type") != "asr_result":
            continue
        turn_id = str(event.get("turn_id") or "")
        if turn_id and turn_id not in observed_turn_ids:
            observed_turn_ids.append(turn_id)

    result: dict[str, Any] = {
        "observed_turn_ids": observed_turn_ids,
        "measured_turn_id": "",
        "marks": {},
        "recognized_text": "",
        "assistant_text": "",
        "final_insight": None,
        "tts_audio": bytearray(),
        "asr_source": "",
    }
    if not observed_turn_ids:
        return result

    measured_turn_id = observed_turn_ids[-1]
    marks: dict[str, float] = {}
    recognized_text = ""
    assistant_text = ""
    final_insight: dict[str, Any] | None = None
    tts_audio = bytearray()
    asr_source = ""
    latest_vad_end_at: float | None = None

    for arrived_at, event in timeline:
        event_type = event.get("type")

        if event_type == "vad_end":
            latest_vad_end_at = arrived_at
            continue

        if event_type == "soulx_state" and str(event.get("text") or "").strip():
            marks.setdefault("soulx_first_text", arrived_at)
            continue

        if str(event.get("turn_id") or "") != measured_turn_id:
            continue

        if event_type == "asr_result":
            marks.setdefault("asr_result", arrived_at)
            if latest_vad_end_at is not None:
                marks.setdefault("vad_end", latest_vad_end_at)
            recognized_text = str(event.get("text") or "")
            asr_source = str(event.get("source") or "unknown")

        elif event_type == "ai_response_chunk":
            marks.setdefault("first_ai_text", arrived_at)
            assistant_text += str(event.get("text") or "")

        elif event_type == "ai_response":
            marks.setdefault("first_ai_text", arrived_at)
            if not assistant_text:
                assistant_text = str(event.get("text") or "")

        elif event_type == "tts_start":
            marks.setdefault("tts_start", arrived_at)

        elif event_type in {"tts_audio", "tts_chunk"}:
            marks.setdefault("first_tts_byte", arrived_at)
            encoded = event.get("audio") or event.get("chunk")
            if encoded:
                try:
                    tts_audio.extend(base64.b64decode(encoded))
                except (ValueError, TypeError):
                    pass

        elif event_type == "tts_end":
            marks.setdefault("tts_end", arrived_at)

        elif event_type == "turn_insight" and event.get("state") == "final":
            final_insight = event

    result.update(
        measured_turn_id=measured_turn_id,
        marks=marks,
        recognized_text=recognized_text,
        assistant_text=assistant_text,
        final_insight=final_insight,
        tts_audio=tts_audio,
        asr_source=asr_source,
    )
    return result


async def _measure_one_turn(
    *,
    websocket_url: str,
    cookie: str,
    speech_frames: list[bytes],
    timeout_s: float,
    patient_id: str | None,
    create_patient: bool,
    profile_name: str,
    long_term_memory_enabled: bool,
    long_term_memory_writes_enabled: bool,
    emotion_enabled: bool,
    verify_effective_flags: bool,
    verify_tls: bool,
    tts_audio_output: Path | None,
) -> dict[str, Any]:

    async with _connect_websocket(
        websocket_url,
        cookie,
        verify_tls=verify_tls,
    ) as websocket:

        handshake = await _start_session(
            websocket,
            patient_id=patient_id,
            create_patient=create_patient,
            profile_name=profile_name,
            long_term_memory_enabled=long_term_memory_enabled,
            long_term_memory_writes_enabled=long_term_memory_writes_enabled,
            emotion_enabled=emotion_enabled,
            verify_effective_flags=verify_effective_flags,
            timeout_s=timeout_s,
        )

        stream_started_at = time.perf_counter()
        # 接收必须先于推流启动，否则音频期间已经到达的 SoulX 增量事件只会在
        # 说完后才被读取，所谓“首字延迟”会退化成音频时长。
        collector_task = asyncio.create_task(
            _collect_turn_events(
                websocket,
                deadline=(
                    stream_started_at
                    + len(speech_frames) * FRAME_INTERVAL_S
                    + timeout_s
                ),
            )
        )
        last_speech_frame_at = await _stream_frames_in_realtime(
            websocket,
            speech_frames,
            stream_started_at=stream_started_at,
        )

        # 静音在后台一直推到本轮收齐为止，模拟麦克风常开。主协程同时已开始收
        # 事件，不会漏掉推静音期间到达的 vad_end。
        silence_task = asyncio.create_task(
            _stream_silence_until_cancelled(
                websocket,
                stream_started_at=stream_started_at,
                frame_offset=len(speech_frames),
            )
        )

        # 所有延迟都以"说完最后一个字"为零点。
        zero_at = last_speech_frame_at

        try:
            timeline = await collector_task
        finally:
            silence_task.cancel()
            await asyncio.gather(silence_task, return_exceptions=True)
            if not collector_task.done():
                collector_task.cancel()
                await asyncio.gather(collector_task, return_exceptions=True)

    selected = _select_measured_turn(timeline)

    if tts_audio_output is not None and selected["tts_audio"]:
        _write_float32_as_wav(selected["tts_audio"], tts_audio_output)

    return _build_record(
        marks=selected["marks"],
        zero_at=zero_at,
        stream_started_at=stream_started_at,
        speech_frame_count=len(speech_frames),
        observed_turn_ids=selected["observed_turn_ids"],
        recognized_text=selected["recognized_text"],
        assistant_text=selected["assistant_text"],
        final_insight=selected["final_insight"],
        patient_id=handshake["patient_id"],
        patient_has_history=handshake["patient_has_history"],
        session_id=handshake["session_id"],
        requested_flags=handshake["requested_flags"],
        effective_flags=handshake["effective_flags"],
        asr_source=selected["asr_source"],
    )


def _write_float32_as_wav(raw: bytearray, destination: Path) -> None:
    """TTS 下发的是 float32 24kHz，落盘前转成 PCM16。"""
    samples = array("f")
    samples.frombytes(bytes(raw[: len(raw) - len(raw) % 4]))
    pcm16 = array(
        "h",
        (
            round(max(-1.0, min(1.0, value)) * 32767)
            for value in samples
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(24000)
        sink.writeframes(pcm16.tobytes())


def _build_record(
    *,
    marks: dict[str, float],
    zero_at: float,
    stream_started_at: float,
    speech_frame_count: int,
    observed_turn_ids: list[str],
    recognized_text: str,
    assistant_text: str,
    final_insight: dict[str, Any] | None,
    patient_id: str | None,
    patient_has_history: bool,
    session_id: str | None,
    requested_flags: dict[str, Any],
    effective_flags: dict[str, Any],
    asr_source: str,
) -> dict[str, Any]:
    """把原始时间戳换算成指标。所有值以 zero_at 为零点。"""

    def elapsed_ms(mark: str) -> float | None:
        if mark not in marks:
            return None
        return round((marks[mark] - zero_at) * 1000, 1)

    def difference_ms(later: str, earlier: str) -> float | None:
        if later not in marks or earlier not in marks:
            return None
        return round((marks[later] - marks[earlier]) * 1000, 1)

    record: dict[str, Any] = {
        "patient_id": patient_id,
        "session_id": session_id,
        "patient_has_history": patient_has_history,
        "session_flags_requested": requested_flags,
        "session_flags_effective": effective_flags,
        "asr_source": asr_source,
        "speech_duration_ms": round(
            speech_frame_count * FRAME_INTERVAL_S * 1000,
            1,
        ),
        # 大于 1 说明这段话被切成了多个轮次。
        "turn_count": len(observed_turn_ids),
        "recognized_text": recognized_text,
        "assistant_text": assistant_text,
        # 保留相对零点的原始时刻，便于事后核对与排查。
        "raw_marks_ms": {
            mark: elapsed_ms(mark)
            for mark in (
                "vad_end",
                "soulx_first_text",
                "asr_result",
                "first_ai_text",
                "tts_start",
                "first_tts_byte",
                "tts_end",
            )
            if mark in marks
        },
    }

    if "soulx_first_text" in marks:
        record["soulx_first_text_ms"] = round(
            (marks["soulx_first_text"] - stream_started_at) * 1000,
            1,
        )
    record["turn_decision_ms"] = elapsed_ms("vad_end")
    record["post_vad_pipeline_ms"] = difference_ms("asr_result", "vad_end")
    # Backward-compatible aliases.  They remain in raw records but are no
    # longer presented as pure provider timings.
    record["asr_ms"] = record["post_vad_pipeline_ms"]
    record["agent_pipeline_ms"] = difference_ms(
        "first_ai_text", "asr_result"
    )
    record["llm_first_ms"] = record["agent_pipeline_ms"]
    record["tts_queue_ms"] = difference_ms("tts_start", "first_ai_text")
    record["tts_provider_first_byte_ms"] = difference_ms(
        "first_tts_byte", "tts_start"
    )
    record["tts_first_byte_ms"] = difference_ms(
        "first_tts_byte",
        "first_ai_text",
    )
    record["perceived_ms"] = elapsed_ms("first_tts_byte")
    record["full_response_ms"] = elapsed_ms("tts_end")

    if isinstance(final_insight, dict):
        emotion = final_insight.get("emotion")
        if isinstance(emotion, dict):
            record["emotion_dominant"] = emotion.get("dominant")
            record["emotion_source"] = emotion.get("source")
            record["emotion_audio_model_used"] = bool(
                emotion.get("audio_model_used")
            )
        memory = final_insight.get("memory")
        if isinstance(memory, dict):
            record["memory_used_items"] = len(
                memory.get("used_item_ids") or []
            )
            record["memory_retrieval_source"] = memory.get(
                "retrieval_source"
            )
            record["memory_retrieval_kind"] = memory.get("retrieval_kind")
            record["memory_retrieval_hit"] = bool(
                memory.get("retrieval_hit")
            )
            record["memory_retrieval_context_chars"] = int(
                memory.get("retrieval_context_chars") or 0
            )
            record["memory_retrieval_elapsed_ms"] = float(
                memory.get("retrieval_elapsed_ms") or 0.0
            )
            record["memory_writes_enabled"] = bool(
                memory.get("writes_enabled", True)
            )

    record["valid_for_latency"] = len(observed_turn_ids) == 1

    return {key: value for key, value in record.items() if value is not None}


# ============================================================
# 批量运行
# ============================================================

def _build_run_specs(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    total = arguments.warmup_runs + arguments.runs
    if arguments.ablation == "none":
        return [
            {
                "unit_index": index,
                "is_warmup": index < arguments.warmup_runs,
                "pair_index": None,
                "condition": "default",
                "memory_enabled": not arguments.disable_long_term_memory,
                "emotion_enabled": arguments.emotion == "on",
                "memory_writes_enabled": arguments.allow_memory_writes,
            }
            for index in range(total)
        ]

    randomizer = random.Random(arguments.seed)
    specs: list[dict[str, Any]] = []
    for index in range(total):
        if arguments.ablation == "memory":
            conditions = [
                ("memory_off", False, arguments.emotion == "on"),
                ("memory_on", True, arguments.emotion == "on"),
            ]
        else:
            conditions = [
                ("emotion_off", not arguments.disable_long_term_memory, False),
                ("emotion_on", not arguments.disable_long_term_memory, True),
            ]
        randomizer.shuffle(conditions)
        for condition, memory_enabled, emotion_enabled in conditions:
            specs.append(
                {
                    "unit_index": index,
                    "is_warmup": index < arguments.warmup_runs,
                    "pair_index": (
                        None if index < arguments.warmup_runs
                        else index - arguments.warmup_runs + 1
                    ),
                    "condition": condition,
                    "memory_enabled": memory_enabled,
                    "emotion_enabled": emotion_enabled,
                    # Ablation sessions are read-only so the first arm cannot
                    # alter the persistent state observed by the second arm.
                    "memory_writes_enabled": False,
                }
            )
    return specs


def _condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(record.get("condition") or "default") for record in records})
    result: dict[str, Any] = {}
    for condition in conditions:
        selected = [record for record in records if record.get("condition") == condition]
        valid = [record for record in selected if record.get("valid_for_latency")]
        result[condition] = {
            "attempted_records": len(selected),
            "valid_single_turn_records": len(valid),
            "metrics": {
                metric: _summarize(
                    [
                        float(record[metric])
                        for record in (
                            selected if metric == "soulx_first_text_ms" else valid
                        )
                        if isinstance(record.get(metric), (int, float))
                    ]
                )
                for metric in _METRICS
            },
            "quality": {
                "asr_cer": (
                    {
                        "count": len(cer_values),
                        "mean": round(statistics.fmean(cer_values), 4),
                        "median": round(statistics.median(cer_values), 4),
                    }
                    if (
                        cer_values := [
                            float(record["asr_cer"])
                            for record in selected
                            if isinstance(record.get("asr_cer"), (int, float))
                        ]
                    )
                    else {"count": 0}
                ),
                "memory_retrieval_hits": sum(
                    bool(record.get("memory_retrieval_hit"))
                    for record in selected
                ),
            },
        }
    return result


async def _run_benchmark(arguments: argparse.Namespace) -> int:
    audio_cases = _prepare_audio_cases(
        _load_audio_cases(arguments),
        no_trim=arguments.no_trim,
    )
    websocket_url = _websocket_url(arguments.server)
    verify_tls = not arguments.insecure
    health = _server_health(arguments.server, verify_tls=verify_tls)
    release_flags = dict(health.get("release_flags") or {})

    if arguments.ablation != "none":
        if not arguments.patient_id:
            raise RuntimeError("消融实验必须用 --patient-id 固定同一个已有患者")
        if not release_flags.get("turn_insight"):
            raise RuntimeError("消融实验需要服务端 ENABLE_TURN_INSIGHT=true")
        if arguments.ablation == "memory" and not release_flags.get(
            "long_term_memory"
        ):
            raise RuntimeError("服务端长期记忆能力未启用，不能做记忆消融")
        if arguments.ablation == "emotion" and not release_flags.get("emotion"):
            raise RuntimeError("服务端情绪模型未启用，不能做情绪消融")

    cookie = _login(
        arguments.server,
        arguments.username,
        arguments.password,
        verify_tls=verify_tls,
    )

    print(f"音频样本 {len(audio_cases)} 条，按轮次循环并在 A/B 两组内严格配对")
    for case in audio_cases:
        print(
            f"  {case['sample_id']}: {case['speech_seconds']:.2f}s，"
            f"{len(case['frames'])} × {FRAME_INTERVAL_S * 1000:.0f}ms，"
            f"裁剪前/后 {case['raw_seconds']:.2f}s/{case['speech_seconds']:.2f}s"
        )
    unit = "对" if arguments.ablation != "none" else "轮"
    print(
        f"预热 {arguments.warmup_runs} {unit}，计入统计 "
        f"{arguments.runs} {unit}  静音持续推送至本轮收齐"
    )
    if arguments.ablation != "none":
        print(
            f"消融模式 {arguments.ablation}：同一患者配对 A/B，"
            f"顺序随机化（seed={arguments.seed}），长期记忆写入强制关闭"
        )
    if arguments.patient_id:
        print(
            f"复用患者 {arguments.patient_id}；"
            "长期记忆写入="
            f"{'开启' if arguments.allow_memory_writes and arguments.ablation == 'none' else '冻结'}"
        )
    elif arguments.reuse_patient:
        print(
            "首轮新建患者，后续轮次复用同一患者；"
            "长期记忆写入="
            f"{'开启' if arguments.allow_memory_writes and arguments.ablation == 'none' else '冻结'}"
        )
    else:
        print(
            "未指定 --patient-id：每轮新建患者，长期记忆检索为空。"
            "做记忆消融时务必改用固定患者，否则两组都在测空检索。"
        )
    print()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # 首轮拿到的 patient_id 供后续轮次复用，避免每轮都新建。
    active_patient_id = arguments.patient_id

    run_specs = _build_run_specs(arguments)
    for index, spec in enumerate(run_specs):
        audio_case = audio_cases[int(spec["unit_index"]) % len(audio_cases)]
        speech_frames = audio_case["frames"]
        is_warmup = bool(spec["is_warmup"])
        pair_index = spec["pair_index"]
        condition = str(spec["condition"])
        label = (
            f"预热/{condition}"
            if is_warmup
            else (
                f"第 {pair_index} 对/{condition}"
                if pair_index is not None
                else f"第 {len(records) + 1} 轮"
            )
        )

        audio_output = None
        if arguments.save_tts_audio and not is_warmup:
            audio_output = (
                Path(arguments.save_tts_audio)
                / f"{condition}_{pair_index or index:03d}.wav"
            )

        try:
            record = await _measure_one_turn(
                websocket_url=websocket_url,
                cookie=cookie,
                speech_frames=speech_frames,
                timeout_s=arguments.timeout,
                patient_id=active_patient_id,
                create_patient=active_patient_id is None,
                profile_name=f"bench_{uuid.uuid4().hex[:8]}",
                long_term_memory_enabled=bool(spec["memory_enabled"]),
                long_term_memory_writes_enabled=bool(
                    spec["memory_writes_enabled"]
                ),
                emotion_enabled=bool(spec["emotion_enabled"]),
                verify_effective_flags=True,
                verify_tls=verify_tls,
                tts_audio_output=audio_output,
            )
            record["condition"] = condition
            record["pair_index"] = pair_index
            record["sample_id"] = audio_case["sample_id"]
            record["audio_sha256"] = audio_case["sha256"]
            if audio_case.get("reference_text"):
                record["asr_cer"] = _character_error_rate(
                    str(audio_case["reference_text"]),
                    str(record.get("recognized_text") or ""),
                )

            if arguments.ablation == "memory":
                # Memory ablation is a SoulX turn-taking experiment.  If the
                # external service drops and the server falls back to local
                # VAD/Ark ASR, the timing origin changes and the pair is not
                # comparable; reject it instead of silently mixing pipelines.
                if record.get("asr_source") != "soulx_final":
                    raise RuntimeError(
                        "本轮未使用 SoulX（asr_source != soulx_final），已拒绝降级样本"
                    )
                if condition == "memory_on" and not record.get(
                    "patient_has_history"
                ):
                    raise RuntimeError(
                        "memory_on 患者没有历史记忆，A/B 实际会比较空检索"
                    )
                if record.get("memory_writes_enabled"):
                    raise RuntimeError("消融会话仍在写长期记忆，已拒绝污染样本")
                expected_source = "disabled" if condition == "memory_off" else None
                if expected_source and record.get("memory_retrieval_source") != expected_source:
                    raise RuntimeError("memory_off 并未真正关闭长期记忆检索")
                # 查询无关的本地兜底会把三条最近记忆原样倒进 prompt，与
                # "语义检索是否有用"无关。它一旦被当成命中，消融比较的就是
                # "有没有贴记忆卡"而不是"检索是否生效"，必须单独拒绝。
                if (
                    condition == "memory_on"
                    and record.get("memory_retrieval_kind") == "local_fallback"
                ):
                    raise RuntimeError(
                        "memory_on 用的是查询无关的本地兜底（retrieval_kind="
                        "local_fallback），不是语义检索；请先让 Memobase 可用"
                    )
                if (
                    condition == "memory_on"
                    and arguments.require_memory_hit
                    and not record.get("memory_retrieval_hit")
                ):
                    raise RuntimeError("memory_on 本轮未命中记忆，不能计入有效消融")

            if arguments.ablation == "emotion":
                if condition == "emotion_off" and record.get("emotion_source") != "disabled":
                    raise RuntimeError("emotion_off 并未真正关闭情绪推理")
                if condition == "emotion_on" and (
                    record.get("emotion_source") != "emotion2vec_audio+text"
                    or not record.get("emotion_audio_model_used")
                ):
                    raise RuntimeError("emotion_on 没有实际运行 Emotion2Vec 音频模型")
        except Exception as exc:
            message = f"{label} 失败: {type(exc).__name__}: {exc}"
            print(f"  {message}")
            failures.append(
                {
                    "condition": condition,
                    "pair_index": pair_index,
                    "warmup": is_warmup,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            continue

        if arguments.reuse_patient and active_patient_id is None:
            active_patient_id = record.get("patient_id")

        turn_note = ""
        if record.get("turn_count", 1) > 1:
            turn_note = f"  ⚠️ 被切成 {record['turn_count']} 个轮次"

        print(
            f"  {label}  "
            f"判定 {record.get('turn_decision_ms', float('nan')):.0f}ms  "
            f"SoulX首字 {record.get('soulx_first_text_ms', float('nan')):.0f}ms  "
            f"VAD后处理 {record.get('post_vad_pipeline_ms', float('nan')):.0f}ms  "
            f"Agent {record.get('agent_pipeline_ms', float('nan')):.0f}ms  "
            f"TTS合成 {record.get('tts_provider_first_byte_ms', float('nan')):.0f}ms  "
            f"感知 {record.get('perceived_ms', float('nan')):.0f}ms"
            f"{turn_note}"
        )
        if arguments.show_text:
            print(f"        识别: {record.get('recognized_text', '')!r}")

        if not is_warmup:
            records.append(record)

        if index + 1 < len(run_specs) and arguments.pause > 0:
            await asyncio.sleep(arguments.pause)

    _print_summary(records, failures)

    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "git": _git_state(),
                    "server_health": health,
                    "configuration": {
                        "server": arguments.server,
                        "audio": str(Path(arguments.audio).resolve()),
                        "audio_manifest": arguments.audio_manifest,
                        "ablation": arguments.ablation,
                        "seed": arguments.seed,
                        "patient_id": arguments.patient_id,
                        "disable_long_term_memory": arguments.disable_long_term_memory,
                        "emotion": arguments.emotion,
                        "reference_text": arguments.reference_text,
                        "require_memory_hit": arguments.require_memory_hit,
                        "allow_memory_writes": arguments.allow_memory_writes,
                    },
                    "audio_manifest": (
                        str(Path(arguments.audio_manifest).resolve())
                        if arguments.audio_manifest
                        else None
                    ),
                    "audio_samples": [
                        {
                            "sample_id": case["sample_id"],
                            "audio_path": case["audio_path"],
                            "audio_sha256": case["sha256"],
                            "speech_duration_s": round(case["speech_seconds"], 3),
                            "reference_text": case.get("reference_text"),
                            "trim_report": case["trim_report"],
                        }
                        for case in audio_cases
                    ],
                    "frame_ms": round(FRAME_INTERVAL_S * 1000, 1),
                    "metric_definitions": {
                        "soulx_first_text_ms": "audio stream start -> first non-empty soulx_state text",
                        "turn_decision_ms": "last speech frame sent -> vad_end",
                        "post_vad_pipeline_ms": "vad_end -> asr_result; includes submission/queue/publication",
                        "agent_pipeline_ms": "asr_result -> first AI text; includes memory and agent setup",
                        "tts_queue_ms": "first AI text -> tts_start",
                        "tts_provider_first_byte_ms": "tts_start -> first TTS audio event",
                        "tts_first_byte_ms": "first AI text -> first TTS audio event",
                        "perceived_ms": "last speech frame sent -> first TTS audio event",
                        "full_response_ms": "last speech frame sent -> tts_end",
                        "asr_ms": "legacy alias of post_vad_pipeline_ms; not pure ASR",
                        "llm_first_ms": "legacy alias of agent_pipeline_ms; not pure LLM",
                    },
                    "warmup_runs": arguments.warmup_runs,
                    "requested_runs": arguments.runs,
                    "successful_records": len(records),
                    "valid_single_turn_records": sum(
                        bool(record.get("valid_for_latency")) for record in records
                    ),
                    "failures": failures,
                    "summary_by_condition": _condition_summary(records),
                    "paired_deltas": _paired_summary(
                        [record for record in records if record.get("valid_for_latency")]
                    ),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n明细已写入 {output_path}")

    return 0 if records else 1


def _print_summary(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 72)
    valid_records = [record for record in records if record.get("valid_for_latency")]
    print(f"汇总（成功 {len(records)}，有效单轮 {len(valid_records)}）")
    print("=" * 72)

    labels = {
        "soulx_first_text_ms": "SoulX首字 (音频开始→首增量)",
        "turn_decision_ms": "轮次判定 (最后一帧→vad_end)",
        "post_vad_pipeline_ms": "VAD后处理 (含提交/排队/发布)",
        "agent_pipeline_ms": "Agent链路 (含记忆/准备/LLM)",
        "tts_queue_ms": "TTS排队 (首文本→tts_start)",
        "tts_provider_first_byte_ms": "TTS合成 (tts_start→首音频)",
        "tts_first_byte_ms": "TTS总等待 (首文本→首音频)",
        "perceived_ms": "用户感知 (最后一帧→首音频)",
        "full_response_ms": "完整响应 (最后一帧→tts_end)",
    }

    conditions = sorted(
        {str(record.get("condition") or "default") for record in records}
    ) or ["default"]
    for condition in conditions:
        condition_records = [
            record
            for record in records
            if str(record.get("condition") or "default") == condition
        ]
        condition_valid = [
            record for record in condition_records if record.get("valid_for_latency")
        ]
        if len(conditions) > 1:
            print(
                f"\n  [{condition}] 成功 {len(condition_records)}，"
                f"有效单轮 {len(condition_valid)}"
            )
        for metric in _METRICS:
            source_records = (
                condition_records
                if metric == "soulx_first_text_ms"
                else condition_valid
            )
            values = [
                float(record[metric])
                for record in source_records
                if isinstance(record.get(metric), (int, float))
            ]
            if not values:
                suffix = "（TTS 未产出音频）" if metric in _TTS_DEPENDENT_METRICS else ""
                print(f"  {labels[metric]:<32} 无数据{suffix}")
                continue

            summary = _summarize(values)
            line = (
                f"  {labels[metric]:<32} "
                f"中位数 {summary['p50_ms']:>7.0f}ms  "
                f"均值 {summary['mean_ms']:>7.0f}ms"
            )
            if "p95_ms" in summary:
                line += f"  p95 {summary['p95_ms']:>7.0f}ms"
            line += f"  最大 {summary['max_ms']:>7.0f}ms"
            print(line)

    minimum_condition_n = min(
        (
            sum(
                bool(record.get("valid_for_latency"))
                for record in records
                if str(record.get("condition") or "default") == condition
            )
            for condition in conditions
        ),
        default=0,
    )
    if minimum_condition_n < _MIN_SAMPLES_FOR_P95:
        print(
            f"\n  每组有效样本不足 {_MIN_SAMPLES_FOR_P95} 个，已省略 p95："
            "小样本下它等于最大值，报出来会被误读为独立统计量。"
        )

    split_turns = [
        record for record in records
        if record.get("turn_count", 1) > 1
    ]
    if split_turns:
        print(
            f"\n  ⚠️ {len(split_turns)}/{len(records)} 轮输入被切成多个轮次，"
            "这些样本已从分阶段延迟统计中剔除。"
        )

    if len({record.get("condition") for record in records}) > 1:
        paired = _paired_summary(valid_records)
        print(f"\n  完整有效配对: {paired.get('complete_pairs', 0)}")
        for metric, summary in paired.get("metrics", {}).items():
            print(
                f"    Δ {labels[metric]} ({summary['definition']}): "
                f"中位数 {summary['p50_ms']:+.0f}ms，"
                f"均值 {summary['mean_ms']:+.0f}ms"
            )

    if failures:
        print(f"\n  失败 {len(failures)} 轮:")
        for failure in failures:
            print(
                f"    {failure.get('condition')} pair={failure.get('pair_index')}: "
                f"{failure.get('error_type')}: {failure.get('detail')}"
            )


# ============================================================
# 命令行
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按真实说话节奏（32ms 帧）测量完整语音链路延迟，"
            "计时零点为最后一帧语音送出时刻。"
        ),
    )

    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8426",
        help="语音服务地址，默认 http://127.0.0.1:8426",
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--audio",
        default=str(_DEFAULT_AUDIO_PATH),
        help=(
            "单声道 PCM16 16kHz WAV，默认用 "
            f"tests/fixtures/{_DEFAULT_AUDIO_PATH.name}"
            "（旧 benchmark 用的同一段素材，便于新旧对比）"
        ),
    )
    parser.add_argument(
        "--audio-manifest",
        help=(
            "JSON 音频清单；格式为 samples 数组，每项含 sample_id、"
            "audio_path、reference_text。指定后覆盖 --audio，"
            "不同轮次循环取样且每个 A/B 配对使用同一条音频"
        ),
    )

    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=20,
        help="计入统计的轮次数，默认 20（p95 需要至少 20）",
    )
    parser.add_argument(
        "--warmup-runs",
        type=_non_negative_int,
        default=3,
        help=(
            "预热轮次，不计入统计，默认 3。"
            "首轮要付 TTS 建连、SoulX 首次推理等冷启动成本。"
        ),
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="轮次之间的间隔秒数，默认 1.0",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help=(
            "不裁首尾静音，整段原样送出。"
            "默认会裁，因为服务落盘的录音末尾常带 1 秒以上静音，"
            "不裁会让轮次判定耗时被低估同样的量。"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="单轮等待上限秒数，默认 60",
    )

    patient = parser.add_mutually_exclusive_group()
    patient.add_argument(
        "--patient-id",
        help=(
            "复用已有患者。做记忆消融时必须指定，"
            "否则新患者的记忆检索为空，两组消融的都是空检索。"
        ),
    )
    patient.add_argument(
        "--reuse-patient",
        action="store_true",
        help="首轮新建患者，后续轮次复用它，让上下文自然增长",
    )

    parser.add_argument(
        "--disable-long-term-memory",
        action="store_true",
        help="普通单组测试时关闭长期记忆；配对消融请用 --ablation memory",
    )
    parser.add_argument(
        "--emotion",
        choices=("on", "off"),
        default="on",
        help="普通测试的会话级情绪推理开关，默认 on",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "memory", "emotion"),
        default="none",
        help=(
            "运行真正的配对消融：同一患者、同一音频、A/B 顺序随机，"
            "并冻结长期记忆写入"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260910,
        help="配对条件顺序的随机种子",
    )
    parser.add_argument(
        "--require-memory-hit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="记忆消融中要求 memory_on 确实命中相关长期记忆（默认开启）",
    )
    parser.add_argument(
        "--reference-text",
        help="音频的人工参考转写；提供后逐轮计算 CER",
    )
    parser.add_argument(
        "--allow-memory-writes",
        action="store_true",
        help=(
            "允许 benchmark 回写长期记忆。默认关闭以冻结患者状态；"
            "消融模式下即使指定也仍强制关闭"
        ),
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="打印每轮的识别文本，用于核对漏字",
    )
    parser.add_argument(
        "--save-tts-audio",
        help="把每轮的 TTS 音频存到该目录",
    )
    parser.add_argument(
        "--output",
        help="把汇总与明细写入该 JSON 文件",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="跳过 TLS 证书校验（仅用于自签名证书的测试环境）",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.ablation == "memory" and arguments.disable_long_term_memory:
        parser.error("--ablation memory 不能和 --disable-long-term-memory 同时使用")
    if arguments.ablation != "none" and arguments.reuse_patient:
        parser.error("配对消融必须显式提供 --patient-id，不能使用 --reuse-patient")
    try:
        return asyncio.run(_run_benchmark(arguments))
    except KeyboardInterrupt:
        print("\n已中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
