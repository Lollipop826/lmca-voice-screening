"""FastAPI entry point for the Alzheimer's voice-screening application.

This module is deliberately limited to configuration and dependency wiring.
Voice-session behavior lives in ``src.voice`` and HTTP behavior in ``src.web``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def _load_environment() -> None:
    """Load the deployment-specific .env before reading configuration."""
    if getattr(sys, "frozen", False):
        env_path = os.path.join(
            os.path.dirname(sys.executable),
            ".env",
        )
    else:
        env_path = ".env"
    load_dotenv(env_path)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


_load_environment()

from src.context_management.emotion_memobase import EmotionMemobase
from src.agents.wellbeing_companion_agent import WellbeingCompanionAgent
from src.tools.emotion import get_audio_emotion_classifier
from src.db.database import (
    assign_patient as db_assign_patient,
    create_patient as db_create_patient,
    create_session as db_create_session,
    get_patient as db_get_patient,
    init_db,
    link_session_patient as db_link_session_patient,
    update_patient_profile as db_update_patient_profile,
    record_patient_audit_event,
)
from src.voice import (
    PatientMemoryService,
    VoiceEndpointApplication,
    VoiceEndpointConfig,
    VoiceModelRuntime,
    VoiceRecognitionService,
)
from src.voice.media import (
    AIORTC_IMPORT_ERROR as _AIORTC_IMPORT_ERROR,
    HybridMediaRegistry,
    HybridVoiceTransport,
    RTCPeerConnection,
    RTCSessionDescription,
    WebRTCPeerSession,
)
from src.web import (
    ApplicationHttpController,
    ApplicationLogBroker,
    AuthController,
    AuthService,
    PublicApiController,
    PublicApiKeyService,
    PublicApiSessionStore,
    load_context_registry,
    MemoryApiController,
)
from src.voice_modes import normalize_session_mode
from src.voice.health import SoulXHealthProbe


_BASE_DIR = Path(__file__).parent
_PIXI_VIEWER_DIR = Path(os.getenv("PIXI_VIEWER_ROOT", "/data/student1/yzy/pixijs-live2d-spine-viewer")).resolve()
_IS_FROZEN = getattr(sys, "frozen", False)

_TTS_PREVIEW_TEXT = "您好，我会陪您慢慢聊，按自己的节奏就好。"
_TTS_VOICES = (
    {"id": "zh_female_vv_uranus_bigtts", "name": "Vivi 2.0"},
    {"id": "zh_female_shuangkuaisisi_moon_bigtts", "name": "爽快思思"},
    {"id": "zh_male_yuanboxiaoshu_moon_bigtts", "name": "元波小叔"},
    {"id": "zh_male_jieshuonansheng_mars_bigtts", "name": "解说男声"},
)
_TTS_VOICE_BY_ID = {item["id"]: item for item in _TTS_VOICES}
_TTS_PREVIEW_DIR = Path(
    os.getenv("TTS_PREVIEW_DIR", str(_BASE_DIR / "data" / "tts_previews"))
).resolve()
_TTS_SETTINGS_PATH = Path(
    os.getenv("TTS_SETTINGS_PATH", str(_BASE_DIR / "data" / "tts_settings.json"))
).resolve()
_TTS_PREVIEW_LOCK = asyncio.Lock()


def _load_saved_tts_voice() -> str | None:
    try:
        payload = json.loads(_TTS_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    voice_id = str(payload.get("voice_id") or "").strip()
    return voice_id if voice_id in _TTS_VOICE_BY_ID else None


def _env_bool(name: str, default: bool) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_DEFAULT_SESSION_MODE = normalize_session_mode(None)
_ENABLE_COGNITIVE_SCREENING = _env_bool("ENABLE_COGNITIVE_SCREENING", True)
_ENABLE_TURN_INSIGHT = _env_bool("ENABLE_TURN_INSIGHT", True)
_ENABLE_LONG_TERM_MEMORY = _env_bool("ENABLE_LONG_TERM_MEMORY", True)
_ENABLE_LONG_TERM_MEMORY_WRITES = _env_bool("ENABLE_LONG_TERM_MEMORY_WRITES", True)
_ENABLE_EMOTION = _env_bool("ENABLE_EMOTION", True)
_ENABLE_PATIENT_MEMORY_DELETE = _env_bool("ENABLE_PATIENT_MEMORY_DELETE", True)
_ENABLE_DEMO_DATA = _env_bool("ENABLE_DEMO_DATA", False)
_USE_ARK_ASR = (
    os.getenv("USE_ARK_ASR", "true" if _IS_FROZEN else "false").lower()
    == "true"
)
_USE_ARK_TTS = (
    os.getenv("USE_ARK_TTS", "true" if _IS_FROZEN else "false").lower()
    == "true"
)
if not os.getenv("VOLC_TTS_SPEAKER_ID", "").strip():
    _saved_tts_voice = _load_saved_tts_voice()
    if _saved_tts_voice:
        os.environ["VOLC_TTS_VOICE"] = _saved_tts_voice
_USE_SPEAKER_VERIFIER = (
    os.getenv(
        "USE_SPEAKER_VERIFIER",
        "false" if _IS_FROZEN else "true",
    ).lower()
    == "true"
)
_USE_LOCAL_EMBEDDING = (
    os.getenv(
        "USE_LOCAL_EMBEDDING",
        "false" if _IS_FROZEN else "true",
    ).lower()
    == "true"
)
_USE_LLM_STREAMING = (
    os.getenv("USE_LLM_STREAMING", "true").lower() == "true"
)
_ENABLE_FULL_DUPLEX = (
    os.getenv("ENABLE_FULL_DUPLEX", "true").lower() == "true"
)
_USE_SOULX_TURN_TAKING = (
    os.getenv("USE_SOULX_TURN_TAKING", "true").lower() == "true"
)
_SOULX_TURN_URL = os.getenv(
    "SOULX_TURN_URL",
    "ws://127.0.0.1:8000/turn",
).strip()
_SOULX_TIMEOUT_S = float(os.getenv("SOULX_TIMEOUT_S", "3.0"))
_SOULX_RETRY_INTERVAL_S = float(
    os.getenv("SOULX_RETRY_INTERVAL_S", "5.0")
)
_SOULX_HEALTH = SoulXHealthProbe(_SOULX_TURN_URL)
_SOULX_MIN_UTTERANCE_RMS = float(
    os.getenv("SOULX_MIN_UTTERANCE_RMS", "0.008")
)
# nonidle 打断的单 chunk 能量下限。默认 0 表示保持原行为（不设闸），
# 需要按现场回声实测标定后再调高，定太高会让真实打断失灵。
_SOULX_BARGE_IN_MIN_CHUNK_RMS = float(
    os.getenv("SOULX_BARGE_IN_MIN_CHUNK_RMS", "0")
)
_SOULX_PRE_ROLL_S = float(os.getenv("SOULX_PRE_ROLL_S", "2.0"))
_VAD_PRE_END_ARM_WINDOW_S = float(
    os.getenv("VAD_PRE_END_ARM_WINDOW_S", "0.25")
)
_VAD_MIN_POST_TTS_CHUNKS = max(
    1,
    int(os.getenv("VAD_MIN_POST_TTS_CHUNKS", "2")),
)
_FULL_DUPLEX_TRIGGER_PROB = float(
    os.getenv("FULL_DUPLEX_TRIGGER_PROB", "0.72")
)
_FULL_DUPLEX_MIN_TRIGGER_DURATION = float(
    os.getenv("FULL_DUPLEX_MIN_TRIGGER_DURATION", "0.45")
)
_FULL_DUPLEX_STOP_DURATION = float(
    os.getenv("FULL_DUPLEX_STOP_DURATION", "0.20")
)
_FULL_DUPLEX_MIN_RMS = float(
    os.getenv("FULL_DUPLEX_MIN_RMS", "0.010")
)
_FULL_DUPLEX_MIN_CONSECUTIVE_CHUNKS = max(
    1,
    int(os.getenv("FULL_DUPLEX_MIN_CONSECUTIVE_CHUNKS", "2")),
)
_FULL_DUPLEX_COMPLETE_SILENCE_S = float(
    os.getenv("FULL_DUPLEX_COMPLETE_SILENCE_S", "0.45")
)
_FULL_DUPLEX_MIN_COMPLETE_AUDIO_S = float(
    os.getenv("FULL_DUPLEX_MIN_COMPLETE_AUDIO_S", "0.45")
)
_ANSWER_COMPLETION_OBSERVATION_WINDOW_S = float(
    os.getenv("ANSWER_COMPLETION_OBSERVATION_WINDOW_S", "0")
)
_ENABLE_REALTIME_COMPANION = (
    os.getenv("ENABLE_REALTIME_COMPANION", "true").lower() == "true"
)
_REALTIME_EMOTION_INTERVAL_S = max(
    0.2,
    float(os.getenv("REALTIME_EMOTION_INTERVAL_S", "0.8")),
)
_REALTIME_EMOTION_WINDOW_S = max(
    1.0,
    float(os.getenv("REALTIME_EMOTION_WINDOW_S", "3.0")),
)
_MEMORY_RETRIEVAL_TIMEOUT_S = max(
    0.05,
    float(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_S", "0.25")),
)
_MEMORY_PREFETCH_STABILITY_S = max(
    0.05,
    float(os.getenv("MEMORY_PREFETCH_STABILITY_S", "0.35")),
)
_MEMORY_PREFETCH_MIN_INTERVAL_S = max(
    0.05,
    float(os.getenv("MEMORY_PREFETCH_MIN_INTERVAL_S", "0.5")),
)
_LEGACY_SYSTEM_URL = os.getenv("LEGACY_SYSTEM_URL", "").strip()


from src.agents.screening_agent_function_calling import (
    ADScreeningAgentFunctionCalling as ADScreeningAgent,
)

AGENT_TYPE = "FunctionCalling"

if _USE_ARK_TTS:
    from src.tools.voice.ark_tts import ArkTTS as VoiceTTS
else:
    from src.tools.voice.zipvoice_tts import ZipVoiceTTS as VoiceTTS


def _load_ice_servers_from_env() -> list[dict]:
    """Build browser ICE configuration from STUN/TURN environment values."""
    stun_urls = os.getenv(
        "WEBRTC_STUN_URLS",
        (
            "stun:stun.l.google.com:19302,"
            "stun:stun1.l.google.com:19302,"
            "stun:stun.miwifi.com:3478"
        ),
    )
    servers = [
        {"urls": url.strip()}
        for url in stun_urls.split(",")
        if url.strip()
    ]
    turn_url = os.getenv("WEBRTC_TURN_URL", "").strip()
    if not turn_url:
        return servers

    turn_server = {"urls": turn_url}
    username = os.getenv("WEBRTC_TURN_USERNAME", "").strip()
    credential = os.getenv("WEBRTC_TURN_CREDENTIAL", "").strip()
    if username:
        turn_server["username"] = username
    if credential:
        turn_server["credential"] = credential
    servers.append(turn_server)
    return servers


LOG_BROKER = ApplicationLogBroker()
LOG_BROKER.install_stdio()
EMOTION_MEMORY = EmotionMemobase(
    audit_event=lambda **event: record_patient_audit_event(
        actor_username="system",
        **event,
    ),
)

PATIENT_MEMORY_SERVICE = PatientMemoryService(
    get_patient=db_get_patient,
    create_patient=db_create_patient,
    update_patient_profile=db_update_patient_profile,
    link_session_patient=db_link_session_patient,
    assign_patient=db_assign_patient,
    long_term_memory=EMOTION_MEMORY if _ENABLE_LONG_TERM_MEMORY else None,
    long_term_memory_writes=_ENABLE_LONG_TERM_MEMORY_WRITES,
)
MODEL_RUNTIME = VoiceModelRuntime(
    base_dir=_BASE_DIR,
    agent_factory=ADScreeningAgent,
    agent_type=AGENT_TYPE,
    tts_factory=VoiceTTS,
    use_ark_asr=_USE_ARK_ASR,
    use_ark_tts=_USE_ARK_TTS,
    use_speaker_verifier=_USE_SPEAKER_VERIFIER,
    use_local_embedding=_USE_LOCAL_EMBEDDING,
    wellbeing_agent_factory=WellbeingCompanionAgent,
)
RECOGNITION = VoiceRecognitionService(MODEL_RUNTIME)

app = FastAPI()
AUTH = AuthService(data_dir=_BASE_DIR / "data")
VOICE_APPLICATION = VoiceEndpointApplication(
    auth=AUTH,
    models=MODEL_RUNTIME,
    recognition=RECOGNITION,
    patient_memory_service=PATIENT_MEMORY_SERVICE,
    render_manager=None,
    config=VoiceEndpointConfig(
        use_ark_asr=_USE_ARK_ASR,
        use_ark_tts=_USE_ARK_TTS,
        use_llm_streaming=_USE_LLM_STREAMING,
        enable_full_duplex=_ENABLE_FULL_DUPLEX,
        use_soulx_turn_taking=_USE_SOULX_TURN_TAKING,
        soulx_turn_url=_SOULX_TURN_URL,
        soulx_timeout_s=_SOULX_TIMEOUT_S,
        soulx_retry_interval_s=_SOULX_RETRY_INTERVAL_S,
        soulx_minimum_utterance_rms=_SOULX_MIN_UTTERANCE_RMS,
        soulx_barge_in_minimum_chunk_rms=(
            _SOULX_BARGE_IN_MIN_CHUNK_RMS
        ),
        soulx_pre_roll_s=_SOULX_PRE_ROLL_S,
        vad_pre_end_arm_window_s=_VAD_PRE_END_ARM_WINDOW_S,
        vad_minimum_post_tts_chunks=_VAD_MIN_POST_TTS_CHUNKS,
        interrupt_trigger_probability=_FULL_DUPLEX_TRIGGER_PROB,
        interrupt_minimum_duration=_FULL_DUPLEX_MIN_TRIGGER_DURATION,
        interrupt_stop_duration=_FULL_DUPLEX_STOP_DURATION,
        interrupt_minimum_rms=_FULL_DUPLEX_MIN_RMS,
        interrupt_minimum_consecutive_chunks=(
            _FULL_DUPLEX_MIN_CONSECUTIVE_CHUNKS
        ),
        interrupt_complete_silence_s=(
            _FULL_DUPLEX_COMPLETE_SILENCE_S
        ),
        interrupt_minimum_complete_audio_s=(
            _FULL_DUPLEX_MIN_COMPLETE_AUDIO_S
        ),
        answer_completion_observation_window_s=(
            _ANSWER_COMPLETION_OBSERVATION_WINDOW_S
        ),
        enable_realtime_companion=_ENABLE_REALTIME_COMPANION,
        realtime_emotion_interval_s=_REALTIME_EMOTION_INTERVAL_S,
        realtime_emotion_window_s=_REALTIME_EMOTION_WINDOW_S,
        memory_timeout_s=_MEMORY_RETRIEVAL_TIMEOUT_S,
        memory_prefetch_stability_s=_MEMORY_PREFETCH_STABILITY_S,
        memory_prefetch_min_interval_s=_MEMORY_PREFETCH_MIN_INTERVAL_S,
    ),
)
AUTH_CONTROLLER = AuthController(
    AUTH,
    static_dir=_BASE_DIR / "static",
    legacy_system_url=_LEGACY_SYSTEM_URL,
)
PUBLIC_API_CONTROLLER = PublicApiController(
    key_service=PublicApiKeyService(
        key_file=_BASE_DIR / "data" / ".public_api_key",
    ),
    session_store=PublicApiSessionStore(),
    context_registry=load_context_registry(),
    agent_factory=lambda: MODEL_RUNTIME.create_agent(
        mode="cognitive_screening"
    ),
    service_ready=lambda: MODEL_RUNTIME.ready,
    create_session=lambda session_id, **kwargs: db_create_session(
        session_id,
        **kwargs,
    ),
    static_dir=_BASE_DIR / "static",
    memory=EMOTION_MEMORY,
)
APPLICATION_HTTP_CONTROLLER = ApplicationHttpController(
    auth=AUTH,
    logs=LOG_BROKER,
    static_dir=_BASE_DIR / "static",
    voice_calls_dir=Path(
        os.getenv("VOICE_CALLS_DIR", "data/voice_calls")
    ),
    get_agent=lambda: MODEL_RUNTIME.agent,
)
MEMORY_API_CONTROLLER = MemoryApiController(
    EMOTION_MEMORY,
    auth=AUTH,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
AUTH_CONTROLLER.install(app)
PUBLIC_API_CONTROLLER.install(app)
APPLICATION_HTTP_CONTROLLER.install(app)
MEMORY_API_CONTROLLER.install(app)
app.mount(
    "/static",
    StaticFiles(directory=str(_BASE_DIR / "static")),
    name="static",
)
if _PIXI_VIEWER_DIR.is_dir():
    app.mount(
        "/pixi-viewer",
        StaticFiles(directory=str(_PIXI_VIEWER_DIR), html=True),
        name="pixi_viewer",
    )

try:
    init_db()
    AUTH.initialize_bootstrap_admin()
except Exception as db_init_error:
    print(f"[DB] ⚠️ 数据库初始化失败: {db_init_error}")


_WEBRTC_ICE_SERVERS = _load_ice_servers_from_env()
_WEBRTC_PEERS: set[Any] = set()
_HYBRID_MEDIA = HybridMediaRegistry()
_STARTUP_STATUS = {
    "ready": False,
    "memory_worker": False,
    "agent": False,
    "tts": False,
    "emotion": os.getenv("USE_MODELSCOPE", "false").lower()
    not in {"1", "true", "yes", "on"},
}
_STARTUP_TIMING = {
    "launch_started_at": os.getenv("LMCA_LAUNCH_STARTED_AT"),
    "started_at": None,
    "completed_at": None,
    "duration_s": None,
    "launch_to_ready_s": None,
    "stages": {},
}


def _utc_timestamp(epoch: float | None = None) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() if epoch is None else epoch),
    )


def _record_startup_stage(name: str, started: float) -> None:
    duration = round(time.perf_counter() - started, 3)
    _STARTUP_TIMING["stages"][name] = duration
    print(f"[启动耗时] {name}={duration:.3f}s")


@app.on_event("startup")
async def startup_event() -> None:
    startup_started = time.perf_counter()
    _STARTUP_TIMING.update(
        {
            "started_at": _utc_timestamp(),
            "completed_at": None,
            "duration_s": None,
            "launch_to_ready_s": None,
            "stages": {},
        }
    )
    print(f"[启动计时] 主服务启动阶段开始: {_STARTUP_TIMING['started_at']}")
    try:
        stage_started = time.perf_counter()
        if _ENABLE_LONG_TERM_MEMORY:
            EMOTION_MEMORY.start_memory_worker()
            _STARTUP_STATUS["memory_worker"] = True
        _record_startup_stage("memory_worker", stage_started)

        stage_started = time.perf_counter()
        MODEL_RUNTIME.initialize()
        _STARTUP_STATUS["agent"] = MODEL_RUNTIME.agent is not None
        _record_startup_stage("runtime_initialize", stage_started)

        stage_started = time.perf_counter()
        await MODEL_RUNTIME.prewarm_tts()
        _STARTUP_STATUS["tts"] = MODEL_RUNTIME.tts is not None
        _record_startup_stage("tts_prewarm", stage_started)

        stage_started = time.perf_counter()
        await MODEL_RUNTIME.prewarm_llm()
        _record_startup_stage("llm_prewarm", stage_started)

        stage_started = time.perf_counter()
        if os.getenv("USE_MODELSCOPE", "false").lower() in {
            "1", "true", "yes", "on"
        }:
            try:
                if _ENABLE_EMOTION:
                    emotion_ready = await asyncio.to_thread(
                        get_audio_emotion_classifier().prewarm
                    )
                else:
                    emotion_ready = False

                _STARTUP_STATUS["emotion"] = bool(emotion_ready)
                print(
                    "[初始化] "
                    + (
                        "✅ Emotion2Vec+ 预热完成"
                        if emotion_ready
                        else "⚠️ Emotion2Vec+ 预热失败（回退文本情绪）"
                    )
                )
            except Exception as exc:
                _STARTUP_STATUS["emotion"] = False
                print(
                    "[初始化] ⚠️ Emotion2Vec+ 预热异常（回退文本情绪）: "
                    f"{type(exc).__name__}"
                )
        _record_startup_stage("emotion_prewarm", stage_started)
        _STARTUP_STATUS["ready"] = all(
            (
                _STARTUP_STATUS["agent"],
                _STARTUP_STATUS["tts"],
                _STARTUP_STATUS["emotion"] if _ENABLE_EMOTION else True,
                _STARTUP_STATUS["memory_worker"] if _ENABLE_LONG_TERM_MEMORY else True,
            )
        )
    finally:
        completed_at = time.time()
        _STARTUP_TIMING["completed_at"] = _utc_timestamp(completed_at)
        _STARTUP_TIMING["duration_s"] = round(
            time.perf_counter() - startup_started,
            3,
        )
        launch_started = os.getenv("LMCA_LAUNCH_STARTED_EPOCH")
        try:
            _STARTUP_TIMING["launch_to_ready_s"] = round(
                completed_at - float(launch_started),
                3,
            )
        except (TypeError, ValueError):
            _STARTUP_TIMING["launch_to_ready_s"] = None
        print(
            "[启动耗时] 主服务启动及预热总耗时="
            f"{_STARTUP_TIMING['duration_s']:.3f}s"
        )
        if _STARTUP_TIMING["launch_to_ready_s"] is not None:
            print(
                "[启动耗时] 从启动脚本开始到主服务就绪="
                f"{_STARTUP_TIMING['launch_to_ready_s']:.3f}s"
            )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    _STARTUP_STATUS["ready"] = False
    EMOTION_MEMORY.stop_memory_worker()
    await asyncio.gather(
        *[
            transport.close()
            for transport in _HYBRID_MEDIA.transports()
        ],
        return_exceptions=True,
    )
    await asyncio.gather(
        *[peer.close() for peer in list(_WEBRTC_PEERS)],
        return_exceptions=True,
    )
    _WEBRTC_PEERS.clear()


@app.get("/health")
async def health() -> dict:
    turn_health = (
        await _SOULX_HEALTH.check()
        if _USE_SOULX_TURN_TAKING
        else {"available": True, "error": None}
    )
    ready = bool(_STARTUP_STATUS["ready"] and turn_health["available"])
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "webrtc": _AIORTC_IMPORT_ERROR is None,
        "turn_taking": (
            "soulx" if _USE_SOULX_TURN_TAKING and turn_health["available"] else "local"
        ),
        "configured_turn_taking": (
            "soulx" if _USE_SOULX_TURN_TAKING else "local"
        ),
        "turn_taking_health": turn_health,
        "soulx_turn_url": (
            _SOULX_TURN_URL if _USE_SOULX_TURN_TAKING else None
        ),
        "memory": {
            "backend": "sqlite",
            "long_term_enabled": _ENABLE_LONG_TERM_MEMORY,
            "worker_ready": _STARTUP_STATUS["memory_worker"],
            "mirror_configured": bool(
                os.getenv("MEMOBASE_PROJECT_URL")
                and os.getenv("MEMOBASE_API_KEY")
            ),
        },
        "startup": {
            "ready": _STARTUP_STATUS["ready"],
            **_STARTUP_TIMING,
            "prewarm": {
                key: _STARTUP_STATUS[key]
                for key in ("agent", "tts", "emotion")
            },
        },
        "release_flags": {
            "default_session_mode": _DEFAULT_SESSION_MODE,
            "cognitive_screening": _ENABLE_COGNITIVE_SCREENING,
            "turn_insight": _ENABLE_TURN_INSIGHT,
            "long_term_memory": _ENABLE_LONG_TERM_MEMORY,
            "long_term_memory_writes": _ENABLE_LONG_TERM_MEMORY_WRITES,
            "emotion": _ENABLE_EMOTION,
            "patient_memory_delete": _ENABLE_PATIENT_MEMORY_DELETE,
            "demo_data": _ENABLE_DEMO_DATA,
        },
    }


def _tts_auth_error(request: Request) -> JSONResponse | None:
    if AUTH.current_user(request):
        return None
    return JSONResponse({"success": False, "error": "UNAUTHORIZED"}, status_code=401)


def _current_tts_voice() -> str:
    voice_id = str(getattr(getattr(MODEL_RUNTIME, "tts", None), "voice", "") or "").strip()
    if voice_id in _TTS_VOICE_BY_ID:
        return voice_id
    configured = str(os.getenv("VOLC_TTS_VOICE", "") or "").strip()
    return configured if configured in _TTS_VOICE_BY_ID else _TTS_VOICES[0]["id"]


def _save_tts_voice(voice_id: str) -> None:
    _TTS_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _TTS_SETTINGS_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"voice_id": voice_id}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(_TTS_SETTINGS_PATH)


def _tts_preview_path(voice_id: str) -> Path:
    return _TTS_PREVIEW_DIR / f"{voice_id}.wav"


async def _generate_tts_preview(voice_id: str, output_path: Path) -> None:
    tts = getattr(MODEL_RUNTIME, "tts", None)
    stream = getattr(tts, "text_to_speech_streaming", None)
    if tts is None or not callable(stream):
        raise RuntimeError("TTS 当前不可用")
    chunks: list[np.ndarray] = []
    async for chunk in stream(_TTS_PREVIEW_TEXT, emotion="neutral", voice=voice_id):
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
    if not chunks:
        raise RuntimeError("TTS 未返回音频")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.wav")
    sf.write(
        temporary_path,
        np.concatenate(chunks),
        24000,
        format="WAV",
        subtype="PCM_16",
    )
    temporary_path.replace(output_path)


@app.get("/api/tts/voices")
async def tts_voices(request: Request) -> JSONResponse:
    auth_error = _tts_auth_error(request)
    if auth_error:
        return auth_error
    return JSONResponse(
        {
            "voices": [
                {**item, "preview_url": f"/api/tts/preview/{item['id']}"}
                for item in _TTS_VOICES
            ],
            "current_voice": _current_tts_voice(),
            "preview_text": _TTS_PREVIEW_TEXT,
        }
    )


@app.post("/api/tts/voice")
async def set_tts_voice(request: Request) -> JSONResponse:
    auth_error = _tts_auth_error(request)
    if auth_error:
        return auth_error
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    voice_id = str(payload.get("voice_id") or payload.get("voiceId") or "").strip()
    if voice_id not in _TTS_VOICE_BY_ID:
        return JSONResponse({"success": False, "error": "不支持的音色"}, status_code=400)
    tts = getattr(MODEL_RUNTIME, "tts", None)
    set_voice = getattr(tts, "set_voice", None)
    if not _USE_ARK_TTS or not callable(set_voice):
        return JSONResponse({"success": False, "error": "当前 TTS 不支持切换预置音色"}, status_code=503)
    try:
        set_voice(voice_id)
        _save_tts_voice(voice_id)
    except Exception as exc:
        print(f"[TTS] 音色切换失败: {type(exc).__name__}")
        return JSONResponse({"success": False, "error": "音色切换失败"}, status_code=503)
    return JSONResponse({"success": True, "voice_id": voice_id})


@app.get("/api/tts/preview/{voice_id}")
async def tts_preview(voice_id: str, request: Request):
    auth_error = _tts_auth_error(request)
    if auth_error:
        return auth_error
    if voice_id not in _TTS_VOICE_BY_ID:
        return JSONResponse({"success": False, "error": "不支持的音色"}, status_code=404)
    output_path = _tts_preview_path(voice_id)
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        async with _TTS_PREVIEW_LOCK:
            if not output_path.is_file() or output_path.stat().st_size <= 44:
                try:
                    await _generate_tts_preview(voice_id, output_path)
                except Exception as exc:
                    print(f"[TTS] 试听生成失败 voice={voice_id}: {type(exc).__name__}")
                    return JSONResponse({"success": False, "error": "试听音频生成失败"}, status_code=503)
    return FileResponse(
        output_path,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/webrtc/ice-config")
async def webrtc_ice_config(request: Request) -> JSONResponse:
    if not AUTH.current_user(request):
        return JSONResponse(
            {"success": False, "error": "UNAUTHORIZED"},
            status_code=401,
        )
    return JSONResponse({"iceServers": _WEBRTC_ICE_SERVERS})


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request) -> JSONResponse:
    if not AUTH.current_user(request):
        return JSONResponse(
            {"success": False, "error": "UNAUTHORIZED"},
            status_code=401,
        )
    if RTCPeerConnection is None or RTCSessionDescription is None:
        return JSONResponse(
            {
                "success": False,
                "error": "WEBRTC_UNAVAILABLE",
                "detail": str(_AIORTC_IMPORT_ERROR),
            },
            status_code=503,
        )

    try:
        payload = await request.json()
        offer_sdp = payload["sdp"]
        offer_type = payload.get("type", "offer")
        raw_media_token = payload.get("media_token")
        media_token = _HYBRID_MEDIA.normalize_token(
            raw_media_token
        )
        if raw_media_token is not None and not media_token:
            raise ValueError("invalid hybrid media token")
    except Exception:
        return JSONResponse(
            {"success": False, "error": "INVALID_SDP_OFFER"},
            status_code=400,
        )

    debug_dump_enabled = (
        os.getenv("WEBRTC_MIC_DEBUG_DUMP", "true").lower()
        in {"1", "true", "yes"}
    )
    media_transport = None
    if media_token:
        media_transport = _HYBRID_MEDIA.get(media_token)
        if media_transport is None or media_transport.closed:
            return JSONResponse(
                {
                    "success": False,
                    "error": "HYBRID_CONTROL_NOT_READY",
                    "detail": "WebSocket control channel is not registered",
                },
                status_code=409,
            )
        request_session_token = request.cookies.get(AUTH.cookie_name, "")
        control_session_token = media_transport.cookies.get(
            AUTH.cookie_name,
            "",
        )
        if not request_session_token or (
            request_session_token != control_session_token
        ):
            return JSONResponse(
                {"success": False, "error": "HYBRID_BIND_FORBIDDEN"},
                status_code=403,
            )
        old_media_session = media_transport.media_session
        if old_media_session is not None:
            await old_media_session.close("replaced by new media peer")
        peer_session = WebRTCPeerSession.create_media_only(
            peer_id=uuid.uuid4().hex[:10],
            ice_servers=_WEBRTC_ICE_SERVERS,
            media_transport=media_transport,
            peer_registry=_WEBRTC_PEERS,
            audio_writer=sf.write,
            debug_dump_enabled=debug_dump_enabled,
        )
    else:
        # Backward-compatible path for older pages which still use DataChannel
        # as the complete voice transport.
        peer_session = WebRTCPeerSession.create(
            cookies=request.cookies,
            peer_id=uuid.uuid4().hex[:10],
            ice_servers=_WEBRTC_ICE_SERVERS,
            session_runner=VOICE_APPLICATION.handle,
            peer_registry=_WEBRTC_PEERS,
            audio_writer=sf.write,
            debug_dump_enabled=debug_dump_enabled,
        )
    try:
        answer = await peer_session.negotiate(offer_sdp, offer_type)
        if media_transport is not None:
            answer["media_only"] = True
            answer["control_transport"] = "websocket"
        return JSONResponse(answer)
    except Exception as exc:
        await peer_session.close("offer error")
        print(f"[WebRTC] SDP 协商失败: {exc}")
        return JSONResponse(
            {
                "success": False,
                "error": "SDP_NEGOTIATION_FAILED",
                "detail": str(exc),
            },
            status_code=500,
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    media_token = _HYBRID_MEDIA.normalize_token(
        websocket.query_params.get("media_token")
    )
    transport = websocket
    if media_token:
        transport = HybridVoiceTransport(
            websocket,
            media_token=media_token,
            registry=_HYBRID_MEDIA,
        )
    try:
        await VOICE_APPLICATION.handle(transport)
    finally:
        if isinstance(transport, HybridVoiceTransport):
            await transport.close()


if __name__ == "__main__":
    import uvicorn

    ssl_certfile = os.getenv("VOICE_SSL_CERTFILE") or str(_BASE_DIR / "certs" / "voice_server.crt")
    ssl_keyfile = os.getenv("VOICE_SSL_KEYFILE") or str(_BASE_DIR / "certs" / "voice_server.key")
    if not Path(ssl_certfile).is_file() or not Path(ssl_keyfile).is_file():
        ssl_certfile = ssl_keyfile = None
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("VOICE_PORT", "8426")),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
        timeout_keep_alive=120,
    )
