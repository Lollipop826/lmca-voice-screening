"""FastAPI entry point for the Alzheimer's voice-screening application.

This module is deliberately limited to configuration and dependency wiring.
Voice-session behavior lives in ``src.voice`` and HTTP behavior in ``src.web``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


_BASE_DIR = Path(__file__).parent
_IS_FROZEN = getattr(sys, "frozen", False)


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
    os.getenv("USE_SOULX_TURN_TAKING", "false").lower() == "true"
)
_SOULX_TURN_URL = os.getenv(
    "SOULX_TURN_URL",
    "ws://127.0.0.1:8000/turn",
).strip()
_SOULX_TIMEOUT_S = float(os.getenv("SOULX_TIMEOUT_S", "3.0"))
_SOULX_RETRY_INTERVAL_S = float(
    os.getenv("SOULX_RETRY_INTERVAL_S", "5.0")
)
_SOULX_MIN_UTTERANCE_RMS = float(
    os.getenv("SOULX_MIN_UTTERANCE_RMS", "0.008")
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
    float(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_S", "3.0")),
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
EMOTION_MEMORY = EmotionMemobase()

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
        soulx_pre_roll_s=_SOULX_PRE_ROLL_S,
        vad_pre_end_arm_window_s=_VAD_PRE_END_ARM_WINDOW_S,
        vad_minimum_post_tts_chunks=_VAD_MIN_POST_TTS_CHUNKS,
        interrupt_trigger_probability=_FULL_DUPLEX_TRIGGER_PROB,
        interrupt_minimum_duration=_FULL_DUPLEX_MIN_TRIGGER_DURATION,
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

try:
    init_db()
    AUTH.initialize_bootstrap_admin()
except Exception as db_init_error:
    print(f"[DB] ⚠️ 数据库初始化失败: {db_init_error}")


_WEBRTC_ICE_SERVERS = _load_ice_servers_from_env()
_WEBRTC_PEERS: set[Any] = set()
_HYBRID_MEDIA = HybridMediaRegistry()


@app.on_event("startup")
async def startup_event() -> None:
    MODEL_RUNTIME.initialize()
    await MODEL_RUNTIME.prewarm_tts()
    if os.getenv("USE_MODELSCOPE", "false").lower() in {"1", "true", "yes", "on"}:
        asyncio.create_task(asyncio.to_thread(get_audio_emotion_classifier().prewarm))


@app.on_event("shutdown")
async def shutdown_event() -> None:
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
    return {
        "status": "ok",
        "webrtc": _AIORTC_IMPORT_ERROR is None,
        "turn_taking": (
            "soulx" if _USE_SOULX_TURN_TAKING else "local"
        ),
        "soulx_turn_url": (
            _SOULX_TURN_URL if _USE_SOULX_TURN_TAKING else None
        ),
        "release_flags": {
            "default_session_mode": _DEFAULT_SESSION_MODE,
            "cognitive_screening": _ENABLE_COGNITIVE_SCREENING,
            "turn_insight": _ENABLE_TURN_INSIGHT,
            "long_term_memory": _ENABLE_LONG_TERM_MEMORY,
            "long_term_memory_writes": _ENABLE_LONG_TERM_MEMORY_WRITES,
            "patient_memory_delete": _ENABLE_PATIENT_MEMORY_DELETE,
            "demo_data": _ENABLE_DEMO_DATA,
        },
    }


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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8502,
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
        timeout_keep_alive=120,
    )
