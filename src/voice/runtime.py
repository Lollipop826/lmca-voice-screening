from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import soundfile as sf

from src.voice.interrupt_intent import is_backchannel_text


def clean_for_tts(text: str) -> str:
    """Remove common Markdown markers before speech synthesis."""
    if not text:
        return text
    substitutions = (
        (r"\*\*([^*]+)\*\*", r"\1"),
        (r"\*([^*]+)\*", r"\1"),
        (r"__([^_]+)__", r"\1"),
        (r"_([^_]+)_", r"\1"),
        (r"~~([^~]+)~~", r"\1"),
        (r"`([^`]+)`", r"\1"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    return text


def audio_signal_stats(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> dict:
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "duration_s": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "dbfs": -120.0,
            "nonzero_ratio": 0.0,
        }
    rms = float(np.sqrt(np.mean(np.square(array))))
    peak = float(np.max(np.abs(array)))
    dbfs = 20.0 * np.log10(max(rms, 1e-12))
    return {
        "duration_s": float(array.size) / float(sample_rate or 16000),
        "rms": rms,
        "peak": peak,
        "dbfs": float(max(dbfs, -120.0)),
        "nonzero_ratio": (
            float(np.count_nonzero(np.abs(array) > 1e-5))
            / float(array.size)
        ),
    }


def audio_is_effectively_silent(stats: dict) -> bool:
    return (
        float(stats.get("duration_s") or 0.0) > 0.0
        and float(stats.get("rms") or 0.0) < 0.0005
        and float(stats.get("peak") or 0.0) < 0.005
    )


def decode_recorded_audio_blob(
    audio_bytes: bytes,
    mime_type: str = "",
) -> tuple[np.ndarray, int]:
    """Decode a browser MediaRecorder blob into mono 16 kHz float PCM."""
    suffix = (
        ".mp4"
        if "mp4" in mime_type or "aac" in mime_type
        else ".webm"
    )
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as source_file:
        source_path = source_file.name
        source_file.write(audio_bytes)
    wav_path = source_path + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                source_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        audio, sample_rate = sf.read(
            wav_path,
            dtype="float32",
            always_2d=False,
        )
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=1)
        return array.reshape(-1), int(sample_rate)
    finally:
        for path in (source_path, wav_path):
            try:
                os.remove(path)
            except OSError:
                pass


class SileroVADModel:
    """Silero VAD v5 ONNX wrapper."""

    _CONTEXT_SIZE = 64

    def __init__(
        self,
        model_path: str | None = None,
        *,
        session=None,
    ) -> None:
        self._session = session or ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(
            (1, self._CONTEXT_SIZE),
            dtype=np.float32,
        )

    def clone(self) -> "SileroVADModel":
        """Share immutable ONNX weights while isolating recurrent state."""
        return SileroVADModel(session=self._session)

    def __call__(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> float:
        samples = np.asarray(
            audio_chunk,
            dtype=np.float32,
        ).reshape(1, -1)
        samples = np.concatenate(
            [self._context, samples],
            axis=1,
        )
        output, self._state = self._session.run(
            None,
            {
                "input": samples,
                "sr": np.array(sample_rate, dtype=np.int64),
                "state": self._state,
            },
        )
        self._context = samples[:, -self._CONTEXT_SIZE :]
        return float(output.squeeze())

    def predict_stateless(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> float:
        samples = np.asarray(
            audio_chunk,
            dtype=np.float32,
        ).reshape(1, -1)
        samples = np.concatenate(
            [
                np.zeros(
                    (1, self._CONTEXT_SIZE),
                    dtype=np.float32,
                ),
                samples,
            ],
            axis=1,
        )
        output, _state = self._session.run(
            None,
            {
                "input": samples,
                "sr": np.array(sample_rate, dtype=np.int64),
                "state": np.zeros(
                    (2, 1, 128),
                    dtype=np.float32,
                ),
            },
        )
        return float(output.squeeze())

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(
            (1, self._CONTEXT_SIZE),
            dtype=np.float32,
        )


def load_silero_vad(*, logger=print) -> SileroVADModel:
    paths = [
        os.path.join(
            getattr(__import__("sys"), "_MEIPASS", ""),
            "silero_vad.onnx",
        ),
        str(
            Path(__file__).resolve().parents[2]
            / "models"
            / "silero_vad.onnx"
        ),
        "/app/models/silero_vad.onnx",
        os.path.expanduser(
            "~/.cache/torch/hub/snakers4_silero-vad_master/"
            "src/silero_vad/data/silero_vad.onnx"
        ),
    ]
    for model_path in paths:
        if not os.path.exists(model_path):
            continue
        try:
            logger(f"[VAD] 从 ONNX 缓存加载 Silero VAD: {model_path}")
            model = SileroVADModel(model_path)
            logger("[VAD] ✅ Silero VAD ONNX 加载完成（无 PyTorch）")
            return model
        except Exception as exc:
            logger(f"[VAD] ⚠️ 加载失败: {exc}")
    raise RuntimeError(f"找不到 silero_vad.onnx，检查路径: {paths}")


DEFAULT_VAD_MODEL = load_silero_vad()


class VADBuffer:
    """Connection-local VAD buffer backed by the shared ONNX model."""

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        vad_model=None,
        environ=None,
        logger=print,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_model = vad_model or DEFAULT_VAD_MODEL.clone()
        self.environ = environ if environ is not None else os.environ
        self._log = logger
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        try:
            self.max_silence_chunks = max(
                1,
                math.ceil(
                    float(self.environ.get("VAD_END_SILENCE_S", "1.2"))
                    * sample_rate
                    / 512
                ),
            )
        except (TypeError, ValueError, OverflowError):
            self.max_silence_chunks = math.ceil(1.2 * sample_rate / 512)
        self._speech_chunk_count = 0
        self.start_threshold = float(
            self.environ.get("VAD_START_THRESHOLD", "0.50")
        )
        self.weak_speech_threshold = float(
            self.environ.get("VAD_WEAK_SPEECH_THRESHOLD", "0.35")
        )
        self.minimum_rms = float(
            self.environ.get("VAD_MIN_RMS", "0.008")
        )
        self.minimum_speech_seconds = float(
            self.environ.get("VAD_MIN_SPEECH_SECONDS", "0.22")
        )
        pre_roll_chunks = max(
            1,
            int(self.environ.get("VAD_PRE_ROLL_CHUNKS", "6")),
        )
        self._pre_roll_buffer = deque(maxlen=pre_roll_chunks)
        self._weak_speech_run = 0
        self.last_drop_reason = None
        self.last_drop_duration_s = None
        self._last_frame_audio = None
        self._last_frame_probability = 0.0

    def add_chunk(self, audio_chunk):
        chunk_size = 512
        self.last_drop_reason = None
        self.last_drop_duration_s = None
        self._last_frame_audio = audio_chunk
        self._last_frame_probability = 0.0
        for offset in range(0, len(audio_chunk), chunk_size):
            chunk = audio_chunk[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                if self.is_speaking:
                    self.buffer.append(chunk)
                elif len(chunk) > 0:
                    self._pre_roll_buffer.append(chunk)
                continue

            probability = self.vad_model(chunk, self.sample_rate)
            # Reuse continuous inference for barge-in; resetting Silero for
            # every 32 ms frame makes sustained speech look like isolated noise.
            # For batched input require every full chunk to pass the gate.
            self._last_frame_probability = (
                probability if offset == 0
                else min(self._last_frame_probability, probability)
            )
            chunk_rms = self._chunk_rms(chunk)
            strong_speech = probability >= self.start_threshold
            weak_speech = (
                probability >= self.weak_speech_threshold
                and chunk_rms >= self.minimum_rms
            )
            self._weak_speech_run = (
                self._weak_speech_run + 1 if weak_speech else 0
            )

            if not self.is_speaking:
                self._pre_roll_buffer.append(chunk)
                if strong_speech or self._weak_speech_run >= 2:
                    self._log(
                        "[VAD] 🎤 正在说话... "
                        f"(prob={probability:.2f}, rms={chunk_rms:.4f})"
                    )
                    self.is_speaking = True
                    self.buffer = list(self._pre_roll_buffer)
                    self.silence_chunks = 0
                    self._speech_chunk_count = max(
                        1,
                        self._weak_speech_run,
                    )
                    continue

            if self.is_speaking and (strong_speech or weak_speech):
                self.silence_chunks = 0
                self._speech_chunk_count += 1
                self.buffer.append(chunk)
                continue

            if not self.is_speaking:
                continue
            self.buffer.append(chunk)
            self.silence_chunks += 1
            speech_duration = (
                self._speech_chunk_count
                * chunk_size
                / self.sample_rate
            )
            if self.silence_chunks < self.max_silence_chunks:
                continue
            if speech_duration < self.minimum_speech_seconds:
                self._log(
                    f"[VAD] ⚠️ 语音过短({speech_duration:.2f}s)，"
                    "可能是噪音，跳过"
                )
                self.last_drop_reason = "speech_too_short"
                self.last_drop_duration_s = speech_duration
                self.reset()
                return None
            self._log(
                f"[VAD] ⏹️ 说话结束 (语音{speech_duration:.1f}s, "
                f"静音阈值{self.max_silence_chunks}块)"
            )
            complete_audio = np.concatenate(self.buffer)
            self.reset()
            return complete_audio
        return None

    def consume_drop_feedback(self):
        result = (
            self.last_drop_reason,
            self.last_drop_duration_s,
        )
        self.last_drop_reason = None
        self.last_drop_duration_s = None
        return result

    def has_speech(self, audio_chunk) -> float:
        """Read the score from add_chunk without advancing/resetting Silero."""
        if len(audio_chunk) < 512 or audio_chunk is not self._last_frame_audio:
            return 0.0
        return self._last_frame_probability

    def reset(self) -> None:
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        self._speech_chunk_count = 0
        self._pre_roll_buffer.clear()
        self._weak_speech_run = 0
        self._last_frame_audio = None
        self._last_frame_probability = 0.0
        self.vad_model.reset_states()

    @staticmethod
    def _chunk_rms(chunk) -> float:
        if len(chunk) == 0:
            return 0.0
        array = np.asarray(chunk, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(array))))


class VoiceModelRuntime:
    """Own the process-wide ASR, reference Agent, TTS and speaker models."""

    def __init__(
        self,
        *,
        base_dir: Path,
        agent_factory: Callable[..., Any],
        agent_type: str,
        tts_factory: Callable[..., Any],
        use_ark_asr: bool,
        use_ark_tts: bool,
        use_speaker_verifier: bool,
        use_local_embedding: bool,
        logger=print,
        wellbeing_agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.agent_factory = agent_factory
        self.wellbeing_agent_factory = wellbeing_agent_factory
        self.agent_type = agent_type
        self.tts_factory = tts_factory
        self.use_ark_asr = bool(use_ark_asr)
        self.use_ark_tts = bool(use_ark_tts)
        self.use_speaker_verifier = bool(use_speaker_verifier)
        self.use_local_embedding = bool(use_local_embedding)
        self._log = logger
        self.asr_model = None
        self.agent = None
        self.tts = None
        self.speaker_verifier = None

    @property
    def ready(self) -> bool:
        return self.agent is not None and self.tts is not None

    def create_agent(self, mode: str = "wellbeing"):
        if self.wellbeing_agent_factory is None:
            # Keep the narrow legacy factory contract used by integrations and
            # tests that have not opted into the mode-aware runtime yet.
            return self.agent_factory(use_local=False)
        from src.agents.mode_aware_agent import ModeAwareAgent

        return ModeAwareAgent(
            cognitive_factory=self.agent_factory,
            wellbeing_factory=self.wellbeing_agent_factory,
            mode=mode,
        )

    def initialize(self) -> None:
        self._initialize_asr()
        self._initialize_embedding()
        self._log(f"[初始化] 加载 Agent ({self.agent_type})...")
        self.agent = self.create_agent()
        self._log(f"[初始化] ✅ Agent 类型: {self.agent_type}")
        self._initialize_tts()
        self._initialize_speaker_verifier()
        self._initialize_location()
        self._log(
            "[初始化] 💡 LLM 使用 API 模式"
            "（无需加载本地 7B-GPTQ，节省 ~6GB 显存）"
        )
        self._log("[初始化] ✅ 所有本地模型加载+预热完成")

    async def prewarm_tts(self) -> None:
        if not self.use_ark_tts or self.tts is None:
            return
        try:
            prewarm = getattr(self.tts, "prewarm", None)
            if callable(prewarm):
                await prewarm()
            else:
                await self.tts._ensure_connected()
            self._log("[初始化] 🔥 ArkTTS WebSocket 连接预热完成")
        except Exception as exc:
            self._log(
                "[初始化] ⚠️ ArkTTS 预热失败（不影响启动）: "
                f"{exc}"
            )

    async def prewarm_llm(self) -> None:
        """Create the default companion LLM and complete one tiny probe call."""
        if self.agent is None:
            return

        def probe() -> None:
            agent = self.agent
            get_agent = getattr(agent, "_get_agent", None)
            if callable(get_agent):
                agent = get_agent()
            get_llm = getattr(agent, "_get_llm", None)
            llm = get_llm() if callable(get_llm) else getattr(agent, "llm", None)
            invoke = getattr(llm, "invoke", None)
            if not callable(invoke):
                return
            invoke([{"role": "user", "content": "只回复一个字：好"}])

        self._log("[初始化] 🔥 预热陪伴 Agent/LLM...")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(probe),
                timeout=max(1.0, float(os.getenv("LLM_PREWARM_TIMEOUT_S", "20"))),
            )
            self._log("[初始化] ✅ 陪伴 Agent/LLM 预热完成")
        except asyncio.TimeoutError:
            self._log("[初始化] ⚠️ 陪伴 Agent/LLM 预热超时（不影响启动）")
        except Exception as exc:
            self._log(
                "[初始化] ⚠️ 陪伴 Agent/LLM 预热失败（不影响启动）: "
                f"{type(exc).__name__}"
            )

    def _initialize_asr(self) -> None:
        if self.use_ark_asr:
            self._log(
                "[初始化] 🔥 ASR 使用火山引擎 BigASR API"
                "（无需加载本地模型）"
            )
            return
        self._log(
            "[初始化] 加载 SenseVoice-Small "
            "(多语言+情绪识别+事件检测)..."
        )
        from funasr import AutoModel

        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=(
                "cuda:0"
                if __import__("torch").cuda.is_available()
                else "cpu"
            ),
            disable_pbar=True,
            disable_log=False,
            disable_update=True,
        )
        self._log("[初始化] ✅ SenseVoice-Small 加载完成")
        self._log("[初始化] 🔥 预热 SenseVoice ASR...")
        self.asr_model.generate(
            input=np.zeros(16000, dtype=np.float32),
            cache={},
            language="auto",
            use_itn=True,
        )
        self._log("[初始化] ✅ SenseVoice ASR 预热完成")

    def _initialize_embedding(self) -> None:
        if not self.use_local_embedding:
            self._log(
                "[初始化] ⏭️ 跳过 BGE-M3"
                "（USE_LOCAL_EMBEDDING=false，知识检索功能不可用）"
            )
            return
        self._log("[初始化] 加载 Embedding 模型池 (BGE-M3)...")
        from src.tools.retrieval.embedding_pool import get_embedding_pool

        get_embedding_pool()

    def _initialize_tts(self) -> None:
        self._log("[初始化] 加载 TTS...")
        if self.use_ark_tts:
            self.tts = self.tts_factory()
            self._log(
                "[初始化] 🔥 TTS 使用火山引擎 API"
                "（无需加载本地模型）"
            )
            return
        model_dir = self.base_dir / "models" / "zipvoice_distill"
        reference_dir = self.base_dir / "static" / "audio"
        reference_audio = reference_dir / "参考音频.wav"
        reference_text_path = reference_dir / "参考音频.txt"
        reference_text = (
            reference_text_path.read_text(encoding="utf-8").strip()
            if reference_text_path.exists()
            else None
        )
        self.tts = self.tts_factory(
            model_dir=str(model_dir),
            prompt_wav=str(reference_audio),
            prompt_text=reference_text,
        )

    def _initialize_speaker_verifier(self) -> None:
        if not self.use_speaker_verifier:
            self._log(
                "[初始化] ⏭️ 跳过声纹验证器"
                "（USE_SPEAKER_VERIFIER=false，可按需懒加载）"
            )
            return
        self._log("[初始化] 加载声纹验证器 (ONNX Runtime)...")
        try:
            from src.tools.voice.speaker_verification import (
                get_speaker_verifier,
            )

            self.speaker_verifier = get_speaker_verifier()
            self._log("[初始化] ✅ 声纹验证器加载完成")
        except Exception as exc:
            self._log(f"[初始化] ⚠️ 声纹验证器加载失败: {type(exc).__name__}")
            self.speaker_verifier = None

    def _initialize_location(self) -> None:
        from src.utils.location_service import (
            get_deployment_location,
            get_realtime_context,
        )

        location = get_deployment_location()
        self._log(
            "[初始化] 📍 当前位置: "
            f"{location.get('province', '未知')} "
            f"{location.get('city', '未知')}"
        )
        context = get_realtime_context(fetch_weather=False)
        weather = context["weather"]
        self._log(
            "[初始化] 🌤️ 天气占位: "
            f"{weather.get('weather', '未知')} "
            f"({weather.get('source', '')})"
        )


class VoiceRecognitionService:
    """Own fast ASR and LLM-based interruption/completion judgements."""

    _SHORT_PATTERNS = (
        re.compile(r"^[0-9]+$"),
        re.compile(r"^[零〇一二三四五六七八九十百千万两]+$"),
        re.compile(r"^[A-Da-d甲乙丙丁]$"),
    )
    _SHORT_ANSWERS = {
        "是",
        "不是",
        "有",
        "没有",
        "会",
        "不会",
        "能",
        "不能",
        "记得",
        "不记得",
        "知道",
        "不知道",
        "左",
        "右",
        "上",
        "下",
        "前",
        "后",
        "看到了",
        "没看到",
        "做完了",
        "完成了",
        "做好了",
    }

    def __init__(
        self,
        models: VoiceModelRuntime,
        *,
        logger=print,
    ) -> None:
        self.models = models
        self._log = logger

    @staticmethod
    def parse_sensevoice_result(result) -> dict:
        parsed = {
            "text": "",
            "emotion": "neutral",
            "language": "zh",
            "event": "Speech",
        }
        if not result:
            return parsed
        raw_text = result[0].get("text", "")
        from funasr.utils.postprocess_utils import (
            rich_transcription_postprocess,
        )

        parsed["text"] = rich_transcription_postprocess(raw_text)
        for tag, value in {
            "<|HAPPY|>": "happy",
            "<|SAD|>": "sad",
            "<|ANGRY|>": "angry",
            "<|FEARFUL|>": "fearful",
            "<|DISGUSTED|>": "disgusted",
            "<|SURPRISED|>": "surprised",
        }.items():
            if tag in raw_text:
                parsed["emotion"] = value
                break
        for tag, value in {
            "<|zh|>": "zh",
            "<|en|>": "en",
            "<|yue|>": "yue",
            "<|ja|>": "ja",
            "<|ko|>": "ko",
        }.items():
            if tag in raw_text:
                parsed["language"] = value
                break
        for tag, value in {
            "<|Applause|>": "Applause",
            "<|Laughter|>": "Laughter",
            "<|Cry|>": "Cry",
            "<|Cough|>": "Cough",
            "<|Sneeze|>": "Sneeze",
            "<|Breath|>": "Breath",
            "<|BGM|>": "BGM",
        }.items():
            if tag in raw_text:
                parsed["event"] = value
                break
        return parsed

    async def quick_asr(self, audio_data: np.ndarray) -> str:
        try:
            self._log(
                "[快速ASR] 开始识别，音频长度: "
                f"{len(audio_data) / 16000:.2f}秒"
            )
            started_at = time.time()
            if self.models.use_ark_asr:
                from src.tools.voice.ark_asr import (
                    ArkASRError,
                    ark_asr_recognize,
                )

                try:
                    parsed = await ark_asr_recognize(audio_data)
                except ArkASRError as exc:
                    self._log(f"[快速ASR] ⚠️ BigASR 不可用: {type(exc).__name__}")
                    return ""
            else:
                result = await asyncio.to_thread(
                    self.models.asr_model.generate,
                    input=audio_data,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
                parsed = self.parse_sensevoice_result(result)
            text = parsed["text"]
            if (
                parsed["emotion"] != "neutral"
                or parsed["event"] != "Speech"
            ):
                self._log(
                    f"[快速ASR] 情绪: {parsed['emotion']}, "
                    f"事件: {parsed['event']}"
                )
            self._log(
                f"[快速ASR] 识别完成: text_chars={len(text)} "
                f"(耗时 {time.time() - started_at:.2f}秒)"
            )
            return text
        except Exception as exc:
            self._log(f"[快速ASR] 错误: {type(exc).__name__}")
            return ""

    async def judge_interrupt_intent(self, text: str) -> str:
        normalized = self.normalize_interrupt_text(text)
        if is_backchannel_text(normalized):
            self._log(f"[语义判断] 应答词 text_chars={len(text)}")
            return "backchannel"
        corrections = {
            "不对",
            "不对不对",
            "不是",
            "不是不是",
            "说错了",
            "我说错了",
            "更正一下",
            "重新说",
            "改一下",
        }
        if any(normalized.startswith(item) for item in corrections):
            self._log(f"[语义判断] 纠正信号 text_chars={len(text)}")
            return "incomplete"
        interruptions = {
            "等一下",
            "等等",
            "停",
            "停一下",
            "等下",
            "慢着",
            "别说了",
            "停下",
            "打断一下",
        }
        if any(item in normalized for item in interruptions):
            self._log(f"[语义判断] 打断词 text_chars={len(text)}")
            return "complete"
        if self._looks_like_complete_short(normalized):
            self._log(f"[语义判断] 简短但有效的完整回答 text_chars={len(text)}")
            return "complete"
        if self._looks_like_complete_modal(normalized):
            self._log(f"[语义判断] 带语气词的完整句 text_chars={len(text)}")
            return "complete"
        if len(normalized) < 2:
            self._log(f"[语义判断] 太短 text_chars={len(text)}")
            return "backchannel"
        try:
            prompt = (
                "判断下面这句话的意图，只输出一个字母：\n"
                'B - 应答词（如"嗯"、"啊"、"对"等简短回应）\n'
                "C - 完整的问题或陈述\n"
                "I - 不完整的句子\n\n"
                f'句子："{normalized or text}"\n\n'
                "只输出B、C或I其中一个字母："
            )
            response = await asyncio.to_thread(
                self.models.agent.llm.invoke,
                prompt,
            )
            result = (
                response.content.strip().upper()
                if hasattr(response, "content")
                else str(response).strip().upper()
            )
            if "B" in result:
                return "backchannel"
            if "I" in result:
                return "incomplete"
            return "complete"
        except Exception as exc:
            self._log(
                f"[语义判断] LLM出错: {exc}，默认为完整"
            )
            return "complete"

    async def judge_answer_completion(
        self,
        question_text: str,
        answer_text: str,
    ) -> dict:
        answer = str(answer_text or "").strip()
        if not answer:
            return {
                "label": "uncertain",
                "reason": "empty_answer",
                "confidence": 0.0,
            }
        llm = self._answer_completion_llm()
        if llm is None:
            return {
                "label": "uncertain",
                "reason": "llm_unavailable",
                "confidence": 0.0,
            }
        prompt = self._answer_completion_prompt(
            question_text,
            answer,
        )
        try:
            response = await asyncio.to_thread(llm.invoke, prompt)
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            payload = self._parse_completion_payload(raw)
            label = str(
                payload.get("label", "uncertain") or "uncertain"
            ).strip().lower()
            accepted = {
                "incomplete",
                "likely_complete",
                "ask_repeat",
                "explicit_no_answer",
                "uncertain",
            }
            if label not in accepted:
                label = "uncertain"
            try:
                confidence = float(
                    payload.get("confidence", 0.0) or 0.0
                )
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(confidence, 1.0))
            reason = str(payload.get("reason") or "").strip()
            self._log(
                f"[作答判定] LLM结果 label={label}, "
                f"confidence={confidence:.2f}, "
                f"reason_chars={len(reason)}, answer_chars={len(answer)}"
            )
            return {
                "label": label,
                "reason": reason,
                "confidence": confidence,
            }
        except Exception as exc:
            self._log(f"[作答判定] LLM出错: {type(exc).__name__}")
            return {
                "label": "uncertain",
                "reason": f"llm_error:{exc}",
                "confidence": 0.0,
            }

    @staticmethod
    def normalize_interrupt_text(text: str) -> str:
        normalized = re.sub(r"\s+", "", (text or "").strip())
        return re.sub(
            r"[，。！？、,.!?]+(?=[了吧吗呢呀啊啦喽])",
            "",
            normalized,
        )

    @staticmethod
    def extract_latest_assistant_utterance(
        chat_history: list[dict],
    ) -> str:
        for item in reversed(chat_history or []):
            if item.get("role") != "assistant":
                continue
            content = str(item.get("content") or "").strip()
            if content:
                return content
        return ""

    def _answer_completion_llm(self):
        agent = self.models.agent
        if agent is not None and agent.use_local:
            from src.llm.model_pool import get_pooled_llm

            return get_pooled_llm(pool_key="small_classify")
        from src.llm.http_client_pool import get_chat_openai

        return get_chat_openai(
            temperature=0.1,
            max_tokens=96,
            timeout=8,
            max_retries=1,
            streaming=False,
            disable_thinking=True,
        )

    def _looks_like_complete_short(self, text: str) -> bool:
        normalized = re.sub(r"[，。！？、,.!?\s]+", "", text or "")
        return bool(
            normalized
            and (
                normalized in self._SHORT_ANSWERS
                or any(
                    pattern.fullmatch(normalized)
                    for pattern in self._SHORT_PATTERNS
                )
            )
        )

    def _looks_like_complete_modal(self, text: str) -> bool:
        normalized = re.sub(
            r"[，。！？、,.!?\s]+",
            "",
            self.normalize_interrupt_text(text),
        )
        return bool(
            len(normalized) >= 4
            and re.search(r"(了吧|对吧|是吧|吧|吗|呢)$", normalized)
            and re.search(
                r"(今年|现在|今天|年|月|号|日|星期|礼拜|"
                r"点|分|季|春|夏|秋|冬|是|有|在|叫|住|"
                r"来自|属于)",
                normalized,
            )
        )

    @staticmethod
    def _parse_completion_payload(raw_text: str) -> dict:
        raw = str(raw_text or "").strip()
        if not raw:
            return {}
        candidates = [raw]
        candidates.extend(
            re.findall(
                r"```(?:json)?\s*([\s\S]*?)```",
                raw,
                flags=re.IGNORECASE,
            )
        )
        candidates.extend(re.findall(r"\{[\s\S]*?\}", raw))
        seen = set()
        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        payload = {}
        label_match = re.search(
            "(likely_complete|explicit_no_answer|ask_repeat|"
            "incomplete|uncertain)",
            raw,
            flags=re.IGNORECASE,
        )
        if label_match:
            payload["label"] = label_match.group(1).lower()
        confidence_match = re.search(
            r"(?:confidence|置信度)[^0-9]*([01](?:\.\d+)?)",
            raw,
            flags=re.IGNORECASE,
        )
        if confidence_match:
            payload["confidence"] = confidence_match.group(1)
        reason_match = re.search(
            r"(?:reason|理由)\s*[:：]?\s*([^\n\r]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if reason_match:
            payload["reason"] = reason_match.group(1).strip()
        return payload

    @staticmethod
    def _answer_completion_prompt(
        question_text: str,
        answer: str,
    ) -> str:
        return f"""
你是阿尔茨海默病认知评估场景中的“作答完成性判定器”。
判断患者当前累计回答是否已经足够提交处理。

规则：
- 患者可能在字词之间长时间停顿，不要因为说得慢判为未完成。
- 语义完整、能独立回答问题时判 likely_complete。
- 请求系统重说判 ask_repeat。
- 明确表示不知道、想不起来或不会答判 explicit_no_answer。
- 明显停在半句或断裂表达判 incomplete。
- 只有上下文不足、确实无法判断时才用 uncertain。

当前问题：{question_text or '（未知）'}
当前累计作答：{answer}

只输出 JSON：
{{"label":"incomplete|likely_complete|ask_repeat|explicit_no_answer|uncertain","confidence":0.0,"reason":"一句简短中文原因"}}
""".strip()
