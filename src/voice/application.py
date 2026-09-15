from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from fastapi import WebSocketDisconnect

from src.db import database
from src.tools.voice.soulx_turn_taking import (
    SoulXAudioAccumulator,
    SoulXTurnTakingClient,
    SoulXUnavailable,
)
from src.tools.voice.ark_asr import ark_asr_streaming_supported

from .answer_completion import (
    AnswerCompletionController,
    AnswerCompletionState,
)
from .connection import VoiceConnectionController, VoiceConnectionIO
from .handlers import (
    AgentOutputPresenter,
    AnswerCompletionMessageHandler,
    ClientControlHandler,
    LiveAudioConfig,
    LiveAudioInputHandler,
    ManualAudioBlobProcessor,
    ManualAudioInputHandler,
    ManualInterruptHandler,
    SessionLifecycleHandler,
    SessionResumeHandler,
    SoulXAudioHandler,
    SpeechTurnConfig,
    SpeechTurnProcessor,
    SpeakerVerificationController,
    TextTurnHandler,
    VisionMessageHandler,
    VoiceFeedbackPresenter,
    VoiceInterruptionController,
    VoiceMessageRouter,
)
from .media import resample_audio
from .persistence import (
    VoiceAudioStore,
    VoiceDatasetManifestStore,
    VoiceHistoryStore,
)
from .realtime_companion import RealtimeCompanion, RealtimeCompanionConfig
from .runtime import (
    VADBuffer,
    VoiceModelRuntime,
    VoiceRecognitionService,
    audio_is_effectively_silent,
    audio_signal_stats,
    clean_for_tts,
    decode_recorded_audio_blob,
)
from .services import (
    AgentResumeStateService,
    PatientMemoryService,
    PatientProfileService,
    SpeechProcessingCoordinator,
    VoiceConnectionCleanup,
    VoiceSessionBootstrap,
    VoiceTTSStreamer,
)
from .session import VoiceSession


@dataclass(frozen=True)
class VoiceEndpointConfig:
    use_ark_asr: bool
    use_ark_tts: bool
    use_llm_streaming: bool
    enable_full_duplex: bool
    use_soulx_turn_taking: bool
    soulx_turn_url: str
    soulx_timeout_s: float
    soulx_retry_interval_s: float
    soulx_minimum_utterance_rms: float
    soulx_barge_in_minimum_chunk_rms: float
    soulx_pre_roll_s: float
    vad_pre_end_arm_window_s: float
    vad_minimum_post_tts_chunks: int
    interrupt_trigger_probability: float
    interrupt_minimum_duration: float
    interrupt_minimum_rms: float
    interrupt_minimum_consecutive_chunks: int
    interrupt_complete_silence_s: float
    interrupt_minimum_complete_audio_s: float
    answer_completion_observation_window_s: float
    vision_lock_timeout_s: float = 30.0
    enable_realtime_companion: bool = True
    realtime_emotion_interval_s: float = 0.8
    realtime_emotion_window_s: float = 3.0
    memory_timeout_s: float = 0.25
    memory_prefetch_stability_s: float = 0.35
    memory_prefetch_min_interval_s: float = 0.5
    interrupt_stop_duration: float = 0.20


class VoiceEndpointApplication:
    """Composition root for one WebSocket or WebRTC voice connection."""

    def __init__(
        self,
        *,
        auth,
        models: VoiceModelRuntime,
        recognition: VoiceRecognitionService,
        patient_memory_service: PatientMemoryService,
        config: VoiceEndpointConfig,
        repository=database,
        render_manager=None,
        logger=print,
    ) -> None:
        self.auth = auth
        self.models = models
        self.recognition = recognition
        self.patient_memory_service = patient_memory_service
        self.config = config
        self.repository = repository
        self.render_manager = render_manager
        self._log = logger

    async def handle(self, transport) -> None:
        connection = VoiceConnectionIO(transport)
        authenticated_user = self.auth.user_from_session_token(
            connection.cookies.get(self.auth.cookie_name, "")
        )
        if not authenticated_user:
            await connection.close(code=1008)
            return

        await connection.accept()
        try:
            session_agent = await asyncio.to_thread(
                self.models.create_agent
            )
        except Exception as exc:
            self._log(f"[连接] ❌ 创建独立 Agent 失败: {type(exc).__name__}")
            await connection.close(code=1011)
            return

        session = VoiceSession(
            connection=connection,
            agent=session_agent,
            owner_username=str(
                authenticated_user.get("username") or ""
            ).strip(),
        )
        connection.bind_session(session)
        bind_agent = getattr(self.patient_memory_service, "bind_agent", None)
        if callable(bind_agent):
            bind_agent(session.agent)
        client_id = connection.client_id
        self._log(f"[连接] 新客户端: {client_id}")
        self._log(
            "[配置] 全双工自动打断: "
            f"{'开启' if self.config.enable_full_duplex else '关闭'}"
        )
        self._log(
            "[配置] 轮次判断: "
            + (
                f"SoulX {self.config.soulx_turn_url}"
                if self.config.use_soulx_turn_taking
                else "本地 Silero VAD"
            )
        )

        try:
            components = await self._build_components(
                connection,
                session,
                client_id,
                authenticated_user,
            )
        except Exception as exc:
            self._log(f"[连接] ❌ 组装会话组件失败: {type(exc).__name__}")
            await connection.close(code=1011)
            return
        if components is None:
            return
        controller, cleanup = components

        try:
            await controller.run()
        except WebSocketDisconnect:
            self._log(f"[断开] 客户端 {client_id}")
        except RuntimeError as exc:
            if "WebSocket is not connected" in str(exc):
                self._log(
                    f"[断开] WebSocket 连接已断开 {client_id}"
                )
            else:
                raise
        except Exception as exc:
            self._log(f"[错误] WebSocket 端点异常: {type(exc).__name__}")
        finally:
            await cleanup.close(client_id)

    async def _build_components(
        self,
        connection: VoiceConnectionIO,
        session: VoiceSession,
        client_id: str,
        authenticated_user: dict,
    ) -> tuple[VoiceConnectionController, VoiceConnectionCleanup] | None:
        config = self.config
        answer_completion = AnswerCompletionState()
        vad_buffer = VADBuffer()
        soulx_client = self._create_soulx_client()
        soulx_audio = SoulXAudioAccumulator(
            sample_rate=16000,
            pre_roll_seconds=config.soulx_pre_roll_s,
        )
        speaker = SpeakerVerificationController(
            connection,
            verifier=self.models.speaker_verifier,
            uploads=session.enroll_sample_uploads,
        )
        profile_service = PatientProfileService()
        resume_state_service = AgentResumeStateService(session.agent)
        bootstrap = VoiceSessionBootstrap(
            connection,
            session=session,
            create_session=self.repository.create_session,
        )
        interruption = VoiceInterruptionController(
            connection,
            session=session,
            vad_buffer=vad_buffer,
            soulx_client=soulx_client,
            soulx_audio=soulx_audio,
            quick_asr=self.recognition.quick_asr,
            judge_answer_completion=(
                self.recognition.judge_answer_completion
            ),
            judge_interrupt_intent=(
                self.recognition.judge_interrupt_intent
            ),
            extract_latest_assistant_utterance=(
                self.recognition.extract_latest_assistant_utterance
            ),
            normalize_interrupt_text=(
                self.recognition.normalize_interrupt_text
            ),
            soulx_session_id_factory=self._soulx_session_id,
            cancel_render_jobs=(
                self.render_manager.cancel if self.render_manager else None
            ),
        )
        manifest_store = VoiceDatasetManifestStore(
            session,
            list_audio=self.repository.list_audio_records,
            normalize_profile=profile_service.normalize,
        )
        def enqueue_render_audio(audio_meta: dict, content_text: str) -> None:
            if self.render_manager is None:
                return
            identity = connection.current_output_identity() or {}
            processing = session.processing
            if not identity:
                turn_id = str(
                    processing.active_turn_id
                    or f"system-{session.session_id or 'session'}"
                )
                generation = int(processing.generation or 0)
                identity = {
                    "session_id": session.session_id,
                    "turn_id": turn_id,
                    "generation": generation,
                    "playback_id": str(
                        processing.active_playback_id
                        or f"playback-{turn_id}-{generation}"
                    ),
                }
            self.render_manager.enqueue(
                session=session,
                connection=connection,
                audio_meta=audio_meta,
                reply_text=content_text,
                emotion="neutral",
                output_identity=identity,
            )

        audio_store = VoiceAudioStore(
            session,
            save_audio=self.repository.save_audio_record,
            manifest_store=manifest_store,
            on_assistant_persist=enqueue_render_audio,
        )
        history_store = VoiceHistoryStore(
            session,
            save_message=self.repository.save_message,
            manifest_store=manifest_store,
        )
        tts_streamer = VoiceTTSStreamer(
            connection,
            session=session,
            audio_store=audio_store,
            tts=self.models.tts,
            clean_for_tts=clean_for_tts,
        )
        realtime_companion = RealtimeCompanion(
            connection,
            session=session,
            patient_memory_service=self.patient_memory_service,
            stream_tts_audio=tts_streamer.stream,
            config=RealtimeCompanionConfig(
                enabled=(
                    config.enable_realtime_companion
                    and (
                        config.use_soulx_turn_taking
                        or (
                            config.use_ark_asr
                            and ark_asr_streaming_supported()
                        )
                    )
                ),
                external_asr=config.use_soulx_turn_taking,
                emotion_interval_s=config.realtime_emotion_interval_s,
                emotion_window_s=config.realtime_emotion_window_s,
                memory_timeout_s=config.memory_timeout_s,
                memory_prefetch_stability_s=config.memory_prefetch_stability_s,
                memory_prefetch_min_interval_s=config.memory_prefetch_min_interval_s,
            ),
        )
        def authorize_patient(patient_id: str, action: str = "voice") -> bool:
            self.auth.authorize_patient(authenticated_user, patient_id, action)
            return True
        try:
            bootstrap.create_fresh()
        except Exception as exc:
            self._log(f"[DB] ❌ 初始化会话失败: {type(exc).__name__}")
            await connection.close(code=1011)
            return None

        self._log(
            f"[连接] 会话 {session.session_id} - 等待患者信息..."
        )
        presenters = self._build_presenters(connection, session)
        output_presenter, voice_feedback = presenters
        speech_processor = SpeechTurnProcessor(
            connection,
            session=session,
            speaker=speaker,
            audio_store=audio_store,
            history_store=history_store,
            presenter=output_presenter,
            stream_tts_audio=tts_streamer.stream,
            notify_voice_input_feedback=voice_feedback.send,
            normalize_profile=profile_service.normalize,
            parse_sensevoice_result=(
                self.recognition.parse_sensevoice_result
            ),
            audio_signal_stats=audio_signal_stats,
            audio_is_effectively_silent=audio_is_effectively_silent,
            asr_model=self.models.asr_model,
            tts=self.models.tts,
            clean_for_tts=clean_for_tts,
            config=SpeechTurnConfig(
                use_ark_asr=config.use_ark_asr,
                use_ark_tts=config.use_ark_tts,
                use_llm_streaming=config.use_llm_streaming,
                vision_lock_timeout=config.vision_lock_timeout_s,
                memory_timeout_s=config.memory_timeout_s,
            ),
            capture_memory_turn=getattr(
                self.patient_memory_service,
                "capture_turn",
                None,
            ),
            get_turn_memory=getattr(
                self.patient_memory_service,
                "get_turn_context",
                None,
            ),
            get_cross_session_memory=getattr(
                self.patient_memory_service,
                "get_cross_session_context",
                None,
            ),
            update_memory_emotion=getattr(
                self.patient_memory_service,
                "update_turn_emotion",
                None,
            ),
            record_safety_event=getattr(
                self.repository,
                "record_safety_event",
                None,
            ),
            update_memory_status=getattr(
                self.patient_memory_service,
                "update_turn_status",
                None,
            ),
        )
        processing = SpeechProcessingCoordinator(
            session,
            processor=speech_processor,
            enabled_full_duplex=config.enable_full_duplex,
            is_ai_speaking=interruption.is_ai_speaking,
            stop_playback=interruption.stop_playback,
            disconnect_error_type=WebSocketDisconnect,
            client_id=client_id,
            cancel_render_jobs=(
                self.render_manager.cancel if self.render_manager else None
            ),
        )
        answer_controller = AnswerCompletionController(
            connection,
            session=session,
            state=answer_completion,
            processing_coordinator=processing,
            quick_asr=self.recognition.quick_asr,
            judge_answer_completion=(
                self.recognition.judge_answer_completion
            ),
            extract_latest_assistant_utterance=(
                self.recognition.extract_latest_assistant_utterance
            ),
            observation_window_s=(
                config.answer_completion_observation_window_s
            ),
        )
        blob_processor = ManualAudioBlobProcessor(
            connection,
            processing_coordinator=processing,
            feedback=voice_feedback,
            decode_recorded_audio_blob=decode_recorded_audio_blob,
            resample_audio=resample_audio,
            audio_signal_stats=audio_signal_stats,
            audio_is_effectively_silent=audio_is_effectively_silent,
        )
        router = self._build_router(
            connection=connection,
            session=session,
            client_id=client_id,
            answer_completion=answer_completion,
            answer_controller=answer_controller,
            audio_store=audio_store,
            blob_processor=blob_processor,
            bootstrap=bootstrap,
            history_store=history_store,
            interruption=interruption,
            manifest_store=manifest_store,
            output_presenter=output_presenter,
            processing=processing,
            profile_service=profile_service,
            resume_state_service=resume_state_service,
            soulx_audio=soulx_audio,
            soulx_client=soulx_client,
            speaker=speaker,
            tts_streamer=tts_streamer,
            vad_buffer=vad_buffer,
            voice_feedback=voice_feedback,
            realtime_companion=realtime_companion,
            authorize_patient=authorize_patient,
        )
        cleanup = VoiceConnectionCleanup(
            session,
            answer_completion=answer_completion,
            turn_client=soulx_client,
            patient_memory_service=self.patient_memory_service,
            end_session=self.repository.end_session,
            realtime_companion=realtime_companion,
            cancel_render_jobs=(
                self.render_manager.cancel if self.render_manager else None
            ),
        )
        await bootstrap.send_waiting_for_info()
        return VoiceConnectionController(connection, router), cleanup

    def _build_router(
        self,
        *,
        connection,
        session,
        client_id,
        answer_completion,
        answer_controller,
        audio_store,
        blob_processor,
        bootstrap,
        history_store,
        interruption,
        manifest_store,
        output_presenter,
        processing,
        profile_service,
        resume_state_service,
        soulx_audio,
        soulx_client,
        speaker,
        tts_streamer,
        vad_buffer,
        voice_feedback,
        realtime_companion,
        authorize_patient,
    ) -> VoiceMessageRouter:
        config = self.config
        router = VoiceMessageRouter()
        client_control = ClientControlHandler(connection)
        resume_handler = SessionResumeHandler(
            connection,
            session=session,
            manifest_store=manifest_store,
            get_resume=self.repository.get_session_for_resume,
            get_session_binding=(
                self.repository.get_session_resume_binding
            ),
            assign_session_owner=self.repository.assign_session_owner,
            normalize_profile=profile_service.normalize,
            sync_manual_location=profile_service.sync_location,
            build_score_payload=resume_state_service.build_score_payload,
            sync_agent_state=resume_state_service.sync,
            reset_turn_session=interruption.reset_turn_taking,
            authorize_patient=authorize_patient,
        )
        answer_handler = AnswerCompletionMessageHandler(
            answer_completion,
            notify_status=answer_controller.notify,
            request_check=answer_controller.request_check,
        )
        lifecycle_handler = SessionLifecycleHandler(
            connection,
            session=session,
            answer_completion=answer_completion,
            history_store=history_store,
            manifest_store=manifest_store,
            patient_memory_service=self.patient_memory_service,
            create_fresh_session=bootstrap.create_fresh,
            end_session=self.repository.end_session,
            update_profile=self.repository.update_session_profile,
            normalize_profile=profile_service.normalize,
            sync_manual_location=profile_service.sync_location,
            reset_turn_session=interruption.reset_turn_taking,
            reset_vad=vad_buffer.reset,
            stream_tts_audio=tts_streamer.stream,
            clean_for_tts=clean_for_tts,
            send_waiting_for_info=bootstrap.send_waiting_for_info,
            client_id=client_id,
            cancel_active_processing=processing.cancel_active,
            reset_realtime=realtime_companion.reset,
            authorize_patient=authorize_patient,
        )
        manual_audio_handler = ManualAudioInputHandler(
            connection,
            uploads=session.manual_audio_blob_uploads,
            process_recorded_blob=blob_processor.process,
            submit_speech=processing.submit_or_queue,
            normalize_doctor_markers=(
                blob_processor.normalize_doctor_markers
            ),
            resample_audio=resample_audio,
        )
        manual_interrupt_handler = ManualInterruptHandler(
            session=session,
            stop_playback=interruption.stop_playback,
            reset_interrupt_capture=interruption.reset_capture,
            interrupt_active_processing=(
                lambda reason: processing.interrupt_active(
                    reason,
                    force_playback=True,
                )
            ),
        )
        vision_handler = VisionMessageHandler(
            connection,
            session=session,
            history_store=history_store,
            presenter=output_presenter,
            stream_tts_audio=tts_streamer.stream,
            clean_for_tts=clean_for_tts,
            normalize_profile=profile_service.normalize,
        )
        text_handler = TextTurnHandler(
            connection,
            session=session,
            history_store=history_store,
            audio_store=audio_store,
            presenter=output_presenter,
            stream_tts_audio=tts_streamer.stream,
            clean_for_tts=clean_for_tts,
            normalize_profile=profile_service.normalize,
            use_llm_streaming=config.use_llm_streaming,
            use_ark_tts=config.use_ark_tts,
            tts=self.models.tts,
            capture_memory_turn=getattr(
                self.patient_memory_service,
                "capture_turn",
                None,
            ),
            record_safety_event=getattr(
                self.repository,
                "record_safety_event",
                None,
            ),
            update_memory_status=getattr(
                self.patient_memory_service,
                "update_turn_status",
                None,
            ),
        )
        soulx_handler = SoulXAudioHandler(
            connection,
            client=soulx_client,
            accumulator=soulx_audio,
            turn_state=session.turn_taking,
            speaker=speaker,
            answer_completion=answer_completion,
            reset_local_vad=vad_buffer.reset,
            is_ai_speaking=interruption.is_ai_speaking,
            stop_playback=interruption.stop_playback,
            is_processing=lambda: session.processing.is_active,
            interrupt_active_processing=(
                lambda reason: processing.interrupt_active(reason, force_playback=True)
            ),
            notify_answer_completion=answer_controller.notify,
            arm_answer_completion_window=(
                answer_controller.maybe_arm_window
            ),
            submit_speech=processing.submit_or_queue,
            audio_signal_stats=audio_signal_stats,
            unavailable_error_type=SoulXUnavailable,
            server_url=config.soulx_turn_url,
            enabled_full_duplex=config.enable_full_duplex,
            minimum_utterance_rms=(
                config.soulx_minimum_utterance_rms
            ),
            barge_in_minimum_chunk_rms=(
                config.soulx_barge_in_minimum_chunk_rms
            ),
            realtime_companion=realtime_companion,
            ensure_session_on_speech=(
                lifecycle_handler.start_next_on_speech
            ),
        )
        live_handler = LiveAudioInputHandler(
            connection,
            session=session,
            vad_buffer=vad_buffer,
            soulx_audio_handler=soulx_handler,
            speaker=speaker,
            answer_completion=answer_completion,
            history_store=history_store,
            is_ai_speaking=interruption.is_ai_speaking,
            reset_interrupt_capture=interruption.reset_capture,
            remember_interrupt_judgement=(
                interruption.remember_judgement
            ),
            reuse_interrupt_judgement=interruption.reuse_judgement,
            judge_waiting_completion=(
                interruption.judge_waiting_completion
            ),
            interrupt_active_processing=processing.interrupt_active,
            stop_playback=interruption.stop_playback,
            get_active_processing_audio=processing.active_audio_copy,
            notify_voice_input_feedback=voice_feedback.send,
            submit_speech=processing.submit_or_queue,
            notify_answer_completion=answer_controller.notify,
            arm_answer_completion_window=(
                answer_controller.maybe_arm_window
            ),
            quick_asr=self.recognition.quick_asr,
            judge_interrupt_intent=(
                self.recognition.judge_interrupt_intent
            ),
            stream_tts_audio=tts_streamer.stream,
            realtime_companion=realtime_companion,
            ensure_session_on_speech=(
                lifecycle_handler.start_next_on_speech
            ),
            config=LiveAudioConfig(
                enabled_full_duplex=config.enable_full_duplex,
                pre_end_arm_window_s=(
                    config.vad_pre_end_arm_window_s
                ),
                minimum_post_tts_chunks=(
                    config.vad_minimum_post_tts_chunks
                ),
                interrupt_min_duration=(
                    config.interrupt_minimum_duration
                ),
                interrupt_stop_duration=config.interrupt_stop_duration,
                interrupt_trigger_probability=(
                    config.interrupt_trigger_probability
                ),
                interrupt_min_rms=config.interrupt_minimum_rms,
                interrupt_min_consecutive_chunks=(
                    config.interrupt_minimum_consecutive_chunks
                ),
                interrupt_complete_silence_s=(
                    config.interrupt_complete_silence_s
                ),
                interrupt_min_complete_audio_s=(
                    config.interrupt_minimum_complete_audio_s
                ),
            ),
        )
        handlers = (
            (ClientControlHandler.MESSAGE_TYPES, client_control),
            (SpeakerVerificationController.MESSAGE_TYPES, speaker),
            (SessionResumeHandler.MESSAGE_TYPES, resume_handler),
            (
                AnswerCompletionMessageHandler.MESSAGE_TYPES,
                answer_handler,
            ),
            (
                SessionLifecycleHandler.MESSAGE_TYPES,
                lifecycle_handler,
            ),
            (ManualAudioInputHandler.MESSAGE_TYPES, manual_audio_handler),
            (
                ManualInterruptHandler.MESSAGE_TYPES,
                manual_interrupt_handler,
            ),
            (VisionMessageHandler.MESSAGE_TYPES, vision_handler),
            (TextTurnHandler.MESSAGE_TYPES, text_handler),
            (LiveAudioInputHandler.MESSAGE_TYPES, live_handler),
        )
        for message_types, handler in handlers:
            router.register_many(
                message_types,
                handler.handle_message,
            )
        return router

    @staticmethod
    def _build_presenters(connection, session):
        return (
            AgentOutputPresenter(connection, session=session),
            VoiceFeedbackPresenter(connection),
        )

    def _create_soulx_client(self):
        if not self.config.use_soulx_turn_taking:
            return None
        return SoulXTurnTakingClient(
            self.config.soulx_turn_url,
            session_id=self._soulx_session_id(),
            timeout=self.config.soulx_timeout_s,
            retry_interval=self.config.soulx_retry_interval_s,
        )

    @staticmethod
    def _soulx_session_id() -> str:
        return f"voice8502_{uuid.uuid4().hex}"
