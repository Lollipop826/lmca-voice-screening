"""Public facade for the function-calling cognitive screening agent."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.agents.screening.catalog import ScreeningTaskCatalog
from src.agents.screening.state import ScreeningSessionState
from src.agents.screening.turn_pipeline import TurnPipeline
from src.domain.dimensions import MMSE_DIMENSIONS
from src.utils.tool_logger import log_summary


class ADScreeningAgentFunctionCalling:
    """Assemble the screening domain and expose its stable public API."""

    def __init__(self, use_local: bool = True) -> None:
        self.use_local = use_local
        self.state = ScreeningSessionState()
        self.catalog = ScreeningTaskCatalog()
        self.dimension_map = {
            dimension.get("id"): dimension for dimension in MMSE_DIMENSIONS
        }
        self._verbose_logs = os.getenv(
            "AGENT_VERBOSE_LOGS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}

        self._log_summary_card(
            "Agent Init",
            {
                "use_local": use_local,
                "verbose_logs": self._verbose_logs,
            },
        )
        self.turn_pipeline = TurnPipeline(
            use_local=use_local,
            state=self.state,
            catalog=self.catalog,
            dimension_map=self.dimension_map,
            log_summary=self._log_summary_card,
            log_verbose=self._log_verbose,
            verbose_logs=self._verbose_logs,
        )
        self.tool_gateway = self.turn_pipeline.tool_gateway
        self.question_classifier = self.turn_pipeline.question_classifier
        self.session_lifecycle = self.turn_pipeline.session_lifecycle
        self.task_planner = self.turn_pipeline.task_planner
        self.conversation_policy = self.turn_pipeline.policy
        self.answer_evaluator = self.turn_pipeline.answer_evaluator
        self.question_generator = self.turn_pipeline.question_generator
        self.background_analysis = self.turn_pipeline.background_analysis

        self.llm, interrupt_llm = self._create_interrupt_llm(use_local)
        self._log_summary_card(
            "Agent Ready",
            {
                "interrupt_llm": interrupt_llm,
                "tool_mode": "local" if use_local else "api_mix",
            },
        )

    def process_turn(
        self,
        user_input: str,
        dimension: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        current_emotion: str = "neutral",
    ) -> Dict[str, Any]:
        return self.turn_pipeline.process_turn(
            user_input=user_input,
            dimension=dimension,
            session_id=session_id,
            patient_profile=patient_profile,
            chat_history=chat_history,
            current_emotion=current_emotion,
        )

    def get_current_dimension(self) -> Dict[str, Any]:
        return self.state.current_dimension

    def set_dimension(self, dimension_id: str) -> None:
        if dimension := self.dimension_map.get(dimension_id):
            self.state.current_dimension = dimension

    def _create_interrupt_llm(self, use_local: bool):
        if use_local:
            from src.llm.model_pool import get_pooled_llm

            return get_pooled_llm(pool_key="small_classify"), "small_classify"

        from src.llm.http_client_pool import get_chat_openai

        return (
            get_chat_openai(
                temperature=0.1,
                max_tokens=8,
                timeout=5,
                max_retries=1,
            ),
            "api_lightweight",
        )

    def _log_summary_card(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        payload = {
            key: self._preview_log_value(value)
            for key, value in (items or {}).items()
            if value is not None and value != ""
        }
        if payload:
            log_summary(title, payload)

    def _log_verbose(self, message: str) -> None:
        if self._verbose_logs:
            print(f"[AgentFC] 🔍 message_chars={len(str(message))}")

    @staticmethod
    def _preview_log_value(value: Any, limit: int = 72) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return f"text_chars={len(value)}"
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"bytes={len(value)}"
        if isinstance(value, dict):
            return f"dict_fields={len(value)}"
        if isinstance(value, (list, tuple, set, frozenset)):
            return f"{type(value).__name__}_items={len(value)}"
        return str(value)
