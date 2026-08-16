from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import Any, Optional

from .catalog import ScreeningTaskCatalog
from .state import ScreeningSessionState


class ScreeningQuestionClassifier:
    """Classify generated questions without sharing executor state implicitly."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        log_summary: Callable[[str, dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self._log_summary = log_summary
        self._log_verbose = log_verbose

    def classify(self, question: str, candidates: list[str]) -> str:
        if not candidates:
            return "buffer_chat"
        normalized = question.strip()
        for task_id, keywords in self.catalog.task_keyword_rules:
            if task_id in candidates and any(
                keyword in normalized for keyword in keywords
            ):
                self._log_verbose(f"规则分类: {task_id} (命中关键词)")
                return task_id
        self._log_verbose("规则未命中，归为闲聊")
        return "buffer_chat"

    def start(
        self,
        question: str,
        candidates: Optional[list[str]] = None,
    ) -> None:
        self.reset()
        selected_candidates = list(
            self.state._available_candidates
            if candidates is None
            else candidates
        )
        preview = question[:50] + "..." if len(question) > 50 else question
        self._log_verbose(
            "后台分类任务启动: "
            f"question='{preview}', "
            f"candidates={selected_candidates[:5]}"
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.state._classification_executor = executor
        self.state._pending_classification_task = executor.submit(
            self.classify,
            question,
            selected_candidates,
        )

    def result(self, timeout: float = 1.0) -> Optional[str]:
        pending = self.state._pending_classification_task
        if pending is None:
            return self.state._last_classification_result
        try:
            result = pending.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._log_verbose("分类超时，使用上次结果")
            return self.state._last_classification_result
        except Exception as exc:
            self._log_summary(
                "Classification",
                {"status": "failed", "error": str(exc)},
            )
            self.state._pending_classification_task = None
            return "buffer_chat"
        self.state._last_classification_result = result
        self.state._pending_classification_task = None
        return result

    def reset(self) -> None:
        executor = self.state._classification_executor
        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        self.state._classification_executor = None
        self.state._pending_classification_task = None
        self.state._last_classification_result = None
        self.state._available_candidates = []
