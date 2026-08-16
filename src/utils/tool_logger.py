"""
统一工具日志系统
提供醒目、结构化的日志输出，清晰展示工具执行流程和数据传递
"""
import threading
import time
import textwrap
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager


_PRINT_LOCK = threading.Lock()
_CARD_INNER_WIDTH = 76


def _box_border(char: str, left: str, right: str) -> str:
    return f"{left}{char * (_CARD_INNER_WIDTH + 2)}{right}"


def _box_line(text: str = "") -> str:
    return f"║ {text[:_CARD_INNER_WIDTH].ljust(_CARD_INNER_WIDTH)} ║"


def _safe_display_value(value: Any) -> str:
    """Serialize diagnostics without exposing user-provided content."""
    if isinstance(value, str):
        return f"text_chars={len(value)}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"bytes={len(value)}"
    if isinstance(value, dict):
        return f"dict_fields={len(value)}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__}_items={len(value)}"
    return str(value)


def _normalize_text(value: Any) -> str:
    return _safe_display_value(value).replace("\n", " / ").strip()


def _wrap_with_prefix(prefix: str, value: Any, width: int = _CARD_INNER_WIDTH) -> list[str]:
    text = _normalize_text(value)
    available = max(10, width - len(prefix))
    wrapped = textwrap.wrap(text, width=available, replace_whitespace=False, drop_whitespace=False) or [""]
    indent = " " * len(prefix)
    return [prefix + wrapped[0], *[indent + part for part in wrapped[1:]]]


def _print_lines(lines: list[str]) -> None:
    with _PRINT_LOCK:
        print("\n".join(lines), flush=True)


class ToolLogger:
    """工具日志记录器"""
    
    # 工具图标映射
    TOOL_ICONS = {
        'AgentFC': '🤖',
        'ResistanceTool': '🛡️',
        'ComfortTool': '💬',
        'QuestionGenTool': '❓',
        'AnswerEvalTool': '📝',
        'ScoreTool': '📊',
        'TaskPool': '📋',
        'RetrievalTool': '🔎',
        'MemoryTool': '🧠',
        'StandardQuestion': '📌',
        'ImageTool': '🖼️',
        'TTS': '🎵',
        'ASR': '🎤',
        'VAD': '👂',
    }
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.icon = self.TOOL_ICONS.get(tool_name, '🔧')
        self._start_time = None
        self._inputs = {}
        self._outputs = {}
        self._timeline = []
        self._start_display_time = None
        self._log_id = f"{tool_name}_{int(time.time() * 1000)}_{id(self)}"
    
    def start(self, **inputs):
        """开始工具执行，记录输入"""
        self._start_time = time.time()
        self._inputs = inputs
        self._timeline = []
        now = datetime.now().strftime("%H:%M:%S")
        self._start_display_time = now
        
        emit_tool_event(
            {
                "phase": "start",
                "log_id": self._log_id,
                "tool_name": self.tool_name,
                "icon": self.icon,
                "timestamp": now,
                "inputs": _truncate_dict(inputs, 120),
            }
        )
        
        return self
    
    def log(self, message: str, level: str = "info"):
        """记录中间日志"""
        safe_message = f"message_chars={len(str(message))}"
        self._timeline.append((level, safe_message))
        emit_tool_event(
            {
                "phase": "log",
                "log_id": self._log_id,
                "tool_name": self.tool_name,
                "icon": self.icon,
                "level": level,
                "message": safe_message,
            }
        )
    
    def step(self, message: str):
        """记录执行步骤"""
        safe_message = f"message_chars={len(str(message))}"
        self._timeline.append(("step", safe_message))
        emit_tool_event(
            {
                "phase": "step",
                "log_id": self._log_id,
                "tool_name": self.tool_name,
                "icon": self.icon,
                "message": safe_message,
            }
        )
    
    def end(self, **outputs):
        """结束工具执行，记录输出和耗时"""
        self._outputs = outputs
        elapsed = time.time() - self._start_time if self._start_time else 0

        lines = [
            "",
            _box_border("═", "╔", "╗"),
            _box_line(f"⏰ [{self._start_display_time or datetime.now().strftime('%H:%M:%S')}] {self.icon} {self.tool_name}"),
            _box_border("═", "╠", "╣"),
        ]

        if self._inputs:
            lines.append(_box_line("📥 输入"))
            for key, value in self._inputs.items():
                for line in _wrap_with_prefix(f"  - {key}: ", _truncate_value(value, 240)):
                    lines.append(_box_line(line))

        if self._timeline:
            lines.append(_box_line("🧭 过程"))
            level_icons = {
                "info": "ℹ️",
                "warn": "⚠️",
                "error": "❌",
                "success": "✅",
                "step": "•",
            }
            for level, message in self._timeline:
                prefix = f"  {level_icons.get(level, 'ℹ️')} "
                for line in _wrap_with_prefix(prefix, _truncate_value(message, 320)):
                    lines.append(_box_line(line))

        if outputs:
            lines.append(_box_line("📤 输出"))
            for key, value in outputs.items():
                for line in _wrap_with_prefix(f"  - {key}: ", _truncate_value(value, 240)):
                    lines.append(_box_line(line))

        lines.append(_box_line(f"⏱️  耗时: {elapsed:.3f}s"))
        lines.append(_box_border("═", "╚", "╝"))
        _print_lines(lines)

        emit_tool_event(
            {
                "phase": "end",
                "log_id": self._log_id,
                "tool_name": self.tool_name,
                "icon": self.icon,
                "elapsed_s": round(elapsed, 3),
                "outputs": _truncate_dict(outputs, 120),
            }
        )
        
        return outputs
    
    def end_with_arrow(self, next_tool: str, data_passed: str, **outputs):
        """结束并显示数据流向下一个工具"""
        self.end(**outputs)
        next_icon = self.TOOL_ICONS.get(next_tool, '🔧')
        _print_lines([
            "         │",
            f"         ▼ ({data_passed})",
            f"         │  {next_icon} {next_tool}",
        ])


@contextmanager
def tool_context(tool_name: str, **inputs):
    """
    工具执行上下文管理器
    
    Usage:
        with tool_context("ResistanceTool", question=q, answer=a) as logger:
            # do work
            logger.step("BERT预测完成")
            logger.end(label="normal", confidence=0.95)
    """
    logger = ToolLogger(tool_name)
    logger.start(**inputs)
    try:
        yield logger
    except Exception as e:
        logger.log(f"执行失败: {e}", level="error")
        logger.end(error=str(e))
        raise


def log_tool_start(tool_name: str, **inputs) -> ToolLogger:
    """快速启动工具日志"""
    logger = ToolLogger(tool_name)
    logger.start(**inputs)
    return logger


_EVENT_CALLBACKS: Dict[str, Callable[[Dict[str, Any]], None]] = {}
_SCORE_CALLBACKS: Dict[str, Callable[[str], None]] = {}
_CONTEXT = threading.local()


def _truncate_value(value: Any, max_len: int = 120) -> str:
    text = _safe_display_value(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _truncate_dict(data: Dict[str, Any], max_len: int = 120) -> Dict[str, str]:
    return {str(k): _truncate_value(v, max_len) for k, v in (data or {}).items()}


def register_tool_event_callback(session_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
    _EVENT_CALLBACKS[session_id] = callback


def unregister_tool_event_callback(session_id: str) -> None:
    _EVENT_CALLBACKS.pop(session_id, None)


def register_score_event_callback(session_id: str, callback: Callable[[str], None]) -> None:
    _SCORE_CALLBACKS[session_id] = callback


def unregister_score_event_callback(session_id: str) -> None:
    _SCORE_CALLBACKS.pop(session_id, None)


def emit_score_event(session_id: str) -> None:
    callback = _SCORE_CALLBACKS.get(session_id)
    if not callback:
        return
    try:
        callback(session_id)
    except Exception:
        pass


def set_current_tool_log_session(session_id: Optional[str]) -> None:
    _CONTEXT.session_id = session_id


def get_current_tool_log_session() -> Optional[str]:
    return getattr(_CONTEXT, "session_id", None)


def emit_tool_event(event: Dict[str, Any]) -> None:
    session_id = get_current_tool_log_session()
    if not session_id:
        return
    callback = _EVENT_CALLBACKS.get(session_id)
    if not callback:
        return
    try:
        callback(dict(event, session_id=session_id))
    except Exception:
        pass


def log_data_flow(from_tool: str, to_tool: str, data: str):
    """显示工具间数据流动"""
    from_icon = ToolLogger.TOOL_ICONS.get(from_tool, '🔧')
    to_icon = ToolLogger.TOOL_ICONS.get(to_tool, '🔧')
    _print_lines([
        f"\n  {from_icon} {from_tool}",
        "         │",
        f"         ▼ (data_chars={len(str(data))})",
        "         │",
        f"  {to_icon} {to_tool}\n",
    ])


def log_phase(phase_name: str, phase_num: int = None):
    """记录执行阶段"""
    now = datetime.now().strftime("%H:%M:%S")
    if phase_num:
        _print_lines([f"\n{'─'*20} ⏰ [{now}] 阶段 {phase_num}: {phase_name} {'─'*20}\n"])
    else:
        _print_lines([f"\n{'─'*20} ⏰ [{now}] {phase_name} {'─'*20}\n"])


def log_summary(title: str, items: Dict[str, Any]):
    """记录摘要信息"""
    now = datetime.now().strftime("%H:%M:%S")
    lines = [f"\n┌{'─'*60}┐", f"│ ⏰ [{now}] 📊 {title:50} │", f"├{'─'*60}┤"]
    for key, value in items.items():
        str_val = _safe_display_value(value)
        if len(str_val) > 45:
            str_val = str_val[:42] + "..."
        lines.append(f"│   {key}: {str_val:52} │")
    lines.append(f"└{'─'*60}┘")
    _print_lines(lines)
