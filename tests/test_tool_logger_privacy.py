from src.utils.tool_logger import (
    ToolLogger,
    log_summary,
    register_tool_event_callback,
    set_current_tool_log_session,
    unregister_tool_event_callback,
)


def test_tool_logger_redacts_text_from_stdout_and_events(capsys):
    secret = "private patient transcript"
    events = []
    session_id = "tool-logger-privacy-test"
    register_tool_event_callback(session_id, events.append)
    set_current_tool_log_session(session_id)
    try:
        logger = ToolLogger("ASR").start(answer=secret)
        logger.log(secret)
        logger.step(secret)
        logger.end(response=secret)
        log_summary("summary", {"content": secret})
    finally:
        set_current_tool_log_session(None)
        unregister_tool_event_callback(session_id)

    assert secret not in capsys.readouterr().out
    assert secret not in repr(events)
    assert any("text_chars=" in repr(event) for event in events)
