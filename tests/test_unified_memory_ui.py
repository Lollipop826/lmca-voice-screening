from pathlib import Path
import re


VOICE_PAGE = Path(__file__).parents[1] / "static" / "voice_chat.html"


def _memory_panel_markup() -> str:
    html = VOICE_PAGE.read_text(encoding="utf-8")
    start = html.index("<!-- 记忆侧边栏 -->")
    end = html.index("<!-- Controls -->", start)
    return html[start:end]


def _css_block(html: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>[^}}]*)\}}", html)
    assert match, f"missing CSS block for {selector}"
    return match.group("body")


def test_unified_memory_panel_contains_non_mmse_memory_management_views():
    panel = _memory_panel_markup()

    for marker in (
        'id="memory-tab-profile"',
        'id="memory-tab-emotion"',
        'id="memory-view-profile"',
        'id="memory-view-emotion"',
        'id="memory-profile-form"',
        'id="memory-emotion-list"',
    ):
        assert marker in panel

    assert "记录MMSE" not in panel
    assert "/emotion-trajectory" in VOICE_PAGE.read_text(encoding="utf-8")


def test_memory_tabs_keep_horizontal_scroll_without_visible_scrollbar():
    html = VOICE_PAGE.read_text(encoding="utf-8")

    assert "overflow-x: auto" in html
    assert "scrollbar-width: none" in html
    assert ".memory-panel-tabs::-webkit-scrollbar" in html


def test_memory_tabs_cannot_shrink_over_scroll_body_and_view_switch_resets_scroll():
    html = VOICE_PAGE.read_text(encoding="utf-8")
    tabs_css = _css_block(html, ".memory-panel-tabs")
    body_css = _css_block(html, "#memory-panel .mem-body,\n        #memory-panel.expanded .mem-body")
    set_view = html[html.index("function setMemoryPanelView"):html.index("function toggleMemoryPanel")]

    assert "display: flex;" in tabs_css
    assert "flex: 0 0 auto;" in tabs_css
    assert "min-height: 48px;" in tabs_css
    assert "flex: 1 1 auto;" in body_css
    assert "min-height: 0;" in body_css
    assert "memoryBody.scrollTop = 0" in set_view


def test_wellbeing_session_summary_uses_reflection_api_after_session_end():
    html = VOICE_PAGE.read_text(encoding="utf-8")

    assert "心理健康陪伴模式不生成会话摘要" not in html
    assert "function fetchSessionReflection()" in html
    assert "/session-reflection?session_id=" in html
    assert "previous_reflection" in html
    assert "上一轮摘要" in html
    assert "会话结束后异步生成摘要" in html
    assert "startMemoryPoll({ afterSessionEnd: true })" in html


def test_emotion_trajectory_has_color_coded_comparison_and_safe_failure_copy():
    html = VOICE_PAGE.read_text(encoding="utf-8")

    assert "memory-emotion-heatmap" in html
    assert "memory-emotion-fill emotion-" in html
    assert "emotionColors" in html
    assert "display: block;" in _css_block(html, ".memory-emotion-fill")
    assert "failure_reason" in html
    assert "next_retry_at" in html
