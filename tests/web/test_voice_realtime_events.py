from pathlib import Path


VOICE_HTML = Path(__file__).resolve().parents[2] / "static" / "voice_chat.html"


def test_voice_page_renders_and_finalizes_partial_asr_text():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "data.type === 'asr_partial'" in html
    assert "function normalizeDisplayedAsrText(text)" in html
    assert "const displayedText = normalizeDisplayedAsrText(data.text);" in html
    assert "updatePartialTranscript(displayedText)" in html
    assert "finalizePartialTranscript(displayedText)" in html
    assert "discardPartialTranscript();" in html
    assert "data.type === 'companion_message'" in html


def test_voice_page_rebinds_only_the_patient_after_disconnect():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "localStorage.setItem('voice_patient_id', patientId)" in html
    assert "type: 'prepare_patient'" in html
    assert "data.type === 'patient_memory'" in html
    assert "data.type === 'session_waiting_next_speech'" in html
    assert "awaitingNextUtterance" in html


def test_voice_page_ignores_stale_playback_but_keeps_stop_controls():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "OUTPUT_STOP_EVENT_TYPES" in html
    assert "data.playback_id !== _latestOutputPlaybackId" in html
    assert "data.type === 'tts_start'" in html
    assert "OUTPUT_STOP_EVENT_TYPES.has(data.type)" in html


def test_voice_page_defaults_to_wellbeing_and_sends_session_mode():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "<title>记忆守护 · 心理健康陪伴</title>" in html
    assert '<body class="mode-wellbeing">' in html
    assert "记忆守护 · 心理健康陪伴" in html
    assert "开始陪伴" in html
    assert "let currentSessionMode = 'wellbeing';" in html
    assert "mode: currentSessionMode" in html
    assert "if (!isCognitiveMode()) delete profile.education_years;" in html


def test_voice_page_hides_cognitive_ui_until_explicit_mode():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "body.mode-wellbeing [data-cognitive-only]" in html
    assert 'id="education-group" class="input-group settings-grid-wide" data-cognitive-only' in html
    assert 'id="score-settings-section"' in html
    assert '<img id="drawing-ref-img" alt="参考五边形">' in html
    assert '<img id="display-image" alt="认知筛查题目图片"' in html
    assert 'src=""' not in html
    assert 'src="/api/mmse-image/pentagons"' not in html


def test_voice_page_consumes_turn_insight_by_turn_id():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "const turnRecords = new Map();" in html
    assert "function ensureTurnRecord(turnId)" in html
    assert "data.type === 'turn_insight'" in html
    assert "handleTurnInsight(data)" in html
    assert "updateTurnRecord(data.turn_id || latestTurnId" in html


def test_voice_page_ignores_cognitive_events_in_wellbeing_mode():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert "data.type === 'update_score'" in html
    assert "data.type === 'show_image'" in html
    assert "data.type === 'vision_capture'" in html
    assert "if (!isCognitiveMode(data.mode || currentSessionMode))" in html
    assert "已忽略非认知会话评分事件" in html
    assert "已忽略非认知会话题图" in html
    assert "已忽略非认知会话视觉任务" in html


def test_voice_page_uses_a_dedicated_memory_sidebar_and_drag_safe_log_button():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert 'id="memory-overlay"' in html
    assert 'aria-label="记忆侧边栏"' in html
    assert "panel.classList.toggle('open', shouldOpen)" in html
    assert "overlay?.classList.toggle('open', shouldOpen)" in html
    assert "const FAB_POSITION_KEY = 'debug_log_fab_position';" in html
    assert "fab.addEventListener('pointerdown'" in html
    assert "const textInputBar = document.getElementById('text-input-bar');" in html
    assert "const overlapsInput = left < inputRect.right" in html
    assert "positionPanelNearFab();" in html


def test_voice_page_separates_long_term_session_and_turn_memory_views():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert 'role="tablist" aria-label="记忆视图"' in html
    assert 'id="memory-view-longterm"' in html
    assert 'id="memory-view-session"' in html
    assert 'id="memory-view-turn"' in html
    assert "function setMemoryPanelView(view = 'longterm')" in html
    assert "toggleMemoryPanel(true, 'longterm')" in html
    assert "心理健康陪伴模式不生成会话摘要。" in html


def test_voice_page_keeps_the_initial_welcome_hint_compact():
    html = VOICE_HTML.read_text(encoding="utf-8")

    assert 'class="message-row ai welcome-message"' in html
    assert "div.className = 'message-row ai welcome-message';" in html
    assert ".message-row.welcome-message .bubble" in html
    assert "background: #edf8f8 !important;" in html
    assert "border: 1px solid #c7e3e3 !important;" in html
