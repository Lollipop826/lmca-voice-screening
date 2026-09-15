from pathlib import Path


HISTORY_HTML = Path(__file__).resolve().parents[2] / "static" / "history.html"


def test_history_page_is_companion_record_surface():
    html = HISTORY_HTML.read_text(encoding="utf-8")

    assert "<title>陪伴记录 · 心语陪伴</title>" in html
    assert "返回陪伴" in html
    assert "历史会话" in html
    assert 'id="stat-messages"' in html
    assert 'id="metric-mode"' in html
    assert "function renderMessages(messages)" in html
    assert "function renderAudio(sessionId, audios)" in html
    assert "sessionMode(s) !== 'wellbeing'" in html
    for removed in ("scores-tab-btn", "tab-scores", "isCognitiveSession", "认知", "评分", "MMSE"):
        assert removed not in html


def test_history_exports_only_conversations_without_screening_reports():
    html = HISTORY_HTML.read_text(encoding="utf-8")

    assert "function buildTxtContent(data)" in html
    assert "function buildCsvContent(data)" in html
    assert "downloadRecord(buildCsvContent(currentDetail)" in html
    assert "link.download = '陪伴记录_'" in html
    assert "export/audio-merged.wav" in html
    assert "export/audio-clips.zip" in html
    assert "/export/csv" not in html
    assert "score_records" not in html
    assert "mmse_total" not in html
