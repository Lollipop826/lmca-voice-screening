from pathlib import Path


HISTORY_HTML = Path(__file__).resolve().parents[2] / "static" / "history.html"


def test_history_page_is_patient_companion_record_surface():
    html = HISTORY_HTML.read_text(encoding="utf-8")

    assert "<title>陪伴记录 · 心理健康陪伴</title>" in html
    assert "历史评估记录" not in html
    assert "返回陪伴" in html
    assert "function isCognitiveSession(session)" in html
    assert "document.getElementById('scores-tab-btn').hidden = !cognitive" in html
    assert "本会话无认知筛查评分。" in html
    assert "const prefix = currentDetail && isCognitiveSession(currentDetail.session) ? '认知筛查记录' : '陪伴记录'" in html
