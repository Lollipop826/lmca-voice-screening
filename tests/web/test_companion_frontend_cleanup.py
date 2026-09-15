"""Companion UI regressions; execute the shipped inline JS without a live server."""

import csv
import io
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[2] / "static"
PAGES = ("voice_chat", "history", "login", "admin", "api_portal")


class PageElements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


def inline_functions(page, *names):
    """Copy complete top-level functions, using their shared closing indentation."""
    html = (STATIC / f"{page}.html").read_text(encoding="utf-8")
    functions = []
    for name in names:
        match = re.search(
            rf"^(?P<indent>[ \t]*)(?:async )?function {re.escape(name)}\([^\n]*\) \{{"
            r"[\s\S]*?^(?P=indent)\}",
            html,
            re.MULTILINE,
        )
        assert match, f"Missing inline function: {page}.{name}"
        functions.append(match.group())
    return "\n".join(functions)


def inline_constant(page, name):
    html = (STATIC / f"{page}.html").read_text(encoding="utf-8")
    match = re.search(rf"\bconst {re.escape(name)} = [\s\S]*?;", html)
    assert match, f"Missing inline constant: {page}.{name}"
    return match.group()


def run_javascript(source, checks):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute inline JavaScript regressions")
    result = subprocess.run(
        [node, "-e", "const assert = require('node:assert/strict');\n" + source + "\n" + checks],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def feedback_source():
    return inline_constant("voice_chat", "BACKGROUND_ONLY_FEEDBACK_SOURCES") + inline_functions(
        "voice_chat", "isBackgroundOnlyVoiceFeedback", "handleVoiceInputFeedback"
    )


FEEDBACK_STATE = """
let currentAppState = 'speaking';
let isRecording = true;
let calls = [];
const spy = name => (...args) => calls.push([name, ...args]);
const hideSessionLoading = spy('hideLoading');
const hideThinking = spy('hideThinking');
const showVoiceFeedbackToast = spy('toast');
const showSpeakerRejectedToast = spy('speakerToast');
const flashPassiveStatus = spy('flash');
const schedulePassiveStateRestore = spy('restore');
function transitionTo(state, ...args) {
    calls.push(['transition', state, ...args]);
    currentAppState = state;
}
"""


@pytest.mark.parametrize("page", PAGES)
def test_entrypoint_branding_and_legacy_ui_are_clean(page):
    html = (STATIC / f"{page}.html").read_text(encoding="utf-8")
    assert "心语陪伴" in re.search(r"<title>(.*?)</title>", html).group(1)
    for removed in (
        "记忆守护", "Memory Guardian", "Clinical AI", "认知筛查", "MMSE", "MoCA",
        "阿尔茨", "患者", "医生", "医院", "病房", "诊室", "旧系统",
        "data-cognitive-only", "/v1/screening", "/v1/vision",
    ):
        assert removed not in html, (page, removed)
    elements = PageElements(html).elements
    ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
    assert len(ids) == len(set(ids)), f"Duplicate DOM ids in {page}"


def test_short_interrupt_and_background_feedback_never_touch_active_output():
    run_javascript(feedback_source(), FEEDBACK_STATE + """
for (const source of ['interrupt', 'interrupt_buffer', 'interrupt_complete', 'interrupt_future', 'processing_complete']) {
    for (const state of ['idle', 'listening', 'processing', 'speaking']) {
        for (const recording of [false, true]) {
            calls = [];
            currentAppState = state;
            isRecording = recording;
            handleVoiceInputFeedback({
                reason: 'speech_too_short', source,
                message: '刚才这句太短或不够清晰，没有触发打断。您可以再完整说一遍。',
                duration_s: 0.2,
            });
            assert.deepEqual(calls, [], source + '/' + state);
            assert.equal(currentAppState, state);
        }
    }
}
""")


def test_short_main_input_is_silent_but_releases_processing_state():
    run_javascript(feedback_source(), FEEDBACK_STATE + """
for (const source of [undefined, 'main', 'live']) {
    for (const state of ['idle', 'listening', 'processing', 'speaking']) {
        for (const recording of [false, true]) {
            calls = [];
            currentAppState = state;
            isRecording = recording;
            handleVoiceInputFeedback({reason: 'speech_too_short', source, status_text: '请再说一遍'});
            const fallback = recording ? 'listening' : 'idle';
            const expected = state === 'processing'
                ? [['hideLoading'], ['hideThinking'], ['transition', fallback]]
                : [];
            assert.deepEqual(calls, expected, source + '/' + state);
            assert.equal(currentAppState, state === 'processing' ? fallback : state);
        }
    }
}
""")


@pytest.mark.parametrize("reason", ["asr_empty", "audio_too_quiet", "speaker_non_target", "unexpected"])
def test_actionable_voice_feedback_remains_visible(reason):
    run_javascript(feedback_source(), FEEDBACK_STATE + f"const reason = {json.dumps(reason)};\n" + """
for (const source of ['main', 'interrupt_buffer']) {
    calls = [];
    currentAppState = 'processing';
    handleVoiceInputFeedback({reason, source, message: '需要处理的错误', duration_s: 1, similarity: 0.1});
    assert.equal(calls[0][0], reason === 'speaker_non_target' ? 'speakerToast' : 'toast');
    assert.ok(calls.some(call => call[0] === (source === 'main' ? 'transition' : 'flash')));
    assert.equal(currentAppState, source === 'main' ? 'listening' : 'processing');
}
""")


def test_legacy_screening_events_return_before_any_media_or_state_handling():
    source = inline_constant("voice_chat", "LEGACY_SCREENING_EVENT_TYPES") + inline_functions("voice_chat", "handleMessage")
    run_javascript(source, """
const calls = [];
const console = {log: () => calls.push('log')};
function recordProtocolEvent() { calls.push('protocol'); }
function acceptOutputGeneration() { calls.push('generation'); return true; }
for (const type of [
    'update_score', 'show_image', 'hide_image', 'vision_capture', 'vision_stop',
    'drawing_evaluating', 'drawing_result', 'eval_result', 'image_display',
]) {
    handleMessage({type, mode: 'cognitive', audio: 'stale', playback_id: 'old'});
    assert.deepEqual(calls, [], type);
}
// Only the legacy events are ignored; other events still enter the normal dispatcher.
handleMessage({type: 'unknown_future_event'});
assert.deepEqual(calls, ['log', 'protocol', 'generation']);
""")


def test_profile_fields_are_optional_and_do_not_collect_clinical_defaults():
    html = (STATIC / "voice_chat.html").read_text(encoding="utf-8")
    elements = PageElements(html).elements
    ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
    age_input = next(attrs for _, attrs in elements if attrs.get("id") == "patient-age")
    assert not age_input.get("value")
    assert "education_years" not in html
    assert "age: 70" not in html
    source = inline_functions(
        "voice_chat", "getProfileFromForm", "applyProfileToForm", "updateSettingsProfileSummary",
        "validateRequiredProfileFields", "selectPatientSource"
    )
    run_javascript(source, "const ids = " + json.dumps(ids) + ";\n" + """
const elements = Object.fromEntries(ids.map(id => [id, {
    _value: '',
    get value() { return this._value; },
    set value(value) { this._value = String(value); },
    classList: {add() {}, remove() {}, toggle() {}},
    setAttribute() {}, focus() {},
}]));
const document = {getElementById(id) { assert.ok(id in elements, 'Missing DOM element: ' + id); return elements[id]; }};
let patientSourceMode = 'new';
let selectedPatientId = '';
let currentPatientId = null;
let sessionStarted = false;
const patientOptions = [];
function clearRequiredProfileMarkers() {}
function clearSavedPatient() { currentPatientId = null; }
function setPatientSelectorStatus() {}
function loadPatientOptions() {}
function renderPatientOptions() {}
applyProfileToForm();
assert.deepEqual(getProfileFromForm(), {});
assert.equal(validateRequiredProfileFields(), true);
applyProfileToForm({name: ' 小林 ', age: 35, gender: '男', city: '杭州', education_years: 6, hospital_name: '旧字段'});
assert.deepEqual(getProfileFromForm(), {name: '小林', age: '35', gender: '男', city: '杭州'});
patientSourceMode = 'existing';
selectedPatientId = 'pt-existing';
assert.equal(getProfileFromForm().patient_id, 'pt-existing');
for (const mode of ['new', 'existing']) {
    selectPatientSource(mode);
    assert.deepEqual(getProfileFromForm(), {});
    assert.equal(elements['patient-age'].value, '');
    assert.equal(elements['patient-sex'].value, '');
}
for (const age of ['', '1', '120', '35']) {
    elements['patient-age'].value = age;
    assert.equal(validateRequiredProfileFields(), true, age);
}
for (const age of ['0', '-1', '121', 'not-a-number']) {
    elements['patient-age'].value = age;
    assert.equal(validateRequiredProfileFields(), false, age);
}
""")


def test_history_keeps_old_dialogue_and_audio_without_rendering_or_exporting_scores():
    html = (STATIC / "history.html").read_text(encoding="utf-8")
    ids = [attrs["id"] for _, attrs in PageElements(html).elements if "id" in attrs]
    source = inline_functions(
        "history", "normalizePatientName", "sessionMode", "modeLabel", "escapeHtml",
        "formatDateTime", "formatDuration", "resolveApiUrl", "renderDetail", "renderMessages",
        "renderAudio", "buildTxtContent", "buildCsvContent",
    )
    run_javascript(source, "const ids = " + json.dumps(ids) + ";\n" + """
const elements = Object.fromEntries(ids.map(id => [id, {innerHTML: '', textContent: '', disabled: false}]));
const document = {getElementById(id) { assert.ok(id in elements, 'Missing DOM element: ' + id); return elements[id]; }};
const API_BASE = '';
const window = {location: {origin: 'https://companion.example'}};
const data = {
    session: {session_id: 'old-session', mode: 'cognitive', patient_name: '小林', mmse_total: 19},
    messages: [{role: 'user', content: '今天想聊聊 <生活>'}, {role: 'assistant', content: '我在听。'}],
    audio_files: [{role: 'user', audio_url: '/api/sessions/old-session/audio/1', duration_s: 2}],
    score_records: [{dimension: '记忆力', score: 3}],
};
renderDetail(data);
assert.equal(elements['metric-mode'].textContent, '历史会话');
assert.equal(elements['metric-messages'].textContent, 2);
assert.equal(elements['metric-audio'].textContent, 1);
assert.equal(elements['resume-btn'].disabled, true);
assert.ok(elements['tab-messages'].innerHTML.includes('今天想聊聊 &lt;生活&gt;'));
assert.ok(elements['tab-audio'].innerHTML.includes('/api/sessions/old-session/audio/1'));
for (const output of [buildTxtContent(data), buildCsvContent(data)]) {
    assert.ok(output.includes('今天想聊聊 <生活>'));
    assert.ok(output.includes('我在听。'));
    assert.doesNotMatch(output, /MMSE|评分|记忆力|认知筛查|score_records/);
}
data.session.mode = 'wellbeing';
renderDetail(data);
assert.equal(elements['metric-mode'].textContent, '心理健康陪伴');
assert.equal(elements['resume-btn'].disabled, false);
data.session.ended_at = '2026-09-13T20:00:00';
renderDetail(data);
assert.equal(elements['resume-btn'].disabled, true);
""")


def test_history_csv_preserves_quotes_newlines_and_neutralizes_formulas():
    contents = ['普通对话', '含有,逗号和"引号"\n换行', '=1+1', '+cmd', '-1', '@sum(A1)', '\t=1', '\r=1']
    payload = {"messages": [{"role": "user", "content": value} for value in contents]}
    output = run_javascript(
        inline_functions("history", "buildCsvContent"),
        "process.stdout.write(JSON.stringify(buildCsvContent(" + json.dumps(payload) + ")));",
    )
    csv_text = json.loads(output)
    assert csv_text.startswith("\ufeff")
    rows = list(csv.reader(io.StringIO(csv_text.removeprefix("\ufeff"))))
    assert rows[0] == ["序号", "角色", "时间", "内容"]
    assert [row[3] for row in rows[1:]] == contents[:2] + ["'" + value for value in contents[2:]]
    assert [row[0] for row in rows[1:]] == [str(i) for i in range(1, len(contents) + 1)]
