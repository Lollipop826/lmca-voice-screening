"""Settings-panel contracts and shipped JavaScript behavior, without a live session."""

import re
from html.parser import HTMLParser
from pathlib import Path

from test_companion_frontend_cleanup import inline_functions, run_javascript


ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "static" / "voice_chat.html"
CSS = ROOT / "static" / "voice_chat_glass.css"


class SettingsElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs, tuple(self.stack)))
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, attrs.get("id"), attrs.get("class", "")))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                self.stack = self.stack[:i]
                break


def elements():
    parser = SettingsElements()
    parser.feed(HTML.read_text(encoding="utf-8"))
    return parser.elements


# A small deterministic DOM double. The tests execute the actual functions from
# voice_chat.html; they do not connect WebSocket, request media, or call an API.
DOM = r"""
let document;
const nodes = {};
const focusHistory = [];
class Element {
    constructor(id, attrs = {}, owner = null) {
        this.id = id;
        this.attrs = {...attrs};
        this.owner = owner;
        this._value = '';
        this.textContent = '';
        this.hidden = false;
        this.inert = false;
        this.disabled = false;
        this.visible = true;
        this.isConnected = true;
        this.tabIndex = 0;
        this.classes = new Set();
        this.classList = {
            contains: value => this.classes.has(value),
            add: value => this.classes.add(value),
            remove: value => this.classes.delete(value),
            toggle: (value, force) => {
                const next = force === undefined ? !this.classes.has(value) : force;
                if (next) this.classes.add(value); else this.classes.delete(value);
                return next;
            },
        };
        nodes[id] = this;
    }
    get value() { return this._value; }
    set value(value) { this._value = String(value); }
    getAttribute(name) { return this.attrs[name] ?? null; }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    closest(selector) {
        if (selector === '[role="tab"]') return this.attrs.role === 'tab' ? this : null;
        if ((selector.includes('[inert]') && this.inert) || (selector.includes('[hidden]') && this.hidden)) return this;
        return this.owner?.closest(selector) || null;
    }
    getClientRects() { return this.visible && !this.hidden ? [{}] : []; }
    focus() { document.activeElement = this; focusHistory.push(this.id); }
    click() { if (!this.disabled) this.onClick?.(); }
    contains(el) { return el === this || (el?.owner ? this.contains(el.owner) : false); }
    querySelector(selector) {
        const tabs = this.id === 'drawer' ? settingsTabs : memoryTabs;
        return tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || null;
    }
    querySelectorAll(selector) {
        if (selector === '[role="tab"]:not(:disabled)') return this.tabs.filter(tab => !tab.disabled);
        return (this.focusables || []).filter(el => !el.disabled);
    }
}
const drawer = new Element('drawer', {'aria-hidden': 'true'});
const memory = new Element('memory-panel', {'aria-hidden': 'true'});
drawer.inert = memory.inert = true;
new Element('drawer-overlay');
new Element('memory-overlay');
const trigger = new Element('drawer-toggle', {'aria-expanded': 'false'});
const settingsViews = ['profile', 'memory', 'preferences'].map(name => new Element(`settings-view-${name}`, {}, drawer));
const settingsTabs = ['profile', 'memory', 'preferences'].map(name => {
    const tab = new Element(`settings-tab-${name}`, {role: 'tab', 'aria-selected': String(name === 'profile')}, drawer);
    tab.tabIndex = name === 'profile' ? 0 : -1;
    tab.onClick = () => setSettingsTab(name);
    return tab;
});
const memoryViews = ['profile', 'longterm', 'emotion', 'session', 'turn'].map(name => new Element(`memory-view-${name}`, {}, memory));
const memoryTabs = ['profile', 'longterm', 'emotion', 'session', 'turn'].map(name => {
    const tab = new Element(`memory-tab-${name}`, {role: 'tab', 'aria-selected': String(name === 'longterm')}, memory);
    tab.tabIndex = name === 'longterm' ? 0 : -1;
    tab.onClick = () => setMemoryPanelView(name);
    return tab;
});
const settingsTablist = new Element('settings-tablist');
settingsTablist.tabs = settingsTabs;
const memoryTablist = new Element('memory-tablist');
memoryTablist.tabs = memoryTabs;
const settingsBody = {scrollTop: 125};
const memoryBody = {scrollTop: 125};
document = {
    body: new Element('body'),
    activeElement: trigger,
    getElementById: id => nodes[id] || null,
    querySelectorAll: selector => ({
        '.settings-view': settingsViews, '.settings-tab': settingsTabs,
        '.memory-panel-view': memoryViews, '.memory-panel-tab': memoryTabs,
    }[selector] || []),
    querySelector: selector => {
        if (selector === '#drawer .settings-body') return settingsBody;
        if (selector === '#memory-panel .mem-body') return memoryBody;
        if (selector === '#memory-panel.open, #drawer.open') return memory.classList.contains('open') ? memory : drawer.classList.contains('open') ? drawer : null;
        return null;
    },
};
for (const id of ['patient-name', 'patient-age', 'patient-sex', 'location-province', 'location-city', 'location-district',
    'settings-profile-initial', 'settings-profile-name', 'settings-profile-age', 'settings-profile-gender', 'settings-profile-location', 'settings-profile-source',
    'btn-model-mini', 'btn-model-lite', 'btn-model-current']) new Element(id);
let companionPanelReturnFocus = null;
let patientSourceMode = 'existing';
let selectedPatientId = '';
let activeMemoryPanelView = 'longterm';
let lastMemorySnapshot = null;
const refreshes = [];
function refreshMemoryProfile() { refreshes.push('profile'); }
function refreshMemoryItems() { refreshes.push('longterm'); }
function refreshEmotionTrajectory() { refreshes.push('emotion'); }
function renderMemoryPanel(snapshot) { refreshes.push('session'); }
function event(key, target = null, currentTarget = null, extra = {}) {
    return {key, target, currentTarget, prevented: false, preventDefault() {this.prevented = true;}, ...extra};
}
"""


def panel_script():
    return DOM + inline_functions(
        "voice_chat", "setSettingsTab", "updateSettingsProfileSummary", "setCompanionPanelOpen",
        "toggleDrawer", "openSettingsMemory", "handleCompanionTabKeydown", "handleCompanionPanelKeydown",
        "setMemoryPanelView", "toggleMemoryPanel",
    )


def test_settings_structure_keeps_three_real_tabs_and_a_separate_footer():
    rows = elements()
    by_id = {attrs["id"]: (tag, attrs, parents) for tag, attrs, parents in rows if "id" in attrs}
    ids = [attrs["id"] for _, attrs, _ in rows if "id" in attrs]
    assert len(ids) == len(set(ids))
    assert by_id["drawer"][1]["aria-labelledby"] == "drawer-title"
    assert by_id["drawer"][1]["role"] == "dialog"
    assert by_id["drawer"][1]["aria-modal"] == "true"
    assert "inert" in by_id["drawer"][1]
    assert by_id["drawer-toggle"][1]["aria-expanded"] == "false"
    for name in ("profile", "memory", "preferences"):
        tab = by_id[f"settings-tab-{name}"][1]
        panel = by_id[f"settings-view-{name}"][1]
        assert tab["role"] == "tab"
        assert tab["aria-controls"] == panel["id"]
        assert panel["aria-labelledby"] == tab["id"]
        assert tab["aria-selected"] == str(name == "profile").lower()
        assert ("hidden" in panel) == (name != "profile")
    footer = next(row for row in rows if row[1].get("class") == "settings-action-panel")
    assert footer[2][-1][1] == "drawer"
    for identifier in ("start-btn", "long-term-memory-toggle"):
        assert any("settings-action-panel" in ancestor[2] for ancestor in by_id[identifier][2])
        assert not any("settings-body" in ancestor[2] for ancestor in by_id[identifier][2])
    assert "hidden" in by_id["memory-view-profile"][1]
    assert by_id["memory-panel"][1]["aria-labelledby"] == "memory-panel-title"


def test_existing_profile_voice_and_memory_controls_remain_accessible():
    rows = elements()
    by_id = {attrs["id"]: attrs for _, attrs, _ in rows if "id" in attrs}
    for identifier, handler in {
        "start-btn": "startSession()",
        "patient-list-refresh": "loadPatientOptions(true)",
        "tts-preview-btn": "previewTtsVoice()",
        "enroll-btn": "toggleEnrollRecording()",
        "save-speaker-btn": "saveSpeaker()",
        "btn-model-mini": "switchModel('doubao-seed-2-0-mini-260215')",
        "btn-model-lite": "switchModel('doubao-seed-2-0-lite-260215')",
    }.items():
        assert by_id[identifier]["onclick"] == handler
    for identifier in ("patient-name", "patient-age", "patient-sex", "location-province", "location-city", "location-district", "speaker-name-input", "saved-speakers", "tts-voice-select", "threshold-slider"):
        assert any(tag == "label" and attrs.get("for") == identifier for tag, attrs, _ in rows)
    assert "checked" in by_id["long-term-memory-toggle"]
    assert "display:none" in by_id["drawer-admin-link"]["style"]
    assert by_id["drawer-admin-link"]["href"] == "/admin"
    assert all(f"sample-{i}" in by_id for i in range(1, 9))
    details = next(attrs for tag, attrs, _ in rows if tag == "details" and attrs.get("class") == "settings-advanced")
    assert "open" not in details


def test_settings_tabs_update_selection_without_resetting_profile():
    run_javascript(panel_script(), r"""
nodes['patient-name'].value = '小林';
for (const name of ['memory', 'preferences', 'profile']) {
    setSettingsTab(name);
    assert.equal(settingsBody.scrollTop, 0);
    for (const tab of settingsTabs) {
        const selected = tab.id === `settings-tab-${name}`;
        assert.equal(tab.getAttribute('aria-selected'), String(selected));
        assert.equal(tab.tabIndex, selected ? 0 : -1);
        assert.equal(tab.classList.contains('active'), selected);
    }
    assert.deepEqual(settingsViews.filter(v => !v.hidden).map(v => v.id), [`settings-view-${name}`]);
    assert.equal(nodes['patient-name'].value, '小林');
}
setSettingsTab('unknown');
assert.equal(nodes['settings-view-profile'].hidden, false);
""")


def test_both_tab_strips_support_keyboard_navigation_and_wraparound():
    run_javascript(panel_script(), r"""
for (const [list, tabs] of [[settingsTablist, settingsTabs], [memoryTablist, memoryTabs]]) {
    let e = event('ArrowLeft', tabs[0], list);
    handleCompanionTabKeydown(e);
    assert.equal(document.activeElement, tabs.at(-1));
    assert.equal(tabs.at(-1).getAttribute('aria-selected'), 'true');
    assert.equal(e.prevented, true);
    e = event('ArrowRight', tabs.at(-1), list);
    handleCompanionTabKeydown(e);
    assert.equal(document.activeElement, tabs[0]);
    handleCompanionTabKeydown(event('End', tabs[0], list));
    assert.equal(document.activeElement, tabs.at(-1));
    handleCompanionTabKeydown(event('Home', tabs.at(-1), list));
    assert.equal(document.activeElement, tabs[0]);
    e = event('Tab', tabs[0], list);
    handleCompanionTabKeydown(e);
    assert.equal(e.prevented, false);
    e = event('ArrowRight', tabs[0], list, {ctrlKey: true});
    handleCompanionTabKeydown(e);
    assert.equal(e.prevented, false);
}
settingsTabs[1].disabled = true;
handleCompanionTabKeydown(event('ArrowRight', settingsTabs[0], settingsTablist));
assert.equal(document.activeElement, settingsTabs[2]);
""")


def test_dialogs_are_mutually_exclusive_and_restore_the_original_focus():
    run_javascript(panel_script(), r"""
toggleDrawer(true);
assert.equal(drawer.classList.contains('open'), true);
assert.equal(drawer.inert, false);
assert.equal(memory.inert, true);
assert.equal(trigger.getAttribute('aria-expanded'), 'true');
assert.equal(document.activeElement.id, 'settings-tab-profile');
setSettingsTab('preferences');
toggleDrawer(true); // Idempotent: a reconnect must not close an open drawer.
assert.equal(drawer.classList.contains('open'), true);
assert.equal(document.activeElement.id, 'settings-tab-preferences');
toggleMemoryPanel(true, 'emotion');
assert.equal(drawer.inert, true);
assert.equal(drawer.classList.contains('open'), false);
assert.equal(nodes['drawer-overlay'].classList.contains('open'), false);
assert.equal(memory.inert, false);
assert.equal(memory.classList.contains('expanded'), true);
assert.equal(nodes['memory-overlay'].classList.contains('open'), true);
assert.equal(document.body.classList.contains('memory-sidebar-open'), true);
assert.equal(document.activeElement.id, 'memory-tab-emotion');
assert.equal(activeMemoryPanelView, 'emotion');
assert.deepEqual(refreshes, ['emotion']);
openSettingsMemory();
assert.equal(memory.inert, true);
assert.equal(nodes['memory-overlay'].classList.contains('open'), false);
assert.equal(document.activeElement.id, 'settings-tab-memory');
toggleDrawer(false);
assert.equal(drawer.inert, true);
assert.equal(drawer.getAttribute('aria-hidden'), 'true');
assert.equal(trigger.getAttribute('aria-expanded'), 'false');
assert.equal(document.activeElement, trigger);
const before = focusHistory.length;
toggleDrawer(false);
assert.equal(focusHistory.length, before);
""")


def test_escape_closes_only_the_open_panel_and_focus_return_has_a_fallback():
    run_javascript(panel_script(), r"""
const outside = new Element('outside-trigger');
document.activeElement = outside;
toggleMemoryPanel(true, 'session');
const e = event('Escape');
handleCompanionPanelKeydown(e);
assert.equal(e.prevented, true);
assert.equal(memory.classList.contains('open'), false);
assert.equal(document.activeElement, outside);
const noop = event('Escape');
handleCompanionPanelKeydown(noop);
assert.equal(noop.prevented, false);
for (const previous of [document.body, outside]) {
    document.activeElement = previous;
    toggleDrawer(true);
    outside.isConnected = false;
    handleCompanionPanelKeydown(event('Escape'));
    assert.equal(document.activeElement, trigger);
}
""")


def test_focus_is_trapped_without_including_hidden_or_disabled_controls():
    run_javascript(panel_script(), r"""
toggleDrawer(true);
const first = new Element('first', {}, drawer);
const last = new Element('last', {}, drawer);
const hidden = new Element('hidden', {}, drawer); hidden.hidden = true;
const disabled = new Element('disabled', {}, drawer); disabled.disabled = true;
const inactiveTab = new Element('inactive-tab', {}, drawer); inactiveTab.tabIndex = -1;
drawer.focusables = [hidden, first, disabled, inactiveTab, last];
last.focus();
let e = event('Tab');
handleCompanionPanelKeydown(e);
assert.equal(document.activeElement, first);
assert.equal(e.prevented, true);
e = event('Tab', null, null, {shiftKey: true});
handleCompanionPanelKeydown(e);
assert.equal(document.activeElement, last);
assert.equal(e.prevented, true);
trigger.focus();
handleCompanionPanelKeydown(event('Tab'));
assert.equal(document.activeElement, first);
first.focus();
e = event('Tab');
handleCompanionPanelKeydown(e);
assert.equal(e.prevented, false); // Native Tab moves between interior controls.
drawer.focusables = [];
handleCompanionPanelKeydown(event('Tab'));
assert.equal(document.activeElement, drawer);
""")


def test_profile_overview_tracks_real_data_resets_and_detected_location():
    source = DOM + inline_functions("voice_chat", "updateSettingsProfileSummary", "applyProfileToForm", "prefillDetectedLocation")
    run_javascript(source, r"""
applyProfileToForm();
assert.equal(nodes['settings-profile-name'].textContent, '未填写称呼');
assert.equal(nodes['settings-profile-age'].textContent, '未填写');
assert.equal(nodes['settings-profile-gender'].textContent, '未填写');
assert.equal(nodes['settings-profile-source'].textContent, '未选档案');
selectedPatientId = 'profile-existing';
applyProfileToForm({name: ' 小林 ', age: 35, gender: '男', province: '浙江省', city: '杭州市'});
assert.equal(nodes['settings-profile-name'].textContent, '小林');
assert.equal(nodes['settings-profile-initial'].textContent, '小');
assert.equal(nodes['settings-profile-age'].textContent, '35 岁');
assert.equal(nodes['settings-profile-source'].textContent, '已有档案');
assert.equal(nodes['settings-profile-location'].textContent, '浙江省 · 杭州市');
prefillDetectedLocation({province: '其他省', city: '其他市', district: '西湖区'});
assert.equal(nodes['settings-profile-location'].textContent, '浙江省 · 杭州市 · 西湖区');
nodes['patient-name'].value = '<img src=x onerror=alert(1)>';
updateSettingsProfileSummary();
assert.equal(nodes['settings-profile-name'].textContent, '<img src=x onerror=alert(1)>');
assert.equal(nodes['settings-profile-name'].innerHTML, undefined);
patientSourceMode = 'new';
selectedPatientId = '';
applyProfileToForm();
assert.equal(nodes['settings-profile-source'].textContent, '新档案');
assert.equal(nodes['settings-profile-age'].textContent, '未填写');
assert.equal(nodes['settings-profile-gender'].textContent, '未填写');
prefillDetectedLocation({province: '北京市', city: '北京市'});
assert.equal(nodes['settings-profile-location'].textContent, '北京市');
""")
    source = HTML.read_text(encoding="utf-8")
    for event_name in ("input", "change"):
        assert f"document.querySelector('.settings-profile-fields').addEventListener('{event_name}', updateSettingsProfileSummary)" in source


def test_model_selection_uses_accessible_state_instead_of_inline_colors():
    run_javascript(DOM + inline_functions("voice_chat", "updateModelUI"), r"""
updateModelUI('doubao-seed-2-0-mini-260215');
assert.equal(nodes['btn-model-mini'].getAttribute('aria-pressed'), 'true');
assert.equal(nodes['btn-model-lite'].getAttribute('aria-pressed'), 'false');
updateModelUI('doubao-seed-2-0-lite-260215');
assert.equal(nodes['btn-model-mini'].getAttribute('aria-pressed'), 'false');
assert.equal(nodes['btn-model-lite'].getAttribute('aria-pressed'), 'true');
updateModelUI('qwen3.7-flash', ['qwen3.7-flash']);
assert.equal(nodes['btn-model-mini'].hidden, true);
assert.equal(nodes['btn-model-lite'].hidden, true);
assert.equal(nodes['btn-model-mini'].getAttribute('aria-pressed'), 'false');
assert.equal(nodes['btn-model-lite'].getAttribute('aria-pressed'), 'false');
assert.equal(nodes['btn-model-current'].hidden, false);
assert.equal(nodes['btn-model-current'].textContent, 'Qwen3.7 Flash');
assert.equal(nodes['btn-model-current'].getAttribute('aria-pressed'), 'true');
updateModelUI('doubao-seed-2-0-mini-260215');
assert.equal(nodes['btn-model-mini'].hidden, false);
assert.equal(nodes['btn-model-lite'].hidden, false);
assert.equal(nodes['btn-model-current'].hidden, true);
""")
    assert '#drawer #model-switcher button[aria-pressed="true"]' in CSS.read_text(encoding="utf-8")


def test_settings_theme_keeps_internal_scrolling_safe_areas_and_reduced_motion():
    css = CSS.read_text(encoding="utf-8")
    body = re.search(r"#drawer > \.settings-body\s*\{([^}]+)\}", css).group(1)
    footer = re.search(r"#drawer > \.settings-action-panel\s*\{([^}]+)\}", css).group(1)
    assert "min-height: 0" in body
    assert "overflow-y: auto" in body
    assert "padding: 0 var(--settings-padding) 24px !important" in body
    assert "flex: 0 0 auto" in footer
    assert "margin: 0 !important" in footer
    assert "env(safe-area-inset-bottom)" in footer
    assert '#drawer .settings-profile-fields .input.required-missing' in css
    assert "@media (max-width: 359px)" in css
    reduced = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    assert "#drawer," in reduced and "#memory-panel," in reduced
    assert "transition: none !important" in reduced
    transparency = css[css.index("@media (prefers-reduced-transparency: reduce)"):]
    assert "#memory-panel," in transparency
    assert '/static/voice_chat_glass.css?v=20260914-unified-1' in HTML.read_text(encoding="utf-8")
