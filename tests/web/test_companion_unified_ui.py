"""Shared companion theme contracts and shipped JS behavior (no live services).

These DOM doubles check interaction state, not browser layout or native focus trapping.
"""

import json
import re
from urllib.parse import urlsplit

import pytest

from test_companion_frontend_cleanup import (
    STATIC,
    PageElements,
    inline_functions,
    run_javascript,
)


PAGES = (
    "voice_chat", "login", "history", "admin", "api_portal",
    "test_camera", "test_3step_action_demo",
)
SHARED_JS = (STATIC / "companion_ui.js").read_text(encoding="utf-8")
SHARED_CSS = (STATIC / "companion_ui.css").read_text(encoding="utf-8")

# Just enough DOM to execute the production functions; no copied UI implementation.
DOM = r"""
const documentEvents = {};
let modalUnsupported = false;
class Element {
    constructor(tag = 'div') {
        this.tagName = tag; this.attributes = {}; this.children = []; this.parentElement = null;
        this.listeners = {}; this.style = {}; this.dataset = {}; this.className = '';
        this.hidden = false; this.disabled = false; this.inert = false; this.visible = true;
        this.open = false; this.tabIndex = 0; this.focusCount = 0; this.clickCount = 0;
        this.resetCount = 0; this.value = ''; this.returnValue = ''; this._text = ''; this._html = '';
        this.classList = {
            contains: value => this.className.split(/\s+/).includes(value),
            add: (...values) => { this.className = [...new Set([...this.className.split(/\s+/), ...values])].filter(Boolean).join(' '); },
            remove: (...values) => { this.className = this.className.split(/\s+/).filter(value => !values.includes(value)).join(' '); },
            toggle: (value, force) => {
                const on = force === undefined ? !this.classList.contains(value) : force;
                on ? this.classList.add(value) : this.classList.remove(value);
                return on;
            },
        };
    }
    get isConnected() { return this === document.body || document.body.contains(this); }
    get textContent() { return this._text; }
    set textContent(value) { this.clear(); this._text = String(value); }
    get innerHTML() { return this._html; }
    set innerHTML(value) { this.clear(); this._html = value; }
    clear() { for (const child of this.children) child.parentElement = null; this.children = []; }
    setAttribute(name, value) {
        this.attributes[name] = String(value);
        if (name === 'id') this.id = value;
        if (name === 'class') this.className = value;
        if (['hidden', 'inert', 'disabled', 'open'].includes(name)) this[name] = true;
    }
    getAttribute(name) {
        if (name === 'id') return this.id || null;
        if (name === 'class') return this.className;
        return this.attributes[name] ?? null;
    }
    removeAttribute(name) { delete this.attributes[name]; if (['hidden', 'inert', 'open'].includes(name)) this[name] = false; }
    append(...nodes) { for (const node of nodes) { node.remove(); node.parentElement = this; this.children.push(node); } }
    appendChild(node) { this.append(node); return node; }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this); this.parentElement = null; }
    contains(node) { return this === node || this.children.some(child => child.contains(node)); }
    matches(selector) {
        return selector.split(',').some(part => {
            let value = part.trim();
            const not = value.match(/:not\(([^)]+)\)/);
            if (not) { if (this.matches(not[1])) return false; value = value.replace(not[0], ''); }
            if (value === ':disabled') return this.disabled;
            const tag = value.match(/^[a-z][\w-]*/i);
            if (tag && tag[0] !== this.tagName) return false;
            const id = value.match(/#([\w-]+)/);
            if (id && this.id !== id[1]) return false;
            if ([...value.matchAll(/\.([\w-]+)/g)].some(match => !this.classList.contains(match[1]))) return false;
            return [...value.matchAll(/\[([\w-]+)(?:=["']?([^\]"']+)["']?)?\]/g)].every(([, name, expected]) => {
                const actual = ['hidden', 'inert', 'disabled', 'open'].includes(name) ? (this[name] ? '' : null) : this.getAttribute(name);
                return expected === undefined ? actual !== null : actual === expected;
            });
        });
    }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    getClientRects() { return this.visible && !this.closest('[hidden], [inert]') ? [{}] : []; }
    getBoundingClientRect() { return {left: 50, top: 50, right: 450, bottom: 350}; }
    focus() { document.activeElement = this; this.focusCount++; }
    click() { this.clickCount++; this.dispatch('click'); }
    reset() { this.resetCount++; }
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
    dispatch(type, options = {}) {
        const event = keyEvent(undefined, this, options);
        for (const listener of this.listeners[type] || []) listener(event);
        return event;
    }
    showModal() { if (modalUnsupported) throw new Error('unsupported'); this.open = true; }
    close(value = '') { this.open = false; this.returnValue = value; this.dispatch('close'); }
}
const document = {
    body: null, activeElement: null,
    createElement: tag => new Element(tag),
    getElementById: id => document.body.querySelector('#' + id),
    querySelector: selector => document.body.querySelector(selector),
    querySelectorAll: selector => document.body.querySelectorAll(selector),
    addEventListener: (type, listener) => (documentEvents[type] ||= []).push(listener),
};
document.body = new Element('body');
document.activeElement = document.body;
const window = {};
function element(tag, id, attributes = {}, parent = document.body) {
    const node = new Element(tag);
    if (id) node.setAttribute('id', id);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    parent.append(node);
    return node;
}
function keyEvent(key, target, extra = {}) {
    return {key, target, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra};
}
function dispatchKey(event) { for (const listener of documentEvents.keydown || []) listener(event); }
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; }
"""


def run_async(source, checks):
    run_javascript("", """
const watchdog = setTimeout(() => { throw new Error('Async UI test did not finish'); }, 5000);
(async () => {
""" + source + "\n" + checks + """
})().then(() => clearTimeout(watchdog), error => { clearTimeout(watchdog); console.error(error); process.exitCode = 1; });
""")


@pytest.mark.parametrize("page", PAGES)
def test_all_entrypoints_use_local_shared_assets_and_valid_aria_targets(page):
    html = (STATIC / f"{page}.html").read_text(encoding="utf-8")
    rows = PageElements(html).elements
    ids = [attrs["id"] for _, attrs in rows if "id" in attrs]
    assert len(ids) == len(set(ids))
    assert "心语陪伴" in re.search(r"<title>(.*?)</title>", html).group(1)
    for name, tag, attr in (("css", "link", "href"), ("js", "script", "src")):
        url = next(attrs[attr] for item, attrs in rows if item == tag and f"/static/companion_ui.{name}" in attrs.get(attr, ""))
        assert urlsplit(url).query == "v=20260914-unified-1"
        assert (STATIC / f"companion_ui.{name}").is_file()
    assert any(tag == "meta" and attrs.get("name") == "theme-color" and attrs["content"] == "#edf5ef" for tag, attrs in rows)
    assert "/static/vendor/fontawesome/css/all.min.css" in html
    assert "fonts.googleapis.com" not in html
    for _, attrs in rows:
        for key in ("aria-controls", "aria-labelledby", "aria-describedby"):
            for target in attrs.get(key, "").split():
                assert target in ids, (page, key, target)
    if page != "voice_chat":
        assert "<style" not in html
        assert any(tag == "main" for tag, _ in rows)
        assert any(tag == "a" and attrs.get("class") == "ui-skip-link" for tag, attrs in rows)


def test_shared_theme_covers_controls_feedback_mobile_and_accessibility():
    for token in ("ink", "muted", "accent", "surface", "line", "success", "warning", "danger", "font", "primary-fill", "panel-fill"):
        assert f"--ui-{token}:" in SHARED_CSS
    for selector in (".login-page", ".history-page", ".admin-page", ".api-page", ".diagnostics-page", ".companion-dialog"):
        assert selector in SHARED_CSS
    for contract in ("min-height: 44px", ":focus-visible", "[hidden]", "prefers-reduced-motion", "prefers-reduced-transparency", "forced-colors", "max-width: 700px", "max-height: 760px", "overflow-y: auto"):
        assert contract in SHARED_CSS
    # A connection pill also gets a loading class. It must not inherit empty-state padding.
    assert ".loading:not(.connection-pill)" in SHARED_CSS
    assert ".permission-copy" in SHARED_CSS
    assert ".modal-feedback.flash-error" in SHARED_CSS
    for page in ("voice_chat", "history", "api_portal"):
        html = (STATIC / f"{page}.html").read_text(encoding="utf-8")
        assert not re.search(r"(?<![.\w])(?:alert|confirm)\(", html)


@pytest.mark.parametrize("action", ["confirm", "cancel", "escape", "backdrop"])
def test_shared_confirmation_is_safe_named_and_restores_focus(action):
    run_async(DOM + SHARED_JS, "const action = " + json.dumps(action) + r""";
const trigger = element('button', 'trigger'); trigger.focus();
const result = window.CompanionUI.confirm({title: '删除记录？', message: '<img src=x onerror=alert(1)>', confirmText: '删除', destructive: true});
const dialog = document.querySelector('dialog');
assert.ok(dialog.open);
assert.equal(dialog.getAttribute('aria-labelledby'), dialog.querySelector('h2').id);
assert.equal(dialog.querySelector('p').textContent, '<img src=x onerror=alert(1)>');
assert.equal(dialog.querySelector('p').children.length, 0);
const buttons = dialog.querySelectorAll('button');
assert.equal(buttons.length, 2);
assert.equal(document.activeElement.value, 'cancel');
assert.ok(buttons[1].classList.contains('ui-button-danger'));
// Clicking empty space inside the panel must not be treated as clicking its backdrop.
dialog.dispatch('click', {clientX: 100, clientY: 100});
assert.ok(dialog.open);
if (action === 'escape') assert.ok(dialog.dispatch('cancel').defaultPrevented);
else if (action === 'backdrop') dialog.dispatch('click', {clientX: 5, clientY: 5});
else dialog.close(action);
assert.equal(await result, action === 'confirm');
assert.equal(document.querySelector('dialog'), null);
assert.equal(document.activeElement, trigger);
assert.equal(trigger.focusCount, 2);
""")


def test_shared_alert_concurrent_calls_and_unsupported_dialog_fail_closed():
    run_async(DOM + SHARED_JS, r"""
const first = window.CompanionUI.alert({message: '连接失败，请重试'});
const dialog = document.querySelector('dialog');
assert.equal(dialog.querySelectorAll('button').length, 1);
assert.equal(document.activeElement.value, 'confirm');
assert.equal(await window.CompanionUI.confirm({message: 'second request'}), false);
assert.equal(document.querySelectorAll('dialog').length, 1);
dialog.close('confirm');
assert.equal(await first, true);
modalUnsupported = true;
assert.equal(await window.CompanionUI.confirm({destructive: true}), false);
assert.equal(document.querySelector('dialog'), null);
modalUnsupported = false;
const recovered = window.CompanionUI.confirm();
document.querySelector('dialog').close('cancel');
assert.equal(await recovered, false);
assert.ok(Object.isFrozen(window.CompanionUI));
""")


@pytest.mark.parametrize("unavailable", ["removed", "disabled", "hidden", "inert", "invisible"])
def test_shared_dialog_does_not_restore_focus_into_unavailable_content(unavailable):
    run_async(DOM + SHARED_JS, "const unavailable = " + json.dumps(unavailable) + r""";
const wrapper = element('div', 'wrapper');
const trigger = element('button', 'trigger', {}, wrapper); trigger.focus();
const result = window.CompanionUI.confirm();
const dialog = document.querySelector('dialog');
if (unavailable === 'removed') trigger.remove();
if (unavailable === 'disabled') trigger.disabled = true;
if (unavailable === 'hidden') wrapper.hidden = true;
if (unavailable === 'inert') wrapper.inert = true;
if (unavailable === 'invisible') trigger.visible = false;
dialog.close('cancel'); await result;
assert.equal(trigger.focusCount, 1);
""")


def test_shared_keyboard_tabs_skip_unavailable_tabs_and_respect_modifiers():
    run_javascript(DOM + SHARED_JS, r"""
const group = element('div', 'tabs', {'data-keyboard-tabs': '', role: 'tablist'});
const tabs = Array.from({length: 6}, (_, i) => element('button', 'tab-' + i, {role: 'tab'}, group));
tabs[1].disabled = true; tabs[2].hidden = true; tabs[3].setAttribute('aria-disabled', 'true');
const hiddenParent = element('div', 'hidden-parent', {inert: ''}, group); hiddenParent.append(tabs[4]);
let event = keyEvent('ArrowRight', tabs[0]); dispatchKey(event);
assert.ok(event.defaultPrevented); assert.equal(document.activeElement, tabs[5]);
event = keyEvent('ArrowRight', tabs[5]); dispatchKey(event); assert.equal(document.activeElement, tabs[0]);
event = keyEvent('ArrowLeft', tabs[0]); dispatchKey(event); assert.equal(document.activeElement, tabs[5]);
for (const [key, expected] of [['Home', tabs[0]], ['End', tabs[5]]]) { dispatchKey(keyEvent(key, tabs[5])); assert.equal(document.activeElement, expected); }
for (const modifier of ['altKey', 'ctrlKey', 'metaKey', 'defaultPrevented']) {
    const before = tabs.map(tab => tab.clickCount);
    dispatchKey(keyEvent('ArrowRight', tabs[0], {[modifier]: true}));
    assert.deepEqual(tabs.map(tab => tab.clickCount), before);
}
event = keyEvent('ArrowDown', tabs[0]); dispatchKey(event); assert.equal(event.defaultPrevented, false);
dispatchKey(keyEvent('ArrowRight', {}));
const nestedGroup = element('div', 'nested', {'data-keyboard-tabs': ''}, group);
element('button', 'nested-tab', {role: 'tab'}, nestedGroup);
dispatchKey(keyEvent('End', tabs[0])); assert.equal(document.activeElement, tabs[5]);
""")


@pytest.mark.parametrize("page", ["login", "history", "api_portal"])
def test_utility_tabs_keep_visible_panel_selection_and_roving_focus_in_sync(page):
    function = {"login": "switchMode", "history": "switchTab", "api_portal": "updateCode"}[page]
    checks = {
        "login": r"""
const notifications = []; function showStatus(...args) { notifications.push(args); }
for (const mode of ['login', 'register']) { element('button', 'tab-' + mode); element('div', mode + '-panel'); }
switchMode('register');
assert.equal(document.getElementById('login-panel').hidden, true);
assert.equal(document.getElementById('register-panel').hidden, false);
assert.equal(document.getElementById('tab-register').getAttribute('aria-selected'), 'true');
assert.equal(document.getElementById('tab-login').tabIndex, -1);
assert.equal(document.activeElement.id, 'tab-register');
switchMode('invalid');
assert.equal(document.activeElement.id, 'tab-login');
assert.equal(document.getElementById('register-panel').hidden, true);
assert.equal(document.getElementById('tab-login').tabIndex, 0);
""",
        "history": r"""
for (const tab of ['messages', 'audio']) {
    const button = element('button', 'record-tab-' + tab, {class: 'tab-btn'}); button.dataset.tab = tab;
    element('div', 'tab-' + tab, {class: 'tab-content'});
}
switchTab('audio');
assert.equal(document.getElementById('tab-audio').hidden, false);
assert.equal(document.getElementById('tab-messages').hidden, true);
assert.equal(document.getElementById('record-tab-audio').getAttribute('aria-selected'), 'true');
assert.equal(document.getElementById('record-tab-messages').tabIndex, -1);
switchTab('invalid');
assert.equal(document.getElementById('tab-audio').hidden, true);
assert.equal(document.getElementById('record-tab-messages').tabIndex, 0);
""",
        "api_portal": r"""
const state = {}; const codeSamples = {curl: 'curl sample', javascript: 'js sample', python: 'python sample'};
const elements = {codeContent: element('code', 'codeContent')}; element('pre', 'code-panel-content');
for (const lang of Object.keys(codeSamples)) { const tab = element('button', 'code-tab-' + lang, {class: 'code-tab'}); tab.dataset.lang = lang; }
updateCode('python');
assert.equal(elements.codeContent.textContent, 'python sample');
assert.equal(document.getElementById('code-panel-content').getAttribute('aria-labelledby'), 'code-tab-python');
assert.equal(document.getElementById('code-tab-python').tabIndex, 0);
assert.equal(document.getElementById('code-tab-curl').getAttribute('aria-selected'), 'false');
updateCode('invalid'); assert.equal(state.codeLang, 'curl'); assert.equal(elements.codeContent.textContent, 'curl sample');
""",
    }
    run_javascript(DOM + inline_functions(page, function), checks[page])


ADMIN_DOM = r"""
let modalState = {type: 'edit', username: 'demo'};
const dialog = element('dialog', 'modal-backdrop');
element('div', 'modal-feedback', {}, dialog);
const editForm = element('form', 'edit-user-form', {}, dialog);
const resetForm = element('form', 'reset-password-form', {}, dialog);
element('input', 'edit-display-name', {type: 'text'}, editForm);
element('select', 'edit-role', {}, editForm).value = 'user';
element('input', 'reset-password-input', {type: 'password'}, resetForm);
element('input', 'reset-password-confirm', {type: 'password'}, resetForm);
for (const id of ['flash-message', 'flash-icon', 'flash-title', 'flash-text']) element('div', id);
"""


def admin_source(*extra):
    return DOM + ADMIN_DOM + inline_functions("admin", "openModal", "closeModal", "handleModalBackdrop", "showFlash", *extra)


def test_admin_native_modal_lifecycle_and_feedback_do_not_use_inert_background():
    run_javascript(admin_source(), r"""
openModal('edit');
assert.ok(dialog.open); assert.ok(document.body.classList.contains('modal-open'));
assert.equal(document.activeElement.id, 'edit-display-name');
assert.equal(resetForm.style.display, 'none');
showFlash('请输入显示名', 'error');
assert.equal(document.getElementById('modal-feedback').textContent, '请输入显示名');
assert.equal(document.getElementById('modal-feedback').hidden, false);
assert.ok(document.getElementById('modal-feedback').classList.contains('flash-error'));
assert.equal(document.getElementById('flash-message').className, '');
handleModalBackdrop({target: dialog, clientX: 80, clientY: 80}); assert.ok(dialog.open);
handleModalBackdrop({target: editForm, clientX: 0, clientY: 0}); assert.ok(dialog.open);
handleModalBackdrop({target: dialog, clientX: 0, clientY: 0});
assert.equal(dialog.open, false); assert.equal(modalState.username, '');
assert.equal(document.body.classList.contains('modal-open'), false);
assert.equal(editForm.resetCount, 1); assert.equal(resetForm.resetCount, 1);
showFlash('保存成功', 'success'); assert.equal(document.getElementById('flash-text').textContent, '保存成功');
openModal('reset-password');
assert.equal(document.activeElement.id, 'reset-password-input');
assert.equal(document.getElementById('modal-feedback').hidden, true);
assert.equal(document.getElementById('modal-feedback').textContent, '');
""")


def test_admin_modal_validation_stays_visible_and_does_not_submit():
    run_async(admin_source("submitEditUser", "submitResetPassword"), r"""
let requests = 0; async function jsonRequest() { requests++; }
openModal('edit');
await submitEditUser(keyEvent('submit', editForm));
assert.equal(requests, 0); assert.ok(dialog.open);
assert.equal(document.activeElement.id, 'edit-display-name');
assert.match(document.getElementById('modal-feedback').textContent, /显示名/);
closeModal(); modalState = {type: 'reset-password', username: 'demo'}; openModal('reset-password');
document.getElementById('reset-password-input').value = 'example';
document.getElementById('reset-password-confirm').value = 'different';
await submitResetPassword(keyEvent('submit', resetForm));
assert.equal(requests, 0); assert.ok(dialog.open);
assert.equal(document.activeElement.id, 'reset-password-confirm');
assert.match(document.getElementById('modal-feedback').textContent, /不一致/);
""")


HISTORY_DOM = r"""
const API_BASE = 'https://example.invalid';
let currentDetail = {session: {session_id: 'old'}};
let selectedSessionId = 'old'; let detailRequestId = 0; let listRequestId = 0;
let sessions = [{session_id: 'a'}, {session_id: 'b'}]; let filteredSessions = sessions;
const layout = element('main', 'history-main', {class: 'layout'});
element('a', 'admin-link');
for (const id of ['session-list', 'detail-empty', 'detail-content', 'detail-name', 'detail-sub', 'tab-messages', 'tab-audio']) element('div', id, {}, layout);
const actions = Array.from({length: 6}, (_, i) => element('button', 'action-' + i, {}, layout));
const values = Array.from({length: 5}, (_, i) => element('span', 'metric-' + i, {}, layout));
const rows = ['a', 'b'].map(id => {
    const row = element('div', 'row-' + id, {class: 'session-row'}, layout); row.dataset.sessionId = id;
    element('button', 'open-' + id, {class: 'session-open'}, row); return row;
});
const queryAll = document.querySelectorAll;
document.querySelectorAll = selector => selector === '.detail-actions button' ? actions : selector === '.metric-strip .value' ? values : queryAll(selector);
const rendered = []; const switched = []; const requests = [];
function switchTab(name) { switched.push(name); }
function renderDetail(data) { rendered.push(data); document.getElementById('detail-name').textContent = data.session.session_id; }
function escapeHtml(text) { return String(text).replace(/</g, '&lt;'); }
function normalizePatientName() { return '演示用户'; }
function applyFilters() { filteredSessions = sessions; }
function jsonFetch(url, options = {}) { const pending = deferred(); requests.push({url, options, ...pending}); return pending.promise; }
"""


def history_source(*extra):
    return DOM + HISTORY_DOM + inline_functions("history", "resetDetailSelection", "openDetail", *extra)


@pytest.mark.parametrize("newer_first", [True, False])
def test_history_loading_disables_old_actions_and_ignores_stale_detail(newer_first):
    run_async(history_source(), "const newerFirst = " + json.dumps(newer_first) + r""";
const a = openDetail('a'); const b = openDetail('b');
assert.equal(currentDetail, null); assert.equal(selectedSessionId, 'b');
assert.ok(actions.every(button => button.disabled));
assert.ok(values.every(value => value.textContent === '—'));
assert.equal(document.getElementById('detail-sub').textContent, '');
assert.equal(document.getElementById('detail-content').getAttribute('aria-busy'), 'true');
assert.equal(document.getElementById('open-b').getAttribute('aria-current'), 'true');
const oldData = {session: {session_id: 'a'}}; const newData = {session: {session_id: 'b'}};
if (newerFirst) { requests[1].resolve(newData); await b; requests[0].resolve(oldData); await a; }
else {
    requests[0].resolve(oldData); await a;
    assert.equal(currentDetail, null); assert.ok(actions.every(button => button.disabled));
    assert.equal(document.getElementById('detail-content').getAttribute('aria-busy'), 'true');
    requests[1].resolve(newData); await b;
}
assert.equal(currentDetail, newData); assert.deepEqual(rendered, [newData]);
assert.ok(actions.every(button => !button.disabled));
assert.equal(document.getElementById('detail-content').getAttribute('aria-busy'), 'false');
""")


def test_history_failed_detail_and_permission_change_have_safe_recovery_states():
    run_async(history_source("showPermissionError", "refreshSessions"), r"""
const first = openDetail('a'); requests[0].reject(new Error('offline')); await first;
assert.equal(currentDetail, null); assert.ok(actions.every(button => button.disabled));
assert.equal(document.getElementById('detail-content').getAttribute('aria-busy'), 'false');
assert.match(document.getElementById('tab-messages').innerHTML, /重新选择会话或刷新/);
const second = openDetail('b'); showPermissionError(); requests[1].reject(new Error('FORBIDDEN')); await second;
assert.equal(currentDetail, null); assert.equal(selectedSessionId, '');
assert.equal(document.getElementById('detail-content'), null);
assert.equal(document.getElementById('admin-link').hidden, true);
assert.match(layout.innerHTML, /permission-copy/);
await refreshSessions(); await openDetail('a');
assert.equal(requests.length, 2);
""")


def test_deleting_a_loading_history_record_invalidates_its_pending_response():
    run_async(history_source("deleteSession"), r"""
window.CompanionUI = {confirm: async () => true};
sessions = [{session_id: 'a'}]; filteredSessions = sessions;
const detail = openDetail('a'); const deletion = deleteSession('a', actions[0]);
await Promise.resolve();
assert.equal(requests[1].options.method, 'DELETE');
requests[1].resolve({success: true}); await deletion;
requests[0].resolve({session: {session_id: 'a'}}); await detail;
assert.equal(selectedSessionId, ''); assert.equal(currentDetail, null); assert.deepEqual(sessions, []);
assert.equal(document.getElementById('detail-content').classList.contains('open'), false);
assert.equal(document.getElementById('detail-empty').style.display, ''); assert.deepEqual(rendered, []);
""")


def test_latest_history_refresh_wins_over_older_responses():
    source = DOM + HISTORY_DOM + inline_functions("history", "resetDetailSelection", "refreshSessions")
    run_async(source, r"""
const opened = []; function openDetail(id) { selectedSessionId = id; opened.push(id); }
selectedSessionId = ''; currentDetail = null;
const first = refreshSessions(); const second = refreshSessions();
requests[1].resolve({sessions: [{session_id: 'b'}]}); await second;
requests[0].resolve({sessions: [{session_id: 'a'}]}); await first;
assert.deepEqual(sessions, [{session_id: 'b'}]); assert.deepEqual(opened, ['b']);
assert.equal(document.getElementById('session-list').getAttribute('aria-busy'), 'false');
""")


def test_history_rows_use_native_sibling_actions_not_nested_buttons():
    source = inline_functions("history", "renderSessionRow", "normalizePatientName", "escapeHtml", "formatDateShort", "getLastActivity", "modeLabel", "sessionMode")
    output = run_javascript(source, "console.log(renderSessionRow({session_id: 'record-1', patient_name: '<用户>', mode: 'wellbeing'}));")
    rows = PageElements(output).elements
    buttons = [attrs for tag, attrs in rows if tag == "button"]
    assert len(buttons) == 2
    assert all(attrs["type"] == "button" for attrs in buttons)
    assert "role=\"button\"" not in output
    assert "this.closest('.session-row').dataset.sessionId" in output
    assert "<用户>" not in output and "&lt;用户&gt;" in output
    assert output.index("</button>") < output.index("session-delete")


@pytest.mark.parametrize("page,function", [("history", "deleteSession"), ("api_portal", "revokeKey")])
def test_cancelled_record_deletion_or_key_revocation_sends_no_request(page, function):
    run_async(inline_functions(page, function), r"""
const calls = [];
const window = {CompanionUI: {confirm: async () => false}};
let sessions = [{session_id: 'a'}]; let currentDetail = null;
function normalizePatientName() { return '用户'; }
function jsonFetch(...args) { calls.push(args); }
function adminFetch(...args) { calls.push(args); }
const trigger = {disabled: false};
""" + ("await deleteSession('a', trigger);" if page == "history" else "await revokeKey({name: 'key', key_id: 'key-1'});") + r"""
assert.deepEqual(calls, []); assert.equal(trigger.disabled, false);
""")


@pytest.mark.parametrize("function", ["endSession", "newConversation"])
@pytest.mark.parametrize("outcome", ["cancel", "reconnect", "new-session"])
def test_call_confirmations_never_act_on_a_changed_session(function, outcome):
    run_async(DOM + inline_functions("voice_chat", function), "const outcome = " + json.dumps(outcome) + r""";
let sessionStarted = true;
const sent = []; let ws = {send: value => sent.push(value)};
window._pendingSessionId = 'session-a';
const confirmation = deferred(); window.CompanionUI = {confirm: () => confirmation.promise};
""" + f"const completion = {function}();\n" + r"""
if (outcome === 'reconnect') ws = {send: value => sent.push(value)};
if (outcome === 'new-session') window._pendingSessionId = 'session-b';
confirmation.resolve(outcome !== 'cancel'); await completion;
assert.deepEqual(sent, []); assert.equal(sessionStarted, true);
""")


def test_nested_confirmation_owns_escape_instead_of_closing_settings_underneath():
    run_javascript(DOM + inline_functions("voice_chat", "handleCompanionPanelKeydown"), r"""
const calls = []; function toggleDrawer(...args) { calls.push(args); }
const dialog = element('dialog', 'confirmation'); dialog.showModal();
for (const key of ['Escape', 'Tab']) {
    const event = keyEvent(key, dialog); handleCompanionPanelKeydown(event);
    assert.equal(event.defaultPrevented, false);
}
assert.deepEqual(calls, []);
""")


def test_speaker_validation_focus_moves_only_after_dismissing_the_dialog():
    run_async(DOM + inline_functions("voice_chat", "saveSpeaker"), r"""
const WebSocket = {OPEN: 1}; const sent = []; const ws = {readyState: 1, send: data => sent.push(data)};
const input = element('input', 'speaker-name-input');
const confirmation = deferred(); window.CompanionUI = {alert: () => confirmation.promise};
saveSpeaker(); assert.equal(input.focusCount, 0); assert.deepEqual(sent, []);
confirmation.resolve(true); await Promise.resolve(); assert.equal(input.focusCount, 1);
input.value = '常用声纹'; saveSpeaker(); assert.deepEqual(JSON.parse(sent[0]), {type: 'save_speaker', name: '常用声纹'});
""")
