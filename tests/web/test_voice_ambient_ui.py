import re
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOICE_HTML = PROJECT_ROOT / "static" / "voice_chat.html"
VOICE_STYLES = PROJECT_ROOT / "static" / "voice_chat_glass.css"


class PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


def read_page():
    page = PageElements()
    page.feed(VOICE_HTML.read_text(encoding="utf-8"))
    return page.elements


def test_ambient_theme_keeps_the_avatar_scene_background():
    styles = VOICE_STYLES.read_text(encoding="utf-8")
    scene_rules = re.findall(
        r"\.avatar-call-pane\.avatar-reply-pane\s*\{([^}]+)\}", styles
    )
    backgrounds = [
        declaration.strip()
        for rule in scene_rules
        for declaration in rule.split(";")
        if declaration.strip().startswith("background:")
    ]

    assert len(backgrounds) == 1
    assert "/pixi-viewer/assets/room-background.png" in backgrounds[0]
    assert "transparent" not in backgrounds[0]


def test_ambient_theme_keeps_media_elements_and_renderer_host():
    elements = read_page()
    identifiers = [attributes["id"] for _, attributes in elements if "id" in attributes]

    for identifier in (
        "avatar-user-video",
        "avatar-user-composite",
        "avatar-reply-video",
        "avatar-reply-3d",
        "avatar-connection-state",
        "avatar-model-select",
        "avatar-scale-range",
        "avatar-position-reset",
        "camera-background-select",
        "camera-background-status",
    ):
        assert identifiers.count(identifier) == 1

    user_video = next(
        attributes for tag, attributes in elements
        if tag == "video" and attributes.get("id") == "avatar-user-video"
    )
    assert all(name in user_video for name in ("autoplay", "muted", "playsinline"))
    assert any(
        tag == "button" and attributes.get("data-avatar-call-action") == "toggle-camera"
        for tag, attributes in elements
    )


def test_virtual_background_is_opt_in_local_and_keeps_original_as_default():
    elements = read_page()
    select = next(
        attributes for tag, attributes in elements
        if tag == "select" and attributes.get("id") == "camera-background-select"
    )
    assert select["aria-describedby"] == "camera-background-status"

    options = [
        attributes for tag, attributes in elements
        if tag == "option" and attributes.get("value") in {"original", "blur", "room"}
    ]
    assert [attributes["value"] for attributes in options] == ["original", "blur", "room"]
    assert "selected" in options[0]
    assert all("selected" not in attributes for attributes in options[1:])

    source = VOICE_HTML.read_text(encoding="utf-8")
    worker = PROJECT_ROOT / "static" / "camera_background_worker.js"
    worker_source = worker.read_text(encoding="utf-8")
    assert "new Worker('/static/camera_background_worker.js', { type: 'module' })" in source
    assert "https://" not in worker_source
    assert "/static/vendor/mediapipe/tasks-vision/vision_bundle.mjs" in worker_source
    assert "/static/vendor/mediapipe/models/selfie_segmenter_landscape.tflite" in worker_source
    assert (PROJECT_ROOT / "static/vendor/mediapipe/tasks-vision/vision_bundle.mjs").is_file()
    assert (PROJECT_ROOT / "static/vendor/mediapipe/tasks-vision/wasm/vision_wasm_module_internal.wasm").is_file()
    assert (PROJECT_ROOT / "static/vendor/mediapipe/models/selfie_segmenter_landscape.tflite").is_file()


def test_ambient_theme_preserves_primary_button_events():
    elements = read_page()
    buttons = {
        attributes.get("id"): attributes
        for tag, attributes in elements if tag == "button" and attributes.get("id")
    }
    expected_handlers = {
        "interrupt-btn": "interruptAI()",
        "answer-completion-btn": "handleAnswerCompletionButton(event)",
        "voice-mode-btn": "toggleVoiceInputMode()",
        "mic-btn": "toggleRecording()",
        "end-btn": "endSession()",
        "new-session-btn": "newConversation()",
    }

    assert "doctor-pause-btn" not in buttons
    assert "toggleDoctorPromptMarker" not in VOICE_HTML.read_text(encoding="utf-8")
    for identifier, handler in expected_handlers.items():
        assert buttons[identifier]["onclick"] == handler
    assert buttons["answer-completion-btn"]["ondblclick"] == "disableAnswerCompletionMode(event)"
    assert "disabled" in buttons["end-btn"]
    assert any(
        tag == "button" and attributes.get("onclick") == "exportChatHistory()"
        for tag, attributes in elements
    )


def test_reference_layout_keeps_camera_and_secondary_controls_discoverable():
    elements = read_page()
    camera_controls = [
        attributes for tag, attributes in elements
        if tag == "button" and attributes.get("data-avatar-call-action") == "toggle-camera"
    ]
    assert len(camera_controls) == 1
    assert "ctrl-btn" in camera_controls[0]["class"].split()

    source = VOICE_HTML.read_text(encoding="utf-8")
    disclosure_start = source.index('<details id="call-more-controls"')
    disclosure_end = source.index("</details>", disclosure_start)
    disclosure = PageElements()
    disclosure.feed(source[disclosure_start:disclosure_end])
    assert any(tag == "summary" for tag, _ in disclosure.elements)
    assert {
        attributes.get("onclick") for tag, attributes in disclosure.elements
        if tag == "button"
    } == {
        "switchInputMode()",
        "toggleVoiceInputMode()",
        "exportChatHistory()",
        "newConversation()",
    }

    styles = VOICE_STYLES.read_text(encoding="utf-8")
    desktop_camera = re.search(
        r"\.avatar-call-pane:not\(\.avatar-reply-pane\)\s*\{([^}]+)\}", styles
    ).group(1)
    composite = "\n".join(re.findall(r"\.avatar-user-composite\s*\{([^}]+)\}", styles))
    assert "margin: 116px 3vw calc(var(--call-controls-clearance) + 42px) 3.5vw" in desktop_camera
    assert "position: absolute" in composite
    assert "inset: 0" in composite
    assert "background: transparent" in composite
    assert "background: #dbe7df" not in composite

    camera_script = source[source.index("let userVideoStream = null;"):]
    assert "if (backgroundMode === 'blur')" in camera_script
    assert "drawCover(compositeContext, roomImage)" not in camera_script
