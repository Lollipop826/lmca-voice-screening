"""Structural regression tests for the object-oriented voice handlers."""

import ast
from pathlib import Path
import unittest


HANDLERS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "voice" / "handlers"
)
LEGACY_HANDLER_FILE = HANDLERS_DIR.with_suffix(".py")

PUBLIC_TYPE_MODULES = {
    "MessageHandlerAction": "message_handler_action.py",
    "VoiceRouteResult": "voice_route_result.py",
    "VoiceMessageRouter": "voice_message_router.py",
    "AgentOutputPresenter": "agent_output_presenter.py",
    "ClientControlHandler": "client_control_handler.py",
    "SessionResumeHandler": "session_resume_handler.py",
    "AnswerCompletionMessageHandler": (
        "answer_completion_message_handler.py"
    ),
    "SessionLifecycleHandler": "session_lifecycle_handler.py",
    "VoiceFeedbackPresenter": "voice_feedback_presenter.py",
    "ManualAudioBlobProcessor": "manual_audio_blob_processor.py",
    "ManualAudioInputHandler": "manual_audio_input_handler.py",
    "ManualInterruptHandler": "manual_interrupt_handler.py",
    "VoiceInterruptionController": "voice_interruption_controller.py",
    "VisionMessageHandler": "vision_message_handler.py",
    "TextTurnHandler": "text_turn_handler.py",
    "SpeechTurnConfig": "speech_turn_config.py",
    "SpeechTurnProcessor": "speech_turn_processor.py",
    "SoulXAudioHandler": "soulx_audio_handler.py",
    "LiveAudioConfig": "live_audio_config.py",
    "LiveAudioInputHandler": "live_audio_input_handler.py",
    "SpeakerVerificationController": (
        "speaker_verification_controller.py"
    ),
}


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _method_node(
    path: Path,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    class_node = _class_node(path, class_name)
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class VoiceHandlerArchitectureTests(unittest.TestCase):
    def test_public_handler_types_live_in_independent_modules(self):
        self.assertFalse(LEGACY_HANDLER_FILE.exists())
        self.assertTrue((HANDLERS_DIR / "__init__.py").exists())

        for class_name, filename in PUBLIC_TYPE_MODULES.items():
            path = HANDLERS_DIR / filename
            self.assertTrue(path.exists(), filename)
            class_names = [
                node.name
                for node in ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                ).body
                if isinstance(node, ast.ClassDef)
            ]
            self.assertEqual(class_names, [class_name], filename)

    def test_large_entry_methods_are_thin_orchestrators(self):
        speech_process = _method_node(
            HANDLERS_DIR / "speech_turn_processor.py",
            "SpeechTurnProcessor",
            "process",
        )
        live_handle = _method_node(
            HANDLERS_DIR / "live_audio_input_handler.py",
            "LiveAudioInputHandler",
            "handle_message",
        )

        self.assertLessEqual(
            speech_process.end_lineno - speech_process.lineno + 1,
            60,
        )
        self.assertLessEqual(
            live_handle.end_lineno - live_handle.lineno + 1,
            30,
        )

    def test_large_entry_methods_do_not_rebind_object_graph(self):
        methods = [
            _method_node(
                HANDLERS_DIR / "speech_turn_processor.py",
                "SpeechTurnProcessor",
                "process",
            ),
            _method_node(
                HANDLERS_DIR / "live_audio_input_handler.py",
                "LiveAudioInputHandler",
                "handle_message",
            ),
        ]

        for method in methods:
            self_aliases = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and _root_name(node.value) == "self"
                and any(
                    isinstance(target, ast.Name)
                    for target in node.targets
                )
            ]
            nested_functions = [
                node
                for node in method.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            self.assertEqual(self_aliases, [], method.name)
            self.assertEqual(nested_functions, [], method.name)

    def test_no_handler_method_exceeds_two_hundred_lines(self):
        oversized = []
        for path in HANDLERS_DIR.glob("*.py"):
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            for class_node in (
                node for node in tree.body if isinstance(node, ast.ClassDef)
            ):
                for method in class_node.body:
                    if not isinstance(
                        method,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        continue
                    line_count = (
                        method.end_lineno - method.lineno + 1
                    )
                    if line_count > 200:
                        oversized.append(
                            (
                                path.name,
                                class_node.name,
                                method.name,
                                line_count,
                            )
                        )

        self.assertEqual(oversized, [])

    def test_extracted_state_objects_are_connection_local(self):
        speech_context = _class_node(
            HANDLERS_DIR / "speech_turn_context.py",
            "SpeechTurnContext",
        )
        live_context = _class_node(
            HANDLERS_DIR / "live_audio_frame_context.py",
            "LiveAudioFrameContext",
        )

        self.assertTrue(speech_context.decorator_list)
        self.assertTrue(live_context.decorator_list)
