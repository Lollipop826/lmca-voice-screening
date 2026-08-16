import ast
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.voice import (
    VoiceAudioStore,
    VoiceDatasetManifestStore,
    VoiceHistoryStore,
    VoiceSession,
)


VOICE_SERVER_PATH = Path(__file__).resolve().parents[2] / "voice_server.py"
VOICE_APPLICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "voice"
    / "application.py"
)
FIXED_NOW = datetime(2026, 7, 16, 18, 30, 45, 123000)


def _voice_application_class():
    tree = ast.parse(
        VOICE_APPLICATION_PATH.read_text(encoding="utf-8"),
        filename=str(VOICE_APPLICATION_PATH),
    )
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VoiceEndpointApplication"
    )


class _ManifestSpy:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class VoiceAudioStoreTests(unittest.TestCase):
    def test_user_audio_is_persisted_with_sidecar_and_session_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor-a",
            )
            session.bind_session("call-audio")
            session.history_file = str(Path(temp_dir) / "messages.json")
            saved_rows = []
            manifest = _ManifestSpy()

            def save_audio(session_id, file_path, **kwargs):
                saved_rows.append((session_id, file_path, kwargs))

            def write_audio(path, audio, sample_rate):
                self.assertEqual(audio.dtype, np.float32)
                self.assertEqual(sample_rate, 16000)
                path.write_bytes(b"test-wav")

            store = VoiceAudioStore(
                session,
                save_audio=save_audio,
                manifest_store=manifest,
                write_audio=write_audio,
                now_factory=lambda: FIXED_NOW,
                token_factory=lambda: "abc123",
            )

            result = store.persist_user(
                np.ones(16000, dtype=np.float64),
                "  今天晴天  ",
                extra_meta={"source": "webrtc"},
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["duration_s"], 1.0)
            self.assertEqual(result["asr_text"], "今天晴天")
            self.assertEqual(result["extra_meta"], {"source": "webrtc"})
            self.assertEqual(manifest.refresh_count, 1)
            self.assertEqual(len(saved_rows), 1)
            self.assertEqual(saved_rows[0][0], "call-audio")
            self.assertEqual(saved_rows[0][2]["role"], "user")
            self.assertEqual(saved_rows[0][2]["asr_text"], "今天晴天")

            audio_path = Path(result["file_path"])
            self.assertEqual(
                audio_path.name,
                "user_20260716_183045_123000_abc123.wav",
            )
            sidecar = json.loads(
                audio_path.with_suffix(".wav.meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["session_id"], "call-audio")
            self.assertEqual(sidecar["sample_rate"], 16000)
            self.assertEqual(sidecar["extra_meta"], {"source": "webrtc"})

    def test_user_audio_drops_runtime_metadata_before_json_persistence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = VoiceSession(
                connection=object(), agent=object(), owner_username="doctor-a"
            )
            session.bind_session("call-audio")
            session.history_file = str(Path(temp_dir) / "messages.json")
            manifest = _ManifestSpy()
            store = VoiceAudioStore(
                session,
                save_audio=lambda *_args, **_kwargs: None,
                manifest_store=manifest,
                write_audio=lambda path, *_args: path.write_bytes(b"test-wav"),
                now_factory=lambda: FIXED_NOW,
                token_factory=lambda: "abc123",
            )
            runtime_turn = object()

            result = store.persist_user(
                np.ones(16000, dtype=np.float32),
                "今天晴天",
                extra_meta={
                    "source": "webrtc",
                    "realtime_turn": runtime_turn,
                    "nested": {"asr_source": "stream", "runtime": runtime_turn},
                    "events": ["started", runtime_turn],
                },
            )

            expected_meta = {
                "source": "webrtc",
                "nested": {"asr_source": "stream"},
                "events": ["started"],
            }
            self.assertEqual(result["extra_meta"], expected_meta)
            sidecar = json.loads(
                Path(result["file_path"])
                .with_suffix(".wav.meta.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["extra_meta"], expected_meta)


class VoiceHistoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_append_preserves_legacy_files_and_deduplicates_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor-a",
            )
            session.bind_session("call-history")
            session.history_file = str(Path(temp_dir) / "messages.json")
            saved_messages = []
            manifest = _ManifestSpy()
            store = VoiceHistoryStore(
                session,
                save_message=lambda *args: saved_messages.append(args),
                manifest_store=manifest,
                now_factory=lambda: FIXED_NOW,
            )
            audio_meta = {"file_path": str(Path(temp_dir) / "user.wav")}

            await store.append(
                "user",
                "今天晴天",
                "neutral",
                "zh",
                audio_meta,
            )
            await store.append(
                "user",
                "今天晴天",
                "neutral",
                "zh",
                audio_meta,
            )

            self.assertEqual(len(saved_messages), 1)
            self.assertEqual(saved_messages[0][0], "call-history")
            self.assertEqual(manifest.refresh_count, 1)
            messages = json.loads(
                Path(session.history_file).read_text(encoding="utf-8")
            )
            self.assertEqual(
                messages,
                [
                    {
                        "role": "user",
                        "content": "今天晴天",
                        "timestamp": "2026-07-16 18:30:45.123",
                    }
                ],
            )
            transcript = [
                json.loads(line)
                for line in (Path(temp_dir) / "transcript.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(transcript[0]["emotion"], "neutral")
            self.assertEqual(transcript[0]["language"], "zh")
            self.assertEqual(transcript[0]["meta"], audio_meta)
            transcript_text = (Path(temp_dir) / "transcript.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("emotion=neutral", transcript_text)
            self.assertIn("language=zh", transcript_text)
            self.assertIn("audio=user.wav", transcript_text)

    async def test_append_drops_runtime_metadata_before_writing_transcript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = VoiceSession(
                connection=object(), agent=object(), owner_username="doctor-a"
            )
            session.bind_session("call-history")
            session.history_file = str(Path(temp_dir) / "messages.json")
            store = VoiceHistoryStore(
                session,
                save_message=lambda *_args: None,
                manifest_store=_ManifestSpy(),
                now_factory=lambda: FIXED_NOW,
            )

            await store.append(
                "user",
                "今天晴天",
                meta={
                    "file_path": str(Path(temp_dir) / "user.wav"),
                    "extra_meta": {"source": "webrtc", "realtime_turn": object()},
                },
            )

            transcript = json.loads(
                (Path(temp_dir) / "transcript.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                transcript["meta"],
                {
                    "file_path": str(Path(temp_dir) / "user.wav"),
                    "extra_meta": {"source": "webrtc"},
                },
            )


class VoiceDatasetManifestStoreTests(unittest.TestCase):
    def test_refresh_builds_audio_pairs_and_summary_from_session_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            session = VoiceSession(
                connection=object(),
                agent=object(),
                owner_username="doctor-a",
            )
            session.bind_session("call-manifest")
            session.history_file = str(session_dir / "messages.json")
            session.patient_profile.update({"name": " 张阿姨 "})
            (session_dir / "transcript.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"role": "user", "content": "今天晴天"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"role": "assistant", "content": "那是什么季节？"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            audio_path = session_dir / "user_audio" / "sample.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"wav")
            audio_path.with_suffix(".wav.meta.json").write_text(
                json.dumps({"extra_meta": {"source": "webrtc"}}),
                encoding="utf-8",
            )
            audio_rows = [
                {
                    "file_path": str(audio_path),
                    "duration_s": 1.25,
                    "role": "user",
                    "asr_text": "今天晴天",
                    "content_text": "",
                    "created_at": "2026-07-16T18:30:45",
                }
            ]
            store = VoiceDatasetManifestStore(
                session,
                list_audio=lambda session_id: audio_rows,
                normalize_profile=lambda profile: {
                    "name": str((profile or {}).get("name") or "").strip()
                },
                now_factory=lambda: FIXED_NOW,
            )

            manifest = store.refresh()

            self.assertEqual(manifest["session_id"], "call-manifest")
            self.assertEqual(manifest["profile"], {"name": "张阿姨"})
            self.assertEqual(manifest["stats"]["message_count"], 2)
            self.assertEqual(manifest["stats"]["user_message_count"], 1)
            self.assertEqual(manifest["stats"]["assistant_message_count"], 1)
            self.assertEqual(manifest["stats"]["audio_segment_count"], 1)
            self.assertEqual(manifest["stats"]["audio_duration_s"], 1.25)
            self.assertEqual(
                manifest["audio_samples"][0]["audio_file"],
                "user_audio/sample.wav",
            )
            self.assertEqual(
                manifest["audio_samples"][0]["extra_meta"],
                {"source": "webrtc"},
            )
            persisted = json.loads(
                (session_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, manifest)


class VoicePersistenceEndpointWiringTests(unittest.TestCase):
    def test_endpoint_delegates_persistence_instead_of_defining_io_closures(self):
        application = _voice_application_class()
        nested_function_names = {
            node.name
            for node in ast.walk(application)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertNotIn("_persist_audio_segment", nested_function_names)
        self.assertNotIn("_refresh_session_dataset_manifest", nested_function_names)
        self.assertNotIn("append_history", nested_function_names)

        constructed_names = {
            node.func.id
            for node in ast.walk(application)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("VoiceAudioStore", constructed_names)
        self.assertIn("VoiceDatasetManifestStore", constructed_names)
        self.assertIn("VoiceHistoryStore", constructed_names)


if __name__ == "__main__":
    unittest.main()
