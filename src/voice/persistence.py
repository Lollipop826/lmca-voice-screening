from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .session import VoiceSession


_DROP_JSON_VALUE = object()


def _json_safe_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                key: converted
                for key, item in value.items()
                if isinstance(key, str)
                and (converted := convert(item)) is not _DROP_JSON_VALUE
            }
        if isinstance(value, (list, tuple)):
            return [
                converted
                for item in value
                if (converted := convert(item)) is not _DROP_JSON_VALUE
            ]
        return _DROP_JSON_VALUE

    safe_meta = convert(meta or {})
    return safe_meta if isinstance(safe_meta, dict) else {}


class VoiceDatasetManifestStore:
    """Build the transcript/audio manifest for one connection-owned session."""

    def __init__(
        self,
        session: VoiceSession,
        *,
        list_audio: Callable[[str], list[dict[str, Any]]],
        normalize_profile: Callable[[dict[str, Any] | None], dict[str, Any]],
        now_factory: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.session = session
        self._list_audio = list_audio
        self._normalize_profile = normalize_profile
        self._now_factory = now_factory

    def refresh(self) -> dict[str, Any] | None:
        try:
            if not self.session.history_file:
                return None
            session_dir = Path(self.session.history_file).parent
            session_dir.mkdir(parents=True, exist_ok=True)

            transcript_jsonl_file = session_dir / "transcript.jsonl"
            transcript_txt_file = session_dir / "transcript.txt"
            audio_pairs_file = session_dir / "audio_transcript_pairs.jsonl"
            manifest_file = session_dir / "manifest.json"
            messages_path = Path(self.session.history_file)

            transcript_entries = []
            if transcript_jsonl_file.exists():
                with transcript_jsonl_file.open("r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            transcript_entries.append(json.loads(line))
                        except Exception:
                            continue

            audio_rows = self._list_audio(self.session.session_id)
            audio_pairs = []
            total_audio_duration_s = 0.0
            for index, row in enumerate(audio_rows, start=1):
                file_path = row.get("file_path")
                if not file_path:
                    continue
                audio_path = Path(file_path)
                try:
                    relative_audio_path = str(
                        audio_path.relative_to(session_dir)
                    ).replace("\\", "/")
                except ValueError:
                    relative_audio_path = str(audio_path).replace("\\", "/")
                extra_meta = {}
                try:
                    sidecar_path = audio_path.with_suffix(audio_path.suffix + ".meta.json")
                    if sidecar_path.exists():
                        with sidecar_path.open("r", encoding="utf-8") as sidecar_file:
                            sidecar_payload = json.load(sidecar_file)
                        if (
                            isinstance(sidecar_payload, dict)
                            and isinstance(sidecar_payload.get("extra_meta"), dict)
                        ):
                            extra_meta = sidecar_payload.get("extra_meta") or {}
                except Exception:
                    extra_meta = {}
                duration_s = float(row.get("duration_s") or 0.0)
                total_audio_duration_s += duration_s
                role = row.get("role") or "user"
                content_text = row.get("content_text") or ""
                asr_text = row.get("asr_text") or ""
                audio_pairs.append(
                    {
                        "index": index,
                        "session_id": self.session.session_id,
                        "timestamp": row.get("created_at", ""),
                        "role": role,
                        "transcript": content_text or asr_text,
                        "asr_text": asr_text or content_text,
                        "content_text": content_text,
                        "audio_file": relative_audio_path,
                        "duration_s": duration_s,
                        "extra_meta": extra_meta,
                    }
                )

            with audio_pairs_file.open("w", encoding="utf-8") as file:
                for item in audio_pairs:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")

            profile_snapshot = (
                self._normalize_profile(self.session.patient_profile)
                if self.session.patient_profile
                else {}
            )
            user_message_count = sum(
                1 for entry in transcript_entries if entry.get("role") == "user"
            )
            assistant_message_count = sum(
                1 for entry in transcript_entries if entry.get("role") == "assistant"
            )

            manifest = {
                "session_id": self.session.session_id,
                "generated_at": self._now_factory().isoformat(),
                "profile": profile_snapshot,
                "paths": {
                    "messages_json": str(messages_path.name),
                    "transcript_jsonl": str(transcript_jsonl_file.name),
                    "transcript_txt": str(transcript_txt_file.name),
                    "audio_dir": "user_audio",
                    "assistant_audio_dir": "assistant_audio",
                    "audio_pairs_jsonl": str(audio_pairs_file.name),
                },
                "stats": {
                    "message_count": len(transcript_entries),
                    "user_message_count": user_message_count,
                    "assistant_message_count": assistant_message_count,
                    "audio_segment_count": len(audio_pairs),
                    "audio_duration_s": round(total_audio_duration_s, 3),
                },
                "audio_samples": audio_pairs,
            }

            with manifest_file.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)
            return manifest
        except Exception as exc:
            print(f"[Manifest] ⚠️ 刷新数据集清单失败: {type(exc).__name__}")
            return None


class VoiceAudioStore:
    """Persist user/assistant audio and attach it to the current session."""

    def __init__(
        self,
        session: VoiceSession,
        *,
        save_audio: Callable[..., Any],
        manifest_store: VoiceDatasetManifestStore,
        write_audio: Callable[[Path, np.ndarray, int], Any] | None = None,
        now_factory: Callable[[], datetime] = datetime.now,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.session = session
        self._save_audio = save_audio
        self._manifest_store = manifest_store
        self._write_audio = write_audio or self._write_soundfile
        self._now_factory = now_factory
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex[:6])

    def persist_user(
        self,
        audio_data: Any,
        asr_text: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self.persist(
            audio_data,
            role="user",
            sample_rate=16000,
            folder_name="user_audio",
            file_prefix="user",
            asr_text=asr_text,
            extra_meta=extra_meta,
        )

    def persist_assistant(
        self,
        audio_data: Any,
        content_text: str = "",
    ) -> dict[str, Any] | None:
        return self.persist(
            audio_data,
            role="assistant",
            sample_rate=24000,
            folder_name="assistant_audio",
            file_prefix="assistant",
            content_text=content_text,
        )

    def persist(
        self,
        audio_data: Any,
        *,
        role: str,
        sample_rate: int,
        folder_name: str,
        file_prefix: str,
        asr_text: str = "",
        content_text: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            if audio_array.size == 0:
                return None
            session_dir = Path(self.session.history_file).parent
            audio_dir = session_dir / folder_name
            audio_dir.mkdir(parents=True, exist_ok=True)
            file_name = (
                f"{file_prefix}_{self._now_factory().strftime('%Y%m%d_%H%M%S_%f')}_"
                f"{self._token_factory()}.wav"
            )
            file_path = audio_dir / file_name
            self._write_audio(file_path, audio_array, sample_rate)
            duration_s = float(audio_array.size) / float(sample_rate)
            clean_asr_text = (asr_text or "").strip()
            clean_content_text = (content_text or "").strip()
            self._save_audio(
                self.session.session_id,
                str(file_path),
                duration_s=duration_s,
                asr_text=clean_asr_text or None,
                role=role,
                content_text=clean_content_text or None,
            )
            safe_extra_meta = _json_safe_metadata(extra_meta)
            sidecar_path = file_path.with_suffix(file_path.suffix + ".meta.json")
            sidecar_payload = {
                "session_id": self.session.session_id,
                "role": role,
                "file_path": str(file_path),
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_text": clean_asr_text,
                "content_text": clean_content_text,
                "created_at": self._now_factory().isoformat(),
                "extra_meta": safe_extra_meta,
            }
            try:
                with sidecar_path.open("w", encoding="utf-8") as sidecar_file:
                    json.dump(sidecar_payload, sidecar_file, ensure_ascii=False, indent=2)
            except Exception as sidecar_error:
                print(f"[Audio] ⚠️ 保存语音元数据失败: {type(sidecar_error).__name__}")
            self._manifest_store.refresh()
            print(f"[Audio] ✅ 已保存{role}语音: duration_s={duration_s:.2f}")
            return {
                "file_path": str(file_path),
                "duration_s": duration_s,
                "asr_text": clean_asr_text,
                "content_text": clean_content_text,
                "role": role,
                "extra_meta": safe_extra_meta,
            }
        except Exception as exc:
            print(f"[Audio] ⚠️ 保存{role}语音失败: {type(exc).__name__}")
            return None

    @staticmethod
    def _write_soundfile(file_path: Path, audio_array: np.ndarray, sample_rate: int) -> None:
        import soundfile as sf

        sf.write(str(file_path), audio_array, sample_rate)


class VoiceHistoryStore:
    """Serialize conversation history and transcript files for one session."""

    def __init__(
        self,
        session: VoiceSession,
        *,
        save_message: Callable[..., Any],
        manifest_store: VoiceDatasetManifestStore,
        now_factory: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.session = session
        self._save_message = save_message
        self._manifest_store = manifest_store
        self._now_factory = now_factory
        self._lock = asyncio.Lock()

    async def append(
        self,
        role: str,
        content: str,
        emotion: str | None = None,
        language: str | None = None,
        meta: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> None:
        async with self._lock:
            normalized_turn_id = str(turn_id or "").strip()
            safe_meta = _json_safe_metadata(meta)
            message_key = (role, content, normalized_turn_id)
            if message_key in self.session.history_message_keys:
                return

            try:
                saved = self._save_message(
                    self.session.session_id,
                    role,
                    content,
                    emotion,
                    language,
                    normalized_turn_id or None,
                )
            except Exception as exc:
                print(f"[DB] ⚠️ 保存消息失败: {type(exc).__name__}")
                raise
            if saved is False:
                self.session.history_message_keys.add(message_key)
                return
            self.session.last_message_key = (role, content)
            self.session.history_message_keys.add(message_key)

            history_path = Path(self.session.history_file)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = []
            if history_path.exists():
                try:
                    with history_path.open("r", encoding="utf-8") as file:
                        history = json.load(file)
                except Exception:
                    history = []

            timestamp = self._now_factory().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            history_entry = {"role": role, "content": content, "timestamp": timestamp}
            if normalized_turn_id:
                history_entry["turn_id"] = normalized_turn_id
            history.append(history_entry)

            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)

            transcript_entry = {
                "timestamp": timestamp,
                "role": role,
                "content": content,
            }
            if emotion:
                transcript_entry["emotion"] = emotion
            if language:
                transcript_entry["language"] = language
            if safe_meta or normalized_turn_id:
                transcript_meta = dict(safe_meta)
                if normalized_turn_id:
                    transcript_meta["turn_id"] = normalized_turn_id
                transcript_entry["meta"] = transcript_meta

            transcript_jsonl_file = history_path.parent / "transcript.jsonl"
            with transcript_jsonl_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(transcript_entry, ensure_ascii=False) + "\n")

            transcript_txt_meta = []
            if emotion:
                transcript_txt_meta.append(f"emotion={emotion}")
            if language:
                transcript_txt_meta.append(f"language={language}")
            if safe_meta.get("file_path"):
                transcript_txt_meta.append(
                    f"audio={Path(safe_meta['file_path']).name}"
                )
            transcript_txt_prefix = f"[{timestamp}] {role}"
            if transcript_txt_meta:
                transcript_txt_prefix += f" ({', '.join(transcript_txt_meta)})"

            transcript_txt_file = history_path.parent / "transcript.txt"
            with transcript_txt_file.open("a", encoding="utf-8") as file:
                file.write(f"{transcript_txt_prefix}: {content}\n")

            self._manifest_store.refresh()
