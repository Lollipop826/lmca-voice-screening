from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "memory_scale_fixture.py"
    spec = importlib.util.spec_from_file_location("memory_scale_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_memory_scale_corpus_is_unique_and_diverse():
    module = _module()
    records = [module.memory_record(index) for index in range(5000)]

    assert len({record["text"] for record in records}) == 5000
    assert {record["category"] for record in records} == {"facts", "preferences", "events"}
    assert len({record["topic"] for record in records}) == 10
    assert len({record["emotion"] for record in records}) == 7
    assert "心情不好" in module.memory_record(2)["text"]


def test_memory_scale_targets_and_embedding_batch_limit():
    module = _module()

    assert tuple(module._target(value) for value in module.SCALES) == module.SCALES
    with pytest.raises(argparse.ArgumentTypeError):
        module._target(11)

    parser = module._parser()
    args = parser.parse_args(["grow-memobase", "--target", "10", "--batch-size", "21"])
    with pytest.raises(ValueError, match="between 1 and 20"):
        import asyncio
        asyncio.run(module.grow_memobase(args.target, args.batch_size, args.retries))


def test_init_local_creates_empty_memory_fixture(tmp_path):
    module = _module()
    from src.db import database

    previous_db_path = database.DB_PATH
    previous_env = os.environ.get("DB_PATH")
    try:
        db_path = str(tmp_path / "voice.db")
        module.init_local(db_path, "memory-bench", "secret")
        assert module.verify_local(db_path, 0)["local"] == {
            "turns": 0,
            "memory_items": 0,
            "emotion_trajectory": 0,
        }
    finally:
        database.DB_PATH = previous_db_path
        if previous_env is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = previous_env
