#!/usr/bin/env python3
"""Run the existing application with a dedicated database and process-local direct routing."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent != ROOT/'output' or not out.name.startswith('real_voice_ablation_'):
        raise ValueError('Use a fresh output/real_voice_ablation_* directory')
    out.mkdir(exist_ok=False)
    runtime = out/'isolated_runtime'
    runtime.mkdir()
    from dotenv import dotenv_values
    os.environ.update({k:v for k,v in dotenv_values(ROOT/'.env').items() if v is not None})
    overrides = {
        'DB_PATH': str(runtime/'test_only.sqlite3'),
        'VOICE_CALLS_DIR': str(runtime/'data/voice_calls'),
        'TTS_PREVIEW_DIR': str(runtime/'data/tts_previews'),
        'TTS_SETTINGS_PATH': str(runtime/'data/tts_settings.json'),
        'PIXI_VIEWER_ROOT': str(runtime/'no-viewer'),
        'VOICE_PORT': '18427', 'USE_SOULX_TURN_TAKING': 'false',
        'ENABLE_REALTIME_COMPANION': 'true', 'ENABLE_TURN_INSIGHT': 'true',
        'USE_SPEAKER_VERIFIER': 'false', 'USE_LOCAL_EMBEDDING': 'false',
        'ENABLE_COGNITIVE_SCREENING': 'false', 'ENABLE_LONG_TERM_MEMORY': 'true',
        'ENABLE_LONG_TERM_MEMORY_WRITES': 'false', 'ENABLE_EMOTION': 'true',
        'ENABLE_DEMO_DATA': 'false', 'WEBRTC_MIC_DEBUG_DUMP': 'false',
        'MEMOBASE_PROJECT_URL': '', 'MEMOBASE_API_KEY': '',
        'MEMORY_LOCAL_FALLBACK_ENABLED': 'false',
        'USE_MODELSCOPE': 'true', 'EMOTION_USE_GPU': 'true',
        'HF_HOME': '/data/luyang/cache/hf', 'MODELSCOPE_CACHE': '/data/luyang/cache/modelscope',
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
    }
    model = Path('/data/luyang/cache/modelscope/models/iic/emotion2vec_plus_large')
    view = runtime/'emotion_model_readonly'
    view.mkdir()
    for child in model.iterdir():
        if child.name != 'requirements.txt':
            (view/child.name).symlink_to(child, target_is_directory=child.is_dir())
    overrides['EMOTION2VEC_MODEL'] = str(view)
    os.environ.update(overrides)
    os.environ['AUTH_SECRET_KEY'] = secrets.token_urlsafe(48)
    from scripts.realtime_study_network import enable_direct
    enable_direct()
    # Explicit HTTP transports also cover uvloop-backed asynchronous HTTP.
    import httpx
    from src.llm import http_client_pool
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=60)
    http_client_pool._shared_sync_client = httpx.Client(
        transport=httpx.HTTPTransport(local_address='192.168.5.4', limits=limits),
        timeout=httpx.Timeout(60, connect=10), trust_env=False)
    http_client_pool._shared_async_client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(local_address='192.168.5.4', limits=limits),
        timeout=httpx.Timeout(60, connect=10), trust_env=False)
    creds = {'username': 'study_'+out.name[-15:], 'password': secrets.token_urlsafe(24)}
    path = runtime/'test_credentials.json'
    path.write_text(json.dumps(creds))
    path.chmod(0o600)
    plan = {
        'created_at': datetime.now().astimezone().isoformat(),
        'pid': os.getpid(), 'server': 'http://127.0.0.1:18427',
        'production_env_sha256': hashlib.sha256((ROOT/'.env').read_bytes()).hexdigest(),
        'environment_overrides': {k:v for k,v in overrides.items() if 'KEY' not in k},
        'network': 'External IPv4 sockets bound to 192.168.5.4 in test processes only; no global proxy change',
        'boundaries': ['Real application, local Silero VAD; SoulX unavailable',
                       'Frozen persisted SQLite authoritative memory card; no Memobase semantic retrieval',
                       'Paced synthetic audio; no microphone, physical display or speaker timing',
                       'Actual Emotion2Vec predictions; synthesis emotion requests are not human ground truth'],
    }
    (out/'preflight_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (out/'server.pid').write_text(str(os.getpid()))
    log = (out/'isolated_server.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.chdir(runtime)
    # Observe the actual prompt passed to the unchanged production agent.
    from src.agents.wellbeing_companion_agent import WellbeingCompanionAgent
    original = WellbeingCompanionAgent._build_messages
    signature = inspect.signature(original)
    prompt_log = (out/'agent_prompt_events.jsonl').open('a', buffering=1)
    lock = threading.Lock()

    def observed(self, *a, **kw):
        messages = original(self, *a, **kw)
        bound = signature.bind(self, *a, **kw)
        inputs = {k:v for k,v in bound.arguments.items() if k != 'self'}
        row = {'perf_counter': time.perf_counter(), 'inputs': inputs,
               'messages': [{'role':m.type, 'content':m.content} for m in messages]}
        with lock:
            prompt_log.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
        return messages

    WellbeingCompanionAgent._build_messages = observed
    import uvicorn
    uvicorn.run('voice_server:app', host='127.0.0.1', port=18427, access_log=False)


if __name__ == '__main__':
    main()
