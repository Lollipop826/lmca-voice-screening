#!/usr/bin/env python3
"""Isolated 2x2 real-time voice ablation: real models, no patient recordings.

Only operates on a test-only SQLite DB provisioned by the companion launcher.
Records monotonic client timestamps, actual audio, failures, flags and DB hashes.
Does not equate TTS events/received bytes with physical speaker playback.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import wave
from array import array
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_voice_realtime import (
    _connect_websocket, _start_session, _receive_json, _prepare_audio_cases,
    _frame_rms_envelope,
)
from scripts.benchmark_voice_sqlite_ablation import (
    Recorder, state_snapshot, test_paths,
)

SERVER = 'http://127.0.0.1:18427'
DT = 512 / 16000
ARMS = {
    'M1E1': {'memory': True, 'emotion': True, 'label': '记忆开·情绪开'},
    'M0E1': {'memory': False, 'emotion': True, 'label': '记忆关·情绪开'},
    'M1E0': {'memory': True, 'emotion': False, 'label': '记忆开·情绪关'},
    'M0E0': {'memory': False, 'emotion': False, 'label': '记忆关·情绪关'},
}
METRIC_PAIRS = {
    'speech_end_to_vad_end_ms': ('speech_end', 'vad_end'),
    'vad_end_to_asr_final_ms': ('vad_end', 'asr_final'),
    'speech_end_to_asr_final_ms': ('speech_end', 'asr_final'),
    'asr_final_to_first_ai_text_ms': ('asr_final', 'first_ai_text'),
    'first_ai_text_to_first_audio_ms': ('first_ai_text', 'first_audio'),
    'speech_end_to_first_ai_text_ms': ('speech_end', 'first_ai_text'),
    'speech_end_to_first_audio_ms': ('speech_end', 'first_audio'),
    'first_ai_text_to_tts_end_ms': ('first_ai_text', 'tts_end'),
    'speech_end_to_tts_end_ms': ('speech_end', 'tts_end'),
    'stream_start_to_first_partial_ms': ('stream_start', 'first_partial'),
}


def now():
    return datetime.now().astimezone().isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ''.join(c.lower() for c in str(text) if c.isalnum())


def cer(ref, hyp):
    ref, hyp = normalized(ref), normalized(hyp)
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        nxt = [i]
        for j, b in enumerate(hyp, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j-1] + (a != b)))
        row = nxt
    return row[-1] / max(1, len(ref))


def metrics(marks):
    # Preserve negatives: never silently clip malformed/multi-turn intervals to 0.
    return {name: round(1000 * (marks[b] - marks[a]), 3)
            for name, (a, b) in METRIC_PAIRS.items() if a in marks and b in marks}


def numeric_summary(values):
    xs = sorted(float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x))
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'mean': round(statistics.fmean(xs), 3),
            'median': round(statistics.median(xs), 3),
            'p95_nearest_rank': round(xs[math.ceil(.95 * len(xs))-1], 3),
            'std': round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            'min': round(xs[0], 3), 'max': round(xs[-1], 3)}


def design(samples, repeats, seed):
    rng = random.Random(seed)
    blocks = [(rep, sample['sample_id']) for rep in range(1, repeats+1) for sample in samples]
    rng.shuffle(blocks)
    schedule = []
    for block, (rep, sample_id) in enumerate(blocks, 1):
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            schedule.append({'trial_id': f'formal-{len(schedule)+1:03}',
                             'stage': 'formal', 'block_id': block, 'repeat': rep,
                             'sample_id': sample_id, 'arm': arm})
    arms = list(ARMS)
    rng.shuffle(arms)
    warmup = [{'trial_id': f'warmup-{i:02}', 'stage': 'warmup', 'block_id': 0,
               'repeat': 0, 'sample_id': samples[0]['sample_id'], 'arm': arm}
              for i, arm in enumerate(arms, 1)]
    return warmup, schedule


def prepare(out, repeats=3, seed=20260913):
    out, db = test_paths(out)
    if (out/'realtime_protocol.json').exists():
        raise RuntimeError('Frozen realtime protocol already exists; no overwrite')
    os.environ.update(DB_PATH=str(db), MEMOBASE_PROJECT_URL='', MEMOBASE_API_KEY='',
                      MEMORY_LOCAL_FALLBACK_ENABLED='false', USE_LOCAL_EMBEDDING='false')
    from scripts.prepare_memory_ablation_fixture import prepare_fixture
    from src.context_management.emotion_memobase import EmotionMemobase
    fixture = json.loads((ROOT/'tests/fixtures/memory_ablation_cases.json').read_text())
    fixture['fixture_id'] = out.name + '-realtime-v1'
    fixture['patient_id'] = 'pt-synthetic-rt-' + out.name.removeprefix('real_voice_ablation_')
    seeded = prepare_fixture(fixture, db_path=str(db), apply=True)
    save(out/'fixture.json', fixture)
    save(out/'fixture_seed_result.json', seeded)
    # This instance is used only to read the frozen card, not for trial inference.
    memory = EmotionMemobase(db_path=str(db), emotion_classifier=lambda _: {}, logger=lambda _: None)
    card = memory.get_authoritative_card(fixture['patient_id'])
    (out/'authoritative_memory_card.txt').write_text(card, encoding='utf-8')
    manifest = ROOT/'output/memory_ablation_audio_manifest.json'
    cases = [{**x, 'audio_path': str((manifest.parent/x['audio_path']).resolve())}
             for x in json.loads(manifest.read_text())['samples']]
    prepared = _prepare_audio_cases(cases, no_trim=False)
    assets = out/'stimuli'
    assets.mkdir()
    samples = []
    for item in prepared:
        frames = item.pop('frames')
        raw = b''.join(frames)
        values = array('h'); values.frombytes(raw)
        envelope = _frame_rms_envelope(values)
        threshold = max(envelope) * .08
        voiced = [i for i, rms in enumerate(envelope) if rms > threshold]
        if not voiced:
            raise RuntimeError('Synthetic fixture contains no energy-detectable speech')
        path = assets/(item['sample_id']+'.wav')
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(raw)
        samples.append({**item, 'source_audio_path': item['audio_path'], 'audio_path': str(path),
                        'source_wav_sha256': sha(item['audio_path']), 'prepared_wav_sha256': sha(path),
                        'frame_count': len(frames), 'last_energy_frame_index': voiced[-1],
                        'first_energy_frame_index': voiced[0], 'energy_threshold_rms_pcm16': threshold,
                        'provenance': 'pre-existing synthetic TTS fixture, NOT a patient recording'})
    warmup, schedule = design(samples, repeats, seed)
    protocol = {
        'schema_version': 'isolated-realtime-2x2-v2', 'created_at': now(), 'seed': seed,
        'server': SERVER, 'patient_id': fixture['patient_id'], 'profile': fixture['profile'],
        'arms': ARMS, 'samples': samples, 'warmup': warmup, 'schedule': schedule,
        'repeats_per_sample_per_arm': repeats, 'timeout_after_input_s': 75, 'settle_s': 2.0,
        'maximum_speech_frame_lag_ms': 150,
        'input': 'mono 16kHz PCM16, 512 samples every 32ms over the real binary WebSocket audio path',
        'zero': 'monotonic client send-completion time of the last energy-active speech frame (8% of peak frame RMS); frame/energy estimate, not an observed physical utterance endpoint',
        'metrics': METRIC_PAIRS, 'greeting': 'wait for nonempty greeting audio and normal TTS_END, then duration+0.25s for server ai_speaking_until before measured input; not physical playback',
        'design': '4 synthetic utterances x 3 repeats x 4 arms, randomized paired blocks, fresh session each trial; four warmups excluded',
        'controlled': ['same immutable profile and SQLite card', 'same WAV per paired block', 'same ASR/LLM/TTS/VAD config',
                       'long-term writes disabled', 'same real Emotion2Vec GPU model', 'single sequential load'],
        'boundary': ['local Silero VAD, NOT SoulX', 'SQLite authoritative memory-card on/off, NOT Memobase semantic retrieval',
                     'loopback WebSocket transport, NOT browser playback', 'synthetic speech, NOT human clinical/subjective quality',
                     'first AI event is a nonempty text segment, NOT necessarily LLM first token',
                     'TTS completion interval overlaps generation and synthesis; do not add it to first-audio latency'],
        'failure_policy': 'all trials retained; no automatic replacement or hidden retry; stop on profile/memory drift; report incomplete trials separately',
        'source_sha256': {'scripts/benchmark_voice_realtime_ablation.py': sha(__file__),
                          'scripts/benchmark_voice_realtime.py': sha(ROOT/'scripts/benchmark_voice_realtime.py')},
    }
    save(out/'realtime_protocol.json', protocol)
    save(out/'memory_state_before.json', state_snapshot(db, fixture['patient_id']))
    import requests
    client = requests.Session(); client.trust_env = False
    creds = json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    r = client.post(SERVER+'/api/auth/register', json={**creds, 'display_name': '实时消融隔离测试账户'}, timeout=15)
    if r.status_code not in (200, 201):
        raise RuntimeError(f'Test-only account registration failed: {r.status_code}')
    health = client.get(SERVER+'/health', timeout=5).json()
    save(out/'isolated_health.json', health)
    if not (health.get('startup', {}).get('ready') and health.get('turn_taking') == 'local'
            and health.get('release_flags', {}).get('emotion')):
        raise RuntimeError('Isolated service not ready for real-model ablation')
    print(json.dumps({'prepared': str(out), 'formal_trials': len(schedule), 'warmup': len(warmup),
                      'card_chars': len(card), 'samples': [{k: s[k] for k in ['sample_id','speech_seconds','last_energy_frame_index']} for s in samples]}, ensure_ascii=False), flush=True)


def greeting_idle_guard(duration):
    duration = float(duration)
    if not math.isfinite(duration) or not 0 < duration <= 120:
        raise ValueError("Greeting TTS_END must contain a finite positive duration <=120s")
    return duration + .25


async def wait_for_greeting_idle(recorder, timeout_s=45):
    """Wait for server's real speaking-until guard, not just synthesis completion.

    The current greeting handler sets ai_speaking_until = now + duration AFTER
    TTS_END. Waiting for this guard avoids inadvertently benchmarking barge-in.
    This is not a physical playback observation or a fabricated playback ACK.
    """
    started = time.perf_counter()
    deadline = started + timeout_s
    greeting_id = ''
    audio_bytes = 0
    while True:
        e = await _receive_json(recorder, deadline)
        kind = e.get('type')
        if kind in {'error', 'tts_error'}:
            raise RuntimeError('Greeting service error: '+str(kind))
        eid = str(e.get('turn_id') or '')
        if kind == 'tts_start': greeting_id = eid
        if kind in {'tts_chunk', 'tts_audio'} and eid == greeting_id:
            payload = e.get('chunk') or e.get('audio') or e.get('data')
            if payload: audio_bytes += len(base64.b64decode(payload, validate=True))
        if kind == 'tts_end' and eid == greeting_id:
            if not greeting_id or e.get('reason') or not audio_bytes:
                raise RuntimeError('Incomplete/non-normal greeting')
            ended = recorder.last_received_at
            guard = greeting_idle_guard(e.get('duration'))
            await asyncio.sleep(max(0, ended + guard - time.perf_counter()))
            ready = time.perf_counter()
            return {'turn_id': greeting_id, 'audio_bytes': audio_bytes,
                    'tts_end_perf': ended, 'duration_s': float(e['duration']),
                    'server_idle_guard_s': guard, 'ready_perf': ready,
                    'server_idle_wait_completed': ready >= ended + guard,
                    'setup_wait_ms': (ready-started)*1000}


async def stream_audio(recorder, frames, last_energy_index, marks, frame_log, audio_done):
    start = time.perf_counter()
    marks['stream_start'] = start
    silence = b'\x00' * 1024
    i = 0
    while True:
        # A browser can send a frame only after capturing its last sample.
        target = start + (i+1)*DT
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        before = time.perf_counter()
        await recorder.send(frames[i] if i < len(frames) else silence)
        sent = time.perf_counter()
        frame_log.append({'index': i, 'is_stimulus': i < len(frames), 'target_perf': target,
                          'send_start_perf': before, 'sent_perf': sent, 'lag_ms': (sent-target)*1000})
        if i == 0: marks['first_input_frame'] = sent
        if i == last_energy_index: marks['speech_end'] = sent
        if i == len(frames)-1:
            marks['last_input_frame'] = sent
            audio_done.set()
        i += 1


def analyze_events(events, marks):
    """Associate all measured outputs to one nonempty ASR turn; decode audio first."""
    turn_ids = list(dict.fromkeys(str(e.get('turn_id')) for _, e in events
                                if e.get('type') == 'asr_result' and e.get('turn_id')))
    result = {'observed_turn_ids': turn_ids, 'observed_asr_results': [], 'partial_events': 0,
              'final_partial_events': 0, 'ai_chunks': [], 'final_insight': None, 'error_events': []}
    selected = turn_ids[0] if len(turn_ids) == 1 else None
    audio = bytearray()
    last_vad = None
    for received, e in events:
        kind = e.get('type')
        if kind in {'error','asr_error','tts_error'} or (kind == 'ai_response' and e.get('error')):
            result['error_events'].append(e)
        if kind == 'vad_end':
            last_vad = received
        if kind == 'asr_partial' and str(e.get('text') or '').strip():
            if e.get('final'):
                result['final_partial_events'] += 1
                result['final_partial_text'] = e['text']
                marks.setdefault('stream_final_partial', received)
            else:
                result['partial_events'] += 1
                marks.setdefault('first_partial', received)
        if kind == 'asr_result':
            result['observed_asr_results'].append(e)
        if not selected or str(e.get('turn_id') or '') != selected:
            continue
        if kind == 'asr_result':
            result['asr_text'] = str(e.get('text') or '')
            result['asr_source'] = e.get('source')
            marks.setdefault('asr_final', received)
            if last_vad is not None: marks.setdefault('vad_end', last_vad)
        elif kind in {'ai_response','ai_response_chunk'}:
            text = str(e.get('text') or '')
            if text.strip():
                marks.setdefault('first_ai_text', received)
                if kind == 'ai_response_chunk': result['ai_chunks'].append(text)
                else: result['final_ai_text'] = text
        elif kind == 'turn_insight' and e.get('state') == 'final':
            result['final_insight'] = e
        elif kind == 'tts_start':
            marks.setdefault('tts_start', received)
        elif kind in {'tts_chunk','tts_audio'}:
            encoded = e.get('chunk') or e.get('audio') or e.get('data')
            if encoded:
                decoded = base64.b64decode(encoded, validate=True)
                if decoded:
                    marks.setdefault('first_audio', received)
                    audio.extend(decoded)
                    result['audio_dtype'] = e.get('dtype') or result.get('audio_dtype') or 'float32'
                    result['audio_rate'] = e.get('sample_rate') or result.get('audio_rate') or 24000
        elif kind == 'tts_end':
            marks.setdefault('tts_end', received)
            result['tts_end_event'] = e
    result['ai_text'] = result.get('final_ai_text') or ''.join(result['ai_chunks'])
    return result, audio


async def run_trial(out, protocol, spec, cookie):
    _, db = test_paths(out)
    folder = out/'realtime_trials'/spec['trial_id']
    folder.mkdir(parents=True, exist_ok=False)
    result = {**spec, 'started_at': now(), 'status': 'failed', 'valid_for_latency': False}
    sample = next(s for s in protocol['samples'] if s['sample_id'] == spec['sample_id'])
    arm = ARMS[spec['arm']]
    result['reference_text'] = sample['reference_text']
    result['sample_sha256'] = sha(sample['audio_path'])
    marks, events, frame_log = {}, [], []
    recorder = producer = None
    audio = bytearray()
    before = state_snapshot(db, protocol['patient_id'])
    result['memory_sha256_before'] = before['tables_sha256']
    baseline = json.loads((out/'memory_state_before.json').read_text())['tables_sha256']
    try:
        if before['tables_sha256'] != baseline:
            raise RuntimeError('Frozen memory/profile state changed before trial')
        if result['sample_sha256'] != sample['prepared_wav_sha256']:
            raise RuntimeError('Frozen stimulus hash changed')
        with wave.open(sample['audio_path']) as w:
            pcm = w.readframes(w.getnframes())
        frames = [pcm[i:i+1024] for i in range(0, len(pcm), 1024)]
        async with _connect_websocket(SERVER.replace('http','ws')+'/ws', cookie, verify_tls=True) as ws:
            recorder = Recorder(ws, folder/'events.jsonl', profile=protocol['profile'])
            result['session'] = await _start_session(recorder, patient_id=protocol['patient_id'], create_patient=False,
                profile_name='合成实时消融实验', long_term_memory_enabled=arm['memory'],
                long_term_memory_writes_enabled=False, emotion_enabled=arm['emotion'],
                verify_effective_flags=True, timeout_s=30)
            profile_echo = (recorder.session_started_event or {}).get('profile')
            result['profile_echo'] = profile_echo
            if profile_echo != protocol['profile']:
                raise RuntimeError('Server profile differs from the frozen fixture')
            result['greeting'] = await wait_for_greeting_idle(recorder, timeout_s=45)
            recorder.submitted_at = time.perf_counter()
            done = asyncio.Event()
            producer = asyncio.create_task(stream_audio(recorder, frames, sample['last_energy_frame_index'], marks, frame_log, done))
            deadline = time.perf_counter() + len(frames)*DT + protocol['timeout_after_input_s']
            last_completed = None
            while True:
                limit = deadline
                if last_completed is not None and done.is_set():
                    limit = min(limit, last_completed + protocol['settle_s'])
                try:
                    e = await _receive_json(recorder, limit)
                except (TimeoutError, asyncio.TimeoutError):
                    if last_completed is not None and done.is_set():
                        break
                    raise RuntimeError('Timed out before a complete TTS response')
                t = recorder.last_received_at
                events.append((t, e))
                kind = e.get('type')
                if kind in {'error','asr_error','tts_error'} or (kind == 'ai_response' and e.get('error')):
                    raise RuntimeError('Real service returned '+str(kind))
                if kind == 'asr_result': last_completed = None
                if kind == 'tts_end': last_completed = t
            result.update({'session_events': recorder.event_count})
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if producer is not None:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        if recorder is not None:
            recorder.close()
        try:
            analyzed, audio = analyze_events(events, marks)
            result.update(analyzed)
        except Exception as exc:
            result['event_analysis_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['marks_perf_counter'] = marks
        result['metrics'] = metrics(marks)
        result['asr_cer'] = cer(result['reference_text'], result.get('asr_text', ''))
        result['output_text_chars'] = len(result.get('ai_text',''))
        stimulus_frames = [x for x in frame_log if x['is_stimulus']]
        result['pacing'] = {'expected_input_frames': sample['frame_count'], 'sent_input_frames': len(stimulus_frames),
                            'speech_frame_lag_ms': numeric_summary(x['lag_ms'] for x in stimulus_frames)}
        if len(stimulus_frames) > 1:
            result['pacing']['realtime_factor'] = (stimulus_frames[-1]['sent_perf'] - stimulus_frames[0]['sent_perf']) / ((len(stimulus_frames)-1)*DT)
        if audio:
            (folder/'tts_received.bin').write_bytes(audio)
            result['audio_bytes'] = len(audio)
            result['audio_sha256'] = hashlib.sha256(audio).hexdigest()
            if result.get('audio_dtype') == 'float32' and len(audio) % 4 == 0:
                floats = array('f'); floats.frombytes(audio)
                result['audio_finite'] = all(math.isfinite(v) for v in floats)
                result['audio_non_silent'] = result['audio_finite'] and max(abs(v) for v in floats) > 1e-6
                result['output_audio_duration_s'] = len(floats) / result['audio_rate']
                if result['audio_finite']:
                    pcm16 = array('h', (max(-32768,min(32767,int(v*32767))) for v in floats))
                    with wave.open(str(folder/'tts_response.wav'), 'wb') as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(result['audio_rate']); w.writeframes(pcm16.tobytes())
        after = state_snapshot(db, protocol['patient_id'])
        result['memory_sha256_after'] = after['tables_sha256']
        emotion = (result.get('final_insight') or {}).get('emotion', {})
        memory = (result.get('final_insight') or {}).get('memory', {})
        required = {'speech_end','vad_end','asr_final','first_ai_text','first_audio','tts_end'}
        result['integrity_checks'] = {
            'profile_frozen': result.get('profile_echo') == protocol['profile'],
            'greeting_idle_waited': (result.get('greeting') or {}).get('server_idle_wait_completed') is True,
            'memory_frozen': before['tables_sha256'] == after['tables_sha256'] == baseline,
            'single_asr_turn': len(result.get('observed_turn_ids', [])) == 1,
            'all_required_marks': required <= marks.keys(),
            'full_stimulus_sent': len(stimulus_frames) == sample['frame_count'],
            'pacing_within_limit': result['pacing']['speech_frame_lag_ms'].get('max', 1e9) <= protocol['maximum_speech_frame_lag_ms'],
            'real_streaming_final': result.get('partial_events',0) > 0 and result.get('final_partial_events',0) > 0
                and result.get('asr_source') != 'fallback_final'
                and normalized(result.get('final_partial_text','')) == normalized(result.get('asr_text','')),
            'real_nonempty_audio': bool(audio) and result.get('audio_finite') is True and result.get('audio_non_silent') is True,
            'normal_tts_completion': bool(result.get('tts_end_event')) and not (result.get('tts_end_event') or {}).get('reason'),
            'nonempty_reply': bool(result.get('ai_text','').strip()),
            'emotion_actual': (emotion.get('source') == 'emotion2vec_audio+text' and emotion.get('audio_model_used') is True)
                if arm['emotion'] else (emotion.get('source') == 'disabled' and emotion.get('audio_model_used') is False),
            'no_long_term_writes': memory.get('writes_enabled') is False and not memory.get('written_item_ids'),
            'nonnegative_pipeline_intervals': all(result['metrics'].get(k, -1) >= 0 for k in [
                'speech_end_to_vad_end_ms','vad_end_to_asr_final_ms','asr_final_to_first_ai_text_ms','first_ai_text_to_first_audio_ms']),
            'no_service_error': not result.get('error') and not result.get('error_events') and not result.get('event_analysis_error'),
        }
        if all(result['integrity_checks'].values()):
            result['status'] = 'ok'; result['valid_for_latency'] = True
        result['finished_at'] = now()
        save(folder/'frame_times.json', frame_log)
        save(folder/'result.json', result)
        print(json.dumps({'trial': spec['trial_id'], 'arm': spec['arm'], 'status': result['status'],
                          'speech_end_to_first_audio_ms': result['metrics'].get('speech_end_to_first_audio_ms'),
                          'bad_checks': [k for k,v in result['integrity_checks'].items() if not v],
                          'error': result.get('error')}, ensure_ascii=False), flush=True)
    return result


def summary(out):
    protocol = json.loads((out/'realtime_protocol.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((out/'realtime_trials').glob('*/result.json'))]
    formal = [r for r in records if r['stage'] == 'formal']
    groups = {}
    for arm in ARMS:
        rows = [r for r in formal if r['arm'] == arm]
        valid = [r for r in rows if r['valid_for_latency']]
        groups[arm] = {'attempted': len(rows), 'valid': len(valid), 'failed': len(rows)-len(valid),
                       'metrics': {k: numeric_summary(r['metrics'].get(k) for r in valid) for k in METRIC_PAIRS},
                       'asr_cer': numeric_summary(r['asr_cer'] for r in rows),
                       'output_text_chars': numeric_summary(r['output_text_chars'] for r in valid),
                       'output_audio_duration_s': numeric_summary(r.get('output_audio_duration_s') for r in valid),
                       'emotion_inference_ms': numeric_summary((r.get('final_insight') or {}).get('emotion',{}).get('inference_ms') for r in valid)}
    paired = {}
    for left, right in [('M1E1','M0E1'), ('M1E0','M0E0'), ('M1E1','M1E0'), ('M0E1','M0E0')]:
        pairs = []
        for block in sorted({r['block_id'] for r in formal}):
            row = {r['arm']: r for r in formal if r['block_id'] == block and r['valid_for_latency']}
            if left in row and right in row:
                pairs.append({k: row[left]['metrics'][k] - row[right]['metrics'][k]
                              for k in METRIC_PAIRS if k in row[left]['metrics'] and k in row[right]['metrics']})
        paired[left+' - '+right] = {'complete_pairs': len(pairs),
            'metrics': {k: numeric_summary(p.get(k) for p in pairs) for k in METRIC_PAIRS}}
    preflight = json.loads((out/'preflight_plan.json').read_text())
    result = {'schema_version': 'isolated-realtime-2x2-summary-v1', 'updated_at': now(),
              'protocol_sha256': sha(out/'realtime_protocol.json'), 'planned_formal': len(protocol['schedule']),
              'attempted_formal': len(formal), 'valid_formal': sum(r['valid_for_latency'] for r in formal),
              'warmup_attempted': sum(r['stage']=='warmup' for r in records),
              'warmup_valid': sum(r['stage']=='warmup' and r['valid_for_latency'] for r in records),
              'env_file_unchanged': sha(ROOT/'.env') == preflight['production_env_sha256'],
              'groups': groups, 'paired_deltas': paired,
              'failures': [{k:r.get(k) for k in ['trial_id','arm','error','integrity_checks']} for r in records if not r['valid_for_latency']]}
    save(out/'realtime_summary.json', result)
    return result


async def run(out, stage):
    out, _ = test_paths(out)
    protocol = json.loads((out/'realtime_protocol.json').read_text())
    import requests
    client = requests.Session(); client.trust_env = False
    creds = json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    response = client.post(SERVER+'/api/auth/login', json=creds, timeout=15)
    response.raise_for_status()
    cookie = '; '.join(f'{k}={v}' for k,v in response.cookies.items())
    if not cookie: raise RuntimeError('No test session cookie')
    if stage == 'formal':
        previous = summary(out)
        if previous['warmup_valid'] != len(protocol['warmup']):
            raise RuntimeError('All four warmup arms must pass before formal trials')
    specs = protocol['warmup'] if stage == 'warmup' else protocol['schedule']
    for spec in specs:
        if (out/'realtime_trials'/spec['trial_id']/'result.json').exists():
            raise RuntimeError('Refusing duplicate trial; recorded results are immutable')
        r = await run_trial(out, protocol, spec, cookie)
        summary(out)
        if not r['integrity_checks']['memory_frozen'] or not r['integrity_checks']['profile_frozen']:
            raise RuntimeError('Stopping study: profile or memory integrity failure')
        await asyncio.sleep(.3)
    x = summary(out)
    print(json.dumps({k:x[k] for k in ['attempted_formal','valid_formal','warmup_attempted','warmup_valid','env_file_unchanged']},ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','warmup','formal','summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    if args.stage == 'prepare': prepare(args.output, args.repeats)
    elif args.stage == 'summarize': print(json.dumps(summary(args.output),ensure_ascii=False,indent=2))
    else: asyncio.run(run(args.output, args.stage))


if __name__ == '__main__':
    main()
