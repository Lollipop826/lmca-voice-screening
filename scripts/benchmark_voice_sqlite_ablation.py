#!/usr/bin/env python3
"""Auditable isolated, real-ASR/LLM/TTS/E2V, SQLite-card 2x2 ablation.

No fake outputs, no hidden retries, no real-patient data. `manual_audio` is a
batch-input protocol, NOT microphone end-of-speech or SoulX latency.
Only accepts the test-only DB created by tmp/real-voice-ablation-20260913/launch_isolated.py.
"""
from __future__ import annotations
import argparse, asyncio, base64, hashlib, json, math, os, random, re, sqlite3, sys, time, wave
from array import array
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_voice_realtime import _connect_websocket, _start_session, _receive_json

ARMS = {
    'M1E1': {'memory': True, 'emotion': True},
    'M0E1': {'memory': False, 'emotion': True},
    'M1E0': {'memory': True, 'emotion': False},
    'M0E0': {'memory': False, 'emotion': False},
}
SERVER = 'http://127.0.0.1:18427'

def now(): return datetime.now().astimezone().isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def stable_hash(obj): return sha(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode())
def test_paths(out):
    out = out.resolve()
    if out.parent != ROOT/'output' or not out.name.startswith('real_voice_ablation_'):
        raise ValueError('Only dedicated real_voice_ablation_* output directories are accepted')
    db = out/'isolated_runtime/test_only.sqlite3'
    if not db.is_file(): raise ValueError('Isolated test DB must already exist')
    if not (out/'preflight_plan.json').is_file(): raise ValueError('Missing isolation preflight')
    return out, db

def state_snapshot(db, patient_id):
    result = {}
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name): continue
            if 'memory' not in name and name not in {'patients','emotion_memobase_snapshots','emotion_memobase_turns'}: continue
            cols = {r['name'] for r in conn.execute(f'PRAGMA table_info("{name}")')}
            if 'patient_id' in cols:
                rows = [dict(r) for r in conn.execute(f'SELECT * FROM "{name}" WHERE patient_id=? ORDER BY rowid', (patient_id,))]
                if name == 'patients':
                    rows=[{k:r.get(k) for k in ('patient_id','name','gender','age','education_years','extra_profile')} for r in rows]
                result[name] = rows
    return {'captured_at': now(), 'tables': result, 'tables_sha256': stable_hash(result)}

def prepare(out):
    out, db = test_paths(out)
    if (out/'protocol.json').exists(): raise RuntimeError('Frozen protocol already exists; will not overwrite')
    # Imports can initialize DB helpers: explicitly isolate BEFORE importing.
    os.environ.update(DB_PATH=str(db), MEMOBASE_PROJECT_URL='', MEMOBASE_API_KEY='',
                      MEMORY_LOCAL_FALLBACK_ENABLED='false', USE_LOCAL_EMBEDDING='false')
    from scripts.prepare_memory_ablation_fixture import prepare_fixture
    from src.context_management.emotion_memobase import EmotionMemobase
    fixture = json.loads((ROOT/'tests/fixtures/memory_ablation_cases.json').read_text())
    fixture['fixture_id'] = out.name+'-sqlite-real-audio-v1'
    fixture['patient_id'] = 'pt-synthetic-'+out.name.removeprefix('real_voice_ablation_')
    seeded = prepare_fixture(fixture, db_path=str(db), apply=True)
    save(out/'fixture.json', fixture)
    save(out/'fixture_seed_result.json', seeded)
    memory = EmotionMemobase(db_path=str(db), emotion_classifier=lambda _text: {}, logger=lambda _msg: None)
    card = memory.get_authoritative_card(fixture['patient_id'])
    (out/'authoritative_memory_card.txt').write_text(card, encoding='utf-8')
    snapshot = state_snapshot(db, fixture['patient_id'])
    save(out/'memory_state_before.json', snapshot)
    manifest = ROOT/'output/memory_ablation_audio_manifest.json'
    samples = []
    for item in json.loads(manifest.read_text())['samples']:
        p = (manifest.parent/item['audio_path']).resolve()
        with wave.open(str(p)) as w:
            assert w.getnchannels()==1 and w.getframerate()==16000 and w.getsampwidth()==2
            pcm = w.readframes(w.getnframes())
            dur = w.getnframes()/w.getframerate()
        samples.append({**item, 'audio_path': str(p), 'wav_sha256': sha(p.read_bytes()),
                        'pcm16_sha256': sha(pcm), 'input_audio_duration_s': dur,
                        'stimulus_provenance': 'pre-existing synthesized TTS fixture, NOT a patient recording',
                        'emotion_ground_truth': None})
    rng = random.Random(20260913)
    blocks = [(rep, sample) for rep in range(1,4) for sample in samples]
    rng.shuffle(blocks)
    schedule = []
    for rep, sample in blocks:
        arms = list(ARMS); rng.shuffle(arms)
        for arm in arms:
            schedule.append({'trial_id': f'study-{len(schedule)+1:03d}', 'repeat': rep,
                             'sample_id':sample['sample_id'], 'arm':arm})
    warmup = [{'trial_id':f'warmup-{i+1:02d}', 'repeat':0, 'sample_id':samples[0]['sample_id'], 'arm':arm}
              for i,arm in enumerate(ARMS)]
    protocol = {
        'version':'sqlite-card-real-audio-1', 'frozen_at':now(), 'server':SERVER,
        'patient_id':fixture['patient_id'], 'profile':fixture['profile'], 'arms':ARMS, 'samples':samples,
        'design': {'independent_scenarios':4, 'arms':4, 'repeats_per_cell':3,
                   'formal_attempts':48, 'warmup_excluded':4, 'schedule_seed':20260913,
                   'sequential_no_concurrency':True, 'new_session_every_attempt':True,
                   'greeting_audio_drained_before_input':True, 'memory_writes':False, 'hidden_retries':False, 'server_internal_retries':'existing ASR/TTS retry behavior retained and logged', 'timeout_s':100,
                   'post_completion_drain_s':0.6, 'cooldown_s':1.5},
        'memory_state_sha256':snapshot['tables_sha256'], 'card_sha256':sha(card.encode()),
        'runner_sha256_at_freeze':sha(Path(__file__).read_bytes()),
        'schedule':schedule, 'warmup':warmup,
        'components':{'ASR':'real configured cloud BigASR', 'LLM':'real configured cloud LLM',
                      'TTS':'real configured cloud ArkTTS', 'emotion':'real cached Emotion2Vec+ large on GPU',
                      'memory':'real persisted SQLite authoritative-card read path; fixture seeding is direct setup'},
        'boundaries':[
            'manual_audio batch submission, not realtime microphone streaming; no SoulX measured',
            'Memobase offline; semantic retrieval intentionally disabled; card != semantic retrieval',
            'Synthetic stimuli have no validated vocal-emotion labels; cannot estimate emotion accuracy',
            'Only 4 scenarios; repeated generations are NOT 48 independent users/scenarios',
            'Cloud LLM stochastic generation; longer replies confound time-to-complete',
            'Emotion model execution alone does not prove causal improvement to a response',
        ],
        'metric_definitions':{
            'input_to_asr_ms':'client submit start -> first matching non-empty ASR text event',
            'vad_end_to_asr_ms':'manual handler VAD_END event -> ASR result; not true speech endpoint latency',
            'asr_to_first_ai_event_ms':'ASR result -> first non-empty AI sentence event; NOT internal first token',
            'first_ai_to_tts_start_ms':'first AI sentence -> TTS start notification',
            'tts_start_to_first_audio_ms':'TTS start notification -> first NONEMPTY decoded audio bytes',
            'first_ai_to_first_audio_ms':'first AI sentence -> first NONEMPTY decoded audio bytes',
            'input_to_first_audio_ms':'batch submit start -> client receive first audio bytes; NOT speaker playback',
            'input_to_tts_end_ms':'batch submit start -> synthesis stream complete; NOT playback complete',
        },
        'prespecified_quality_audit':[
            'Inspect ASR errors, observed actual emotion source, and fixed-memory integrity',
            'Describe references to fixture-only historical facts separately from current-input paraphrases',
            'Audit unrelated-control for off-topic historical insertions',
            'No opaque LLM-judge score is used as a primary result; no invented user satisfaction',
        ],
    }
    save(out/'protocol.json', protocol)
    creds = json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    import requests
    client = requests.Session(); client.trust_env = False
    response = client.post(SERVER+'/api/auth/register', json={**creds,'display_name':'隔离消融实验测试账户'}, timeout=15)
    if response.status_code not in (200,201):
        raise RuntimeError(f'Isolated registration failed: HTTP {response.status_code}')
    health = client.get(SERVER+'/health', timeout=10).json()
    save(out/'isolated_health.json', health)
    print(json.dumps({'prepared':str(out),'formal_trials':len(schedule),'memory_items':len(seeded['memory_items']),
                      'card_chars':len(card),'protocol_sha256':sha((out/'protocol.json').read_bytes())}, ensure_ascii=False))

class Recorder:
    def __init__(self, websocket, path, profile=None):
        self.profile=profile; self.session_started_event=None
        self.websocket=websocket; self.file=path.open('w',encoding='utf-8'); self.submitted_at=None
        self.last_received_at=None; self.event_count=0
    async def recv(self):
        raw=await self.websocket.recv(); self.last_received_at=time.perf_counter()
        row={'received_at':now(),'perf_counter':self.last_received_at,
             'ms_since_submit':None if self.submitted_at is None else (self.last_received_at-self.submitted_at)*1000}
        if isinstance(raw, bytes): row['raw_binary_base64']=base64.b64encode(raw).decode()
        else:
            try:
                row['event']=json.loads(raw)
                if isinstance(row['event'],dict) and row['event'].get('type')=='session_started':
                    self.session_started_event=row['event']
            except json.JSONDecodeError: row['raw_text']=raw
        self.file.write(json.dumps(row,ensure_ascii=False)+'\n'); self.file.flush(); self.event_count+=1
        return raw
    async def send(self, raw):
        if self.profile is not None and isinstance(raw,str):
            event=json.loads(raw)
            if event.get('type')=='start_session':
                event['profile']=dict(self.profile)
                raw=json.dumps(event,ensure_ascii=False)
        return await self.websocket.send(raw)
    def close(self): self.file.close()

def calc_metrics(marks, base):
    result={}
    for key,value in marks.items(): result['input_to_'+key+'_ms']=round((value-base)*1000,3)
    pairs={
        'vad_end_to_asr_ms':('vad_end','asr'),
        'asr_to_first_ai_event_ms':('asr','first_ai_event'),
        'first_ai_to_tts_start_ms':('first_ai_event','tts_start'),
        'tts_start_to_first_audio_ms':('tts_start','first_audio'),
        'first_ai_to_first_audio_ms':('first_ai_event','first_audio'),
        'asr_to_first_audio_ms':('asr','first_audio'),
    }
    for label,(a,b) in pairs.items():
        if a in marks and b in marks: result[label]=round((marks[b]-marks[a])*1000,3)
    return result

async def wait_for_greeting(recorder, timeout_s=35):
    """Existing patients ALSO receive greetings in the current server version.

    Wait for actual nonempty greeting audio and its matching normal TTS_END,
    before starting the measured trial. Never overlap setup TTS with input.
    """
    deadline=time.perf_counter()+timeout_s
    greeting_id='';audio_bytes=0;start=time.perf_counter()
    while True:
        event=await _receive_json(recorder,deadline)
        kind=event.get('type');eid=str(event.get('turn_id') or '')
        if kind in {'error','tts_error'}:
            raise RuntimeError('Greeting service error: '+str(kind))
        if kind=='tts_start': greeting_id=eid
        if kind in {'tts_chunk','tts_audio'} and eid==greeting_id:
            payload=event.get('chunk') or event.get('audio') or event.get('data')
            if payload: audio_bytes+=len(base64.b64decode(payload,validate=True))
        if kind=='tts_end' and eid==greeting_id:
            if event.get('reason') or not audio_bytes:
                raise RuntimeError('Greeting failed; refusing a contaminated trial')
            return {'turn_id':greeting_id,'audio_bytes':audio_bytes,
                    'wait_ms':round((time.perf_counter()-start)*1000,3)}

async def run_trial(out, protocol, trial, cookie):
    sample=next(s for s in protocol['samples'] if s['sample_id']==trial['sample_id'])
    arm=ARMS[trial['arm']]
    folder=out/'trials'/trial['trial_id']; folder.mkdir(parents=True,exist_ok=False)
    result={**trial,'started_at':now(),'status':'failed','sample':sample,'events_path':str(folder/'events.jsonl')}
    marks={}; audio=bytearray(); chunks=[]; final_text=''; final_insight=None; turn_id=''; recorder=None
    base=None; audio_dtype=None; audio_rate=None
    try:
        p=Path(sample['audio_path'])
        if sha(p.read_bytes())!=sample['wav_sha256']: raise ValueError('Input wave changed since freeze')
        with wave.open(str(p)) as w: pcm=w.readframes(w.getnframes())
        async with _connect_websocket('ws://127.0.0.1:18427/ws',cookie,verify_tls=True) as ws:
            recorder=Recorder(ws,folder/'events.jsonl',profile=protocol['profile'])
            setup_start=time.perf_counter()
            result['session']=await _start_session(recorder,patient_id=protocol['patient_id'],create_patient=False,
                profile_name='隔离合成实验用户',long_term_memory_enabled=arm['memory'],
                long_term_memory_writes_enabled=False,emotion_enabled=arm['emotion'],verify_effective_flags=True,timeout_s=30)
            result['session_start_ms']=round((time.perf_counter()-setup_start)*1000,3)
            actual_profile=(recorder.session_started_event or {}).get('profile')
            result['profile_echo']=actual_profile
            if actual_profile != protocol['profile']:
                raise RuntimeError('Server profile differs from frozen profile')
            result['greeting']=await wait_for_greeting(recorder)
            # JSON serialization is prepared before the measured submission begins.
            payload=json.dumps({'type':'manual_audio','audio':base64.b64encode(pcm).decode(),'sample_rate':16000})
            base=time.perf_counter(); recorder.submitted_at=base
            await recorder.send(payload)
            result['input_send_ms']=round((time.perf_counter()-base)*1000,3)
            deadline=base+protocol['design']['timeout_s']
            while True:
                complete='tts_end' in marks and final_insight is not None
                recv_deadline=min(deadline,time.perf_counter()+protocol['design']['post_completion_drain_s']) if complete else deadline
                try: event=await _receive_json(recorder,recv_deadline)
                except (TimeoutError,asyncio.TimeoutError):
                    if complete: break
                    raise
                received=recorder.last_received_at; kind=event.get('type'); eid=str(event.get('turn_id') or '')
                if kind in {'error','asr_error','tts_error'}:
                    result.setdefault('error_events',[]).append(event)
                    raise RuntimeError('Service error event: '+str(kind))
                if kind=='vad_end': marks.setdefault('vad_end',received)
                if kind=='asr_result':
                    if turn_id and eid!=turn_id: raise RuntimeError('Unexpected second ASR turn from one manual input')
                    if not eid or not str(event.get('text') or '').strip(): raise RuntimeError('Empty ASR text or turn id')
                    turn_id=eid; result['turn_id']=turn_id; result['asr_text']=event.get('text')
                    result['asr_event']=event; marks.setdefault('asr',received)
                if not turn_id or eid!=turn_id: continue
                if kind=='turn_insight' and event.get('state')=='final': final_insight=event
                elif kind=='ai_response_chunk':
                    text=str(event.get('text') or '')
                    if text.strip(): marks.setdefault('first_ai_event',received); chunks.append(text)
                elif kind=='ai_response':
                    if event.get('error'): raise RuntimeError('AI returned error response')
                    text=str(event.get('text') or '')
                    if text.strip(): marks.setdefault('first_ai_event',received); final_text=text
                    result['ai_response_event']=event
                elif kind=='tts_start':
                    marks.setdefault('tts_start',received); result.setdefault('tts_start_event',event)
                elif kind in {'tts_chunk','tts_audio'}:
                    payload=event.get('chunk') or event.get('audio') or event.get('data')
                    if payload:
                        decoded=base64.b64decode(payload,validate=True)
                        if decoded:
                            marks.setdefault('first_audio',received); audio.extend(decoded)
                            audio_dtype=event.get('dtype') or audio_dtype
                            audio_rate=event.get('sample_rate') or audio_rate
                elif kind=='tts_end':
                    marks.setdefault('tts_end',received); result['tts_end_event']=event
                    if event.get('reason'): raise RuntimeError('TTS ended with non-normal reason: '+str(event['reason']))
            result['ai_text']=final_text or ''.join(chunks)
            result['ai_chunks']=chunks
            result['final_insight']=final_insight
            result['event_count']=recorder.event_count
            needed={'asr','first_ai_event','tts_start','first_audio','tts_end'}
            missing=needed-set(marks)
            if missing: raise RuntimeError('Missing completion marks: '+','.join(sorted(missing)))
            emotion=(final_insight or {}).get('emotion',{})
            memory=(final_insight or {}).get('memory',{})
            checks={
                'all_flags_echoed_as_requested':True,
                'emotion_audio_verified': emotion.get('audio_model_used') is True and emotion.get('source')=='emotion2vec_audio+text' if arm['emotion'] else emotion.get('audio_model_used') is False and emotion.get('analysis_status')=='disabled',
                'memory_writes_disabled':memory.get('writes_enabled') is False,
                'no_memory_items_written':not memory.get('written_item_ids'),
                'received_audio':bool(audio),
                'reply_nonempty':bool(result['ai_text'].strip()),
            }
            result['integrity_checks']=checks
            if not all(checks.values()): raise RuntimeError('Component verification failed: '+','.join(k for k,v in checks.items() if not v))
            result['status']='ok'
    except Exception as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        if recorder: recorder.close()
        if base is not None: result['metrics']=calc_metrics(marks,base)
        result['ai_text']=result.get('ai_text') or final_text or ''.join(chunks)
        result['final_insight']=result.get('final_insight') or final_insight
        result['output_text_chars']=len(result['ai_text'])
        if audio:
            # Preserve received native bytes even if encoding verification fails.
            native=folder/'tts_received.bin'; native.write_bytes(audio)
            result['tts_native']={'path':str(native),'bytes':len(audio),'sha256':sha(audio),
                                  'dtype':audio_dtype,'sample_rate':audio_rate}
            rate=int(audio_rate or 24000)
            if audio_dtype in (None,'float32','f32') and len(audio)%4==0:
                floats=array('f'); floats.frombytes(audio)
                if sys.byteorder!='little': floats.byteswap()
                if all(math.isfinite(x) for x in floats):
                    pcm16=array('h',(round(max(-1.0,min(1.0,x))*32767) for x in floats))
                    if sys.byteorder!='little': pcm16.byteswap()
                    wavpath=folder/'tts_response.wav'
                    with wave.open(str(wavpath),'wb') as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm16.tobytes())
                    result['output_audio_path']=str(wavpath); result['output_audio_duration_s']=len(floats)/rate
        result['finished_at']=now()
        save(folder/'result.json',result)
    return result

async def run(out,phase):
    out,db=test_paths(out)
    protocol=json.loads((out/'protocol.json').read_text())
    if phase=='study':
        warmup_path=out/'warmup_results.jsonl'
        warmup=[json.loads(x) for x in warmup_path.read_text().splitlines()] if warmup_path.exists() else []
        if len(warmup)!=4 or any(x['status']!='ok' for x in warmup):
            raise RuntimeError('Require all 4 verified warmup arms before formal study')
    creds=json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    import requests
    client=requests.Session(); client.trust_env=False
    login=client.post(SERVER+'/api/auth/login',json=creds,timeout=15); login.raise_for_status()
    cookie='; '.join(f'{k}={v}' for k,v in login.cookies.items())
    if not cookie: raise RuntimeError('No login cookie')
    if 'profile' not in protocol:
        raise RuntimeError('Old pilot protocol lacks frozen profile; create a new isolated run')
    schedule=protocol['warmup' if phase=='warmup' else 'schedule']
    path=out/f'{phase}_results.jsonl'
    existing=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={r['trial_id'] for r in existing}
    failures=0
    with path.open('a',encoding='utf-8') as stream:
        for trial in schedule:
            if trial['trial_id'] in done: continue
            before=state_snapshot(db,protocol['patient_id'])
            if before['tables_sha256']!=protocol['memory_state_sha256']:
                save(out/'state_drift_abort.json',before)
                raise RuntimeError('Frozen memory/profile drift detected before trial; aborting')
            result=await run_trial(out,protocol,trial,cookie)
            after=state_snapshot(db,protocol['patient_id'])
            drift=after['tables_sha256']!=protocol['memory_state_sha256']
            if drift:
                result['status']='failed'; result['state_drift_detected']=True
                save(out/'state_drift_abort.json',after)
                save(out/'trials'/trial['trial_id']/'result.json',result)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n'); stream.flush()
            if drift: raise RuntimeError('Frozen memory/profile changed during trial; aborting')
            print(json.dumps({'trial':trial['trial_id'],'arm':trial['arm'],'sample':trial['sample_id'],
                              'status':result['status'],'error':result.get('error'),
                              'asr':result.get('asr_text'),'first_audio_ms':result.get('metrics',{}).get('input_to_first_audio_ms'),
                              'emotion':(result.get('final_insight') or {}).get('emotion')},ensure_ascii=False),flush=True)
            failures=failures+1 if result['status']!='ok' else 0
            if failures>=3: print('Aborted: 3 consecutive failures; no synthetic replacement.',flush=True); break
            await asyncio.sleep(protocol['design']['cooldown_s'])
    save(out/f'memory_state_after_{phase}.json',state_snapshot(db,protocol['patient_id']))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','warmup','study'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='prepare': prepare(args.output)
    else: asyncio.run(run(args.output,args.phase))
if __name__=='__main__': main()
