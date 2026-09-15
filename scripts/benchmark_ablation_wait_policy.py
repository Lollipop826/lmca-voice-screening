#!/usr/bin/env python3
"""Production wait/cache classes with a deliberately synthetic delayed backend.

Measures only final-query waiting after an assumed stable ASR partial, NOT ASR,
real semantic retrieval, LLM, TTS, client playback or end-to-end voice latency.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import ssl
import statistics
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.context_management.retrieval_cache import RetrievalCache
from src.voice.realtime_companion import RealtimeCompanionConfig, RealtimeTurn

POLICIES={
    'no_cache_250':{'cache':False,'warm':False,'lead_ms':0,'budget_ms':250,'mismatch':False},
    'cold_cache_250':{'cache':True,'warm':False,'lead_ms':0,'budget_ms':250,'mismatch':False},
    'warm_cache_250':{'cache':True,'warm':True,'lead_ms':0,'budget_ms':250,'mismatch':False},
    'prefetch_match_250':{'cache':True,'warm':False,'lead_ms':200,'budget_ms':250,'mismatch':False},
    'prefetch_changed_query_250':{'cache':True,'warm':False,'lead_ms':200,'budget_ms':250,'mismatch':True},
    'cold_cache_100':{'cache':True,'warm':False,'lead_ms':0,'budget_ms':100,'mismatch':False},
    'cold_cache_500':{'cache':True,'warm':False,'lead_ms':0,'budget_ms':500,'mismatch':False},
    'prefetch_match_100':{'cache':True,'warm':False,'lead_ms':200,'budget_ms':100,'mismatch':False},
}


def quantile_nearest(values, fraction):
    if not values: return None
    ordered=sorted(values)
    return ordered[max(0,math.ceil(len(values)*fraction)-1)]


class DelayedMemoryService:
    def __init__(self, delay_ms, enabled):
        self.delay_ms=delay_ms
        self.cache=RetrievalCache(ttl_s=60 if enabled else 0,max_entries=128 if enabled else 0,
                                  max_inflight=2,wait_timeout_s=.25)
        self.lock=threading.Lock(); self.loader_calls=0; self.events=[]; self.sources=[]

    def get_turn_context(self, session, text):
        def loader():
            done=threading.Event()
            with self.lock:
                self.loader_calls+=1; self.events.append(done)
            try:
                time.sleep(self.delay_ms/1000)
                return 'controlled-memory-for:'+text
            finally: done.set()
        value,source=self.cache.get((session.lifecycle.current_patient_id,1,text),loader)
        with self.lock: self.sources.append(source)
        return value


async def measure_trial(case_id, delay_ms, policy_name):
    policy=POLICIES[policy_name]
    logs=[]
    session=SimpleNamespace(long_term_memory_enabled=True,
                            lifecycle=SimpleNamespace(current_patient_id='synthetic-cache-patient'))
    service=DelayedMemoryService(delay_ms,policy['cache'])
    companion=SimpleNamespace(config=RealtimeCompanionConfig(enabled=True,external_asr=True,
                                 memory_timeout_s=policy['budget_ms']/1000),
                              session=session,patient_memory_service=service,
                              _now=time.monotonic,_log=logs.append)
    turn=RealtimeTurn(companion)
    final_text='今天想接着聊上次说的散步安排'
    partial_text='今天不想聊上次说的散步安排' if policy['mismatch'] else final_text
    if policy['warm']:
        await asyncio.to_thread(service.get_turn_context,session,final_text)
    prewarm_calls=service.loader_calls
    actual_prefetch_started=False
    if policy['lead_ms']:
        now=time.monotonic()
        # Explicit assumption: this partial already survived the real 350 ms
        # stability gate. The fake timestamp is not a measured ASR observation.
        turn._partial_history.append((now-.36,partial_text))
        turn._maybe_prefetch(partial_text,now)
        actual_prefetch_started=turn._memory_task is not None
        await asyncio.sleep(policy['lead_ms']/1000)
    t=time.perf_counter()
    result=await turn.memory_for_final(final_text)
    wait_ms=(time.perf_counter()-t)*1000
    stale=bool(result) and result!='controlled-memory-for:'+final_text
    turn.cancel_memory_prefetch()
    # Cancelling asyncio.to_thread cannot terminate the loader; drain it outside
    # the measured waiting window so later trials cannot inherit resource load.
    await asyncio.sleep(0)
    for done in list(service.events):
        await asyncio.to_thread(done.wait,2)
    return {'case_id':case_id,'policy':policy_name,'loader_delay_ms':delay_ms,
            'budget_ms':policy['budget_ms'],'assumed_prefetch_lead_ms':policy['lead_ms'],
            'wait_ms':round(wait_ms,4),'returned_context':bool(result),
            'timeout_skipped':not bool(result),'stale_context':stale,
            'prefetch_started':actual_prefetch_started,
            'measured_phase_loader_calls':service.loader_calls-prewarm_calls,
            'prewarm_loader_calls':prewarm_calls,'cache_sources':service.sources,
            'log_count':len(logs)}


def make_trials(count, seed):
    rng=random.Random(seed)
    trials=[]
    for i in range(count):
        center=(80,200,400)[i%3]
        delay=round(center+rng.uniform(-.1*center,.1*center),3)
        order=list(POLICIES); rng.shuffle(order)
        for policy in order: trials.append((f'delay-{i+1:02d}',delay,policy))
    return trials


def environment_preflight(output):
    checks=[]
    for name,url in [('voice_health','https://127.0.0.1:8427/health'),
                     ('soulx','http://127.0.0.1:8001/health'),
                     ('memobase','http://127.0.0.1:18019/health')]:
        try:
            with urllib.request.urlopen(url,timeout=3,context=ssl._create_unverified_context()) as response:
                data=json.loads(response.read())
            # Only safe public health fields; no headers, tokens or DB content.
            checks.append({'name':name,'url':url,'reachable':True,
                           'status':data.get('status'),'turn_taking':data.get('turn_taking'),
                           'memory':data.get('memory')})
        except Exception as exc:
            checks.append({'name':name,'url':url,'reachable':False,'error_type':type(exc).__name__})
    manifest=ROOT/'tests/emotion_benchmark/testset/manifest.json'
    x=json.loads(manifest.read_text()); samples=x['samples']
    audio=sum(bool(s.get('audio_path')) and ((manifest.parent/str(s['audio_path'])).is_file()
                   or Path(str(s['audio_path'])).is_file()) for s in samples)
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'service_checks':checks,
            'emotion_manifest':{'samples':len(samples),'source_counts':dict(Counter(s.get('source') for s in samples)),
                                'available_audio_files':audio},
            'live_service_restarted':False,'live_configuration_modified':False,'live_database_written':False}
    (output/'environment_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


async def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    environment_preflight(args.output)
    specs=make_trials(args.trials,args.seed); records=[]
    for i,spec in enumerate(specs,1):
        records.append(await measure_trial(*spec))
        if i%24==0: print(f'wait microbenchmark {i}/{len(specs)}',flush=True)
    summary={}
    for policy in POLICIES:
        rows=[r for r in records if r['policy']==policy]
        waits=[r['wait_ms'] for r in rows]
        summary[policy]={'trials':len(rows),'p50_wait_ms':round(statistics.median(waits),3),
                         'p95_wait_ms':round(quantile_nearest(waits,.95),3),
                         'context_return_rate':sum(r['returned_context'] for r in rows)/len(rows),
                         'timeout_skip_rate':sum(r['timeout_skipped'] for r in rows)/len(rows),
                         'stale_context_count':sum(r['stale_context'] for r in rows),
                         'mean_measured_loader_calls':statistics.fmean(r['measured_phase_loader_calls'] for r in rows)}
    payload={'schema_version':'ablation-synthetic-wait-microbenchmark-v1',
             'created_at':datetime.now(timezone.utc).isoformat(),'seed':args.seed,
             'policies':POLICIES,'delay_cases':args.trials,'total_trials':len(records),
             'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/'src/context_management/retrieval_cache.py',ROOT/'src/voice/realtime_companion.py',Path(__file__)]},
             'claim_boundary':'Actual production RetrievalCache and RealtimeTurn.memory_for_final code with a synthetic sleep-based memory backend. No real network retrieval, ASR, Emotion2Vec, LLM, TTS, playback or clinical effect.',
             'timing_origin':'Entry into memory_for_final; prefetch lead and prewarm are excluded from the final-wait timer.',
             'prefetch_assumption':'Artificial stable partial available 200 ms before final; real 350 ms stability predicate invoked with seeded history, not measured speech.',
             'summary':summary,'records':records}
    (args.output/'wait_policy_microbenchmark.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/ablation_study_20260913')
    parser.add_argument('--trials',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20260913)
    args=parser.parse_args()
    if args.trials<3: parser.error('At least 3 trials required')
    asyncio.run(run(args))


if __name__=='__main__': main()
