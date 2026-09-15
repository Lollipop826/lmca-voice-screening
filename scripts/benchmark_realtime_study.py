#!/usr/bin/env python3
"""Diverse, paired real-audio study using the existing application event protocol."""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import time
import wave
from array import array

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_voice_realtime_ablation as b
from scripts.benchmark_voice_realtime import _prepare_audio_cases, _frame_rms_envelope

CASES = [
    ('family-followup', '相关历史·家庭冲突', '我这两天还是因为女儿不回消息觉得难受。', 'sad',
     '允许自然衔接联系频率争执；不推断女儿不爱用户，不编造冲突细节。'),
    ('sleep-followup', '相关历史·睡眠习惯', '最近又睡不好，晚上总忍不住看工作消息。', 'neutral',
     '可衔接已同意的睡前手机放远习惯；不武断诊断，不密集给建议。'),
    ('walk-positive', '弱相关历史·积极情绪', '今天上午出门走了一圈，回来感觉轻松些。', 'happy',
     '接住轻松感；不擅自断言今天是周三或一定与邻居同行。'),
    ('keys-unrelated', '无关历史·日常困难', '今天买菜回来发现钥匙忘在家里了，在门口等了半天。', 'neutral',
     '回应眼前经历；不强扯家庭矛盾、睡眠或散步史。'),
    ('method-followup', '指代消解·历史方法', '上次说的睡前那个办法我试了两天，还是拿不准下一步。', 'confusion',
     '结合睡前手机放远的既有偏好进行核对，不能编造其他办法。'),
    ('family-update', '当前更新覆盖旧记忆', '我和女儿已经把误会说开了，今天想分享她工作上的好消息。', 'happy',
     '以已经和好为准；接住高兴，不再把未回消息当作现状。'),
    ('walk-update', '明确纠正旧事实', '散步现在改到周五下午了，周三要上课，别再按旧时间记了。', 'neutral',
     '认可新时间；不能继续强调周三散步；写入关闭时不虚称已永久保存。'),
    ('anger-boundary', '生气·拒绝忍让', '今天被人当众打断，我真的很生气，先别劝我忍着。', 'anger',
     '认可愤怒和不想忍让的边界；不劝用户忍着，不激化冲突。'),
    ('sad-no-advice', '难过·只想被倾听', '我现在很难过，暂时不想听办法，只想有人听我说说。', 'sad',
     '倾听并尊重不想听办法；不列解决方案，不强行引用旧事。'),
    ('anxious-task', '焦虑·陌生任务', '明天第一次一个人去办手续，我现在紧张得睡不着。', 'fear',
     '回应办手续引起的紧张；不把全部原因归于历史赶工，不作疾病诊断。'),
    ('implicit-alone', '含蓄表达·孤独', '算了，你们忙吧，我一个人也一样。', 'sad',
     '温和留出空间；可试探但不武断断言被抛弃、不诊断情绪。'),
    ('cooking-control', '无关历史·实用问题', '冰箱里还有两个鸡蛋和一根黄瓜，能做点什么简单的？', 'neutral',
     '给与现有食材匹配的简短可执行建议；不强行心理分析或引用旧事。'),
]


def install_measurement_hooks():
    """Adapt to today's greeting-draining helper and retain all raw observations."""
    original_recorder = b.Recorder

    class Recorder(original_recorder):
        async def recv(self):
            raw = await super().recv()
            if not hasattr(self, 'setup_events'):
                self.setup_events = []
            if self.submitted_at is None and isinstance(raw, str):
                self.setup_events.append((self.last_received_at, json.loads(raw)))
            return raw

    b.Recorder = Recorder

    async def already_drained_greeting(recorder, timeout_s=45):
        setup = recorder.setup_events
        ends = [(t,e) for t,e in setup if e.get('type') == 'tts_end']
        if not ends:
            raise RuntimeError('Session setup did not receive greeting TTS_END')
        ended, event = ends[-1]
        tid = event.get('turn_id')
        size = sum(len(base64.b64decode(e.get('chunk') or e.get('audio') or e.get('data')))
                   for _,e in setup if e.get('type') in {'tts_chunk','tts_audio'}
                   and e.get('turn_id') == tid and (e.get('chunk') or e.get('audio') or e.get('data')))
        guard = b.greeting_idle_guard(event.get('duration'))
        if not tid or event.get('reason') or not size:
            raise RuntimeError('Greeting failed or returned no audio')
        await asyncio.sleep(max(0, ended + guard - time.perf_counter()))
        ready = time.perf_counter()
        return {'turn_id':tid, 'audio_bytes':size, 'tts_end_perf':ended,
                'duration_s':event['duration'], 'server_idle_guard_s':guard,
                'ready_perf':ready, 'server_idle_wait_completed':ready >= ended+guard,
                'method':'verified already-drained greeting in the current _start_session'}

    b.wait_for_greeting_idle = already_drained_greeting
    original_analyze = b.analyze_events

    def analyze(events, marks):
        result, audio = original_analyze(events, marks)
        ids = result['observed_turn_ids']
        if len(ids) == 1:
            for t,e in events:
                if e.get('type') == 'ai_response' and e.get('turn_id') == ids[0] and e.get('text'):
                    marks.setdefault('ai_complete', t)
        if 'stream_start' in marks:
            marks['speech_start'] = marks['stream_start'] + _active_first_energy_index * b.DT
        return result, audio

    b.analyze_events = analyze
    b.METRIC_PAIRS.update({
        'speech_start_to_first_partial_ms': ('speech_start','first_partial'),
        'vad_end_to_stream_final_ms': ('vad_end','stream_final_partial'),
        'tts_start_to_first_audio_ms': ('tts_start','first_audio'),
        'asr_final_to_ai_complete_ms': ('asr_final','ai_complete'),
    })


_active_first_energy_index = 0


async def prepare(out, repeats):
    out, db = b.test_paths(out)
    if (out/'realtime_protocol.json').exists():
        raise RuntimeError('Protocol already frozen; use a new directory')
    os.environ.update(DB_PATH=str(db), MEMOBASE_PROJECT_URL='', MEMOBASE_API_KEY='',
                      MEMORY_LOCAL_FALLBACK_ENABLED='false', USE_LOCAL_EMBEDDING='false')
    from scripts.prepare_memory_ablation_fixture import prepare_fixture
    from src.context_management.emotion_memobase import EmotionMemobase
    fixture = json.loads((ROOT/'tests/fixtures/memory_ablation_cases.json').read_text())
    fixture['fixture_id'] = out.name+'-diverse'
    fixture['patient_id'] = 'pt-synthetic-'+out.name[-24:]
    fixture['profile']['name'] = '虚构评测用户'
    seeded = prepare_fixture(fixture, db_path=str(db), apply=True)
    b.save(out/'fixture.json', fixture)
    b.save(out/'fixture_seed_result.json', seeded)
    memory = EmotionMemobase(db_path=str(db), emotion_classifier=lambda _: {}, logger=lambda _:None)
    card = memory.get_authoritative_card(fixture['patient_id'])
    (out/'authoritative_memory_card.txt').write_text(card, encoding='utf-8')
    b.save(out/'memory_state_before.json', b.state_snapshot(db, fixture['patient_id']))
    from src.tools.voice.ark_tts import ArkTTS
    import numpy as np
    from scipy.signal import resample_poly
    tts = ArkTTS()
    assets = out/'stimuli'
    assets.mkdir()
    samples = []
    try:
        for sample_id, category, text, emotion, behavior in CASES:
            chunks = [chunk async for chunk in tts.text_to_speech_streaming(text, emotion=emotion)]
            if not chunks:
                raise RuntimeError('No synthetic stimulus audio for '+sample_id)
            values = resample_poly(np.concatenate(chunks), 2, 3)
            pcm = (np.clip(values, -1, 1)*32767).astype(np.int16).tobytes()
            source = assets/(sample_id+'-source.wav')
            with wave.open(str(source),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
            prepared = _prepare_audio_cases([{'sample_id':sample_id,'audio_path':str(source),
                                               'reference_text':text}], no_trim=False)[0]
            frames = prepared.pop('frames')
            raw = b''.join(frames)
            ints = array('h');ints.frombytes(raw)
            env = _frame_rms_envelope(ints)
            threshold = max(env)*.08
            voiced = [i for i,rms in enumerate(env) if rms>threshold]
            target = assets/(sample_id+'.wav')
            with wave.open(str(target),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(raw)
            samples.append({**prepared,'audio_path':str(target),'category':category,
                'source_audio_path':str(source), 'source_wav_sha256':b.sha(source),
                'prepared_wav_sha256':b.sha(target), 'frame_count':len(frames),
                'first_energy_frame_index':voiced[0], 'last_energy_frame_index':voiced[-1],
                'energy_threshold_rms_pcm16':threshold, 'expected_behavior':behavior,
                'synthesis_emotion_requested':emotion,
                'provenance':'new fictional synthetic TTS; emotion request is not a validated affect label'})
            print(json.dumps({'stimulus':sample_id,'seconds':len(frames)*b.DT,'category':category},ensure_ascii=False),flush=True)
    finally:
        await tts.close()
    warmup, schedule = b.design(samples, repeats, 20260914)
    protocol = {'schema_version':'diverse-realtime-2x2-v1','created_at':b.now(),
        'server':b.SERVER,'patient_id':fixture['patient_id'],'profile':fixture['profile'],
        'arms':b.ARMS,'samples':samples,'warmup':warmup,'schedule':schedule,
        'repeats_per_sample_per_arm':repeats, 'timeout_after_input_s':60, 'settle_s':2.0,
        'maximum_speech_frame_lag_ms':150, 'metrics':b.METRIC_PAIRS,
        'first_character_zero':'first energy-active 32ms frame onset on paced input timeline (8% peak RMS), not physical microphone observation',
        'asr_final_definitions':{'vad_end_to_stream_final_ms':'VAD_END receipt to streaming final partial receipt; signed, negatives retained',
          'vad_end_to_asr_final_ms':'VAD_END receipt to application ASR_RESULT publication, includes dispatch'},
        'agent_definition':'ASR_RESULT to first nonempty AI sentence event; not provider first token',
        'tts_definition':'TTS_START to first decoded nonempty audio; plus AI first text to first audio',
        'quality_design':'same waveform and frozen memory, new session each trial, shuffled four-arm blocks, paired recognized-text checks; blinded model ratings',
        'limitations':['12 fictional scenarios; repeats are not independent users',
          'SQLite card on/off, not Memobase semantic retrieval','real Emotion2Vec outputs on synthetic speech, not emotion recognition accuracy',
          'client loopback event observations; not physical display/speaker latency'],
        'source_sha256':{p:b.sha(ROOT/p) for p in ['scripts/benchmark_realtime_study.py',
          'scripts/launch_realtime_study.py','scripts/realtime_study_network.py',
          'scripts/benchmark_voice_realtime_ablation.py','scripts/benchmark_voice_realtime.py']}}
    b.save(out/'realtime_protocol.json', protocol)
    import requests
    client=requests.Session();client.trust_env=False
    creds=json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    resp=client.post(b.SERVER+'/api/auth/register',json={**creds,'display_name':'多样语音隔离评测'},timeout=15)
    resp.raise_for_status()
    health=client.get(b.SERVER+'/health',timeout=5).json()
    b.save(out/'isolated_health.json',health)
    if not health.get('startup',{}).get('ready') or health.get('turn_taking')!='local':
        raise RuntimeError('Isolated service not ready')
    print(json.dumps({'prepared':str(out),'formal_trials':len(schedule),'warmup':len(warmup),'card_chars':len(card)},ensure_ascii=False),flush=True)


async def run(out, stage):
    global _active_first_energy_index
    out,_=b.test_paths(out)
    protocol=json.loads((out/'realtime_protocol.json').read_text())
    import requests
    client=requests.Session();client.trust_env=False
    creds=json.loads((out/'isolated_runtime/test_credentials.json').read_text())
    resp=client.post(b.SERVER+'/api/auth/login',json=creds,timeout=15);resp.raise_for_status()
    cookie='; '.join(f'{k}={v}' for k,v in resp.cookies.items())
    if stage=='formal' and b.summary(out)['warmup_valid'] != len(protocol['warmup']):
        raise RuntimeError('All four warmups must pass before formal measurement')
    specs=protocol['warmup'] if stage=='warmup' else protocol['schedule']
    for spec in specs:
        if (out/'realtime_trials'/spec['trial_id']/'result.json').exists():
            continue  # Resume only never-started trials; do not replace failures.
        _active_first_energy_index=next(s['first_energy_frame_index'] for s in protocol['samples'] if s['sample_id']==spec['sample_id'])
        r=await b.run_trial(out,protocol,spec,cookie)
        b.summary(out)
        if not r['integrity_checks']['memory_frozen'] or not r['integrity_checks']['profile_frozen']:
            raise RuntimeError('Stopping: memory/profile drift')
        if stage=='warmup' and not r['valid_for_latency']:
            raise RuntimeError('Warmup failed; inspect recorded evidence before running formal trials')
        await asyncio.sleep(.3)
    summary=b.summary(out)
    print(json.dumps({k:summary[k] for k in ['attempted_formal','valid_formal','warmup_valid']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','warmup','formal','summarize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    from dotenv import dotenv_values
    os.environ.update({k:v for k,v in dotenv_values(ROOT/'.env').items() if v is not None})
    from scripts.realtime_study_network import enable_direct
    enable_direct()
    install_measurement_hooks()
    out=args.output.resolve()
    if args.stage=='prepare':asyncio.run(prepare(out,args.repeats))
    elif args.stage=='summarize':print(json.dumps(b.summary(out),ensure_ascii=False,indent=2))
    else:asyncio.run(run(out,args.stage))


if __name__=='__main__':
    main()
