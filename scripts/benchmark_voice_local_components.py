#!/usr/bin/env python3
"""Real local-model and SQLite-service checks after cloud access failure.
These are component tests, NOT a substitute for the blocked end-to-end study.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,sqlite3,sys,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_voice_sqlite_ablation import test_paths,save,sha,stable_hash

def main(out):
    out,db=test_paths(out)
    protocol=json.loads((out/'protocol.json').read_text())
    local=out/'local_components';local.mkdir(exist_ok=False)
    plan={'frozen_at':datetime.now().astimezone().isoformat(),
          'purpose':'Actual local components only; cloud real-voice study aborted due resource-not-granted 403',
          'emotion':{'samples':4,'repetitions_per_sample':5,'warmup_excluded':2,
                     'input_text':'frozen reference transcript, not newly obtained ASR',
                     'function':'src.tools.emotion.classify_multimodal_with_metadata',
                     'human_emotion_ground_truth':None},
          'memory':{'runs_per_arm':20,'arms':['sqlite_card_on','sqlite_card_off'],
                    'function':'PatientMemoryService.resolve_for_session',
                    'fixture':'same persisted synthetic patient; new DB-backed session each call',
                    'long_term_writes':False,'semantic_memobase':False,'LLM_calls':0},
          'seed':20260913,'source_sha256':sha(Path(__file__).read_bytes())}
    save(local/'protocol.json',plan)
    os.environ.update(DB_PATH=str(db),MEMOBASE_PROJECT_URL='',MEMOBASE_API_KEY='',
        MEMORY_LOCAL_FALLBACK_ENABLED='false',USE_LOCAL_EMBEDDING='false',
        ENABLE_LONG_TERM_MEMORY_WRITES='false',USE_MODELSCOPE='true',EMOTION_USE_GPU='true',
        EMOTION2VEC_MODEL=str(out/'isolated_runtime/emotion_model_readonly'),
        EMOTION_USE_TRANSFORMERS='false',EMOTION_LOCAL_FILES_ONLY='true',
        HF_HOME='/data/luyang/cache/hf',MODELSCOPE_CACHE='/data/luyang/cache/modelscope',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    os.chdir(out/'isolated_runtime')
    import torch
    from src.tools.emotion import classify_multimodal_with_metadata,classify_emotion,get_audio_emotion_classifier
    from src.context_management.emotion_memobase import EmotionMemobase
    from src.db import database
    from src.voice.services import PatientMemoryService
    from src.voice.session import VoiceSession
    database.DB_PATH=str(db)
    classifier=get_audio_emotion_classifier()
    load_start=time.perf_counter();classifier._load_model()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    load_ms=(time.perf_counter()-load_start)*1000
    save(local/'runtime.json',{'model_load_ms':load_ms,'model_available':classifier.model_available,
         'device':classifier.device,'torch':torch.__version__,'python':sys.version,
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'text_component':'built-in rule classifier; transformers disabled, consistent with tested service default',
         'dependency_install':'No install; symlink view excludes model requirements.txt',
         'audio_model_path':classifier.model_name})
    if not classifier.model_available: raise RuntimeError('Real Emotion2Vec unavailable: '+str(classifier.last_error))
    first=protocol['samples'][0]
    def infer(sample,rep,warmup=False):
        if sha(Path(sample['audio_path']).read_bytes())!=sample['wav_sha256']: raise RuntimeError('Input modified')
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        scores,metadata=classify_multimodal_with_metadata(sample['reference_text'],sample['audio_path'])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        wall_ms=(time.perf_counter()-start)*1000
        return {'sample_id':sample['sample_id'],'repeat':rep,'warmup':warmup,
                'wav_sha256':sample['wav_sha256'],'reference_text':sample['reference_text'],
                'audio_duration_s':sample['input_audio_duration_s'],
                'wall_ms':wall_ms,'audio_model_inference_ms':metadata['inference_ms'],
                'scores':scores,'dominant':max(scores,key=scores.get),'metadata':metadata,
                'valid_real_audio_model':metadata.get('audio_model_used') is True and metadata.get('source')=='emotion2vec_audio+text'}
    warm=[infer(first,i+1,True) for i in range(2)];save(local/'emotion_warmup.json',warm)
    schedule=[(s,rep) for rep in range(1,6) for s in protocol['samples']]
    random.Random(20260913).shuffle(schedule)
    with (local/'emotion_trials.jsonl').open('w') as f:
        for sample,rep in schedule:
            row=infer(sample,rep);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({k:row[k] for k in ['sample_id','repeat','dominant','wall_ms','audio_model_inference_ms','valid_real_audio_model']},ensure_ascii=False),flush=True)
    component_scores=[]
    for s in protocol['samples']:
        text=classify_emotion(s['reference_text']);audio=classifier.classify_audio(s['audio_path'])
        component_scores.append({'sample_id':s['sample_id'],'text_rule_scores':text,'audio_7d_scores':audio,
                                 'text_scores_tied':len(set(round(x,8) for x in text.values()))==1})
    save(local/'separate_component_scores.json',component_scores)
    memory=EmotionMemobase(db_path=str(db),emotion_classifier=classify_emotion,logger=lambda _msg:None)
    patient=protocol['patient_id']; fixture=json.loads((out/'fixture.json').read_text())
    facts=[item['text'] for category in ['facts','events','preferences'] for item in fixture['memory'][category]]
    def authoritative_state():
        with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
            c.row_factory=sqlite3.Row
            return {name:[dict(r) for r in c.execute(f'SELECT * FROM {name} WHERE patient_id=? ORDER BY rowid',(patient,))]
                    for name in ['emotion_memobase_memory_items','emotion_memobase_memory_deletions','emotion_memobase_snapshots','emotion_memobase_turns']}
    before=authoritative_state();save(local/'memory_state_before.json',before)
    log=[]
    service=PatientMemoryService(get_patient=database.get_patient,create_patient=database.create_patient,
        update_patient_profile=database.update_patient_profile,link_session_patient=database.link_session_patient,
        assign_patient=database.assign_patient,long_term_memory=memory,long_term_memory_writes=False,logger=log.append)
    memory_schedule=[(enabled,rep) for rep in range(1,21) for enabled in [True,False]]
    random.Random(20260913).shuffle(memory_schedule)
    with (local/'memory_trials.jsonl').open('w') as f:
        for i,(enabled,rep) in enumerate(memory_schedule):
            session_id=f'local_card_test_{i+1:03d}_{out.name[-6:]}'
            owner='ablation_test_20260913'
            database.create_session(session_id,owner_username=owner,patient_id=patient,mode='wellbeing')
            session=VoiceSession(connection=None,agent=None,owner_username=owner,session_id=session_id,
                long_term_memory_enabled=enabled,long_term_memory_writes_enabled=False,
                emotion_enabled=False)
            start=time.perf_counter();context=service.resolve_for_session(session,requested_patient_id=patient)
            elapsed=(time.perf_counter()-start)*1000
            present=[fact for fact in facts if fact in context.memory_card]
            row={'repeat':rep,'memory_enabled':enabled,'session_id':session_id,'elapsed_ms':elapsed,
                 'card_chars':len(context.memory_card),'has_history':context.has_history,
                 'card_sha256':sha(context.memory_card.encode()),'fixture_facts_in_card':len(present),
                 'valid':len(present)==(len(facts) if enabled else 0),'card':context.memory_card}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    after=authoritative_state();save(local/'memory_state_after.json',after)
    (local/'memory_service.log').write_text('\n'.join(log)+'\n')
    card_after=memory.get_authoritative_card(patient)
    (local/'authoritative_card_after.txt').write_text(card_after)
    save(local/'memory_integrity.json',{'before_sha256':stable_hash(before),'after_sha256':stable_hash(after),
       'authoritative_state_unchanged':before==after,
       'card_equal_to_before_cloud_pilot':card_after==(out/'authoritative_memory_card.txt').read_text(),
       'semantic_memobase_available':memory.is_memobase_available(),
       'semantic_evidence_by_case':{s['sample_id']:memory.get_relevant_evidence(patient,s['reference_text']) for s in protocol['samples']}})
    print('LOCAL_COMPONENT_CHECKS_COMPLETE',str(local),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
