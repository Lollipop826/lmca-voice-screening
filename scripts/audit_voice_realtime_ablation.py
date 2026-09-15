#!/usr/bin/env python3
"""Independently recompute realtime ablation timestamps from raw receive/send logs."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_voice_sqlite_ablation import state_snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_disposition(record, quota_ids, early_stop):
    """Preserve failures; distinguish a logged rejection from operator cancellation."""
    if record['valid_for_latency']:
        return 'valid'
    if quota_ids:
        return 'llm_quota_failed'
    if early_stop and not record.get('metrics') and not record.get('pacing', {}).get('sent_input_frames'):
        stopped = datetime.fromisoformat(early_stop['stopped_at'])
        if datetime.fromisoformat(record['started_at']) <= stopped <= datetime.fromisoformat(record['finished_at']):
            return 'cancelled_before_measured_input'
    return 'other_failure'


def audit(out: Path):
    protocol = json.loads((out/'realtime_protocol.json').read_text())
    preflight = json.loads((out/'preflight_plan.json').read_text())
    samples = {s['sample_id']: s for s in protocol['samples']}
    schedule = {s['trial_id']: s for s in protocol['schedule']}
    stop_path = out/'early_stop.json'
    early_stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    source_log = (out/'isolated_server.log').read_text(errors='replace')
    latency_pattern = re.compile(r'\[Latency\] session=(\S+) memory_card_ms=([\d.]+) memory_card_chars=(\d+)')
    sessions = {m.group(1): {'card_ms': float(m.group(2)), 'card_chars': int(m.group(3)), 'offset': m.start()}
                for m in latency_pattern.finditer(source_log)}
    all_offsets = sorted(x['offset'] for x in sessions.values()) + [len(source_log)]
    baseline = json.loads((out/'memory_state_before.json').read_text())['tables_sha256']
    rows = []
    for p in sorted((out/'realtime_trials').glob('*/result.json')):
        r = json.loads(p.read_text())
        if r['stage'] != 'formal': continue
        issues = []
        spec = schedule.get(r['trial_id'])
        if spec is None or any(r.get(k) != v for k, v in spec.items()):
            issues.append('frozen_schedule_mismatch')
        trace = [json.loads(line) for line in (p.parent/'events.jsonl').read_text().splitlines()]
        measured = [x for x in trace if x.get('ms_since_submit') is not None and isinstance(x.get('event'),dict)]
        frames = json.loads((p.parent/'frame_times.json').read_text())
        sample = samples[r['sample_id']]
        active_frame = next((f for f in frames if f['index'] == sample['last_energy_frame_index']), None)
        times = {}
        if active_frame is not None: times['speech_end'] = active_frame['sent_perf']
        if frames: times['stream_start'] = frames[0]['target_perf'] - 512/16000
        asr_rows = [x for x in measured if x['event'].get('type') == 'asr_result']
        ids = list(dict.fromkeys(str(x['event'].get('turn_id')) for x in asr_rows))
        tid = ids[0] if len(ids) == 1 else None
        audio = bytearray()
        last_vad = None
        for x in measured:
            e = x['event']; t = x['perf_counter']; kind = e.get('type')
            if kind == 'vad_end': last_vad = t
            if kind == 'asr_partial' and str(e.get('text') or '').strip() and not e.get('final'):
                times.setdefault('first_partial', t)
            if not tid or str(e.get('turn_id') or '') != tid: continue
            if kind == 'asr_result':
                times.setdefault('asr_final', t)
                if last_vad is not None: times.setdefault('vad_end', last_vad)
            if kind in {'ai_response_chunk','ai_response'} and str(e.get('text') or '').strip():
                times.setdefault('first_ai_text', t)
            if kind in {'tts_chunk','tts_audio'}:
                encoded = e.get('chunk') or e.get('audio') or e.get('data')
                if encoded:
                    chunk = base64.b64decode(encoded, validate=True)
                    if chunk:
                        times.setdefault('first_audio', t)
                        audio.extend(chunk)
            if kind == 'tts_end': times.setdefault('tts_end', t)
        recalculated = {}
        for metric, (a,b) in protocol['metrics'].items():
            if a in times and b in times:
                value = (times[b]-times[a])*1000
                recalculated[metric] = round(value,3)
                if abs(value - r['metrics'].get(metric, float('inf'))) > .0006:
                    issues.append('metric_mismatch:'+metric)
            elif metric in r.get('metrics',{}):
                issues.append('metric_without_raw_marks:'+metric)
        for k,t in times.items():
            if abs(t - r['marks_perf_counter'].get(k,float('inf'))) > 1e-7:
                issues.append('mark_mismatch:'+k)
        if r['valid_for_latency']:
            if len(ids) != 1: issues.append('non_single_turn')
            if not all(r.get('integrity_checks',{}).values()): issues.append('unchecked_success')
            if not audio or hashlib.sha256(audio).hexdigest() != r.get('audio_sha256'):
                issues.append('received_audio_hash_mismatch')
            if digest(p.parent/'tts_received.bin') != r.get('audio_sha256'):
                issues.append('native_audio_file_hash_mismatch')
            chain = sum(recalculated.get(k, -1e9) for k in ['speech_end_to_vad_end_ms','vad_end_to_asr_final_ms',
                                      'asr_final_to_first_ai_text_ms','first_ai_text_to_first_audio_ms'])
            if abs(chain - recalculated.get('speech_end_to_first_audio_ms',1e9)) > .003:
                issues.append('non_additive_chain')
        if r.get('memory_sha256_before') != baseline or r.get('memory_sha256_after') != baseline:
            issues.append('memory_or_profile_drift')
        if digest(sample['audio_path']) != sample['prepared_wav_sha256']:
            issues.append('stimulus_changed')
        sid = r.get('session',{}).get('session_id')
        card = sessions.get(sid)
        path_counts = {}
        quota_ids = []
        if card:
            end = next(i for i in all_offsets if i > card['offset'])
            segment = source_log[card['offset']:end]
            quota_lines = [line for line in segment.splitlines() if 'AllocationQuota.FreeTierOnly' in line and '403' in line]
            quota_ids = sorted(set(re.findall(r"'request_id': '([^']+)'", '\n'.join(quota_lines))))
            path_counts = {
                'streaming_final_used': segment.count('使用 BigASR 流式最终文字'),
                'streaming_fallbacks': segment.count('流式结果未通过最终性校验'),
                'barge_in_after_greeting': segment.count('用户主动打断或回答'),
            }
            if r['valid_for_latency'] and path_counts['streaming_final_used'] != 1:
                issues.append('streaming_path_not_exactly_one')
            if path_counts['streaming_fallbacks'] or path_counts['barge_in_after_greeting']:
                issues.append('unexpected_fallback_or_barge_in')
        else:
            issues.append('missing_actual_memory_card_log')
        disposition = classify_disposition(r, quota_ids, early_stop)
        rows.append({'disposition': disposition, 'llm_quota_request_ids': quota_ids, 'trial_id': r['trial_id'], 'arm': r['arm'], 'session_id': sid,
                     'valid_for_latency': r['valid_for_latency'], 'metric_count': len(recalculated),
                     'recomputed_metrics': recalculated, 'actual_memory_card_chars': card['card_chars'] if card else None,
                     'memory_card_setup_ms': card['card_ms'] if card else None,
                     'path_counts': path_counts, 'issues': issues,
                     'raw_sha256': {name:digest(p.parent/name) for name in ['events.jsonl','frame_times.json','result.json']}})
    current = state_snapshot(out/'isolated_runtime/test_only.sqlite3', protocol['patient_id'])
    by_arm = {arm: sorted({r['actual_memory_card_chars'] for r in rows if r['arm']==arm and r['actual_memory_card_chars'] is not None})
              for arm in protocol['arms']}
    on = {n for arm in ['M1E1','M1E0'] for n in by_arm[arm]}
    off = {n for arm in ['M0E1','M0E0'] for n in by_arm[arm]}
    source_checks = {k: digest(ROOT/k)==v for k,v in preflight['source_sha256'].items()}
    recorded_ids = [r['trial_id'] for r in rows]
    schedule_matches = len(recorded_ids) == len(set(recorded_ids)) and set(recorded_ids) <= set(schedule)
    missing_ids = [tid for tid in schedule if tid not in recorded_ids]
    complete_schedule = schedule_matches and not missing_ids
    quota_evidence_path = out/'llm_quota_error_evidence.json'
    evidence = json.loads(quota_evidence_path.read_text()) if quota_evidence_path.exists() else {}
    expected_quota_ids = {e['request_id'] for e in evidence.get('unique_request_errors', [])}
    observed_quota_ids = {qid for row in rows for qid in row['llm_quota_request_ids']}
    result = {
        'created_at': datetime.now().astimezone().isoformat(), 'formal_records_audited': len(rows),
        'formal_records_with_issues': sum(bool(r['issues']) for r in rows),
        'recomputed_metric_count': sum(r['metric_count'] for r in rows),
        'complete_schedule': complete_schedule,
        'planned_formal_records': len(schedule), 'missing_trial_ids': missing_ids,
        'schedule_identity_verified': schedule_matches,
        'study_terminated_early': early_stop is not None and not complete_schedule,
        'early_stop': early_stop,
        'disposition_counts': {kind: sum(r['disposition'] == kind for r in rows) for kind in ['valid', 'llm_quota_failed', 'cancelled_before_measured_input', 'other_failure']},
        'llm_quota_request_ids_verified': observed_quota_ids == expected_quota_ids,
        'unique_llm_quota_requests': len(observed_quota_ids),
        'env_file_unchanged': digest(ROOT/'.env')==preflight['production_env_sha256'],
        'business_source_unchanged': source_checks,
        'measurement_source_unchanged_since_protocol': digest(ROOT/'scripts/benchmark_voice_realtime_ablation.py')==protocol['source_sha256']['scripts/benchmark_voice_realtime_ablation.py'],
        'memory_profile_still_frozen': current['tables_sha256']==baseline,
        'actual_card_chars_by_arm': by_arm,
        'memory_card_on_off_separation_verified': bool(on and off and len(on)==len(off)==1 and min(on)>max(off)),
        'pilot_excluded': protocol.get('pilot_exclusion'), 'rows': rows,
    }
    result['passed'] = (schedule_matches and result['llm_quota_request_ids_verified']
                         and not result['formal_records_with_issues'] and result['env_file_unchanged']
                         and all(source_checks.values()) and result['measurement_source_unchanged_since_protocol']
                         and result['memory_profile_still_frozen'] and result['memory_card_on_off_separation_verified'])
    path = out/('measurement_audit.json' if result['complete_schedule'] or result['study_terminated_early'] else 'measurement_audit_interim.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    x=audit(args.output)
    print(json.dumps({k:v for k,v in x.items() if k!='rows'},ensure_ascii=False,indent=2))
    if not x['passed']: raise SystemExit(1)
