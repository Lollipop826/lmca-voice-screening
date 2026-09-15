"""Offline regression tests; these are NOT substitutes for cloud experiments."""
from copy import deepcopy

import pytest

from scripts.audit_voice_realtime_ablation import classify_disposition
from scripts.benchmark_voice_realtime_ablation import ARMS, METRIC_PAIRS
from scripts.report_voice_realtime_ablation import complete_block_summary


def fixture():
    schedule = []
    records = []
    for block in (1, 2):
        for index, arm in enumerate(ARMS):
            spec = {'trial_id': f'formal-{len(schedule)+1:03d}', 'stage': 'formal',
                    'block_id': block, 'repeat': 1, 'sample_id': f'synthetic-{block}', 'arm': arm}
            schedule.append(spec)
            records.append({**spec, 'valid_for_latency': True,
                            'metrics': {key: float(1000 * block + index) for key in METRIC_PAIRS},
                            'output_text_chars': 30, 'output_audio_duration_s': 5.0})
    return {'schedule': schedule}, records


def test_complete_blocks_exclude_extra_success_without_deleting_it():
    protocol, records = fixture()
    records[-1]['valid_for_latency'] = False
    records[-2]['valid_for_latency'] = False
    result = complete_block_summary(protocol, records)
    assert result['complete_block_ids'] == [1]
    assert result['main_n'] == 4
    assert result['valid_formal'] == 6
    assert result['valid_outside_complete_blocks'] == ['formal-005', 'formal-006']
    assert all(g['n'] == 1 for g in result['groups'].values())
    assert all(p['n'] == 1 for p in result['paired_deltas'].values())
    assert result['main_sample_block_counts'] == {'synthetic-1': 1}


def test_missing_records_not_invented_as_zero_latency_or_success():
    protocol, records = fixture()
    result = complete_block_summary(protocol, records[:5])
    assert result['planned_formal'] == 8
    assert result['attempted_formal'] == 5
    assert result['missing_trial_ids'] == ['formal-006', 'formal-007', 'formal-008']
    assert result['main_n'] == 4
    assert result['groups']['M1E1']['metrics']['speech_end_to_first_audio_ms']['mean'] == 1000


def test_retains_all_complete_blocks_including_very_slow_observations():
    protocol, records = fixture()
    records[4]['metrics']['speech_end_to_first_audio_ms'] = 90000.0
    result = complete_block_summary(protocol, records)
    assert result['complete_block_ids'] == [1, 2]
    assert result['main_n'] == 8
    metric = result['groups']['M1E1']['metrics']['speech_end_to_first_audio_ms']
    assert metric['mean'] == 45500
    assert metric['p95_nearest_rank'] == 90000


def test_pair_deltas_keep_the_sign_and_same_block():
    protocol, records = fixture()
    result = complete_block_summary(protocol, records)
    pair = result['paired_deltas']['M1E1 - M0E1']
    assert pair['n'] == 2
    assert pair['metrics']['speech_end_to_first_audio_ms']['mean'] == -1


def test_no_complete_block_returns_no_fabricated_metrics():
    protocol, records = fixture()
    for r in records:
        r['valid_for_latency'] = False
    result = complete_block_summary(protocol, records)
    assert result['main_n'] == 0
    assert result['groups']['M1E1']['metrics']['speech_end_to_first_audio_ms'] == {'n': 0}


def test_duplicate_observed_trial_rejected():
    protocol, records = fixture()
    with pytest.raises(ValueError, match='Duplicate observed'):
        complete_block_summary(protocol, records + [deepcopy(records[0])])


def test_duplicate_planned_trial_rejected():
    protocol, records = fixture()
    protocol['schedule'].append(deepcopy(protocol['schedule'][0]))
    with pytest.raises(ValueError, match='Duplicate planned'):
        complete_block_summary(protocol, records)


@pytest.mark.parametrize('field,value', [('arm', 'M0E0'), ('sample_id', 'changed'), ('block_id', 9)])
def test_record_must_match_frozen_protocol(field, value):
    protocol, records = fixture()
    records[0][field] = value
    with pytest.raises(ValueError, match='frozen schedule'):
        complete_block_summary(protocol, records)


def test_within_block_material_cannot_silently_differ():
    protocol, records = fixture()
    protocol['schedule'][0]['sample_id'] = records[0]['sample_id'] = 'different'
    with pytest.raises(ValueError, match='stimulus/repeat mismatch'):
        complete_block_summary(protocol, records)


def cancellation_fixture():
    return ({'valid_for_latency': False, 'metrics': {}, 'pacing': {'sent_input_frames': 0},
             'started_at': '2026-09-13T22:22:30+08:00', 'finished_at': '2026-09-13T22:22:44+08:00'},
            {'stopped_at': '2026-09-13T22:22:43+08:00'})


def test_cancelled_trial_requires_stop_in_its_time_interval():
    record, stop = cancellation_fixture()
    assert classify_disposition(record, [], stop) == 'cancelled_before_measured_input'
    stop['stopped_at'] = '2026-09-13T23:00:00+08:00'
    assert classify_disposition(record, [], stop) == 'other_failure'
    assert classify_disposition(record, [], None) == 'other_failure'


def test_partial_measured_input_is_not_labelled_preinput_cancellation():
    record, stop = cancellation_fixture()
    record['pacing']['sent_input_frames'] = 1
    assert classify_disposition(record, [], stop) == 'other_failure'


def test_quota_rejection_requires_session_linked_log_evidence():
    record, stop = cancellation_fixture()
    record['metrics'] = {'vad_end_to_asr_final_ms': 560}
    assert classify_disposition(record, ['real-request-id'], stop) == 'llm_quota_failed'
    assert classify_disposition(record, [], stop) == 'other_failure'


def test_success_is_not_relabelled_as_failed_because_run_stopped():
    record, stop = cancellation_fixture()
    record['valid_for_latency'] = True
    assert classify_disposition(record, [], stop) == 'valid'
