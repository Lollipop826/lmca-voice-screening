#!/usr/bin/env python3
"""Offline integrity checks and explicitly post-hoc diagnostics for the study.

Never calls a provider or opens the production database. Does not rewrite the
frozen plan, generation records, judgements, or original statistical analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_controlled_ablation as study

# Defined after inspecting the results. Mentions are not retrieval accuracy,
# helpfulness, or a confirmatory endpoint. Synonyms outside this list are missed.
FOLLOWUP_ANCHORS = {
    'followup-01': ('手机', '客厅', '一周', '第七天'),
    'followup-02': ('女儿', '周日'),
    'followup-03': ('口腔', '复查', '看牙'),
    'followup-04': ('合唱',),
    'followup-05': ('花架',),
    'followup-06': ('猫',),
}


def judge_attempt_counts(rows):
    """A raw reply plus its parse exception is one attempt, not two calls."""
    counts = Counter()
    for row in rows:
        attempts = row.get('attempts', [])
        counts['logical_requests'] += 1
        counts['logged_invoke_attempts'] += len({a['attempt'] for a in attempts})
        counts['returned_raw_replies'] += sum('raw_response' in a for a in attempts)
        counts['error_records'] += sum('error_type' in a for a in attempts)
        counts['retried_logical_requests'] += len({a['attempt'] for a in attempts}) > 1
        counts['successful_logical_requests'] += row['status'] == 'ok'
    return dict(counts)


def audit(output):
    plan = json.loads((output / 'plan.json').read_text())
    analysis = json.loads((output / 'analysis.json').read_text())
    benchmark = json.loads((output / 'wait_policy_microbenchmark.json').read_text())
    cases = study.load_cases(output / 'source_snapshot/tests/fixtures/controlled_ablation_cases_20260913.json')
    generations = study.read_jsonl(output / 'generations.jsonl')
    judgements = study.read_jsonl(output / 'judgements.jsonl')
    keys = {r['judge_id']: r for r in json.loads((output / 'blinding_keys.json').read_text())['keys']}
    checks = []

    def check(name, passed):
        if not passed:
            raise ValueError('Integrity check failed: ' + name)
        checks.append(name)

    check('frozen_configuration_hash', study.digest(plan['configuration']) == plan['configuration_sha256'])
    check('analysis_uses_frozen_configuration', analysis['configuration_sha256'] == plan['configuration_sha256'])
    source_checks = {}
    for name, expected in plan['source_sha256'].items():
        source_checks[name] = {
            'snapshot_matches': hashlib.sha256((output / 'source_snapshot' / name).read_bytes()).hexdigest() == expected,
            'current_matches': hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
        }
    check('frozen_source_files_match_hashes', all(x['snapshot_matches'] for x in source_checks.values()))
    check('current_prompt_and_harness_match_experiment', all(x['current_matches'] for x in source_checks.values()))
    specs = study.run_specs(cases, plan['configuration']['repeats'], plan['configuration']['schedule_seed'])
    gs = study.latest_ok(generations, 'run_id')
    check('all_planned_generations_present', set(gs) == {s['run_id'] for s in specs})
    check('generation_records_unique', len(generations) == len(gs))
    check('generation_configurations_match', all(g['configuration_sha256'] == plan['configuration_sha256'] for g in gs.values()))
    check('response_hashes_match', all(study.digest(g['response']) == g['response_sha256'] for g in gs.values()))
    by_id = {c['sample_id']: c for c in cases}
    for g in gs.values():
        messages, intervention = study.build_messages(by_id[g['sample_id']], g['condition'])
        encoded = [{'role': m.type, 'content': m.content} for m in messages]
        check_result = study.digest(encoded) == g['prompt_sha256'] and intervention == g['intervention']
        if not check_result:
            raise ValueError('Cannot reproduce prompt ' + g['run_id'])
    checks.append('all_360_generation_prompts_reproduced_offline')
    check('judge_records_unique_and_complete', len(judgements) == len(keys) == plan['judge_request_count']
          and {j['judge_id'] for j in judgements} == set(keys))
    for j in judgements:
        key = keys[j['judge_id']]
        check_result = (j['prompt_sha256'] == key['prompt_sha256']
                        and set(key['label_to_condition'].values()) == set(study.CONDITIONS))
        records = {c: gs[f"{j['sample_id']}__r{j['repeat']}__{c}"] for c in study.CONDITIONS}
        for label, condition in key['label_to_condition'].items():
            check_result &= key['response_sha256_by_label'][label] == records[condition]['response_sha256']
        seed = int(study.digest([plan['configuration']['schedule_seed'], j['judge_id']])[:16], 16)
        prompt, mapping = study.blind_prompt(by_id[j['sample_id']], records, seed)
        check_result &= study.digest(prompt) == j['prompt_sha256'] and mapping == key['label_to_condition']
        if not check_result:
            raise ValueError('Blinding integrity failed ' + j['judge_id'])
    checks.append('all_blind_prompts_keys_and_response_hashes_match')

    # Independent aggregation from raw blinded scores (not case_condition_means).
    raw_scores = defaultdict(list)
    for j in judgements:
        for label, item in j['responses'].items():
            condition = keys[j['judge_id']]['label_to_condition'][label]
            for dimension, value in {**item['scores'], **item['flags']}.items():
                raw_scores[(j['sample_id'], condition, dimension)].append(float(value))
    check('four_ratings_per_case_condition_dimension', all(len(v) == 4 for v in raw_scores.values()))
    check('case_means_reproduce', all(statistics.fmean(v) == analysis['case_condition_means'][sid][cond][dim]
          for (sid, cond, dim), v in raw_scores.items()))
    check('condition_means_reproduce', all(
        round(statistics.fmean(statistics.fmean(raw_scores[(c['sample_id'], cond, dim)]) for c in cases), 4)
        == analysis['condition_means'][cond][dim]
        for cond in study.CONDITIONS for dim in (*study.RUBRIC, *study.FLAGS)))
    endpoints = (
        ('memory_continuity_relevant', 'memory_main', 'continuity', [c for c in cases if c['memory_relevance']]),
        ('text_emotion_fit_all', 'text_emotion_main', 'emotion_fit', cases),
        ('memory_restraint_negative_controls', 'memory_main', 'restraint', [c for c in cases if c['family'] in ('unrelated', 'update')]),
    )
    for label, contrast, dim, subset in endpoints:
        values = [sum(statistics.fmean(raw_scores[(c['sample_id'], cond, dim)]) * weight
                      for cond, weight in study.CONTRASTS[contrast].items()) for c in subset]
        check(label + '_reproduces', study.bootstrap_summary(values) == analysis['primary_endpoints'][label])

    ratings = [item for j in judgements for item in j['responses'].values()]
    distribution = Counter(v for item in ratings for v in item['scores'].values())
    flags = {cond: {flag: 0 for flag in study.FLAGS} for cond in study.CONDITIONS}
    for j in judgements:
        for label, item in j['responses'].items():
            cond = keys[j['judge_id']]['label_to_condition'][label]
            for flag, present in item['flags'].items():
                flags[cond][flag] += present
    probe = {}
    for cond in study.CONDITIONS:
        rows = [g for g in gs.values() if g['sample_id'] in FOLLOWUP_ANCHORS and g['condition'] == cond]
        hits = [g['run_id'] for g in rows if any(word in g['response'] for word in FOLLOWUP_ANCHORS[g['sample_id']])]
        probe[cond] = {'responses': len(rows), 'anchor_mentions': len(hits), 'run_ids': hits}

    # One fully auditable evidence-attribution failure, without generalizing its prevalence.
    mismatch_id = 'followup-05__r2__MrecentE0'
    attribution = []
    for j in judgements:
        if j['sample_id'] == 'followup-05' and j['repeat'] == 2:
            label = next(k for k, v in keys[j['judge_id']]['label_to_condition'].items() if v == 'MrecentE0')
            reason = j['responses'][label]['reason']
            if '花架' in reason and '花架' not in gs[mismatch_id]['response']:
                attribution.append({'judge_id': j['judge_id'], 'blinded_label': label,
                                    'run_id': mismatch_id, 'actual_response': gs[mismatch_id]['response'],
                                    'judge_reason': reason, 'possible_other_response': 'followup-05__r2__M1Eref'})

    # Count only explicit update-action claims, not ambiguous acknowledgements like 记住了.
    update_rows = [g for g in gs.values() if g['sample_id'] == 'update-15']
    explicit_updates = [g['run_id'] for g in update_rows if '已更新' in g['response'] or '已经更新' in g['response']]
    update_ratings = [j for j in judgements if j['sample_id'] == 'update-15']
    update_all_five = sum(all(v == 5 for v in item['scores'].values())
                          for j in update_ratings for item in j['responses'].values())

    records = benchmark['records']
    check('microbenchmark_all_240_trials_unique', len(records) == 240
          and len({(r['case_id'], r['policy']) for r in records}) == 240)
    delays = defaultdict(set)
    for r in records:
        delays[r['case_id']].add(r['loader_delay_ms'])
    check('microbenchmark_paired_backend_delays', len(delays) == 30 and all(len(x) == 1 for x in delays.values()))
    check('microbenchmark_zero_stale_returns', not any(r['stale_context'] for r in records))
    check('prefetch_stability_gate_triggered', all(r['prefetch_started'] for r in records if r['assumed_prefetch_lead_ms']))
    check('warm_cache_avoids_measured_loader_calls', all(r['returned_context'] and r['measured_phase_loader_calls'] == 0
          for r in records if r['policy'] == 'warm_cache_250'))
    check('changed_query_uses_new_loader', all(r['measured_phase_loader_calls'] == 2
          for r in records if r['policy'] == 'prefetch_changed_query_250'))
    benchmark_snapshots = {}
    for name, expected in benchmark['source_hashes'].items():
        data = (ROOT / name).read_bytes()
        check('microbenchmark_source_hash_' + name, hashlib.sha256(data).hexdigest() == expected)
        dest = output / 'microbenchmark_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        benchmark_snapshots[name] = expected

    result = {
        'schema_version': 'controlled-ablation-offline-audit-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'integrity_checks_passed': checks,
        'source_checks': source_checks,
        'judge_attempt_counts': judge_attempt_counts(judgements),
        'judge_attempt_scope': 'Logged invoke calls, not unobservable transport retries inside the provider client; excludes connectivity smoke.',
        'score_distribution': {str(k): v for k, v in sorted(distribution.items())},
        'total_score_cells': sum(distribution.values()),
        'maximum_score_fraction': distribution[5] / sum(distribution.values()),
        'all_dimensions_five_responses': sum(all(v == 5 for v in r['scores'].values()) for r in ratings),
        'total_judged_response_objects': len(ratings),
        'judge_flag_counts': flags,
        'per_condition_flag_denominator': 120,
        'posthoc_followup_anchor_probe': {'anchors': FOLLOWUP_ANCHORS, 'conditions': probe,
            'claim_boundary': 'Post-hoc exact substring diagnostic on 6 authored cases, not retrieval or response quality accuracy.'},
        'verified_judge_evidence_attribution_examples': attribution,
        'posthoc_no_write_update_wording_audit': {
            'case_id': 'update-15', 'generation_count': len(update_rows),
            'explicit_action_claim_run_ids': explicit_updates,
            'judged_response_objects_all_five': update_all_five,
            'judged_response_objects': len(update_ratings) * len(study.CONDITIONS),
            'claim_boundary': 'Memory writing was disabled. Wording such as 已更新 is not evidence of persisted state; qualitative audit, not independently human-adjudicated incidents.'},
        'microbenchmark_snapshot_hashes': benchmark_snapshots,
        'raw_evidence_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ('plan.json', 'generations.jsonl', 'judgements.jsonl', 'blinding_keys.json',
                         'analysis.json', 'wait_policy_microbenchmark.json', 'environment_preflight.json')},
    }
    study.write_json(output / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/ablation_study_20260913')
    args = parser.parse_args()
    result = audit(args.output.resolve())
    print(json.dumps({k: result[k] for k in ('integrity_checks_passed', 'judge_attempt_counts',
          'score_distribution', 'maximum_score_fraction', 'posthoc_no_write_update_wording_audit')},
          ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
