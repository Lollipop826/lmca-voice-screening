"""Fast offline regressions; no network calls or production DB access."""
import asyncio
from collections import Counter

import pytest

from scripts import audit_controlled_ablation as audit
from scripts import benchmark_ablation_wait_policy as benchmark


def test_delay_schedule_has_paired_delays_and_all_eight_policies():
    rows = benchmark.make_trials(30, 20260913)
    assert len(rows) == 240
    assert Counter(r[2] for r in rows) == {name: 30 for name in benchmark.POLICIES}
    for case in {row[0] for row in rows}:
        subset = [r for r in rows if r[0] == case]
        assert len({r[1] for r in subset}) == 1
        assert {r[2] for r in subset} == set(benchmark.POLICIES)
    assert rows == benchmark.make_trials(30, 20260913)
    assert rows != benchmark.make_trials(30, 20260914)


def test_nearest_rank_p95_and_empty_input():
    assert benchmark.quantile_nearest([], .95) is None
    assert benchmark.quantile_nearest(list(range(1, 31)), .95) == 29
    assert benchmark.quantile_nearest([7], .95) == 7


@pytest.mark.parametrize('policy,expected_loads', [
    ('no_cache_250', 1), ('cold_cache_250', 1), ('warm_cache_250', 0),
    ('prefetch_match_250', 1), ('prefetch_changed_query_250', 2),
])
def test_real_wait_path_returns_correct_context_and_tracks_loader_calls(policy, expected_loads):
    row = asyncio.run(benchmark.measure_trial('unit-fast', 1, policy))
    assert row['returned_context']
    assert not row['timeout_skipped']
    assert not row['stale_context']
    assert row['measured_phase_loader_calls'] == expected_loads
    assert row['prewarm_loader_calls'] == int(policy == 'warm_cache_250')
    assert row['prefetch_started'] == policy.startswith('prefetch_')


def test_timeout_skips_context_but_drains_background_thread(monkeypatch):
    monkeypatch.setitem(benchmark.POLICIES, 'unit_timeout', {
        'cache': True, 'warm': False, 'lead_ms': 0, 'budget_ms': 1, 'mismatch': False,
    })
    row = asyncio.run(benchmark.measure_trial('unit-slow', 50, 'unit_timeout'))
    assert row['timeout_skipped'] and not row['returned_context']
    assert row['measured_phase_loader_calls'] == 1
    assert row['cache_sources'] == ['miss']  # Loader finished before the trial returned.


def test_raw_reply_and_parse_error_are_one_judge_invoke_not_two():
    rows = [{'status': 'ok', 'attempts': [
        {'attempt': 1, 'raw_response': 'invalid JSON'},
        {'attempt': 1, 'error_type': 'JSONDecodeError'},
        {'attempt': 2, 'raw_response': '{"valid":true}'},
    ]}, {'status': 'ok', 'attempts': [{'attempt': 1, 'raw_response': '{}'}]}]
    result = audit.judge_attempt_counts(rows)
    assert result == {
        'logical_requests': 2, 'logged_invoke_attempts': 3,
        'returned_raw_replies': 3, 'error_records': 1,
        'retried_logical_requests': 1, 'successful_logical_requests': 2,
    }
