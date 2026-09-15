import copy
import json
from pathlib import Path

import pytest

from scripts import run_controlled_ablation as study

FIXTURE = Path(__file__).parent / 'fixtures/controlled_ablation_cases_20260913.json'


def test_fixture_has_thirty_unique_fictional_cases_and_five_balanced_families():
    cases = study.load_cases(FIXTURE)
    assert len(cases) == 30
    assert len({case['sample_id'] for case in cases}) == 30
    assert set(study.Counter(c['family'] for c in cases).values()) == {6}
    assert all(c['synthetic_patient_id'].startswith('ablation-synthetic-') for c in cases)


def test_schedule_is_balanced_randomized_reproducible_and_non_mutating():
    cases = study.load_cases(FIXTURE)
    before = copy.deepcopy(cases)
    specs = study.run_specs(cases, 2, 17)
    assert len(specs) == 360
    assert specs == study.run_specs(cases, 2, 17)
    assert specs != study.run_specs(cases, 2, 18)
    assert len({s['run_id'] for s in specs}) == 360
    for sid in {c['sample_id'] for c in cases}:
        for repeat in (1, 2):
            assert {s['condition'] for s in specs if s['sample_id']==sid and s['repeat']==repeat} == set(study.CONDITIONS)
    assert cases == before


def test_memory_intervention_has_equal_item_ceiling_and_no_live_retrieval_claim():
    case = study.load_cases(FIXTURE)[0]
    off = study.build_intervention(case, 'M0E0')
    curated = study.build_intervention(case, 'M1E0')
    recent = study.build_intervention(case, 'MrecentE0')
    assert off['memory_context'] == ''
    assert curated['memory_item_ids'] == ['context-1','context-2']
    assert recent['memory_item_ids'] == ['recent-1','recent-2']
    assert len(curated['memory_context']) <= 240
    assert len(recent['memory_context']) <= 240
    assert all(not x['actual_memobase_used'] and not x['actual_audio_model_used'] for x in (off,curated,recent))


def test_real_agent_prompt_keeps_neutral_placeholder_and_only_changes_inputs():
    from src.agents.wellbeing_companion_agent import WellbeingCompanionAgent
    case = study.load_cases(FIXTURE)[0]
    off, _ = study.build_messages(case, 'M0E0')
    memory, _ = study.build_messages(case, 'M1E0')
    assert off[0].content == memory[0].content == WellbeingCompanionAgent._SYSTEM_PROMPT
    assert case['user_text'] in off[1].content
    assert 'neutral' in off[1].content
    assert '【受控记忆背景】' not in off[1].content
    assert '【受控记忆背景】' in memory[1].content
    assert '当前会话近期内容' not in off[1].content


def test_reference_emotion_is_explicitly_diagnostic_not_audio():
    case=study.load_cases(FIXTURE)[-1]
    intervention=study.build_intervention(case,'M1Eref')
    assert intervention['current_emotion']==case['reference_emotion']
    assert intervention['emotion_policy']=='authored_reference'
    assert intervention['actual_audio_model_used'] is False


def response_rows(score=4):
    return {chr(65+i): {'scores':{d:score for d in study.RUBRIC},
                         'flags':{f:False for f in study.FLAGS},'reason':'与当前表达一致'}
            for i in range(len(study.CONDITIONS))}


def test_blinding_does_not_expose_condition_names_or_mapping():
    case=study.load_cases(FIXTURE)[0]
    records={c:{'response':f'独立回复 {i}'} for i,c in enumerate(study.CONDITIONS)}
    prompt,mapping=study.blind_prompt(case,records,42)
    assert set(mapping)==set('ABCDEF')
    assert set(mapping.values())==set(study.CONDITIONS)
    assert all(condition not in prompt for condition in study.CONDITIONS)
    assert all(row['response'] in prompt for row in records.values())
    assert 'recent_two' not in prompt and 'curated_context' not in prompt


def test_judge_parser_accepts_complete_scores_and_booleans():
    rows=response_rows()
    assert study.parse_judgement(json.dumps({'responses':rows}),set(rows))==rows


@pytest.mark.parametrize('mutation',[ 'score_bool', 'score_range', 'missing_dimension', 'missing_label', 'flag_string'])
def test_judge_parser_rejects_invalid_ratings(mutation):
    rows=response_rows()
    if mutation=='score_bool': rows['A']['scores']['safety']=True
    elif mutation=='score_range': rows['A']['scores']['safety']=6
    elif mutation=='missing_dimension': rows['A']['scores'].pop('safety')
    elif mutation=='missing_label': rows.pop('F')
    elif mutation=='flag_string': rows['A']['flags']['internal_signal_leak']='false'
    with pytest.raises(ValueError):
        study.parse_judgement(json.dumps({'responses':rows}),set('ABCDEF'))


def make_ratings(case_ids=('case1','case2')):
    rows=[]; keys={}
    for sid in case_ids:
        for repeat in (1,2):
            for rater in (1,2):
                jid=f'{sid}__r{repeat}__j{rater}'
                mapping=dict(zip('ABCDEF',study.CONDITIONS))
                scores=response_rows(3)
                for label,condition in mapping.items():
                    value=4 if condition.startswith('M1') else 3
                    scores[label]['scores']={d:value for d in study.RUBRIC}
                keys[jid]={'prompt_sha256':'same','label_to_condition':mapping}
                rows.append({'judge_id':jid,'sample_id':sid,'repeat':repeat,'rater':rater,
                             'prompt_sha256':'same','status':'ok','responses':scores})
    return rows,keys


def test_statistics_aggregate_judges_and_generation_repeats_by_original_case():
    rows,keys=make_ratings()
    cases,incomplete=study.aggregate_cases(rows,keys,2,2)
    assert len(rows)==8 and len(cases)==2 and incomplete==[]
    assert cases['case1']['M1E0']['continuity']==4
    effect=[c['M1E0']['continuity']-c['M0E0']['continuity'] for c in cases.values()]
    summary=study.bootstrap_summary(effect)
    assert summary['n_cases']==2 and summary['mean']==1


def test_missing_judge_or_repeat_is_reported_not_silently_counted_as_complete():
    rows,keys=make_ratings()
    rows=[r for r in rows if r['judge_id']!='case1__r2__j2']
    cases,incomplete=study.aggregate_cases(rows,keys,2,2)
    assert set(cases)=={'case2'}
    assert any(row['sample_id']=='case1' for row in incomplete)


def test_key_mismatch_is_rejected():
    rows,keys=make_ratings()
    rows[0]['prompt_sha256']='wrong'
    with pytest.raises(ValueError,match='mismatch'):
        study.aggregate_cases(rows,keys,2,2)


def test_empty_bootstrap_and_zero_effect_are_explicit():
    assert study.bootstrap_summary([])['n_cases']==0
    summary=study.bootstrap_summary([0.,0.,0.])
    assert summary['ci95']==[0.,0.] and summary['tie_cases']==3


def test_resume_does_not_replace_success_with_later_failed_attempt():
    rows=[{'run_id':'one','status':'ok','response':'ok'}, {'run_id':'one','status':'failed'}]
    assert study.latest_ok(rows,'run_id')['one']['response']=='ok'
