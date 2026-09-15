"""Regression tests for the deterministic 2x2 quality scorer.

The subtle parts are what gets protected here: a phrase the user already said
must never count as recall, an inapplicable metric must stay out of the
denominator rather than scoring zero, and a corrupted stimulus must not be
reported as a passed or failed test point.
"""

from __future__ import annotations

import json

import pytest

from scripts import score_realtime_ablation_quality as scorer


def _trial(
    *,
    arm,
    block_id=1,
    sample_id="sleep-followup",
    reply="好的",
    recognized="最近又睡不好，晚上总忍不住看工作消息。",
    status="ok",
):
    memory_on = arm.startswith("M1")
    emotion_on = arm.endswith("E1")
    return {
        "trial_id": f"formal-{block_id:03}-{arm}",
        "arm": arm,
        "block_id": block_id,
        "sample_id": sample_id,
        "status": status,
        "valid_for_latency": status == "ok",
        "final_ai_text": reply,
        "asr_text": recognized,
        "sample_sha256": f"sha-{sample_id}",
        "observed_turn_ids": ["turn-1"],
        "observed_asr_results": [{"text": recognized}],
        "output_text_chars": len(reply),
        "final_insight": {
            "memory": {"retrieval_source": "realtime" if memory_on else "disabled"},
            "emotion": {
                "source": "emotion2vec_audio+text" if emotion_on else "disabled",
                "audio_model_used": emotion_on,
            },
        },
    }


def _complete_block(block_id, sample_id, replies, recognized):
    return [
        _trial(
            arm=arm,
            block_id=block_id,
            sample_id=sample_id,
            reply=replies[arm],
            recognized=recognized,
        )
        for arm in scorer.ARM_IDS
    ]


def test_phrase_already_spoken_by_the_user_is_not_memory_evidence():
    """The measured failure mode: 周三 is in the walk-update transcript.

    Counting it scored all four arms an identical 2/22, which looked like a
    null effect but was really the model echoing the input back.
    """
    recognized = "散步现在改到周五下午了，周三要上课，别再按旧时间记了。"

    usable = scorer.usable_anchor_phrases(
        "wednesday_walk_with_neighbour", recognized
    )
    assert "周三" not in usable
    assert "邻居" in usable

    assert not scorer.cited_anchor("周三的安排我记下了", "wednesday_walk_with_neighbour", recognized)
    assert scorer.cited_anchor("和邻居一起走走", "wednesday_walk_with_neighbour", recognized)


def test_anchor_stays_usable_when_the_user_never_said_it():
    recognized = "最近又睡不好，晚上总忍不住看工作消息。"
    assert scorer.cited_anchor(
        "你之前说愿意把手机放远一点", "phone_away_at_night", recognized
    )
    assert not scorer.cited_anchor("早点休息吧", "phone_away_at_night", recognized)


def test_inapplicable_metrics_are_absent_rather_than_false():
    """An absent key keeps a stimulus out of the denominator entirely."""
    outcomes = scorer.score_reply(
        "你之前说愿意把手机放远一点",
        sample_id="sleep-followup",
        recognized_text="最近又睡不好，晚上总忍不住看工作消息。",
    )
    assert outcomes["cited_expected_memory"] is True
    assert "violated_emotional_boundary" not in outcomes
    assert "acknowledged_correction" not in outcomes


def test_corrupted_stimulus_is_not_scored_for_citation():
    """ASR heard 睡前 as 税前 in every trial, so the reference never arrived.

    Scoring it would report a reference-resolution failure that the model was
    never actually given the chance to pass.
    """
    outcomes = scorer.score_reply(
        "你说的税前那个办法是指什么呀",
        sample_id="method-followup",
        recognized_text="上次说的税前那个办法，我试了两天，还是拿不准，下一步。",
    )
    assert "cited_expected_memory" not in outcomes


def test_unrelated_stimulus_flags_forced_memory():
    recognized = "今天买菜回来发现钥匙忘在家里了，在门口等了半天。"
    clean = scorer.score_reply(
        "在门口等那么久肯定累了吧", sample_id="keys-unrelated", recognized_text=recognized
    )
    forced = scorer.score_reply(
        "这和你上周跟女儿的争执有关系吗",
        sample_id="keys-unrelated",
        recognized_text=recognized,
    )
    assert clean["forced_irrelevant_memory"] is False
    assert forced["forced_irrelevant_memory"] is True


def test_emotional_boundary_violation_is_detected():
    recognized = "今天被人当众打断，我真的很生气，先别劝我忍着。"
    respected = scorer.score_reply(
        "换谁都会窝火，我就在这儿陪着你",
        sample_id="anger-boundary",
        recognized_text=recognized,
    )
    violated = scorer.score_reply(
        "你就先忍着，别生气了", sample_id="anger-boundary", recognized_text=recognized
    )
    assert respected["violated_emotional_boundary"] is False
    assert violated["violated_emotional_boundary"] is True


def test_blocks_missing_an_arm_are_excluded_from_pairing():
    trials = _complete_block(
        1,
        "sleep-followup",
        {arm: "回复" for arm in scorer.ARM_IDS},
        "最近又睡不好，晚上总忍不住看工作消息。",
    )
    assert len(scorer.build_paired_blocks(trials)) == 1

    assert scorer.build_paired_blocks(trials[:-1]) == []


def test_blocks_with_divergent_input_are_excluded():
    """Pairing only buys anything when every arm heard the same thing."""
    trials = _complete_block(
        1,
        "sleep-followup",
        {arm: "回复" for arm in scorer.ARM_IDS},
        "最近又睡不好，晚上总忍不住看工作消息。",
    )
    trials[2]["asr_text"] = "完全不同的一句话"
    assert scorer.build_paired_blocks(trials) == []


def test_trial_without_a_reply_is_not_scorable():
    """sad-no-advice produced perfect ASR but no LLM turn in every trial."""
    empty = _trial(arm="M1E1", reply="")
    assert not scorer.is_scorable(empty)

    multi_turn = _trial(arm="M1E1")
    multi_turn["observed_turn_ids"] = ["turn-1", "turn-2"]
    assert not scorer.is_scorable(multi_turn)


def test_manipulation_check_catches_an_arm_that_kept_memory():
    trials = _complete_block(
        1,
        "sleep-followup",
        {arm: "回复" for arm in scorer.ARM_IDS},
        "最近又睡不好，晚上总忍不住看工作消息。",
    )
    blocks = scorer.build_paired_blocks(trials)
    assert scorer.verify_arm_manipulations(blocks)["all_arms_verified"] is True

    leaky = scorer.build_paired_blocks(trials)
    leaky[0]["arms"]["M0E0"]["final_insight"]["memory"]["retrieval_source"] = "realtime"
    findings = scorer.verify_arm_manipulations(leaky)
    assert findings["all_arms_verified"] is False
    assert findings["M0E0"]["memory_manipulation_verified"] is False


def test_mcnemar_uses_only_discordant_pairs():
    assert scorer.exact_mcnemar_two_sided(0, 0) is None
    # A unanimous split of four discordant pairs is the strongest evidence
    # four pairs can carry, and it still only reaches 0.125.
    assert scorer.exact_mcnemar_two_sided(4, 0) == pytest.approx(0.125)
    assert scorer.exact_mcnemar_two_sided(2, 2) == pytest.approx(1.0)


def test_score_run_reports_memory_effect_and_flags_missing_stimulus(tmp_path):
    """End-to-end: memory arms cite the card, memory-off arms cannot."""
    run_directory = tmp_path / "run"
    trials_directory = run_directory / "realtime_trials"
    recognized = "最近又睡不好，晚上总忍不住看工作消息。"

    trials = []
    for block_id in (1, 2, 3, 4):
        replies = {
            "M1E1": "你之前说愿意把手机放远一点",
            "M1E0": "你之前说愿意把手机放远一点",
            "M0E1": "早点休息吧",
            "M0E0": "早点休息吧",
        }
        trials.extend(
            _complete_block(block_id, "sleep-followup", replies, recognized)
        )
    # A stimulus whose every trial failed, mirroring sad-no-advice.
    for arm in scorer.ARM_IDS:
        trials.append(
            _trial(
                arm=arm,
                block_id=9,
                sample_id="sad-no-advice",
                reply="",
                recognized="我现在很难过，暂时不想听办法，只想有人听我说说。",
                status="failed",
            )
        )

    for trial in trials:
        directory = trials_directory / trial["trial_id"]
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps(trial, ensure_ascii=False), encoding="utf-8"
        )

    report = scorer.score_run(run_directory)

    assert report["complete_paired_blocks"] == 4
    assert report["manipulation_checks"]["all_arms_verified"] is True

    citation = report["metrics"]["cited_expected_memory"]
    assert citation["per_arm"]["M1E1"]["rate"] == 1.0
    assert citation["per_arm"]["M0E1"]["rate"] == 0.0

    memory_contrast = citation["paired_contrasts"]["M1E1 - M0E1"]
    assert memory_contrast["left_only"] == 4
    assert memory_contrast["right_only"] == 0
    assert memory_contrast["exact_mcnemar_p"] == pytest.approx(0.125)

    flagged = {flag["kind"] for flag in report["data_quality_flags"]}
    assert "sample_without_any_scorable_trial" in flagged
