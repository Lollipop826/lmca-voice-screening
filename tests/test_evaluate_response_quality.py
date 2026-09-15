import pytest

from scripts import evaluate_response_quality as quality


def _record(
    condition,
    pair_index,
    *,
    source="soulx_final",
    hit=False,
    sample_id="sample-1",
):
    return {
        "condition": condition,
        "pair_index": pair_index,
        "sample_id": sample_id,
        "valid_for_latency": True,
        "turn_count": 1,
        "asr_source": source,
        "assistant_text": f"response {condition} {pair_index}",
        "recognized_text": "我最近有点难过。",
        "audio_sha256": f"hash-{sample_id}",
        "memory_retrieval_hit": hit,
        "memory_writes_enabled": False,
        "asr_cer": 0.0,
    }


def _benchmark():
    """Two pairs built from two distinct audio samples.

    Each pair gets its own ``sample_id`` so the default fixture represents two
    independent cases.  Re-running one audio file is covered separately by
    ``_repeated_sample_benchmark``.
    """
    return {
        "configuration": {"ablation": "memory"},
        "records": [
            _record("memory_on", 1, hit=True, sample_id="sample-1"),
            _record("memory_off", 1, sample_id="sample-1"),
            _record("memory_on", 2, hit=True, sample_id="sample-2"),
            _record("memory_off", 2, sample_id="sample-2"),
        ],
    }


def _repeated_sample_benchmark():
    """Two pairs that are two runs of the *same* audio sample.

    The voice benchmark loops over its manifest, so running three scenarios
    twice yields six pairs but only three independent samples.  Each repeat
    becomes its own ``pair_index`` and therefore its own blind case, which is
    exactly how repeated measurements sneak in as fake independent samples.
    """
    return {
        "configuration": {"ablation": "memory"},
        "records": [
            _record("memory_on", 1, hit=True, sample_id="sample-repeat"),
            _record("memory_off", 1, sample_id="sample-repeat"),
            _record("memory_on", 2, hit=True, sample_id="sample-repeat"),
            _record("memory_off", 2, sample_id="sample-repeat"),
        ],
    }


def test_prepare_blinds_condition_and_preserves_paired_prompt():
    sheet, key = quality.prepare_blind_sheet(
        _benchmark(), seed=7, raters=["r1"]
    )

    assert sheet["kind"] == "blinded_response_quality_ratings"
    assert len(sheet["cases"]) == 2
    assert {case["prompt"] for case in sheet["cases"]} == {"我最近有点难过。"}
    assert all("condition" not in case for case in sheet["cases"])
    assert {
        key_case["label_a"] for key_case in key["cases"]
    } | {key_case["label_b"] for key_case in key["cases"]} == {
        "memory_on",
        "memory_off",
    }


def test_prepare_skips_pairs_without_a_streaming_final():
    """A reply built from a non-streaming transcript is not a comparable turn.

    ``final`` is the local-VAD counterpart of ``soulx_final`` and stays
    eligible; only sources outside the whitelist, such as a whole-utterance
    re-recognition fallback, disqualify a pair.
    """
    benchmark = _benchmark()
    benchmark["records"][0]["asr_source"] = "nostream_recognize"

    sheet, _ = quality.prepare_blind_sheet(benchmark, seed=1, raters=["r1"])
    assert len(sheet["cases"]) == 1


def test_prepare_accepts_local_vad_finals():
    """The isolated 2x2 ablation runs on Silero VAD, which finalizes as ``final``."""
    benchmark = _benchmark()
    for record in benchmark["records"]:
        record["asr_source"] = "final"

    sheet, _ = quality.prepare_blind_sheet(benchmark, seed=1, raters=["r1"])
    assert len(sheet["cases"]) == 2


def test_prepare_accepts_sqlite_card_memory_without_a_retrieval_hit():
    """Memory can be live via the authoritative card while Memobase is down.

    ``memory_retrieval_hit`` stays false in that configuration, so requiring it
    would reject an entire protocol-conformant run. The manipulation is carried
    by ``memory_retrieval_source``.
    """
    benchmark = _benchmark()
    for record in benchmark["records"]:
        record["memory_retrieval_hit"] = False
        record["memory_retrieval_source"] = (
            "realtime" if record["condition"] == "memory_on" else "disabled"
        )

    sheet, _ = quality.prepare_blind_sheet(benchmark, seed=1, raters=["r1"])
    assert len(sheet["cases"]) == 2


def test_prepare_skips_pairs_whose_off_arm_still_had_memory():
    """If the off arm kept memory, the pair measures nothing."""
    benchmark = _benchmark()
    for record in benchmark["records"]:
        record["memory_retrieval_hit"] = False
        record["memory_retrieval_source"] = "realtime"

    with pytest.raises(quality.QualityEvaluationError, match="没有可用于盲评"):
        quality.prepare_blind_sheet(benchmark, seed=1, raters=["r1"])


def test_analyse_maps_blinded_scores_back_to_conditions():
    sheet, key = quality.prepare_blind_sheet(
        _benchmark(), seed=3, raters=["r1"]
    )
    for case in sheet["cases"]:
        ratings = case["ratings"]["r1"]
        for dimension in quality.MEMORY_RUBRIC:
            ratings["a"][dimension] = 5
            ratings["b"][dimension] = 3
        ratings["preference"] = "a"

    result = quality.analyse_ratings(sheet, key)
    assert result["completed_ratings"] == 2
    assert result["preference"]["on"] in {0, 1, 2}
    for dimension in quality.MEMORY_RUBRIC:
        assert result["dimensions"][dimension]["on"]["count"] == 2
        assert result["dimensions"][dimension]["off"]["count"] == 2
        assert result["dimensions"][dimension]["paired_delta_on_minus_off"]["count"] == 2


def test_significance_does_not_count_repeated_raters_as_samples():
    """Repeated raters on one case must not inflate the apparent sample size.

    Three raters scoring two cases yields six paired differences, but only two
    independent samples. Treating all six as independent is pseudo-replication
    and drives the sign flip p value far below what the data supports.
    """
    raters = ["r1", "r2", "r3"]
    sheet, key = quality.prepare_blind_sheet(
        _benchmark(), seed=11, raters=raters
    )
    for case in sheet["cases"]:
        for rater in raters:
            ratings = case["ratings"][rater]
            for dimension in quality.MEMORY_RUBRIC:
                ratings["a"][dimension] = 5
                ratings["b"][dimension] = 3
            ratings["preference"] = "a"

    result = quality.analyse_ratings(sheet, key)

    assert result["completed_ratings"] == 6
    assert result["completed_cases"] == 2
    assert result["distinct_samples"] == 2
    for dimension in quality.MEMORY_RUBRIC:
        summary = result["dimensions"][dimension]
        assert summary["significance_unit"] == "sample_id"
        # Two audio samples, so the test runs on two values regardless of how
        # many raters or repeats produced them.
        assert summary["n_samples_tested"] == 2
        assert summary["n_cases_tested"] == 2
        # All six ratings are still reported for transparency; they simply are
        # not what the significance test consumes.
        assert summary["paired_delta_on_minus_off"]["count"] == 6
        assert summary["case_mean_delta_on_minus_off"]["count"] == 2
        assert summary["sample_mean_delta_on_minus_off"]["count"] == 2
        # Two same-signed samples can never beat p=0.5 under sign flipping.
        assert summary["exact_sign_flip_p"] == 0.5


def test_repeated_runs_of_one_audio_collapse_into_a_single_sample():
    """Re-running the same audio must not double the significance sample size.

    The benchmark loops over its manifest, so three scenarios run twice produce
    six blind cases carrying only three independent samples.  Testing on cases
    would treat each repeat as new evidence and understate the p value, which
    is what made a 6-pair memory ablation look statistically significant.
    """
    raters = ["r1"]
    sheet, key = quality.prepare_blind_sheet(
        _repeated_sample_benchmark(), seed=13, raters=raters
    )
    key_by_case = {case["case_id"]: case for case in key["cases"]}

    assert len(sheet["cases"]) == 2, "each repeat still gets its own blind case"
    assert {case["sample_id"] for case in key["cases"]} == {"sample-repeat"}

    for case in sheet["cases"]:
        on_is_a = key_by_case[case["case_id"]]["label_a"] == "memory_on"
        on_side, off_side = ("a", "b") if on_is_a else ("b", "a")
        ratings = case["ratings"]["r1"]
        for dimension in quality.MEMORY_RUBRIC:
            ratings[on_side][dimension] = 5
            ratings[off_side][dimension] = 3
        ratings["preference"] = on_side

    result = quality.analyse_ratings(sheet, key)

    assert result["completed_cases"] == 2
    assert result["distinct_samples"] == 1
    for dimension in quality.MEMORY_RUBRIC:
        summary = result["dimensions"][dimension]
        assert summary["n_cases_tested"] == 2
        assert summary["n_samples_tested"] == 1, "repeats are one sample"
        # A single sample cannot produce a two-sided sign flip p below 1.0.
        assert summary["exact_sign_flip_p"] == 1.0
        # The by-case p value stays available for diagnosis and is more
        # permissive precisely because it double counts the repeat.
        assert summary["exact_sign_flip_p_by_case"] == 0.5


def test_case_means_average_disagreeing_raters_before_testing():
    """A case where raters disagree contributes its mean, not each opinion."""
    raters = ["r1", "r2"]
    sheet, key = quality.prepare_blind_sheet(
        _benchmark(), seed=5, raters=raters
    )
    key_by_case = {case["case_id"]: case for case in key["cases"]}
    for case in sheet["cases"]:
        on_is_a = key_by_case[case["case_id"]]["label_a"] == "memory_on"
        # r1 prefers the on condition by 2 points, r2 by nothing.
        for rater, on_score in (("r1", 5), ("r2", 3)):
            ratings = case["ratings"][rater]
            on_side, off_side = ("a", "b") if on_is_a else ("b", "a")
            for dimension in quality.MEMORY_RUBRIC:
                ratings[on_side][dimension] = on_score
                ratings[off_side][dimension] = 3

    result = quality.analyse_ratings(sheet, key)

    for dimension in quality.MEMORY_RUBRIC:
        summary = result["dimensions"][dimension]
        assert summary["n_cases_tested"] == 2
        # Per-case mean of a +2 and a 0 rater difference.
        assert summary["case_mean_delta_on_minus_off"]["mean"] == 1.0


def test_analyse_rejects_out_of_range_scores():
    sheet, key = quality.prepare_blind_sheet(
        _benchmark(), seed=1, raters=["r1"]
    )
    for dimension in quality.MEMORY_RUBRIC:
        sheet["cases"][0]["ratings"]["r1"]["a"][dimension] = 6
        sheet["cases"][0]["ratings"]["r1"]["b"][dimension] = 3

    with pytest.raises(quality.QualityEvaluationError, match="1 到 5"):
        quality.analyse_ratings(sheet, key)


def test_prepare_skips_emotion_fallback_pairs():
    benchmark = {
        "configuration": {"ablation": "emotion"},
        "records": [
            {
                **_record("emotion_on", 1),
                "emotion_source": "text_fallback_model_unavailable",
                "emotion_audio_model_used": False,
            },
            {
                **_record("emotion_off", 1),
                "emotion_source": "disabled",
                "emotion_audio_model_used": False,
            },
        ],
    }

    with pytest.raises(quality.QualityEvaluationError, match="没有可用于盲评"):
        quality.prepare_blind_sheet(benchmark, seed=1, raters=["r1"])


def test_parse_llm_judgement_accepts_fenced_json():
    dimensions = list(quality.MEMORY_RUBRIC)
    payload = {
        "a": {dimension: 4 for dimension in dimensions},
        "b": {dimension: 2 for dimension in dimensions},
        "preference": "a",
        "reason": "A 更贴合上下文。",
    }
    parsed = quality._parse_llm_judgement(
        "```json\n" + __import__("json").dumps(payload) + "\n```", dimensions
    )
    assert parsed["a"]["relevance"] == 4
    assert parsed["preference"] == "a"


def test_judge_blind_sheet_maps_swapped_positions(monkeypatch):
    sheet, _ = quality.prepare_blind_sheet(_benchmark(), seed=3, raters=["placeholder"])
    dimensions = list(quality.MEMORY_RUBRIC)
    payload = {
        "a": {dimension: 5 for dimension in dimensions},
        "b": {dimension: 1 for dimension in dimensions},
        "preference": "a",
        "reason": "测试",
    }

    def fake_completion(**kwargs):
        return __import__("json").dumps(payload)

    monkeypatch.setattr(quality, "_chat_completion", fake_completion)
    judged = quality.judge_blind_sheet(
        sheet,
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        raters=["llm_1"],
        seed=3,
    )
    assert judged["raters"] == ["llm_1"]
    assert len(judged["cases"]) == 2
    for case in judged["cases"]:
        rating = case["ratings"]["llm_1"]
        assert rating["a"]["relevance"] in {1, 5}
        assert rating["b"]["relevance"] in {1, 5}
        assert rating["preference"] in {"a", "b"}
