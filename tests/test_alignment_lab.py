import copy

import pytest

from autosub.alignment_lab import assess, validate_result, restore_unit_text, atom_ownership, stitch_atoms


def sample():
    case = {"id": "case", "text": "今日はテスト。", "window_ms": [1000, 6000], "baseline_ms": [2000, 4000]}
    result = {"id": "case", "text": case["text"], "units": [
        {"text": "今日は", "start": 1.0, "end": 2.0}, {"text": "テスト", "start": 2.0, "end": 3.0}]}
    return case, result


def test_aligned_units_retain_source_clock_and_never_claim_human_accuracy():
    case, result = sample()
    before = copy.deepcopy((case, result))
    checked = assess(case, result)
    assert checked == {"status": "candidate", "errors": [], "warnings": [],
                       "span_ms": [2000, 4000], "human_verified": False}
    assert (case, result) == before


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "1"])
def test_missing_and_invalid_times_cannot_be_adopted(bad):
    case, result = sample()
    result["units"][0]["start"] = bad
    assert assess(case, result)["status"] == "failed"


def test_missing_head_character_fails_instead_of_hiding_coverage_loss():
    case, result = sample()
    result["units"][0]["text"] = "日は"
    assert "aligned-text-coverage-mismatch" in assess(case, result)["errors"]


def test_changed_input_text_and_wrong_case_fail():
    case, result = sample()
    result.update(id="other", text="改字")
    assert "case-or-text-mismatch" in assess(case, result)["errors"]


@pytest.mark.parametrize("a,b", [(-.1, 2), (1, 6), (2, 1)])
def test_invalid_window_times_fail(a, b):
    case, result = sample()
    result["units"][0].update(start=a, end=b)
    assert assess(case, result)["status"] == "failed"


def test_nonmonotonic_unit_times_fail():
    case, result = sample()
    result["units"][1]["start"] = .9
    assert "nonmonotonic-time" in assess(case, result)["errors"]


def test_backend_repair_and_zero_length_are_visible_review_warnings():
    case, result = sample()
    result["timestamp_corrections"] = [{"changed_indices": [0], "raw": [4000, 2000], "fixed": [2000, 2000]}]
    result["units"][0]["end"] = 1
    checked = assess(case, result)
    assert checked["status"] == "review-required"
    assert set(checked["warnings"]) == {"backend-repaired-timestamps", "zero-duration-unit"}


def test_large_shift_and_window_edge_require_review():
    case, result = sample()
    case["baseline_ms"][0] = 2100
    result["units"][0]["start"] = 0
    assert set(assess(case, result)["warnings"]) == {"boundary-shift-over-1000ms", "input-window-edge"}


@pytest.mark.parametrize("results", [[], ["case", "case"], ["other"]])
def test_missing_or_duplicate_results_rejected(results):
    case, result = sample()
    manifest = {"manifest_sha256": "same", "cases": [case]}
    output = {"manifest_sha256": "same", "results": [{**result, "id": name} for name in results]}
    with pytest.raises(ValueError, match="missing/duplicate"):
        validate_result(manifest, output)


def test_input_manifest_hash_required():
    case, result = sample()
    with pytest.raises(ValueError, match="different input"):
        validate_result({"manifest_sha256": "new", "cases": [case]},
                        {"manifest_sha256": "old", "results": [result]})
    with pytest.raises(ValueError, match="different input"):
        validate_result({"cases": [case]}, {"results": [result]})


@pytest.mark.parametrize("text,parts", [
    ("「今日は、東京駅へ。」", ["今日", "は", "東京", "駅", "へ"]),
    ("We're playing APEX!", ["We", "re", "playing", "apex"]),
    ("今天玩ＡＰＥＸ。", ["今", "天", "玩", "apex"]),
    ("오늘 게임을 합니다!", ["오늘", "게임", "을", "합니다"]),
    ("ハ\u309aンです。", ["パン", "です"]),
])
def test_unit_mapping_preserves_exact_original_text_and_model_times(text, parts):
    units = [{"text": p, "start": i, "end": i + 1} for i, p in enumerate(parts)]
    mapped = restore_unit_text(text, units)
    assert "".join(u["text"] for u in mapped) == text
    assert [(u["start"], u["end"]) for u in mapped] == [(u["start"], u["end"]) for u in units]
    assert [u["text"] for u in units] == parts


def test_mapping_rejects_missing_text_and_splitting_a_normalized_source_character():
    with pytest.raises(ValueError, match="cover"):
        restore_unit_text("東京駅", [{"text": "京駅", "start": 0, "end": 1}])
    with pytest.raises(ValueError, match="splits"):
        restore_unit_text("ß", [{"text": "s", "start": 0, "end": 1}, {"text": "s", "start": 1, "end": 2}])


def test_whole_lexical_atom_is_owned_once_across_overlapping_windows():
    span = [24000, 27000]
    assert atom_ownership(*span, [0, 26000], [0, 25000], 50000) == "context"
    assert atom_ownership(*span, [24000, 50000], [25000, 50000], 50000) == "owned"


def test_outside_context_unit_does_not_invalidate_owned_words():
    assert atom_ownership(25500, 26080, [0, 26000], [0, 25000], 50000) == "context"
    assert atom_ownership(23000, 24000, [0, 26000], [0, 25000], 50000) == "owned"


def test_invalid_owned_unit_cannot_be_silently_clamped_into_the_audio():
    assert atom_ownership(25500, 26080, [0, 26000], [0, 50000], 50000) == "invalid-owned"
    assert atom_ownership(49500, 50080, [48000, 50000], [49000, 50000], 50000) == "invalid-owned"


def test_stitching_does_not_interleave_duplicate_counter_from_overlapping_context():
    left = [{"text": "結構", "start_ms": 23000, "end_ms": 24000},
            {"text": "1分", "start_ms": 24000, "end_ms": 25600}]
    right = [{"text": "1分", "start_ms": 24200, "end_ms": 25800},
             {"text": "できる。", "start_ms": 25800, "end_ms": 27000}]
    windows = [{"id": "a", "atoms": left, "window_ms": [0, 27000], "owner_ms": [0, 25000]},
               {"id": "b", "atoms": right, "window_ms": [23000, 40000], "owner_ms": [25000, 40000]}]
    before = copy.deepcopy(windows)
    selected, seams = stitch_atoms(windows)
    assert "".join(a["text"] for a in selected) == "結構1分できる。"
    assert windows == before
    assert all(atom in left + right for atom in selected)
    assert seams[0]["status"] == "lexical-seam-candidate"


def test_real_repeated_words_with_distinct_times_survive_stitching():
    both = [{"text": "もう", "start_ms": 23500, "end_ms": 24000},
            {"text": "1分", "start_ms": 24000, "end_ms": 24500},
            {"text": "1分", "start_ms": 25000, "end_ms": 25500},
            {"text": "やる。", "start_ms": 25500, "end_ms": 27000}]
    windows = [{"id": "a", "atoms": both, "window_ms": [0, 28000], "owner_ms": [0, 24750]},
               {"id": "b", "atoms": both, "window_ms": [23000, 40000], "owner_ms": [24750, 40000]}]
    selected, seams = stitch_atoms(windows)
    assert "".join(a["text"] for a in selected) == "もう1分1分やる。"


def test_unresolvable_seam_keeps_baseline_and_marks_review():
    windows = [{"id": "a", "atoms": [{"text": "前", "start_ms": 0, "end_ms": 1000}],
                "window_ms": [0, 2000], "owner_ms": [0, 10000]},
               {"id": "b", "atoms": [{"text": "後", "start_ms": 20000, "end_ms": 21000}],
                "window_ms": [19000, 22000], "owner_ms": [10000, 22000]}]
    selected, seams = stitch_atoms(windows)
    assert "".join(a["text"] for a in selected) == "前後"
    assert seams[0]["status"] == "unresolved-use-midpoint-ownership"


def test_same_words_at_different_audio_times_are_not_deduplicated():
    windows = [{"id": "a", "atoms": [{"text": "1分", "start_ms": 24000, "end_ms": 24500}],
                "window_ms": [23000, 27000], "owner_ms": [0, 24750]},
               {"id": "b", "atoms": [{"text": "1分", "start_ms": 25000, "end_ms": 25500}],
                "window_ms": [23000, 27000], "owner_ms": [24750, 30000]}]
    selected, seams = stitch_atoms(windows)
    assert "".join(a["text"] for a in selected) == "1分1分"
    assert seams[0]["matched_overlap_characters"] == 0


def test_seam_prefers_preserving_active_speech_over_small_distance_to_grid():
    left = [{"text": "前", "start_ms": 22000, "end_ms": 24000},
            {"text": "完整詞", "start_ms": 24000, "end_ms": 26500}]
    right = [{"text": "別的詞", "start_ms": 25000, "end_ms": 26500},
             {"text": "後", "start_ms": 26500, "end_ms": 27500}]
    windows = [{"id": "a", "atoms": left, "window_ms": [20000, 28000], "owner_ms": [0, 25000]},
               {"id": "b", "atoms": right, "window_ms": [23000, 29000], "owner_ms": [25000, 30000]}]
    selected, seams = stitch_atoms(windows, speech=[[24000, 26500]])
    assert "".join(a["text"] for a in selected) == "前完整詞後"
    assert seams[0]["active_speech_gap_ms"] == 0
