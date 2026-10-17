import copy

import pytest

from autosub.quality_gate import compare_quality


def report():
    return {"results": [{"clip": name, "policy": policy, "project_sha256": "same-project-" + name,
        "reference_sha256": "same-reference-" + name, "tolerance_ms": 100, "reference_offset_ms": 0,
        "summary": {"reference_events": 10,
        "strict_time_estimate_rate": .5, "group_outer_boundary_estimate_rates": {"100ms": .5, "250ms": .8}}}
        for name in ("a", "b") for policy in ("legacy", "conservative")]}


def test_each_clip_must_pass_even_when_overall_average_improves():
    data = report()
    data["results"][1]["summary"]["strict_time_estimate_rate"] = .4
    data["results"][3]["summary"]["strict_time_estimate_rate"] = .9
    before = copy.deepcopy(data)
    result = compare_quality(data, data)
    assert not result["passed"] and result["rows"][0]["regressed_metrics"] == ["strict_100ms"]
    assert data == before


def test_temporal_pass_never_automatically_promotes_default():
    result = compare_quality(report(), report())
    assert result["passed"] and not result["default_promotion_authorized"]


@pytest.mark.parametrize("change", ["missing", "changed_source", "changed_reference", "nan", "duplicate", "tolerance", "offset"])
def test_incomparable_or_invalid_reports_are_rejected(change):
    candidate = report()
    if change == "missing":
        candidate["results"].pop()
    elif change == "changed_source":
        candidate["results"][1]["project_sha256"] = "different"
    elif change == "changed_reference":
        candidate["results"][1]["reference_sha256"] = "different"
    elif change == "nan":
        candidate["results"][1]["summary"]["strict_time_estimate_rate"] = float("nan")
    elif change == "tolerance":
        candidate["results"][1]["tolerance_ms"] = 500
    elif change == "offset":
        candidate["results"][1]["reference_offset_ms"] = 500
    else:
        candidate["results"].append(copy.deepcopy(candidate["results"][1]))
    with pytest.raises(ValueError):
        compare_quality(report(), candidate)
