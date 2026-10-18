from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from autosub.asr_complementarity import assess_case, complementarity_report, render_report


def case(primary="red kite", auxiliary="blue kite", reference="red kite", scope="dataset-complete"):
    def run(text):
        return {"status": "complete", "text": text, "input": {
            "samples_sha256": "a" * 64, "sample_rate": 16000,
            "sample_count": 48000, "language": "en"}}
    return {"id": "sample", "language": "en", "primary": run(primary),
            "auxiliary": run(auxiliary), "reference": {
                "text": reference, "scope": scope, "uncertain": False}}


def test_oracle_is_whole_window_and_counts_corruption_separately():
    a = case("red kite", "blue kite", "red kite")
    b = case("tall boat", "small boat", "small boat")
    b["id"] = "other"
    report = complementarity_report({"schema_version": 1, "cases": [a, b]})
    summary = report["by_language"]["en"]
    assert summary["whole_window_oracle_edits"] == 0
    assert summary["primary_edits"] > 0 and summary["auxiliary_edits"] > 0
    assert summary["potential_edits_saved"] > 0
    assert summary["always_auxiliary_extra_edits"] > 0
    assert summary["primary_exact_auxiliary_not"] == 1
    assert summary["auxiliary_exact_primary_not"] == 1
    # A character-level oracle could reconstruct this reference; window oracle cannot.
    score = assess_case(case("axc", "abx", "abc"))["scores"]
    assert score["whole_window_oracle_edits"] == 1
    assert score["both_nonexact"] is True


@pytest.mark.parametrize("change", ["hash", "missing_hash", "rate", "count", "language", "failed", "bool_count"])
def test_unpaired_never_gets_scores(change):
    value = case()
    other = value["auxiliary"]
    if change == "hash":
        other["input"]["samples_sha256"] = "b" * 64
    elif change == "missing_hash":
        other["input"].pop("samples_sha256")
    elif change == "rate":
        other["input"]["sample_rate"] = 48000
    elif change == "count":
        other["input"]["sample_count"] += 1
    elif change == "language":
        other["input"]["language"] = "ja"
    elif change == "bool_count":
        other["input"]["sample_count"] = True
    else:
        other["status"] = "failed"
    report = complementarity_report({"schema_version": 1, "cases": [value]})
    assert report["cases"][0]["status"] == "unpaired"
    assert report["cases"][0]["scores"] is None
    assert report["by_language"]["en"]["primary_edits"] is None
    assert report["cases"][0]["auxiliary"] == other


@pytest.mark.parametrize("scope", ["partial", "unreviewed", "unknown", "human-complete"])
def test_incomplete_references_only_produce_differences(scope):
    value = case(scope=scope)
    row = assess_case(value)
    assert row["scores"] is None and row["differences"]
    assert row["status"] == "paired-unrated"


def test_human_review_must_be_explicit_and_uncertainty_is_not_ignored():
    value = case(scope="human-complete")
    value["reference"]["reviewer_confirmed"] = True
    assert assess_case(value)["status"] == "paired-rated"
    for uncertainty in (True, None, [], "false"):
        value["reference"]["uncertain"] = uncertainty
        assert assess_case(value)["scores"] is None


def test_raw_difference_offsets_are_not_normalized_word_positions():
    value = case("ＡＢ， c", "ab c", "ABC")
    before = deepcopy(value)
    row = assess_case(value)
    assert row["normalized_text_equal"] is True and row["raw_text_equal"] is False
    assert row["scores"]["primary"]["character_edits"] == 0
    assert row["scores"]["auxiliary"]["character_edits"] == 0
    for diff in row["differences"]:
        i, j = diff["primary_offsets"]
        k, l = diff["auxiliary_offsets"]
        assert value["primary"]["text"][i:j] == diff["primary_text"]
        assert value["auxiliary"]["text"][k:l] == diff["auxiliary_text"]
    assert value == before


def test_no_cross_language_pooling_or_silent_exclusions():
    first, second = case(), case(scope="partial")
    second["id"] = "unrated"
    second["language"] = "ja"
    for name in ("primary", "auxiliary"):
        second[name]["input"]["language"] = "ja"
    report = complementarity_report({"schema_version": 1, "cases": [first, second]})
    assert set(report["by_language"]) == {"en", "ja"}
    assert report["by_language"]["ja"]["unrated_cases"] == 1
    assert report["by_language"]["ja"]["whole_window_oracle_edits"] is None
    assert len(report["cases"]) == 2


def test_empty_hypothesis_is_scored_but_empty_reference_is_not():
    assert assess_case(case("", "abc", "abc"))["scores"]["primary"]["deletion"] == 3
    assert assess_case(case(reference="!"))["scores"] is None


def test_model_agreement_does_not_imply_correct_text():
    value = case("blue ship", "BLUE SHIP!", "red ship")
    row = assess_case(value)
    assert row["scores"]["agreement_both_nonexact"] is True
    assert row["scores"]["whole_window_oracle_edits"] > 0


def test_short_window_limits_prevent_large_edit_grids():
    assert assess_case(case("a" * 10001))["status"] == "unrated"
    assert assess_case(case("a" * 1100, "b" * 1100, "c" * 1100))["scores"] is None


@pytest.mark.parametrize("manifest", [{}, {"schema_version": 1, "cases": [case(), case()]},
    {"schema_version": 1, "cases": [{}]}])
def test_bad_manifests_rejected(manifest):
    with pytest.raises(ValueError):
        complementarity_report(manifest)


def test_html_escapes_all_untrusted_text():
    value = case("<script>alert(1)</script>")
    value["id"] = "<img onerror=x>"
    rendered = render_report(complementarity_report({"schema_version": 1, "cases": [value]}))
    assert "<script>" not in rendered and "<img onerror=x>" not in rendered
    assert "&lt;script&gt;" in rendered and "&lt;img onerror=x&gt;" in rendered


def test_cli_preserves_inputs_and_refuses_existing_output(tmp_path):
    manifest, output = tmp_path / "manifest.json", tmp_path / "report"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case()]}), encoding="utf-8")
    before = manifest.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_asr_complementarity.py"
    command = [sys.executable, str(script), str(manifest), "--output", str(output)]
    first = subprocess.run(command, capture_output=True)
    assert first.returncode == 0, first.stderr
    saved = (output / "report.json").read_bytes()
    second = subprocess.run(command, capture_output=True)
    assert second.returncode != 0
    assert (output / "report.json").read_bytes() == saved and manifest.read_bytes() == before
