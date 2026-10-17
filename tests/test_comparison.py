import copy
import json
from threading import Event

import pytest

from autosub.comparison import (ComparisonSettings, compare, compare_file, export_report,
    load_session, read_reference, repair_pair, save_session, set_decision, summary, validate_session)
from autosub.domain import Cue, Project
from autosub.evaluation import ReferenceCue
from autosub.media import Cancelled


def project(intervals):
    return Project(duration_ms=20000, cues=[Cue(f"gen-{i}", a, b, text, a, b)
                                          for i, (a, b, text) in enumerate(intervals)])


def refs(intervals):
    return [ReferenceCue(a, b, text, "Default") for a, b, text in intervals]


def test_exact_and_shifted_pairs_keep_direction_and_unreviewed_denominator():
    p = project([(1020, 1950, "日文"), (4200, 5000, "他の言葉")])
    r = refs([(1000, 2000, "中文翻譯"), (4000, 5000, "另一句"), (10000, 11000, "未辨識")])
    before = copy.deepcopy(p)
    session = compare(p, r)
    assert p == before
    assert session["reference_used_by_asr"] is False
    assert session["rows"][0]["start_delta_ms"] == 20
    assert session["rows"][0]["end_delta_ms"] == -50
    result = summary(session)
    assert result["strict_time_estimate_rate"] == pytest.approx(1 / 3)
    assert result["human_confirmed_usable_rate"] is None
    assert result["mapping_counts"]["unmatched_reference"] == 1
    validate_session(session)


@pytest.mark.parametrize("reference,generated,kind", [
    ([(1000, 3000, "一句")], [(1000, 1900, "前"), (2100, 3000, "後")], "1:2"),
    ([(1000, 1900, "前"), (2100, 3000, "後")], [(1000, 3000, "一句")], "2:1"),
    ([(1000, 4000, "一句")], [(1000, 1800, "一"), (1900, 2800, "二"), (2900, 4000, "三")], "1:3"),
])
def test_split_merge_matching_without_reusing_events(reference, generated, kind):
    session = compare(project(generated), refs(reference))
    assert [r["kind"] for r in session["rows"]] == [kind]
    assert summary(session)["strict_time_estimate_rate"] == 0
    validate_session(session)


def test_overlap_and_gaps_do_not_become_ready_estimates():
    p = project([(1000, 3000, "甲"), (1500, 2500, "乙")])
    session = compare(p, refs([(1000, 3000, "A"), (1500, 2500, "B")]))
    assert all(r["ambiguous"] for r in session["rows"])
    assert summary(session)["strict_time_estimate_rate"] == 0
    # An envelope covering a long silent gap cannot justify a merged match.
    session = compare(project([(1000, 11000, "跨越空白")]), refs([(1000, 2000, "A"), (10000, 11000, "B")]))
    assert all(r["kind"] not in {"2:1", "1:1"} for r in session["rows"])


def test_offset_and_window_edges_are_explicit():
    p = project([(1000, 2000, "A")])
    session = compare(p, refs([(61000, 62000, "翻譯"), (59000, 60500, "跨界")]),
                      ComparisonSettings(reference_offset_ms=60000))
    assert session["clipped_reference_events_excluded"] == 1
    assert summary(session)["strict_time_estimate_rate"] == 1
    assert session["rows"][0]["start_delta_ms"] == 0


def test_manual_decision_requires_timeline_and_reviewer_and_never_inflates():
    session = compare(project([(1000, 2000, "A")]), refs([(1000, 2000, "甲"), (8000, 9000, "乙")]))
    with pytest.raises(ValueError, match="偏移"):
        set_decision(session, 0, "usable", "reviewer")
    session["settings"]["timeline_confirmed"] = True
    with pytest.raises(ValueError, match="複核者"):
        set_decision(session, 0, "usable", "")
    set_decision(session, 0, "usable", "reviewer")
    assert summary(session)["human_confirmed_usable_rate"] == .5
    with pytest.raises(ValueError, match="未配對"):
        set_decision(session, 1, "usable", "reviewer")
    set_decision(session, 1, "exclude", "reviewer", "非對話事件")
    assert summary(session)["human_confirmed_usable_rate"] == .5


def test_manual_pair_correction_displaces_but_does_not_drop_other_events():
    session = compare(project([(1000, 2000, "A"), (2200, 3000, "B")]),
                      refs([(1000, 2000, "甲"), (2200, 3000, "乙")]))
    repair_pair(session, ["ref-0"], ["gen-1"])
    validate_session(session)
    assert len(session["rows"]) == 3
    repaired = next(r for r in session["rows"] if r["pairing"] == "manual")
    assert repaired["generated_ids"] == ["gen-1"] and repaired["decision"] == "unreviewed"


def test_srt_reader_preserves_translated_content(tmp_path):
    path = tmp_path / "reference.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n<i>翻譯</i>\n第二行\n\n2\n00:00:03,000 --> 00:00:04,000\n另一句\n", encoding="utf-8-sig")
    events, video = read_reference(path)
    assert [(r.start_ms, r.end_ms) for r in events] == [(1000, 2000), (3000, 4000)]
    assert events[0].text == "翻譯 第二行" and video == ""


def test_session_resume_refuses_changed_subtitle_or_project(tmp_path):
    ref = tmp_path / "ref.srt"
    ref.write_text("1\n00:00:01,000 --> 00:00:02,000\n原參考\n", encoding="utf-8")
    p = project([(1000, 2000, "A")])
    session = compare_file(p, ref)
    out = tmp_path / "session.json"
    save_session(session, out)
    assert load_session(out, p)["rows"] == session["rows"]
    p.cues[0].text = "changed"
    with pytest.raises(ValueError, match="專案"):
        load_session(out, p)
    p.cues[0].text = "A"
    ref.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="參考"):
        load_session(out, p)


def test_safe_reports_and_source_preservation(tmp_path):
    p = project([(1000, 2000, "<script>alert(1)</script>")])
    session = compare(p, refs([(1000, 2000, "=FORMULA()")]))
    export_report(session, tmp_path / "report.html")
    assert "<script>" not in (tmp_path / "report.html").read_text(encoding="utf-8")
    export_report(session, tmp_path / "report.csv")
    assert "'=FORMULA()" in (tmp_path / "report.csv").read_text(encoding="utf-8-sig")
    protected = tmp_path / "original.ass"
    protected.write_text("original", encoding="utf-8")
    session["reference_fingerprint"] = {"path": str(protected.resolve())}
    with pytest.raises(ValueError, match="覆寫"):
        save_session(session, protected)
    assert protected.read_text(encoding="utf-8") == "original"


def test_cancel_and_empty_comparison():
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        compare(project([]), [], cancel=cancel)
    result = summary(compare(project([]), []))
    assert result["strict_time_estimate_rate"] is None
    assert result["human_confirmed_usable_rate"] is None


def test_identical_reference_can_be_relocated_without_losing_decisions(tmp_path):
    old = tmp_path / "old.srt"
    new = tmp_path / "new.srt"
    old.write_text("1\n00:00:01,000 --> 00:00:02,000\n翻譯\n", encoding="utf-8")
    p = project([(1000, 2000, "A")])
    session = compare_file(p, old, ComparisonSettings(timeline_confirmed=True))
    set_decision(session, 0, "usable", "reviewer")
    path = tmp_path / "session.json"
    save_session(session, path)
    new.write_bytes(old.read_bytes())
    old.unlink()
    resumed = load_session(path, p, new)
    assert resumed["rows"][0]["decision"] == "usable"
    assert resumed["reference_fingerprint"]["path"] == str(new.resolve())
