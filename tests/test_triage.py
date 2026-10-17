import copy

from autosub.comparison import ComparisonSettings, ReferenceCue, compare, set_decision
from autosub.domain import Cue, Project
from autosub.triage import quality_diagnostics, review_order, review_priority


def test_priority_preserves_session_and_places_unmatched_and_large_errors_first():
    p = Project(duration_ms=12000, cues=[Cue("a", 1000, 2000, "A", 1000, 1800),
        Cue("b", 5000, 7000, "B", 5000, 6800), Cue("c", 10000, 11000, "C", 10000, 10800)])
    session = compare(p, [ReferenceCue(1000, 2000, "甲", "Default"),
                          ReferenceCue(4500, 6400, "乙", "Default")],
                      ComparisonSettings(timeline_confirmed=True))
    before = copy.deepcopy(session)
    order = review_order(session)
    assert session == before
    assert [review_priority(session["rows"][i])["score"] for i in order] == [100, 60, 0]
    assert session["rows"][order[0]]["kind"] == "unmatched_generated"
    set_decision(session, order[0], "redo", "tester", "需試聽")
    assert order[0] not in review_order(session)


def test_diagnostics_marks_only_observable_risks_without_modifying_cues():
    p = Project(duration_ms=12000, cues=[
        Cue("a", 1000, 9000, "字" * 50, 1000, 8800,
            timing_trace={"raw": [1000, 8900]}), Cue("b", 10000, 11000, "好", 10000, 10800)])
    before = copy.deepcopy(p)
    risks = quality_diagnostics(p)
    assert len(risks) == 1 and len(risks[0]["reasons"]) == 3
    assert p == before
