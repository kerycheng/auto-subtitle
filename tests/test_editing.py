import pytest

from autosub.domain import Cue, Project, Word, word_key
from autosub.editing import merge_cues, review_cues, set_lock_state, split_cue
from autosub.timing import apply_timing


def sample():
    p = Project(duration_ms=5000, language="ja", words=[Word(1000, 1700, "前半"), Word(1700, 2500, "後半。")],
                speech=[[1000, 2500]])
    p.cues = apply_timing(p)
    return p


def test_split_and_merge_preserve_text_ownership_and_locks():
    p = sample()
    before = p.cues[0]
    p.cues = split_cue(p, 0, 1700, "前半", "後半。")
    assert p.cues[0].end_ms == p.cues[1].start_ms == 1700
    assert all(c.locked for c in p.cues)
    assert set(p.cues[0].source_tokens + p.cues[1].source_tokens) == set(before.source_tokens)
    assert [c.text for c in apply_timing(p)] == ["前半", "後半。"]
    p.cues = merge_cues(p, 0)
    assert p.cues[0].text == before.text
    assert (p.cues[0].start_ms, p.cues[0].end_ms) == (before.start_ms, before.end_ms)
    assert p.cues[0].locked


def test_split_inside_one_word_and_retime_does_not_duplicate_text():
    p = Project(duration_ms=4000, words=[Word(1000, 3000, "前半後半。")], speech=[[1000, 3000]])
    p.cues = apply_timing(p)
    p.cues = split_cue(p, 0, 2000, "前半", "後半。")
    assert "".join(c.text for c in apply_timing(p)) == "前半後半。"


def test_invalid_split_does_not_mutate_project():
    p = sample()
    for point, left, right in [(980, "前半", "後半"), (1500, "", "後半")]:
        with pytest.raises(ValueError):
            split_cue(p, 0, point, left, right)
        assert len(p.cues) == 1


def test_english_merge_inserts_word_separator():
    p = Project(duration_ms=5000, language="en", cues=[Cue("1", 100, 900, "Hello", 100, 800),
                                                     Cue("2", 900, 2000, "world", 920, 1800)])
    assert merge_cues(p, 0)[0].text == "Hello world"


def test_review_detects_nested_overlaps_and_removes_stale_warnings():
    p = Project(duration_ms=10000, cues=[Cue("1", 0, 9000, "long", 0, 8800),
                                       Cue("2", 1000, 2000, "short", 1000, 1800),
                                       Cue("3", 3000, 4000, "later", 3000, 3800)])
    review_cues(p)
    assert all(any("重疊" in f for f in cue.flags) for cue in p.cues)
    p.cues[0].end_ms = 900
    review_cues(p)
    assert not any("重疊" in f for cue in p.cues for f in cue.flags)


def test_refined_edge_does_not_lose_word_ownership_on_lock():
    p = Project(duration_ms=5000, words=[Word(700, 1100, "こんな"), Word(1100, 1800, "感じ。")],
                speech=[[1000, 1800]])
    p.cues = apply_timing(p)
    assert p.cues[0].speech_start_ms == 1000
    p.cues[0].locked = True
    p.cues[0].text = "人工修正。"
    assert [c.text for c in apply_timing(p)] == ["人工修正。"]


def test_new_warning_reopens_previously_reviewed_cue():
    p = sample()
    review_cues(p)
    p.cues[0].reviewed = True
    p.cues[0].end_ms = p.cues[0].speech_end_ms + 1000
    review_cues(p)
    assert not p.cues[0].reviewed
    assert "尾留超過上限" in p.cues[0].flags


def test_unlocking_one_in_word_fragment_regenerates_group_without_duplicate():
    p = Project(duration_ms=4000, words=[Word(1000, 3000, "前半後半。")], speech=[[1000, 3000]])
    p.cues = apply_timing(p)
    p.cues = split_cue(p, 0, 2000, "前半", "後半。")
    p.cues = set_lock_state(p, 0, False)
    assert not any(c.locked for c in p.cues)
    assert "".join(c.text for c in apply_timing(p)) == "前半後半。"
    p.cues = set_lock_state(p, 1, True)
    assert all(c.locked for c in p.cues)


def test_merge_joins_existing_split_groups_for_safe_unlock():
    p = Project(duration_ms=8000, words=[Word(1000, 3000, "前半後半。"), Word(4000, 6000, "另一段。")],
                speech=[[1000, 3000], [4000, 6000]])
    p.cues = apply_timing(p)
    p.cues = split_cue(p, 0, 2000, "前半", "後半。")
    p.cues = split_cue(p, 2, 5000, "另一", "段。")
    p.cues = merge_cues(p, 1)
    p.cues = set_lock_state(p, 1, False)
    assert not any(c.locked for c in p.cues)
    assert "".join(c.text for c in apply_timing(p)) == "前半後半。另一段。"
