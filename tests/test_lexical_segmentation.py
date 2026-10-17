from collections import Counter
import copy

import pytest

pytest.importorskip("sudachipy")
pytest.importorskip("sudachidict_core")
pytest.importorskip("budoux")

from autosub.domain import Cue, Project, TimingSettings, Word, word_key
from autosub.lexical_segmentation import boundary_map, lexical_groups, segment_lexically
from autosub.timing import apply_timing


def project(texts, gap_at=1, gap=650):
    words, cursor = [], 100
    for i, text in enumerate(texts):
        if i == gap_at:
            cursor += gap
        words.append(Word(cursor, cursor + 140, text))
        cursor += 140
    return Project(language="ja", duration_ms=cursor + 1000, words=words,
                   speech=[[w.start_ms, w.end_ms] for w in words])


@pytest.mark.parametrize("unit", ["東京駅", "プランク", "配置", "嬉しい", "来週", "毎日", "旅行", "お母さん", "あんた"])
def test_every_internal_character_pause_keeps_lexical_unit_together(unit):
    for at in range(1, len(unit)):
        p = project(["終わった。", *unit, "です。"], at + 1)
        groups, _, mapping = lexical_groups(p)
        assert any(unit in "".join(w.text for w in group) for group in groups)
        assert Counter(word_key(w) for group in groups for w in group) == Counter(map(word_key, p.words))


def test_false_pause_moves_onset_with_complete_word_to_following_cue():
    p = project(["終わった", "プ", "ランク", "をする。"], 2)
    groups, decisions, _ = lexical_groups(p)
    assert ["".join(w.text for w in group) for group in groups] == ["終わった", "プランクをする。"]
    assert decisions[0]["chosen_token"] == 1


def test_digit_counter_and_particle_do_not_start_as_orphans():
    p = project(["昨日は", "1", "分", "だけ", "やった。"], 2)
    groups, _, mapping = lexical_groups(p)
    assert 2 not in mapping.allowed and 3 not in mapping.allowed
    assert any("1分だけ" in "".join(w.text for w in group) for group in groups)


def test_non_independent_verb_stays_with_question_word():
    p = project(["言った。", "何", "して", "いるの？"], 2)
    assert 2 not in boundary_map(p.words).allowed


def test_an_acoustic_island_cannot_join_two_unrelated_oov_nouns():
    p = project(["トア", "ドア"], 1, 3000)
    groups, _, _ = lexical_groups(p)
    assert ["".join(w.text for w in group) for group in groups] == ["トア", "ドア"]


def test_existing_whisper_token_is_never_split_even_if_dictionary_would_split_it():
    p = project(["最初。", "今日東京駅に行った。"], 1, 650)
    p.settings.max_chars = 2
    cues = apply_timing(p, segmenter=segment_lexically)
    assert any(c.text == "今日東京駅に行った。" for c in cues)
    assert any("字數上限" in f for c in cues for f in c.flags)


def test_oversized_atomic_word_is_flagged_and_never_cut_or_retimed_by_characters():
    p = project(["プ", "ランク"], 1, 650)
    p.words[1].end_ms = 9000
    p.duration_ms = 10000
    cues = apply_timing(p, segmenter=segment_lexically)
    assert len(cues) == 1 and cues[0].text == "プランク"
    assert any("分段上限" in f for f in cues[0].flags)


def test_manual_locked_text_times_and_source_words_are_preserved():
    p = project(["手動。", "次", "の", "話。"], 2)
    p.cues = [Cue("manual", 50, 260, "人工修正", 100, 240, locked=True,
                  source_tokens=[word_key(p.words[0])])]
    before = copy.deepcopy(p)
    cues = apply_timing(p, segmenter=segment_lexically)
    locked = next(c for c in cues if c.id == "manual")
    assert {k: v for k, v in vars(locked).items() if k != "flags"} == {
        k: v for k, v in vars(before.cues[0]).items() if k != "flags"}
    assert p == before
    assert Counter(t for c in cues for t in c.source_tokens) == Counter(map(word_key, p.words))


def test_neighbour_vad_cannot_expand_across_nonoverlapping_raw_groups():
    p = project(["できた。", "二", "人", "で行く。"], 2)
    p.speech = [[100, 550], [750, 2000]]
    cues = apply_timing(p, segmenter=segment_lexically)
    assert all(a.end_ms <= b.start_ms for a, b in zip(cues, cues[1:]))


def test_rejects_other_languages_instead_of_applying_japanese_dictionary():
    p = project(["Hello", " world."])
    p.language = "en"
    with pytest.raises(ValueError, match="日文"):
        segment_lexically(p)


def test_empty_project_returns_no_cues():
    assert segment_lexically(Project(language="ja")) == []
