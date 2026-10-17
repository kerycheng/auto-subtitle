from collections import Counter
import copy
import pytest

pytest.importorskip("budoux")
pytest.importorskip("sudachipy")
pytest.importorskip("sudachidict_core")

from autosub.domain import Project, Word, word_key, Cue
from autosub.native_segmentation import segment_native
from autosub.timing import apply_timing


def test_complete_native_sentence_survives_vad_pauses():
    words = [Word(100, 400, "お母さんが"), Word(1000, 1400, "家に"), Word(2200, 3000, "来るんですよね")]
    p = Project(duration_ms=5000, words=words, speech=[[100, 400], [1000, 1400], [2200, 3000]])
    assert [c.text for c in segment_native(p, [words])] == ["お母さんが家に来るんですよね"]


def test_native_connective_stays_at_start_of_following_sentence():
    a = [Word(1000, 2000, "東京駅に売店があるよ")]
    b = [Word(2500, 2700, "で"), Word(2700, 4000, "お母さんと妹がさ")]
    p = Project(duration_ms=6000, words=a+b, speech=[[1000, 2000], [2500, 4000]])
    assert [c.text for c in segment_native(p, [a,b])] == [a[0].text, "でお母さんと妹がさ"]


def test_native_character_boundary_is_repaired_before_complete_word():
    a = [Word(100, 500, "昨日は"), Word(600, 700, "プ")]
    b = [Word(700, 1000, "ランク"), Word(1000, 1600, "をした。")]
    p = Project(duration_ms=3000, words=a+b, speech=[[100, 1600]])
    cues = segment_native(p, [a,b])
    assert [c.text for c in cues] == ["昨日は", "プランクをした。"]


def test_long_native_sentence_only_splits_at_protected_boundaries():
    words = [Word(0, 2000, "今日は"), Word(2500, 4500, "東京駅に"), Word(5000, 7000, "行って"), Word(7500, 9000, "遊んだ。")]
    p = Project(duration_ms=10000, words=words, speech=[[0, 9000]])
    cues = segment_native(p, [words])
    assert len(cues) >= 2
    assert "".join(c.text for c in cues) == "".join(w.text for w in words)
    assert all(c.speech_end_ms-c.speech_start_ms <= 7000 for c in cues)
    assert Counter(t for c in cues for t in c.source_tokens) == Counter(map(word_key, words))


def test_manual_lock_splits_available_native_sentence_without_bridging_it():
    words = [Word(100, 700, "最初。"), Word(900, 1600, "手動。"), Word(1900, 2500, "最後。")]
    p = Project(duration_ms=4000, words=words, speech=[[100, 2500]])
    lock = Cue("manual", 880, 1700, "人工修正", 900, 1600, locked=True, source_tokens=[word_key(words[1])])
    p.cues=[lock]
    before=copy.deepcopy(p)
    result = apply_timing(p, segmenter=lambda remaining: segment_native(remaining, [words]))
    assert [c.text for c in result] == ["最初。", "人工修正", "最後。"]
    assert p == before
    assert result[1].start_ms == lock.start_ms and result[1].end_ms == lock.end_ms


def test_vad_expansion_cannot_create_overlap_between_disjoint_native_sentences():
    a=[Word(1000,2000,"前です。")]
    b=[Word(2000,3000,"後です。")]
    p=Project(duration_ms=4000,words=a+b,speech=[[1000,2200],[2500,3000]])
    cues=apply_timing(p,segmenter=lambda remaining:segment_native(remaining,[a,b]))
    assert cues[0].end_ms <= cues[1].start_ms


def test_true_raw_overlap_is_retained_and_flagged():
    a=[Word(1000,2200,"前です。")]
    b=[Word(2000,3000,"後です。")]
    p=Project(duration_ms=4000,words=a+b,speech=[[1000,3000]])
    cues=apply_timing(p,segmenter=lambda remaining:segment_native(remaining,[a,b]))
    assert cues[0].end_ms > cues[1].start_ms
    assert any("時間重疊" in f for c in cues for f in c.flags)


def test_oov_guess_cannot_join_native_sentences_across_a_clear_pause():
    a=[Word(100,900,"バカ")]
    b=[Word(2000,4000,"文房具を思い出した")]
    p=Project(duration_ms=5000,words=a+b,speech=[[100,900],[2000,4000]])
    cues=segment_native(p,[a,b])
    assert [c.text for c in cues] == ["バカ","文房具を思い出した"]


def test_dictionary_partial_right_token_cannot_move_native_sentence_ending():
    a = [Word(100, 400, "よかったよ"), Word(400, 500, "な")]
    b = [Word(500, 1000, "いや"), Word(1000, 2000, "今日は休み")]
    p = Project(duration_ms=3000, words=a+b, speech=[[100, 2000]])
    assert [c.text for c in segment_native(p, [a, b])] == ["よかったよな", "いや今日は休み"]


def test_dictionary_partial_left_token_cannot_move_unrelated_text_with_prefix():
    a = [Word(100, 300, "今日は"), Word(300, 500, "楽しエ")]
    b = [Word(500, 800, "ステ"), Word(800, 1200, "に行く")]
    p = Project(duration_ms=2000, words=a+b, speech=[[100, 1200]])
    assert [c.text for c in segment_native(p, [a, b])] == ["今日は楽しエ", "ステに行く"]
