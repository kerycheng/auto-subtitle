import copy

import pytest

from autosub.alignment_lab import propose_word_alignment
from autosub.domain import Word, word_key


def test_complete_character_evidence_preserves_words_spelling_and_ownership():
    words = [Word(1200, 1500, 'ab, ', .8), Word(1600, 1900, 'cd!', .7)]
    units = [{'text': char, 'start': a, 'end': a + .1}
             for char, a in zip('abcd', [.3, .5, .8, 1.0])]
    before = copy.deepcopy((words, units))
    result = propose_word_alignment(words, units, [1000, 3000])
    assert result['span_ms'] == [1300, 2100]
    assert [p['proposed_ms'] for p in result['words']] == [[1300, 1600], [1800, 2100]]
    assert result['source_text'] == 'ab, cd!'
    assert result['source_tokens'] == list(map(word_key, words))
    assert [p['probability'] for p in result['words']] == [.8, .7]
    assert result['production_adoption'] is False
    assert (words, units) == before


@pytest.mark.parametrize('fault', ['missing', 'zero', 'overlap', 'reverse', 'outside',
                                  'nonfinite', 'bool', 'text', 'submillisecond'])
def test_unsafe_unit_evidence_is_rejected_without_filling_or_clamping(fault):
    words = [Word(100, 200, 'ab', .9)]
    units = [{'text': 'a', 'start': .1, 'end': .2},
             {'text': 'b', 'start': .3, 'end': .4}]
    if fault == 'missing': units[0]['start'] = None
    elif fault == 'zero': units[0]['end'] = .1
    elif fault == 'overlap': units[1]['start'] = .15
    elif fault == 'reverse': units[0]['end'] = .05
    elif fault == 'outside': units[1]['end'] = 2
    elif fault == 'nonfinite': units[0]['start'] = float('nan')
    elif fault == 'bool': units[0]['start'] = True
    elif fault == 'text': units[1]['text'] = 'x'
    else: units[0].update(start=.10001, end=.10002)
    before = copy.deepcopy(units)
    with pytest.raises(ValueError):
        propose_word_alignment(words, units, [0, 1000])
    # NaN does not compare equal, so compare its representation as well.
    assert repr(units) == repr(before)


def test_timed_unit_crossing_an_original_word_boundary_is_not_split():
    words = [Word(100, 200, 'a', .9), Word(200, 300, 'b', .8)]
    with pytest.raises(ValueError):
        propose_word_alignment(words, [{'text': 'ab', 'start': .1, 'end': .3}], [0, 1000])


def test_backend_repairs_are_not_hidden_by_valid_looking_final_times():
    with pytest.raises(ValueError):
        propose_word_alignment([Word(100, 200, 'a', .9)],
                               [{'text': 'a', 'start': .1, 'end': .2}], [0, 1000],
                               timestamp_corrections=[{'changed_indices': [1]}])


def test_duplicate_original_tokens_keep_separate_indices_and_all_occurrences():
    words = [Word(100, 200, 'a', .9), Word(100, 200, 'a', .9)]
    result = propose_word_alignment(words, [{'text': 'a', 'start': .1, 'end': .2},
                                            {'text': 'a', 'start': .3, 'end': .4}], [0, 1000])
    assert result['source_tokens'] == [word_key(words[0])] * 2
    assert [p['original_index'] for p in result['words']] == [0, 1]
