import copy

import pytest

from autosub.gap_context_lab import observed_clip_edges


def test_seek_ownership_rejects_first_outside_word_instead_of_selecting_later():
    clips = [[0, 1000], [2005, 4000]]
    segments = [{'seek': 200, 'words': [{'start': 2, 'end': 2},
                                       {'start': 2, 'end': 2.1},
                                       {'start': 2.15, 'end': 2.3}]},
                {'seek': 200, 'words': [{'start': 2.4, 'end': 2.5}]}]
    before = copy.deepcopy((clips, segments))
    result = observed_clip_edges(clips, segments, 5000, frames_per_second=100)
    assert result['effective_clip_frames'] == [[0, 100], [200, 400]]
    assert result['edge_words'] == {}
    assert result['excluded'][0]['clip_index'] == 1
    assert result['excluded'][0]['first_word_ms'] == [2000, 2100]
    assert (clips, segments) == before


def test_first_positive_word_can_follow_an_empty_or_zero_duration_segment():
    segments = [{'seek': 0, 'words': []},
                {'seek': 0, 'words': [{'start': .1, 'end': .1}]},
                {'seek': 0, 'words': [{'start': .2, 'end': .3}]},
                {'seek': 200, 'words': [{'start': 2, 'end': 2.2}]}]
    result = observed_clip_edges([[0, 1000], [2000, 4000]], segments, 5000,
                                 frames_per_second=100)
    assert result['edge_words'] == {0: [200, 300], 1: [2000, 2200]}


@pytest.mark.parametrize('fault', ['unknown-seek', 'decreasing-seek', 'bad-rate',
                                  'collapsed', 'nonfinite-word', 'reverse-word',
                                  'bool-rate'])
def test_ambiguous_clock_evidence_is_rejected(fault):
    clips = [[0, 1000], [2000, 4000]]
    segments = [{'seek': 200, 'words': [{'start': 2, 'end': 2.1}]}]
    rate = 100
    if fault == 'unknown-seek': segments[0]['seek'] = 150
    elif fault == 'decreasing-seek': segments.append({'seek': 0, 'words': []})
    elif fault == 'bad-rate': rate = 0
    elif fault == 'collapsed': clips[0] = [0, 1]
    elif fault == 'nonfinite-word': segments[0]['words'][0]['start'] = float('nan')
    elif fault == 'reverse-word': segments[0]['words'][0]['end'] = 1.9
    else: rate = True
    with pytest.raises(ValueError):
        observed_clip_edges(clips, segments, 5000, frames_per_second=rate)
