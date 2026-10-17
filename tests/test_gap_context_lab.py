import copy
import pytest
from autosub.gap_context_lab import propose_internal_context
from autosub.gap_context_lab import propose_short_internal_context


def test_supported_internal_head_changes_without_merging_or_moving_old_ends():
    clips = [[1000, 3000], [10000, 14000], [18000, 20000]]
    words = {0: [1000, 1400], 1: [10002, 10500], 2: [18500, 19000]}
    support = [[500, 1500], [6500, 12000], [16000, 19500]]
    before = copy.deepcopy((clips, words, support))
    result = propose_internal_context(clips, words, support, 22000)
    assert result['candidate_clips_ms'] == [[1000, 3000], [6500, 14000], [18000, 20000]]
    assert result['changed_clip_indices'] == [1]
    assert (clips, words, support) == before
    assert result['production_adoption'] is False and result['confirmed_missing_text'] is False


def test_context_cannot_take_previous_owned_audio_and_is_bounded():
    result = propose_internal_context([[100, 6000], [7000, 12000]], {1: [7000, 7500]}, [[0, 9000]], 13000)
    assert result['candidate_clips_ms'] == [[100, 6000], [6001, 12000]]
    result = propose_internal_context([[100, 1000], [10000, 12000]], {1: [10000, 10500]}, [[0, 11000]], 13000)
    assert result['candidate_clips_ms'][1][0] == 5000


def test_an_independent_span_must_reach_the_entire_observed_word():
    result = propose_internal_context([[100, 1000], [4000, 6000]], {1: [4000, 4500]}, [[2000, 4200]], 7000)
    assert not result['changed_clip_indices']


def test_no_word_evidence_and_leading_or_trailing_noise_do_not_create_new_clips():
    result = propose_internal_context([[1000, 3000], [7000, 9000]], {}, [[0, 10000]], 10000)
    assert result['candidate_clips_ms'] == result['original_clips_ms']


@pytest.mark.parametrize('bad', ['overlap', 'nan', 'bool', 'unknown-word', 'outside-word', 'negative-limit'])
def test_invalid_or_ambiguous_clocks_are_rejected(bad):
    clips = [[100, 1000], [4000, 6000]]; words = {1: [4000, 4500]}; support = [[2000, 5000]]
    kwargs = {}
    if bad == 'overlap': support = [[0, 3000], [2000, 5000]]
    elif bad == 'nan': clips[1][0] = float('nan')
    elif bad == 'bool': clips[0][0] = True
    elif bad == 'unknown-word': words = {7: [4000, 4500]}
    elif bad == 'outside-word': words = {1: [3900, 4500]}
    else: kwargs['max_extension_ms'] = -1
    with pytest.raises(ValueError): propose_internal_context(clips, words, support, 7000, **kwargs)

def test_minimal_synthetic_word_is_the_most_permissive_edge_probe():
    clips = [[100, 1000], [4000, 6000]]
    for support_end in [4000, 4001, 4030, 4200, 5000]:
        support = [[2000, support_end]]
        minimal = propose_internal_context(clips, {1: [4000, 4001]}, support, 7000)
        for offset in [0, 20, 50]:
            for duration in [1, 10, 200]:
                observed = propose_internal_context(clips, {1: [4000+offset, 4000+offset+duration]}, support, 7000)
                if not minimal['changed_clip_indices']:
                    assert not observed['changed_clip_indices']


def test_short_cap_checks_entire_proposal_and_preserves_unrelated_groups():
    clips = [[0, 1000], [6000, 11000], [17000, 22000]]
    words = {1: [6000, 6100], 2: [17000, 17100]}
    support = [[3000, 8000], [14000, 19000]]
    before = copy.deepcopy((clips, words, support))
    # Original clips fit, but both extensions would exceed this cap.
    result = propose_short_internal_context(clips, words, support, 25000, max_clip_ms=7000)
    assert result['candidate_clips_ms'] == clips
    assert not result['changed_clip_indices']
    assert result['decisions'][1]['reason'] == 'proposed-clip-exceeds-limit'
    assert (clips, words, support) == before
    assert result['production_adoption'] is False


def test_exact_cap_is_allowed_and_long_original_is_never_shortened():
    result = propose_short_internal_context(
        [[0, 1000], [6000, 11000], [17000, 30000]],
        {1: [6000, 6100], 2: [17000, 17100]},
        [[3000, 8000], [14000, 19000]], 31000, max_clip_ms=8000)
    assert result['candidate_clips_ms'] == [[0, 1000], [3000, 11000], [17000, 30000]]
    assert result['changed_clip_indices'] == [1]
    assert result['decisions'][2]['reason'] == 'proposed-clip-exceeds-limit'


@pytest.mark.parametrize('limit', [0, -1, True, float('nan'), float('inf'), 12.5, '30'])
def test_short_cap_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        propose_short_internal_context([[0, 1000]], {}, [], 2000, max_clip_ms=limit)
