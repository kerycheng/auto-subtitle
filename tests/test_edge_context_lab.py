import pytest
from autosub.edge_context_lab import edge_window,prefix_word_candidates


def test_synthetic_edge_uses_bounded_gap_and_retains_source_frames():
    result=edge_window(7000,14000,16500,50000)
    assert result['source_window_ms']==[9000,16500]
    assert result['input_frames']==[144000,264000]
    assert result['original_relative_clips']==[5.,7.5]
    assert result['expanded_relative_clips']==[0.,7.5]
    assert result['automatic_adoption'] is False


def test_context_does_not_enter_preceding_owned_audio_and_caps_long_right_side():
    result=edge_window(20000,24000,40000,50000)
    assert result['source_window_ms']==[20000,32000]
    assert result['original_relative_clips']==[4.,12.]


@pytest.mark.parametrize('values',[(0,0,2000,3000),(1000,900,2000,3000),(0,1000,4000,3000),
                                    (0,True,2000,3000),(0,float('nan'),2000,3000)])
def test_invalid_or_fractional_clocks_are_rejected(values):
    with pytest.raises(ValueError):edge_window(*values)


def test_generated_prefix_candidates_do_not_auto_delete_or_claim_hallucinations():
    rows=prefix_word_candidates([{'words':[{'start':1.,'end':2.,'word':'a'},
        {'start':4.9,'end':5.2,'word':'b'},{'start':5.2,'end':6.,'word':'c'}]}],0,5000)
    assert [r['relation'] for r in rows]==['wholly-before-source-speech','straddles-source-speech-start']
    assert all(r['independent_review']=='unreviewed' for r in rows)
