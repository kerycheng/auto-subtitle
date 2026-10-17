import copy

import numpy as np
import pytest

from autosub.level_compensation_lab import compensate,union_regions


def test_loud_neighbor_cannot_clip_or_change_and_source_clock_is_preserved():
    source=np.concatenate((np.full(16000,.9,dtype=np.float32),np.full(16000,.002,dtype=np.float32),
                           np.zeros(16000,dtype=np.float32)))
    before=source.copy();processed,trace=compensate(source)
    np.testing.assert_array_equal(source,before)
    np.testing.assert_array_equal(processed[:16000],source[:16000])
    np.testing.assert_array_equal(processed[32000:],source[32000:])
    assert processed.shape==source.shape and processed.dtype==np.float32
    assert not np.shares_memory(source,processed)
    assert np.max(np.abs(processed))==pytest.approx(.9)
    assert processed[24000]>.002 and trace['time_shift_samples']==0
    assert all(1<=b['gain']<=10**(24/20) for b in trace['blocks'])


def test_short_final_block_and_single_sample_are_not_dropped():
    for count in (0,1,16001):
        source=np.full(count,.05,dtype=np.float32)
        output,trace=compensate(source)
        assert len(output)==count==trace['source_frames']
        assert not len(output) or np.max(np.abs(output))<=.500001


@pytest.mark.parametrize('source',[np.array([np.nan],dtype=np.float32),np.array([2],dtype=np.float32),
    np.zeros((2,2),dtype=np.float32),np.array([.02],dtype=np.float64)],ids=['nonfinite','range','channels','dtype'])
def test_invalid_audio_is_rejected(source):
    with pytest.raises(ValueError): compensate(source)


def test_region_union_never_removes_raw_spans_or_mutates_inputs():
    raw=[{'start':20,'end':40},{'start':100,'end':120}]
    boosted=[{'start':0,'end':30},{'start':60,'end':70}]
    before=copy.deepcopy((raw,boosted))
    assert union_regions(raw,boosted)==[{'start':0,'end':40},{'start':60,'end':70},{'start':100,'end':120}]
    assert (raw,boosted)==before
