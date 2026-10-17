import copy
import pytest
from autosub.domain import Project,Word
from autosub.input_context_audit import native_input_ranges,point_coverage,clip_edge_targets


def project():
    p=Project(duration_ms=20000,words=[Word(1000,2000,'a'),Word(14002,15000,'b')])
    p.review_events=[{'action':'asr_trace_context','strategy':'native-v1',
        'requested_options':{'vad_filter':False,'clip_timestamps':[1.,3.,14.,16.5]},'input':{'start_ms':0}}]
    return p


def test_saved_input_can_exclude_confirmed_earlier_speech_without_moving_any_word():
    p=project();before=copy.deepcopy(p)
    result=point_coverage(p,12000)
    assert result['requested_input_covered'] is False
    assert result['preceding_clip_end_ms']==3000 and result['following_clip_start_ms']==14000
    assert point_coverage(p,14002)['requested_input_covered'] is True
    edges=clip_edge_targets(p)['targets']
    assert edges[1]['distance_from_input_start_ms']==2
    assert edges[1]['confirmed_missing_text'] is False and p==before


def test_missing_metadata_is_unknown_and_not_fabricated_from_current_vad():
    p=project();p.review_events=[];p.speech=[[11000,16000]]
    assert native_input_ranges(p) is None and point_coverage(p,12000)['requested_input_covered'] is None
    assert clip_edge_targets(p)['status']=='unknown-input-metadata'


def test_resumed_overlapping_context_ranges_merge_without_inventing_gaps():
    p=project();p.review_events.append({'action':'asr_trace_context','strategy':'native-v1',
        'requested_options':{'vad_filter':False,'clip_timestamps':[0.,2.]},'input':{'start_ms':15000}})
    assert native_input_ranges(p)==[[1000,3000],[14000,17000]]


@pytest.mark.parametrize('bad',[float('nan'),-1,20001])
def test_invalid_source_point_rejected(bad):
    with pytest.raises(ValueError):point_coverage(project(),bad)
