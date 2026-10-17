import numpy as np
import pytest

from autosub.speech_detection import detect_speech, external_vad_clips, quiet_retry_applied
from autosub.domain import Project


def test_quiet_retry_uses_bounded_copy_and_exact_original_clock():
    audio=np.array([.001,-.002,.001],dtype=np.float32)
    before=audio.copy(); calls=[]; options=object()
    def detector(samples,opts,**kwargs):
        assert opts is options and kwargs=={'sampling_rate':16000}
        calls.append(samples)
        return [] if len(calls)==1 else [{'start':0,'end':3}]
    stamps,trace=detect_speech(audio,options,detector)
    assert stamps==[{'start':0,'end':3}] and np.array_equal(audio,before)
    assert calls[0] is audio and not np.shares_memory(calls[1],audio)
    assert calls[1].shape==audio.shape and calls[1].dtype==audio.dtype
    assert trace['analysis_gain_db']==pytest.approx(24)
    assert trace['asr_audio_changed'] is False


@pytest.mark.parametrize('case',['speech','normal-empty','silence','disabled','empty'])
def test_retry_does_not_change_existing_speech_or_run_without_conditions(case):
    audio=np.array([.001],dtype=np.float32)
    if case=='normal-empty': audio[:]=.2
    if case=='silence': audio[:]=0
    if case=='empty': audio=audio[:0]
    calls=[]; expected=[{'start':0,'end':1}] if case=='speech' else []
    def detector(samples,*args,**kwargs):
        calls.append(samples)
        assert len(calls)==1
        return expected
    stamps,trace=detect_speech(audio,None,detector,retry_quiet=case!='disabled')
    assert stamps is expected and trace is None


def test_retry_cannot_turn_silence_into_automatic_subtitles():
    audio=np.full(160,.01,dtype=np.float32)
    stamps,trace=detect_speech(audio,None,lambda *a,**k:[])
    assert not stamps and trace['retry_regions']==0
    assert trace['analysis_gain']<=10**(24/20)


def test_peak_target_does_not_clip_analysis():
    observed=[]
    def detector(samples,*a,**k):
        observed.append(float(np.max(np.abs(samples))))
        return []
    detect_speech(np.array([-.09,.03],dtype=np.float32),None,detector)
    assert observed[1]==pytest.approx(.5)


def test_external_clips_merge_padding_and_keep_window_relative_clock():
    assert external_vad_clips([[100,200],[300,500],[1500,1800],[4000,4500]],1000,3000)==[.3,1.0]
    assert external_vad_clips([[100,300],[500,800]],0,1000)==[0,1.0]
    assert external_vad_clips([[4000,5000]],0,1000)==[]


def test_retry_audit_persists_without_new_project_schema(tmp_path):
    p=Project(review_events=[{'action':'vad_quiet_retry','policy':'quiet-empty-v1','retry_regions':1}])
    path=tmp_path/'project.json';p.save(path)
    assert quiet_retry_applied(Project.load(path))
    p.review_events[0]['retry_regions']=0
    assert not quiet_retry_applied(p)
