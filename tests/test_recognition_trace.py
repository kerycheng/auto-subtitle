import copy
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest

from autosub.domain import Project
from autosub.media import Cancelled
from autosub.pipeline import run_pipeline
from autosub.recognition_trace import RecognitionTrace, digest, words_digest
from test_native_pipeline import setup


def events(project, action):
    return [e for e in project.review_events if e['action'] == action]


def test_native_trace_checkpoints_follow_whole_batches_and_resume(tmp_path, monkeypatch):
    p, path, calls = setup(tmp_path, monkeypatch)
    cancel = Event()
    def stop(_, message):
        if '已保存至 26.0' in message:
            cancel.set()
    with pytest.raises(Cancelled):
        run_pipeline(p, path, cancel, stop)
    saved = Project.load(path)
    prefix = copy.deepcopy(saved.review_events)
    checkpoints = events(saved, 'asr_trace_checkpoint')
    assert len(checkpoints) == 1
    assert checkpoints[0]['accepted_words_sha256'] == words_digest(saved.words)
    assert checkpoints[0]['checkpoint']['resume_to_ms'] == 26000
    assert len(checkpoints[0]['segments']) == 2
    assert all(s['temperature'] is None for s in checkpoints[0]['segments'])
    run_pipeline(saved, path, Event(), lambda *_: None)
    result = Project.load(path)
    assert result.review_events[:len(prefix)] == prefix
    assert len(events(result, 'asr_trace_context')) == 2
    assert len(events(result, 'asr_trace_checkpoint')) == 2
    assert events(result, 'asr_trace_context')[1]['requested_options']['clip_timestamps'] == [26.,66.]
    final = events(result, 'asr_trace_checkpoint')[-1]
    assert final['accepted_words_sha256'] == words_digest(result.words[2:])
    run_pipeline(result, path, Event(), lambda *_: None)
    assert result.review_events == Project.load(path).review_events
    assert len(calls) == 2


def test_native_error_discards_partial_trace_and_words(tmp_path, monkeypatch):
    p, path, _ = setup(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='simulated'):
        run_pipeline(p, path, Event(), lambda *_: None)
    result = Project.load(path)
    assert len(events(result, 'asr_trace_checkpoint')) == 1
    assert len(events(result, 'asr_trace_checkpoint')[0]['segments']) == 2
    assert len(result.words) == 2


@pytest.mark.parametrize('failure', ['cancel', 'error'])
@pytest.mark.parametrize('language', ['en', 'mixed'])
def test_chunk_partial_trace_never_persisted(tmp_path, monkeypatch, failure, language):
    import autosub.pipeline as pipeline
    import faster_whisper
    source = tmp_path/'input.wav'
    source.write_bytes(b'fixture')
    audio = np.zeros(61*16000, dtype=np.float32)
    monkeypatch.setattr(pipeline, 'decode_audio', lambda *_: None)
    monkeypatch.setattr(pipeline, 'load_audio', lambda _: audio)
    monkeypatch.setattr(pipeline, 'check_runtime', lambda _: {'compute_type':'int8'})
    cancel = Event()
    class Model:
        calls = 0
        def __init__(self,*a,**k): pass
        def transcribe(self, samples, **kwargs):
            Model.calls += 1
            current = Model.calls
            def output():
                w=NS(start=1.,end=2.,word=' words',probability=.9)
                yield NS(words=[w],text=w.word,temperature=.4)
                if current == (4 if language == 'mixed' else 2):
                    if failure == 'cancel': cancel.set()
                    else: raise RuntimeError('interrupted')
            return output(), None
    monkeypatch.setattr(faster_whisper, 'WhisperModel', Model)
    p=Project(media_path=str(source),language=language,speech=[[0,61000]])
    path=tmp_path/'project.json'
    with pytest.raises(Cancelled if failure=='cancel' else RuntimeError):
        run_pipeline(p,path,cancel,lambda *_:None)
    saved=Project.load(path)
    assert saved.completed_chunks == [0]
    assert len(events(saved,'asr_trace_context')) == len(events(saved,'asr_trace_checkpoint')) == (2 if language == 'mixed' else 1)
    first=events(saved,'asr_trace_checkpoint')[0]
    assert first['segments'][0]['temperature']==.4
    assert first['accepted_words_sha256']==words_digest(saved.words[:1])
    prefix=copy.deepcopy(saved.review_events)
    run_pipeline(saved,path,Event(),lambda *_:None)
    assert saved.review_events[:len(prefix)]==prefix
    assert saved.completed_chunks==[0,1,2]


def test_actual_observations_are_distinct_from_requests_and_missing_data():
    @dataclass
    class Options:
        temperatures: tuple = (0., .2, .4)
        initial_prompt: str = 'private glossary'
    info=NS(transcription_options=Options(),language='ja',language_probability=.91)
    audio=np.array([.1,.2],dtype=np.float32)
    before=audio.copy()
    options={'beam_size':5,'initial_prompt':'private glossary','vad_filter':False}
    trace=RecognitionTrace(Project(),NS(model=NS(device='cpu',compute_type='int8')),
                           'default',audio,options,info,2000,2000.125)
    context=trace.context
    assert context['compute_type_requested']=='default'
    assert context['compute_type_observed']=='int8'
    assert context['resolved_options']['temperatures']==[0.,.2,.4]
    assert context['requested_options']['initial_prompt']['sha256']==digest('private glossary')
    assert options['initial_prompt']=='private glossary'
    assert 'private glossary' not in str(context)
    assert context['rejected_fallback_attempts'] is None
    assert context['model_weights_sha256'] is None
    segment=trace.segment(NS(words=[],temperature=.4,avg_logprob=float('nan')))
    assert segment['temperature']==.4 and segment['avg_logprob'] is None
    assert segment['no_speech_prob'] is None and segment['text_sha256'] is None
    assert np.array_equal(audio,before)


def test_old_completed_project_does_not_invent_trace(tmp_path,monkeypatch):
    p,path,_=setup(tmp_path,monkeypatch)
    run_pipeline(p,path,Event(),lambda *_:None)
    p.review_events=[]
    p.save(path)
    run_pipeline(p,path,Event(),lambda *_:None)
    assert not Project.load(path).review_events
