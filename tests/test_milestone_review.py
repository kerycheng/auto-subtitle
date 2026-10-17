from pathlib import Path
import wave

import numpy as np
import pytest

from autosub.alignment_lab import file_hash
from autosub.domain import Project, Word, Cue, word_key
from autosub.media import load_audio
from autosub.milestone_review import targets,build_review


def project(tmp_path):
    private=tmp_path/'private';private.mkdir()
    source=private/'original.mp4';source.write_bytes(b'unchanged original')
    cache=tmp_path/'cache.wav'
    pcm=np.rint(np.sin(np.arange(48000)*.17)*4000).astype('<i2')
    with wave.open(str(cache),'wb') as f:
        f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(pcm.tobytes())
    word=Word(1200,1600,'テスト')
    p=Project(media_path=str(source),audio_path=str(cache),duration_ms=3000,status='complete',
              asr_strategy='fixed30-v1',speech=[[1200,1600]],words=[word],
              cues=[Cue('a',1180,1800,word.text,1200,1600,source_tokens=[word_key(word)])])
    p.review_events=[{'action':'asr_trace_context','trace_id':'one','input':{'start_ms':1000}},
                     {'action':'asr_trace_checkpoint','trace_id':'one','segments':[
                         {'temperature':.8,'start':.2,'end':.6},
                         {'temperature':None,'start':0,'end':.1},
                         {'temperature':0.,'start':0,'end':.1}]}]
    return p


def test_listening_uses_context_clock_and_keeps_unknown_temperatures_unknown(tmp_path):
    p=project(tmp_path);rows=targets(p)
    assert len(rows)==1 and rows[0]['source_ms']==[1200,1600]
    assert rows[0]['temperature']==.8 and rows[0]['independent_review']=='unreviewed'
    assert 'rejected attempts unknown' in rows[0]['note']


def test_excerpts_preserve_original_samples_and_frozen_baseline(tmp_path):
    p=project(tmp_path);path=tmp_path/'baseline.json';p.save(path)
    before=file_hash(path);original=file_hash(p.media_path)
    output=tmp_path/'review';r=build_review(path,output)
    assert file_hash(path)==before and file_hash(p.media_path)==original
    assert r['human_completion_rate'] is None and len(r['targets'])==1
    row=r['targets'][0]
    assert np.array_equal(load_audio(output/row['audio_file']),load_audio(Path(p.audio_path)))
    with pytest.raises(ValueError,match='new review directory'):build_review(path,output)


def test_missing_decoder_context_and_unfinished_project_are_rejected(tmp_path):
    p=project(tmp_path);p.review_events.pop(0)
    with pytest.raises(ValueError,match='Missing decoder'):targets(p)
    p.status='transcribing';path=tmp_path/'partial.json';p.save(path)
    with pytest.raises(ValueError,match='complete baseline'):build_review(path,tmp_path/'review')
