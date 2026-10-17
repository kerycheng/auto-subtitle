import copy
import json
import runpy
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from autosub.alignment_lab import file_hash
from autosub.media import load_audio
from autosub.domain import Project


def write_json(path,data): path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')


def wav(path,samples):
    with wave.open(str(path),'wb') as target:
        target.setparams((1,2,16000,0,'NONE','not compressed'));target.writeframes(samples.astype('<i2').tobytes())


def invoke(monkeypatch,name,*arguments):
    monkeypatch.setattr(sys,'argv',[name,*map(str,arguments)])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/name),run_name='__main__')


def fixture(tmp_path,monkeypatch):
    import faster_whisper.vad
    monkeypatch.setattr(faster_whisper.vad,'get_speech_timestamps',lambda audio,*a,**k:[{'start':0,'end':len(audio)}])
    samples=[];windows=[];chunks=[];offset=0
    for language in ('en','zh','ja','ko'):
        for position,count in ((8,1600),(9,3200)):
            name=f'{language}-{position}';path=tmp_path/(name+'.wav')
            pcm=np.resize(np.array([10000,-5000,2000],dtype='<i2'),count);wav(path,pcm)
            samples.append({'id':name,'language':language,'audio_path':str(path),'audio_sha256':file_hash(path),
                'reference_text':f'原語 {language} {position}'})
            windows.append({'id':name,'language':language,'window_ms':[offset,offset+count//16],
                'owner_ms':[offset,offset+count//16],'source_audio_sha256':file_hash(path),'padding_samples':0})
            offset+=count//16;chunks.append(pcm)
    dataset=tmp_path/'dataset.json';write_json(dataset,{'samples':samples})
    audio=tmp_path/'all.wav';wav(audio,np.concatenate(chunks))
    manifest=tmp_path/'manifest.json';write_json(manifest,{'text_provided':False,'audio_path':str(audio),
        'audio_sha256':file_hash(audio),'duration_ms':offset,'source_manifest_sha256':file_hash(dataset),'windows':windows})
    return dataset,manifest


def test_fixed_pairs_keep_complete_source_frames_text_separation_and_sources(tmp_path,monkeypatch):
    dataset,manifest=fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    output=tmp_path/'prepared'
    invoke(monkeypatch,'prepare_level_pairs.py',dataset,manifest,'--output',output)
    cases=json.loads((output/'cases.json').read_text())
    assert len(cases['cases'])==16 and cases['text_provided'] is False
    assert '原語' not in (output/'cases.json').read_text() and 'reference_text' not in (output/'cases.json').read_text()
    for row in cases['cases']:
        audio=load_audio(Path(row['audio_path']))
        np.testing.assert_array_equal(audio[:1600],load_audio(tmp_path/(row['language']+'-8.wav')))
        gain=10**((-18 if 'db18' in row['id'] else -30)/20)
        expected=np.rint(load_audio(tmp_path/(row['language']+'-9.wav'))*32768*gain).astype('<i2').astype(np.float32)/32768
        np.testing.assert_array_equal(audio[-3200:],expected)
        assert len(audio)==4800+(12800 if 'gap800' in row['id'] else 0)
    assert all(p.read_bytes()==b for p,b in before.items())


@pytest.mark.parametrize('changed',['source','combined'])
def test_stale_sources_fail_before_creating_pair_output(tmp_path,monkeypatch,changed):
    dataset,manifest=fixture(tmp_path,monkeypatch)
    (tmp_path/('en-8.wav' if changed=='source' else 'all.wav')).write_bytes(b'changed')
    output=tmp_path/'prepared'
    with pytest.raises((ValueError,wave.Error,EOFError)):
        invoke(monkeypatch,'prepare_level_pairs.py',dataset,manifest,'--output',output)
    assert not output.exists()


@pytest.mark.parametrize('mode',['production','full-window'])
def test_ablation_changes_only_research_call_options_and_restores_factory(tmp_path,monkeypatch,mode):
    import faster_whisper
    import autosub.pipeline
    dataset,manifest=fixture(tmp_path,monkeypatch)
    inputs=tmp_path/'prepared'
    invoke(monkeypatch,'prepare_level_pairs.py',dataset,manifest,'--output',inputs)
    path=inputs/'cases.json';data=json.loads(path.read_text());data['cases']=data['cases'][:1];write_json(path,data)
    model=tmp_path/'model';model.mkdir();(model/'model.bin').write_bytes(b'fake weights')
    observed=[]
    class FakeModel:
        frames_per_second=100
        def __init__(self,*a,**kw):pass
        def transcribe(self,audio,**kw):
            observed.append((audio.copy(),copy.deepcopy(kw)));return iter([]),None
    monkeypatch.setattr(faster_whisper,'WhisperModel',FakeModel)
    def pipeline(p,path,*a):
        audio=load_audio(Path(p.media_path));model=faster_whisper.WhisperModel()
        model.transcribe(audio,language=p.language,initial_prompt=None,vad_filter=True,
            vad_parameters={'threshold':.5},clip_timestamps=[.01,.05])
        p.audio_path=p.media_path;p.duration_ms=round(len(audio)/16);p.status='complete';p.save(path)
    monkeypatch.setattr(autosub.pipeline,'run_pipeline',pipeline)
    invoke(monkeypatch,'benchmark_level_pairs.py',path,'--mode',mode,'--model',model,'--output',tmp_path/'run')
    assert faster_whisper.WhisperModel is FakeModel
    raw,options=observed[0]
    assert options['initial_prompt'] is None and options['language']=='en'
    assert options['vad_filter']==(mode=='production')
    assert options['clip_timestamps']==([0,len(raw)/16000] if mode=='full-window' else [.01,.05])
    np.testing.assert_array_equal(raw,load_audio(Path(data['cases'][0]['audio_path'])))


@pytest.mark.parametrize('changed',['reference','project','incomplete'])
def test_report_refuses_changed_labels_projects_or_incomplete_run(tmp_path,monkeypatch,changed):
    dataset,manifest=fixture(tmp_path,monkeypatch)
    inputs=tmp_path/'prepared'
    invoke(monkeypatch,'prepare_level_pairs.py',dataset,manifest,'--output',inputs)
    cases=json.loads((inputs/'cases.json').read_text())
    run=tmp_path/'run';run.mkdir();rows=[]
    for case in cases['cases']:
        folder=run/case['id'];folder.mkdir();path=folder/'generated.autosub.json'
        Project(media_path=case['audio_path'],duration_ms=case['duration_ms'],language=case['language'],status='complete').save(path)
        rows.append({'id':case['id'],'project_sha256':file_hash(path)})
    report={'status':'running' if changed=='incomplete' else 'complete',
        'cases_sha256':file_hash(inputs/'cases.json'),'model_identity':{},'mode':'production','cases':rows}
    write_json(run/'report.json',report)
    if changed=='reference':
        path=inputs/'labels.json';d=json.loads(path.read_text(encoding='utf-8'))
        d['cases'][0]['regions'][0]['reference_text']='changed reference';write_json(path,d)
    elif changed=='project':
        (run/rows[0]['id']/'generated.autosub.json').write_text('{}')
    output=tmp_path/'comparison'
    with pytest.raises(ValueError):
        invoke(monkeypatch,'report_level_pairs.py',inputs,run,'--output',output)
    assert not output.exists()
