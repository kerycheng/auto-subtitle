import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from autosub.alignment_lab import file_hash
from autosub.compensation_evaluation import evaluate_study,noise_candidates,source_matches
from autosub.domain import Project,Word,Cue,word_key
from autosub.export import export_subtitles
from autosub.level_compensation_lab import compensate
from autosub.media import load_audio


def test_source_matches_are_text_evidence_not_a_word_error_rate():
    result=source_matches([{'sample_id':'first','reference_text':'alpha'},
        {'sample_id':'second','reference_text':'beta'}],'alpha bet')
    assert [r['exact_matched_characters'] for r in result['regions']]==[5,3]
    assert 'not a source-specific error rate' in result['scope']


def test_noise_candidates_keep_straddling_and_fully_after_distinct():
    p=Project(words=[Word(0,80,'before'),Word(80,120,'tail'),Word(120,160,'extra')])
    result=noise_candidates(p,[1600,3200])
    assert [r['relation'] for r in result]==['straddles-generated-nonspeech','starts-in-generated-nonspeech']
    assert all(r['independently_reviewed'] is False for r in result)


def save_json(path,data):path.write_text(json.dumps(data),encoding='utf-8')


def wav(path,samples):
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(samples.astype('<i2').tobytes())


def study(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    first,second=tmp_path/'first.wav',tmp_path/'second.wav'
    a=np.full(1600,6500,dtype='<i2');b=np.full(3200,650,dtype='<i2')
    wav(first,a);wav(second,b)
    source=tmp_path/'whole.wav';wav(source,np.concatenate((a,b)));raw=load_audio(source)
    dataset=tmp_path/'dataset.json';save_json(dataset,{'samples':[
        {'id':'first','language':'en','reference_text':'alpha'},
        {'id':'second','language':'en','reference_text':'beta'}]})
    plan=tmp_path/'plan.json';save_json(plan,{'fixed':'test'})
    protocol={'plan_sha256':file_hash(plan),'source_dataset_path':str(dataset),'dataset_sha256':file_hash(dataset)}
    case={'id':'pair','language':'en','kind':'speech-pair','audio_path':str(source),'audio_sha256':file_hash(source),'duration_ms':300}
    regions=[{'sample_id':'first','source_path':str(first),'source_sha256':file_hash(first),
        'sample_frames':[0,1600],'reference_text':'alpha','attenuation_db':0},
        {'sample_id':'second','source_path':str(second),'source_sha256':file_hash(second),
        'sample_frames':[1600,4800],'reference_text':'beta','attenuation_db':0}]
    save_json(inputs/'cases.json',{'text_provided':False,'protocol':protocol,'cases':[case]})
    save_json(inputs/'labels.json',{'protocol':protocol,'cases':[{**case,'regions':regions}]})
    dirs=[];processed,trace=compensate(raw)
    def sample_hash(a):return hashlib.sha256(a.tobytes()).hexdigest()
    for mode in ('raw','vad-gain','both-gain'):
        folder=tmp_path/mode;dest=folder/'pair';dest.mkdir(parents=True);dirs.append(folder)
        words=[Word(0,100,'alpha'),Word(100,300,'beta')]
        cues=[Cue(str(i),w.start_ms,w.end_ms,w.text,w.start_ms,w.end_ms,source_tokens=[word_key(w)]) for i,w in enumerate(words)]
        p=Project(media_path=str(source),audio_path=str(source),duration_ms=300,language='en',
            asr_strategy='fixed30-v1',status='complete',words=words,cues=cues)
        path=dest/'generated.autosub.json';p.save(path)
        for suffix in ('.ass','.srt'):export_subtitles(p,dest/('generated'+suffix))
        if mode!='raw':np.save(dest/'processed-asr-or-analysis.npy',processed)
        row={'id':'pair','source_sha256':case['audio_sha256'],'original_samples_sha256':sample_hash(raw),
            'project_path':str(path),'project_sha256':file_hash(path),'reused':False,'gain_trace':trace,
            'processed_samples_sha256':sample_hash(processed),'decoder_segments':[{'temperature':0.0}],
            'decoder_calls':[{'options':{'initial_prompt':None},'asr_samples_sha256':sample_hash(processed if mode=='both-gain' else raw),'samples':len(raw)}]}
        save_json(folder/'report.json',{'mode':mode,'status':'complete','cases_sha256':file_hash(inputs/'cases.json'),
            'plan_sha256':file_hash(plan),'model_identity':{},'reference_text_provided':False,'cases':[row]})
    return inputs,dirs,plan


def test_validated_study_is_read_only_and_retains_all_strategies(tmp_path):
    inputs,dirs,plan=study(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=evaluate_study(inputs,dirs,plan)
    assert len(result['runs'])==3
    assert all(r['summary']['en/original-level']['character_edits']==0 for r in result['runs'])
    assert 'research-only' in result['production_decision']
    assert all(p.read_bytes()==b for p,b in before.items())


@pytest.mark.parametrize('changed',['source','label','processed','decoder','incomplete','duplicate','missing-mode'])
def test_report_rejects_changed_or_incomplete_evidence(tmp_path,changed):
    inputs,dirs,plan=study(tmp_path)
    if changed=='source':(tmp_path/'whole.wav').write_bytes(b'changed')
    elif changed=='label':
        path=inputs/'labels.json';d=json.loads(path.read_text());d['cases'][0]['regions'][1]['reference_text']='wrong';save_json(path,d)
    elif changed=='processed':np.save(dirs[-1]/'pair/processed-asr-or-analysis.npy',np.zeros(4800,dtype=np.float32))
    elif changed=='missing-mode':dirs=dirs[:-1]
    else:
        path=dirs[-1]/'report.json';d=json.loads(path.read_text())
        if changed=='decoder':d['cases'][0]['decoder_calls'][0]['asr_samples_sha256']='wrong'
        elif changed=='incomplete':d['status']='running'
        else:d['cases']*=2
        save_json(path,d)
    with pytest.raises(ValueError):evaluate_study(inputs,dirs,plan)
