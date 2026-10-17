import hashlib
import json
import wave

import numpy as np
import pytest

from autosub.alignment_lab import file_hash
from autosub.edge_context_lab import edge_window
from autosub.edge_context_evaluation import evaluate


def write(path,data):path.write_text(json.dumps(data),encoding='utf-8')


def fixture(tmp_path):
    path=tmp_path/'audio.wav'
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(bytes(64000))
    source=tmp_path/'original.json';write(source,{'unaltered':True})
    geometry=edge_window(0,1000,2000,2000)
    signature=hashlib.sha256(np.zeros(32000,dtype=np.float32).tobytes()).hexdigest()
    plan={'text_provided':False,'cases':[{'id':'synthetic-case','kind':'unlabelled-natural','audio_path':str(path),
        'audio_sha256':file_hash(path),'asr_samples_sha256':signature,'geometry':geometry}],
        'source_projects':[{'path':str(source),'sha256':file_hash(source)}],
        'strategies':['original-clips','expanded-context'],'decoding':{'initial_prompt':None},'model_identity':{'sha256':'test'}}
    plan_path=tmp_path/'plan.json';write(plan_path,plan)
    rows=[]
    for mode,text in [('original-clips','beta'),('expanded-context','alpha beta')]:
        rows.append({'id':'synthetic-case','mode':mode,'status':'complete','text':text,
            'asr_samples_sha256':signature,'samples':32000,
            'options':{'initial_prompt':None,'clip_timestamps':geometry[
                'original_relative_clips' if mode=='original-clips' else 'expanded_relative_clips']},
            'resolved_options':{'initial_prompt':None},'segments':[{'text':text,'temperature':0.,'words':[]}]})
    run={'status':'complete','reference_text_provided':False,'plan_sha256':file_hash(plan_path),
         'model_identity':plan['model_identity'],'results':rows}
    run_path=tmp_path/'run.json';write(run_path,run)
    return plan_path,run_path


def test_evaluation_retains_pairs_and_scores_only_confirmed_case_without_mutation(tmp_path):
    plan,run=fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    result=evaluate(plan,run,'alpha beta',confirmed_case_id='synthetic-case')
    pair=result['pairs'][0]
    assert pair['modes']['expanded-context']['confirmed_case_text_score']['character_edits']==0
    assert pair['modes']['original-clips']['confirmed_case_text_score']['character_edits']>0
    assert result['production_adoption'] is False and result['human_completion_rate'] is None
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('change',['source','decoder','duplicate','incomplete','missing','project'])
def test_changed_or_incomplete_evidence_is_rejected(tmp_path,change):
    plan,run=fixture(tmp_path);data=json.loads(run.read_text(encoding='utf-8'))
    if change=='source':(tmp_path/'audio.wav').write_bytes(b'changed')
    elif change=='project':(tmp_path/'original.json').write_bytes(b'changed')
    else:
        if change=='decoder':data['results'][1]['options']['clip_timestamps']=[.1,2.]
        elif change=='duplicate':data['results'].append(data['results'][0])
        elif change=='missing':data['results'].pop()
        else:data['status']='running'
        write(run,data)
    with pytest.raises(ValueError):evaluate(plan,run)
