import importlib.util
from pathlib import Path

import pytest

from autosub.alignment_lab import file_hash
from autosub.domain import Project, Word, Cue, word_key


def runner():
    path=Path(__file__).resolve().parents[1]/'scripts'/'run_unlabelled_milestone.py'
    spec=importlib.util.spec_from_file_location('milestone_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sample(tmp_path):
    source=tmp_path/'private source.mp4';source.write_bytes(b'private source unchanged')
    output=tmp_path/'derived';output.mkdir()
    words=[Word(100,700,'テスト'),Word(900,1500,'ですね。')]
    cue=Cue('a',80,1700,'テストですね。',100,1500,source_tokens=list(map(word_key,words)))
    p=Project(media_path=str(source),duration_ms=2000,words=words,cues=[cue],
              asr_strategy='native-v1',native_groups=[[0,1]],native_resume_ms=2000,native_complete=True,status='complete')
    p.save(output/'generated.autosub.json')
    return p,output,{'source':{'path':str(source),'sha256':file_hash(source)}}


def test_unlabelled_baseline_keeps_source_project_and_uncertainty(tmp_path):
    module=runner();p,output,plan=sample(tmp_path)
    before=(output/'generated.autosub.json').read_bytes()
    result=module.report(p,output,plan)
    assert result['human_completion_rate'] is None and result['word_omission_rate'] is None
    assert result['name_accuracy'] is None and result['source_word_order_preserved'] is True
    assert result['exports']['.ass']['events']==result['exports']['.srt']['events']==1
    terms=module.json.loads((output/'term-review.json').read_text(encoding='utf-8'))
    assert terms['terms']==[{'recognized_form':'テスト','occurrences':1,'verified_spelling':None,'decision':'unreviewed'}]
    assert (output/'generated.autosub.json').read_bytes()==before
    assert file_hash(plan['source']['path'])==plan['source']['sha256']


@pytest.mark.parametrize('mutation',['drop','duplicate','reorder','display'])
def test_unlabelled_report_rejects_changed_words_or_ownership(tmp_path,mutation):
    module=runner();p,output,plan=sample(tmp_path)
    if mutation=='drop':p.cues[0].source_tokens.pop()
    elif mutation=='duplicate':p.cues[0].source_tokens.append(p.cues[0].source_tokens[-1])
    elif mutation=='reorder':p.cues[0].source_tokens.reverse()
    else:p.cues[0].text='猜測修正的名字'
    with pytest.raises(ValueError):module.report(p,output,plan)


def test_unlabelled_report_detects_changed_original(tmp_path):
    module=runner();p,output,plan=sample(tmp_path)
    Path(p.media_path).write_bytes(b'changed independently')
    with pytest.raises(ValueError,match='Original media changed'):module.report(p,output,plan)


def test_milestone_refuses_outputs_next_to_original(tmp_path,monkeypatch):
    module=runner();source=tmp_path/'source.mp4';source.write_bytes(b'private')
    monkeypatch.setattr(module,'model_identity',lambda *_:pytest.fail('must reject before model access'))
    monkeypatch.setattr('sys.argv',['run',str(source),'--model',str(tmp_path/'model'),'--output',str(tmp_path/'derived')])
    with pytest.raises(ValueError,match='outside the original'):module.main()
