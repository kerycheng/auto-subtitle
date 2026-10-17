import json
import runpy
import sys
from pathlib import Path

import pytest

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.target_text import target_distance


def test_context_is_explicitly_unscored_but_internal_missing_text_counts():
    result = target_distance('日本には島がある', '前置き日本に島がある後書き')
    assert result['character_edits'] == 1
    assert result['matched_hypothesis_text'] == '日本に島がある'
    assert result['unscored_context_characters'] == 6
    assert 'cannot measure whole-file' in result['scope']


def test_internal_extra_speech_is_not_free_context():
    assert target_distance('abcxyz', 'abcQxyz')['character_edits'] == 1


def test_equal_cost_prefers_exact_matches_and_reports_the_span():
    r = target_distance('ab', 'xay')
    assert r['character_edits'] == 1
    start, end = r['normalized_hypothesis_span']
    assert 'xay'[start:end] == r['matched_hypothesis_text']
    assert 'a' in r['matched_hypothesis_text']


def test_normalization_and_missing_hypothesis():
    assert target_distance('ＡＢ、が！', 'abか\u3099')['target_only_cer'] == 0
    r = target_distance('全部欠落', '')
    assert r['target_only_cer'] == 1 and r['matched_hypothesis_text'] == ''


@pytest.mark.parametrize('a,b', [('', 'text'), ('あ'*3000, 'い'*3000)], ids=['empty', 'grid-limit'])
def test_unusable_targets_fail_before_grid_allocation(a, b):
    with pytest.raises(ValueError):
        target_distance(a, b)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def report_fixture(tmp_path):
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    audio, reference, project = [tmp_path/name for name in ('audio.wav', 'reference.srt', 'project.json')]
    for path in (audio, reference, project):
        path.write_bytes(path.name.encode())
    window = {'id':'w1', 'language':'ja', 'window_ms':[0, 30000], 'owner_ms':[0, 30000]}
    manifest = {'audio_path':str(audio), 'audio_sha256':file_hash(audio), 'duration_ms':30000,
        'windows':[window], 'text_provided':False}
    labels = {'reference_sha256':file_hash(reference), 'project_sha256':file_hash(project),
        'windows':[{'id':'w1', 'reference_text':'日本には島がある', 'source_rows':[[1]], 'reference_span_ms':[500, 1000]}],
        'scope':'target only', 'overlap_reconstruction':[], 'unscored_reference_ids':[2]}
    write_json(inputs/'manifest.json',manifest)
    write_json(inputs/'evidence.json',labels)
    write_json(inputs/'provenance.json',{'manifest_sha256':file_hash(inputs/'manifest.json'),
        'evidence_sha256':file_hash(inputs/'evidence.json'), 'reference_path':str(reference), 'project_path':str(project)})
    result = tmp_path/'result.json'
    write_json(result,{'manifest_sha256':file_hash(inputs/'manifest.json'), 'backend':'whisper',
        'model':'local model', 'model_identity':{}, 'configuration':{}, 'status':'complete',
        'text_provided':False, 'results':[{**window,'text':'前置き日本には島がある後書き'}]})
    return inputs, result, audio, reference, project


def run_report(monkeypatch, inputs, result, output):
    monkeypatch.setattr(sys, 'argv', ['report_target_asr.py',str(inputs),str(result),'--output',str(output)])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/report_target_asr.py'),run_name='__main__')


def test_report_preserves_full_hypotheses_unscored_events_and_sources(tmp_path, monkeypatch):
    inputs, result, *sources = report_fixture(tmp_path)
    before = {p:p.read_bytes() for p in [*sources,result,*inputs.iterdir()]}
    output = tmp_path/'report'
    run_report(monkeypatch,inputs,result,output)
    r = json.loads((output/'report.json').read_text(encoding='utf-8'))
    assert r['unscored_reference_ids'] == [2]
    assert r['runs'][0]['target_only_cer'] == 0
    assert r['runs'][0]['samples'][0]['hypothesis'].startswith('前置き')
    assert r['runs'][0]['samples'][0]['unscored_context_characters'] == 6
    assert all(p.read_bytes()==b for p,b in before.items())
    with pytest.raises(ValueError,match='exists'):
        run_report(monkeypatch,inputs,result,output)


@pytest.mark.parametrize('change',['audio','reference','project','evidence','window','incomplete','duplicate'])
def test_changed_source_or_incomplete_results_never_publish_report(tmp_path,monkeypatch,change):
    inputs,result,audio,reference,project = report_fixture(tmp_path)
    if change in {'audio','reference','project','evidence'}:
        target = {'audio':audio,'reference':reference,'project':project,'evidence':inputs/'evidence.json'}[change]
        if change == 'evidence':
            d=json.loads(target.read_text()); d['scope']='changed'; write_json(target,d)
        else:
            target.write_bytes(b'changed')
    else:
        d=json.loads(result.read_text())
        if change == 'window':
            d['results'][0]['window_ms']=[1,30000]
        elif change == 'incomplete':
            d['status']='running'
        else:
            d['results']*=2
        write_json(result,d)
    output=tmp_path/'report'
    with pytest.raises(ValueError):
        run_report(monkeypatch,inputs,result,output)
    assert not output.exists()


@pytest.mark.parametrize('complete',[True,False])
def test_preparation_keeps_labels_outside_asr_and_rejects_unfinished_source(tmp_path,monkeypatch,complete):
    audio=tmp_path/'audio.wav'; audio.write_bytes(b'reference-free PCM fixture')
    project=tmp_path/'project.json'
    Project(audio_path=str(audio),duration_ms=60000,asr_strategy='native-v1',
        native_complete=complete,native_resume_ms=60000 if complete else 0).save(project)
    reference=tmp_path/'reference.srt'
    reference.write_text('1\n00:00:01,000 --> 00:00:02,000\n秘密の原語\n\n2\n00:00:29,800 --> 00:00:30,500\n境界の原語\n',encoding='utf-8')
    before=[p.read_bytes() for p in (audio,project,reference)]
    output=tmp_path/'prepared'
    monkeypatch.setattr(sys,'argv',['prepare_target_asr.py',str(project),str(reference),'--output',str(output)])
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_target_asr.py'
    if not complete:
        with pytest.raises(ValueError,match='completed'):
            runpy.run_path(str(script),run_name='__main__')
        assert not output.exists()
    else:
        runpy.run_path(str(script),run_name='__main__')
        manifest=(output/'manifest.json').read_text()
        labels=json.loads((output/'evidence.json').read_text(encoding='utf-8'))
        assert '原語' not in manifest and 'reference_text' not in manifest
        assert labels['windows'][0]['reference_text']=='秘密の原語'
        assert labels['unscored_reference_ids']==[2]
    assert before==[p.read_bytes() for p in (audio,project,reference)]
