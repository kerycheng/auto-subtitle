import json
from copy import deepcopy

import pytest

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import character_alignment, compare_runs, model_identity, validate_input, validate_resume
from autosub.paired_evaluation import _distance


@pytest.mark.parametrize('a,b,counts', [('abc', 'ac', (0, 1, 0)), ('abc', 'abxc', (0, 0, 1)),
    ('abc', 'adc', (1, 0, 0)), ('ＡＢ，日', 'ab日', (0, 0, 0)), ('한국어', '', (0, 3, 0)),
    ('漢', '汉', (1, 0, 0)), ('2', '二', (1, 0, 0))])
def test_edit_evidence(a, b, counts):
    row = character_alignment(a, b)
    assert tuple(row[k] for k in ('substitution', 'deletion', 'insertion')) == counts
    assert row['character_edits'] == sum(counts)
    assert len(row['differences']) == sum(counts)


def test_alignment_matches_levenshtein_for_ties():
    for a in ('abc', 'abab', '日本語', 'a'):
        for b in ('', 'baba', 'abc', '日本', 'aaaaaa'):
            row = character_alignment(a, b)
            assert row['character_edits'] == _distance(a, b)
            assert row['character_edits'] == sum(row[k] for k in ('substitution', 'deletion', 'insertion'))
    with pytest.raises(ValueError):
        character_alignment('！', 'abc')


def evidence(tmp_path):
    audio = tmp_path / 'a.wav'
    audio.write_bytes(b'fixed audio')
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'dataset': 'test', 'license': 'test', 'samples': [
        {'id': 'ja-0', 'language': 'ja', 'audio_path': str(audio), 'audio_sha256': file_hash(audio),
         'reference_text': 'abc'}]}), encoding='utf-8')
    manifest = {'duration_ms': 1000, 'audio_path': str(audio), 'audio_sha256': file_hash(audio),
        'source_manifest_sha256': file_hash(dataset), 'text_provided': False,
        'windows': [{'id': 'ja-0', 'language': 'ja', 'window_ms': [0, 1000],
                     'source_audio_sha256': file_hash(audio)}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.bin').write_bytes(b'weights')
    run = {'manifest_sha256': file_hash(path), 'backend': 'whisper', 'text_provided': False,
        'model_identity': model_identity(model), 'configuration': {'prompt': None}, 'model': str(model),
        'status': 'complete', 'load_seconds': 1, 'results': [{**manifest['windows'][0], 'text': 'ac', 'seconds': 2}]}
    return dataset, path, manifest, run, model


@pytest.mark.parametrize('change', ['model', 'config', 'input', 'duplicate', 'window', 'text'])
def test_resume_refuses_cross_model_and_changed_evidence(tmp_path, change):
    _, path, manifest, run, model = evidence(tmp_path)
    identity, config = deepcopy(run['model_identity']), deepcopy(run['configuration'])
    signature = file_hash(path)
    validate_resume(run, signature, 'whisper', identity, config, manifest)
    if change == 'model':
        (model / 'model.bin').write_bytes(b'other weights')
        identity = model_identity(model)
    elif change == 'config':
        config['prompt'] = 'reference leak'
    elif change == 'input':
        signature = 'different'
    elif change == 'duplicate':
        run['results'] *= 2
    elif change == 'window':
        run['results'][0]['window_ms'] = [1, 1000]
    else:
        run['text_provided'] = True
    with pytest.raises(ValueError):
        validate_resume(run, signature, 'whisper', identity, config, manifest)


@pytest.mark.parametrize('change', ['prompt', 'duplicate', 'bounds'])
def test_input_no_reference_and_valid_windows(tmp_path, change):
    _, _, manifest, _, _ = evidence(tmp_path)
    validate_input(manifest)
    if change == 'prompt':
        manifest['windows'][0]['reference_text'] = 'leaked'
    elif change == 'duplicate':
        manifest['windows'] *= 2
    else:
        manifest['windows'][0]['window_ms'] = [-1, 1001]
    with pytest.raises(ValueError):
        validate_input(manifest)


@pytest.mark.parametrize('change', [None, 'partial', 'dataset', 'result_input', 'source', 'duplicate_model'])
def test_comparison_checks_pairing_and_keeps_raw_text(tmp_path, change):
    dataset, path, manifest, run, _ = evidence(tmp_path)
    result = tmp_path / 'result.json'
    if change == 'partial':
        run['status'] = 'cancelled'
    elif change == 'dataset':
        dataset.write_text('{}', encoding='utf-8')
    elif change == 'result_input':
        run['manifest_sha256'] = 'other'
    elif change == 'source':
        (tmp_path / 'a.wav').write_bytes(b'changed')
    result.write_text(json.dumps(run), encoding='utf-8')
    paths = [result, result] if change == 'duplicate_model' else [result]
    if change:
        with pytest.raises(ValueError):
            compare_runs(dataset, path, paths)
    else:
        before = result.read_bytes()
        report = compare_runs(dataset, path, paths)
        row = report['runs'][0]['samples'][0]
        assert row['recognized_text'] == 'ac' and row['deletion'] == 1
        assert report['runs'][0]['by_language']['ja']['mechanical_cer'] == pytest.approx(1 / 3)
        assert 'unrated' in report['semantic_omissions_hallucinations_proper_names']
        assert result.read_bytes() == before


def test_comparison_refuses_different_shared_compute_conditions(tmp_path):
    dataset, path, _, run, _ = evidence(tmp_path)
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    first.write_text(json.dumps(run), encoding='utf-8')
    run['model_identity']['sha256'] = 'other-model'
    run['configuration']['device'] = 'cpu'
    second.write_text(json.dumps(run), encoding='utf-8')
    with pytest.raises(ValueError, match='common ASR conditions'):
        compare_runs(dataset, path, [first, second])


def test_parenthetical_caution_does_not_silently_change_reference_or_score(tmp_path):
    dataset, path, manifest, run, _ = evidence(tmp_path)
    data = json.loads(dataset.read_text(encoding='utf-8'))
    data['samples'][0]['reference_text'] = 'ab(Name)c'
    dataset.write_text(json.dumps(data), encoding='utf-8')
    manifest['source_manifest_sha256'] = file_hash(dataset)
    path.write_text(json.dumps(manifest), encoding='utf-8')
    run['manifest_sha256'] = file_hash(path)
    result = tmp_path / 'result.json'
    result.write_text(json.dumps(run), encoding='utf-8')
    row = compare_runs(dataset, path, [result])['runs'][0]['samples'][0]
    assert row['reference_parentheticals'] == ['(Name)']
    assert row['reference_text'] == 'ab(Name)c'
    assert row['character_edits'] == 5 and row['reference_characters'] == 7
