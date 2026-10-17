import importlib
import json
import sys
import urllib.error
import urllib.parse
import wave
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import pytest

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import validate_input
from autosub.asr_experiments import prepare_contexts, target_word_evidence, validate_selection_plan, repetition_evidence


def plan():
    return {'schema': 'fleurs-selection-v1', 'dataset': 'google/fleurs', 'split': 'validation',
        'configs': {'en': 'en_us', 'zh': 'cmn_hans_cn', 'ja': 'ja_jp', 'ko': 'ko_kr'},
        'development_positions': [2], 'reserved_positions': [3]}


@pytest.mark.parametrize('change', ['overlap', 'observed', 'float', 'duplicate', 'dataset', 'empty'])
def test_frozen_plan_rejects_leakage_and_invalid_selections(change):
    p = plan()
    validate_selection_plan(p)
    if change == 'overlap':
        p['reserved_positions'] = [2]
    elif change == 'observed':
        p['development_positions'] = [0]
    elif change == 'float':
        p['development_positions'] = [2.5]
    elif change == 'duplicate':
        p['development_positions'] = [2, 2]
    elif change == 'dataset':
        p['dataset'] = 'other'
    else:
        p['reserved_positions'] = []
    with pytest.raises(ValueError):
        validate_selection_plan(p)


def cases_file(tmp_path):
    audio = tmp_path / 'source.wav'
    with wave.open(str(audio), 'wb') as target:
        target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        target.writeframes(b'\x10\x00' * 16000 * 12)
    cases = [{'id': 'c1', 'sample': 'ja-01', 'reason': 'test', 'audio_path': str(audio),
        'audio_sha256': file_hash(audio), 'target_ms': [5000, 5500],
        'primary_context_text': 'REFERENCE MUST NOT BE IN ASR INPUT'}]
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(cases), encoding='utf-8')
    return path, audio, cases


def test_contexts_keep_source_clock_whole_pcm_and_no_text(tmp_path):
    cases, source, _ = cases_file(tmp_path)
    before = source.read_bytes()
    a = prepare_contexts(cases, tmp_path / 'short', 3000)
    b = prepare_contexts(cases, tmp_path / 'long', 8000)
    assert a['windows'][0]['owner_ms'] == [3000, 3500]
    assert b['windows'][0]['owner_ms'] == [5000, 5500]
    assert a['duration_ms'] == 6500 and b['duration_ms'] == 12000
    validate_input(a)
    validate_input(b)
    assert 'REFERENCE' not in json.dumps(a) and 'REFERENCE' not in json.dumps(b)
    assert a['source_cases_sha256'] == b['source_cases_sha256']
    assert a['evidence_sha256'] == file_hash(tmp_path / 'short/evidence.json')
    with wave.open(a['audio_path'], 'rb') as decoded:
        assert decoded.getnframes() == 6500 * 16
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match='exists'):
        prepare_contexts(cases, tmp_path / 'short', 3000)


@pytest.mark.parametrize('change', ['source', 'target', 'duplicate', 'padding'])
def test_context_rejects_bad_inputs_before_creating_output(tmp_path, change):
    path, audio, cases = cases_file(tmp_path)
    if change == 'source':
        audio.write_bytes(b'changed')
    elif change == 'target':
        cases[0]['target_ms'] = [12000, 13000]
    elif change == 'duplicate':
        cases *= 2
    path.write_text(json.dumps(cases), encoding='utf-8')
    output = tmp_path / 'out'
    with pytest.raises(ValueError):
        prepare_contexts(path, output, -1 if change == 'padding' else 3000)
    assert not output.exists()


def test_target_evidence_never_splits_edge_words_or_deduplicates_replies():
    row = {'window_ms': [10000, 15000], 'owner_ms': [11000, 12000], 'segments': [{'words': [
        {'start': .5, 'end': 1.2, 'word': '前半'}, {'start': 1.2, 'end': 1.7, 'word': 'はい'},
        {'start': 1.7, 'end': 2.3, 'word': 'はい'}, {'start': 2.3, 'end': 3, 'word': '後半'}]}]}
    before = deepcopy(row)
    result = target_word_evidence(row)
    assert result['text'] == '前半はいはい'
    assert [w['crosses_target_edge'] for w in result['words']] == [True, False, True]
    assert row == before
    assert target_word_evidence({'window_ms': [0, 1], 'owner_ms': [0, 1]})['text'] == ''


def test_repetition_is_observation_only_and_preserves_elongation():
    text = 'はいはい、いや' + 'ー' * 25 + '！'
    observed = repetition_evidence(text)
    assert observed['max_same_character_run'] == 25
    assert observed['long_runs'] == [{'character': 'ー', 'length': 25, 'text_offset': [7, 32]}]
    assert text == 'はいはい、いや' + 'ー' * 25 + '！'
    assert repetition_evidence('')['max_same_character_run'] == 0
    assert repetition_evidence('はいはい')['consecutive_phrase_repetition'] is None
    assert repetition_evidence('ー' * 30)['consecutive_phrase_repetition'] is None
    phrase = 'この動画の字幕は視聴者によって作成されました。'
    repeated = repetition_evidence('前' + phrase * 10 + '後')['consecutive_phrase_repetition']
    assert repeated['text'] == phrase and repeated['copies'] == 10
    assert repeated['text_offset'] == [1, 1 + len(phrase) * 10]


def script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module(name)


def test_download_fixed_positions_only_and_resume_without_network(tmp_path, monkeypatch):
    module = script(monkeypatch, 'fetch_asr_development')
    p = tmp_path / 'plan.json'
    p.write_text(json.dumps(plan()), encoding='utf-8')
    output = tmp_path / 'download'
    calls = []
    def fake_fetch(url, limit):
        calls.append(url)
        if '/rows?' in url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            assert query['offset'] == ['2']  # never touch reserved position 3
            config = query['config'][0]
            return json.dumps({'rows': [{'row_idx': 2, 'row': {'id': 22, 'raw_transcription': 'text',
                'audio': [{'src': 'https://example.invalid/' + config, 'type': 'audio/wav'}]}}]}).encode()
        return b'fixed audio'
    monkeypatch.setattr(module, 'fetch', fake_fetch)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['fetch', str(p), '--output', str(output)])
    module.main()
    assert len(calls) == 8
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and len(manifest['samples']) == 4
    monkeypatch.setattr(sys, 'argv', ['fetch', str(p), '--output', str(output), '--resume'])
    module.main()
    assert len(calls) == 8
    Path(manifest['samples'][0]['audio_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        module.main()
    assert len(calls) == 8


def test_download_checks_server_position(tmp_path, monkeypatch):
    module = script(monkeypatch, 'fetch_asr_development')
    p = tmp_path / 'plan.json'
    p.write_text(json.dumps(plan()), encoding='utf-8')
    monkeypatch.setattr(module, 'fetch', lambda *_: b'{"rows":[{"row_idx":3}]}')
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['fetch', str(p), '--output', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='different position'):
        module.main()


def test_rate_limit_respects_retry_after(monkeypatch):
    module = script(monkeypatch, 'fetch_language_samples')
    attempts, delays = [], []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return b'ok'
    def open_url(url, timeout):
        attempts.append(url)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(url, 429, 'rate limited', {'Retry-After': '17'}, None)
        return Response()
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    assert module.fetch('https://example.invalid/sample', 10) == b'ok'
    assert len(attempts) == 2 and delays == [17]


@pytest.mark.parametrize('change', [None, 'partial', 'audio', 'pairing'])
def test_context_report_never_adopts_and_checks_saved_evidence(tmp_path, monkeypatch, change):
    module = script(monkeypatch, 'report_context_experiment')
    cases, audio, _ = cases_file(tmp_path)
    inputs = tmp_path / 'input'
    manifest = prepare_contexts(cases, inputs, 3000)
    manifest_path = inputs / 'manifest.json'
    result = {'manifest_sha256': file_hash(manifest_path), 'backend': 'whisper', 'text_provided': False,
        'model_identity': {'sha256': 'test'}, 'configuration': {'prompt': None}, 'model': 'test-model',
        'status': 'complete', 'results': [{**manifest['windows'][0], 'text': '<full text>',
            'segments': [{'words': [{'start': 2.9, 'end': 3.2, 'word': '前半'}]}]}]}
    if change == 'partial':
        result['status'] = 'running'
    elif change == 'audio':
        audio.write_bytes(b'changed source')
    elif change == 'pairing':
        result['manifest_sha256'] = 'different'
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    before = path.read_bytes()
    output = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['report', str(cases), '--run', 'variant', str(manifest_path), str(path),
                                    '--output', str(output)])
    if change:
        with pytest.raises(ValueError):
            module.main()
        assert not output.exists()
    else:
        module.main()
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        assert report['accuracy'] is None and report['adopted'] is False
        assert report['runs'][0]['results'][0]['text'] == '<full text>'
        assert report['runs'][0]['results'][0]['target_evidence']['words'][0]['source_ms'] == [4900, 5200]
        assert '&lt;full text&gt;' in (output / 'index.html').read_text(encoding='utf-8')
    assert path.read_bytes() == before


@pytest.mark.parametrize('vocabulary,temperature_zero', [(False, False), (True, False), (True, True), (False, True)])
def test_probe_passes_only_explicit_prompt_and_decode_options_and_rejects_changed_resume(
        tmp_path, monkeypatch, vocabulary, temperature_zero):
    import faster_whisper
    import autosub.runtime
    module = script(monkeypatch, 'probe_asr_backends')
    cases, _, _ = cases_file(tmp_path)
    manifest = tmp_path / 'inputs'
    prepare_contexts(cases, manifest, 3000)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.bin').write_bytes(b'fixed weights')
    prompt = tmp_path / 'vocabulary.txt'
    prompt.write_text('APEX、SF6', encoding='utf-8')
    output = tmp_path / 'probe'
    captured, gpu_calls = [], []
    @dataclass
    class Segment:
        text: str
    @dataclass
    class Options:
        initial_prompt: str | None
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, audio, **kwargs):
            captured.append(kwargs)
            return iter([Segment('raw hypothesis')]), type('Info', (), {
                'transcription_options': Options(kwargs['initial_prompt'])})()
    monkeypatch.setattr(faster_whisper, 'WhisperModel', Model)
    monkeypatch.setattr(autosub.runtime, 'check_runtime', lambda *_: None)
    def gpu(*args, **kwargs):
        gpu_calls.append(True)
        return 'Test GPU, 6000, 1000, 0'
    monkeypatch.setattr(module.subprocess, 'check_output', gpu)
    argv = ['probe', str(manifest / 'manifest.json'), '--backend', 'whisper', '--model', str(model),
            '--output', str(output)]
    if vocabulary:
        argv.extend(['--prompt-file', str(prompt)])
    if temperature_zero:
        argv.append('--temperature-zero')
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    assert len(captured) == 1
    actual = captured[0]
    assert actual['initial_prompt'] == ('APEX、SF6' if vocabulary else None)
    assert ('temperature' in actual) == temperature_zero
    if temperature_zero:
        assert actual['temperature'] == 0.0
    assert actual['vad_filter'] is False and actual['condition_on_previous_text'] is False
    assert 'REFERENCE' not in json.dumps(actual)
    result_path = output / 'result.json'
    before = result_path.read_bytes()
    calls_after_first_run = len(gpu_calls)
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    module.main()
    assert result_path.read_bytes() == before and len(captured) == 1 and len(gpu_calls) == calls_after_first_run
    if vocabulary:
        prompt.write_text('Changed vocabulary', encoding='utf-8')
        with pytest.raises(ValueError, match='Resume mismatch'):
            module.main()
        assert result_path.read_bytes() == before and len(captured) == 1
