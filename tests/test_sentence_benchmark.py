import json
import wave
from pathlib import Path

import pytest

from autosub.alignment_lab import file_hash
from autosub.domain import Project, Word, Cue, word_key
from autosub.sentence_benchmark import assess_pair, prepare_pairs, matched_positions
from autosub.paired_evaluation import _distance


def controlled_project(carry=False, combined=False):
    words = [Word(100, 500, '前の文章。'), Word(1100, 1200, '日本'), Word(1200, 1800, 'には島がある。')]
    p = Project(duration_ms=2000, words=words)
    groups = [words] if combined else [words[:2], words[2:]] if carry else [words[:1], words[1:]]
    p.cues = [Cue(str(i), group[0].start_ms, group[-1].end_ms, ''.join(w.text for w in group),
        group[0].start_ms, group[-1].end_ms, source_tokens=[word_key(w) for w in group]) for i, group in enumerate(groups)]
    case = {'regions': [{'reference_text': '前の文章。', 'sample_frames': [0, 16000]},
                        {'reference_text': '日本には島がある。', 'sample_frames': [16000, 32000]}]}
    return p, case


def test_prefix_carry_detected_without_changing_any_words_or_cues():
    p, case = controlled_project(carry=True)
    before = repr(p)
    report = assess_pair(p, case)
    assert report['prefix_mapping_eligible'] and report['character_edits'] == 0
    assert len(report['next_prefix_carry_candidates']) == 1
    assert report['next_prefix_carry_candidates'][0]['prefix'] == '日本'
    assert not report['next_prefix_carry_candidates'][0]['independently_reviewed']
    assert report['source_tokens_preserved'] and report['normalized_characters_preserved']
    assert report['acoustic_timing_accuracy'] is None and repr(p) == before


@pytest.mark.parametrize('combined', [False, True])
def test_complete_two_sentences_together_are_not_partial_prefix_carry(combined):
    p, case = controlled_project(combined=combined)
    r = assess_pair(p, case)
    assert not r['next_prefix_carry_candidates']
    assert len(r['mixed_source_cues']) == int(combined)


@pytest.mark.parametrize('change', ['low_match', 'shared_anchor', 'duplicate', 'missing_ownership'])
def test_insufficient_evidence_is_not_reported_as_success(change):
    p, case = controlled_project(carry=True)
    if change == 'low_match':
        case['regions'][1]['reference_text'] = '全く違う言葉を言っている。'
    elif change == 'shared_anchor':
        case['regions'][0]['reference_text'] = '日本には前の文章。'
    elif change == 'duplicate':
        p.words.append(p.words[-1])
    else:
        p.cues[0].source_tokens = []
    r = assess_pair(p, case)
    if change == 'missing_ownership':
        assert not r['source_tokens_preserved']
    else:
        assert r['status'].startswith('unrated')


def test_early_raw_word_times_are_separate_from_display_lead_and_not_corrected():
    p, case = controlled_project()
    p.words[1].start_ms = 900
    p.cues[1].source_tokens[0] = word_key(p.words[1])
    r = assess_pair(p, case)
    assert r['second_source_words_timestamped_before_source_audio'][0]['text'] == '日本'
    assert p.words[1].start_ms == 900


def test_matching_agrees_with_existing_distance_and_only_maps_equal_characters():
    for a in ('日本には島がある', '前の文章', 'ああああ次'):
        for b in ('', a, a + 'え', a[:-1], '全く違う'):
            edits, matches = matched_positions(a, b)
            assert edits == _distance(a, b)
            assert all(a[j] == b[i] for i, j in matches.items())
            assert len(set(matches.values())) == len(matches)


def test_tokens_reordered_without_changing_visible_text_do_not_pass_ownership():
    p, case = controlled_project()
    p.cues[1].source_tokens.reverse()
    r = assess_pair(p, case)
    assert r['source_tokens_preserved'] and r['normalized_characters_preserved']
    assert not r['source_token_order_preserved']


def fixture(tmp_path):
    dataset_dir = tmp_path / 'fleurs-development'
    converted = tmp_path / 'fleurs-input'
    dataset_dir.mkdir()
    converted.mkdir()
    samples, windows, cursor, all_pcm = [], [], 0, b''
    for i, text in enumerate(('前の文章。', '日本には島がある。')):
        name = f'ja-{i}'
        pcm = bytes([i + 1, 0]) * 16000
        original = dataset_dir / (name + '.audio')
        original.write_bytes(pcm)
        with wave.open(str(converted / (name + '.wav')), 'wb') as target:
            target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            target.writeframes(pcm)
        samples.append({'id': name, 'language': 'ja', 'reference_text': text,
            'audio_path': str(original), 'audio_sha256': file_hash(original)})
        windows.append({'id': name, 'window_ms': [cursor, cursor + 1000], 'padding_samples': 0,
            'source_audio_sha256': file_hash(original)})
        cursor += 1000
        all_pcm += pcm
    dataset = dataset_dir / 'manifest.json'
    dataset.write_text(json.dumps({'samples': samples}), encoding='utf-8')
    combined = converted / 'input.wav'
    with wave.open(str(combined), 'wb') as target:
        target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        target.writeframes(all_pcm)
    (converted / 'manifest.json').write_text(json.dumps({'source_manifest_sha256': file_hash(dataset),
        'audio_path': str(combined), 'audio_sha256': file_hash(combined), 'windows': windows}), encoding='utf-8')
    return dataset, converted


def test_complete_pcm_and_source_clock_are_preserved(tmp_path):
    dataset, converted = fixture(tmp_path)
    before = {p: file_hash(p) for p in converted.iterdir()}
    result = prepare_pairs(dataset, tmp_path / 'out', [('ja-0', 'ja-1')])
    assert len(result['cases']) == 3
    for case in result['cases']:
        assert case['regions'][1]['sample_frames'][0] == (1000 + case['gap_ms']) * 16
        with wave.open(case['audio_path'], 'rb') as source:
            data = source.readframes(source.getnframes())
        assert data == b'\x01\x00' * 16000 + bytes(case['gap_ms'] * 32) + b'\x02\x00' * 16000
    assert all(file_hash(p) == h for p, h in before.items())


@pytest.mark.parametrize('change', ['converted', 'original', 'duplicates', 'negative'])
def test_changed_sources_or_bad_plan_rejected_before_output(tmp_path, change):
    dataset, converted = fixture(tmp_path)
    if change == 'converted':
        with wave.open(str(converted / 'ja-0.wav'), 'wb') as target:
            target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            target.writeframes(b'\x10\x00' * 16000)
    elif change == 'original':
        (dataset.parent / 'ja-0.audio').write_bytes(b'changed')
    pairs = [('ja-0', 'ja-1')] * (2 if change == 'duplicates' else 1)
    with pytest.raises(ValueError):
        prepare_pairs(dataset, tmp_path / 'out', pairs, gaps=(-1,) if change == 'negative' else (0,))
    assert not (tmp_path / 'out').exists()
