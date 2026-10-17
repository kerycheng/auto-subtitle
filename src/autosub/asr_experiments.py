"""Frozen selection rules and source-clock evidence for ASR ablations."""
from __future__ import annotations

import json
import itertools
import wave
from pathlib import Path

from .alignment_lab import file_hash


def validate_selection_plan(plan):
    if (plan.get('schema') != 'fleurs-selection-v1' or plan.get('dataset') != 'google/fleurs'
            or plan.get('split') != 'validation' or plan.get('configs') !=
            {'en': 'en_us', 'zh': 'cmn_hans_cn', 'ja': 'ja_jp', 'ko': 'ko_kr'}):
        raise ValueError('Unexpected selection plan')
    dev, held = plan['development_positions'], plan['reserved_positions']
    if (not dev or not held or any(type(x) is not int or x < 2 for x in dev + held)
            or len(set(dev + held)) != len(dev + held)):
        raise ValueError('Invalid or overlapping selection positions')


def prepare_contexts(cases_path, output, padding_ms):
    """Re-extract same targets with one explicit context variable; never read text for ASR."""
    if not 0 <= padding_ms <= 15000:
        raise ValueError('Invalid context padding')
    cases_path, output = Path(cases_path), Path(output)
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate cases')
    # All private source audio is opened read-only. Validate before creating outputs.
    chunks, evidence, cursor, windows = [], [], 0, []
    for case in cases:
        if file_hash(case['audio_path']) != case['audio_sha256']:
            raise ValueError('Source audio changed')
        with wave.open(case['audio_path'], 'rb') as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Expected 16kHz mono PCM16')
            duration = source.getnframes() // 16
            a, b = case['target_ms']
            if not 0 <= a < b <= duration:
                raise ValueError('Target outside source')
            lo, hi = max(0, a - padding_ms), min(duration, b + padding_ms)
            source.setpos(lo * 16)
            data = source.readframes((hi - lo) * 16)
            if len(data) != (hi - lo) * 32:
                raise ValueError('Truncated source')
        chunks.append(data)
        windows.append({'id': case['id'], 'language': 'ja', 'window_ms': [cursor, cursor + hi - lo],
                        'owner_ms': [cursor + a - lo, cursor + b - lo]})
        evidence.append({'id': case['id'], 'sample': case['sample'], 'target_ms': [a, b],
            'source_context_ms': [lo, hi], 'source_audio_path': case['audio_path'],
            'source_audio_sha256': case['audio_sha256'], 'reason': case['reason'],
            'source_primary_context_text': case['primary_context_text']})
        cursor += hi - lo
    if output.exists():
        raise ValueError('Output exists')
    output.mkdir(parents=True)
    audio = output / 'contexts.wav'
    with wave.open(str(audio), 'wb') as target:
        target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for data in chunks:
            target.writeframes(data)
    manifest = {'schema': 'asr-lab-v1', 'language': 'ja', 'duration_ms': cursor,
        'audio_path': str(audio.resolve()), 'audio_sha256': file_hash(audio), 'text_provided': False,
        'source_cases_sha256': file_hash(cases_path), 'context_padding_ms': padding_ms, 'windows': windows}
    (output / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest['evidence_sha256'] = file_hash(output / 'evidence.json')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def target_word_evidence(row):
    """Whole raw words near the target, for orientation only; never a replacement text."""
    start, end = row['owner_ms']
    origin = row['window_ms'][0]
    words = []
    for segment in row.get('segments', []):
        for word in segment.get('words') or []:
            a, b = origin + word['start'] * 1000, origin + word['end'] * 1000
            if a < end and b > start:
                words.append({'text': word['word'], 'audio_ms': [a, b],
                    'crosses_target_edge': a < start or b > end, 'zero_duration': a == b})
    return {'words': words, 'text': ''.join(w['text'] for w in words),
        'interpretation': 'whole overlapping raw words; timestamps may drift; not gold or an accepted replacement'}


def repetition_evidence(text):
    """Literal output shape, not a judgement about elongation or repeated speech."""
    maximum, offset, runs = 0, 0, []
    for char, group in itertools.groupby(text):
        count = sum(1 for _ in group)
        maximum = max(maximum, count)
        if count >= 12:
            runs.append({'character': char, 'length': count, 'text_offset': [offset, offset + count]})
        offset += count
    best = None
    for size in range(2, min(64, len(text) // 3) + 1):
        for start in range(len(text) - size * 3 + 1):
            block = text[start:start + size]
            if len(set(block)) == 1 or text[start + size:start + size * 3] != block * 2:
                continue
            end = start + size * 3
            while text.startswith(block, end):
                end += size
            if end - start >= 12 and (best is None or end - start > best['covered_characters']):
                best = {'text': block, 'copies': (end - start) // size, 'text_offset': [start, end],
                        'covered_characters': end - start}
    return {'characters': len(text), 'max_same_character_run': maximum, 'long_runs': runs,
        'consecutive_phrase_repetition': best,
        'interpretation': 'literal repeated characters only; do not automatically delete screams, laughter or elongation'}
