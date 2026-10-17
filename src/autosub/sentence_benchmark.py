"""Known source-utterance regions, never acoustic word-onset gold."""
from __future__ import annotations

from collections import Counter
import json
import wave
from pathlib import Path

from .alignment_lab import file_hash
from .domain import word_key
from .language_validation import normalized_text


def prepare_pairs(dataset_path, output, pairs, gaps=(0, 300, 800)):
    """Join complete cached PCM utterances; reference labels stay in a separate file."""
    dataset_path, output = Path(dataset_path), Path(output)
    data = json.loads(dataset_path.read_text(encoding='utf-8'))
    samples = {s['id']: s for s in data['samples']}
    if (not pairs or not gaps or len(samples) != len(data['samples']) or len(set(tuple(p) for p in pairs)) != len(pairs)
            or any(type(g) is not int or g < 0 for g in gaps) or len(set(gaps)) != len(gaps)):
        raise ValueError('Invalid pairs/gaps')
    prepared = []
    for left, right in pairs:
        a, b = samples[left], samples[right]
        if a['language'] != 'ja' or b['language'] != 'ja' or left == right:
            raise ValueError('Native sentence benchmark currently uses distinct Japanese utterances')
        parts = []
        input_manifest = dataset_path.parent.parent / 'fleurs-input' / 'manifest.json'
        cache = json.loads(input_manifest.read_text(encoding='utf-8'))
        if cache['source_manifest_sha256'] != file_hash(dataset_path) or file_hash(cache['audio_path']) != cache['audio_sha256']:
            raise ValueError('Cached inputs differ from source dataset')
        for sample in (a, b):
            # The earlier conversion produced id.wav alongside the reference-free input.
            path = dataset_path.parent.parent / 'fleurs-input' / (sample['id'] + '.wav')
            with wave.open(str(path), 'rb') as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                    raise ValueError('Expected cached mono16k PCM16')
                pcm = source.readframes(source.getnframes())
            window = next(w for w in cache['windows'] if w['id'] == sample['id'])
            if (window['source_audio_sha256'] != sample['audio_sha256']
                    or file_hash(sample['audio_path']) != sample['audio_sha256']):
                raise ValueError('Original dataset audio changed')
            with wave.open(cache['audio_path'], 'rb') as combined:
                start, end = window['window_ms']
                combined.setpos(start * 16)
                expected = combined.readframes((end - start) * 16)
            if pcm + bytes(window['padding_samples'] * 2) != expected:
                raise ValueError('Converted source no longer matches frozen PCM input')
            parts.append((path, pcm, sample))
        prepared.append(parts)
    if output.exists():
        raise ValueError('Output exists')
    output.mkdir(parents=True)
    cases = []
    for index, parts in enumerate(prepared):
        for gap in gaps:
            name = f'pair-{index + 1}-gap-{gap}'
            audio = output / (name + '.wav')
            cursor, regions = 0, []
            with wave.open(str(audio), 'wb') as target:
                target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                for number, (path, pcm, sample) in enumerate(parts):
                    if number:
                        target.writeframes(bytes(gap * 32))
                        cursor += gap * 16
                    start = cursor
                    target.writeframes(pcm)
                    cursor += len(pcm) // 2
                    regions.append({'sample_id': sample['id'], 'source_path': str(path.resolve()),
                        'source_sha256': file_hash(path), 'sample_frames': [start, cursor],
                        'reference_text': sample['reference_text']})
            cases.append({'id': name, 'language': 'ja', 'audio_path': str(audio.resolve()),
                'audio_sha256': file_hash(audio), 'gap_ms': gap, 'regions': regions})
    manifest = {'scope': 'synthetic complete-utterance concatenations; not natural dialogue or word-onset gold',
        'dataset': data.get('dataset'), 'license': data.get('license'),
        'dataset_sha256': file_hash(dataset_path), 'cases': cases}
    (output / 'labels.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def matched_positions(reference, hypothesis):
    """Deterministic minimum-edit matches, ambiguous repeated text is not resolved acoustically."""
    n, m = len(reference), len(hypothesis)
    grid = [list(range(m + 1))]
    for i, a in enumerate(reference, 1):
        row = [i]
        for j, b in enumerate(hypothesis, 1):
            row.append(min(grid[i - 1][j - 1] + (a != b), grid[i - 1][j] + 1, row[-1] + 1))
        grid.append(row)
    i, j, matches = n, m, {}
    while i or j:
        if i and j and grid[i][j] == grid[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            if reference[i - 1] == hypothesis[j - 1]:
                matches[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif i and grid[i][j] == grid[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return grid[-1][-1], matches


def assess_pair(project, case):
    """Report reference-matched prefix carry candidates; never count this as gold accuracy."""
    regions = case['regions']
    references = [normalized_text(r['reference_text']) for r in regions]
    if len(regions) != 2 or not all(references):
        raise ValueError('Expected two source utterances')
    text = ''.join(normalized_text(w.text) for w in project.words)
    reference = ''.join(references)
    edits, matches = matched_positions(reference, text)
    split = len(references[0])
    prefix_anchor = references[1][:min(4, len(references[1]))]
    reverse = {ref: hyp for hyp, ref in matches.items()}
    anchor_positions = [reverse.get(p) for p in range(split, split + len(prefix_anchor))]
    normalized_raw = normalized_text(''.join(w.text for w in project.words))
    eligible = (len(matches) >= .9 * len(reference) and text == normalized_raw
        and reference.find(prefix_anchor) == reference.rfind(prefix_anchor)
        and 0 <= text.find(prefix_anchor) == text.rfind(prefix_anchor)
        and all(p is not None for p in anchor_positions)
        and anchor_positions == list(range(anchor_positions[0], anchor_positions[0] + len(prefix_anchor))))
    keys = [word_key(w) for w in project.words]
    expected = Counter(keys)
    actual = Counter(t for c in project.cues for t in c.source_tokens or [])
    order_preserved = [t for c in project.cues for t in c.source_tokens or []] == keys
    if len(set(keys)) != len(keys):
        return {'status': 'unrated-duplicate-word-identity', 'source_tokens_preserved': expected == actual,
            'source_token_order_preserved': order_preserved,
            'normalized_characters_preserved': normalized_raw == normalized_text(''.join(c.text for c in project.cues))}
    owners, cursor = {}, 0
    for word, key in zip(project.words, keys):
        positions = [matches[i] for i in range(cursor, cursor + len(normalized_text(word.text))) if i in matches]
        owners[key] = positions
        cursor += len(normalized_text(word.text))
    mixed, candidates = [], []
    second_start = regions[1]['sample_frames'][0] / 16
    early = []
    for word, key in zip(project.words, keys):
        positions = owners[key]
        if (eligible and positions and all(p >= split for p in positions)
                and len(positions) == len(normalized_text(word.text)) and word.start_ms < second_start):
            early.append({'text': word.text, 'raw_word_ms': [word.start_ms, word.end_ms],
                'second_source_audio_start_ms': second_start, 'independently_reviewed': False})
    for index, cue in enumerate(project.cues):
        positions = [p for k in cue.source_tokens or [] for p in owners.get(k, [])]
        left, right = [p for p in positions if p < split], [p - split for p in positions if p >= split]
        if left and right:
            mixed.append({'cue_id': cue.id, 'text': cue.text, 'second_source_matched_characters': len(right)})
            later = {p for c in project.cues[index + 1:] for k in c.source_tokens or []
                     for p in owners.get(k, []) if p >= split}
            # Only contiguous 1-3 exact matched prefix characters, not entire second sentences.
            if eligible and sorted(right) == list(range(len(right))) and 0 < len(right) <= 3 and later:
                candidates.append({'cue_id': cue.id, 'text': cue.text, 'prefix': references[1][:len(right)],
                    'matched_only': True, 'independently_reviewed': False})
    return {'status': 'measured-reference-matches-only' if eligible else 'unrated-insufficient-or-ambiguous-reference-match',
        'prefix_mapping_eligible': eligible, 'source_tokens_preserved': expected == actual,
        'source_token_order_preserved': order_preserved,
        'normalized_characters_preserved': normalized_raw == normalized_text(''.join(c.text for c in project.cues)),
        'reference_characters': len(reference), 'character_edits': edits,
        'matched_reference_characters': len(matches), 'mixed_source_cues': mixed,
        'second_source_words_timestamped_before_source_audio': early,
        'next_prefix_carry_candidates': candidates, 'acoustic_timing_accuracy': None,
        'note': 'Exact edit-path matches only; no word-onset gold; substitutions/insertions unassigned; mixed cues are not automatically errors'}
