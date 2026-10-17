"""Original-language reference evidence; never used as recognition input."""
from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path

from .domain import word_key
from .language_validation import normalized_text
from .sentence_benchmark import matched_positions


def read_srt_events(path):
    text = Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    events = []
    stamp = r'(\d+):(\d{2}):(\d{2})[,.](\d{3})'
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        match = re.fullmatch(stamp + r'\s*-->\s*' + stamp + r'[^\n]*', lines[0]) if lines else None
        if not match or len(lines) < 2:
            raise ValueError('Invalid SRT event; no rows were silently skipped')
        values = [int(x) for x in match.groups()]
        if any(values[i] >= 60 for i in (1, 2, 5, 6)):
            raise ValueError('Invalid SRT clock')
        a, b = [((values[i]*60 + values[i+1])*60 + values[i+2])*1000 + values[i+3] for i in (0, 4)]
        if a >= b:
            raise ValueError('Nonpositive reference event')
        content = [line.strip() for line in lines[1:]]
        if not any(content):
            raise ValueError('Empty reference text')
        events.append({'id': len(events)+1, 'start_ms': a, 'end_ms': b, 'lines': content,
                       'text': '\n'.join(content), 'source_rows': [len(events)+1]})
    if not events or any(a['start_ms'] > b['start_ms'] for a, b in zip(events, events[1:])):
        raise ValueError('Empty or unsorted reference')
    return events


def reconstruct_overlap_bridges(events):
    """Secondary evaluation view only. Never merge ordinary identical replies.

    A <=50ms two-line event, exactly touching two different single-line neighbors
    and containing exactly their complete texts, can be a split overlap display.
    Keep the raw view and record every inferred change.
    """
    result, removed, trace = copy.deepcopy(events), set(), []
    for i in range(1, len(events)-1):
        left, current, right = events[i-1:i+2]
        if (i-1 in removed or len(current['lines']) != 2 or len(left['lines']) != 1 or len(right['lines']) != 1
                or not 0 < current['end_ms']-current['start_ms'] <= 50
                or left['text'] == right['text'] or set(current['lines']) != {left['text'], right['text']}
                or left['end_ms'] != current['start_ms'] or current['end_ms'] != right['start_ms']):
            continue
        result[i-1]['end_ms'] = current['end_ms']
        result[i+1]['start_ms'] = current['start_ms']
        result[i-1]['source_rows'] = sorted(set(result[i-1]['source_rows'] + current['source_rows']))
        result[i+1]['source_rows'] = sorted(set(result[i+1]['source_rows'] + current['source_rows']))
        removed.add(i)
        trace.append({'kind': 'inferred-short-overlap-display', 'removed_display_row': current['id'],
            'neighbor_rows': [left['id'], right['id']], 'interval_ms': [current['start_ms'], current['end_ms']],
            'independently_confirmed': False})
    return [e for i, e in enumerate(result) if i not in removed], trace


def compare_text_reference(project, events):
    reference = ''.join(normalized_text(e['text']) for e in events)
    token_hypothesis = ''.join(normalized_text(w.text) for w in project.words)
    hypothesis = normalized_text(''.join(w.text for w in project.words))
    if not reference:
        raise ValueError('Empty normalized reference')
    # The controlled benchmark uses short references. Bound this exact Python
    # traceback grid; never unexpectedly allocate a 20-minute quadratic matrix.
    if len(reference)*len(hypothesis) > 4_000_000:
        raise ValueError('Reference comparison exceeds exact-grid limit; split evaluation into shorter independently defined excerpts')
    edits, matches = matched_positions(reference, hypothesis)
    identities = Counter(word_key(w) for w in project.words)
    ambiguous = [key for key, count in identities.items() if count > 1]
    if token_hypothesis != hypothesis or ambiguous:
        return {'reference_characters': len(reference), 'hypothesis_characters': len(hypothesis),
            'character_edits': edits, 'mechanical_cer': edits/len(reference), 'prefix_carry_candidates': [],
            'previous_suffix_carry_candidates': [], 'unrated_reference_boundaries': [e['id'] for e in events[1:]],
            'ownership_mapping': ('unrated-duplicate-word-identities' if ambiguous else
                                  'unrated-cross-token-unicode-normalization'),
            'ambiguous_word_identities': ambiguous, 'acoustic_timing_accuracy': None}
    offsets = [0]
    for event in events:
        offsets.append(offsets[-1] + len(normalized_text(event['text'])))
    by_key, cursor = {}, 0
    for word in project.words:
        key = word_key(word)
        length = len(normalized_text(word.text))
        by_key[key] = [matches[i] for i in range(cursor, cursor+length) if i in matches]
        cursor += length
    cue_positions = [[p for k in c.source_tokens or [] for p in by_key.get(k, [])] for c in project.cues]
    reverse = {r: h for h, r in matches.items()}
    candidates, suffix_candidates, unrateable = [], [], []
    for index in range(1, len(events)):
        start, end = offsets[index:index+2]
        anchor = reference[start:min(start+4, end)]
        positions = [reverse.get(p) for p in range(start, start+len(anchor))]
        lo, hi = max(0, start-24), min(len(reference), start+24)
        local_matches = sum(p in reverse for p in range(lo, hi)) / max(1, hi-lo)
        unique = (len(anchor) >= 3 and reference.find(anchor) == reference.rfind(anchor)
                  and 0 <= hypothesis.find(anchor) == hypothesis.rfind(anchor))
        eligible = (unique and local_matches >= .85 and all(p is not None for p in positions)
                    and positions == list(range(positions[0], positions[0]+len(anchor))))
        if not eligible:
            unrateable.append(events[index]['id'])
            continue
        for cue_index, (cue, matched) in enumerate(zip(project.cues, cue_positions)):
            prefix = sorted(p-start for p in matched if start <= p < end)
            if (0 < len(prefix) <= 3 and prefix == list(range(len(prefix)))
                    and any(offsets[index-1] <= p < start for p in matched)
                    and any(p >= start+len(prefix) for row in cue_positions[cue_index+1:] for p in row)):
                candidates.append({'cue_id': cue.id, 'cue_text': cue.text, 'reference_next_id': events[index]['id'],
                    'prefix': reference[start:start+len(prefix)], 'reference_display_ms':
                    [events[index]['start_ms'], events[index]['end_ms']], 'independently_reviewed': False})
            suffix = sorted(p-offsets[index-1] for p in matched if offsets[index-1] <= p < start)
            previous_length = start-offsets[index-1]
            if (0 < len(suffix) <= 3 and len(suffix) < previous_length
                    and suffix == list(range(previous_length-len(suffix), previous_length))
                    and any(start <= p < end for p in matched)
                    and any(offsets[index-1] <= p < start-len(suffix) for row in cue_positions[:cue_index] for p in row)):
                suffix_candidates.append({'cue_id': cue.id, 'cue_text': cue.text, 'reference_previous_id': events[index-1]['id'],
                    'suffix': reference[start-len(suffix):start], 'independently_reviewed': False})
    return {'reference_characters': len(reference), 'hypothesis_characters': len(hypothesis),
        'character_edits': edits, 'mechanical_cer': edits/len(reference),
        'prefix_carry_candidates': candidates, 'unrated_reference_boundaries': unrateable,
        'previous_suffix_carry_candidates': suffix_candidates,
        'scope': 'user supplied original-language text; edit-path prefix candidates, not acoustic timing or semantic error rate',
        'acoustic_timing_accuracy': None}
