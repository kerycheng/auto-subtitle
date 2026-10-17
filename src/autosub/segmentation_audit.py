"""Mechanical segmentation checks; never a substitute for listening or gold text."""
from collections import Counter

from .domain import word_key
from .lexical_segmentation import boundary_map


def uncovered_speech(project, minimum_ms=700):
    """Source-clock VAD intervals not represented by positive-duration ASR words.

    A timestamp gap can also be bad alignment, noise or nonverbal speech. It is
    only an audition target, and text omissions inside a long word remain hidden.
    """
    words = sorted((w.start_ms, w.end_ms) for w in project.words if w.end_ms > w.start_ms)
    spans = []
    for start, end in sorted(project.speech):
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    missing = []
    for start, end in spans:
        cursor = start
        for a, b in words:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a-cursor >= minimum_ms:
                missing.append([cursor, min(a,end)])
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if end-cursor >= minimum_ms:
            missing.append([cursor,end])
    return missing


def audit(project):
    mapping = boundary_map(project.words)
    positions = {}
    for index, word in enumerate(project.words):
        positions.setdefault(word_key(word), []).append(index+1)
    cuts, ambiguous = [], []
    for cue in project.cues:
        if not cue.source_tokens:
            ambiguous.append(cue.id)
            continue
        ids = positions.get(cue.source_tokens[-1], [])
        if len(ids) != 1:
            ambiguous.append(cue.id)
            continue
        i = ids[0]
        if i == len(project.words):
            continue
        offset = mapping.offsets[i]
        for lexeme in mapping.lexemes:
            if lexeme['start'] < offset < lexeme['end']:
                cuts.append({'cue_id':cue.id,'text':cue.text,'next_text':''.join(w.text for w in project.words[i:i+8]),
                             'lexeme':lexeme['text'],'oov':lexeme['oov'],'source_token':i})
    expected = Counter(map(word_key,project.words))
    actual = Counter(token for cue in project.cues for token in cue.source_tokens or [])
    active, overlaps = [], []
    for cue in sorted(project.cues,key=lambda c:(c.start_ms,c.end_ms)):
        active = [other for other in active if other.end_ms > cue.start_ms]
        overlaps.extend([other.id,cue.id] for other in active)
        active.append(cue)
    return {'words':len(project.words),'cues':len(project.cues),
        'source_tokens_preserved':actual == expected,
        'duplicate_source_keys':sum(count-1 for count in expected.values()),
        'ambiguous_cut_cues':ambiguous,
        'known_word_internal_cuts':sum(not x['oov'] for x in cuts),
        'oov_internal_cuts':sum(x['oov'] for x in cuts),'cuts':cuts,
        'overlap_pairs':overlaps,
        'over_duration_cues':[c.id for c in project.cues if c.end_ms-c.start_ms>project.settings.max_duration_ms],
        'uncovered_vad_ms':uncovered_speech(project),
        'low_confidence_words':[{'start_ms':w.start_ms,'end_ms':w.end_ms,'text':w.text,'probability':w.probability}
                                for w in project.words if w.probability<.45],
        'human_completion_rate':None,'acoustic_timing_accuracy':None}
