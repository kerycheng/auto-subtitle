"""Preserve Whisper's sentence boundaries; only split oversized sentences.

Native groups have explicit provenance; legacy projects retain their strategy.
"""
from collections import Counter

from .domain import word_key
from .lexical_segmentation import boundary_map
from .timing import PAUSE, cues_from_groups, joined, visible_length


def segment_native(project, native_groups):
    if project.language != "ja":
        raise ValueError("原生句界策略目前支援日文")
    project.settings.validate()
    stream = [w for group in native_groups for w in group if w.text.strip()]
    bm = boundary_map(stream)
    safe_words = set()
    token_offsets = set(bm.offsets)
    partial = {}
    for i, offset in enumerate(bm.offsets):
        crossed = [m for m in bm.lexemes if m["start"] < offset < m["end"]]
        gap = stream[i].start_ms-stream[i-1].end_ms if 0 < i < len(stream) else 0
        # Unknown dictionary compounds can span two genuine sentences when ASR
        # omitted punctuation. Native evidence + a pause outranks that OOV guess.
        # A dictionary parse may fuse the previous ending with only part of a
        # complete next token (な | いや -> ない + や). Moving the cut then damages
        # the native sentence. Repair only lexemes fully bounded by source tokens.
        partial_crossing = [m for m in crossed if m["start"] not in token_offsets or m["end"] not in token_offsets]
        if partial_crossing:
            partial[i] = [m["text"] for m in partial_crossing]
        if not crossed or partial_crossing or (all(m["oov"] for m in crossed) and gap >= project.settings.silence_split_ms):
            safe_words.add(i)
    cuts, cursor = {0, len(stream)}, 0
    moved = []
    retained = []
    for group in native_groups[:-1]:
        cursor += sum(bool(w.text.strip()) for w in group)
        if cursor in safe_words:
            cuts.add(cursor)
            if cursor in partial:
                retained.append({"boundary_token": cursor, "dictionary_text": partial[cursor],
                    "reason": "dictionary-lexeme-only-partially-covers-source-token"})
        else:
            cut = max(i for i in safe_words if i < cursor)
            cuts.add(cut)
            moved.append({"from_token": cursor, "to_token": cut})
    available = Counter(word_key(w) for w in project.words if w.text.strip())
    groups = []
    cuts = sorted(cuts)
    for a, b in zip(cuts, cuts[1:]):
        current = []
        for word in stream[a:b]:
            key = word_key(word)
            if available[key]:
                available[key] -= 1
                current.append(word)
            elif current:
                # A manual lock removed part of this sentence. Do not bridge it.
                groups.append(current)
                current = []
        if current:
            groups.append(current)
    if any(available.values()):
        raise ValueError("Native groups do not cover the source tokens")
    result_groups = []
    settings = project.settings
    for original in groups:
        rest = original
        while rest:
            duration = max(w.end_ms for w in rest) - rest[0].start_ms
            if duration <= settings.max_duration_ms and visible_length(joined(rest)) <= settings.max_chars:
                result_groups.append(rest)
                break
            mapping = boundary_map(rest)
            preferred = mapping.allowed & mapping.phrases
            options = [i for i in range(1, len(rest)) if i in mapping.allowed and
                       max(w.end_ms for w in rest[:i]) - rest[0].start_ms <= settings.max_duration_ms and
                       visible_length(joined(rest[:i])) <= settings.max_chars]
            if not options:
                cut = min((i for i in mapping.allowed if 0 < i < len(rest)), default=len(rest))
            else:
                phrase_options = [i for i in options if i in preferred]
                def score(i):
                    elapsed = rest[i - 1].end_ms - rest[0].start_ms
                    gap = max(0, rest[i].start_ms - rest[i - 1].end_ms)
                    return min(gap, 1000) + 500 * bool(PAUSE.search(rest[i - 1].text.strip())) - abs(elapsed - settings.soft_duration_ms) / 10
                cut = max(phrase_options or options, key=score)
            result_groups.append(rest[:cut])
            rest = rest[cut:]
    cues = cues_from_groups(project, result_groups, separate_vad=True)
    for cue in cues:
        cue.timing_trace["segmentation"] = "native-sentence-v1"
        cue.timing_trace["native_lexical_repairs"] = moved
        if retained:
            cue.timing_trace["native_partial_lexemes_retained"] = retained
    return cues
