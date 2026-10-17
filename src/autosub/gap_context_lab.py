"""Read-only internal-gap context proposals. Not used by the ASR pipeline."""
import math


def _clock(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 and value == round(value))


def _ranges(spans, duration):
    result = []
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not all(_clock(v) for v in span) or not 0 <= span[0] < span[1] <= duration
                or result and span[0] < result[-1][1]):
            raise ValueError('Expected ordered, disjoint source-clock millisecond ranges')
        result.append(list(map(int, span)))
    return result


def propose_internal_context(clips, edge_words, support, duration_ms, *,
                             tolerance_ms=50, max_extension_ms=5000):
    """Keep every original clip end/group and the first clip unchanged.

    edge_words maps an original clip index to its first observed word interval.
    A second detector must cover that whole word and cross the old clip start.
    Only an internal gap may be exposed; prior owned audio is never entered.
    These are geometry proposals, not evidence of missing text or correct ASR.
    """
    if (not _clock(duration_ms) or duration_ms <= 0 or not _clock(tolerance_ms)
            or not _clock(max_extension_ms) or max_extension_ms <= 0):
        raise ValueError('Invalid context bounds')
    original = _ranges(clips, duration_ms)
    support = _ranges(support, duration_ms)
    if not original or not isinstance(edge_words, dict):
        raise ValueError('Explicit original clips and observed edge words required')
    for index, word in edge_words.items():
        if type(index) is not int or not 0 <= index < len(original):
            raise ValueError('Unknown observed clip index')
        checked = _ranges([word], duration_ms)[0]
        if not original[index][0] <= checked[0] < checked[1] <= original[index][1]:
            raise ValueError('Observed word is outside its original clip')
    candidate = [span.copy() for span in original]
    decisions = []
    for index, (start, end) in enumerate(original):
        reason = None; proposed = start
        if index == 0:
            reason = 'first-clip-preserved'
        elif index not in edge_words:
            reason = 'no-observed-edge-word'
        else:
            word_start, word_end = edge_words[index]
            if word_start - start > tolerance_ms:
                reason = 'word-not-near-input-start'
            else:
                crossing = [a for a, b in support if a < start and b >= word_end]
                if not crossing:
                    reason = 'no-crossing-independent-support'
                else:
                    # One millisecond is clock separation, not acoustic precision.
                    proposed = max(original[index-1][1] + 1, start-max_extension_ms, crossing[0])
                    proposed = min(start, proposed)
                    reason = 'internal-context-preview' if proposed < start else 'no-unowned-extension'
        candidate[index][0] = int(proposed)
        decisions.append({'clip_index': index, 'original_ms': [start, end],
                          'candidate_ms': candidate[index].copy(), 'reason': reason})
    return {'policy': 'internal-context-lab-v1', 'original_clips_ms': original,
            'candidate_clips_ms': candidate, 'decisions': decisions,
            'changed_clip_indices': [i for i in range(len(original)) if original[i] != candidate[i]],
            'production_adoption': False, 'confirmed_missing_text': False}
