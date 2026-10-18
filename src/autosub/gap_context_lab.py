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


def propose_short_internal_context(clips, edge_words, support, duration_ms, *,
                                   max_clip_ms, tolerance_ms=50,
                                   max_extension_ms=5000):
    """Research-only exposure cap; rejected extensions keep the original clip.

    The caller supplies the pinned decoder's input-window duration. This limits
    the entire proposed clip, not just added audio, and does not guarantee one
    decoder call or correct transcription. Never shorten a rejected proposal
    to make it fit: that would create an unsupported new input boundary.
    """
    if not _clock(max_clip_ms) or max_clip_ms <= 0:
        raise ValueError('A positive integer-clock clip limit is required')
    result = propose_internal_context(
        clips, edge_words, support, duration_ms,
        tolerance_ms=tolerance_ms, max_extension_ms=max_extension_ms)
    for index in result['changed_clip_indices']:
        start, end = result['candidate_clips_ms'][index]
        if end - start > max_clip_ms:
            original = result['original_clips_ms'][index].copy()
            result['candidate_clips_ms'][index] = original
            result['decisions'][index].update(
                candidate_ms=original.copy(), reason='proposed-clip-exceeds-limit')
    result['changed_clip_indices'] = [
        i for i, (old, new) in enumerate(zip(result['original_clips_ms'],
                                           result['candidate_clips_ms'])) if old != new]
    result.update(policy='short-internal-context-lab-v1', max_clip_ms=int(max_clip_ms))
    return result


def observed_clip_edges(clips, segments, duration_ms, *, frames_per_second):
    """Associate emitted segments by SDK seek frame before validating words.

    An invalid first positive-duration word excludes the edge. Never filter
    words into a requested time range and then choose a later word. Requested
    source clocks remain separate from the SDK's rounded decoder frame grid.
    """
    if (not _clock(duration_ms) or duration_ms <= 0
            or isinstance(frames_per_second, bool)
            or not isinstance(frames_per_second, (int, float))
            or not math.isfinite(frames_per_second) or frames_per_second <= 0):
        raise ValueError('Valid duration and decoder frame rate are required')
    original = _ranges(clips, duration_ms)
    frames = [[round(v / 1000 * frames_per_second) for v in span] for span in original]
    if not frames or any(a >= b or i and a < frames[i-1][1]
                         for i, (a, b) in enumerate(frames)):
        raise ValueError('Requested clips collapse or overlap on the decoder grid')
    edges, excluded, seen, previous = {}, [], set(), -1
    for segment in segments:
        seek = segment['seek']
        if not _clock(seek) or seek < previous:
            raise ValueError('Invalid or decreasing decoder seek')
        previous = seek
        owners = [i for i, (a, b) in enumerate(frames) if a <= seek < b]
        if len(owners) != 1:
            raise ValueError('Segment seek has no unique requested clip')
        index = owners[0]
        if index in seen:
            continue
        for word in segment.get('words') or []:
            a, b = word['start'], word['end']
            if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in (a, b)) or a > b):
                raise ValueError('Invalid observed word time')
            if a == b:
                continue
            seen.add(index)
            interval = [round(a * 1000), round(b * 1000)]
            start, end = original[index]
            if not start <= interval[0] < interval[1] <= end:
                excluded.append({'clip_index': index, 'first_word_ms': interval,
                                 'reason': 'first-positive-word-outside-raw-clip'})
            else:
                edges[index] = interval
            break
    return {'edge_words': edges, 'excluded': excluded, 'effective_clip_frames': frames,
            'frames_per_second': frames_per_second}
