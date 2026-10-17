"""Stable VAD utterance windows for experimental mixed-language recognition."""

MIXED_JOIN_GAP_MS = 750
MIXED_MAX_WINDOW_MS = 15000
MIXED_CONTEXT_MS = 250


def monolingual_windows(speech, duration_ms, *, target_ms=20000, maximum_ms=25000, context_ms=1000):
    """Experimental single-language ASR context, independent of subtitle cuts.

    Prefer an acoustic pause near 20s. Keep silence *inside* each input intact;
    ownership determines which aligned units survive overlapping context.
    """
    if not 6000 <= target_ms <= maximum_ms or context_ms < 0:
        raise ValueError("Invalid recognition context limits")
    regions = sorted((max(0, s), min(duration_ms, e)) for s, e in speech if max(0, s) < min(duration_ms, e))
    if not regions:
        return []
    candidates = [((a[1] + b[0]) // 2, b[0] - a[1]) for a, b in zip(regions, regions[1:])
                  if b[0] - a[1] >= 300]
    cuts, cursor = [0], regions[0][0]
    while regions[-1][1] - cursor > maximum_ms:
        options = [(center, gap) for center, gap in candidates if cursor + 6000 <= center <= cursor + maximum_ms]
        cut = max(options, key=lambda p: min(p[1], 2000) / 2 - abs(p[0] - cursor - target_ms))[0] if options else cursor + maximum_ms
        cuts.append(cut)
        cursor = cut
    cuts.append(duration_ms)
    windows = []
    for lo, hi in zip(cuts, cuts[1:]):
        local = [(s, e) for s, e in regions if s < hi and e > lo]
        if local:
            start = max(0, lo - context_ms, local[0][0] - context_ms)
            end = min(duration_ms, hi + context_ms, local[-1][1] + context_ms)
            windows.append((start, end, lo, hi))
    return windows


def mixed_windows(speech, duration_ms):
    """Return input and ownership bounds on the source clock, independent of checkpoints.

    Silence separates languages more reliably than choosing a language for 30 seconds.
    Long utterances still need bounded windows. Context may overlap; ownership cannot.
    """
    groups = []
    for start, end in sorted(speech):
        start, end = max(0, start), min(duration_ms, end)
        if start >= end:
            continue
        if groups and start - groups[-1][1] <= MIXED_JOIN_GAP_MS:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    windows = []
    for index, (start, end) in enumerate(groups):
        owner_start = (groups[index - 1][1] + start) // 2 if index else 0
        owner_end = (end + groups[index + 1][0]) // 2 if index + 1 < len(groups) else duration_ms
        for part in range(start, end, MIXED_MAX_WINDOW_MS):
            stop = min(end, part + MIXED_MAX_WINDOW_MS)
            windows.append((max(0, part - MIXED_CONTEXT_MS), min(duration_ms, stop + MIXED_CONTEXT_MS),
                            owner_start if part == start else part, owner_end if stop == end else stop))
    return windows
