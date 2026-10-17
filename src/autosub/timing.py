"""Pure timing rules. All values are milliseconds on the original media clock."""
from __future__ import annotations

import copy
import re
import hashlib
import json
from bisect import bisect_left, bisect_right
from typing import Callable

from .domain import Cue, Project, Word, word_key

END = re.compile(r"[。！？!?…][\"'」』）)]*$|(?<!\.)\.(?:[\"')])?$")
PAUSE = re.compile(r"[、，,;；:：]$")


def joined(words: list[Word]) -> str:
    # Whisper already supplies leading spaces for space-delimited languages.
    return "".join(w.text for w in words).strip()


def visible_length(text: str) -> int:
    return len(re.sub(r"\s", "", text))


class PauseIndex:
    """Index once per retime, instead of scanning every VAD region per word."""
    def __init__(self, speech):
        pairs = sorted(((a[1] + b[0]) / 2, b[0] - a[1])
                       for a, b in zip(speech, speech[1:]) if a[1] < b[0])
        self.centers = [center for center, _ in pairs]
        self.lengths = [length for _, length in pairs]

    def between(self, left: Word, right: Word) -> int:
        lo = (left.start_ms + left.end_ms) / 2
        hi = (right.start_ms + right.end_ms) / 2
        if hi <= lo:
            return 0
        first, last = bisect_left(self.centers, lo), bisect_right(self.centers, hi)
        return max(self.lengths[first:last], default=0)


def preferred_cut(project: Project, group: list[Word], pauses: PauseIndex) -> int:
    """Look back for a pause before a hard cap, instead of cutting mid-phrase."""
    options = []
    for index in range(1, len(group)):
        left, right = group[index - 1], group[index]
        duration = left.end_ms - group[0].start_ms
        if duration < max(500, project.settings.min_duration_ms):
            continue
        gap = max(0, right.start_ms - left.end_ms, pauses.between(left, right))
        punctuation = 500 if PAUSE.search(left.text.strip()) else 0
        # Require an actual cue to prefer an early boundary; otherwise use last word.
        if gap < 120 and not punctuation:
            continue
        score = min(gap, 1500) + punctuation - abs(duration - project.settings.soft_duration_ms) / 15
        options.append((score, index))
    return max(options)[1] if options else len(group)


def boundary_evidence(project: Project, group: list[Word]) -> dict:
    start, end = group[0].start_ms, max(w.end_ms for w in group)
    legacy = project.boundary_policy == "legacy"
    start_radius = project.settings.boundary_search_ms if legacy else project.settings.start_search_ms
    end_radius = project.settings.boundary_search_ms if legacy else project.settings.end_search_ms
    # Associate each edge with VAD speech overlapping that edge word. A nearby
    # unrelated utterance must never pull the cue across a silent gap.
    first, last = group[0], group[-1]
    start_candidates = [s for s, e in project.speech
                        if s < first.end_ms and e > first.start_ms and abs(s - start) <= start_radius]
    end_candidates = [e for s, e in project.speech
                      if s < last.end_ms and e > last.start_ms and abs(e - end) <= end_radius]
    proposed_start = min(start_candidates, key=lambda s: abs(s - start)) if start_candidates else start
    proposed_end = min(end_candidates, key=lambda e: abs(e - end)) if end_candidates else end
    return {"raw": [start, end], "vad": [proposed_start, proposed_end],
            "policy": project.boundary_policy}


def refine_boundaries(project: Project, group: list[Word]) -> tuple[int, int, list[str]]:
    evidence = boundary_evidence(project, group)
    start, end = evidence["raw"]
    proposed_start, proposed_end = evidence["vad"]
    flags = []
    if project.boundary_policy == "conservative" and proposed_end < end:
        proposed_end = end
        flags.append("VAD 建議裁短尾音，保留原始句尾待複核")
    if proposed_start < proposed_end:
        if proposed_start > start or proposed_end < end:
            flags.append("VAD 縮窄邊界，請確認輕聲／尾音")
        start, end = proposed_start, proposed_end
    if not any(s < end and e > start for s, e in project.speech):
        flags.append("未對應到 VAD 語音，請試聽")
    return start, end, flags


def segment_words(project: Project) -> list[Cue]:
    """Choose reliable word boundaries; never invent character-level times."""
    settings = project.settings
    settings.validate()
    words = sorted(project.words, key=lambda w: (w.start_ms, w.end_ms))
    pauses = PauseIndex(project.speech)
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if not word.text.strip():
            continue
        if current:
            duration = word.end_ms - current[0].start_ms
            gap = max(word.start_ms - current[-1].end_ms, pauses.between(current[-1], word))
            too_long = duration > settings.max_duration_ms
            too_many = visible_length(joined(current + [word])) > settings.max_chars
            soft_pause = (current[-1].end_ms - current[0].start_ms >= settings.soft_duration_ms
                          and (gap >= 150 or PAUSE.search(current[-1].text.strip())))
            if gap >= settings.silence_split_ms or soft_pause:
                groups.append(current)
                current = []
            elif too_long or too_many:
                cut = preferred_cut(project, current, pauses)
                groups.append(current[:cut])
                current = current[cut:]
                # A single oversized word/phrase can remain; do not fabricate times.
                if current and (word.end_ms - current[0].start_ms > settings.max_duration_ms or
                                visible_length(joined(current + [word])) > settings.max_chars):
                    groups.append(current)
                    current = []
        current.append(word)
        if END.search(word.text.strip()):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return cues_from_groups(project, groups)


def cues_from_groups(project: Project, groups: list[list[Word]]) -> list[Cue]:
    """Convert token groups to cues without changing recognized text or token times."""
    settings = project.settings
    result = []
    for group in groups:
        start, end, flags = refine_boundaries(project, group)
        if any(w.probability < 0.45 for w in group):
            flags.append("辨識信心偏低")
        if end - start > settings.max_duration_ms:
            flags.append("缺乏可靠切點，超過分段上限")
        if visible_length(joined(group)) > settings.max_chars:
            flags.append("單一詞組超過字數上限")
        end = min(project.duration_ms, max(start + 1, end))
        if end <= start:
            continue
        tokens = [word_key(w) for w in group]
        digest = hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest()[:16]
        result.append(Cue(f"cue-{digest}-{len(result)}", start, end, joined(group), start, end, flags,
                          source_tokens=tokens, timing_trace={**boundary_evidence(project, group),
                          "base": [start, end], "reasons": list(flags)}))
    return result


def apply_timing(project: Project, base: list[Cue] | None = None, *,
                 segmenter: Callable[[Project], list[Cue]] | None = None) -> list[Cue]:
    """Locked manual edits survive resegmentation. Speech coverage outranks overlap."""
    s = project.settings
    s.validate()
    if base is not None:
        cues = copy.deepcopy(base)
    else:
        locked = [copy.deepcopy(c) for c in project.cues if c.locked]
        remaining = copy.copy(project)
        # Filter owned words BEFORE regrouping: removing an intersecting generated
        # cue afterwards could silently drop unlocked words in that same cue.
        owned = {token for lock in locked for token in (lock.source_tokens or [])}
        legacy = [lock for lock in locked if lock.source_tokens is None]
        remaining.words = [w for w in project.words if word_key(w) not in owned and not any(
            lock.speech_start_ms <= (w.start_ms + w.end_ms) / 2 < lock.speech_end_ms for lock in legacy)]
        if segmenter is None and project.asr_strategy == "native-v1":
            from .native_segmentation import segment_native
            project.validate()
            native = [[project.words[i] for i in group] for group in project.native_groups]
            cues = segment_native(remaining, native)
        else:
            cues = (segmenter or segment_words)(remaining)
        for c in cues:
            c.elongated = any(old.elongated and old.speech_end_ms == c.speech_end_ms
                              for old in project.cues)
        cues.extend(locked)
    cues.sort(key=lambda c: (c.speech_start_ms, c.start_ms))
    for c in cues:
        if c.locked:
            continue
        c.start_ms = max(0, c.speech_start_ms - s.lead_ms)
        cap = min(project.duration_ms, c.speech_end_ms + (0 if c.elongated else s.max_tail_ms))
        target = max(c.speech_end_ms + (0 if c.elongated else s.tail_ms),
                     c.start_ms + s.min_duration_ms)
        c.end_ms = min(cap, target)
        if c.elongated:
            c.end_ms = c.speech_end_ms
    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    for index, c in enumerate(cues):
        next_cue = cues[index + 1] if index + 1 < len(cues) else None
        if next_cue:
            # Do not allow an automatically added lead-in to overlap existing speech.
            if not next_cue.locked and next_cue.start_ms < c.speech_end_ms <= next_cue.speech_start_ms:
                next_cue.start_ms = c.speech_end_ms
            if not c.locked:
                if c.speech_end_ms <= next_cue.start_ms:
                    c.end_ms = min(c.end_ms, next_cue.start_ms)
                    gap = next_cue.start_ms - c.end_ms
                    cap = min(project.duration_ms, c.speech_end_ms + s.max_tail_ms)
                    if not c.elongated and 0 < gap < s.close_gap_ms and next_cue.start_ms <= cap:
                        c.end_ms = next_cue.start_ms
                else:
                    c.end_ms = max(c.speech_end_ms, min(c.end_ms, next_cue.start_ms))
            if c.end_ms > next_cue.start_ms:
                c.flags.append("時間重疊，請複核邊界／多人發話")
                next_cue.flags.append("時間重疊，請複核邊界／多人發話")
        if c.end_ms - c.start_ms < s.min_duration_ms:
            c.flags.append("受邊界限制，未達最短時長")
        c.flags = list(dict.fromkeys(c.flags))
        if not c.locked:
            c.timing_trace["base"] = [c.speech_start_ms, c.speech_end_ms]
            c.timing_trace["display"] = [c.start_ms, c.end_ms]
    return cues
