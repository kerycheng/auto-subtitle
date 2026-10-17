"""Manual review operations, independent of Qt and ASR."""
from __future__ import annotations

import copy
import uuid

from .domain import Cue, Project, word_key
from .timing import visible_length


def owned_tokens(project: Project, cue: Cue) -> list[str]:
    if cue.source_tokens is not None:
        return list(cue.source_tokens)
    return [word_key(w) for w in project.words
            if cue.speech_start_ms <= (w.start_ms + w.end_ms) / 2 < cue.speech_end_ms]


def split_cue(project: Project, index: int, at_ms: int, left_text: str, right_text: str) -> list[Cue]:
    original = project.cues[index]
    if not original.start_ms < at_ms < original.end_ms:
        raise ValueError("切點必須位於字幕開始與結束之間")
    if not original.speech_start_ms < at_ms < original.speech_end_ms:
        raise ValueError("切點須位於估計發聲區間內；請先調整字幕或選擇其他切點")
    if not left_text.strip() or not right_text.strip():
        raise ValueError("前後兩句都必須有文字")
    tokens = owned_tokens(project, original)
    token_set = set(tokens)
    left_tokens = {word_key(w) for w in project.words if word_key(w) in token_set
                   and (w.start_ms + w.end_ms) / 2 < at_ms}
    left = copy.deepcopy(original)
    right = copy.deepcopy(original)
    left.id, right.id = uuid.uuid4().hex, uuid.uuid4().hex
    left.edit_group = right.edit_group = original.edit_group or uuid.uuid4().hex
    left.end_ms, left.speech_end_ms = at_ms, at_ms
    right.start_ms, right.speech_start_ms = at_ms, at_ms
    left.text, right.text = left_text.strip(), right_text.strip()
    left.source_tokens = [token for token in tokens if token in left_tokens]
    right.source_tokens = [token for token in tokens if token not in left_tokens]
    left.locked = right.locked = True
    left.reviewed = right.reviewed = False
    left.elongated = False
    for cue in (left, right):
        cue.candidate = cue.candidate_original = None
        cue.timing_trace = {"manual": "split", "base": [cue.speech_start_ms, cue.speech_end_ms],
                            "display": [cue.start_ms, cue.end_ms], "parent": original.id}
    result = copy.deepcopy(project.cues)
    result[index:index + 1] = [left, right]
    for cue in result:
        if cue.edit_group == left.edit_group:
            cue.locked = True
    return result


def merge_cues(project: Project, index: int) -> list[Cue]:
    if not 0 <= index < len(project.cues) - 1:
        raise ValueError("請選取一則後面還有字幕的項目")
    first, second = project.cues[index:index + 2]
    if second.start_ms < first.start_ms:
        raise ValueError("兩句時間順序相反，請先調整時間")
    merged = copy.deepcopy(first)
    merged.id = uuid.uuid4().hex
    merged.end_ms = max(first.end_ms, second.end_ms)
    merged.speech_start_ms = min(first.speech_start_ms, second.speech_start_ms)
    merged.speech_end_ms = max(first.speech_end_ms, second.speech_end_ms)
    separator = " " if project.language in {"en", "ko"} else ""
    merged.text = first.text.rstrip() + separator + second.text.lstrip()
    merged.source_tokens = list(dict.fromkeys(owned_tokens(project, first) + owned_tokens(project, second)))
    merged.flags = list(dict.fromkeys(first.flags + second.flags))
    merged.locked = True
    merged.reviewed = False
    merged.elongated = second.elongated
    merged.candidate = merged.candidate_original = None
    merged.timing_trace = {"manual": "merge", "parents": [first.id, second.id],
                           "base": [merged.speech_start_ms, merged.speech_end_ms],
                           "display": [merged.start_ms, merged.end_ms]}
    result = copy.deepcopy(project.cues)
    result[index:index + 2] = [merged]
    groups = {cue.edit_group for cue in (first, second) if cue.edit_group is not None}
    if groups:
        merged.edit_group = uuid.uuid4().hex
        for cue in result:
            if cue.edit_group in groups:
                cue.edit_group = merged.edit_group
                cue.locked = True
    return result


def set_lock_state(project: Project, index: int, locked: bool) -> list[Cue]:
    """Split fragments must regenerate together, especially for a cut within a word.

    Otherwise unlocking one fragment can regenerate the whole source word while
    retaining the other fragment, duplicating text. Whole edit groups toggle together.
    """
    selected = project.cues[index]
    result = copy.deepcopy(project.cues)
    for i, cue in enumerate(result):
        if i == index or (selected.edit_group is not None and cue.edit_group == selected.edit_group):
            cue.locked = locked
            if not locked:
                cue.reviewed = False
    return result


def review_cues(project: Project) -> None:
    """Recompute timing warnings after manual edits, including non-adjacent overlaps."""
    dynamic = {"受邊界限制，未達最短時長", "時間重疊，請複核邊界／多人發話",
               "顯示超過分段上限", "字幕超過字數上限", "起點晚於估計語音",
               "終點早於估計語音", "句首先行超過 50ms", "尾留超過上限"}
    previous = {cue.id: set(cue.flags) for cue in project.cues}
    for cue in project.cues:
        cue.flags = [f for f in cue.flags if f not in dynamic]
        if cue.end_ms - cue.start_ms < project.settings.min_duration_ms:
            cue.flags.append("受邊界限制，未達最短時長")
        if cue.end_ms - cue.start_ms > project.settings.max_duration_ms:
            cue.flags.append("顯示超過分段上限")
        if visible_length(cue.text) > project.settings.max_chars:
            cue.flags.append("字幕超過字數上限")
        if cue.start_ms > cue.speech_start_ms:
            cue.flags.append("起點晚於估計語音")
        if cue.end_ms < cue.speech_end_ms:
            cue.flags.append("終點早於估計語音")
        if cue.speech_start_ms - cue.start_ms > 50:
            cue.flags.append("句首先行超過 50ms")
        if cue.end_ms - cue.speech_end_ms > project.settings.max_tail_ms:
            cue.flags.append("尾留超過上限")
    active = []
    for cue in sorted(project.cues, key=lambda c: c.start_ms):
        active = [other for other in active if other.end_ms > cue.start_ms]
        for other in active:
            other.flags.append("時間重疊，請複核邊界／多人發話")
            cue.flags.append("時間重疊，請複核邊界／多人發話")
        active.append(cue)
    for cue in project.cues:
        cue.flags = list(dict.fromkeys(cue.flags))
        if set(cue.flags) - previous.get(cue.id, set()):
            cue.reviewed = False
