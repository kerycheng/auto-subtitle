"""Experimental Japanese segmentation, evaluated separately from app defaults.

Lexical boundaries are constraints; phrase boundaries and acoustic pauses are
evidence. No model transcript, token timestamp or manual lock is rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .domain import Project, Word
from .timing import END, PAUSE, PauseIndex, cues_from_groups, joined, visible_length

STRATEGY = "ja-lexical-v1"


@lru_cache(maxsize=1)
def _models():
    import budoux
    from sudachipy import Dictionary, SplitMode
    return Dictionary(dict="core").tokenizer(mode=SplitMode.C), budoux.load_default_japanese_parser()


@dataclass
class BoundaryMap:
    words: list[Word]
    offsets: list[int]
    allowed: set[int]
    phrases: set[int]
    lexemes: list[dict]


def boundary_map(words: list[Word]) -> BoundaryMap:
    """Intersect linguistic cuts with *existing* Whisper token boundaries.

    Long acoustic gaps are separate analysis islands so a dictionary OOV token
    cannot join unrelated utterances. Punctuation terminates an island as well.
    """
    tokenizer, parser = _models()
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word.text))
    allowed, phrases, lexemes = {0, len(words)}, set(), []
    islands, start = [], 0
    for i in range(1, len(words)):
        if words[i].start_ms - words[i - 1].end_ms >= 1500 or END.search(words[i - 1].text.strip()):
            islands.append((start, i))
            start = i
    islands.append((start, len(words)))
    for lo, hi in islands:
        text = "".join(w.text for w in words[lo:hi])
        if not text:
            continue
        # Sudachi limits UTF-8 input size. Bound analysis windows, never split a
        # Whisper token merely to satisfy a model's input limit.
        if len(text.encode("utf-8")) > 45000:
            raise ValueError("日文分段分析區間過長，請先分成較短專案")
        morphemes = list(tokenizer.tokenize(text))
        local = {0, len(text)}
        for j in range(1, len(morphemes)):
            left, right = morphemes[j - 1], morphemes[j]
            lp, rp = left.part_of_speech(), right.part_of_speech()
            # Keep particles, inflections, suffixes and counters on their head.
            if rp[0] in {"助詞", "助動詞", "接尾辞", "補助記号", "空白"} or lp[0] == "接頭辞":
                continue
            if lp[1] == "数詞" and (rp[1] == "数詞" or rp[2] == "助数詞可能"):
                continue
            if rp[:2] == ("動詞", "非自立可能") and (lp[0] == "代名詞" or lp[2] == "サ変可能"):
                continue
            local.add(right.begin())
        phrase_offsets, offset = set(), 0
        for phrase in parser.parse(text):
            phrase_offsets.add(offset)
            offset += len(phrase)
        if offset != len(text):
            raise ValueError("短語分析改變原文，已停止生成候選")
        for i in range(lo, hi + 1):
            position = offsets[i] - offsets[lo]
            if position in local:
                allowed.add(i)
                if position in phrase_offsets:
                    phrases.add(i)
        for m in morphemes:
            lexemes.append({"start": offsets[lo] + m.begin(), "end": offsets[lo] + m.end(),
                            "text": m.surface(), "pos": list(m.part_of_speech()), "oov": m.is_oov()})
        allowed.update((lo, hi))
    return BoundaryMap(words, offsets, allowed, phrases, lexemes)


def lexical_groups(project: Project):
    if project.language != "ja":
        raise ValueError("完整詞語分段候選目前只評估日文；請勿套用其他語言")
    project.settings.validate()
    words = [w for w in sorted(project.words, key=lambda w: (w.start_ms, w.end_ms)) if w.text.strip()]
    mapping = boundary_map(words)
    if not words:
        return [], [], mapping
    settings, pauses = project.settings, PauseIndex(project.speech)
    cuts, decisions = {0, len(words)}, []
    safe = sorted(mapping.allowed)
    for i in range(1, len(words)):
        gap = max(words[i].start_ms - words[i - 1].end_ms, pauses.between(words[i - 1], words[i]))
        if END.search(words[i - 1].text.strip()):
            cuts.add(i)
        elif gap >= settings.silence_split_ms:
            if i in mapping.allowed:
                chosen = i
            else:
                # An onset before a pause must travel WITH its complete lexical
                # unit, not leave a prefix in the preceding cue. Do not snap
                # forward past the rest of that word.
                previous = max((j for j in safe if j < i), default=0)
                chosen = previous if (words[i].start_ms - words[previous].start_ms <= 2000
                                      and visible_length(joined(words[previous:i])) <= 16) else None
            if chosen is not None:
                cuts.add(chosen)
            decisions.append({"proposed_token": i, "chosen_token": chosen,
                              "pause_ms": gap, "moved": chosen != i,
                              "reason": "pause" if chosen == i else "protect-word-and-attached-particles"})
    # Enforce reading limits only at protected boundaries. Oversized indivisible
    # spans remain intact and are flagged by cues_from_groups.
    anchors = sorted(cuts)
    for lo, hi in zip(anchors, anchors[1:]):
        start = lo
        while start < hi:
            exceeds = (words[hi - 1].end_ms - words[start].start_ms > settings.max_duration_ms or
                       visible_length(joined(words[start:hi])) > settings.max_chars)
            if not exceeds:
                break
            options = [j for j in safe if start < j < hi and
                       words[j - 1].end_ms - words[start].start_ms <= settings.max_duration_ms and
                       visible_length(joined(words[start:j])) <= settings.max_chars]
            if not options:
                # The first lexical atom itself exceeds a limit. Keep it whole,
                # then continue splitting later material if possible.
                options = [j for j in safe if start < j < hi][:1]
                if not options:
                    break
            def score(j):
                duration = words[j - 1].end_ms - words[start].start_ms
                gap = max(0, words[j].start_ms - words[j - 1].end_ms)
                return (int(j in mapping.phrases) * 350 + min(gap, 700) +
                        int(bool(PAUSE.search(words[j - 1].text.strip()))) * 500 -
                        abs(duration - settings.soft_duration_ms) / 10)
            chosen = max(options, key=score)
            cuts.add(chosen)
            decisions.append({"proposed_token": None, "chosen_token": chosen,
                              "reason": "reading-limit-at-lexical-boundary", "moved": False})
            start = chosen
    cuts = sorted(cuts)
    return [words[a:b] for a, b in zip(cuts, cuts[1:])], decisions, mapping


def segment_lexically(project: Project):
    groups, decisions, mapping = lexical_groups(project)
    cues = cues_from_groups(project, groups)
    for i, cue in enumerate(cues):
        raw_start = groups[i][0].start_ms
        raw_end = max(w.end_ms for w in groups[i])
        # Moving a cut back can expose a VAD edge belonging to its neighbour.
        # Clamp only such VAD expansion, never crop overlapping raw word times.
        if i and (previous_end := max(w.end_ms for w in groups[i - 1])) <= raw_start:
            cue.speech_start_ms = max(cue.speech_start_ms, previous_end)
        if i + 1 < len(groups) and raw_end <= (next_start := groups[i + 1][0].start_ms):
            cue.speech_end_ms = min(cue.speech_end_ms, next_start)
        cue.start_ms, cue.end_ms = cue.speech_start_ms, cue.speech_end_ms
        cue.timing_trace["segmentation"] = STRATEGY
        cue.flags.append("日文完整詞語分段候選，請複核語意與發聲時間")
    # Annotate affected cuts rather than pretending the proposed character times
    # are newly aligned speech times.
    for decision in decisions:
        if decision["moved"]:
            chosen = decision["chosen_token"]
            if chosen is not None and chosen < len(mapping.words):
                start = mapping.words[chosen].start_ms
                for group, cue in zip(groups, cues):
                    if group[0].start_ms == start:
                        cue.timing_trace.setdefault("segmentation_decisions", []).append(decision)
                        cue.flags.append("停頓切點移至完整詞語之前；原始詞級時間尚未重對齊")
                        break
    return cues
