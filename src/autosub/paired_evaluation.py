"""Fixed cue pairs, with a strict distinction between suggestions and human gold."""
from __future__ import annotations

from collections import Counter
from statistics import median
import math

from .domain import Project


def annotation_template(project: Project, offset_ms=0) -> dict:
    return {"schema_version": 1, "source_fingerprint": project.source_fingerprint,
            "audio_stream": project.audio_stream, "offset_ms": offset_ms,
            "time_basis": "project-relative milliseconds; source offset stored separately",
            "split": "development", "records": [
                {"cue_id": c.id, "recognized_text": c.text, "suggested_base_ms":
                 [c.speech_start_ms, c.speech_end_ms], "original_text": "",
                 "speech_start_ms": None, "speech_end_ms": None,
                 "confirmed": False, "reviewer": "", "uncertain": False,
                 "overlap": False, "elongated": False, "notes": ""} for c in project.cues]}


def _errors(values):
    return {"median_ms": median(values) if values else None,
            "p90_ms": sorted(values)[math.ceil(len(values) * .9) - 1] if values else None}


def _distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def paired_report(project: Project, annotation: dict) -> dict:
    if annotation.get("schema_version") != 1:
        raise ValueError("不支援的配對標註版本")
    if annotation.get("source_fingerprint") != project.source_fingerprint or annotation.get("audio_stream") != project.audio_stream:
        raise ValueError("配對標註的來源或音軌不符")
    cues = {c.id: c for c in project.cues}
    excluded, accepted, seen = Counter(), [], set()
    for record in annotation["records"]:
        cue_id = record["cue_id"]
        if cue_id in seen:
            raise ValueError("配對標註包含重複 cue_id")
        seen.add(cue_id)
        cue = cues.get(cue_id)
        if record.get("confirmed") is not True or not record.get("reviewer", "").strip():
            excluded["unconfirmed"] += 1
            continue
        if record.get("uncertain"):
            excluded["uncertain"] += 1
            continue
        if cue is None or record.get("recognized_text") != cue.text:
            excluded["pair_or_text_changed"] += 1
            continue
        a, b = record.get("speech_start_ms"), record.get("speech_end_ms")
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= project.duration_ms
                or not record.get("original_text", "").strip()):
            excluded["invalid_acoustic_reference"] += 1
            continue
        accepted.append((cue, record))

    def metrics(pairs):
        lead, end_error, tail, cuts, long_tail, chars, edits = [], [], [], 0, [], 0, 0
        for cue, ref in pairs:
            lead.append(ref["speech_start_ms"] - cue.start_ms)
            end_error.append(abs(cue.speech_end_ms - ref["speech_end_ms"]))
            tail.append(cue.end_ms - ref["speech_end_ms"])
            cuts += cue.end_ms < ref["speech_end_ms"]
            if ref.get("elongated"):
                long_tail.append(abs(cue.end_ms - ref["speech_end_ms"]))
            truth, hypothesis = "".join(ref["original_text"].split()), "".join(cue.text.split())
            chars += len(truth)
            edits += _distance(truth, hypothesis)
        return {"samples": len(pairs), "start_0_to_50ms_rate": sum(0 <= x <= 50 for x in lead) / len(lead) if lead else None,
                "late_start_count": sum(x < 0 for x in lead),
                "late_start_error": _errors([-x for x in lead if x < 0]),
                "start_absolute_error": _errors([abs(x) for x in lead]),
                "base_end_absolute_error": _errors(end_error), "display_early_cut_count": cuts,
                "display_tail_ms": {"median": median(tail) if tail else None, "values": tail},
                "confirmed_long_end_absolute_error": _errors(long_tail),
                "cer_whitespace_removed": edits / chars if chars else None}
    counts = Counter(e["action"] for e in project.review_events)
    minutes = project.duration_ms / 60000
    return {"metric": "fixed original-language pairs; human-confirmed acoustic reference only",
            "review_activity": {"counts": dict(counts), "operations_per_media_minute":
                                {k: n / minutes for k, n in counts.items()} if minutes else {},
                                "human_review_elapsed_seconds": None},
            "excluded": dict(excluded), "all": metrics(accepted),
            "clear": metrics([p for p in accepted if not p[1].get("overlap")]),
            "overlap": metrics([p for p in accepted if p[1].get("overlap")])}
