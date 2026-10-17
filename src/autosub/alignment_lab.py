"""Pure contracts for isolated alignment experiments; never a default backend."""
import hashlib
import math
import unicodedata
from pathlib import Path

from .domain import Project


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if unicodedata.category(c)[0] in {"L", "N"})


def restore_unit_text(text, units):
    """Map model units back to exact source spelling/punctuation, without new times."""
    truth = normalized(text)
    if not truth or normalized("".join(u["text"] for u in units)) != truth:
        raise ValueError("Aligned units do not cover the input text")
    positions = {}
    for position in range(len(text) + 1):
        prefix = normalized(text[:position])
        if prefix == truth[:len(prefix)]:
            positions[len(prefix)] = position
    result, cursor, count = [], 0, 0
    for unit in units:
        length = len(normalized(unit["text"]))
        if not length:
            raise ValueError("Empty normalized alignment unit")
        count += length
        if count not in positions:
            raise ValueError("Alignment unit splits a source character")
        stop = positions[count]
        result.append({**unit, "text": text[cursor:stop]})
        cursor = stop
    if cursor != len(text) or "".join(u["text"] for u in result) != text:
        raise ValueError("Source text changed during unit mapping")
    return result


def atom_ownership(start_ms, end_ms, window_ms, owner_ms, duration_ms):
    """Keep whole lexical atoms; invalid unowned context must not poison a window."""
    if not owner_ms[0] <= (start_ms + end_ms) / 2 < owner_ms[1]:
        return "context"
    if not 0 <= window_ms[0] <= start_ms <= end_ms <= window_ms[1] <= duration_ms:
        return "invalid-owned"
    return "owned"


def _text_time_span(atoms, first, last):
    spans, cursor = [], 0
    for atom in atoms:
        for unit in atom.get("unit_evidence", [atom]):
            stop = cursor + len(normalized(unit["text"]))
            if cursor < last and stop > first:
                spans.append((unit["start_ms"], unit["end_ms"]))
            cursor = stop
    return (min(a for a, b in spans), max(b for a, b in spans)) if spans else None


def stitch_atoms(windows, radius_ms=2500, speech=()):
    """Choose complete lexical boundaries in overlapping ASR hypotheses.

    No text is rewritten, no word is split and no timestamp is synthesized.
    Independent hypotheses for the same acoustic interval cannot be interleaved.
    """
    starts, ends = [], []
    for window in windows:
        owned = [i for i, a in enumerate(window["atoms"]) if
                 window["owner_ms"][0] <= (a["start_ms"] + a["end_ms"]) / 2 < window["owner_ms"][1]]
        starts.append(min(owned) if owned else 0)
        ends.append(max(owned) + 1 if owned else 0)
    decisions = []
    for n, (left, right) in enumerate(zip(windows, windows[1:])):
        boundary = left["owner_ms"][1]
        lt = normalized("".join(a["text"] for a in left["atoms"]))
        rt = normalized("".join(a["text"] for a in right["atoms"]))
        overlap = 0
        for length in range(min(60, len(lt), len(rt)), 1, -1):
            if lt[-length:] != rt[:length]:
                continue
            a = _text_time_span(left["atoms"], len(lt) - length, len(lt))
            b = _text_time_span(right["atoms"], 0, length)
            if a and b and (min(a[1], b[1]) >= max(a[0], b[0]) or abs(a[0] - b[0]) <= 500):
                overlap = length
                break
        options = []
        for i in range(starts[n] + 1, len(left["atoms"]) + 1):
            a = left["atoms"][i - 1]
            if a["end_ms"] <= a["start_ms"]:
                continue
            if abs(a["end_ms"] - boundary) > radius_ms:
                continue
            for j, b in enumerate(right["atoms"]):
                if b["end_ms"] <= b["start_ms"]:
                    continue
                if j >= ends[n + 1] or abs(b["start_ms"] - boundary) > radius_ms:
                    continue
                gap = b["start_ms"] - a["end_ms"]
                if gap < -80:
                    continue
                score = abs(a["end_ms"] - boundary) + abs(b["start_ms"] - boundary) + 2 * abs(gap)
                active_gap = sum(max(0, min(e, b["start_ms"]) - max(s, a["end_ms"])) for s, e in speech)
                score += max(0, active_gap - 80) * 20
                if overlap:
                    kept_left = max(0, len(normalized("".join(x["text"] for x in left["atoms"][:i]))) - len(lt) + overlap)
                    kept_right = max(0, overlap - len(normalized("".join(x["text"] for x in right["atoms"][:j]))))
                    score += abs(kept_left + kept_right - overlap) * 2000
                # Edge-truncated hypotheses are less useful than words for which
                # the recognizer had context on both sides.
                score += 1000 * int(left["window_ms"][1] - a["end_ms"] < 100)
                score += 1000 * int(b["start_ms"] - right["window_ms"][0] < 100)
                options.append((score, i, j, gap, active_gap))
        if options:
            score, i, j, gap, active_gap = min(options)
            ends[n], starts[n + 1] = i, j
            decisions.append({"left": left["id"], "right": right["id"], "boundary_ms": boundary,
                              "left_end_ms": left["atoms"][i - 1]["end_ms"],
                              "right_start_ms": right["atoms"][j]["start_ms"],
                              "gap_ms": gap, "active_speech_gap_ms": active_gap,
                              "matched_overlap_characters": overlap, "status": "lexical-seam-candidate"})
        else:
            decisions.append({"left": left["id"], "right": right["id"], "boundary_ms": boundary,
                              "status": "unresolved-use-midpoint-ownership"})
    selected = [a for i, w in enumerate(windows) for a in w["atoms"][starts[i]:ends[i]]]
    return selected, decisions


def make_cases(project: Project, project_path: Path, padding_ms=1000):
    project.validate()
    if type(padding_ms) is not int or not 0 <= padding_ms <= 3000:
        raise ValueError("Invalid context padding")
    cases = []
    for cue in project.cues:
        start = max(0, cue.speech_start_ms - padding_ms)
        end = min(project.duration_ms, cue.speech_end_ms + padding_ms)
        if end - start > 30000:
            raise ValueError("Alignment experiment window exceeds 30s")
        cases.append({"id": cue.id, "text": cue.text, "language": project.language,
                      "window_ms": [start, end], "baseline_ms": [cue.speech_start_ms, cue.speech_end_ms]})
    return {"schema": "alignment-lab-v1", "source_project": str(Path(project_path).resolve()),
            "project_sha256": file_hash(project_path), "audio_path": project.audio_path,
            "audio_sha256": file_hash(project.audio_path), "duration_ms": project.duration_ms,
            "reference_text_used": False, "cases": cases}


def assess(case, result):
    """Reject text loss/invalid timing. A valid span is still an unverified candidate."""
    errors, warnings = [], []
    if result.get("id") != case["id"] or result.get("text") != case["text"]:
        errors.append("case-or-text-mismatch")
    units = result.get("units", [])
    if not units:
        errors.append("no-aligned-units")
    if normalized("".join(u.get("text", "") for u in units)) != normalized(case["text"]):
        errors.append("aligned-text-coverage-mismatch")
    duration = (case["window_ms"][1] - case["window_ms"][0]) / 1000
    previous = -1
    for unit in units:
        a, b = unit.get("start"), unit.get("end")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (a, b)):
            errors.append("missing-or-nonfinite-time")
            continue
        if not 0 <= a <= b <= duration + .001:
            errors.append("time-outside-input-window")
        if a < previous:
            errors.append("nonmonotonic-time")
        previous = a
        if a == b:
            warnings.append("zero-duration-unit")
    correction = result.get("timestamp_corrections", [])
    if any(item.get("changed_indices") for item in correction):
        warnings.append("backend-repaired-timestamps")
    if result.get("error"):
        errors.append("backend-error")
    span = None
    if not errors:
        a, b = units[0]["start"], max(u["end"] for u in units)
        if b <= a:
            errors.append("empty-aligned-span")
        else:
            if a < .05 or duration - b < .05:
                warnings.append("input-window-edge")
            span = [case["window_ms"][0] + round(a * 1000), case["window_ms"][0] + round(b * 1000)]
            if max(abs(x - y) for x, y in zip(span, case["baseline_ms"])) > 1000:
                warnings.append("boundary-shift-over-1000ms")
    return {"status": "failed" if errors else "review-required" if warnings else "candidate",
            "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings)),
            "span_ms": span, "human_verified": False}


def validate_result(manifest, result):
    if not manifest.get("manifest_sha256") or result.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Alignment result belongs to a different input manifest")
    cases = {c["id"]: c for c in manifest["cases"]}
    ids = [c["id"] for c in result["results"]]
    if len(ids) != len(set(ids)) or set(ids) != set(cases):
        raise ValueError("Alignment results have missing/duplicate cases")
    return {r["id"]: assess(cases[r["id"]], r) for r in result["results"]}
