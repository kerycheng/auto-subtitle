"""Read-only reference evaluation. Translated text is never fed to recognition.

Nearest-boundary coverage is a segmentation-sensitive proxy, NOT aligned accuracy.
It is intentionally reported in both directions and is not called 50ms accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .domain import Cue


@dataclass
class ReferenceCue:
    start_ms: int
    end_ms: int
    text: str
    style: str


def parse_ass_time(value: str) -> int:
    hours, minutes, seconds = value.strip().split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def read_ass(path: Path, exclude_styles=("註解", "注释", "注釋")) -> tuple[list[ReferenceCue], str]:
    text = path.read_text(encoding="utf-8-sig")
    fields = []
    result = []
    video = ""
    in_events = False
    for line in text.splitlines():
        if line.startswith("Video File:"):
            video = line.split(":", 1)[1].strip()
        if line.startswith("["):
            in_events = line.strip() == "[Events]"
        if not in_events:
            continue
        if line.startswith("Format:"):
            fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
        if not line.startswith("Dialogue:") or not fields:
            continue
        values = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
        if len(values) != len(fields):
            continue
        event = dict(zip(fields, values))
        style = event.get("style", "")
        raw = event.get("text", "")
        content = re.sub(r"\{[^}]*\}", "", raw).replace(r"\N", " ").strip()
        if style in exclude_styles or not content or re.search(r"\\p[1-9]", raw):
            continue
        start, end = parse_ass_time(event["start"]), parse_ass_time(event["end"])
        if end > start:
            result.append(ReferenceCue(start, end, content, style))
    return sorted(result, key=lambda c: c.start_ms), video


def boundary_report(reference: list[ReferenceCue], generated: list[Cue], start_ms=0, end_ms=None) -> dict:
    if end_ms is None:
        end_ms = max((c.end_ms for c in reference), default=0)
    # Exclude boundaries clipped by evaluation window; never invent edge matches.
    result = {"window_ms": [start_ms, end_ms], "reference_cues": sum(
        c.start_ms < end_ms and c.end_ms > start_ms for c in reference),
        "generated_cues": sum(c.start_ms < end_ms and c.end_ms > start_ms for c in generated),
        "metric": "nearest-boundary coverage; not text-aligned accuracy",
        "reference_overlap_pairs": sum(
            a.start_ms < b.end_ms and b.start_ms < a.end_ms
            for i, a in enumerate(reference) for b in reference[i + 1:]
            if a.start_ms < end_ms and b.start_ms < end_ms and a.end_ms > start_ms and b.end_ms > start_ms)}
    for kind in ["start_ms", "end_ms"]:
        ref = [getattr(c, kind) for c in reference if start_ms < getattr(c, kind) < end_ms]
        gen = [getattr(c, kind) for c in generated if start_ms < getattr(c, kind) < end_ms]
        # Maximum-cardinality ordered matching. A dense output cannot reuse one
        # generated boundary to claim multiple matches, unlike nearest coverage.
        exclusive = {}
        for tolerance in [50, 100, 250, 500]:
            left, right = sorted(ref), sorted(gen)
            i = j = matches = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j] - tolerance:
                    i += 1
                elif right[j] < left[i] - tolerance:
                    j += 1
                else:
                    matches += 1
                    i += 1
                    j += 1
            precision = matches / len(gen) if gen else 0.0
            recall = matches / len(ref) if ref else 0.0
            exclusive[f"{tolerance}ms"] = {
                "matches": matches, "precision": precision, "recall": recall,
                "f1": 2 * matches / (len(ref) + len(gen)) if ref or gen else 0.0,
            }
        result[f"{kind}_one_to_one"] = exclusive
        for label, source, other in [("reference_to_generated", ref, gen), ("generated_to_reference", gen, ref)]:
            errors = [min(abs(a - b) for b in other) for a in source] if other else []
            result[f"{kind}_{label}"] = {
                "boundaries": len(source), "matched_candidates": len(errors),
                "median_nearest_ms": median(errors) if errors else None,
                **{f"within_{threshold}ms": (sum(e <= threshold for e in errors) / len(source)
                                               if source else None) for threshold in [50, 100, 250, 500]},
            }
    return result
