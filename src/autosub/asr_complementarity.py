"""Post-recognition paired text evidence, without selecting or editing subtitles.

Input fingerprints are supplied by an external, audited adapter. This module
checks their equality, not whether a decoder actually consumed those samples.
"""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import html
import json
import re

from .asr_comparison import character_alignment
from .language_validation import normalized_text


def _identity(run):
    fingerprint = run.get("input", {})
    if (run.get("status") != "complete" or not isinstance(run.get("text"), str)
            or not isinstance(fingerprint, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint.get("samples_sha256", "")))
            or type(fingerprint.get("sample_rate")) is not int
            or fingerprint["sample_rate"] <= 0
            or type(fingerprint.get("sample_count")) is not int
            or fingerprint["sample_count"] <= 0
            or fingerprint.get("language") not in {"en", "zh", "ja", "ko"}):
        return None
    return tuple(fingerprint[k] for k in
                 ("samples_sha256", "sample_rate", "sample_count", "language"))


def assess_case(case):
    """Measure whole-window oracle potential only; never build a fused transcript.

    An oracle is allowed to read the reference and choose one COMPLETE window.
    It is an optimistic diagnostic, not an implementable blind selection rule.
    """
    result = deepcopy(case)
    primary, auxiliary = case["primary"], case["auxiliary"]
    left, right = _identity(primary), _identity(auxiliary)
    result.update(status="unpaired", scores=None, differences=None)
    if left is None or right is None or left != right or left[-1] != case["language"]:
        result["reason"] = "Incomplete run or missing/mismatched input fingerprint"
        return result
    a, b = primary["text"], auxiliary["text"]
    # Bound the exact edit grid and descriptive diff before allocating either.
    if max(len(a), len(b)) > 10000:
        result.update(status="unrated", reason="Text exceeds short-window analysis limit")
        return result
    result.update(status="paired-unrated", reason="Complete reference unavailable")
    result["raw_text_equal"] = a == b
    result["normalized_text_equal"] = normalized_text(a) == normalized_text(b)
    result["differences"] = [
        {"kind": kind, "primary_offsets": [i, j], "auxiliary_offsets": [k, l],
         "primary_text": a[i:j], "auxiliary_text": b[k:l]}
        for kind, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        if kind != "equal"
    ]
    reference = case.get("reference") or {}
    scope = reference.get("scope")
    if (scope not in {"dataset-complete", "human-complete"}
            or (scope == "human-complete" and reference.get("reviewer_confirmed") is not True)
            or reference.get("uncertain") is not False
            or not isinstance(reference.get("text"), str)):
        return result
    ref = normalized_text(reference["text"])
    if not ref:
        result["reason"] = "Empty normalized reference is not a rated transcript"
        return result
    if (len(ref) > 10000 or max(len(ref) * len(normalized_text(t)) for t in (a, b)) > 1_000_000):
        result["reason"] = "Reference exceeds exact-grid analysis limit"
        return result
    p, q = (character_alignment(reference["text"], t) for t in (a, b))
    pe, qe = p["character_edits"], q["character_edits"]
    result.update(status="paired-rated", reason=None)
    result["scores"] = {
        "reference_characters": len(ref), "primary": p, "auxiliary": q,
        "whole_window_oracle_edits": min(pe, qe),
        "potential_edits_saved": max(0, pe - qe),
        "always_auxiliary_extra_edits": max(0, qe - pe),
        "primary_exact_auxiliary_not": pe == 0 and qe > 0,
        "auxiliary_exact_primary_not": qe == 0 and pe > 0,
        "both_nonexact": pe > 0 and qe > 0,
        "agreement_both_nonexact": pe > 0 and qe > 0 and result["normalized_text_equal"],
    }
    return result


def complementarity_report(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("Expected schema_version 1 and cases list")
    seen, cases = set(), []
    for case in manifest["cases"]:
        name = case.get("id")
        if (not isinstance(name, str) or not name or name in seen
                or case.get("language") not in {"en", "zh", "ja", "ko"}
                or not isinstance(case.get("primary"), dict)
                or not isinstance(case.get("auxiliary"), dict)):
            raise ValueError("Duplicate or malformed case")
        seen.add(name)
        cases.append(assess_case(case))
    summaries = {}
    for language in sorted({c["language"] for c in cases}):
        rows = [c for c in cases if c["language"] == language]
        scores = [c["scores"] for c in rows if c["status"] == "paired-rated"]
        rated = len(scores)
        summary = {"cases": len(rows), "rated_cases": rated,
                   "unpaired_cases": sum(c["status"] == "unpaired" for c in rows),
                   "unrated_cases": sum(c["status"] in {"paired-unrated", "unrated"} for c in rows)}
        # No paired reference is unknown, never a zero-error observation.
        keys = ("reference_characters", "whole_window_oracle_edits", "potential_edits_saved",
                "always_auxiliary_extra_edits", "primary_exact_auxiliary_not",
                "auxiliary_exact_primary_not", "both_nonexact", "agreement_both_nonexact")
        summary.update({k: sum(s[k] for s in scores) if rated else None for k in keys})
        for model in ("primary", "auxiliary"):
            summary[model + "_edits"] = sum(s[model]["character_edits"] for s in scores) if rated else None
        summaries[language] = summary
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        "schema_version": 1, "status": "analysis-only-no-adoption",
        "input_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalization": "NFKC/casefold/remove whitespace and Unicode punctuation; no numeral/script equivalence",
        "limits": [
            "Fingerprints are caller-supplied evidence, not independently observed decoder inputs.",
            "Raw differences use raw-text offsets, not source-word ownership or audio times.",
            "Oracle reads references and selects whole windows; it is not an automatic selector or per-character fusion.",
            "Partial, uncertain, unreviewed or unpaired references are not scored.",
            "Dataset references are not independently audited acoustic gold; edits are not semantic error diagnoses.",
            "This analyzer invokes no inference and edits no subtitles; its metrics do not validate timing, natural-stream quality or deployment.",
        ],
        "by_language": summaries, "cases": cases,
    }


def render_report(report):
    """Standalone escaped report with every candidate retained for external review."""
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>Paired ASR evidence</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{border:1px solid #ccc;'
             'padding:8px;text-align:left;vertical-align:top}table{border-collapse:collapse;width:100%}</style>',
             '<h1>Paired ASR evidence — no automatic adoption</h1><ul>']
    parts += ['<li>' + escape(limit) + '</li>' for limit in report["limits"]]
    parts += ['</ul><h2>Per-language diagnostics</h2><pre>',
              escape(json.dumps(report["by_language"], ensure_ascii=False, indent=2)), '</pre>']
    for case in report["cases"]:
        parts += ['<h2>' + escape(case["id"]) + '</h2><p>' + escape(case["status"]) + '</p>',
                  '<table><tr><th>Primary</th><th>Auxiliary</th></tr><tr><td><pre>',
                  escape(case["primary"].get("text")), '</pre></td><td><pre>',
                  escape(case["auxiliary"].get("text")), '</pre></td></tr></table><pre>',
                  escape(json.dumps({k: case.get(k) for k in
                         ("reference", "reason", "scores", "differences")}, ensure_ascii=False, indent=2)), '</pre>']
    return '\n'.join(parts)
