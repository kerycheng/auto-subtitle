"""Build Aegisub candidates from full-window ASR + alignment + lexical ownership.

This exports diagnostics, not resumable app projects for unintegrated backends.
"""
import argparse
import copy
import html
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from autosub.alignment_lab import file_hash, restore_unit_text, validate_result, atom_ownership, stitch_atoms, normalized
from autosub.domain import Project, Word
from autosub.export import export_subtitles
from autosub.lexical_segmentation import boundary_map, segment_lexically
from autosub.timing import apply_timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stitch-windows", action="store_true")
    args = parser.parse_args()
    p = Project.load(args.project)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = file_hash(args.manifest)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    checked = validate_result(manifest, result)
    if file_hash(p.audio_path) != manifest["audio_sha256"]:
        raise ValueError("Source audio mismatch")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Select an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    original = {path: file_hash(path) for path in [args.project, Path(p.media_path), Path(p.audio_path)]}
    atoms, rows, windows = [], [], []
    cases = {c["id"]: c for c in manifest["cases"]}
    for raw in result["results"]:
        case = cases[raw["id"]]
        check = checked[raw["id"]]
        detail = {"id": raw["id"], "assessment": check, "input_text": case["text"],
                  "selected_atoms": [], "context_atoms_excluded": [], "invalid_owned_atoms": [],
                  "fully_skipped": False}
        rows.append(detail)
        if set(check["errors"]) - {"time-outside-input-window"}:
            detail["fully_skipped"] = True
            continue
        mapped = restore_unit_text(case["text"], raw["units"])
        # Probability is only a neutral internal adapter placeholder, never
        # exported as a measured ASR confidence (these backends do not expose it).
        words = [Word(case["window_ms"][0] + round(u["start"] * 1000),
                      case["window_ms"][0] + round(u["end"] * 1000), u["text"]) for u in mapped]
        bm = boundary_map(words)
        # Dictionary units alone can split compounds such as 芋煮会. Require
        # agreement with the phrase parser for cross-window ownership.
        safe = sorted((bm.allowed & bm.phrases) | {0, len(words)})
        window = {"id": raw["id"], "window_ms": case["window_ms"], "owner_ms": case["owner_ms"], "atoms": []}
        windows.append(window)
        for a, b in zip(safe, safe[1:]):
            group = words[a:b]
            lo, hi = group[0].start_ms, max(w.end_ms for w in group)
            atom = {"text": "".join(w.text for w in group), "start_ms": lo, "end_ms": hi,
                    "case": raw["id"], "warnings": check["warnings"],
                    "unit_evidence": [{**asdict(w), "probability": None} for w in group]}
            ownership = atom_ownership(lo, hi, case["window_ms"], case["owner_ms"], p.duration_ms)
            if 0 <= case["window_ms"][0] <= lo <= hi <= case["window_ms"][1] <= p.duration_ms:
                window["atoms"].append(atom)
            if ownership == "owned":
                atoms.append(atom)
                detail["selected_atoms"].append(atom)
            elif ownership == "context":
                detail["context_atoms_excluded"].append(atom)
            else:
                detail["invalid_owned_atoms"].append(atom)
    seams = []
    if args.stitch_windows:
        atoms, seams = stitch_atoms(windows, speech=p.speech)
    candidate = copy.deepcopy(p)
    candidate.cues = []
    # Keep the full lexical atom as one timing unit. Flattening its component
    # character times would re-interleave two windows (e.g. 1,1,分,分).
    unanchored = [a for a in atoms if a["end_ms"] <= a["start_ms"]]
    display_atoms = [a for a in atoms if a["end_ms"] > a["start_ms"]]
    candidate.words = [Word(a["start_ms"], a["end_ms"], a["text"]) for a in display_atoms]
    candidate.words.sort(key=lambda w: (w.start_ms, w.end_ms))
    candidate.cues = apply_timing(candidate, segmenter=segment_lexically)
    if normalized("".join(c.text for c in candidate.cues)) != normalized("".join(a["text"] for a in display_atoms)):
        raise ValueError("Candidate reordered or lost selected source text")
    for cue in candidate.cues:
        cue.flags.append("全語音窗實驗候選；模型信心未知，需人工複核")
    candidate.validate()
    for suffix in (".ass", ".srt"):
        export_subtitles(candidate, args.output / (manifest["asr_backend"] + "-context-candidate" + suffix))
    preserved = all(file_hash(path) == value for path, value in original.items())
    if not preserved:
        raise RuntimeError("Original changed")
    report = {"schema": "asr-candidate-v1", "asr_backend": manifest["asr_backend"],
              "alignment_backend": result["backend"], "source_project_sha256": original[args.project],
              "manifest_sha256": manifest["manifest_sha256"], "sources_unchanged": preserved,
              "human_verified": False, "default_changed": False, "not_a_resumable_app_project": True,
              "whole_lexical_atom_ownership": True, "cases": rows,
              "stitch_windows": args.stitch_windows, "seams": seams, "selected_atoms": atoms,
              "unanchored_selected_atoms": unanchored,
              "words": [{**asdict(w), "probability": None} for w in candidate.words],
              "cues": [asdict(c) for c in candidate.cues],
              "summary": {"input_cases": len(cases), "cases_with_alignment_errors": sum(bool(row["assessment"]["errors"]) for row in rows),
                          "fully_skipped_cases": sum(row["fully_skipped"] for row in rows),
                          "invalid_owned_atoms": sum(len(row["invalid_owned_atoms"]) for row in rows),
                          "unanchored_selected_atoms": len(unanchored),
                          "words": len(candidate.words), "cues": len(candidate.cues),
                          "overlap_pairs": sum(a.end_ms > b.start_ms for a, b in zip(candidate.cues, candidate.cues[1:]))}}
    (args.output / "candidate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    esc = lambda x: html.escape(str(x))
    body = ['<!doctype html><meta charset="utf-8"><title>全語音窗候選</title>',
        '<style>body{font:16px system-ui;margin:28px}td,th{border:1px solid #aaa;padding:8px;vertical-align:top}table{border-collapse:collapse}</style>',
        '<h1>全語音窗辨識 → 獨立對齊 → 完整詞語分段</h1>',
        '<p>僅供 Aegisub 比較。重疊輸入的文字以完整詞組時間中點歸屬，不從詞中間切開。',
        '未經人工語意／聲學驗收，仍可能漏字、幻覺或跨窗重複。沒有模型詞信心值。JSON 為診斷資料，不是可續跑專案。</p>',
        '<p>' + esc(report["summary"]) + '</p><table><tr><th>時間</th><th>字幕</th><th>警告</th></tr>']
    for cue in candidate.cues:
        body.append('<tr><td>' + f'{cue.start_ms/1000:.2f}–{cue.end_ms/1000:.2f}' + '</td><td>' + esc(cue.text) + '</td><td>' + esc(cue.flags) + '</td></tr>')
    body.append('</table><h2>每窗文字與所有權</h2>')
    for row in rows:
        body.append('<h3>' + esc(row["id"]) + '</h3><p>' + esc(row["input_text"]) + '</p><p>保留：' + esc([a["text"] for a in row["selected_atoms"]]) + '</p><p>上下文排除：' + esc([a["text"] for a in row["context_atoms_excluded"]]) + '</p><p>' + esc(row["assessment"]) + '</p>')
    (args.output / "index.html").write_text(''.join(body), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
