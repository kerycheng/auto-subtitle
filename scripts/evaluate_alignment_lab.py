"""Export fixed-text alignment candidates with explicit failure/repair evidence."""
import argparse
import copy
import html
import json
from collections import Counter
from pathlib import Path

from autosub.alignment_lab import file_hash, validate_result
from autosub.comparison import ComparisonSettings, compare_file, summary
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.timing import apply_timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = file_hash(args.manifest)
    original = Path(manifest["source_project"])
    if file_hash(original) != manifest["project_sha256"] or file_hash(manifest["audio_path"]) != manifest["audio_sha256"]:
        raise ValueError("Experiment source changed")
    baseline = Project.load(original)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Select an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {p: file_hash(p) for p in [original, args.reference, Path(baseline.media_path), Path(baseline.audio_path)]}
    report = {"scope": "fixed recognized Japanese text; temporal comparison only; no acoustic gold",
              "manifest_sha256": manifest["manifest_sha256"], "default_changed": False,
              "human_verified": False, "backends": {}}
    labels = []
    cases = {c["id"]: c for c in manifest["cases"]}
    for path in args.results:
        raw = json.loads(path.read_text(encoding="utf-8"))
        backend = raw["backend"]
        if backend in report["backends"] or backend not in {"qwen", "whisperx"}:
            raise ValueError("Duplicate or unknown backend")
        checked = validate_result(manifest, raw)
        candidate = copy.deepcopy(baseline)
        applied = []
        for cue in candidate.cues:
            check = checked[cue.id]
            if cue.locked:
                continue
            cue.flags.append("獨立對齊實驗候選，未經人工聲學驗收")
            cue.timing_trace["alignment_lab"] = {"backend": backend, **check}
            # Repaired times and input-edge hits remain visible in the report,
            # while the preview safely retains the old span for those cases.
            if check["span_ms"] and not set(check["warnings"]) & {"backend-repaired-timestamps", "input-window-edge"}:
                cue.speech_start_ms, cue.speech_end_ms = check["span_ms"]
                applied.append(cue.id)
            else:
                cue.flags.append("對齊證據不足，保留原始時間")
            cue.flags.extend(check["warnings"])
        candidate.cues = apply_timing(candidate, base=candidate.cues)
        assert [c.text for c in candidate.cues] == [c.text for c in baseline.cues], "Alignment must not reorder/change text"
        assert candidate.words == baseline.words
        candidate.validate()
        prefix = args.output / (backend + "-candidate")
        candidate.save(prefix.with_suffix(".autosub.json"))
        for suffix in (".ass", ".srt"):
            export_subtitles(candidate, prefix.with_suffix(suffix))
        stats = {"statuses": dict(Counter(v["status"] for v in checked.values())),
                 "warnings": dict(Counter(w for c in checked.values() for w in c["warnings"])),
                 "preview_applied": len(applied), "cases": checked,
                 "inference_seconds": sum(row["seconds"] for row in raw["results"]),
                 "load_seconds": raw["load_seconds"],
                 "peak_tensor_allocated_bytes": max(row["peak_allocated_bytes"] for row in raw["results"]),
                 "peak_tensor_reserved_bytes": max(row["peak_reserved_bytes"] for row in raw["results"]),
                 "time_pairing_only": summary(compare_file(candidate, args.reference, ComparisonSettings()))}
        report["backends"][backend] = stats
        labels.append(backend)
    report["sources_unchanged"] = all(file_hash(p) == value for p, value in sources.items())
    if not report["sources_unchanged"]:
        raise RuntimeError("Source changed during report generation")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    esc = lambda s: html.escape(str(s))
    body = ['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>獨立對齊比較</title>',
        '<style>body{font:16px system-ui;margin:28px;line-height:1.6}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}table{border-collapse:collapse;width:100%}th{background:#edf1f7}</style>',
        '<h1>同原文的獨立對齊比較</h1><p>兩版輸入文字完全相同。只比較對齊，不補漏字、不改切句。未經人工聲學驗收；中文翻譯軸只作時間參考。',
        '補值或碰觸輸入窗邊緣時，候選字幕保留原時間。大幅時間變動與零時長單位仍列待複核。</p>',
        '<p>記憶體數字為 PyTorch CUDA 張量配置／保留峰值，未包含桌面、驅動與其他程式。</p>',
        '<table><tr><th>後端</th><th>狀態</th><th>預覽更新</th><th>推論秒數</th><th>峰值配置 MiB</th></tr>']
    for label in labels:
        r = report["backends"][label]
        body.append('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in [label, r["statuses"], r["preview_applied"],
                     round(r["inference_seconds"], 3), round(r["peak_tensor_allocated_bytes"] / 1024**2, 1)]) + '</tr>')
    body.append('</table><h2>逐句原始發聲時間候選（毫秒）</h2><table><tr><th>原文／基準</th>' + ''.join('<th>' + esc(v) + '</th>' for v in labels) + '</tr>')
    for case in manifest["cases"]:
        body.append('<tr><td>' + esc(case["text"]) + '<br>' + esc(case["baseline_ms"]) + '</td>')
        for label in labels:
            c = report["backends"][label]["cases"][case["id"]]
            body.append('<td>' + esc(c["span_ms"]) + '<br>' + esc(c["status"]) + '<br>' + esc(c["warnings"] + c["errors"]) + '</td>')
        body.append('</tr>')
    body.append('</table></html>')
    (args.output / "index.html").write_text(''.join(body), encoding="utf-8")
    print(json.dumps({"backends": {k: {n: v[n] for n in ["statuses", "preview_applied", "inference_seconds", "peak_tensor_allocated_bytes"]}
                                   for k, v in report["backends"].items()}, "sources_unchanged": report["sources_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
