"""Post-ASR development comparison. Never reads held-out clips or changes projects."""
import argparse
import copy
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from autosub.comparison import ComparisonSettings, compare_file, summary
from autosub.domain import Project
from autosub.timing import apply_timing
from autosub.triage import quality_diagnostics, review_order, review_priority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("請選擇空白輸出目錄，避免覆蓋先前報告或人工判定")
    args.output.mkdir(parents=True, exist_ok=True)
    results, queue = [], []
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for clip in manifest["clips"]:
        if clip["split"] != "development":
            continue
        name = clip["id"]
        if Path(name).name != name:
            raise ValueError("invalid clip id")
        project_path = args.projects / name / "generated.autosub.json"
        project = Project.load(project_path)
        original = project_path.read_bytes()
        # Same recognized words, baseline versus tail-preserving policy. No ASR or tuning on reference text.
        for policy in ("legacy", "conservative"):
            working = copy.deepcopy(project)
            working.boundary_policy = policy
            working.cues = apply_timing(working)
            session = compare_file(working, Path(clip["reference"]),
                ComparisonSettings(reference_offset_ms=clip["offset"] * 1000))
            risks = quality_diagnostics(working)
            results.append({"clip": name, "policy": policy, "summary": summary(session),
                "risk_counts": dict(Counter(reason for risk in risks for reason in risk["reasons"])),
                "risks": risks, "project_sha256": hashlib.sha256(original).hexdigest(),
                "reference_sha256": session["reference_sha256"],
                "tolerance_ms": session["settings"]["tolerance_ms"],
                "reference_offset_ms": session["settings"]["reference_offset_ms"]})
            if policy == "legacy":
                for index in review_order(session):
                    row = session["rows"][index]
                    queue.append({"clip": name, **review_priority(row), "row": row})
        if project_path.read_bytes() != original:
            raise RuntimeError("來源專案意外改變")
    queue.sort(key=lambda item: (-item["score"], -item["max_delta_ms"], item["clip"]))
    report = {"scope": "development-only; post-ASR; temporal suggestions, not acoustic gold",
        "human_completion_rate": None, "results": results, "priority_queue": queue}
    (args.output / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    escape = lambda value: html.escape(str(value))
    body = ['<!doctype html><meta charset="utf-8"><title>第五階段品質診斷</title>',
        '<style>body{font:16px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}tr:nth-child(even){background:#f4f6fa}</style>',
        '<h1>第五階段品質診斷與優先複核</h1><p>僅使用開發集。分數代表複核順序；時間差不是實際發聲誤差。人工完成度尚無，預設策略維持不變。原始專案及人工成品不修改。</p>',
        '<h2>風險診斷</h2><table><tr><th>片段／策略</th><th>人工／自動句數</th><th>一對一時間估計 ≤100ms</th><th>可觀察風險（需試聽）</th></tr>']
    for result in results:
        s = result["summary"]
        rate = s["strict_time_estimate_rate"]
        body.append(f'<tr><td>{escape(result["clip"])} / {escape(result["policy"])}</td><td>{s["reference_events"]} / {s["generated_events"]}</td><td>{rate:.1%}</td><td>{escape(result["risk_counts"])}</td></tr>')
    body.append('</table><h2>基準策略優先複核清單</h2><table><tr><th>片段／範圍 ms</th><th>複核理由</th><th>人工字幕</th><th>自動字幕</th><th>頭／尾差 ms</th></tr>')
    for item in queue:
        row = item["row"]
        values = [f'{item["clip"]} / {row["reference_span"] or row["generated_span"]}',
            "；".join(item["reasons"]), row["reference_text"], row["generated_text"],
            f'{row["start_delta_ms"]} / {row["end_delta_ms"]}']
        body.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in values) + '</tr>')
    body.append('</table>')
    (args.output / "index.html").write_text(''.join(body), encoding="utf-8")
    print(json.dumps({"variants": len(results), "review_rows": len(queue), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
