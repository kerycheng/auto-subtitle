"""Compare two diagnose_quality outputs using identical development inputs."""
import argparse
import html
import json
from pathlib import Path

from autosub.quality_gate import compare_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-policy", default="legacy")
    parser.add_argument("--candidate-policy", default="conservative")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_quality(json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")), args.baseline_policy, args.candidate_policy)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("請選空白輸出目錄，避免覆蓋先前報告")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    body = ['<!doctype html><meta charset="utf-8"><title>品質回歸檢查</title>',
        '<style>body{font:16px system-ui;margin:32px}td,th{padding:10px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
        '<h1>開發集時間回歸檢查</h1><p>' + html.escape(result["note"]) + '</p>',
        '<p>結果：' + ('通過' if result["passed"] else '未通過，保留現行預設') + '</p>',
        '<table><tr><th>片段</th><th>人工句數</th><th>一對一 100ms 差值</th><th>外側 100ms 差值</th><th>外側 250ms 差值</th><th>結果</th></tr>']
    for row in result["rows"]:
        values = [row["clip"], str(row["reference_events"]),
                  *[f'{value:+.1%}' for value in row["deltas"].values()],
                  "通過" if row["passed"] else "退步：" + ", ".join(row["regressed_metrics"])]
        body.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in values) + '</tr>')
    (args.output / "index.html").write_text(''.join(body) + '</table>', encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "default_promotion_authorized": False,
                      "regressed_clips": [r["clip"] for r in result["rows"] if not r["passed"]]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
