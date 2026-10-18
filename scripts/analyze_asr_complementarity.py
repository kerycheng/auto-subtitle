"""Analyze saved paired transcripts only; no inference or subtitle modifications.

Supply a schema_version=1 JSON with cases: id, language, primary, auxiliary,
reference. Each run needs status=complete, text and input (samples_sha256,
sample_rate, sample_count, language). Reference scope is dataset-complete,
human-complete, partial or unreviewed; rating requires uncertain=false, and
human-complete additionally requires reviewer_confirmed=true. Input evidence
must be prepared and audited separately; equality is checked, not inferred.
"""
import argparse
import json
from pathlib import Path

from autosub.asr_complementarity import complementarity_report, render_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Fail before writing anything when an earlier output exists.
    if args.output.exists():
        raise ValueError("Output exists; preserve prior evidence")
    report = complementarity_report(json.loads(args.manifest.read_text(encoding="utf-8")))
    rendered = render_report(report)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "index.html").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
