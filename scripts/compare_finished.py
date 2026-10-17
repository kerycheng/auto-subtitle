"""Post-ASR finished-subtitle comparison. Never passes reference text to ASR."""
import argparse
from pathlib import Path
import json

from autosub.comparison import ComparisonSettings, compare_file, export_report, save_session, summary
from autosub.domain import Project

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--offset-ms", type=int, default=0, help="reference time minus this offset = project time")
parser.add_argument("--tolerance-ms", type=int, default=100)
parser.add_argument("--timeline-confirmed", action="store_true", help="operator has checked media edit, audio track and offset")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
project = Project.load(args.project)
settings = ComparisonSettings(reference_offset_ms=args.offset_ms, tolerance_ms=args.tolerance_ms,
                              timeline_confirmed=args.timeline_confirmed)
session = compare_file(project, args.reference, settings)
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "comparison.json").exists():
    parser.error("output already contains a comparison; choose a new directory to preserve review decisions")
save_session(session, args.output / "comparison.json")
for suffix in (".html", ".csv"):
    export_report(session, args.output / ("report" + suffix))
print(json.dumps(summary(session), ensure_ascii=False, indent=2))
