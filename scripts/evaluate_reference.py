"""Compare a generated project with an ASS reference; all inputs are read-only."""
import argparse
import json
from pathlib import Path

from autosub.domain import Project
from autosub.evaluation import boundary_report, read_ass

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--offset-ms", type=int, default=0, help="Original-clock time corresponding to clip zero")
args = parser.parse_args()
project = Project.load(args.project)
reference, _ = read_ass(args.reference)
for cue in project.cues:
    cue.start_ms += args.offset_ms
    cue.end_ms += args.offset_ms
report = boundary_report(reference, project.cues, args.offset_ms, args.offset_ms + project.duration_ms)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
