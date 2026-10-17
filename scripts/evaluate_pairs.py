"""Create unconfirmed templates or evaluate fixed, human-confirmed cue pairs."""
import argparse
import json
from pathlib import Path

from autosub.domain import Project
from autosub.paired_evaluation import annotation_template, paired_report

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["template", "report"])
parser.add_argument("project", type=Path)
parser.add_argument("--annotation", type=Path)
parser.add_argument("--offset-ms", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
project = Project.load(args.project)
if args.mode == "template":
    result = annotation_template(project, args.offset_ms)
else:
    if not args.annotation:
        parser.error("report requires --annotation")
    result = paired_report(project, json.loads(args.annotation.read_text(encoding="utf-8")))
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    parser.error("output exists; choose a new file to preserve annotations")
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(args.output)
