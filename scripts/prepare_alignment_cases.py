"""Prepare fixed text/audio cases without using the translated reference."""
import argparse
import json
from pathlib import Path

from autosub.alignment_lab import make_cases
from autosub.domain import Project

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Manifest exists; select a new output")
manifest = make_cases(Project.load(args.project), args.project)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f'{len(manifest["cases"])} fixed cases')
