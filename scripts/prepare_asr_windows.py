"""Create identical, transcript-free acoustic inputs for two ASR engines."""
import argparse
import json
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.recognition_windows import mixed_windows, monolingual_windows

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--strategy", choices=["mixed-vad-v1", "mono-vad-context-v1"], default="mono-vad-context-v1")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Output exists")
p = Project.load(args.project)
if p.language != "ja":
    raise ValueError("This experiment currently evaluates Japanese only")
result = {"schema": "asr-lab-v1", "audio_path": p.audio_path, "audio_sha256": file_hash(p.audio_path),
          "strategy": args.strategy,
          "source_project_sha256": file_hash(args.project), "language": p.language, "duration_ms": p.duration_ms,
          "speech": p.speech, "text_provided": False,
          "windows": [{"id": str(i), "window_ms": [a, b], "owner_ms": [c, d]}
                      for i, (a, b, c, d) in enumerate((mixed_windows if args.strategy == "mixed-vad-v1" else monolingual_windows)(p.speech, p.duration_ms))]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f'{len(result["windows"])} identical acoustic windows; no transcript')
