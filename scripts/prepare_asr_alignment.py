"""Bind ASR-only text to its original, unmodified acoustic windows."""
import argparse
import json
from pathlib import Path

from autosub.alignment_lab import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("result", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Output exists")
source = json.loads(args.manifest.read_text(encoding="utf-8"))
result = json.loads(args.result.read_text(encoding="utf-8"))
if result["manifest_sha256"] != file_hash(args.manifest) or result["status"] != "complete":
    raise ValueError("ASR input mismatch or incomplete ASR")
if [r["id"] for r in result["results"]] != [w["id"] for w in source["windows"]]:
    raise ValueError("ASR case mismatch")
manifest = {"schema": "alignment-lab-v1", "source_document": str(args.result.resolve()),
    "source_document_sha256": file_hash(args.result), "audio_path": source["audio_path"],
    "audio_sha256": source["audio_sha256"], "duration_ms": source["duration_ms"],
    "asr_backend": result["backend"], "reference_text_used": False,
    "cases": [{"id": r["id"], "text": r["text"], "language": r.get("language") or source["language"],
               "window_ms": r["window_ms"], "owner_ms": r["owner_ms"], "baseline_ms": r["window_ms"],
               "baseline_kind": "input-window-not-acoustic-gold"} for r in result["results"]]}
args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f'{len(manifest["cases"])} complete ASR window texts')
