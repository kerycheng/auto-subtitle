"""Opt-in network integration check using Whisper's public English test fixture.

Downloads ~75MB tiny model and a short clip, writes only to test-artifacts/models.
Run from repo root: python scripts/verify_real.py
"""
from pathlib import Path
from threading import Event
import json
import os
import urllib.request

from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.pipeline import run_pipeline

root = Path(__file__).resolve().parents[1]
artifacts = root / "test-artifacts"
artifacts.mkdir(exist_ok=True)
os.environ.setdefault("AUTOSUB_MODEL_DIR", str(root / "models"))
source = artifacts / "jfk.flac"
if not source.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", source)
project_path = artifacts / "english.autosub.json"
project = Project.load(project_path) if project_path.exists() else Project(
    media_path=str(source), model="tiny", language="en", device="cpu")
result = run_pipeline(project, project_path, Event(), lambda n, text: print(n, text, flush=True))
assert result.status == "complete" and result.words and result.cues
assert "country" in " ".join(w.text for w in result.words).lower()
for suffix in [".ass", ".srt"]:
    export_subtitles(result, artifacts / ("english" + suffix))
(artifacts / "integration-result.json").write_text(json.dumps({
    "status": result.status, "duration_ms": result.duration_ms,
    "words": len(result.words), "cues": len(result.cues),
    "model": result.model, "device": result.device,
}, indent=2), encoding="utf-8")
print("Real-model integration passed", flush=True)
