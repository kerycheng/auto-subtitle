"""Development-only ASR and finished-reference comparison on prepared excerpts."""
import argparse
import copy
import json
import os
from pathlib import Path
from threading import Event

from autosub.comparison import ComparisonSettings, compare_file, export_report, save_session, summary
from autosub.domain import Project
from autosub.pipeline import run_pipeline
from autosub.timing import apply_timing

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("AUTOSUB_MODEL_DIR", str(Path(__file__).resolve().parents[1] / "models"))
results = []
for clip in json.loads(args.manifest.read_text(encoding="utf-8"))["clips"]:
    if clip["split"] != "development":
        continue  # held-out sources remain unprocessed until strategy and review criteria are frozen
    name = clip["id"]
    if Path(name).name != name:
        raise ValueError("invalid clip id")
    print("Starting " + name, flush=True)
    audio = (args.prepared / name / "excerpt.wav").resolve()
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "generated.autosub.json"
    project = Project.load(path) if path.exists() else Project(media_path=str(audio), model="small", language="ja")
    project = run_pipeline(project, path, Event(), lambda n, text: print(n, text, flush=True))
    for policy in ("legacy", "conservative"):
        working = copy.deepcopy(project)
        working.boundary_policy = policy
        working.cues = apply_timing(working)
        working.save(folder / f"{policy}.autosub.json")
        session = compare_file(working, Path(clip["reference"]),
            ComparisonSettings(reference_offset_ms=clip["offset"] * 1000))
        save_session(session, folder / f"{policy}.comparison.json")
        export_report(session, folder / f"{policy}.html")
        results.append({"clip": name, "policy": policy, "split": "development", **summary(session)})
    (args.output / "suite-summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Finished " + name, flush=True)
