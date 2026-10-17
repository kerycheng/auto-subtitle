"""Run a private source excerpt and compare timing to translated/manual ASS.

The reference is used ONLY after ASR completes. Never writes next to source files.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from threading import Event

from autosub.domain import Project
from autosub.evaluation import boundary_report, read_ass
from autosub.export import export_subtitles
from autosub.media import ffmpeg_path
from autosub.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("media", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", default="small")
parser.add_argument("--seconds", type=int, default=60)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--language", default="ja")
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AUTOSUB_MODEL_DIR", str(Path(__file__).resolve().parents[1] / "models"))
source = args.output / "excerpt.wav"
manifest = args.output / "source.json"
source_key = {"media": str(args.media.resolve()), "size": args.media.stat().st_size,
              "mtime_ns": args.media.stat().st_mtime_ns, "offset": args.offset, "seconds": args.seconds}
if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != source_key:
    raise ValueError("測試輸出資料夾已用於其他來源，請改用新的資料夾")
if not source.exists():
    command = [ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", str(args.media),
               "-ss", str(args.offset), "-t", str(args.seconds), "-map", "0:a:0", "-vn",
               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(source)]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-3000:])
    manifest.write_text(json.dumps(source_key, ensure_ascii=False, indent=2), encoding="utf-8")
path = args.output / "generated.autosub.json"
project = Project.load(path) if path.exists() else Project(media_path=str(source), model=args.model, language=args.language)
if project.model != args.model or project.language != args.language:
    raise ValueError("測試設定已改變，請使用新輸出資料夾")
started = time.perf_counter()
project = run_pipeline(project, path, Event(), lambda n, text: print(n, text, flush=True))
elapsed = time.perf_counter() - started
for suffix in [".ass", ".srt"]:
    export_subtitles(project, args.output / ("generated" + suffix))
reference, _ = read_ass(args.reference)
for cue in project.cues:
    cue.start_ms += args.offset * 1000
    cue.end_ms += args.offset * 1000
report = boundary_report(reference, project.cues, args.offset * 1000, args.offset * 1000 + project.duration_ms)
report.update({"model": project.model, "device": project.device, "run_seconds_including_model_load": round(elapsed, 2),
               "words": len(project.words), "flagged_cues": sum(bool(c.flags) for c in project.cues),
               "reference_text_used_by_asr": False})
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
