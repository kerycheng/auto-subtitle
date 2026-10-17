"""Extract private evaluation audio without running ASR or promoting labels to gold.

Input manifest: clips[{id, media, reference, offset, seconds, split, scenario}].
All outputs belong in ignored test-artifacts. Held-out clips remain unscored.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

from autosub.evaluation import read_ass
from autosub.media import ffmpeg_path, fingerprint

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
clips = json.loads(args.manifest.read_text(encoding="utf-8"))["clips"]
groups, ids, summary = {}, set(), []
for clip in clips:
    name = clip["id"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) or name in ids:
        raise ValueError("clip id must be unique ASCII letters/digits/hyphens")
    ids.add(name)
    source = Path(clip["media"]).resolve()
    split = clip["split"]
    if split not in {"development", "heldout"} or groups.get(str(source), split) != split:
        raise ValueError("one source must stay within one split")
    groups[str(source)] = split
    if clip["offset"] < 0 or not 30 <= clip["seconds"] <= 60:
        raise ValueError("clips must be 30–60 seconds with nonnegative offsets")
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    key = {**clip, "source_fingerprint": fingerprint(source)}
    record_path = folder / "source.json"
    if record_path.exists() and json.loads(record_path.read_text(encoding="utf-8")) != key:
        raise ValueError("output directory already belongs to a different source")
    audio = folder / "excerpt.wav"
    if not audio.exists():
        command = [ffmpeg_path(), "-hide_banner", "-nostdin", "-n", "-i", str(source),
                   "-ss", str(clip["offset"]), "-t", str(clip["seconds"]), "-map", "0:a:0",
                   "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(audio)]
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace")[-2000:])
    record_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    references, _ = read_ass(Path(clip["reference"]))
    lo, hi = clip["offset"] * 1000, (clip["offset"] + clip["seconds"]) * 1000
    # Translated ASS times are orientation aids, never acoustic or original-text gold.
    events = [{"translated_reference_text": r.text,
               "reference_display_ms": [r.start_ms - lo, r.end_ms - lo],
               "original_text": "", "speech_start_ms": None, "speech_end_ms": None,
               "overlap": None, "elongated": None, "uncertain": None,
               "confirmed": False, "reviewer": ""}
              for r in references if lo <= r.start_ms < r.end_ms <= hi]
    labels = folder / "unconfirmed-events.json"
    if not labels.exists():
        labels.write_text(json.dumps({"role": "orientation only; original-language acoustic review pending",
                                     "source": key, "records": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.append({"id": name, "split": split, "seconds": clip["seconds"],
                    "orientation_events": len(events), "human_confirmed_events": 0,
                    "asr_run": False, "scored": False})
    print(name, split, len(events), flush=True)
(args.output / "dataset-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
