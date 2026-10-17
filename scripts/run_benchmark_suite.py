"""Sequential private benchmarks defined by a JSON manifest, with isolated logs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.output = args.output.resolve()
for job in json.loads(args.manifest.read_text(encoding="utf-8")):
    name = job["name"]
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("測試名稱不可包含路徑")
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Starting {name}", flush=True)
    command = [sys.executable, str(root / "scripts/benchmark_clip.py"), job["media"], job["reference"],
               "--output", str(folder), "--model", job.get("model", "small"),
               "--offset", str(job.get("offset", 0)), "--seconds", str(job.get("seconds", 60))]
    with (folder / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [sys.executable, str(root / "scripts/compare_retiming.py"), str(folder / "generated.autosub.json"),
               job["reference"], "--output", str(folder / "comparison"),
               "--offset-ms", str(job.get("offset", 0) * 1000)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", check=True)
    print(name + " " + result.stdout.strip(), flush=True)
