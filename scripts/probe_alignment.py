"""Opt-in local alignment spike. Saves candidates without applying any of them."""
import argparse
import json
import time
from pathlib import Path
from threading import Event

from autosub.alignment import LocalWhisperAligner, analyze_cue
from autosub.domain import Project
from autosub.paired_evaluation import annotation_template, paired_report

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--report-name", default="feasibility.json")
args = parser.parse_args()
project = Project.load(args.project)
if args.model:
    project.model = args.model
backend = LocalWhisperAligner(project.model, project.device)
args.output.mkdir(parents=True, exist_ok=True)
stats = []
for index, cue in enumerate(project.cues[:args.limit or None]):
    started = time.perf_counter()
    candidate = analyze_cue(project, index, args.output / "alignment-cache", Event(), backend=backend)
    elapsed = time.perf_counter() - started
    cue.candidate = candidate
    stats.append({"id": cue.id, "elapsed_seconds": round(elapsed, 3), "status": candidate["status"],
                  "reasons": candidate["reasons"], "cache_hit": candidate.get("cache_hit", False),
                  "base": [cue.speech_start_ms, cue.speech_end_ms], "candidate_base": candidate["base"]})
    project.save(args.output / "candidates.autosub.json")
    print(f"{index + 1}: {candidate['status']} {elapsed:.2f}s", flush=True)
template = annotation_template(project)
annotation_path = args.output / "unconfirmed-pairs.json"
if not annotation_path.exists():
    annotation_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
peak_memory = None
try:
    import ctypes
    from ctypes import wintypes
    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                               ctypes.byref(counters), ctypes.sizeof(counters)):
        peak_memory = counters.PeakWorkingSetSize
except (AttributeError, OSError, ctypes.ArgumentError):
    pass
(args.output / args.report_name).write_text(json.dumps({"identity": backend.identity,
    "run_seconds": sum(c["elapsed_seconds"] for c in stats), "peak_process_working_set_bytes": peak_memory,
    "candidates": stats, "accuracy_report": paired_report(project, template),
    "note": "runtime feasibility only; candidates are not human-confirmed gold"},
    ensure_ascii=False, indent=2), encoding="utf-8")
