"""Controlled repetition load test, not a subtitle accuracy benchmark. Offline only."""
import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from threading import Event

import numpy as np

from autosub.domain import Project
from autosub.media import Cancelled, decode_audio, load_audio
from autosub.model_cache import clear_model
from autosub.pipeline import run_pipeline
from autosub.runtime import check_runtime


def peak_memory_mb():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1048576


def child(args):
    target = args.output / f"{args.seconds}s"
    target.mkdir(parents=True, exist_ok=True)
    original = load_audio(args.source)
    if not len(original):
        raise ValueError("benchmark source is empty")
    samples = np.tile(original, math.ceil(args.seconds * 16000 / len(original)))[:args.seconds * 16000]
    audio_path = target / "repeated-load-test.wav"
    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.rint(samples * 32768).astype("<i2").tobytes())
    del samples, original
    result = {"duration_seconds": args.seconds, "model": str(args.model), "device": "cpu",
              "source_kind": "repeated excerpt; load test only", "runtime": check_runtime("cpu")}
    clear_model()
    start = time.perf_counter()
    project = run_pipeline(Project(media_path=str(audio_path.resolve()), model=str(args.model.resolve()), language=args.language),
                           target / "full.autosub.json", Event(), lambda *_: None)
    elapsed = time.perf_counter() - start
    result.update(elapsed_seconds=elapsed, real_time_factor=elapsed / args.seconds,
                  peak_working_set_mb=peak_memory_mb(), cues=len(project.cues), words=len(project.words))
    if args.seconds == 60:
        # Use a warm model. Request cancel at second chunk; only committed first chunk may survive.
        cancel = Event()
        requested = None
        interrupted = Project(media_path=str(audio_path.resolve()), model=str(args.model.resolve()), language=args.language)
        path = target / "cancel-resume.autosub.json"
        def progress(_, message):
            nonlocal requested
            if "辨識區段 2/" in message:
                requested = time.perf_counter()
                cancel.set()
        try:
            run_pipeline(interrupted, path, cancel, progress)
            raise AssertionError("cancellation was not observed")
        except Cancelled:
            latency = time.perf_counter() - requested
        saved = Project.load(path)
        assert saved.status == "cancelled" and saved.completed_chunks == [0]
        prefix = [vars(word).copy() for word in saved.words]
        start = time.perf_counter()
        resumed = run_pipeline(saved, path, Event(), lambda *_: None)
        assert resumed.status == "complete" and resumed.completed_chunks == [0, 1]
        assert [vars(word) for word in resumed.words[:len(prefix)]] == prefix
        assert [vars(word) for word in resumed.words] == [vars(word) for word in project.words]
        result["cancel_resume"] = {"status": "passed", "cancel_latency_seconds": latency,
            "committed_chunks_on_cancel": [0], "resume_seconds": time.perf_counter() - start,
            "same_words_as_uninterrupted": True}
    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--seconds", type=int)
    args = parser.parse_args()
    if not (args.model / "model.bin").is_file():
        raise ValueError("請指定已下載的本機模型，效能測試不下載模型")
    if args.seconds:
        child(args)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("請選空白輸出目錄")
    args.output.mkdir(parents=True, exist_ok=True)
    decoded = args.output / "source.wav"
    decode_audio(args.source, decoded, 0, Event())
    results = []
    for seconds in (60, 300, 1200):
        print(f"Starting isolated {seconds}s load test", flush=True)
        subprocess.run([sys.executable, __file__, str(decoded), "--model", str(args.model), "--output", str(args.output),
                        "--language", args.language, "--seconds", str(seconds)], check=True)
        results.append(json.loads((args.output / f"{seconds}s/result.json").read_text(encoding="utf-8")))
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
