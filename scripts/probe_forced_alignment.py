"""Run one isolated, local alignment backend. Raw evidence is always retained."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosub.alignment_lab import file_hash
FAILURE_STATE = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=["qwen", "whisperx"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("max-cases must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    signature = file_hash(args.manifest)
    if file_hash(manifest["audio_path"]) != manifest["audio_sha256"]:
        raise ValueError("Audio changed")
    if file_hash(manifest.get("source_project", manifest.get("source_document"))) != manifest.get("project_sha256", manifest.get("source_document_sha256")):
        raise ValueError("Source project changed")
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; select a new path or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    model_manifest = args.model / "experiment-model.json"
    model_identity = json.loads(model_manifest.read_text(encoding="utf-8")) if model_manifest.exists() else str(args.model.resolve())
    report = {"backend": args.backend, "manifest_sha256": signature, "model": model_identity,
              "status": "running", "human_verified": False, "results": []}
    if args.resume and result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report["manifest_sha256"] != signature or report["backend"] != args.backend or report["model"] != model_identity:
            raise ValueError("Resume identity mismatch")
    def save():
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, result_path)
    FAILURE_STATE.update(report=report, path=result_path)
    save()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this controlled experiment")
    report["runtime"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                         "cuda": torch.version.cuda, "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
                         "versions": {p: importlib.metadata.version(p) for p in ["qwen-asr", "whisperx", "transformers", "numpy"]}}
    loaded = time.perf_counter()
    corrections = []
    languages = {case["language"] for case in manifest["cases"]}
    if not languages <= {"ja", "zh", "en", "ko"}:
        raise ValueError("Unsupported experiment language")
    if args.backend == "qwen":
        from qwen_asr import Qwen3ForcedAligner
        model = Qwen3ForcedAligner.from_pretrained(str(args.model.resolve()), device_map="cuda:0",
                    dtype=torch.float16, attn_implementation="eager", local_files_only=True)
        original_fix = model.aligner_processor.fix_timestamp
        def capture_fix(values):
            fixed = original_fix(values)
            raw = values.tolist()
            corrections.append({"raw_ms": raw, "fixed_ms": fixed,
                                "changed_indices": [i for i, (a, b) in enumerate(zip(raw, fixed)) if a != b]})
            return fixed
        model.aligner_processor.fix_timestamp = capture_fix
    else:
        if languages != {"ja"}:
            raise ValueError("The downloaded WhisperX model is Japanese only")
        import nltk
        nltk.data.path.insert(0, str(ROOT / ".cache" / "nltk"))
        nltk.data.find("tokenizers/punkt_tab/english/")
        import whisperx
        model, metadata = whisperx.load_align_model(language_code="ja", device="cuda",
                        model_name=str(args.model.resolve()), model_cache_only=True)
    torch.cuda.synchronize()
    report["load_seconds"] = time.perf_counter() - loaded
    completed = {r["id"] for r in report["results"]}
    with wave.open(manifest["audio_path"], "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected PCM16 mono 16kHz")
        for index, case in enumerate(manifest["cases"]):
            if args.max_cases is not None and index >= args.max_cases:
                break
            if case["id"] in completed:
                continue
            start, end = case["window_ms"]
            audio.setpos(start * 16)
            samples = np.frombuffer(audio.readframes((end - start) * 16), dtype="<i2").astype(np.float32) / 32768
            corrections.clear()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            row = {"id": case["id"], "text": case["text"], "window_ms": case["window_ms"], "units": [],
                   "runner_sha256": file_hash(__file__)}
            try:
                if args.backend == "qwen":
                    language = {"ja": "Japanese", "zh": "Chinese", "en": "English", "ko": "Korean"}[case["language"]]
                    aligned = model.align(audio=(samples, 16000), text=case["text"], language=language)[0]
                    row["units"] = [{"text": unit.text, "start": unit.start_time, "end": unit.end_time} for unit in aligned]
                    row["timestamp_corrections"] = list(corrections)
                else:
                    native = whisperx.align([{"start": 0, "end": len(samples) / 16000, "text": case["text"]}],
                                    model, metadata, samples, "cuda", return_char_alignments=True)
                    row["raw"] = native
                    # Char evidence predates word-level interpolation. Keep missing
                    # char times visible; do not adopt nearest-filled word times.
                    row["units"] = [{"text": char["char"], "start": char.get("start"), "end": char.get("end"),
                                      "score": char.get("score")}
                                    for segment in native["segments"] for char in (segment.get("chars") or [])
                                    if char["char"].isalnum()]
                torch.cuda.synchronize()
            except Exception:
                row["error"] = traceback.format_exc()
            row["seconds"] = time.perf_counter() - started
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["results"].append(row)
            save()
            print(f'{args.backend}: {index + 1}/{len(manifest["cases"])} units={len(row["units"])} error={bool(row.get("error"))}', flush=True)
    report["status"] = "complete" if len(report["results"]) == len(manifest["cases"]) else "partial"
    save()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        if FAILURE_STATE:
            report = FAILURE_STATE["report"]
            report.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                          fatal_error=traceback.format_exc())
            FAILURE_STATE["path"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
