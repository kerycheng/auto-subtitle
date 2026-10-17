"""Local ASR-only experiment on identical windows. No reference text is read."""
import argparse
import json
import os
import sys
import time
import traceback
import wave
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity, validate_resume, validate_input


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=["qwen", "whisper"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-file", type=Path, help="Optional fixed domain vocabulary, never reference transcription")
    parser.add_argument("--temperature-zero", action="store_true", help="Whisper-only controlled ablation without sampling fallback")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if file_hash(manifest["audio_path"]) != manifest["audio_sha256"] or manifest["text_provided"]:
        raise ValueError("Invalid ASR-only input")
    validate_input(manifest)
    prompt = None
    if args.prompt_file:
        if args.backend != "whisper":
            raise ValueError("Vocabulary ablation currently supports Whisper only")
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt or len(prompt) > 500:
            raise ValueError("Vocabulary prompt must contain 1-500 characters")
    configuration = {"device": "cuda", "dtype": "float16", "batch_size": 1,
        "language": "specified per window", "prompt": prompt, "vad_filter": False,
        "backend_options": ({"word_timestamps": True, "beam_size": 5,
            "condition_on_previous_text": False} if args.backend == "whisper" else
            {"attention": "eager", "max_new_tokens": 512, "return_time_stamps": False})}
    if args.backend == "qwen":
        if args.temperature_zero:
            raise ValueError("Temperature control currently supports Whisper only")
        configuration["backend_options"]["weight_loading"] = "CPU FP16 staging then CUDA FP16"
    elif args.temperature_zero:
        configuration["backend_options"]["temperature"] = 0.0
    identity = model_identity(args.model)
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    signature = file_hash(args.manifest)
    report = {"manifest_sha256": signature, "backend": args.backend, "status": "running", "results": [],
              "model": str(args.model.resolve()), "text_provided": False,
              "model_identity": identity, "configuration": configuration,
              "environment": {"python": sys.version, "platform": platform.platform(),
                  "packages": {name: version(name) for name in
                      (["qwen-asr", "torch", "transformers"] if args.backend == "qwen" else
                       ["faster-whisper", "ctranslate2"])}}}
    if args.resume and result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        validate_resume(report, signature, args.backend, identity, configuration, manifest)
        if report.get("status") == "complete" and len(report["results"]) == len(manifest["windows"]):
            print("Already complete; preserved existing results without loading the model", flush=True)
            return
    def save():
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, result_path)
    save()
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        report["gpu_before_load"] = gpu
        free_mb = int(gpu.splitlines()[0].split(",")[1].strip())
        if free_mb < (4500 if args.backend == "qwen" else 3500):
            raise RuntimeError("Insufficient free GPU memory; preserve other running applications and retry later")
        save()
        import numpy as np
        started = time.perf_counter()
        if args.backend == "qwen":
            import torch
            from qwen_asr import Qwen3ASRModel
            torch.set_num_threads(4)
            # Bound this research process rather than evicting another application's model.
            # PyTorch's allocator cap excludes the CUDA context and other applications.
            tensor_budget = min(4608, free_mb - 512) * 1024 * 1024
            torch.cuda.set_per_process_memory_fraction(tensor_budget / torch.cuda.get_device_properties(0).total_memory)
            report["tensor_allocator_limit_bytes"] = tensor_budget
            save()
            # CPU staging avoids temporarily retaining a checkpoint shard plus converted
            # parameters on CUDA; inference stays entirely on the specified CUDA device.
            model = Qwen3ASRModel.from_pretrained(str(args.model.resolve()), device_map="cpu",
                dtype=torch.float16, attn_implementation="eager", local_files_only=True,
                max_inference_batch_size=1, max_new_tokens=512)
            model.model.to("cuda:0")
            model.device = model.model.device
        else:
            from autosub.runtime import check_runtime
            check_runtime("cuda")
            from faster_whisper import WhisperModel
            model = WhisperModel(str(args.model.resolve()), device="cuda", compute_type="float16", local_files_only=True)
        report["load_seconds"] = time.perf_counter() - started
        report.setdefault("load_attempts_seconds", []).append(report["load_seconds"])
        done = {r["id"] for r in report["results"]}
        with wave.open(manifest["audio_path"], "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Expected PCM16 mono 16kHz")
            for i, window in enumerate(manifest["windows"]):
                if window["id"] in done:
                    continue
                a, b = window["window_ms"]
                wav.setpos(a * 16)
                samples = np.frombuffer(wav.readframes((b - a) * 16), dtype="<i2").astype(np.float32) / 32768
                started = time.perf_counter()
                row = {**window}
                language = window.get("language") or manifest["language"]
                if args.backend == "qwen":
                    torch.cuda.reset_peak_memory_stats()
                    name = {"ja": "Japanese", "zh": "Chinese", "en": "English", "ko": "Korean"}[language]
                    result = model.transcribe(audio=(samples, 16000), language=name, return_time_stamps=False)[0]
                    torch.cuda.synchronize()
                    row.update(text=result.text, detected_language=result.language,
                               peak_tensor_bytes=torch.cuda.max_memory_allocated())
                else:
                    from dataclasses import asdict
                    decoding = {"temperature": 0.0} if args.temperature_zero else {}
                    segments, info = model.transcribe(samples, language=language, word_timestamps=True,
                         vad_filter=False, initial_prompt=prompt, condition_on_previous_text=False, beam_size=5, **decoding)
                    rows = [asdict(s) for s in segments]
                    row.update(text="".join(r["text"] for r in rows), segments=rows,
                               transcription_options=asdict(info.transcription_options))
                row["seconds"] = time.perf_counter() - started
                report["results"].append(row)
                save()
                print(f'{args.backend}: {i + 1}/{len(manifest["windows"])} {row["text"]}', flush=True)
        report["status"] = "complete"
        report.pop("error", None)
        report["gpu_after_inference"] = subprocess.check_output(["nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
        save()
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
