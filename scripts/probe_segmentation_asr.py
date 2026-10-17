"""Controlled local ASR diagnosis; references are never inputs to recognition."""
import argparse
import json
import time
import copy
import math
from dataclasses import asdict
from pathlib import Path

from autosub.domain import Project, Word
from autosub.media import load_audio
from autosub.runtime import check_runtime
from autosub.timing import apply_timing
from autosub.export import export_subtitles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", default=32, type=int)
    parser.add_argument("--no-vad-only", action="store_true")
    parser.add_argument("--temperature-zero", action="store_true", help="Disable sampled decoding fallback for diagnosis")
    args = parser.parse_args()
    project = Project.load(args.project)
    if project.language != "ja" or project.asr_strategy not in {"auto", "fixed30-v1"}:
        raise ValueError("此診斷只比較固定區段的單語日文專案")
    if not args.model.is_dir():
        raise ValueError("請提供已下載的本機模型資料夾")
    if args.seconds <= 0:
        raise ValueError("秒數必須大於零")
    if args.output.exists():
        raise ValueError("診斷目錄已存在，請指定新位置")
    args.output.mkdir(parents=True)
    audio = load_audio(Path(project.audio_path))[:args.seconds * 16000]
    check_runtime("cuda")
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model.resolve()), device="cuda", compute_type="float16", local_files_only=True)
    results = []
    for use_vad in ((False,) if args.no_vad_only else (True, False)):
        started = time.perf_counter()
        words, rows = [], []
        # Mirror production 30s ownership + 2s context. Silero still determines
        # whether a checkpoint contains speech; only Whisper's internal silence
        # removal changes. Report window-relative segment times explicitly.
        for chunk in range(math.ceil(len(audio) / (30 * 16000))):
            lo, hi = chunk * 30000, min((chunk + 1) * 30000, len(audio) // 16)
            if not any(s < hi and e > lo for s, e in project.speech):
                continue
            start, end = max(0, lo - 2000), min(len(audio) // 16, hi + 2000)
            segments, info = model.transcribe(audio[start * 16:end * 16], language="ja", word_timestamps=True,
                vad_filter=use_vad, vad_parameters={"threshold": project.vad_threshold,
                    "min_speech_duration_ms": 100, "min_silence_duration_ms": 100, "speech_pad_ms": 200},
                initial_prompt=project.glossary.strip() or None, condition_on_previous_text=False, beam_size=5,
                **({"temperature": 0.0} if args.temperature_zero else {}))
            local = [asdict(s) for s in segments]
            rows.append({"chunk": chunk, "input_start_ms": start, "input_end_ms": end, "segments": local})
            for segment in local:
                for w in segment["words"] or []:
                    a, b = max(0, round(start + w["start"] * 1000)), min(project.duration_ms, round(start + w["end"] * 1000))
                    if lo <= (a + b) / 2 < hi and a <= b:
                        words.append(Word(a, b, w["word"], w["probability"]))
            print(f"vad={use_vad}, checkpoint={chunk + 1}, words={len(words)}", flush=True)
        result = {"vad_filter": use_vad, "elapsed_seconds": time.perf_counter() - started,
                  "source_project": str(args.project.resolve()), "model_path": str(args.model.resolve()),
                  "temperature_zero": args.temperature_zero,
                  "sampled_fallback_used": any(s["temperature"] > 0 for window in rows for s in window["segments"]),
                  "processed_seconds": len(audio) / 16000, "windows": rows}
        results.append(result)
        label = "vad" if use_vad else "no-vad"
        (args.output / (label + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        p = copy.deepcopy(project)
        p.words = sorted(words, key=lambda w: (w.start_ms, w.end_ms))
        p.cues, p.review_events = [], []
        p.completed_chunks = list(range(math.ceil(len(audio) / (30 * 16000))))
        p.status = "complete" if len(audio) // 16 >= p.duration_ms else "cancelled"
        p.cues = apply_timing(p)
        p.save(args.output / (label + ".autosub.json"))
        export_subtitles(p, args.output / (label + ".ass"))
        print(json.dumps({"vad": use_vad, "text": "".join(w.text for w in p.words)}, ensure_ascii=False), flush=True)
    (args.output / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
