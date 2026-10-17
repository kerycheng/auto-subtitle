from __future__ import annotations

import math
import os
import hashlib
import json
from pathlib import Path
from threading import Event
from typing import Callable

from .domain import Project, Word
from .media import Cancelled, check_cancel, decode_audio, fingerprint, load_audio
from .timing import apply_timing
from .runtime import check_runtime, runtime_error
from .model_cache import get_model, clear_model
from .recognition_windows import mixed_windows
from .speech_detection import detect_speech, quiet_retry_applied, external_vad_clips
from .recognition_trace import RecognitionTrace

CHUNK_SECONDS = 30  # fixed grids make interrupted runs deterministic
CONTEXT_SECONDS = 2


def run_pipeline(project: Project, project_path: Path, cancel: Event,
                 progress: Callable[[int, str], None]) -> Project:
    """Silero + Whisper with strategy-specific atomic checkpoints.

    Japanese native sentences retain decoder batches; legacy/mixed strategies use
    30s ownership intervals. All timestamps stay on the source clock.
    """
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    source = Path(project.media_path)
    project.settings.validate()
    retry_quiet = (not project.native_complete and not project.words
                   and not project.completed_chunks and project.status != "complete")
    source_key = fingerprint(source)
    if project.source_fingerprint and project.source_fingerprint != source_key:
        raise ValueError("來源檔案已改變，請建立新專案以避免套用錯誤快取")
    project.source_fingerprint = source_key
    if project.asr_strategy == "auto":
        fresh = not project.words and not project.completed_chunks
        project.asr_strategy = ("native-v1" if fresh and project.language == "ja" else
                                "mixed-vad-v1" if fresh and project.language == "mixed" else "fixed30-v1")
    cache_key = hashlib.sha256(json.dumps([source_key, project.audio_stream], sort_keys=True).encode()).hexdigest()[:16]
    audio_path = project_path.parent / (project_path.stem + ".cache") / f"{cache_key}.wav"
    project.audio_path = str(audio_path.resolve())
    try:
        check_cancel(cancel)
        if not audio_path.exists():
            project.status = "decoding"
            project.save(project_path)
            progress(2, "正在解碼選定音軌…")
            decode_audio(source, audio_path, project.audio_stream, cancel)
        audio = load_audio(audio_path)
        project.duration_ms = round(len(audio) / 16)
        if not len(audio):
            raise ValueError("音訊為空")
        check_cancel(cancel)
        if not project.speech and project.status != "vad_complete":
            project.status = "vad"
            progress(5, "Silero VAD 分析語音邊界…")
            stamps, analysis = detect_speech(audio, VadOptions(
                threshold=project.vad_threshold, min_speech_duration_ms=100,
                min_silence_duration_ms=100, speech_pad_ms=0), get_speech_timestamps,
                retry_quiet=retry_quiet)
            if analysis:
                project.review_events.append(analysis)
                progress(6, "音量偏低，已加強 VAD 分析；辨識與試聽保留原音訊")
            project.speech = [[round(s["start"] / 16), round(s["end"] / 16)] for s in stamps]
            project.status = "vad_complete"
            project.save(project_path)
        check_cancel(cancel)
        if not project.speech:
            if project.asr_strategy == "native-v1":
                project.native_complete = True
                project.native_resume_ms = project.duration_ms
            project.status = "complete"
            project.cues = []
            project.save(project_path)
            progress(100, "未偵測到語音，未產生字幕")
            return project
        if project.asr_strategy == "native-v1":
            from .native_pipeline import transcribe_native
            transcribe_native(project, audio, project_path, cancel, progress, WhisperModel)
        total = math.ceil(len(audio) / (CHUNK_SECONDS * 16000))
        utterances = mixed_windows(project.speech, project.duration_ms) if (
            project.language == "mixed" and project.asr_strategy == "mixed-vad-v1") else None
        model = None
        for chunk in range(0 if project.asr_strategy == "native-v1" else total):
            check_cancel(cancel)
            if chunk in project.completed_chunks:
                continue
            lo, hi = chunk * CHUNK_SECONDS, min((chunk + 1) * CHUNK_SECONDS, len(audio) / 16000)
            if not any(s < hi * 1000 and e > lo * 1000 for s, e in project.speech):
                project.completed_chunks.append(chunk)
                project.save(project_path)
                continue
            if model is None:
                progress(8, "檢查運算裝置與精度支援…")
                runtime = check_runtime(project.device)
                progress(10, "載入辨識模型；尚未快取時會下載，取消將於載入後生效…")
                model_cache = os.environ.get("AUTOSUB_MODEL_DIR")
                model, reused = get_model(WhisperModel, project.model, project.device,
                                          runtime["compute_type"], model_cache or None)
                if reused:
                    progress(10, "沿用已載入的辨識模型…")
                check_cancel(cancel)
            progress(10 + round(85 * chunk / total), f"辨識區段 {chunk + 1}/{total}（{lo:.0f}–{hi:.0f} 秒）…")
            pending = []
            pending_trace = []
            windows = [(max(0, lo * 1000 - CONTEXT_SECONDS * 1000),
                        min(project.duration_ms, hi * 1000 + CONTEXT_SECONDS * 1000), lo * 1000, hi * 1000)] if utterances is None else [
                            window for window in utterances if window[0] < hi * 1000 and window[1] > lo * 1000
                            and window[2] < hi * 1000 and window[3] > lo * 1000]
            for position, (input_start, input_end, owner_start, owner_end) in enumerate(windows):
                check_cancel(cancel)
                if utterances is not None:
                    progress(10 + round(85 * chunk / total), f"混語區段 {chunk + 1}/{total} · 語音段 {position + 1}/{len(windows)}…")
                vad_options = {"vad_filter": True, "vad_parameters": {"threshold": project.vad_threshold,
                    "min_speech_duration_ms": 100, "min_silence_duration_ms": 100, "speech_pad_ms": 200}}
                if quiet_retry_applied(project):
                    clips = external_vad_clips(project.speech, input_start, input_end)
                    if not clips:
                        continue
                    # The internal VAD would repeat the empty result on the raw
                    # quiet audio. Use verified external ranges without changing
                    # ASR samples or resetting their window-relative clock.
                    vad_options = {"vad_filter": False, "clip_timestamps": clips}
                options = dict(
                    language=None if project.language in {"auto", "mixed"} else project.language,
                    multilingual=project.language == "mixed", word_timestamps=True,
                    **vad_options,
                    initial_prompt=project.glossary.strip() or None, condition_on_previous_text=False,
                    beam_size=5,
                )
                samples = audio[round(input_start * 16):round(input_end * 16)]
                segments, info = model.transcribe(samples, **options)
                trace = RecognitionTrace(project, model, runtime["compute_type"], samples,
                                         options, info, input_start, input_end)
                observations, window_words = [], []
                for segment in segments:
                    check_cancel(cancel)
                    observations.append(trace.segment(segment))
                    for w in segment.words or []:
                        absolute_start = max(0, round(input_start + w.start * 1000))
                        absolute_end = min(project.duration_ms, round(input_start + w.end * 1000))
                        midpoint = (absolute_start + absolute_end) / 2
                        if (max(lo * 1000, owner_start) <= midpoint < min(hi * 1000, owner_end)
                                and absolute_end >= absolute_start):
                            word = Word(absolute_start, absolute_end, w.word, float(w.probability))
                            pending.append(word)
                            window_words.append(word)
                pending_trace.extend(trace.checkpoint(
                    observations, window_words, kind="chunk-window", chunk=chunk,
                    owner_start_ms=max(lo * 1000, owner_start),
                    owner_end_ms=min(hi * 1000, owner_end)))
            check_cancel(cancel)
            # Commit only a whole chunk; cancellation never leaves a half-owned interval.
            project.words.extend(pending)
            project.words.sort(key=lambda w: (w.start_ms, w.end_ms))
            project.completed_chunks.append(chunk)
            project.review_events.extend(pending_trace)
            project.status = "transcribing"
            project.save(project_path)
        check_cancel(cancel)
        project.cues = apply_timing(project)
        project.status = "complete"
        project.save(project_path)
        progress(100, f"完成，共 {len(project.cues)} 則字幕；請複核標記項目")
        return project
    except Cancelled:
        project.status = "cancelled"
        if project.words:
            project.cues = apply_timing(project)
        project.save(project_path)
        raise
    except Exception as exc:
        clear_model()
        project.status = "failed"
        project.save(project_path)
        if project.device == "cuda":
            raise RuntimeError(runtime_error(project.device, exc)) from exc
        raise
