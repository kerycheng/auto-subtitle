"""Keep decoder sentences and checkpoint only complete decoder batches.

VAD excludes long nonspeech gaps with context; it never removes short pauses
inside a clip. Whisper receives the original audio and source-clock clips.
"""
from __future__ import annotations

import math
import os

from .domain import Word
from .media import check_cancel
from .model_cache import get_model
from .runtime import check_runtime
from .recognition_trace import RecognitionTrace


def source_clips(speech, duration_ms, resume_ms=0):
    clips = []
    for start, end in speech:
        start, end = max(0, start - 1000), min(duration_ms, end + 1000)
        # Padding consumes 2s, so a remaining gap >3s means >5s raw silence.
        if clips and start - clips[-1][1] <= 3000:
            clips[-1][1] = max(clips[-1][1], end)
        else:
            clips.append([start, end])
    return [value / 1000 for start, end in clips if end > resume_ms
            for value in (max(start, resume_ms), end)]


def transcribe_native(project, audio, path, cancel, progress, factory):
    project.validate()
    if project.native_complete:
        return
    clips = source_clips(project.speech, project.duration_ms, project.native_resume_ms)
    if not clips:
        raise ValueError("無可接續的語音區間，請檢查專案進度")
    progress(8, "檢查運算裝置與精度支援…")
    runtime = check_runtime(project.device)
    progress(10, "載入辨識模型；保留原生句界與音訊上下文…")
    model, _ = get_model(factory, project.model, project.device, runtime["compute_type"],
                         os.environ.get("AUTOSUB_MODEL_DIR") or None)
    check_cancel(cancel)
    fps = model.frames_per_second
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("辨識器未提供有效的解碼進度")
    options = dict(
        language="ja", multilingual=False, word_timestamps=True,
        vad_filter=False, clip_timestamps=clips,
        initial_prompt=project.glossary.strip() or None,
        condition_on_previous_text=False, beam_size=5,
    )
    segments, info = model.transcribe(audio, **options)
    trace = RecognitionTrace(project, model, runtime["compute_type"], audio, options, info,
                             0, project.duration_ms)
    pending, batch_start = [], None
    pending_trace = []

    def commit(next_ms, complete=False):
        first_word = len(project.words)
        accepted = [word for group in pending for word in group]
        for group in pending:
            start = len(project.words)
            project.words.extend(group)
            project.native_groups.append(list(range(start, start + len(group))))
        pending.clear()
        project.review_events.extend(trace.checkpoint(
            pending_trace, accepted, kind="native-batch", resume_from_ms=project.native_resume_ms,
            resume_to_ms=next_ms, first_word_index=first_word, complete=complete))
        pending_trace.clear()
        project.native_resume_ms = next_ms
        project.native_complete = complete
        project.status = "transcribing"
        project.save(path)
        progress(10 + round(85 * next_ms / project.duration_ms),
                 f"保留原生句界 · 已保存至 {next_ms / 1000:.1f} 秒 · {len(project.native_groups)} 句")

    for segment in segments:
        check_cancel(cancel)
        seek_ms = round(segment.seek * 1000 / fps)
        if not project.native_resume_ms <= seek_ms <= project.duration_ms or (
                batch_start is not None and seek_ms < batch_start):
            raise ValueError("辨識器解碼進度倒退或超出媒體範圍，已保留上一批進度")
        if batch_start is not None and seek_ms != batch_start:
            commit(seek_ms)
            check_cancel(cancel)
        batch_start = seek_ms
        group = []
        for w in segment.words or []:
            values = (w.start, w.end, w.probability)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("辨識器回傳無效時間或信心值")
            start, end = round(w.start * 1000), round(w.end * 1000)
            if not 0 <= start <= end <= project.duration_ms:
                raise ValueError("辨識器詞級時間超出媒體範圍，已保留上一批進度")
            group.append(Word(start, end, w.word, float(w.probability)))
        if segment.text.strip() and not group:
            raise ValueError("辨識句子缺少詞級時間，已保留上一批進度")
        if group:
            pending.append(group)
        pending_trace.append(trace.segment(segment))
    check_cancel(cancel)
    commit(project.duration_ms, complete=True)
