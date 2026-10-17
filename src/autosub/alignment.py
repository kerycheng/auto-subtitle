"""Opt-in, local text alignment. Analysis never edits subtitle text or timing."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import Event

from .domain import Cue, Project, word_key
from .media import Cancelled, check_cancel, fingerprint, load_audio
from .timing import apply_timing, joined, refine_boundaries

BACKEND_VERSION = "ct2-local-v1"


class StaleSourceError(ValueError):
    pass


def signature(project: Project, cue: Cue) -> str:
    data = {"source": project.source_fingerprint, "track": project.audio_stream,
            "duration": project.duration_ms, "language": project.language,
            "model": project.model, "device": project.device, "vad": project.speech,
            "words": [asdict(w) for w in project.words
                      if cue.source_tokens is None or word_key(w) in cue.source_tokens],
            "settings": asdict(project.settings), "policy": project.boundary_policy,
            "strategy": project.asr_strategy, "native_groups": project.native_groups,
            "cues": [{k: v for k, v in vars(c).items()
                      if k not in {"candidate", "candidate_original", "reviewed", "flags", "timing_trace"}}
                     for c in project.cues], "selected": cue.id}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LocalWhisperAligner:
    """Reuse an already downloaded CTranslate2 model; no additional model download."""
    def __init__(self, model: str, device: str):
        from faster_whisper.utils import download_model
        path = Path(model)
        if not path.is_dir():
            path = Path(download_model(model, local_files_only=True,
                                       cache_dir=os.environ.get("AUTOSUB_MODEL_DIR")))
        files = [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(path.rglob("*")) if p.is_file()]
        self.identity = {"backend": BACKEND_VERSION, "faster_whisper": version("faster-whisper"),
                         "ctranslate2": version("ctranslate2"), "model": str(path.resolve()),
                         "files": files, "device": device, "compute": "int8" if device == "cpu" else "float16"}
        self.path, self.device, self.model = path, device, None

    def align(self, audio, text, language):
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.audio import pad_or_trim
        if self.model is None:
            self.model = WhisperModel(str(self.path), device=self.device,
                                      compute_type=self.identity["compute"], local_files_only=True)
        model = self.model
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                              task="transcribe", language=language)
        tokens = tokenizer.encode(text)
        if not 0 < len(tokens) <= 400:
            raise ValueError("文字過長或無可對齊文字，請先拆分")
        features = model.feature_extractor(audio)
        frames = min(features.shape[-1], model.feature_extractor.nb_max_frames)
        encoded = model.encode(pad_or_trim(features))
        return model.find_alignment(tokenizer, [tokens], encoded, frames)[0]


def conservative_base(project: Project, cue: Cue):
    words = [w for w in project.words if word_key(w) in (cue.source_tokens or [])]
    if not cue.locked and words and joined(words) == cue.text:
        conservative = copy.copy(project)
        conservative.boundary_policy = "conservative"
        a, b, reasons = refine_boundaries(conservative, words)
        return [a, b], reasons
    return [cue.speech_start_ms, cue.speech_end_ms], ["人工修正／詞對應不足，保留目前基礎邊界"]


def candidate_display(project: Project, index: int, base: list[int]) -> list[int]:
    working = copy.deepcopy(project.cues)
    for cue in working:
        cue.locked = True
    selected = working[index]
    selected.locked = False
    selected.speech_start_ms, selected.speech_end_ms = base
    result = apply_timing(project, working)
    selected = next(c for c in result if c.id == selected.id)
    return [selected.start_ms, selected.end_ms]


def analyze_cue(project: Project, index: int, cache_dir: Path, cancel: Event,
                language: str | None = None, backend=None) -> dict:
    """Return checkpointable candidate; rejected inference falls back conservatively."""
    cue = project.cues[index]
    base, reasons = conservative_base(project, cue)
    result = {"signature": signature(project, cue), "status": "fallback", "base": base,
              "reasons": reasons, "backend": BACKEND_VERSION, "experimental": True}
    language = language or project.language
    result["language"] = language
    try:
        check_cancel(cancel)
        if language not in {"ja", "zh", "en", "ko"}:
            raise ValueError("自動／混合語言專案請指定本句的對齊語言")
        source = fingerprint(Path(project.media_path))
        if project.source_fingerprint and source != project.source_fingerprint:
            raise StaleSourceError("來源檔案已變更，請建立新專案，拒絕使用舊音訊快取")
        audio_path = Path(project.audio_path)
        if not audio_path.is_file():
            raise ValueError("缺少解碼音訊，請先完成辨識")
        check_cancel(cancel)
        backend = backend or LocalWhisperAligner(project.model, project.device)
        lo = max(0, cue.speech_start_ms - 1000)
        hi = min(project.duration_ms, cue.speech_end_ms + 1000)
        if hi - lo > 30000 or hi <= lo:
            raise ValueError("局部對齊限 30 秒，請先拆分字幕")
        key_data = {"source": source, "audio": fingerprint(audio_path), "track": project.audio_stream,
                    "offset_ms": lo, "end_ms": hi, "text": cue.text, "language": language,
                    "backend": backend.identity, "median_filter_width": 7}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache_file = Path(cache_dir) / f"{key}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                words = cached["words"] if cached["key"] == key else None
            except (ValueError, KeyError, TypeError):
                words = None
        else:
            words = None
        cached_hit = words is not None
        if words is None:
            audio = load_audio(audio_path)
            if abs(len(audio) / 16 - project.duration_ms) > 100:
                raise ValueError("解碼音訊長度與專案不符")
            check_cancel(cancel)
            words = backend.align(audio[lo * 16:hi * 16], cue.text, language)
            # CTranslate2 calls cannot be interrupted halfway. Discard that result on cancel.
            check_cancel(cancel)
            words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                      "probability": float(w["probability"])} for w in words]
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"key": key, "words": words}, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, cache_file)
        check_cancel(cancel)
        result.update(identity=backend.identity, offset_ms=lo, cache_key=key, cache_hit=cached_hit)
        if not words or "".join(w["word"] for w in words).strip() != cue.text.strip():
            raise ValueError("對齊文字不匹配")
        previous = 0.0
        for w in words:
            if not all(math.isfinite(w[k]) for k in ("start", "end", "probability")):
                raise ValueError("對齊回傳非有限數值")
            if not 0 <= w["start"] <= w["end"] <= (hi - lo) / 1000 or w["start"] < previous:
                raise ValueError("對齊時間超出區段或順序錯誤")
            previous = w["end"]
        aligned = [lo + round(words[0]["start"] * 1000), lo + round(words[-1]["end"] * 1000)]
        result["aligned_proposal"] = list(aligned)
        score = sum(w["probability"] for w in words) / len(words)
        if score < 0.35 or aligned[1] <= aligned[0]:
            raise ValueError("對齊信心不足或零時長")
        edge_reasons = []
        # DTW can anchor the first text token at frame zero, including leading
        # silence. Do not advertise that endpoint as a refined acoustic onset.
        if aligned[0] <= lo + 20:
            aligned[0] = base[0]
            edge_reasons.append("對齊句首貼齊視窗左緣，句首沿用保守估計")
        if aligned[1] >= hi - 20:
            aligned[1] = base[1]
            edge_reasons.append("對齊句尾貼齊視窗右緣，句尾沿用保守估計")
        if len(edge_reasons) == 2 or aligned[0] >= aligned[1]:
            raise ValueError("對齊兩側缺乏可靠邊界，保留保守估計")
        if max(abs(aligned[0] - cue.speech_start_ms), abs(aligned[1] - cue.speech_end_ms)) > 1000:
            raise ValueError("對齊移動超過 1000ms，疑似錯字／漏字")
        if not any(a < aligned[1] and b > aligned[0] for a, b in project.speech):
            raise ValueError("對齊範圍沒有 VAD 語音證據")
        result.update(status="partial" if edge_reasons else "aligned", base=aligned, words=words, confidence=score,
                      reasons=["局部 Whisper 文字對齊；仍需試聽確認"] + edge_reasons, cache_hit=cached_hit,
                      cache_key=key, identity=backend.identity, offset_ms=lo)
    except (Cancelled, StaleSourceError):
        raise
    except Exception as exc:
        result["reasons"].append(f"精細對齊未採用：{exc}")
    result["display"] = candidate_display(project, index, result["base"])
    return result


def apply_candidate(project: Project, index: int):
    cue = project.cues[index]
    if project.source_fingerprint and fingerprint(Path(project.media_path)) != project.source_fingerprint:
        raise StaleSourceError("來源檔案已變更，拒絕採用舊候選")
    candidate = cue.candidate
    if not candidate or candidate["signature"] != signature(project, cue):
        raise ValueError("候選已過期，請重新分析選取句")
    for kind in ("base", "display"):
        interval = candidate.get(kind)
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(t) is not int for t in interval)
                or not 0 <= interval[0] < interval[1] <= project.duration_ms):
            raise ValueError("候選時間無效，請重新分析")
    if cue.candidate_original is None:
        cue.candidate_original = {k: copy.deepcopy(v) for k, v in asdict(cue).items()
                                  if k not in {"candidate", "candidate_original"}}
    cue.speech_start_ms, cue.speech_end_ms = candidate["base"]
    cue.start_ms, cue.end_ms = candidate["display"]
    cue.locked, cue.reviewed = True, False
    cue.timing_trace = {**cue.timing_trace, "alignment_candidate": copy.deepcopy(candidate),
                        "base": candidate["base"], "display": candidate["display"]}
    cue.flags = list(dict.fromkeys(cue.flags + candidate["reasons"]))
    # Keep manually split fragments as one ownership group.
    if cue.edit_group:
        for member in project.cues:
            if member.edit_group == cue.edit_group:
                member.locked = True


def restore_candidate(project: Project, index: int):
    cue = project.cues[index]
    if cue.candidate_original is None:
        raise ValueError("此句尚未採用候選")
    # Restoring only the selected cue must not partially unlock a split group.
    original = Cue(**copy.deepcopy(cue.candidate_original))
    if original.edit_group:
        original.locked = True
    project.cues[index] = original
