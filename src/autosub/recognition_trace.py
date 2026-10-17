"""Observational decoder evidence, committed with recognition checkpoints.

Only emitted segments are observable: rejected fallback attempts and skipped
silent decoder windows are NOT exposed by faster-whisper's public API.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from uuid import uuid4

from . import __version__


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(value):
    """JSON evidence only; unknown library objects remain unknown."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return None


def options_evidence(options):
    result = safe(options)
    if isinstance(result, dict):
        for key in ("initial_prompt", "prefix", "hotwords"):
            if result.get(key) is not None:
                result[key] = {"sha256": digest(result[key]), "content_saved": False}
    return result


@lru_cache(maxsize=1)
def versions():
    result = {}
    for name in ("faster-whisper", "ctranslate2"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def words_digest(words):
    return digest(safe([[w.start_ms, w.end_ms, w.text, w.probability] for w in words]))


class RecognitionTrace:
    def __init__(self, project, model, compute_type, audio, options, info,
                 input_start_ms, input_end_ms):
        engine = getattr(model, "model", None)
        self.context = {
            "action": "asr_trace_context", "policy": "decoder-evidence-v1",
            "trace_id": uuid4().hex, "strategy": project.asr_strategy,
            "app_version": __version__,
            "source_fingerprint": safe(project.source_fingerprint),
            "speech_regions_sha256": digest(project.speech),
            "model_requested": project.model, "model_weights_sha256": None,
            "device_requested": project.device, "compute_type_requested": compute_type,
            "device_observed": safe(getattr(engine, "device", None)),
            "compute_type_observed": safe(getattr(engine, "compute_type", None)),
            "decoder_frames_per_second": safe(getattr(model, "frames_per_second", None)),
            "packages": dict(versions()),
            "input": {"start_ms": input_start_ms, "end_ms": input_end_ms,
                      "start_sample": round(input_start_ms * 16),
                      "sample_rate": 16000, "sample_count": len(audio),
                      "dtype": str(audio.dtype),
                      "samples_sha256": hashlib.sha256(audio.tobytes()).hexdigest()},
            "requested_options": options_evidence(options),
            "resolved_options": options_evidence(getattr(info, "transcription_options", None)),
            "resolved_vad_options": safe(getattr(info, "vad_options", None)),
            "language_observed": safe(getattr(info, "language", None)),
            "language_probability": safe(getattr(info, "language_probability", None)),
            "duration_after_vad": safe(getattr(info, "duration_after_vad", None)),
            "rejected_fallback_attempts": None,
            "observation_scope": "emitted-segments-only",
        }
        self.saved = False

    def segment(self, segment):
        result = {key: safe(getattr(segment, key, None)) for key in (
            "id", "seek", "start", "end", "temperature", "avg_logprob",
            "compression_ratio", "no_speech_prob")}
        text = getattr(segment, "text", None)
        result["text_sha256"] = digest(text) if isinstance(text, str) else None
        result["word_count"] = len(segment.words or [])
        return result

    def checkpoint(self, segments, words, **checkpoint):
        events = []
        if not self.saved:
            events.append(self.context)
            self.saved = True
        events.append({"action": "asr_trace_checkpoint", "policy": "decoder-evidence-v1",
                       "trace_id": self.context["trace_id"], "checkpoint": checkpoint,
                       "segments": list(segments), "accepted_word_count": len(words),
                       "accepted_words_sha256": words_digest(words)})
        return events
