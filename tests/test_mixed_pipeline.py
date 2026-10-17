import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from autosub.domain import Project, Word
from autosub.media import Cancelled
from autosub.pipeline import run_pipeline
from autosub.recognition_windows import mixed_windows


def test_mixed_windows_merge_short_pauses_and_separate_languages_by_silence():
    assert mixed_windows([[1000, 5000], [5300, 6000], [9000, 10000]], 12000) == [
        (750, 6250, 0, 7500), (8750, 10250, 7500, 12000)]
    assert mixed_windows([[0, 40000]], 40000) == [(0, 15250, 0, 15000),
        (14750, 30250, 15000, 30000), (29750, 40000, 30000, 40000)]


def fake_setup(tmp_path, monkeypatch, cancel=None, cancel_at=2):
    import autosub.pipeline as pipeline
    import faster_whisper
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    monkeypatch.setattr(pipeline, "decode_audio", lambda *_: None)
    monkeypatch.setattr(pipeline, "load_audio", lambda _: np.arange(31 * 16000, dtype=np.float32))
    targets = [(14900, 15200, "境界"), (29900, 30150, "跨區"), (30200, 30800, "한국어")]
    calls = []
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, audio, **kwargs):
            calls.append((len(audio), kwargs))
            if cancel is not None and len(calls) == cancel_at:
                cancel.set()
            offset = float(audio[0]) / 16
            end = offset + len(audio) / 16
            words = [SimpleNamespace(start=(a - offset)/1000, end=(b-offset)/1000, word=text, probability=.9)
                     for a, b, text in targets if offset <= a and b <= end]
            return iter([SimpleNamespace(words=words)]), None
    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    p = Project(media_path=str(source), language="mixed", speech=[[100, 31000]])
    return p, tmp_path / "mixed.autosub.json", calls, targets


def test_words_at_utterance_and_checkpoint_boundaries_are_owned_once(tmp_path, monkeypatch):
    p, path, calls, targets = fake_setup(tmp_path, monkeypatch)
    result = run_pipeline(p, path, Event(), lambda *_: None)
    assert result.asr_strategy == "mixed-vad-v1" and result.completed_chunks == [0, 1]
    assert [(w.start_ms, w.end_ms, w.text) for w in result.words] == targets
    assert len(calls) == 4 and all(length <= 15500 * 16 for length, _ in calls)
    assert all(options["multilingual"] and options["language"] is None for _, options in calls)


@pytest.mark.parametrize("cancel_at", [2, 3])
def test_cancel_commits_no_partial_checkpoint_and_keeps_completed_prefix(tmp_path, monkeypatch, cancel_at):
    cancel = Event()
    p, path, _, targets = fake_setup(tmp_path, monkeypatch, cancel, cancel_at)
    with pytest.raises(Cancelled):
        run_pipeline(p, path, cancel, lambda *_: None)
    saved = Project.load(path)
    assert saved.status == "cancelled"
    assert saved.completed_chunks == ([] if cancel_at == 2 else [0])
    assert [(word.start_ms, word.end_ms, word.text) for word in saved.words] == ([] if cancel_at == 2 else targets[:1])
    assert saved.asr_strategy == "mixed-vad-v1"
    resumed = run_pipeline(saved, path, Event(), lambda *_: None)
    assert [(w.start_ms, w.end_ms, w.text) for w in resumed.words] == targets


def test_schema2_migration_preserves_legacy_strategy_and_exact_backup(tmp_path):
    path = tmp_path / "legacy.autosub.json"
    Project(language="mixed", words=[Word(100, 500, "旧")], duration_ms=1000, completed_chunks=[0]).save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw.pop("asr_strategy")
    original = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)
    migrated = Project.load(path)
    assert migrated.schema_version == 4 and migrated.asr_strategy == "fixed30-v1"
    assert path.read_bytes() == original
    migrated.save(path)
    assert path.with_suffix(".json.v2.bak").read_bytes() == original


def test_legacy_partial_mixed_job_keeps_its_recognition_strategy(tmp_path, monkeypatch):
    p, path, calls, _ = fake_setup(tmp_path, monkeypatch)
    p.asr_strategy = "fixed30-v1"
    p.words = [Word(100, 500, "旧")]
    p.completed_chunks = [0]
    result = run_pipeline(p, path, Event(), lambda *_: None)
    assert result.asr_strategy == "fixed30-v1" and len(calls) == 1
    assert calls[0][0] == 3 * 16000
    assert result.words[0] == Word(100, 500, "旧")
