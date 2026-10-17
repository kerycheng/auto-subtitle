import copy
import json
import wave
from pathlib import Path
from threading import Event

import pytest

from autosub.comparison import compare_file, ComparisonSettings, load_session, save_session, session_key, set_decision, project_signature
from autosub.domain import Project, Cue
from autosub.media import Cancelled, fingerprint
from autosub.paired_evaluation import annotation_template
from autosub.project_tools import clear_alignment_cache, migrate_review_identity, relink_media


def wav(path, value=0):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(int(value).to_bytes(2, "little", signed=True) * 16000)


def test_relink_compares_actual_audio_and_preserves_locked_edits(tmp_path):
    original, moved = tmp_path / "original.wav", tmp_path / "moved.wav"
    wav(original)
    moved.write_bytes(original.read_bytes())
    p = Project(media_path="missing-source.wav", audio_path=str(original), duration_ms=1000,
                cues=[Cue("a", 0, 900, "人工修正", 0, 800, locked=True, reviewed=True, candidate={"old": True})])
    before = copy.deepcopy(p)
    result = relink_media(p, tmp_path / "project.autosub.json", moved, Event())
    assert p == before
    assert result.media_path == str(moved.resolve()) and result.source_fingerprint == fingerprint(moved)
    assert result.cues[0].text == "人工修正" and result.cues[0].locked and result.cues[0].reviewed
    assert result.cues[0].candidate is None
    assert not list((tmp_path / "project.autosub.cache").glob("relink-*"))
    wav(moved, 100)
    with pytest.raises(ValueError, match="不同"):
        relink_media(p, tmp_path / "project.autosub.json", moved, Event())


def test_relink_can_find_moved_cache_and_cancel(tmp_path):
    path = tmp_path / "project.autosub.json"
    cache = tmp_path / "project.autosub.cache"
    cache.mkdir()
    wav(cache / "audio.wav")
    source = tmp_path / "new.wav"
    wav(source)
    p = Project(audio_path=str(tmp_path / "gone" / "audio.wav"), duration_ms=1000)
    result = relink_media(p, path, source, Event())
    assert result.audio_path == str((cache / "audio.wav").resolve())
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        relink_media(p, path, source, cancel)


def test_cache_removal_preserves_audio_progress_and_review(tmp_path):
    path = tmp_path / "project.autosub.json"
    root = tmp_path / "project.autosub.cache"
    alignment = root / "alignment"
    alignment.mkdir(parents=True)
    (alignment / "one.json").write_text("{}")
    (alignment / "keep.txt").write_text("keep")
    (root / "audio.wav").write_bytes(b"keep")
    path.write_text("progress")
    count, size = clear_alignment_cache(path)
    assert count == 1 and size == 2
    assert (root / "audio.wav").read_bytes() == b"keep"
    assert path.read_text() == "progress" and (alignment / "keep.txt").exists()


def test_verified_relink_rebases_review_sidecars_without_deleting_originals(tmp_path):
    old = Project(media_path="old", duration_ms=3000, cues=[Cue("a", 1000, 2000, "A", 1000, 1800)])
    new = copy.deepcopy(old)
    new.media_path = "new"
    new.source_fingerprint = {"path": "new", "size": 1, "mtime_ns": 1}
    ref = tmp_path / "ref.srt"
    ref.write_text("1\n00:00:01,000 --> 00:00:02,000\n翻譯\n", encoding="utf-8")
    session = compare_file(old, ref, ComparisonSettings(timeline_confirmed=True))
    set_decision(session, 0, "usable", "reviewer")
    store = tmp_path / "project.autosub.review"
    save_session(session, store / f"comparison-{session_key(session)}.json")
    gold = annotation_template(old)
    old_annotation = store / f"acoustic-{project_signature(old)[:16]}.json"
    old_annotation.write_text(json.dumps(gold), encoding="utf-8")
    assert migrate_review_identity(old, new, tmp_path / "project.autosub.json") == 2
    assert old_annotation.exists()
    candidates = [p for p in store.glob("comparison-*.json") if json.loads(p.read_text(encoding="utf-8"))["project_signature"] == project_signature(new)]
    assert load_session(candidates[0], new)["rows"][0]["decision"] == "usable"
