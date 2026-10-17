import copy
import json
import wave
from dataclasses import asdict
from threading import Event

import pytest

from autosub.alignment import (StaleSourceError, analyze_cue, apply_candidate,
                               restore_candidate, signature)
from autosub.domain import Cue, Project, TimingSettings, Word, word_key
from autosub.editing import merge_cues, split_cue
from autosub.media import Cancelled, fingerprint
from autosub.paired_evaluation import annotation_template, paired_report
from autosub.timing import apply_timing


def prepared(tmp_path):
    source = tmp_path / "audio.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(16000 * 2 * 4))
    word = Word(1000, 2000, "テスト。")
    project = Project(media_path=str(source), audio_path=str(source), duration_ms=4000,
                      words=[word], speech=[[1100, 1800]], source_fingerprint=fingerprint(source),
                      cues=[Cue("a", 1080, 2000, word.text, 1100, 1800,
                                source_tokens=[word_key(word)])])
    return project


class Backend:
    identity = {"model": "fake-v1", "compute": "test"}

    def __init__(self, words=None):
        self.calls = 0
        self.words = words or [{"word": "テスト。", "start": 1.0, "end": 1.9, "probability": .9}]

    def align(self, audio, text, language):
        self.calls += 1
        return self.words


def test_migration_backs_up_exact_old_file_once(tmp_path):
    project = prepared(tmp_path)
    project.cues[0].locked = project.cues[0].reviewed = True
    project.cues[0].edit_group = "group"
    raw = asdict(project)
    raw["schema_version"] = 1
    raw.pop("boundary_policy")
    for key in ("start_search_ms", "end_search_ms"):
        raw["settings"].pop(key)
    for key in ("timing_trace", "candidate", "candidate_original"):
        raw["cues"][0].pop(key)
    raw["settings"]["boundary_search_ms"] = 250
    path = tmp_path / "old.autosub.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()
    migrated = Project.load(path)
    assert path.read_bytes() == before  # loading is read-only
    assert migrated.schema_version == 4 and migrated.settings.end_search_ms == 250
    assert migrated.cues[0].locked and migrated.cues[0].reviewed
    assert migrated.cues[0].edit_group == "group"
    migrated.save(path)
    backup = path.with_suffix(".json.v1.bak")
    assert backup.read_bytes() == before
    migrated.save(path)
    assert backup.read_bytes() == before


def test_conservative_tail_and_independent_search_preserve_raw(tmp_path):
    project = prepared(tmp_path)
    words = copy.deepcopy(project.words)
    legacy = apply_timing(project)[0]
    assert legacy.speech_end_ms == 1800
    project.boundary_policy = "conservative"
    project.settings.start_search_ms = 0
    candidate = apply_timing(project)[0]
    assert (candidate.speech_start_ms, candidate.speech_end_ms) == (1000, 2000)
    assert candidate.timing_trace["raw"] == [1000, 2000]
    assert candidate.timing_trace["vad"] == [1000, 1800]
    assert candidate.timing_trace["display"] == [980, 2200]
    assert project.words == words
    project.cues[0].locked = True
    assert apply_timing(project) == project.cues


@pytest.mark.parametrize("key", ["start_search_ms", "end_search_ms"])
def test_invalid_search_range(key):
    with pytest.raises(ValueError):
        TimingSettings(**{key: 1001}).validate()


def test_analysis_cache_apply_restore_and_stale_candidate(tmp_path):
    project = prepared(tmp_path)
    before = copy.deepcopy(project)
    backend = Backend()
    cache = tmp_path / "cache"
    result = analyze_cue(project, 0, cache, Event(), backend=backend)
    assert result["status"] == "aligned"
    assert result["base"] == [1100, 2000]  # 100ms context offset
    assert project == before
    again = analyze_cue(project, 0, cache, Event(), backend=backend)
    assert again["cache_hit"] and backend.calls == 1
    project.cues[0].candidate = result
    apply_candidate(project, 0)
    assert project.cues[0].locked and not project.cues[0].reviewed
    assert project.cues[0].text == before.cues[0].text
    assert project.cues[0].end_ms == 2200
    restore_candidate(project, 0)
    assert project.cues[0].start_ms == before.cues[0].start_ms
    assert not project.cues[0].locked
    project.cues[0].candidate = result
    project.settings.tail_ms = 250
    with pytest.raises(ValueError, match="過期"):
        apply_candidate(project, 0)


@pytest.mark.parametrize("change", ["text", "model", "language", "audio", "track", "offset"])
def test_alignment_cache_invalidation(tmp_path, change):
    project = prepared(tmp_path)
    backend = Backend()
    analyze_cue(project, 0, tmp_path / "cache", Event(), backend=backend)
    language = "ja"
    if change == "text":
        project.cues[0].text = "別の文字"
    elif change == "model":
        backend.identity = {"model": "fake-v2"}
    elif change == "language":
        language = "ko"
    elif change == "audio":
        # A separate decoded audio file changes its fingerprint without changing source.
        path = tmp_path / "other.wav"
        path.write_bytes((tmp_path / "audio.wav").read_bytes())
        project.audio_path = str(path)
    elif change == "track":
        project.audio_stream = 1
    else:
        project.cues[0].speech_start_ms += 100
    analyze_cue(project, 0, tmp_path / "cache", Event(), language, backend)
    assert backend.calls == 2


@pytest.mark.parametrize("words", [
    [{"word": "wrong", "start": 1., "end": 1.5, "probability": .9}],
    [{"word": "テスト。", "start": 1., "end": 1.5, "probability": .1}],
    [{"word": "テスト。", "start": 1., "end": 1., "probability": .9}],
    [{"word": "テスト。", "start": float("nan"), "end": 1.5, "probability": .9}],
    [{"word": "テスト。", "start": -1., "end": 1.5, "probability": .9}],
])
def test_rejected_alignment_falls_back_without_editing(tmp_path, words):
    project = prepared(tmp_path)
    result = analyze_cue(project, 0, tmp_path / "cache", Event(), backend=Backend(words))
    assert result["status"] == "fallback"
    assert result["base"] == [1100, 2000]
    assert project.cues[0].speech_end_ms == 1800


def test_cancel_during_inference_discards_output_then_can_resume(tmp_path):
    project = prepared(tmp_path)
    cancel = Event()
    class Cancelling(Backend):
        def align(self, *args):
            cancel.set()
            return super().align(*args)
    cache = tmp_path / "cache"
    with pytest.raises(Cancelled):
        analyze_cue(project, 0, cache, cancel, backend=Cancelling())
    assert not list(cache.glob("*.json"))
    cancel.clear()
    assert analyze_cue(project, 0, cache, cancel, backend=Backend())["status"] == "aligned"


def test_changed_source_rejected_instead_of_adoptable_fallback(tmp_path):
    project = prepared(tmp_path)
    with open(project.media_path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StaleSourceError):
        analyze_cue(project, 0, tmp_path / "cache", Event(), backend=Backend())


def test_missing_backend_and_mixed_language_fallback(tmp_path, monkeypatch):
    import autosub.alignment as alignment
    project = prepared(tmp_path)
    def missing(*args):
        raise ImportError("optional backend unavailable")
    monkeypatch.setattr(alignment, "LocalWhisperAligner", missing)
    assert analyze_cue(project, 0, tmp_path, Event())["status"] == "fallback"
    assert analyze_cue(project, 0, tmp_path, Event(), language="mixed")["status"] == "fallback"


@pytest.mark.parametrize("start,end,status", [(0., 1.9, "partial"), (1., 2.7, "partial"), (0., 2.7, "fallback")])
def test_window_edge_endpoints_use_per_edge_fallback(tmp_path, start, end, status):
    project = prepared(tmp_path)
    backend = Backend([{"word": "テスト。", "start": start, "end": end, "probability": .9}])
    result = analyze_cue(project, 0, tmp_path / "cache", Event(), backend=backend)
    assert result["status"] == status
    if start == 0.:
        assert result["base"][0] == 1100
    if end == 2.7:
        assert result["base"][1] == 2000


def test_corrupted_cache_recomputed_and_neighbor_edit_invalidates_candidate(tmp_path):
    project = prepared(tmp_path)
    backend = Backend()
    cache = tmp_path / "cache"
    result = analyze_cue(project, 0, cache, Event(), backend=backend)
    next(cache.glob("*.json")).write_text("{invalid", encoding="utf-8")
    analyze_cue(project, 0, cache, Event(), backend=backend)
    assert backend.calls == 2
    project.cues[0].candidate = result
    project.cues.append(Cue("neighbor", 2400, 3000, "Next.", 2400, 2800))
    with pytest.raises(ValueError, match="過期"):
        apply_candidate(project, 0)


def test_unusable_vad_evidence_does_not_promote_alignment(tmp_path):
    project = prepared(tmp_path)
    project.speech = []
    assert analyze_cue(project, 0, tmp_path / "cache", Event(), backend=Backend())["status"] == "fallback"


def test_split_merge_drop_parent_candidate_restore_state(tmp_path):
    project = prepared(tmp_path)
    project.cues[0].candidate_original = asdict(copy.deepcopy(project.cues[0]))
    project.cues[0].candidate = {"signature": signature(project, project.cues[0])}
    project.cues = split_cue(project, 0, 1500, "テ", "スト。")
    assert all(c.candidate is None and c.candidate_original is None for c in project.cues)
    project.cues = merge_cues(project, 0)
    assert project.cues[0].candidate_original is None


def test_annotations_never_promote_machine_candidates_to_gold(tmp_path):
    project = prepared(tmp_path)
    annotation = annotation_template(project)
    report = paired_report(project, annotation)
    assert report["excluded"] == {"unconfirmed": 1}
    assert report["all"]["start_0_to_50ms_rate"] is None
    record = annotation["records"][0]
    record.update(confirmed=True, reviewer="human", original_text="テスト。",
                  speech_start_ms=1100, speech_end_ms=2100, elongated=True)
    report = paired_report(project, annotation)
    assert report["all"]["start_0_to_50ms_rate"] == 1
    assert report["all"]["display_early_cut_count"] == 1
    assert report["all"]["base_end_absolute_error"]["median_ms"] == 300
    assert report["all"]["cer_whitespace_removed"] == 0
    record["uncertain"] = True
    assert paired_report(project, annotation)["excluded"] == {"uncertain": 1}


def test_duplicate_pairs_and_source_mismatch_rejected(tmp_path):
    project = prepared(tmp_path)
    annotation = annotation_template(project)
    annotation["records"] *= 2
    with pytest.raises(ValueError, match="重複"):
        paired_report(project, annotation)
    annotation["audio_stream"] = 1
    with pytest.raises(ValueError, match="來源"):
        paired_report(project, annotation)
