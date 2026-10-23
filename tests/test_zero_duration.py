import copy
import json
from collections import Counter
from dataclasses import asdict

import pytest

from autosub.domain import Cue, Project, Word, word_key
from autosub.export import export_subtitles
from autosub.timing import apply_timing, cues_from_groups


@pytest.mark.parametrize("at", [0, 1000, 2000])
def test_zero_duration_group_keeps_text_and_exports_at_media_edges(tmp_path, at):
    word = Word(at, at, "短い。")
    p = Project(duration_ms=2000, words=[word])
    before = copy.deepcopy(p)
    p.cues = apply_timing(p, cues_from_groups(p, [[word]], separate_vad=True))
    p.validate()
    assert p.words == before.words
    assert p.cues[0].text == word.text
    assert p.cues[0].timing_trace["raw"] == [at, at]
    assert any("長度為零" in flag for flag in p.cues[0].flags)
    for suffix in [".ass", ".srt"]:
        export_subtitles(p, tmp_path / ("out" + suffix))


@pytest.mark.parametrize("strategy", ["native", "lexical"])
def test_adjacent_sentences_cannot_collapse_zero_duration_group(tmp_path, monkeypatch, strategy):
    from autosub.native_segmentation import segment_native
    import autosub.lexical_segmentation as lexical

    groups = [[Word(100, 1000, "前です。")],
              [Word(1000, 1000, "短い。")],
              [Word(1000, 1800, "後です。")]]
    words = [w for group in groups for w in group]
    p = Project(duration_ms=2000, words=words)
    if strategy == "native":
        base = segment_native(p, groups)
    else:
        monkeypatch.setattr(lexical, "lexical_groups", lambda _: (groups, [], None))
        base = lexical.segment_lexically(p)
    p.cues = apply_timing(p, base)
    p.validate()
    assert Counter(t for c in p.cues for t in c.source_tokens) == Counter(map(word_key, words))
    assert len(p.cues) == 3
    for suffix in [".ass", ".srt"]:
        export_subtitles(p, tmp_path / (strategy + suffix))


def broken_project():
    word = Word(1000, 1000, "短い。")
    cue = Cue("zero", 980, 1000, word.text, 1000, 1000,
              source_tokens=[word_key(word)], reviewed=True,
              timing_trace={"raw": [1000, 1000], "segmentation": "native-sentence-v1"})
    return Project(duration_ms=2000, words=[word], cues=[cue])


def test_load_recovers_known_bug_and_save_backs_up_exact_original(tmp_path):
    path = tmp_path / "old.autosub.json"
    original = json.dumps(asdict(broken_project()), ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)
    p = Project.load(path)
    assert path.read_bytes() == original
    assert (p.cues[0].speech_start_ms, p.cues[0].speech_end_ms) == (980, 1000)
    assert not p.cues[0].reviewed
    assert p.words[0].start_ms == p.words[0].end_ms == 1000
    export_subtitles(p, tmp_path / "recovered.ass")
    p.save(path)
    backup = path.with_suffix(path.suffix + ".zero-duration.bak")
    assert backup.read_bytes() == original
    assert Project.load(path) == p
    p.save(path)
    assert backup.read_bytes() == original


@pytest.mark.parametrize("damage", ["locked", "raw", "tokens", "range", "strategy"])
def test_recovery_does_not_hide_other_invalid_cues(tmp_path, damage):
    p = broken_project()
    cue = p.cues[0]
    if damage == "locked":
        cue.locked = True
    elif damage == "raw":
        cue.timing_trace["raw"] = [900, 1000]
    elif damage == "tokens":
        cue.source_tokens = ["missing"]
    elif damage == "range":
        cue.end_ms = 3000
    else:
        cue.timing_trace["segmentation"] = "unknown"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(asdict(p)), encoding="utf-8")
    with pytest.raises(ValueError):
        Project.load(path)
