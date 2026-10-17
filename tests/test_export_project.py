import json

import pytest

from autosub.domain import Cue, Project
from autosub.export import export_subtitles, wrap_text


def project():
    return Project(duration_ms=4000, cues=[
        Cue("1", 0, 1235, "こんにちは。", 0, 1100),
        Cue("2", 1235, 4000, r"{\b1} Hello\Nworld <b>中文</b>", 1300, 3800),
    ])


def test_project_round_trip_and_atomic_save(tmp_path):
    path = tmp_path / "測試.autosub.json"
    original = project()
    original.save(path)
    assert Project.load(path) == original
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_unknown_schema_rejected(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError):
        Project.load(path)


def test_phase1_project_without_new_fields_still_loads(tmp_path):
    from dataclasses import asdict
    path = tmp_path / "old.autosub.json"
    raw = asdict(project())
    del raw["settings"]["boundary_search_ms"]
    for cue in raw["cues"]:
        del cue["source_tokens"]
        del cue["reviewed"]
        del cue["edit_group"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = Project.load(path)
    assert loaded.settings.boundary_search_ms == 400
    assert all(c.source_tokens is None for c in loaded.cues)


def test_ass_touching_rounding_and_injection(tmp_path):
    path = tmp_path / "result.ass"
    export_subtitles(project(), path)
    text = path.read_text(encoding="utf-8-sig")
    assert "0:00:00.00,0:00:01.24" in text
    assert "0:00:01.24,0:00:04.00" in text
    assert r"{\b1}" not in text
    assert r"\Nworld" not in text
    assert "｛＼b1｝" in text


def test_srt_and_newline_safety(tmp_path):
    path = tmp_path / "result.srt"
    export_subtitles(project(), path)
    text = path.read_text(encoding="utf-8-sig")
    assert "00:00:01,235 --> 00:00:04,000" in text
    assert "＜b＞中文＜/b＞" in text


def test_unrepresentable_ass_cue_rejected(tmp_path):
    p = Project(duration_ms=2000, cues=[Cue("a", 1000, 1001, "短", 1000, 1001)])
    with pytest.raises(ValueError, match="時長為零"):
        export_subtitles(p, tmp_path / "tiny.ass")


@pytest.mark.parametrize("text", ["This is an English sentence with many words to display.", "これは少し長めの字幕として表示される日本語の文章です。", "x" * 70])
def test_line_wrapping_preserves_text(text):
    wrapped = wrap_text(text)
    assert wrapped.count("\n") <= 1
    assert "".join(wrapped.split()) == "".join(text.split())
