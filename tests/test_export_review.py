import copy
import hashlib

import pytest

from autosub.domain import Cue, Project
from autosub.export import export_subtitles
from autosub.export_review import export_with_review, review_export


def cue(name, start, end, text="Example"):
    return Cue(name, start, end, text, start, end)


def test_review_uses_export_order_rounding_and_custom_threshold(tmp_path):
    p = Project(duration_ms=2003, status="complete", cues=[cue("later", 1501, 2003), cue("first", 4, 499)])
    p.settings.min_duration_ms = 500
    before = copy.deepcopy(p)
    ass, path, error = export_with_review(p, tmp_path / "result.ass")
    srt = review_export(p, ass=False)
    assert not error and p == before
    assert ass["findings"] == []  # ASS rounds 4..499 to 0..500, EOF caps at 2000.
    assert [(f["rows"], f["start_ms"], f["end_ms"]) for f in srt["findings"]] == [([1], 4, 499)]
    text = path.read_text(encoding="utf-8-sig")
    assert hashlib.sha256((tmp_path / "result.ass").read_bytes()).hexdigest() in text
    assert "已聽校" in text and "未發現" in text


def test_nested_overlaps_are_one_group_but_touching_is_separate():
    p = Project(duration_ms=5000, cues=[cue("outer", 0, 3000), cue("nested", 500, 1000),
                                      cue("third", 2000, 3500), cue("touching", 3500, 4500)])
    findings = review_export(p, ass=False)["findings"]
    assert [f["rows"] for f in findings if f["code"] == "overlap"] == [[1, 2, 3]]


def test_empty_uncertain_long_and_incomplete_are_distinct(tmp_path):
    p = Project(duration_ms=10000, cues=[cue("a", 0, 8000, "［聽不清］"), cue("b", 8000, 9000, " ")])
    report, path, error = export_with_review(p, tmp_path / "result.srt")
    assert not error
    assert [f["code"] for f in report["findings"]] == ["long", "uncertain", "empty"]
    assert not report["recognition_complete"]
    assert "部分字幕匯出" in path.read_text(encoding="utf-8-sig")


def test_report_failure_preserves_subtitle_and_previous_report(tmp_path, monkeypatch):
    import autosub.export_review as module
    path = tmp_path / "result.ass"
    companion = tmp_path / "result.ass.review.txt"
    companion.write_text("previous", encoding="utf-8")
    def fail(*args):
        raise PermissionError("locked")
    monkeypatch.setattr(module, "replace_checkpoint", fail)
    p = Project(duration_ms=1000, cues=[cue("a", 0, 800)])
    report, actual, error = export_with_review(p, path)
    assert error == "locked" and actual == companion
    assert "Dialogue:" in path.read_text(encoding="utf-8-sig")
    assert companion.read_text() == "previous"
    assert len(list(tmp_path.glob("*.tmp"))) == 1


def test_reexport_clears_old_warnings_and_library_export_stays_subtitle_only(tmp_path):
    path = tmp_path / "result.ass"
    p = Project(duration_ms=1000, status="complete", cues=[cue("a", 0, 90)])
    export_subtitles(p, path)
    assert len(list(tmp_path.iterdir())) == 1
    _, companion, _ = export_with_review(p, path)
    assert "第 1 句" in companion.read_text(encoding="utf-8-sig")
    p.cues = [cue("b", 0, 800)]
    export_with_review(p, path)
    assert "第 1 句" not in companion.read_text(encoding="utf-8-sig")


def test_invalid_export_does_not_write_report(tmp_path):
    p = Project(duration_ms=1000, cues=[cue("a", 100, 101)])
    with pytest.raises(ValueError, match="時長為零"):
        export_with_review(p, tmp_path / "result.ass")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("report_failure", [False, True])
def test_gui_export_keeps_success_if_project_save_fails(qtbot, monkeypatch, tmp_path, report_failure):
    from autosub.app import MainWindow, QFileDialog, QMessageBox
    import autosub.export_review as module
    w = MainWindow()
    qtbot.addWidget(w)
    w.project = Project(media_path="sample.mp4", duration_ms=1000, status="complete", cues=[cue("a", 0, 90)])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "export.ass"), "ASS 字幕 (*.ass)"))
    monkeypatch.setattr(w, "persist", lambda: False)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    if report_failure:
        monkeypatch.setattr(module, "replace_checkpoint", lambda *args: (_ for _ in ()).throw(OSError("report denied")))
    w.export()
    assert "已匯出" in w.status.text() and "專案未能儲存" in w.status.text()
    assert ("未更新" if report_failure else "1 項檢查提醒") in w.status.text()
    assert (tmp_path / "export.ass").is_file()
