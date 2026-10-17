import copy
import json

from PySide6.QtCore import Qt

from autosub.app import MainWindow, Waveform
from autosub.comparison import load_session, summary
from autosub.domain import Cue, Project
from autosub.review_ui import ComparisonDialog


def prepared(qtbot, tmp_path):
    p = Project(media_path="missing.wav", duration_ms=10000,
                cues=[Cue("a", 1000, 2000, "Original.", 1000, 1800), Cue("b", 4000, 5000, "Next.", 4000, 4800)])
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n翻譯\n\n2\n00:00:04,000 --> 00:00:05,000\n下一句\n", encoding="utf-8")
    dialog = ComparisonDialog(p, tmp_path / "project.autosub.json", Waveform)
    qtbot.addWidget(dialog)
    dialog.reference.setText(str(reference))
    dialog.timeline.setChecked(True)
    return p, dialog


def test_background_compare_manual_judgment_undo_and_resume(qtbot, tmp_path):
    p, dialog = prepared(qtbot, tmp_path)
    before = copy.deepcopy(p)
    dialog.start_compare()
    assert not dialog.compute_button.isEnabled()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    assert dialog.session_path.is_file()
    assert summary(dialog.session)["strict_time_estimate_rate"] == 1
    assert summary(dialog.session)["human_confirmed_usable_rate"] is None
    dialog.reviewer.setText("test reviewer")
    dialog.table.selectRow(0)
    dialog.record_decision()
    assert summary(dialog.session)["human_confirmed_usable_rate"] == .5
    assert load_session(dialog.session_path, p)["rows"][0]["decision"] == "usable"
    dialog.undo_review()
    assert dialog.session["rows"][0]["decision"] == "unreviewed"
    assert p == before
    dialog.close()


def test_annotations_require_explicit_text_and_acoustic_edges(qtbot, tmp_path):
    _, dialog = prepared(qtbot, tmp_path)
    dialog.start_compare()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    dialog.table.selectRow(0)
    assert dialog.gold_start.value() == dialog.gold_end.value() == 0
    assert not dialog.original_text.text()
    errors = []
    dialog.error = errors.append
    dialog.record_annotation()
    assert errors and not dialog.annotation_path.exists()
    dialog.reviewer.setText("test reviewer")
    dialog.original_text.setText("Original.")
    dialog.gold_start.setValue(1020)
    dialog.gold_end.setValue(1810)
    dialog.record_annotation()
    raw = json.loads(dialog.annotation_path.read_text(encoding="utf-8"))
    assert raw["records"][0]["confirmed"]
    assert raw["records"][0]["speech_end_ms"] == 1810
    assert not raw["records"][1]["confirmed"]
    assert dialog.project.cues[0].end_ms == 2000
    dialog.table.selectRow(1)
    assert not dialog.original_text.text()
    assert dialog.wave.speech_region is None
    dialog.close()


def test_setting_changes_preserve_old_review_file_and_reset_display(qtbot, tmp_path):
    _, dialog = prepared(qtbot, tmp_path)
    dialog.start_compare()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    path = dialog.session_path
    dialog.toggle_timer()
    assert dialog.timer_started is not None
    dialog.offset.setValue(500)
    assert dialog.session is None and dialog.table.rowCount() == 0
    assert dialog.timer_started is None and path.exists()
    dialog.close()


def test_next_flag_shortcut_skips_reviewed_items(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(duration_ms=5000, cues=[
        Cue("a", 0, 1000, "A", 0, 900, ["待複核"], reviewed=True),
        Cue("b", 2000, 3000, "B", 2000, 2900, ["待複核"]),
    ])
    window.refresh()
    window.table.selectRow(0)
    window.next_flag()
    assert window.table.currentRow() == 1


def test_priority_navigation_skips_reviewed_and_keeps_rows_in_time_order(qtbot, tmp_path):
    _, dialog = prepared(qtbot, tmp_path)
    dialog.start_compare()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    rows = dialog.session["rows"]
    rows[1]["ambiguous"] = True
    original_ids = [row["id"] for row in rows]
    dialog.table.selectRow(0)
    dialog.audio._resumable = True
    dialog.next_priority()
    assert dialog.table.currentRow() == 1 and not dialog.audio._resumable
    assert "優先複核" in dialog.status.text()
    assert [row["id"] for row in rows] == original_ids
    dialog.reviewer.setText("tester")
    dialog.record_decision()
    dialog.next_priority()
    assert dialog.table.currentRow() == 0
    dialog.close()


def test_escape_cancels_comparison_before_dialog_is_destroyed(qtbot, tmp_path, monkeypatch):
    import autosub.review_ui as review
    from autosub.media import Cancelled
    _, dialog = prepared(qtbot, tmp_path)
    def waiting_compare(project, reference, settings, cancel, progress, styles):
        cancel.wait(3)
        raise Cancelled("cancelled")
    monkeypatch.setattr(review, "compare_file", waiting_compare)
    dialog.start_compare()
    dialog.reject()
    assert dialog.cancel.is_set()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    assert dialog.session is None


def test_invalid_annotation_is_backed_up_before_new_confirmed_record(qtbot, tmp_path):
    _, first = prepared(qtbot, tmp_path)
    path = first.annotation_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("broken old data", encoding="utf-8")
    first.close()
    _, dialog = prepared(qtbot, tmp_path)
    assert dialog.annotation_invalid
    dialog.start_compare()
    qtbot.waitUntil(lambda: dialog.thread is None, timeout=5000)
    dialog.reviewer.setText("test reviewer")
    dialog.original_text.setText("Original.")
    dialog.gold_start.setValue(1000)
    dialog.gold_end.setValue(1800)
    dialog.record_annotation()
    assert path.with_suffix(".invalid.bak").read_text(encoding="utf-8") == "broken old data"
    assert json.loads(path.read_text(encoding="utf-8"))["records"][0]["confirmed"]
    dialog.close()
