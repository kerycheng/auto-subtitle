from PySide6.QtCore import Qt

from autosub.app import MainWindow
from autosub.domain import Cue, Project, Word


def prepared_window(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project_path = tmp_path / "ui.autosub.json"
    window.project = Project(duration_ms=5000, media_path="sample.wav", status="complete",
                             words=[Word(1000, 1800, "原文。")], speech=[[1000, 1800]],
                             cues=[Cue("a", 980, 2000, "原文。", 1000, 1800)])
    window.refresh()
    return window


def test_edit_locks_and_autosaves_with_undo_redo(qtbot, tmp_path):
    window = prepared_window(qtbot, tmp_path)
    window.table.item(0, 2).setText("修改後的文字。")
    assert window.project.cues[0].locked
    qtbot.waitUntil(lambda: window.project_path.exists(), timeout=3000)
    assert Project.load(window.project_path).cues[0].text == "修改後的文字。"
    window.undo()
    assert window.project.cues[0].text == "原文。"
    window.redo()
    assert window.project.cues[0].text == "修改後的文字。"
    window.autosave.stop()
    window.dirty = False


def test_locked_edit_survives_retime_and_problem_filter(qtbot, tmp_path):
    window = prepared_window(qtbot, tmp_path)
    window.table.item(0, 2).setText("APEX 修正。")
    window.retime()
    assert window.project.cues[0].text == "APEX 修正。"
    window.only_flags.setChecked(True)
    assert window.table.isRowHidden(0)
    window.autosave.stop()
    window.dirty = False


def test_manual_elongation_ends_at_speech_boundary(qtbot, tmp_path):
    window = prepared_window(qtbot, tmp_path)
    window.table.item(0, 5).setCheckState(Qt.CheckState.Checked)
    assert window.project.cues[0].end_ms == 1800
    assert window.project.cues[0].locked
    window.autosave.stop()
    window.dirty = False


def test_background_worker_completes_real_silence_pipeline(qtbot, tmp_path):
    import wave
    source = tmp_path / "silent.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(32000))
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(media_path=str(source))
    window.project_path = tmp_path / "thread.autosub.json"
    window.start()
    assert window.thread is not None
    qtbot.waitUntil(lambda: window.thread is None, timeout=20000)
    assert window.project.status == "complete"
    assert Project.load(window.project_path).status == "complete"
    assert window.start_button.isEnabled()


def test_invalid_edit_restores_previous_value(qtbot, tmp_path):
    window = prepared_window(qtbot, tmp_path)
    errors = []
    window.error = errors.append
    window.table.item(0, 0).setText("-10")
    assert errors
    assert window.project.cues[0].start_ms == 980
    assert window.table.item(0, 0).text() == "980"


def test_close_cancels_worker_before_window_is_destroyed(qtbot, tmp_path, monkeypatch):
    import autosub.app as desktop
    from autosub.media import Cancelled
    def slow_run(project, path, cancel, progress):
        cancel.wait(5)
        project.status = "cancelled"
        project.save(path)
        raise Cancelled("cancelled")
    monkeypatch.setattr(desktop, "run_pipeline", slow_run)
    window = prepared_window(qtbot, tmp_path)
    window.start()
    window.close()
    qtbot.waitUntil(lambda: window.thread is None, timeout=10000)
    assert Project.load(window.project_path).status == "cancelled"


def test_manual_split_merge_and_undo_from_desktop(qtbot, tmp_path, monkeypatch):
    import autosub.app as desktop
    window = prepared_window(qtbot, tmp_path)
    window.table.selectRow(0)
    monkeypatch.setattr(desktop.SplitDialog, "exec", lambda self: desktop.QDialog.DialogCode.Accepted)
    window.split_selected()
    assert len(window.project.cues) == 2
    assert "".join(c.text for c in window.project.cues) == "原文。"
    assert all(c.locked for c in window.project.cues)
    window.retime()
    assert len(window.project.cues) == 2
    window.table.selectRow(0)
    window.merge_selected()
    assert len(window.project.cues) == 1
    assert window.project.cues[0].text == "原文。"
    window.undo()
    assert len(window.project.cues) == 2
    window.autosave.stop()
    window.dirty = False


def test_wave_zoom_focus_pan_and_time_mapping(qtbot):
    from autosub.app import Waveform
    wave = Waveform()
    qtbot.addWidget(wave)
    wave.resize(1000, 100)
    wave.duration_ms = 120000
    wave.region = (60000, 63000)
    wave.focus_region()
    lo, hi = wave.bounds()
    assert lo <= 60000 < 63000 <= hi
    assert wave.time_at(0) == lo
    assert wave.time_at(wave.width()) == hi
    wave.zoom(2)
    assert wave.bounds()[1] - wave.bounds()[0] < hi - lo
    wave.pan(1)
    assert 0 <= wave.bounds()[0] < wave.bounds()[1] <= wave.duration_ms
    wave.show_all()
    assert wave.bounds() == (0, 120000)


def test_cursor_can_set_selected_boundary(qtbot, tmp_path, monkeypatch):
    window = prepared_window(qtbot, tmp_path)
    window.table.selectRow(0)
    monkeypatch.setattr(window.audio, "position", lambda: 1100)
    window.set_edge(0)
    assert window.project.cues[0].start_ms == 1100
    assert window.project.cues[0].locked
    window.autosave.stop()
    window.dirty = False


def test_review_checkbox_filters_resolved_item_and_edit_reopens_it(qtbot, tmp_path):
    window = prepared_window(qtbot, tmp_path)
    window.project.cues[0].flags = ["辨識信心偏低"]
    window.refresh()
    window.table.item(0, 7).setCheckState(Qt.CheckState.Checked)
    assert window.project.cues[0].reviewed
    assert not window.project.cues[0].locked
    window.only_flags.setChecked(True)
    assert window.table.isRowHidden(0)
    window.only_flags.setChecked(False)
    window.table.item(0, 2).setText("再次修正。")
    assert not window.project.cues[0].reviewed
    window.autosave.stop()
    window.dirty = False


def test_edit_after_group_unlock_relocks_all_fragments(qtbot, tmp_path, monkeypatch):
    import autosub.app as desktop
    window = prepared_window(qtbot, tmp_path)
    window.table.selectRow(0)
    monkeypatch.setattr(desktop.SplitDialog, "exec", lambda self: desktop.QDialog.DialogCode.Accepted)
    window.split_selected()
    window.unlock()
    assert not any(c.locked for c in window.project.cues)
    window.table.item(0, 2).setText("修正前半")
    assert all(c.locked for c in window.project.cues)
    window.retime()
    assert len(window.project.cues) == 2
    assert window.project.cues[0].text == "修正前半"
    window.autosave.stop()
    window.dirty = False
