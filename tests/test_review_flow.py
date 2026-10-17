from PySide6.QtCore import Qt

from autosub.app import MainWindow
from autosub.domain import Cue, Project


def prepared(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.prepare_replace = lambda: True  # synthetic project has no save destination
    window.project = Project(duration_ms=10000, status="complete", cues=[
        Cue("a", 1000, 2000, "第一句", 1000, 1800, ["check"]),
        Cue("b", 4000, 5000, "第二句", 4000, 4800, ["check"]),
        Cue("c", 7000, 8000, "第三句", 7000, 7800, ["check"])])
    window.refresh()
    return window


def test_confirm_in_basic_mode_advances_without_changing_text_or_times(qtbot):
    window = prepared(qtbot)
    before = [(cue.text, cue.start_ms, cue.end_ms, cue.locked) for cue in window.project.cues]
    assert not window.advanced.isChecked()
    window.confirm_and_next()
    assert window.project.cues[0].reviewed and window.table.currentRow() == 1
    assert "已複核 1／3" in window.selection_label.text()
    assert not window.audio.is_playing()
    assert [(cue.text, cue.start_ms, cue.end_ms, cue.locked) for cue in window.project.cues] == before
    window.undo()
    assert not window.project.cues[0].reviewed


def test_filtered_review_does_not_reselect_hidden_row_and_wraps(qtbot):
    window = prepared(qtbot)
    window.only_flags.setChecked(True)
    window.table.selectRow(1)
    # Direct checkbox editing was the original hidden-selection regression.
    window.table.item(1, 7).setCheckState(Qt.CheckState.Checked)
    assert window.table.isRowHidden(1) and window.table.currentRow() == 2
    assert window.table.selectionModel().selectedRows()[0].row() == 2
    window.confirm_and_next()
    assert window.table.currentRow() == 0
    window.confirm_and_next()
    assert all(cue.reviewed for cue in window.project.cues)
    assert not window.table.selectionModel().hasSelection()
    assert not window.play_button.isEnabled() and not window.confirm_action.isEnabled()
    assert "已複核 3／3" in window.selection_label.text()


def test_review_shortcut_does_not_confirm_uncommitted_cell_edit(qtbot):
    window = prepared(qtbot)
    window.show()
    window.table.editItem(window.table.item(0, 2))
    assert window.table.state() == window.table.State.EditingState
    window.confirm_and_next()
    assert not window.project.cues[0].reviewed
    assert "完成" in window.status.text()


def test_previous_flag_wraps_and_empty_project_is_safe(qtbot):
    window = prepared(qtbot)
    window.project.cues[1].reviewed = True
    window.next_flag(-1)
    assert window.table.currentRow() == 2
    window.next_flag(-1)
    assert window.table.currentRow() == 0
    window.project = Project()
    window.refresh()
    window.next_flag(-1)
    window.confirm_and_next()


def test_ctrl_enter_shortcut_confirms_selected_row(qtbot):
    window = prepared(qtbot)
    window.show()
    window.activateWindow()
    window.table.setFocus()
    qtbot.wait(20)
    qtbot.keyClick(window.table, Qt.Key.Key_Return, modifier=Qt.KeyboardModifier.ControlModifier)
    assert window.project.cues[0].reviewed and window.table.currentRow() == 1
