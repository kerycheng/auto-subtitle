import copy

from autosub.app import MainWindow
from autosub.domain import Project, Cue


def test_public_ui_hides_repair_in_both_modes_and_retains_settings(qtbot,monkeypatch):
    monkeypatch.delenv('AUTOSUB_ENABLE_REPAIR_UI')
    w=MainWindow()
    qtbot.addWidget(w)
    assert not w.repair_enabled and w.advanced.text()=='進階設定'
    for advanced in (False,True):
        w.advanced.setChecked(advanced)
        assert w.edit_panel.isHidden() and w.candidate_panel.isHidden() and w.zoom_panel.isHidden()
        assert all(w.table.isColumnHidden(i) for i in (4,5,7))
        assert w.table.editTriggers()==w.table.EditTrigger.NoEditTriggers
    assert not w.timing_group.isHidden() and not w.recognition_details.isHidden()
    assert not w.compare_button.isVisible() and not w.confirm_action.isVisible()


def test_public_shortcuts_and_hidden_actions_do_not_edit_project(qtbot,monkeypatch):
    monkeypatch.delenv('AUTOSUB_ENABLE_REPAIR_UI')
    w=MainWindow()
    qtbot.addWidget(w)
    w.project=Project(duration_ms=2000,cues=[Cue('a',100,800,'前',100,700),Cue('b',900,1500,'後',900,1400)])
    w.refresh()
    w.table.selectRow(0)
    original=copy.deepcopy(w.project)
    for action in (w.merge_selected,w.split_selected,w.unlock,w.undo,w.redo,w.confirm_and_next,w.analyze_selected,w.accept_candidate,w.revert_candidate):
        action()
    w.set_edge(0)
    w.table.item(0,2).setText('hidden edit')
    assert w.project==original
    assert not w.dirty and not w.history


def test_internal_repair_switch_remains_explicit(qtbot):
    w=MainWindow()
    qtbot.addWidget(w)
    assert w.repair_enabled
    w.advanced.setChecked(True)
    assert not w.edit_panel.isHidden()


def test_public_preview_exports_original_text_and_times(qtbot,monkeypatch,tmp_path):
    from autosub.export import export_subtitles
    from autosub.comparison import read_reference
    monkeypatch.delenv('AUTOSUB_ENABLE_REPAIR_UI')
    w=MainWindow()
    qtbot.addWidget(w)
    w.project=Project(duration_ms=2000,cues=[Cue('a',100,800,'前の文',100,700),Cue('b',900,1500,'次の文',900,1400)])
    expected=[(c.start_ms,c.end_ms,c.text) for c in w.project.cues]
    w.refresh()
    w.table.selectRow(1)
    w.advanced.setChecked(True)
    for suffix in ('.ass','.srt'):
        path=tmp_path/('generated'+suffix)
        export_subtitles(w.project,path)
        cues,_=read_reference(path)
        assert [(c.start_ms,c.end_ms,c.text) for c in cues]==expected
