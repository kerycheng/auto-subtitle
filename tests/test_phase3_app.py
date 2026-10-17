import copy
from threading import Event

from autosub.alignment import analyze_cue
from autosub.app import MainWindow
from autosub.domain import Cue, Project
from autosub.media import Cancelled


def window_for(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(media_path="missing.wav", duration_ms=4000, status="complete",
                             cues=[Cue("one", 980, 2200, "Test.", 1000, 2000, locked=True)])
    window.project_path = tmp_path / "ui.autosub.json"
    window.refresh()
    window.table.selectRow(0)
    return window


def test_background_analysis_preserves_locked_cue_until_explicit_adoption(qtbot, tmp_path, monkeypatch):
    import autosub.app as desktop
    window = window_for(qtbot, tmp_path)
    before = copy.deepcopy(window.project.cues[0])
    def analyze(project, row, cache, cancel, language):
        candidate = analyze_cue(project, row, cache, Event(), language)
        candidate["base"] = [1000, 2100]
        candidate["display"] = [980, 2300]
        return candidate
    monkeypatch.setattr(desktop, "analyze_cue", analyze)
    window.analyze_selected()
    assert not window.apply_candidate_button.isEnabled()
    qtbot.waitUntil(lambda: window.thread is None, timeout=5000)
    assert window.project.cues[0].end_ms == before.end_ms
    assert window.wave.candidate_region == [1000, 2100]
    assert Project.load(window.project_path).cues[0].candidate
    window.accept_candidate()
    assert window.project.cues[0].end_ms == 2300
    assert not window.project.cues[0].reviewed
    window.revert_candidate()
    assert window.project.cues[0].end_ms == 2200
    assert window.project.cues[0].locked
    window.undo()
    assert window.project.cues[0].end_ms == 2300
    window.redo()
    assert window.project.cues[0].end_ms == 2200
    window.autosave.stop()
    window.dirty = False


def test_cancel_alignment_keeps_completed_asr_and_no_candidate(qtbot, tmp_path, monkeypatch):
    import autosub.app as desktop
    window = window_for(qtbot, tmp_path)
    def analyze(project, row, cache, cancel, language):
        cancel.wait(2)
        raise Cancelled("cancel")
    monkeypatch.setattr(desktop, "analyze_cue", analyze)
    window.analyze_selected()
    window.request_cancel()
    qtbot.waitUntil(lambda: window.thread is None, timeout=5000)
    assert window.project.cues[0].candidate is None
    assert window.project.status == "complete"
    assert Project.load(window.project_path).cues[0].text == "Test."


def test_stale_candidate_not_adopted_after_manual_edit(qtbot, tmp_path):
    window = window_for(qtbot, tmp_path)
    window.project.cues[0].candidate = analyze_cue(window.project, 0, tmp_path, Event(), "ja")
    window.table.item(0, 2).setText("Corrected.")
    errors = []
    window.error = errors.append
    window.accept_candidate()
    assert errors and "過期" in str(errors[0])
    assert window.project.cues[0].text == "Corrected."
    assert window.wave.candidate_region is None
    window.autosave.stop()
    window.dirty = False
