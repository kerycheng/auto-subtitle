import copy

import pytest

import autosub.app as desktop
import autosub.proposal_store as store
from autosub.domain import Cue, Project, Word, word_key
from autosub.review_proposals import proposals


def window(qtbot,tmp_path):
    w=desktop.MainWindow()
    qtbot.addWidget(w)
    words=[Word(1000,2000,'声も入っています'),Word(2000,2500,'こちら')]
    w.project=Project(media_path='sample.wav',duration_ms=5000,status='complete',words=words,
        speech=[[1000,2500]],native_groups=[[0],[1]],native_complete=True,native_resume_ms=5000,
        asr_strategy='native-v1',source_fingerprint={'test':'fixed'})
    w.project.cues=[Cue('a',980,2000,words[0].text,1000,2000,source_tokens=[word_key(words[0])]),
                    Cue('b',2000,2700,words[1].text,2000,2500,source_tokens=[word_key(words[1])])]
    w.project_path=tmp_path/'project.autosub.json'
    w.load_controls()
    w.refresh()
    w.table.selectRow(0)
    w.project.save(w.project_path)
    w.error=lambda error: setattr(w,'test_error',str(error))
    return w


def accept(monkeypatch):
    monkeypatch.setattr(desktop.MergePreviewDialog,'exec',lambda _:desktop.QDialog.DialogCode.Accepted)


def test_cancel_preview_has_no_project_file_or_history_effect(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    original=copy.deepcopy(w.project)
    data=w.project_path.read_bytes()
    monkeypatch.setattr(desktop.MergePreviewDialog,'exec',lambda _:desktop.QDialog.DialogCode.Rejected)
    w.merge_selected()
    assert w.project==original and w.project_path.read_bytes()==data
    assert w.history==[] and not list(tmp_path.glob('*.review'))
    assert not w.dirty


def test_adopt_reload_undo_and_redo_use_existing_controls(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    original=copy.deepcopy(w.project.cues)
    accept(monkeypatch)
    w.merge_selected()
    assert len(w.project.cues)==1 and w.project.cues[0].locked
    assert len(list(tmp_path.glob('*.review/merge-receipts/*.json')))==1
    assert Project.load(w.project_path).cues==w.project.cues
    assert not w.dirty
    w.project=Project.load(w.project_path)
    w.reset_history()
    w.load_controls()
    w.refresh()
    w.undo()
    assert w.project.cues==original and Project.load(w.project_path).cues==original
    w.redo()
    assert len(w.project.cues)==1
    w.persist()
    w.reset_history()
    w.undo()
    assert w.project.cues==original
    w.autosave.stop()


def test_later_saved_edit_cannot_be_overwritten_by_persistent_undo(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    w.merge_selected()
    w.table.item(0,2).setText('人工改字')
    w.persist()
    w.reset_history()
    before=w.project_path.read_bytes()
    w.undo()
    assert w.project.cues[0].text=='人工改字' and w.project_path.read_bytes()==before
    assert '後續修改' in w.test_error
    w.autosave.stop()


def test_receipt_failure_prevents_project_adoption(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    data=w.project_path.read_bytes()
    original=copy.deepcopy(w.project)
    def fail(*_): raise OSError('receipt disk full')
    monkeypatch.setattr(store,'save_receipt',fail)
    w.merge_selected()
    assert w.project==original and w.project_path.read_bytes()==data and not w.history
    assert 'disk full' in w.test_error


def test_project_save_failure_keeps_source_and_ram_with_orphan_receipt(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    original=copy.deepcopy(w.project)
    data=w.project_path.read_bytes()
    def fail(*_): raise OSError('project disk full')
    monkeypatch.setattr(Project,'save',fail)
    w.merge_selected()
    assert w.project==original and w.project_path.read_bytes()==data and not w.history
    assert len(list(tmp_path.glob('*.review/merge-receipts/*.json')))==1


def test_missing_or_corrupt_receipt_does_not_revert(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    w.merge_selected()
    w.reset_history()
    receipt=next(tmp_path.glob('*.review/merge-receipts/*.json'))
    receipt.write_text('{}',encoding='utf-8')
    data=w.project_path.read_bytes()
    w.undo()
    assert len(w.project.cues)==1 and w.project_path.read_bytes()==data
    assert '收據' in w.test_error


def test_preview_uses_shared_audition_with_context_and_stops_on_close(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    ranges=[]
    monkeypatch.setattr(w.audio,'audition',lambda a,b:ranges.append((a,b)))
    def preview(d):
        assert d.preview.toPlainText()=='声も入っていますこちら'
        d.play_button.click()
        return desktop.QDialog.DialogCode.Rejected
    monkeypatch.setattr(desktop.MergePreviewDialog,'exec',preview)
    w.merge_selected()
    assert ranges==[(180,3500)]
    assert not w.audio.is_playing()


def test_pending_timing_controls_participate_in_receipt(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    w.timing_inputs['tail_ms'].setValue(100)
    w.merge_selected()
    assert w.project.settings.tail_ms==100
    w.reset_history()
    w.undo()
    assert len(w.project.cues)==2 and w.project.settings.tail_ms==100


@pytest.mark.parametrize('sha',['../escape','abc',None])
def test_receipt_identity_cannot_escape_store(tmp_path,sha):
    with pytest.raises(ValueError):
        store.receipt_path(tmp_path/'project.json',sha)


def test_immutable_receipt_round_trip(tmp_path):
    from autosub.review_proposals import apply_proposal
    # Actual file I/O is covered with a self-contained project, no private fixture.
    words=[Word(100,500,'前'),Word(500,900,'続き')]
    p=Project(duration_ms=1000,words=words,speech=[[100,900]],native_groups=[[0],[1]],
        asr_strategy='native-v1',native_complete=True,native_resume_ms=1000,
        cues=[Cue('a',100,500,'前',100,500,source_tokens=[word_key(words[0])]),
              Cue('b',500,1000,'続き',500,900,source_tokens=[word_key(words[1])])])
    item=next(i for i in proposals(p) if i['kind']=='merge')
    _,receipt=apply_proposal(p,item,'test')
    path=tmp_path/'p.json'
    store.save_receipt(path,receipt)
    store.save_receipt(path,receipt)
    assert store.read_receipt(path,receipt['sha256'])==receipt


def test_persistent_restore_save_failure_does_not_change_ram_or_history(qtbot,tmp_path,monkeypatch):
    w=window(qtbot,tmp_path)
    accept(monkeypatch)
    w.merge_selected()
    w.reset_history()
    before=copy.deepcopy(w.project)
    data=w.project_path.read_bytes()
    def fail(*_): raise OSError('restore disk full')
    monkeypatch.setattr(Project,'save',fail)
    w.undo()
    assert w.project==before and w.project_path.read_bytes()==data
    assert not w.future and 'disk full' in w.test_error
