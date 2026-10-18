import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton

from autosub import __version__
from autosub.app import MainWindow
from autosub.vocabulary import clean_words, library_path, load_library, save_library
from autosub.vocabulary_ui import VocabularyDialog


def collection(text="Moon Harbor\nSky Token"):
    return {"schema_version": 1, "lists": [{"id": "sample", "name": "Sample terms", "text": text}]}


def test_library_round_trip_and_exact_spelling(tmp_path):
    path = tmp_path / "words.json"
    assert load_library(path)["lists"] == []
    raw = collection(" Star Token, mk-II \n星の道\nSTAR Token")
    save_library(raw, path)
    assert load_library(path) == raw
    assert clean_words(" star \nstar\nStar\n\n") == ["star", "Star"]


def test_library_rejects_duplicate_names_without_overwriting(tmp_path):
    path = tmp_path / "words.json"
    raw = collection()
    save_library(raw, path)
    before = path.read_bytes()
    raw["lists"].append({"id": "other", "name": " sample TERMS ", "text": "Example"})
    with pytest.raises(ValueError):
        save_library(raw, path)
    assert path.read_bytes() == before


def test_unreadable_library_is_preserved(tmp_path):
    path = tmp_path / "words.json"
    path.write_text('{"schema_version":99}', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_library(collection(), path)
    assert path.read_bytes() == before


def test_failed_library_replace_retains_old_and_valid_temporary(tmp_path, monkeypatch):
    import autosub.vocabulary as storage
    path = tmp_path / "words.json"
    save_library(collection(), path)
    before = path.read_bytes()
    def deny(*_):
        raise PermissionError("synthetic denial")
    monkeypatch.setattr(storage, "replace_checkpoint", deny)
    with pytest.raises(PermissionError):
        save_library(collection("Changed token"), path)
    assert path.read_bytes() == before
    temporary = next(tmp_path.glob("words.json.*.tmp"))
    assert json.loads(temporary.read_text(encoding="utf-8"))["lists"][0]["text"] == "Changed token"


def test_dialog_cancel_does_not_create_library(qtbot):
    dialog = VocabularyDialog(None)
    qtbot.addWidget(dialog)
    dialog.name.setText("Example")
    dialog.new_word.setText("Sample Token")
    dialog.add_word()
    dialog.reject()
    assert not library_path().exists()


def test_dialog_add_save_and_reopen(qtbot):
    dialog = VocabularyDialog(None)
    qtbot.addWidget(dialog)
    dialog.name.setText("Example")
    dialog.new_word.setText("Sample Token")
    dialog.add_word()
    dialog.new_word.setText("Sample Token")
    dialog.add_word()
    dialog.new_word.setText("Second Token")
    dialog.save()
    saved = load_library()["lists"][0]
    assert saved["text"] == "Sample Token\nSecond Token"
    reopened = VocabularyDialog(None, saved["id"])
    qtbot.addWidget(reopened)
    assert reopened.words.toPlainText() == saved["text"]


def test_enter_adds_word_without_saving_dialog(qtbot):
    dialog = VocabularyDialog(None)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.name.setText("Example")
    dialog.new_word.setText("Sample Token")
    dialog.new_word.setFocus()
    qtbot.keyClick(dialog.new_word, Qt.Key.Key_Return)
    assert dialog.words.toPlainText() == "Sample Token"
    assert not library_path().exists()
    assert dialog.isVisible()


def test_public_layout_hides_tools_and_schedule_copy(qtbot, monkeypatch):
    monkeypatch.delenv("AUTOSUB_ENABLE_REPAIR_UI", raising=False)
    monkeypatch.delenv("AUTOSUB_ENABLE_PROJECT_TOOLS", raising=False)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.tools_button.isHidden()
    assert window.vocabulary_group.isVisible()
    window.advanced.setChecked(True)
    assert window.tools_button.isHidden()
    text = "\n".join(w.text() for w in window.findChildren(QLabel))
    assert not any(term in text for term in ["macOS", "直播剪輯試跑", "50ms 為目標", "忠實保留口語"])
    assert not any(b.text() in {"APEX", "SF6", "其他"} for b in window.findChildren(QPushButton))
    assert __version__ in window.windowTitle()


def test_project_prompt_survives_library_change_and_removal(qtbot):
    save_library(collection())
    window = MainWindow()
    qtbot.addWidget(window)
    window.apply_vocabulary_choice({"ids": ["sample"], "base_text": ""})
    original = window.glossary.toPlainText()
    window.project.glossary = original
    save_library(collection("New terms"))
    window.load_controls()
    assert window.glossary.toPlainText() == original
    assert window.vocabulary_choice == {"ids": [], "base_text": original}
    save_library({"schema_version": 1, "lists": []})
    window.refresh_vocabulary()
    assert window.glossary.toPlainText() == original
    assert window.project.glossary == original
    window.apply_vocabulary_choice({"ids": [], "base_text": ""})
    assert not window.glossary.toPlainText()


def test_vocabulary_controls_lock_during_recognition(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_busy(True)
    assert not window.vocabulary_group.isEnabled()
    window.set_busy(False)
    assert window.vocabulary_group.isEnabled()
    window.project.source_fingerprint = "synthetic"
    window.load_controls()
    assert not window.vocabulary_group.isEnabled()


def test_declining_unsaved_switch_preserves_editor(qtbot, monkeypatch):
    save_library(collection())
    dialog = VocabularyDialog(None, "sample")
    qtbot.addWidget(dialog)
    dialog.words.setPlainText("Unsaved token")
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No)
    dialog.selector.setCurrentIndex(0)
    assert dialog.selector.currentData() == "sample"
    assert dialog.words.toPlainText() == "Unsaved token"


def test_explicit_delete_removes_library_only(qtbot, monkeypatch):
    save_library(collection())
    dialog = VocabularyDialog(None, "sample")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    dialog.delete_list()
    assert load_library()["lists"] == []
    assert dialog.result() == dialog.DialogCode.Accepted


def test_selected_vocabulary_restores_after_app_restart(qtbot):
    save_library(collection())
    window = MainWindow()
    qtbot.addWidget(window)
    window.apply_vocabulary_choice({"ids": ["sample"], "base_text": ""})
    window.save_user_preferences()
    reopened = MainWindow()
    qtbot.addWidget(reopened)
    assert reopened.glossary.toPlainText() == collection()["lists"][0]["text"]
    assert reopened.vocabulary_choice == {"ids": ["sample"], "base_text": ""}
