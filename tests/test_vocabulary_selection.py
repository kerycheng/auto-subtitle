import copy
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit
from autosub.app import MainWindow
from autosub.vocabulary import compose_vocabulary, load_library, save_library
from autosub.vocabulary_ui import VocabularySelectionDialog


def library():
    return {'schema_version': 1, 'lists': [
        {'id': 'names', 'name': 'People', 'text': 'Sky Token\nMoon Harbor'},
        {'id': 'game', 'name': 'Game terms', 'text': 'Moon Harbor\nStar Gate'}]}


def test_merge_deduplicates_without_reordering_or_changing_case():
    raw = library()
    raw['lists'][1]['text'] += '\nmoon harbor'
    assert compose_vocabulary(raw, {'ids': ['game', 'names'], 'base_text': ''}) == (
        'Sky Token\nMoon Harbor\nStar Gate\nmoon harbor')


def test_legacy_snapshot_keeps_exact_prefix_and_deduplicates_additions(qtbot):
    original = '  Moon Harbor  \nLegacy, mk-II\n\n'
    dialog = VocabularySelectionDialog(None, library(), {'ids': [], 'base_text': original})
    qtbot.addWidget(dialog)
    assert dialog.prompt == original
    dialog.checks['names'].setChecked(True)
    assert dialog.prompt == original + 'Sky Token'
    dialog.base_check.setChecked(False)
    assert dialog.prompt == library()['lists'][0]['text']


@pytest.mark.parametrize('accepted', [False, True])
def test_checklist_modal_apply_or_cancel_is_atomic(qtbot, accepted):
    save_library(library())
    before = copy.deepcopy(load_library())
    window = MainWindow()
    qtbot.addWidget(window)
    window.apply_vocabulary_choice({'ids': ['names'], 'base_text': ''})
    original = window.glossary.toPlainText()
    def choose():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, VocabularySelectionDialog)
        dialog.checks['game'].setChecked(True)
        (dialog.accept if accepted else dialog.reject)()
    QTimer.singleShot(0, choose)
    window.choose_vocabulary()
    assert window.glossary.toPlainText() == ('Sky Token\nMoon Harbor\nStar Gate' if accepted else original)
    assert load_library() == before
    assert window.project.glossary == ''


def test_combination_survives_restart_and_changed_library_becomes_snapshot(qtbot):
    save_library(library())
    window = MainWindow()
    qtbot.addWidget(window)
    choice = {'ids': ['names', 'game'], 'base_text': ''}
    window.apply_vocabulary_choice(choice)
    expected = window.glossary.toPlainText()
    window.save_user_preferences()
    reopened = MainWindow()
    qtbot.addWidget(reopened)
    assert reopened.vocabulary_choice == choice
    assert reopened.glossary.toPlainText() == expected
    raw = library()
    raw['lists'][1]['text'] = 'Edited Token'
    save_library(raw)
    reopened.refresh_vocabulary()
    assert reopened.glossary.toPlainText() == expected
    assert reopened.vocabulary_choice == {'ids': [], 'base_text': expected}
    reopened.save_user_preferences()
    again = MainWindow()
    qtbot.addWidget(again)
    assert again.glossary.toPlainText() == expected
    assert again.vocabulary_choice['base_text'] == expected


def test_search_does_not_uncheck_hidden_lists_and_clear_is_explicit(qtbot):
    dialog = VocabularySelectionDialog(None, library(), {'ids': ['names', 'game'], 'base_text': ''})
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit).setText('People')
    assert dialog.checks['game'].isHidden()
    assert dialog.choice['ids'] == ['names', 'game']
    dialog.clear_selection()
    assert dialog.prompt == '' and dialog.choice['ids'] == []


def test_combined_prompt_limit_blocks_apply_without_truncating(qtbot):
    raw = library()
    raw['lists'][0]['text'] = 'a' * 60000
    raw['lists'][1]['text'] = 'b' * 60000
    dialog = VocabularySelectionDialog(None, raw, {'ids': ['names', 'game'], 'base_text': ''})
    qtbot.addWidget(dialog)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.checks['game'].setChecked(False)
    assert dialog.prompt == 'a' * 60000
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert '提示較長' in dialog.summary.text()


def test_invalid_optional_selection_keeps_saved_prompt(qtbot):
    import json
    from autosub.preferences import preferences_path
    window = MainWindow()
    qtbot.addWidget(window)
    window.glossary.setPlainText('Legacy phrase')
    window.save_user_preferences()
    raw = json.loads(preferences_path().read_text(encoding='utf-8'))
    raw['vocabulary_choice'] = {'ids': [None], 'base_text': ''}
    preferences_path().write_text(json.dumps(raw), encoding='utf-8')
    reopened = MainWindow()
    qtbot.addWidget(reopened)
    assert reopened.glossary.toPlainText() == 'Legacy phrase'
    assert reopened.vocabulary_choice == {'ids': [], 'base_text': 'Legacy phrase'}
