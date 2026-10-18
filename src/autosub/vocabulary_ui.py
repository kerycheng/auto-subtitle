"""Vocabulary management dialog; cancellation never writes user data."""
import copy
from uuid import uuid4

from PySide6.QtCore import QEvent, Qt

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from .vocabulary import clean_words, load_library, save_library
from .appearance import ChoiceBox as QComboBox


class VocabularyDialog(QDialog):
    def __init__(self, parent, selected_id=None, current_text=""):
        super().__init__(parent)
        self.setWindowTitle("管理詞彙")
        self.resize(520, 500)
        self.library = copy.deepcopy(load_library())
        self.saved_id = None
        self.current_text = current_text
        self.previous_index = 0
        self.initial_fields = ("", "")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        self.selector = QComboBox()
        self.selector.addItem("新增詞彙清單…", None)
        for item in self.library["lists"]:
            self.selector.addItem(item["name"], item["id"])
        layout.addWidget(self.selector)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("例如：遊戲術語、角色名稱")
        form.addRow("清單名稱", self.name)
        layout.addLayout(form)
        add_row = QHBoxLayout()
        self.new_word = QLineEdit()
        self.new_word.setPlaceholderText("輸入單詞或人名")
        self.add_button = QPushButton("新增單詞")
        self.add_button.clicked.connect(self.add_word)
        self.new_word.returnPressed.connect(self.add_word)
        self.new_word.installEventFilter(self)
        add_row.addWidget(self.new_word, 1)
        add_row.addWidget(self.add_button)
        layout.addLayout(add_row)
        self.words = QPlainTextEdit()
        self.words.setPlaceholderText("每行一個單詞，也可直接貼上多行詞彙")
        layout.addWidget(self.words, 1)
        note = QLabel("清單儲存在本機，可用於之後的辨識。詞彙提示不會強制替換字幕文字。")
        note.setWordWrap(True)
        layout.addWidget(note)
        bottom = QHBoxLayout()
        self.delete_button = QPushButton("刪除清單")
        self.delete_button.clicked.connect(self.delete_list)
        bottom.addWidget(self.delete_button)
        bottom.addStretch()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("儲存並使用")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        bottom.addWidget(self.buttons)
        layout.addLayout(bottom)
        self.selector.currentIndexChanged.connect(self.select_list)
        self.selector.setCurrentIndex(max(0, self.selector.findData(selected_id)))
        self.select_list()

    def select_list(self):
        index = self.selector.currentIndex()
        if index != self.previous_index and ((self.name.text(), self.words.toPlainText()) != self.initial_fields
                                              or self.new_word.text().strip()):
            if QMessageBox.question(self, "尚未儲存", "切換清單會捨棄目前尚未儲存的修改，是否繼續？") != QMessageBox.StandardButton.Yes:
                self.selector.blockSignals(True)
                self.selector.setCurrentIndex(self.previous_index)
                self.selector.blockSignals(False)
                return
        ident = self.selector.currentData()
        item = next((r for r in self.library["lists"] if r["id"] == ident), None)
        self.name.setText(item["name"] if item else "")
        self.words.setPlainText(item["text"] if item else self.current_text)
        self.new_word.clear()
        self.delete_button.setEnabled(item is not None)
        self.previous_index = index
        self.initial_fields = (self.name.text(), self.words.toPlainText())

    def eventFilter(self, watched, event):
        if watched is self.new_word and event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.add_word()
            return True
        return super().eventFilter(watched, event)

    def add_word(self):
        text = self.new_word.text().strip()
        if text:
            words = clean_words(self.words.toPlainText() + "\n" + text)
            self.words.setPlainText("\n".join(words))
            self.new_word.clear()
            self.new_word.setFocus()

    def save(self):
        # Include a typed entry even if the user goes directly to Save.
        self.add_word()
        ident = self.selector.currentData() or uuid4().hex
        result = copy.deepcopy(self.library)
        item = {"id": ident, "name": self.name.text().strip(),
                "text": "\n".join(clean_words(self.words.toPlainText()))}
        result["lists"] = [item if r["id"] == ident else r for r in result["lists"]]
        if not any(r["id"] == ident for r in result["lists"]):
            result["lists"].append(item)
        try:
            save_library(result)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "無法儲存詞彙", str(exc))
            return
        self.saved_id = ident
        self.accept()

    def delete_list(self):
        if QMessageBox.question(self, "刪除詞彙清單", "刪除此清單？已儲存專案中的詞彙內容會保留。") != QMessageBox.StandardButton.Yes:
            return
        ident = self.selector.currentData()
        result = copy.deepcopy(self.library)
        result["lists"] = [r for r in result["lists"] if r["id"] != ident]
        try:
            save_library(result)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "無法刪除清單", str(exc))
            return
        self.accept()
