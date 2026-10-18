"""Vocabulary management dialog; cancellation never writes user data."""
import copy
from uuid import uuid4

from PySide6.QtCore import QEvent, Qt

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .vocabulary import clean_words, compose_vocabulary, load_library, save_library
from .appearance import CheckBox, ChoiceBox as QComboBox


class VocabularySelectionDialog(QDialog):
    def __init__(self, parent, library, choice):
        super().__init__(parent)
        self.setWindowTitle("選擇詞彙提示")
        self.resize(520, 610)
        self.library = copy.deepcopy(library)
        self.base_text = choice["base_text"]
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("勾選本段會出現的人名與用語，可同時選擇多組。"))
        search = QLineEdit()
        search.setPlaceholderText("搜尋清單名稱")
        layout.addWidget(search)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        rows = QVBoxLayout(container)
        self.base_check = None
        if self.base_text:
            self.base_check = CheckBox("保留目前提示（原文快照）")
            self.base_check.setToolTip(self.base_text)
            self.base_check.setChecked(True)
            rows.addWidget(self.base_check)
            self.base_check.toggled.connect(self.update_preview)
        self.checks = {}
        for item in self.library["lists"]:
            check = CheckBox(item["name"])
            check.setToolTip(item["text"])
            check.setChecked(item["id"] in choice["ids"])
            self.checks[item["id"]] = check
            rows.addWidget(check)
            check.toggled.connect(self.update_preview)
        if not self.checks:
            rows.addWidget(QLabel("尚未建立清單，請回主畫面使用「管理詞彙」新增。"))
        rows.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        search.textChanged.connect(lambda text: [check.setVisible(text.casefold() in check.text().casefold())
                                                for check in self.checks.values()])
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(110)
        self.preview.setPlaceholderText("尚未勾選，將不使用詞彙提示")
        layout.addWidget(self.preview)
        note = QLabel("相同詞彙會自動去重。詞彙越多不一定越準，建議只選相關清單。")
        note.setWordWrap(True)
        layout.addWidget(note)
        bottom = QHBoxLayout()
        clear = QPushButton("全部取消勾選")
        clear.clicked.connect(self.clear_selection)
        bottom.addWidget(clear)
        bottom.addStretch()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("套用")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        bottom.addWidget(self.buttons)
        layout.addLayout(bottom)
        self.update_preview()

    def update_preview(self):
        self.choice = {"ids": [ident for ident, check in self.checks.items() if check.isChecked()],
                       "base_text": self.base_text if self.base_check and self.base_check.isChecked() else ""}
        try:
            self.prompt = compose_vocabulary(self.library, self.choice)
        except ValueError as exc:
            self.prompt = ""
            self.summary.setText(str(exc))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        else:
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            text = f"已選 {len(self.choice['ids'])} 組 · {len(clean_words(self.prompt))} 詞 · {len(self.prompt)} 字"
            if len(self.prompt.encode("utf-8")) > 600:
                text += "\n提示較長，模型可能只讀取部分內容；建議減少不相關清單。"
            self.summary.setText(text)
        self.preview.setPlainText(self.prompt)

    def clear_selection(self):
        for check in self.checks.values():
            check.setChecked(False)
        if self.base_check:
            self.base_check.setChecked(False)


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
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("儲存清單")
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
