from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout


class MergePreviewDialog(QDialog):
    def __init__(self, project, proposal, play, parent=None):
        super().__init__(parent)
        self.setWindowTitle("短句合併預覽")
        self.resize(650,440)
        layout=QVBoxLayout(self)
        notice=QLabel("先試聽確認是否為同一句、同一說話者。採用後保留全部文字並鎖定，仍可復原。")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        labels={c.id:c for c in project.cues}
        layout.addWidget(QLabel("目前兩句"))
        before=QPlainTextEdit("\n\n".join(labels[i].text for i in proposal['cue_ids']))
        before.setReadOnly(True)
        layout.addWidget(before)
        layout.addWidget(QLabel("合併預覽"))
        self.preview=QPlainTextEdit(proposal['preview']['text'])
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview)
        a,b=proposal['range_ms']
        layout.addWidget(QLabel(f"{a/1000:.3f}–{b/1000:.3f} 秒 · 原文不改字"))
        self.play_button=QPushButton("試聽上下文")
        layout.addWidget(self.play_button)
        self.message=QLabel("")
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.play_button.clicked.connect(lambda:self.message.setText(play()))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("採用合併")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("保留原句")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
