from __future__ import annotations

import copy
import logging
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QToolButton, QMenu, QSlider,
)

from .domain import Project, TimingSettings
from .editing import merge_cues, review_cues, set_lock_state, split_cue
from .export import export_subtitles
from .media import Cancelled, audio_tracks, waveform
from .pipeline import run_pipeline
from .timing import apply_timing
from .alignment import analyze_cue, apply_candidate, restore_candidate, signature
from .playback import AudioPreview
from .proposal_ui import MergePreviewDialog
from .review_proposals import proposals as sentence_proposals
from .proposal_store import adopt_and_save, restore_and_save


class Waveform(QWidget):
    seek = Signal(int)

    def __init__(self):
        super().__init__()
        self.values: list[float] = []
        self.duration_ms = 1
        self.position_ms = 0
        self.region = (0, 0)
        self.speech_region = None
        self.candidate_region = None
        self.view_start_ms = 0
        self.view_end_ms = 0
        self.setMinimumHeight(100)
        self.setToolTip("點擊定位；滾輪縮放。藍底：字幕顯示；綠線：基礎發聲；橘虛線：候選發聲；黃線：游標。")

    def bounds(self):
        end = min(self.duration_ms, self.view_end_ms or self.duration_ms)
        start = max(0, min(self.view_start_ms, end - 1))
        return start, max(start + 1, end)

    def set_view(self, start, end):
        span = min(max(1, end - start), self.duration_ms)
        start = max(0, min(start, self.duration_ms - span))
        self.view_start_ms, self.view_end_ms = round(start), round(start + span)
        self.update()

    def show_all(self):
        self.set_view(0, self.duration_ms)

    def focus_region(self):
        lo, hi = self.region
        span = max(3000, hi - lo + 2000)
        center = (lo + hi) / 2
        self.set_view(center - span / 2, center + span / 2)

    def zoom(self, factor):
        start, end = self.bounds()
        center = self.position_ms if start <= self.position_ms <= end else (start + end) / 2
        span = min(self.duration_ms, max(1000, (end - start) / factor))
        self.set_view(center - span / 2, center + span / 2)

    def pan(self, direction):
        start, end = self.bounds()
        shift = (end - start) * 0.75 * direction
        self.set_view(start + shift, end + shift)

    def time_at(self, x):
        start, end = self.bounds()
        return max(start, min(end, round(start + x / max(1, self.width()) * (end - start))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151f30"))
        view_start, view_end = self.bounds()
        scale = self.width() / (view_end - view_start)
        lo, hi = self.region
        painter.fillRect(round((lo - view_start) * scale), 0, max(0, round((hi - lo) * scale)), self.height(), QColor("#23456b"))
        painter.setPen(QPen(QColor("#60c5bb"), 1))
        middle = (self.height() - 18) / 2
        count = len(self.values)
        for x in range(self.width()):
            first = max(0, int((view_start + x / scale) / max(1, self.duration_ms) * count))
            last = min(count, max(first + 1, int((view_start + (x + 1) / scale) / max(1, self.duration_ms) * count)))
            amplitude = max(self.values[first:last], default=0)
            height = min(1, amplitude) * (middle - 5)
            painter.drawLine(x, round(middle - height), x, round(middle + height))
        for region, color, style in [(self.speech_region, "#87e892", Qt.PenStyle.SolidLine),
                                      (self.candidate_region, "#ffa858", Qt.PenStyle.DashLine)]:
            if region:
                painter.setPen(QPen(QColor(color), 2, style))
                for edge in region:
                    x = round((edge - view_start) * scale)
                    painter.drawLine(x, 0, x, self.height() - 18)
        painter.setPen(QPen(QColor("#f8d477"), 2))
        x = round((self.position_ms - view_start) * scale)
        painter.drawLine(x, 0, x, self.height())
        painter.setPen(QColor("#c8d3e0"))
        for i in range(5):
            time_ms = round(view_start + (view_end - view_start) * i / 4)
            label = f"{time_ms // 60000:02}:{time_ms % 60000 / 1000:06.3f}"
            width = painter.fontMetrics().horizontalAdvance(label)
            x = min(max(0, round(self.width() * i / 4) - width // 2), max(0, self.width() - width))
            painter.drawText(x, self.height() - 3, label)

    def mousePressEvent(self, event):
        self.seek.emit(self.time_at(event.position().x()))

    def wheelEvent(self, event):
        self.zoom(2 if event.angleDelta().y() > 0 else 0.5)
        event.accept()


class SplitDialog(QDialog):
    def __init__(self, cue, position, parent=None):
        super().__init__(parent)
        self.setWindowTitle("拆分字幕")
        self.resize(540, 320)
        form = QFormLayout(self)
        self.time = QSpinBox()
        self.time.setRange(max(cue.start_ms, cue.speech_start_ms) + 1,
                           min(cue.end_ms, cue.speech_end_ms) - 1)
        middle = (cue.speech_start_ms + cue.speech_end_ms) // 2
        self.time.setValue(position if self.time.minimum() <= position <= self.time.maximum() else middle)
        form.addRow("切點（毫秒）", self.time)
        midpoint = len(cue.text) // 2
        spaces = [i for i, char in enumerate(cue.text) if char.isspace() and 0 < i < len(cue.text) - 1]
        cut = min(spaces, key=lambda i: abs(i - midpoint)) if spaces else midpoint
        self.left = QPlainTextEdit(cue.text[:cut].strip())
        self.right = QPlainTextEdit(cue.text[cut:].strip())
        form.addRow("前半句", self.left)
        form.addRow("後半句", self.right)
        form.addRow(QLabel("請一起確認切點與前後文字。拆分後兩句會鎖定，可用復原還原。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class Worker(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    failure = Signal(str)
    finished = Signal()

    def __init__(self, project: Project, path: Path, cancel: Event):
        super().__init__()
        self.project, self.path, self.cancel = project, path, cancel

    def run(self):
        try:
            self.result.emit(run_pipeline(self.project, self.path, self.cancel, self.progress.emit))
        except Cancelled as exc:
            self.result.emit(self.project)
            self.progress.emit(0, str(exc))
        except Exception as exc:
            details = traceback.format_exc()
            logging.error(details)
            self.path.with_suffix(".error.log").write_text(details, encoding="utf-8")
            self.result.emit(self.project)
            self.failure.emit(f"{exc}\n\n診斷已存至 {self.path.with_suffix('.error.log')}")
        finally:
            self.finished.emit()


class AlignmentWorker(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    failure = Signal(str)
    finished = Signal()

    def __init__(self, project, row, path, cancel, language):
        super().__init__()
        self.project, self.row, self.path = project, row, path
        self.cancel, self.language = cancel, language

    def run(self):
        try:
            self.progress.emit(10, "局部對齊選取句（使用已下載模型）；分析不會修改字幕…")
            candidate = analyze_cue(self.project, self.row,
                                    self.path.parent / (self.path.stem + ".cache") / "alignment",
                                    self.cancel, self.language)
            self.result.emit((self.row, candidate))
        except Cancelled:
            self.progress.emit(0, "已取消對齊；辨識與原字幕保留")
        except Exception as exc:
            self.failure.emit(str(exc))
        finally:
            self.finished.emit()


class RelinkWorker(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    failure = Signal(str)
    finished = Signal()

    def __init__(self, project, path, source, cancel):
        super().__init__()
        self.project, self.path, self.source, self.cancel = project, path, source, cancel

    def run(self):
        from .project_tools import relink_media
        try:
            self.result.emit(relink_media(self.project, self.path, self.source, self.cancel, self.progress.emit))
        except Cancelled:
            self.progress.emit(0, "重新連結已取消，原專案保留")
        except Exception as exc:
            self.failure.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Windows' offscreen Qt plugin does not enumerate system fonts. Register
        # an installed CJK font so test previews remain readable without bundling it.
        if sys.platform == "win32" and not QFontDatabase.families():
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msjh.ttc"
            if font.exists():
                QFontDatabase.addApplicationFont(str(font))
        if "Microsoft JhengHei UI" in QFontDatabase.families():
            self.setFont(QFont("Microsoft JhengHei UI", 10))
        self.setWindowTitle("自動打軸機 · Auto Subtitle 0.9.1 · Windows")
        self.repair_enabled = os.environ.get("AUTOSUB_ENABLE_REPAIR_UI") == "1"
        self.resize(1280, 850)
        self.project = Project()
        self.project_path: Path | None = None
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.cancel = Event()
        self.history: list[Project] = []
        self.future: list[Project] = []
        self.dirty = False
        self.refreshing = False
        self.closing = False
        self.play_end = 0
        self.play_start = 0

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("自動打軸機")
        title.setStyleSheet("font-size:25px; font-weight:700; padding:8px 0")
        layout.addWidget(title)
        layout.addWidget(QLabel("本機辨識 · 中／英／日／韓 · ASS / SRT　｜　建議先用 1–20 分鐘直播剪輯試跑"))
        toolbar = QHBoxLayout()
        self.open_media_button = self.button("匯入媒體", self.open_media, toolbar)
        self.export_button = self.button("匯出字幕", self.export, toolbar)
        self.tools_button = QToolButton()
        self.tools_button.setText("專案與工具")
        self.tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.tools_button)
        for name, label, callback in [("open_project_button", "開啟專案", self.open_project),
            ("save_button", "儲存專案", self.save_project), ("compare_button", "人工成品比較", self.compare_finished),
            ("relink_button", "重新連結媒體", self.relink), ("cache_button", "對齊快取", self.manage_cache)]:
            action = menu.addAction(label)
            action.triggered.connect(callback)
            setattr(self, name, action)
        self.cpu_retry_action = menu.addAction("GPU 失敗後改用 CPU 繼續", self.retry_on_cpu)
        self.cpu_retry_action.setEnabled(False)
        self.confirm_action = menu.addAction("確認選句並前往下一句（Ctrl+Enter）", self.confirm_and_next)
        self.confirm_action.setEnabled(False)
        for action in (self.compare_button,self.cache_button,self.confirm_action):
            action.setVisible(self.repair_enabled)
        self.tools_button.setMenu(menu)
        toolbar.addWidget(self.tools_button)
        self.advanced = QCheckBox("進階設定與修軸" if self.repair_enabled else "進階設定")
        toolbar.addWidget(self.advanced)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.source_label = QLabel("尚未選擇影片或音檔")
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        recognition = QGroupBox("辨識設定")
        form = QFormLayout(recognition)
        self.language = QComboBox()
        for label, code in [("日文", "ja"), ("中文", "zh"), ("英語", "en"), ("韓文", "ko"),
                            ("自動偵測", "auto"), ("多語混用（實驗）", "mixed")]:
            self.language.addItem(label, code)
        form.addRow("主要語言", self.language)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(["small", "large-v3", "turbo", "medium", "base", "tiny"])
        form.addRow("Whisper 模型", self.model)
        basic_form = form
        self.recognition_details = QWidget()
        form = QFormLayout(self.recognition_details)
        self.local_model_button = QPushButton("匯入本機模型資料夾")
        self.local_model_button.clicked.connect(self.local_model)
        form.addRow(self.local_model_button)
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("運算裝置", self.device)
        self.track = QComboBox()
        self.track.addItem("音軌 1", 0)
        form.addRow("音軌", self.track)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        form.addRow("VAD 門檻", self.threshold)
        self.glossary = QPlainTextEdit()
        self.glossary.setPlaceholderText("辨識提示：角色名、遊戲術語、暱稱；不會強制替換文字")
        self.glossary.setMaximumHeight(80)
        form.addRow("詞彙提示", self.glossary)
        games = QHBoxLayout()
        for name, words in [("APEX", "Apex Legends, エーペックス, アーマー, シールド"),
                            ("SF6", "ストリートファイター6, ドライブインパクト, パリィ"),
                            ("其他", "ARK, GTA, RUST")]:
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, text=words: self.glossary.appendPlainText(text))
            games.addWidget(button)
        form.addRow(games)
        basic_form.addRow(self.recognition_details)
        self.recognition_group = recognition
        side_layout.addWidget(recognition)

        timing = QGroupBox("打軸參數（毫秒）")
        timing_form = QFormLayout(timing)
        self.timing_inputs = {}
        labels = {"lead_ms": "句首先行", "tail_ms": "一般尾留", "max_tail_ms": "尾留上限",
                  "min_duration_ms": "最短顯示", "close_gap_ms": "補齊空隙門檻",
                  "soft_duration_ms": "開始尋找切點", "max_duration_ms": "分段時長上限",
                  "max_chars": "每則字數上限", "silence_split_ms": "停頓參考門檻（日文保留原句界）",
                  "boundary_search_ms": "0.2 邊界搜尋範圍", "start_search_ms": "保守策略句首搜尋",
                  "end_search_ms": "保守策略句尾搜尋"}
        for key, value in asdict(self.project.settings).items():
            spin = QSpinBox()
            spin.setRange(0, 50 if key == "lead_ms" else 1000 if key.endswith("search_ms") else 60000)
            spin.setValue(value)
            self.timing_inputs[key] = spin
            timing_form.addRow(labels[key], spin)
        self.boundary_policy = QComboBox()
        self.boundary_policy.addItem("0.2 基準（預設）", "legacy")
        self.boundary_policy.addItem("保守尾音（實驗）", "conservative")
        timing_form.addRow("邊界策略", self.boundary_policy)
        self.retime_button = QPushButton("套用參數／重新分段（保留鎖定句）")
        self.retime_button.clicked.connect(self.retime)
        timing_form.addRow(self.retime_button)
        preset_row = QHBoxLayout()
        self.button("儲存預設", self.save_preset, preset_row)
        self.button("載入預設", self.load_preset, preset_row)
        timing_form.addRow(preset_row)
        self.timing_group = timing
        side_layout.addWidget(timing)
        self.audio_group = QGroupBox("試聽與匯出")
        audio_form = QFormLayout(self.audio_group)
        self.output_devices = QComboBox()
        audio_form.addRow("音訊輸出", self.output_devices)
        self.wrap_export = QCheckBox("匯出時自動換成兩行")
        self.wrap_export.setToolTip("預設不插入換行；開啟後 ASS 會使用 \\N 表示換行。")
        audio_form.addRow(self.wrap_export)
        side_layout.addWidget(self.audio_group)
        note = QLabel("忠實保留口語；不翻譯。\n50ms 為目標，仍需試聽確認。\n長音須人工確認；多人發話未自動分離。\n目前專注 Windows，macOS 暫緩。")
        note.setWordWrap(True)
        side_layout.addWidget(note)
        self.advanced_note = note
        side_layout.addWidget(QLabel("1. 匯入影片或音檔\n2. 選語言及模型，開始辨識\n3. 按需試聽生成內容\n4. 匯出 ASS／SRT 到 Aegisub"))
        side_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(270)
        splitter.addWidget(scroll)

        workbench = QWidget()
        work_layout = QVBoxLayout(workbench)
        actions = QHBoxLayout()
        self.start_button = self.button("開始／繼續辨識", self.start, actions)
        self.cancel_button = self.button("取消並保留進度", self.request_cancel, actions)
        self.cancel_button.setEnabled(False)
        self.only_flags = QCheckBox("只顯示需複核" if self.repair_enabled else "只顯示生成提示")
        self.only_flags.toggled.connect(self.filter_rows)
        actions.addWidget(self.only_flags)
        work_layout.addLayout(actions)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["開始 ms", "結束 ms", "字幕文字", "複核提示", "鎖定", "長音", "ID", "已複核"])
        if not self.repair_enabled:
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.horizontalHeaderItem(3).setText("生成提示")
        self.table.setColumnHidden(6, True)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column, width in {0: 85, 1: 85, 3: 210, 4: 48, 5: 48, 7: 64}.items():
            self.table.setColumnWidth(column, width)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setStyleSheet("QTableWidget::item:selected { background:#2563eb; color:white; }"
            "QTableWidget::item:selected:!active { background:#2563eb; color:white; }")
        self.table.itemChanged.connect(self.edit_cell)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        work_layout.addWidget(self.table, 1)
        self.selection_label = QLabel("尚未選取字幕")
        self.selection_label.setStyleSheet("font-weight:600; color:#1d4ed8; padding:4px")
        work_layout.addWidget(self.selection_label)
        self.subtitle_preview = QLabel("選取字幕後可在此檢視完整文字")
        self.subtitle_preview.setWordWrap(True)
        self.subtitle_preview.setMinimumHeight(55)
        self.subtitle_preview.setTextFormat(Qt.TextFormat.PlainText)
        self.subtitle_preview.setStyleSheet("font-size:16px; padding:6px; background:#e9edf3; color:#172536")
        work_layout.addWidget(self.subtitle_preview)
        edit_actions = QHBoxLayout()
        self.undo_button = self.button("復原", self.undo, edit_actions)
        self.redo_button = self.button("重做", self.redo, edit_actions)
        self.unlock_button = self.button("解除選取句鎖定", self.unlock, edit_actions)
        self.split_button = self.button("拆分", self.split_selected, edit_actions)
        self.merge_button = self.button("合併下一句", self.merge_selected, edit_actions)
        self.merge_button.setToolTip("選取前一句；符合日文短句候選時先顯示預覽，其餘維持一般手動合併。")
        self.undo_button.setToolTip("復原本次操作；重開後可用仍符合目前專案的短句合併收據撤回。")
        edit_actions.addStretch()
        edit_actions.addWidget(QLabel("修改後自動鎖定"))
        self.edit_panel = QWidget()
        self.edit_panel.setLayout(edit_actions)
        work_layout.addWidget(self.edit_panel)
        candidate_actions = QHBoxLayout()
        self.analyze_button = self.button("分析選句候選", self.analyze_selected, candidate_actions)
        self.alignment_language = QComboBox()
        for label, code in [("日", "ja"), ("中", "zh"), ("英", "en"), ("韓", "ko")]:
            self.alignment_language.addItem(label, code)
        candidate_actions.addWidget(QLabel("本句語言"))
        candidate_actions.addWidget(self.alignment_language)
        self.play_candidate_button = self.button("試聽候選", self.play_candidate, candidate_actions)
        self.apply_candidate_button = self.button("採用候選", self.accept_candidate, candidate_actions)
        self.restore_candidate_button = self.button("還原採用前", self.revert_candidate, candidate_actions)
        self.candidate_panel = QWidget()
        self.candidate_panel.setLayout(candidate_actions)
        work_layout.addWidget(self.candidate_panel)
        self.candidate_label = QLabel("精細對齊為實驗功能；分析不改動字幕，試聽後再採用。")
        self.candidate_label.setTextFormat(Qt.TextFormat.PlainText)
        self.candidate_label.setWordWrap(True)
        self.candidate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        work_layout.addWidget(self.candidate_label)
        self.wave = Waveform()
        work_layout.addWidget(self.wave)
        work_layout.addWidget(QLabel("藍底：字幕顯示　綠線：基礎發聲　橘虛線：候選發聲　黃線：播放游標"))
        zoom_controls = QHBoxLayout()
        self.button("＋", lambda: self.wave.zoom(2), zoom_controls)
        self.button("－", lambda: self.wave.zoom(0.5), zoom_controls)
        self.button("前移", lambda: self.wave.pan(-1), zoom_controls)
        self.button("後移", lambda: self.wave.pan(1), zoom_controls)
        self.button("全覽", self.wave.show_all, zoom_controls)
        self.button("選句", self.wave.focus_region, zoom_controls)
        self.follow_cue = QCheckBox("跟隨選句")
        self.follow_cue.setChecked(True)
        zoom_controls.addWidget(self.follow_cue)
        self.set_start_button = self.button("起點←游標", lambda: self.set_edge(0), zoom_controls)
        self.set_end_button = self.button("終點←游標", lambda: self.set_edge(1), zoom_controls)
        self.zoom_panel = QWidget()
        self.zoom_panel.setLayout(zoom_controls)
        work_layout.addWidget(self.zoom_panel)
        playback = QHBoxLayout()
        self.play_button = self.button("試聽選取句", self.play_cue, playback)
        self.play_button.setEnabled(False)
        self.button("暫停／繼續", self.toggle_playback, playback)
        self.loop = QCheckBox("循環")
        playback.addWidget(self.loop)
        self.button("放大", lambda: self.wave.zoom(2), playback)
        self.button("縮小", lambda: self.wave.zoom(.5), playback)
        self.volume_label = QLabel("音量 80%")
        playback.addWidget(self.volume_label)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setMaximumWidth(100)
        playback.addWidget(self.volume)
        self.clock = QLabel("0.00 秒")
        playback.addWidget(self.clock)
        playback.addStretch()
        work_layout.addLayout(playback)
        splitter.addWidget(workbench)
        splitter.setStretchFactor(1, 1)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel("匯入媒體後選擇設定並開始。首次使用模型會下載。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.audio = AudioPreview(self)
        self.player = self.audio.player
        self.audio_output = self.audio.output
        self.audio.positionChanged.connect(self.position_changed)
        self.audio.message.connect(self.status.setText)
        self.volume.valueChanged.connect(lambda value: (self.audio_output.setVolume(value / 100),
                                                         self.volume_label.setText(f"音量 {value}%")))
        self.output_devices.currentIndexChanged.connect(lambda _: self.audio.set_device(self.output_devices.currentData()))
        self.audio.devices.audioOutputsChanged.connect(self.refresh_output_devices)
        self.refresh_output_devices()
        self.wave.seek.connect(self.seek)
        self.autosave = QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.setInterval(500)
        self.autosave.timeout.connect(self.persist)
        for key, callback in [("F5", self.toggle_playback), ("F6", self.next_flag),
                              ("Shift+F6", lambda: self.next_flag(-1)), ("Ctrl+Return", self.confirm_and_next),
                              ("Ctrl+Up", lambda: self.move_selection(-1)),
                              ("Ctrl+Down", lambda: self.move_selection(1)),
                              ("F7", lambda: self.set_edge(0)), ("F8", lambda: self.set_edge(1))]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            if key in {"Ctrl+Return","F7","F8"}:
                shortcut.setEnabled(self.repair_enabled)
        self.clock.setToolTip("F5 暫停／繼續；Ctrl+Enter 確認下一句；Ctrl+↑／↓ 上下句；F6／Shift+F6 前後待複核；F7／F8 設頭尾")
        if not self.repair_enabled:
            self.clock.setToolTip("F5 暫停／繼續；Ctrl+↑／↓ 上下句；F6／Shift+F6 前後生成提示")
        self.advanced.toggled.connect(self.set_advanced)
        self.set_advanced(False)
        splitter.setSizes([280, 1000])
        self.load_user_preferences()

    def load_user_preferences(self):
        from .preferences import load_preferences
        preferences, warning = load_preferences()
        if warning:
            self.status.setText(warning)
        if not preferences:
            return
        self.volume.setValue(preferences["volume"])
        self.language.setCurrentIndex(self.language.findData(preferences["language"]))
        self.model.setCurrentText(preferences["model"])
        self.device.setCurrentText(preferences["device"])
        self.threshold.setValue(preferences["vad_threshold"])
        self.glossary.setPlainText(preferences["glossary"])
        self.boundary_policy.setCurrentIndex(self.boundary_policy.findData(preferences["boundary_policy"]))
        for key, value in preferences["timing"].items():
            self.timing_inputs[key].setValue(value)
        self.wrap_export.setChecked(preferences["wrap_export"])
        device_id = preferences["output_device"]
        index = self.output_devices.findData(bytes.fromhex(device_id)) if device_id is not None else 0
        self.output_devices.setCurrentIndex(max(0, index))
        if index < 0:
            self.status.setText("先前的音訊輸出裝置不存在，已使用系統預設")

    def save_user_preferences(self):
        from .preferences import save_preferences
        try:
            device_id = self.output_devices.currentData()
            save_preferences({"schema_version": 1, "volume": self.volume.value(),
                "output_device": device_id.hex() if device_id is not None else None,
                "language": self.language.currentData(), "model": self.model.currentText().strip(),
                "device": self.device.currentText(), "vad_threshold": self.threshold.value(),
                "glossary": self.glossary.toPlainText(), "timing": asdict(self.read_settings()),
                "boundary_policy": self.boundary_policy.currentData(), "wrap_export": self.wrap_export.isChecked()})
        except Exception as exc:
            logging.warning("使用者設定未保存：%s", exc)
            self.status.setText(f"使用者設定未保存：{exc}")

    def retry_on_cpu(self):
        if self.thread or self.project.device != "cuda" or self.project.status != "failed":
            return
        self.project.review_events.append({"action": "device_fallback", "from": "cuda", "to": "cpu",
            "compute_from": "float16", "compute_to": "int8", "preserved_chunks": list(self.project.completed_chunks),
            "preserved_native_groups": len(self.project.native_groups), "native_resume_ms": self.project.native_resume_ms,
            "at_utc": datetime.now(timezone.utc).isoformat()})
        self.project.device = "cpu"
        self.device.setCurrentText("cpu")
        self.start()

    def set_advanced(self, enabled):
        for widget in (self.recognition_details, self.timing_group, self.audio_group, self.advanced_note):
            widget.setVisible(enabled)
        for widget in (self.edit_panel,self.candidate_panel,self.candidate_label,self.zoom_panel,self.loop):
            widget.setVisible(enabled and self.repair_enabled)
        for column in (4, 5, 7):
            self.table.setColumnHidden(column, not (enabled and self.repair_enabled))
        for row, cue in enumerate(self.project.cues):
            item = self.table.item(row, 3)
            if item:
                self.refreshing = True
                item.setText("；".join(cue.flags) if enabled else
                             "已複核" if cue.reviewed and self.repair_enabled else
                             ("待試聽複核" if self.repair_enabled else "建議試聽") if cue.flags else "")
                self.refreshing = False

    def refresh_output_devices(self):
        from PySide6.QtMultimedia import QMediaDevices
        selected = self.output_devices.currentData()
        self.output_devices.blockSignals(True)
        self.output_devices.clear()
        default = QMediaDevices.defaultAudioOutput()
        self.output_devices.addItem("系統預設 · " + (default.description() or "無裝置"), None)
        for device in QMediaDevices.audioOutputs():
            self.output_devices.addItem(device.description(), bytes(device.id()))
        self.output_devices.setCurrentIndex(max(0, self.output_devices.findData(selected)))
        self.output_devices.blockSignals(False)
        if selected is not None and self.output_devices.currentData() is None:
            self.audio.set_device(None)

    @staticmethod
    def button(text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def error(self, exc):
        QMessageBox.warning(self, "無法完成操作", str(exc))

    def local_model(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇 CTranslate2 Whisper 模型資料夾")
        if folder:
            self.model.setCurrentText(folder)

    def reset_history(self):
        self.history.clear()
        self.future.clear()

    def checkpoint(self):
        self.history.append(copy.deepcopy(self.project))
        self.history = self.history[-50:]
        self.future.clear()

    def persist(self):
        if self.project_path and not self.thread:
            try:
                self.project.save(self.project_path)
                self.dirty = False
            except Exception as exc:
                self.error(exc)

    def prepare_replace(self) -> bool:
        if self.dirty and self.project_path is None:
            answer = QMessageBox.question(self, "尚未儲存", "目前專案尚未儲存，是否先儲存？",
                                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                          QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save and not self.save_project():
                return False
        self.autosave.stop()
        self.persist()
        return not self.dirty or self.project_path is None

    def open_media(self):
        filename, _ = QFileDialog.getOpenFileName(self, "選擇影片或音訊", "", "媒體檔案 (*.*)")
        if not filename or not self.prepare_replace():
            return
        try:
            tracks = audio_tracks(Path(filename))
            if not tracks:
                raise ValueError("檔案中沒有音軌")
            self.project = Project(media_path=str(Path(filename).resolve()))
            self.project_path = None
            self.reset_history()
            self.dirty = True
            self.track.clear()
            for index, name in tracks:
                self.track.addItem(name, index)
            self.source_label.setText(filename)
            self.recognition_group.setEnabled(True)
            self.audio.set_source(filename)
            self.wave.values = []
            self.wave.region = (0, 0)
            self.wave.update()
            self.refresh()
            self.status.setText("已匯入；首次開始前請指定專案存放位置。")
        except Exception as exc:
            self.error(exc)

    def open_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "開啟專案", "", "Auto Subtitle (*.autosub.json)")
        if not filename or not self.prepare_replace():
            return
        try:
            project = Project.load(Path(filename))
            self.project, self.project_path = project, Path(filename)
            self.reset_history()
            self.dirty = False
            self.load_controls()
            self.refresh()
            self.load_playback()
            self.status.setText(f"已載入專案 · {self.project.status}；可繼續辨識或匯出")
            if project.language == "mixed" and project.asr_strategy == "fixed30-v1":
                self.status.setText(self.status.text() + "；此專案沿用舊混語策略，新建專案可使用語音段辨識")
            if project.language == "ja" and project.asr_strategy == "fixed30-v1":
                self.status.setText(self.status.text() + "；此專案沿用舊句界，新建專案可使用原生句界辨識")
        except Exception as exc:
            self.error(exc)

    def load_controls(self):
        p = self.project
        self.source_label.setText(p.media_path)
        self.language.setCurrentIndex(max(0, self.language.findData(p.language)))
        self.model.setCurrentText(p.model)
        self.device.setCurrentText(p.device)
        self.track.clear()
        self.track.addItem(f"音軌 {p.audio_stream + 1}", p.audio_stream)
        self.threshold.setValue(p.vad_threshold)
        self.glossary.setPlainText(p.glossary)
        for key, value in asdict(p.settings).items():
            self.timing_inputs[key].setValue(value)
        self.boundary_policy.setCurrentIndex(max(0, self.boundary_policy.findData(p.boundary_policy)))
        self.alignment_language.setCurrentIndex(max(0, self.alignment_language.findData(p.language)))
        self.recognition_group.setEnabled(not p.source_fingerprint)

    def read_settings(self):
        result = TimingSettings(**{key: spin.value() for key, spin in self.timing_inputs.items()})
        result.validate()
        return result

    def project_from_controls(self):
        working = copy.deepcopy(self.project)
        working.settings = self.read_settings()
        working.boundary_policy = self.boundary_policy.currentData()
        if not working.source_fingerprint:
            working.language = self.language.currentData()
            working.model = self.model.currentText().strip()
            working.device = self.device.currentText()
            working.glossary = self.glossary.toPlainText()
            working.audio_stream = self.track.currentData() or 0
            working.vad_threshold = self.threshold.value()
        return working

    def save_project(self) -> bool:
        try:
            if not self.project.media_path:
                raise ValueError("請先匯入媒體")
            working = self.project_from_controls()
            if self.project_path is None:
                filename, _ = QFileDialog.getSaveFileName(self, "儲存專案", str(Path(self.project.media_path).with_suffix(".autosub.json")),
                                                         "Auto Subtitle (*.autosub.json)")
                if not filename:
                    return False
                self.project_path = Path(filename)
            working.save(self.project_path)
            self.project = working
            self.dirty = False
            self.status.setText(f"已儲存 {self.project_path}")
            return True
        except Exception as exc:
            self.error(exc)
            return False

    def start(self):
        if self.thread or not self.save_project():
            return
        self.autosave.stop()
        self.save_user_preferences()
        self.audio.stop()
        self.reset_history()
        self.cancel = Event()
        self.thread = QThread(self)
        self.worker = Worker(copy.deepcopy(self.project), self.project_path, self.cancel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.report_progress)
        self.worker.result.connect(self.receive_result)
        self.worker.failure.connect(self.error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.work_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.set_busy(True)
        self.thread.start()

    def set_busy(self, busy):
        self.cpu_retry_action.setEnabled(not busy and self.project.device == "cuda" and self.project.status == "failed")
        for widget in [self.open_media_button, self.open_project_button, self.save_button, self.export_button,
                       self.start_button, self.table, self.timing_group, self.undo_button, self.redo_button,
                       self.unlock_button, self.split_button, self.merge_button,
                       self.set_start_button, self.set_end_button, self.analyze_button,
                       self.apply_candidate_button, self.restore_candidate_button, self.alignment_language,
                       self.compare_button, self.relink_button, self.cache_button]:
            widget.setEnabled(not busy)
        self.recognition_group.setEnabled(not busy and not self.project.source_fingerprint)
        self.cancel_button.setEnabled(busy)
        self.play_button.setEnabled(not busy and self.table.currentRow() >= 0 and self.table.selectionModel().hasSelection())
        self.confirm_action.setEnabled(self.play_button.isEnabled())

    def report_progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)

    def request_cancel(self):
        self.cancel.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("正在取消；目前模型載入／推論結束後會保存進度…")

    def receive_result(self, project):
        self.project = project
        self.alignment_language.setCurrentIndex(max(0, self.alignment_language.findData(project.language)))
        self.refresh()
        self.load_playback()
        if project.status == "failed":
            self.status.setText("辨識中斷；已保存進度。修正問題後可按繼續辨識。")

    def work_finished(self):
        self.thread = None
        self.worker = None
        self.set_busy(False)
        if self.closing:
            self.close()

    def retime(self):
        try:
            settings = self.read_settings()
            self.checkpoint()
            self.project.settings = settings
            self.project.boundary_policy = self.boundary_policy.currentData()
            self.project.cues = apply_timing(self.project)
            self.changed("retime")
            self.refresh()
            self.status.setText("已重新分段與打軸；人工鎖定句已保留")
        except Exception as exc:
            self.error(exc)

    def changed(self, action=None):
        if action:
            self.project.review_events.append({"action": action,
                "at_utc": datetime.now(timezone.utc).isoformat()})
        self.dirty = True
        self.autosave.start()

    def refresh(self):
        self.cpu_retry_action.setEnabled(not self.thread and self.project.device == "cuda" and self.project.status == "failed")
        previous_row = self.table.currentRow()
        item = self.table.item(self.table.currentRow(), 6)
        selected_id = item.text() if item else None
        review_cues(self.project)
        self.refreshing = True
        self.table.setRowCount(len(self.project.cues))
        for row, cue in enumerate(self.project.cues):
            values = [str(cue.start_ms), str(cue.end_ms), cue.text, "；".join(cue.flags), "", "", cue.id, ""]
            if not self.advanced.isChecked():
                values[3] = ("已複核" if cue.reviewed else "待試聽複核" if cue.flags else "") if self.repair_enabled else ("建議試聽" if cue.flags else "")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 3:
                    item.setToolTip("；".join(cue.flags))
                if column == 4:
                    item.setToolTip("拆分組會一起鎖定／解鎖；解鎖後重算會使用原始辨識文字")
                if column in {3, 6}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column in {4, 5, 7}:
                    item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
                    item.setCheckState(Qt.CheckState.Checked if (cue.locked if column == 4 else
                                                                 cue.elongated if column == 5 else cue.reviewed)
                                       else Qt.CheckState.Unchecked)
                if cue.flags and not cue.reviewed:
                    item.setBackground(QColor("#fff0cf"))
                    item.setForeground(QColor("#252525"))
                self.table.setItem(row, column, item)
        self.refreshing = False
        self.filter_rows()
        visible = [i for i in range(len(self.project.cues)) if not self.table.isRowHidden(i)]
        fallback = next((i for i in visible if i >= previous_row), visible[0] if visible else -1)
        target = next((i for i in visible if self.project.cues[i].id == selected_id), fallback)
        if target >= 0:
            self.table.selectRow(target)
        else:
            self.table.clearSelection()
        self.select_cue()

    def filter_rows(self):
        for row, cue in enumerate(self.project.cues):
            self.table.setRowHidden(row, self.only_flags.isChecked() and (not cue.flags or cue.reviewed))
        row = self.table.currentRow()
        if row < 0 or self.table.isRowHidden(row):
            indexes = list(range(max(0, row + 1), len(self.project.cues))) + list(range(0, max(0, row + 1)))
            visible = next((i for i in indexes if not self.table.isRowHidden(i)), None)
            if visible is None:
                self.table.clearSelection()
                self.table.setCurrentItem(None)
            else:
                self.table.selectRow(visible)
        self.select_cue()

    def edit_cell(self, item):
        if not self.repair_enabled or self.refreshing or self.thread:
            return
        row, column = item.row(), item.column()
        cue = copy.deepcopy(self.project.cues[row])
        try:
            if column == 0:
                cue.start_ms = int(item.text())
            elif column == 1:
                cue.end_ms = int(item.text())
            elif column == 2:
                if not item.text().strip():
                    raise ValueError("字幕文字不可為空")
                cue.text = item.text()
            elif column == 4:
                cue.locked = item.checkState() == Qt.CheckState.Checked
            elif column == 5:
                cue.elongated = item.checkState() == Qt.CheckState.Checked
                if cue.elongated:
                    cue.end_ms = cue.speech_end_ms
            elif column == 7:
                cue.reviewed = item.checkState() == Qt.CheckState.Checked
            else:
                return
            if not 0 <= cue.start_ms < cue.end_ms <= self.project.duration_ms:
                raise ValueError("時間須滿足 0 ≤ 開始 < 結束 ≤ 媒體時長")
            if column not in {4, 7}:
                cue.locked = True
                cue.reviewed = False
            self.checkpoint()
            if column == 4:
                self.project.cues = set_lock_state(self.project, row, cue.locked)
            else:
                self.project.cues[row] = cue
                if column != 7 and cue.edit_group is not None:
                    self.project.cues = set_lock_state(self.project, row, True)
            self.changed({0: "manual_time", 1: "manual_time", 2: "manual_text",
                          4: "lock", 5: "elongated", 7: "review"}[column])
            self.refresh()
            if not self.table.isRowHidden(row):
                self.table.selectRow(row)
        except Exception as exc:
            self.refresh()
            self.error(exc)

    def undo(self):
        if not self.repair_enabled or self.thread:
            return
        if self.history:
            events = copy.deepcopy(self.project.review_events)
            self.future.append(copy.deepcopy(self.project))
            self.project = self.history.pop()
            self.project.review_events = events
            self.load_controls()
            self.refresh()
            self.changed("undo")
        elif self.project_path and any(e.get("action") == "apply_merge_proposal" for e in self.project.review_events):
            try:
                current = self.project_from_controls()
                restored = restore_and_save(current, self.project_path)
                self.autosave.stop()
                self.future.append(copy.deepcopy(current))
                self.project = restored
                self.load_controls()
                self.refresh()
                self.dirty = False
                self.status.setText("已依收據復原合併並儲存；原文及人工狀態已還原")
            except Exception as exc:
                self.error(exc)

    def redo(self):
        if not self.repair_enabled:
            return
        if self.future:
            events = copy.deepcopy(self.project.review_events)
            self.history.append(copy.deepcopy(self.project))
            self.project = self.future.pop()
            self.project.review_events = events
            self.load_controls()
            self.refresh()
            self.changed("redo")

    def unlock(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row >= 0:
            self.checkpoint()
            self.project.cues = set_lock_state(self.project, row, False)
            self.changed("unlock")
            self.refresh()

    def split_selected(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row < 0 or self.thread:
            return
        cue = self.project.cues[row]
        if len(cue.text.strip()) < 2 or min(cue.end_ms, cue.speech_end_ms) - max(cue.start_ms, cue.speech_start_ms) < 2:
            self.error("這則字幕沒有足夠的文字或時間可拆分")
            return
        dialog = SplitDialog(cue, self.audio.position(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                updated = split_cue(self.project, row, dialog.time.value(),
                                    dialog.left.toPlainText(), dialog.right.toPlainText())
                self.checkpoint()
                self.project.cues = updated
                self.changed("split")
                self.refresh()
                self.table.selectRow(row)
            except Exception as exc:
                self.error(exc)

    def merge_selected(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row < 0 or self.thread:
            return
        if row + 1 < self.table.rowCount() and self.table.isRowHidden(row + 1):
            self.error("下一句被篩選隱藏；請先取消篩選以確認合併對象")
            return
        try:
            if self.project.asr_strategy == "native-v1" and self.project.native_complete:
                working = self.project_from_controls()
                ids = [c.id for c in working.cues[row:row+2]]
                proposal = next((p for p in sentence_proposals(working)
                                 if p['kind'] == 'merge' and p['cue_ids'] == ids), None)
                if proposal is not None:
                    def play_context():
                        a,b = proposal['range_ms']
                        self.audio.audition(max(0,a-800),min(working.duration_ms,b+800))
                        return self.status.text()
                    dialog = MergePreviewDialog(working, proposal, play_context, self)
                    try:
                        accepted = dialog.exec() == QDialog.DialogCode.Accepted
                    finally:
                        self.audio.stop()
                    if not accepted:
                        return
                    if not self.project_path:
                        raise ValueError("請先儲存專案，再採用可撤回的合併")
                    updated = adopt_and_save(working, proposal, self.project_path)
                    self.autosave.stop()
                    self.checkpoint()
                    self.project = updated
                    self.dirty = False
                    self.refresh()
                    self.table.selectRow(row)
                    self.status.setText("已採用合併並保存收據；存檔重開後可按「復原」撤回")
                    return
            updated = merge_cues(self.project, row)
            self.checkpoint()
            self.project.cues = updated
            self.changed("merge")
            self.refresh()
            self.table.selectRow(row)
            self.status.setText("已手動合併；本次作業可按「復原」還原")
        except Exception as exc:
            self.error(exc)

    def set_edge(self, column):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row >= 0 and not self.thread:
            self.table.item(row, column).setText(str(self.audio.position()))

    def selection_changed(self):
        if self.refreshing:
            return
        self.select_cue()
        row = self.table.currentRow()
        if not self.thread and self.table.selectionModel().hasSelection() and 0 <= row < len(self.project.cues):
            self.audio.seek(self.project.cues[row].start_ms)

    def select_cue(self):
        row = self.table.currentRow()
        self.wave.speech_region = None
        self.wave.candidate_region = None
        selected = self.table.selectionModel().hasSelection() and not self.table.isRowHidden(row)
        self.play_button.setEnabled(bool(selected and 0 <= row < len(self.project.cues) and not self.thread))
        self.confirm_action.setEnabled(bool(selected and 0 <= row < len(self.project.cues) and not self.thread))
        reviewed = sum(cue.reviewed for cue in self.project.cues)
        review_progress = f"已複核 {reviewed}／{len(self.project.cues)}" if self.repair_enabled else "人工修整請使用 Aegisub"
        if selected and 0 <= row < len(self.project.cues):
            cue = self.project.cues[row]
            self.selection_label.setText(f"已選取第 {row + 1}／{len(self.project.cues)} 句　"
                f"{cue.start_ms / 1000:.3f}–{cue.end_ms / 1000:.3f} 秒　｜　{review_progress}")
            self.wave.region = (cue.start_ms, cue.end_ms)
            self.wave.speech_region = (cue.speech_start_ms, cue.speech_end_ms)
            candidate = cue.candidate
            trace = cue.timing_trace
            detail = f"原詞 {trace.get('raw', '未記錄')} · VAD {trace.get('vad', '未記錄')} · 基礎 {[cue.speech_start_ms, cue.speech_end_ms]}"
            if candidate and self.repair_enabled:
                fresh = candidate["signature"] == signature(self.project, cue)
                if fresh:
                    self.wave.candidate_region = candidate["base"]
                delta = [candidate["base"][0] - cue.speech_start_ms,
                         candidate["base"][1] - cue.speech_end_ms]
                detail += (f"\n候選 {candidate['base']}（差 {delta[0]:+d} / {delta[1]:+d}ms）"
                           f" · { {'aligned': '對齊', 'partial': '部分對齊／部分回退', 'fallback': '保守回退'}[candidate['status']]}"
                           f" · {'待試聽' if fresh else '已過期／已採用，需重新分析'}\n" + "；".join(candidate["reasons"]))
            self.candidate_label.setText(detail)
            self.wave.update()
            self.subtitle_preview.setText(cue.text)
            if self.follow_cue.isChecked():
                self.wave.focus_region()
        else:
            self.selection_label.setText(f"尚未選取字幕　｜　{review_progress}")
            self.subtitle_preview.setText("選取字幕後可在此檢視完整文字")
            self.wave.region = (0, 0)
            self.candidate_label.setText("選取字幕以分析候選；分析不改動目前字幕。")
        self.wave.update()

    def analyze_selected(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row < 0 or self.thread or not self.save_project():
            return
        self.autosave.stop()
        self.audio.stop()
        self.cancel = Event()
        self.thread = QThread(self)
        self.worker = AlignmentWorker(copy.deepcopy(self.project), row, self.project_path,
                                      self.cancel, self.alignment_language.currentData())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self.receive_candidate)
        self.worker.progress.connect(self.report_progress)
        self.worker.failure.connect(self.error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.work_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.set_busy(True)
        self.thread.start()

    def receive_candidate(self, payload):
        row, candidate = payload
        self.checkpoint()
        self.project.cues[row].candidate = candidate
        self.project.save(self.project_path)
        self.dirty = False
        self.refresh()
        self.table.selectRow(row)
        self.status.setText("候選已保存；目前字幕未變更。請試聽原句與候選後決定是否採用。")

    def accept_candidate(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row < 0 or self.thread:
            return
        try:
            updated = copy.deepcopy(self.project)
            apply_candidate(updated, row)
            updated.validate()
            self.checkpoint()
            self.project = updated
            self.changed("apply_candidate")
            self.refresh()
            self.table.selectRow(row)
        except Exception as exc:
            self.error(exc)

    def revert_candidate(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if row < 0 or self.thread:
            return
        try:
            updated = copy.deepcopy(self.project)
            restore_candidate(updated, row)
            self.checkpoint()
            self.project = updated
            self.changed("restore_candidate")
            self.refresh()
            self.table.selectRow(row)
        except Exception as exc:
            self.error(exc)

    def play_candidate(self):
        row = self.table.currentRow()
        if row < 0:
            return
        cue = self.project.cues[row]
        if not cue.candidate or cue.candidate["signature"] != signature(self.project, cue):
            self.error("候選不存在或已過期，請重新分析")
            return
        start, end = cue.candidate["display"]
        self.play_start, self.play_end = max(0, start - 250), min(self.project.duration_ms, end + 250)
        self.audio.audition(self.play_start, self.play_end, lambda: self.loop.isChecked())

    def load_playback(self):
        self.audio.set_source(self.project.media_path, self.project.audio_path, self.project.audio_stream)
        self.wave.duration_ms = self.project.duration_ms or 1
        self.wave.show_all()
        self.wave.region = (0, 0)
        self.wave.values = []
        if self.project.audio_path and Path(self.project.audio_path).exists():
            try:
                self.wave.values = waveform(Path(self.project.audio_path))
            except Exception as exc:
                self.status.setText(f"波形無法載入：{exc}")
        self.wave.update()
        self.select_cue()
        row = self.table.currentRow()
        if self.table.selectionModel().hasSelection() and 0 <= row < len(self.project.cues):
            self.audio.seek(self.project.cues[row].start_ms)

    def play_cue(self):
        row = self.table.currentRow()
        if row < 0 or not self.table.selectionModel().hasSelection():
            self.status.setText("請先點選一列字幕，再試聽")
            return
        cue = self.project.cues[row]
        self.play_start = max(0, cue.start_ms - 250)
        self.play_end = min(self.project.duration_ms, cue.end_ms + 250)
        self.audio.audition(self.play_start, self.play_end, lambda: self.loop.isChecked())

    def compare_finished(self):
        if self.thread:
            return
        if not self.project.cues:
            self.error("請先完成辨識或載入含字幕的專案")
            return
        if not self.save_project():
            return
        self.audio.pause()
        from .review_ui import ComparisonDialog
        dialog = ComparisonDialog(self.project, self.project_path, Waveform, self)
        dialog.exec()

    def relink(self):
        if self.thread or not self.project_path:
            self.error("請先開啟已儲存的專案")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "選擇相同剪輯版本的新媒體位置", "", "媒體 (*.*)")
        if not filename:
            return
        self.autosave.stop()
        self.persist()
        self.audio.stop()
        self.cancel = Event()
        self.thread = QThread(self)
        self.worker = RelinkWorker(copy.deepcopy(self.project), self.project_path, Path(filename), self.cancel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.report_progress)
        self.worker.result.connect(self.receive_relinked)
        self.worker.failure.connect(self.error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.work_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.set_busy(True)
        self.thread.start()

    def receive_relinked(self, project):
        from .project_tools import migrate_review_identity
        try:
            migrate_review_identity(self.project, project, self.project_path)
        except Exception as exc:
            self.error(f"媒體驗證成功，但複核紀錄遷移失敗，原紀錄保留：{exc}")
        self.checkpoint()
        self.project = project
        self.changed("relink")
        self.project.save(self.project_path)
        self.dirty = False
        self.load_controls()
        self.refresh()
        self.load_playback()

    def manage_cache(self):
        if self.thread or not self.project_path:
            self.error("請先開啟已儲存的專案")
            return
        from .project_tools import alignment_cache_info, clear_alignment_cache
        try:
            _, files, size = alignment_cache_info(self.project_path)
            choice = QMessageBox.question(self, "對齊快取",
                f"本專案有 {len(files)} 個局部對齊快取，共 {size / 1048576:.2f} MiB。\n"
                "是否清除？之後可重新分析。分析 WAV、辨識進度及人工標註會保留。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if choice == QMessageBox.StandardButton.Yes:
                count, _ = clear_alignment_cache(self.project_path)
                self.status.setText(f"已清除 {count} 個對齊快取")
        except Exception as exc:
            self.error(exc)

    def toggle_playback(self):
        if self.audio.is_playing():
            self.audio.pause()
        elif not self.audio.resume():
            self.play_cue()

    def move_selection(self, direction):
        if self.thread:
            return
        row = self.table.currentRow()
        indexes = range(row + 1, self.table.rowCount()) if direction > 0 else range(row - 1, -1, -1)
        for index in indexes:
            if not self.table.isRowHidden(index):
                self.table.selectRow(index)
                self.table.scrollToItem(self.table.item(index, 2))
                return

    def next_flag(self, direction=1):
        if self.thread:
            return
        row, count = self.table.currentRow(), len(self.project.cues)
        if not count:
            return
        if row < 0 and direction < 0:
            row = 0
        indexes = [(row + direction * step) % count for step in range(1, count + 1)]
        for i in indexes:
            cue = self.project.cues[i]
            if cue.flags and not cue.reviewed and not self.table.isRowHidden(i):
                self.table.selectRow(i)
                self.table.scrollToItem(self.table.item(i, 2))
                return

    def confirm_and_next(self):
        if not self.repair_enabled:
            return
        row = self.table.currentRow()
        if self.thread or row < 0 or not self.table.selectionModel().hasSelection() or self.table.isRowHidden(row):
            return
        if self.table.state() == QTableWidget.State.EditingState or self.glossary.hasFocus():
            self.status.setText("請先按 Enter 完成文字／時間編輯，再確認複核")
            return
        self.audio.stop()
        count = len(self.project.cues)
        indexes = list(range(row + 1, count)) + list(range(0, row))
        next_id = next((self.project.cues[i].id for i in indexes if not self.project.cues[i].reviewed
                        and (not self.only_flags.isChecked() or self.project.cues[i].flags)), None)
        if not self.project.cues[row].reviewed:
            self.table.item(row, 7).setCheckState(Qt.CheckState.Checked)
        if next_id is not None:
            index = next((i for i, cue in enumerate(self.project.cues) if cue.id == next_id), None)
            if index is not None and not self.table.isRowHidden(index):
                self.table.selectRow(index)
                self.table.scrollToItem(self.table.item(index, 2))
            self.status.setText("已記錄人工複核，前往下一待確認句；不會自動播放")
        else:
            self.status.setText("已記錄人工複核；目前篩選範圍內沒有下一待確認句")

    def seek(self, position):
        self.play_end = 0
        self.audio.seek(position)

    def position_changed(self, position):
        self.wave.position_ms = position
        self.wave.update()
        self.clock.setText(f"{position / 1000:.2f} 秒")

    def save_preset(self):
        import json
        try:
            settings = self.read_settings()
            filename, _ = QFileDialog.getSaveFileName(self, "儲存打軸預設", "timing-preset.json", "JSON (*.json)")
            if filename:
                Path(filename).write_text(json.dumps({"settings": asdict(settings),
                    "boundary_policy": self.boundary_policy.currentData()}, indent=2), encoding="utf-8")
        except Exception as exc:
            self.error(exc)

    def load_preset(self):
        import json
        filename, _ = QFileDialog.getOpenFileName(self, "載入打軸預設", "", "JSON (*.json)")
        if not filename:
            return
        try:
            raw = json.loads(Path(filename).read_text(encoding="utf-8"))
            settings = TimingSettings(**raw.get("settings", raw))
            settings.validate()
            policy = raw.get("boundary_policy", "legacy")
            if policy not in {"legacy", "conservative"}:
                raise ValueError("不支援的邊界策略")
            self.boundary_policy.setCurrentIndex(self.boundary_policy.findData(policy))
            for key, value in asdict(settings).items():
                self.timing_inputs[key].setValue(value)
            self.status.setText("已載入預設；按「套用參數」重新打軸")
        except Exception as exc:
            self.error(exc)

    def export(self):
        try:
            if not self.project.cues:
                raise ValueError("尚無字幕，請先執行辨識")
            filename, selected = QFileDialog.getSaveFileName(self, "匯出字幕", str(Path(self.project.media_path).with_suffix(".ass")),
                                                            "ASS 字幕 (*.ass);;SRT 字幕 (*.srt)")
            if filename:
                path = Path(filename)
                path = path.with_suffix(".srt" if selected.startswith("SRT") else ".ass")
                export_subtitles(self.project, path, wrap_lines=self.wrap_export.isChecked())
                self.persist()
                suffix = "（辨識尚未完成，目前僅匯出已完成部分）" if self.project.status != "complete" else ""
                self.status.setText(f"已匯出 {path}{suffix}")
        except Exception as exc:
            self.error(exc)

    def closeEvent(self, event):
        if self.thread:
            self.closing = True
            self.request_cancel()
            self.status.setText("正在保存進度，完成後關閉視窗…")
            event.ignore()
            return
        if not self.prepare_replace():
            event.ignore()
            return
        self.audio.stop()
        self.save_user_preferences()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Auto Subtitle")
    app.setOrganizationName("AutoSubtitle")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
