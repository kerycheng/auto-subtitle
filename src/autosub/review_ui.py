"""Post-ASR human-reference review. This module is not imported by the ASR pipeline."""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .comparison import (ComparisonSettings, compare_file, export_report, load_session,
                         repair_pair, save_session, set_decision, summary, session_key)
from .media import Cancelled, waveform
from .paired_evaluation import annotation_template, paired_report
from .playback import AudioPreview
from .triage import review_order, review_priority

DECISION_LABELS = [("可直接使用", "usable"), ("需微調時間", "adjust"), ("需拆分／合併", "split_merge"),
                   ("需重做／補句", "redo"), ("對應錯誤", "wrong_pair"), ("排除（記錄原因）", "exclude")]
LABELS = dict((code, label) for label, code in DECISION_LABELS)
LABELS["unreviewed"] = "未複核"


class ComparisonWorker(QObject):
    result = Signal(object)
    progress = Signal(str)
    failure = Signal(str)
    finished = Signal()

    def __init__(self, project, reference, settings, cancel, styles):
        super().__init__()
        self.project, self.reference, self.settings = project, reference, settings
        self.cancel, self.styles = cancel, styles

    def run(self):
        try:
            self.result.emit(compare_file(self.project, self.reference, self.settings, self.cancel,
                                          lambda _, message: self.progress.emit(message), self.styles))
        except Cancelled:
            self.progress.emit("已取消；先前的比較與人工判定仍保留")
        except Exception as exc:
            self.failure.emit(str(exc))
        finally:
            self.finished.emit()


class PairDialog(QDialog):
    def __init__(self, session, row, parent=None):
        super().__init__(parent)
        self.setWindowTitle("人工修正對應（取代相關自動配對）")
        self.resize(1000, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("勾選屬於同一段內容的人工與自動字幕。受影響的其他事件會保留為未配對；舊判定將清除。"))
        columns = QHBoxLayout()
        self.ref_list, self.gen_list = QListWidget(), QListWidget()
        for widget, events, selected in [(self.ref_list, session["reference_events"], row["reference_ids"]),
                                          (self.gen_list, session["generated_events"], row["generated_ids"])]:
            for event in events:
                item = QListWidgetItem(f"{event['start_ms']}–{event['end_ms']}  {event['text']}")
                item.setData(Qt.ItemDataRole.UserRole, event["id"])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if event["id"] in selected else Qt.CheckState.Unchecked)
                widget.addItem(item)
            columns.addWidget(widget)
        layout.addLayout(columns)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selections(self):
        return [[widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(widget.count())
                 if widget.item(i).checkState() == Qt.CheckState.Checked] for widget in (self.ref_list, self.gen_list)]


class ComparisonDialog(QDialog):
    def __init__(self, project, project_path, wave_class, parent=None):
        super().__init__(parent)
        self.project = copy.deepcopy(project)
        self.project_path = Path(project_path)
        self.store = self.project_path.parent / (self.project_path.stem + ".review")
        self.session = None
        self.session_path = None
        self.history = []
        self.thread = self.worker = None
        self.cancel = Event()
        self.closing = False
        self.timer_started = None
        self.play_end = 0
        self._loading = False
        self.setWindowTitle("人工成品比較與聲學標註 · 評估用途")
        self.resize(1360, 940)
        layout = QVBoxLayout(self)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText("選擇同一版本素材的人工 ASS／SRT；不會輸入辨識模型")
        source = QHBoxLayout()
        source.addWidget(self.reference, 1)
        self.pick_button = self.button("選人工成品", self.choose_reference, source)
        layout.addLayout(source)
        options = QHBoxLayout()
        self.offset = QSpinBox()
        self.offset.setRange(-86_400_000, 86_400_000)
        self.offset.setToolTip("人工字幕時間減去此值後，對應專案時間。例如原片 60 秒剪出的音訊填 60000。")
        self.tolerance = QSpinBox()
        self.tolerance.setRange(0, 1000)
        self.tolerance.setValue(100)
        self.styles = QLineEdit("註解,注释,注釋")
        for label, widget in [("參考偏移 ms", self.offset), ("比較容差 ms", self.tolerance), ("排除樣式", self.styles)]:
            options.addWidget(QLabel(label))
            options.addWidget(widget)
        self.compute_button = self.button("建立比較", self.start_compare, options)
        self.cancel_button = self.button("取消", lambda: self.cancel.set(), options)
        self.cancel_button.setEnabled(False)
        layout.addLayout(options)
        self.timeline = QCheckBox("我已確認影片剪輯版本、音軌與偏移相同（未確認時僅作初步比較）")
        self.timeline.toggled.connect(self.confirm_timeline)
        layout.addWidget(self.timeline)
        self.hint = QLabel("時間配對是建議；翻譯文字不直接計算辨識正確率。正值表示自動軸較晚。")
        self.hint.setWordWrap(True)
        self.hint.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.hint)
        self.stats = QLabel("尚未比較")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["對應", "人工字幕", "自動字幕", "起點差 ms", "終點差 ms", "提示", "人工判定"])
        for column, width in {0: 155, 3: 85, 4: 85, 5: 155, 6: 100}.items():
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeaderItem(0).setToolTip("人工句數：自動句數。未配對區間需聽辨，不能直接當作漏句或多餘辨識。")
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for col in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.select_row)
        layout.addWidget(self.table, 1)
        self.texts = QLabel("選取一組對應後可查看完整文字並輪流試聽")
        self.texts.setWordWrap(True)
        self.texts.setTextFormat(Qt.TextFormat.PlainText)
        self.texts.setMinimumHeight(50)
        layout.addWidget(self.texts)
        decisions = QHBoxLayout()
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("複核者")
        decisions.addWidget(self.reviewer)
        self.decision = QComboBox()
        for label, code in DECISION_LABELS:
            self.decision.addItem(label, code)
        decisions.addWidget(self.decision)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("原因／備註")
        decisions.addWidget(self.notes, 1)
        self.decision_button = self.button("記錄判定", self.record_decision, decisions)
        self.pair_button = self.button("修正對應", self.edit_pair, decisions)
        self.button("復原判定／配對", self.undo_review, decisions)
        self.next_review_button = QPushButton("下一待複核 ▾")
        from PySide6.QtWidgets import QMenu
        self.review_menu = QMenu(self.next_review_button)
        self.review_menu.addAction("依時間順序", self.next_unreviewed)
        self.review_menu.addAction("優先處理大差異", self.next_priority)
        self.next_review_button.setMenu(self.review_menu)
        decisions.addWidget(self.next_review_button)
        self.timer_button = self.button("開始複核計時", self.toggle_timer, decisions)
        layout.addLayout(decisions)
        self.wave = wave_class()
        self.wave.duration_ms = project.duration_ms or 1
        self.wave.show_all()
        if project.audio_path and Path(project.audio_path).is_file():
            self.wave.values = waveform(Path(project.audio_path))
        layout.addWidget(self.wave)
        playback = QHBoxLayout()
        self.button("試聽人工軸", lambda: self.play("reference_span"), playback)
        self.button("試聽自動軸", lambda: self.play("generated_span"), playback)
        self.button("暫停", lambda: self.audio.pause(), playback)
        self.button("＋", lambda: self.wave.zoom(2), playback)
        self.button("－", lambda: self.wave.zoom(.5), playback)
        self.button("全覽", self.wave.show_all, playback)
        self.clock = QLabel("藍底：自動顯示範圍　橘虛線：人工顯示頭尾")
        playback.addWidget(self.clock, 1)
        self.button("匯出報告", self.export, playback)
        self.button("開啟比較紀錄", self.open_review, playback)
        layout.addLayout(playback)

        gold = QGroupBox("實際發聲標註（獨立於人工成品的顯示軸；需聽辨確認）")
        gold_layout = QHBoxLayout(gold)
        self.gold_cue = QComboBox()
        self.gold_cue.setMinimumWidth(150)
        self.gold_cue.currentIndexChanged.connect(self.load_annotation)
        gold_layout.addWidget(self.gold_cue)
        self.gold_start, self.gold_end = QSpinBox(), QSpinBox()
        for spin in (self.gold_start, self.gold_end):
            spin.setRange(0, project.duration_ms)
        gold_layout.addWidget(self.gold_start)
        self.button("頭←游標", lambda: self.gold_start.setValue(self.audio.position()), gold_layout)
        gold_layout.addWidget(self.gold_end)
        self.button("尾←游標", lambda: self.gold_end.setValue(self.audio.position()), gold_layout)
        self.original_text = QLineEdit()
        self.original_text.setPlaceholderText("人工確認的原語文字（不可填中文翻譯）")
        gold_layout.addWidget(self.original_text, 1)
        self.overlap, self.elongated, self.uncertain = QCheckBox("重疊"), QCheckBox("長音"), QCheckBox("不確定")
        for widget in (self.overlap, self.elongated, self.uncertain):
            gold_layout.addWidget(widget)
        self.button("確認並儲存標註", self.record_annotation, gold_layout)
        layout.addWidget(gold)
        self.status = QLabel("比較與人工紀錄會存於專案旁的 .review 資料夾；原媒體、字幕及專案打軸不變。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.audio = AudioPreview(self)
        self.player, self.audio_output = self.audio.player, self.audio.output
        self.audio.set_source(project.media_path, project.audio_path, project.audio_stream)
        if parent is not None and hasattr(parent, "audio_output"):
            self.audio_output.setVolume(parent.audio_output.volume())
            self.audio.set_device(bytes(parent.audio_output.device().id()))
        self.audio.positionChanged.connect(self.position_changed)
        self.wave.seek.connect(self.seek)
        self.audio.message.connect(self.status.setText)
        self.tick = QTimer(self)
        self.tick.setInterval(1000)
        self.tick.timeout.connect(self.update_stats)
        self.tick.start()
        self.annotations = annotation_template(project)
        annotation_key = self.annotations_key()
        self.annotation_path = self.store / f"acoustic-{annotation_key}.json"
        self.annotation_invalid = False
        if self.annotation_path.exists():
            try:
                raw = json.loads(self.annotation_path.read_text(encoding="utf-8"))
                paired_report(project, raw)
                expected = {r["cue_id"]: r["recognized_text"] for r in self.annotations["records"]}
                if {r["cue_id"]: r["recognized_text"] for r in raw["records"]} != expected:
                    raise ValueError("標註與專案的字幕不同")
                for record in raw["records"]:
                    if not isinstance(record["original_text"], str) or any(
                        record[k] is not None and type(record[k]) is not int for k in ("speech_start_ms", "speech_end_ms")):
                        raise ValueError("標註欄位格式不符")
                self.annotations = raw
            except (ValueError, OSError, KeyError, TypeError):
                self.annotation_invalid = True
                self.status.setText("舊聲學標註格式不符；新增標註前會保留原檔備份。")
        for widget in (self.reference, self.styles):
            widget.textChanged.connect(self.invalidate)
        self.offset.valueChanged.connect(self.invalidate)
        self.tolerance.valueChanged.connect(self.invalidate)

    def annotations_key(self):
        from .comparison import project_signature
        return project_signature(self.project)[:16]

    @staticmethod
    def button(text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def error(self, exc):
        QMessageBox.warning(self, "比較未完成", str(exc))

    def choose_reference(self):
        filename, _ = QFileDialog.getOpenFileName(self, "選擇人工成品", "", "字幕 (*.ass *.srt)")
        if filename:
            self.reference.setText(filename)

    def invalidate(self):
        if self._loading or self.thread:
            return
        self.stop_timer()
        self.session = None
        self.history.clear()
        self.table.setRowCount(0)
        self.stats.setText("設定已變更，請重新建立比較")

    def start_compare(self):
        if self.thread:
            return
        if not Path(self.reference.text()).is_file():
            self.error("請選擇有效的人工字幕")
            return
        self.stop_timer()
        self.cancel = Event()
        self.thread = QThread(self)
        settings = ComparisonSettings(reference_offset_ms=self.offset.value(), tolerance_ms=self.tolerance.value(),
                                      timeline_confirmed=self.timeline.isChecked())
        styles = tuple(x.strip() for x in self.styles.text().split(",") if x.strip())
        self.worker = ComparisonWorker(self.project, Path(self.reference.text()), settings, self.cancel, styles)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self.receive)
        self.worker.progress.connect(self.status.setText)
        self.worker.failure.connect(self.error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.work_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.set_busy(True)
        self.thread.start()

    def set_busy(self, busy):
        for widget in (self.reference, self.styles, self.offset, self.tolerance, self.timeline,
                       self.compute_button, self.pick_button, self.table, self.decision_button, self.pair_button):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def work_finished(self):
        self.thread = self.worker = None
        self.set_busy(False)
        if self.closing:
            self.close()

    def receive(self, session):
        timeline_confirmed = self.timeline.isChecked()
        key = session_key(session)
        self.session_path = self.store / f"comparison-{key}.json"
        if self.session_path.exists():
            try:
                session = load_session(self.session_path, self.project, Path(self.reference.text()))
            except ValueError as exc:
                self.error(exc)
                return
        self.session = session
        self.history.clear()
        session["settings"]["timeline_confirmed"] = timeline_confirmed
        self._loading = True
        self.timeline.setChecked(session["settings"]["timeline_confirmed"])
        self._loading = False
        self.persist()
        self.refresh()
        self.hint.setText(f"參考影片提示：{session.get('reference_video_hint') or '未提供'}\n正值＝自動軸較晚；時間配對仍需確認，翻譯文字不做辨識正確率比較。")
        self.status.setText(f"已保存比較：{self.session_path}")

    def persist(self):
        if self.session and self.session_path:
            try:
                save_session(self.session, self.session_path)
            except Exception as exc:
                self.error(exc)

    def refresh(self):
        if not self.session:
            return
        self.table.setRowCount(len(self.session["rows"]))
        for i, row in enumerate(self.session["rows"]):
            hint = "重疊／配對不確定" if row["ambiguous"] else "容差內估計" if row["strict_time_estimate"] else "需複核"
            kind = {"unmatched_reference": "人工有／自動未配對", "unmatched_generated": "自動有／人工未配對"}.get(row["kind"], row["kind"])
            values = [kind, row["reference_text"], row["generated_text"], row["start_delta_ms"],
                      row["end_delta_ms"], hint, LABELS[row["decision"]]]
            for j, value in enumerate(values):
                item = QTableWidgetItem("—" if value is None else str(value))
                item.setToolTip(item.text() + ("\n優先複核理由：" + "；".join(review_priority(row)["reasons"]) if j == 5 else ""))
                if row["decision"] == "unreviewed" and not row["strict_time_estimate"]:
                    item.setBackground(QColor("#fff0cf"))
                self.table.setItem(i, j, item)
        self.update_stats()
        if self.table.rowCount() and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self.select_row()

    def update_stats(self):
        if not self.session:
            return
        result = summary(self.session)
        percent = lambda value: "尚無" if value is None else f"{value:.1%}"
        elapsed = self.session.get("review_active_seconds", 0) + (time.perf_counter() - self.timer_started if self.timer_started else 0)
        self.stats.setText(f"人工 {result['reference_events']} 句／自動 {result['generated_events']} 句 · "
            f"一對一且頭尾均在容差內的估計 {percent(result['strict_time_estimate_rate'])} · "
            f"人工確認可用 {percent(result['human_confirmed_usable_rate'])}（全體人工句為分母） · "
            f"已複核 {result['human_reviewed_reference_events']} 句 · 主動複核計時 {elapsed:.0f} 秒")

    def select_row(self):
        index = self.table.currentRow()
        if not self.session or not 0 <= index < len(self.session["rows"]):
            return
        row = self.session["rows"][index]
        self.audio.stop()
        self.texts.setText("人工：" + row["reference_text"] + "\n自動：" + row["generated_text"])
        self.notes.setText(row["notes"])
        if row["reviewer"]:
            self.reviewer.setText(row["reviewer"])
        if row["decision"] in LABELS:
            self.decision.setCurrentIndex(max(0, self.decision.findData(row["decision"])))
        self.wave.region = row["generated_span"] or row["reference_span"]
        self.wave.candidate_region = row["reference_span"]
        self.wave.speech_region = None
        self.wave.focus_region()
        self.gold_cue.blockSignals(True)
        self.gold_cue.clear()
        by_id = {c.id: c for c in self.project.cues}
        for cue_id in row["generated_ids"]:
            self.gold_cue.addItem(by_id[cue_id].text, cue_id)
        self.gold_cue.blockSignals(False)
        self.load_annotation()

    def confirm_timeline(self, checked):
        if self.session and not self._loading:
            self.session["settings"]["timeline_confirmed"] = checked
            self.persist()
            self.update_stats()

    def record_decision(self):
        row = self.table.currentRow()
        if self.thread or not self.session or row < 0:
            return
        try:
            previous = copy.deepcopy(self.session)
            set_decision(self.session, row, self.decision.currentData(), self.reviewer.text(), self.notes.text())
            self.history.append(previous)
            self.history = self.history[-30:]
            self.persist()
            self.refresh()
            self.table.selectRow(row)
        except Exception as exc:
            self.error(exc)

    def edit_pair(self):
        row = self.table.currentRow()
        if self.thread or not self.session or row < 0:
            return
        dialog = PairDialog(self.session, self.session["rows"][row], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                previous = copy.deepcopy(self.session)
                repair_pair(self.session, *dialog.selections())
                self.history.append(previous)
                self.history = self.history[-30:]
                self.persist()
                self.refresh()
            except Exception as exc:
                self.error(exc)

    def undo_review(self):
        if self.thread or not self.history:
            return
        elapsed = self.session.get("review_active_seconds", 0)
        self.session = self.history.pop()
        self.session["review_active_seconds"] = elapsed
        self.persist()
        self.refresh()

    def next_unreviewed(self):
        if not self.session:
            return
        start = self.table.currentRow() + 1
        for i in list(range(start, len(self.session["rows"]))) + list(range(0, start)):
            if self.session["rows"][i]["decision"] == "unreviewed":
                self.table.selectRow(i)
                return

    def next_priority(self):
        if not self.session:
            return
        order = review_order(self.session)
        current = self.table.currentRow()
        target = next((i for i in order if i != current), order[0] if order else None)
        if target is not None:
            self.table.selectRow(target)
            self.table.scrollToItem(self.table.item(target, 0))
            self.status.setText("優先複核：" + "；".join(review_priority(self.session["rows"][target])["reasons"]))
        else:
            self.status.setText("沒有尚未複核的對應")

    def toggle_timer(self):
        if not self.session:
            return
        if self.timer_started is not None:
            self.stop_timer()
        else:
            self.timer_started = time.perf_counter()
            self.timer_button.setText("暫停複核計時")

    def stop_timer(self):
        if self.timer_started is not None and self.session:
            self.session["review_active_seconds"] += time.perf_counter() - self.timer_started
            self.timer_started = None
            self.timer_button.setText("開始複核計時")
            self.persist()

    def play(self, field):
        index = self.table.currentRow()
        if not self.session or index < 0:
            return
        span = self.session["rows"][index][field]
        if span:
            self.play_end = min(self.project.duration_ms, span[1] + 250)
            self.audio.audition(max(0, span[0] - 250), self.play_end)

    def seek(self, position):
        self.play_end = 0
        self.audio.seek(position)

    def position_changed(self, position):
        self.wave.position_ms = position
        self.wave.update()
        self.clock.setText(f"{position}ms · 藍底：自動軸；橘虛線：人工軸")

    def load_annotation(self):
        self.wave.speech_region = None
        self.wave.update()
        cue_id = self.gold_cue.currentData()
        record = next((r for r in self.annotations["records"] if r["cue_id"] == cue_id), None)
        self.original_text.clear()
        self.gold_start.setValue(0)
        self.gold_end.setValue(0)
        for box in (self.overlap, self.elongated, self.uncertain):
            box.setChecked(False)
        if record:
            self.original_text.setText(record["original_text"])
            self.gold_start.setValue(record["speech_start_ms"] or 0)
            self.gold_end.setValue(record["speech_end_ms"] or 0)
            for box, key in [(self.overlap, "overlap"), (self.elongated, "elongated"), (self.uncertain, "uncertain")]:
                box.setChecked(record[key])
            if record.get("confirmed"):
                self.wave.speech_region = [record["speech_start_ms"], record["speech_end_ms"]]
                self.wave.update()

    def record_annotation(self):
        if self.thread or not self.session:
            return
        try:
            if not self.timeline.isChecked() or not self.reviewer.text().strip():
                raise ValueError("請確認時間基準並填複核者")
            if not self.original_text.text().strip() or not 0 <= self.gold_start.value() < self.gold_end.value() <= self.project.duration_ms:
                raise ValueError("請聽辨後填寫原語文字與合法發聲頭尾；不會自動採用字幕顯示時間")
            record = next((r for r in self.annotations["records"] if r["cue_id"] == self.gold_cue.currentData()), None)
            if record is None:
                raise ValueError("請先選取一則自動字幕")
            record.update(original_text=self.original_text.text().strip(), speech_start_ms=self.gold_start.value(),
                          speech_end_ms=self.gold_end.value(), reviewer=self.reviewer.text().strip(), confirmed=True,
                          uncertain=self.uncertain.isChecked(), overlap=self.overlap.isChecked(), elongated=self.elongated.isChecked())
            self.store.mkdir(parents=True, exist_ok=True)
            if self.annotation_invalid and self.annotation_path.exists():
                backup = self.annotation_path.with_suffix(".invalid.bak")
                if not backup.exists():
                    shutil.copy2(self.annotation_path, backup)
                self.annotation_invalid = False
            self.annotations["offset_ms"] = self.offset.value()
            temporary = self.annotation_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.annotations, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.annotation_path)
            self.load_annotation()
            self.status.setText(f"已保存人工聲學標註：{self.annotation_path}")
        except Exception as exc:
            self.error(exc)

    def export(self):
        if not self.session:
            return
        self.stop_timer()
        filename, selected = QFileDialog.getSaveFileName(self, "匯出比較報告", str(self.store / "report.html"), "HTML (*.html);;CSV (*.csv)")
        if filename:
            try:
                export_report(self.session, Path(filename).with_suffix(".csv" if selected.startswith("CSV") else ".html"))
                self.status.setText("已匯出報告；原始字幕與辨識結果未變更")
            except Exception as exc:
                self.error(exc)

    def open_review(self):
        if self.thread:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "開啟比較紀錄", str(self.store), "JSON (*.json)")
        if not filename:
            return
        try:
            session = load_session(Path(filename), self.project)
            self.stop_timer()
            self._loading = True
            self.reference.setText(session.get("reference_fingerprint", {}).get("path", ""))
            self.offset.setValue(session["settings"]["reference_offset_ms"])
            self.tolerance.setValue(session["settings"]["tolerance_ms"])
            self.styles.setText(",".join(session.get("excluded_styles", [])))
            self.timeline.setChecked(session["settings"]["timeline_confirmed"])
            self._loading = False
            self.session, self.session_path = session, Path(filename)
            self.history.clear()
            self.refresh()
        except Exception as exc:
            self._loading = False
            self.error(exc)

    def closeEvent(self, event):
        if self.thread:
            self.cancel.set()
            self.closing = True
            event.ignore()
            return
        self.stop_timer()
        self.persist()
        self.audio.stop()
        event.accept()

    def reject(self):
        # Escape uses QDialog.reject rather than closeEvent by default. Route it
        # through cancellation so a live worker cannot outlast its dialog.
        self.close()
