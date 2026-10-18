"""Shared desktop colors and control hierarchy."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPolygon
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDoubleSpinBox, QSpinBox, QStyle, QStyleOptionSpinBox,
)


class ChoiceBox(QComboBox):
    """Keep the popup affordance legible with native and offscreen Qt styles."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#52657e" if self.isEnabled() else "#a6b0bd"))
        x, y = self.width() - 12, self.height() // 2
        painter.drawPolygon(QPolygon([QPoint(x - 4, y - 2), QPoint(x + 4, y - 2), QPoint(x, y + 3)]))


class _SpinArrows:
    """Paint arrows inside Qt's actual hit rectangles, including high-DPI layouts."""

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        if option.buttonSymbols != QAbstractSpinBox.ButtonSymbols.UpDownArrows:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for control, flag, direction in (
            (QStyle.SubControl.SC_SpinBoxUp, QAbstractSpinBox.StepEnabledFlag.StepUpEnabled, -1),
            (QStyle.SubControl.SC_SpinBoxDown, QAbstractSpinBox.StepEnabledFlag.StepDownEnabled, 1),
        ):
            rect = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, self)
            if rect.isEmpty():
                continue
            enabled = self.isEnabled() and bool(option.stepEnabled & flag)
            painter.setBrush(QColor("#52657e" if enabled else "#a6b0bd"))
            center = rect.center()
            x, y = center.x(), center.y()
            painter.drawPolygon(QPolygon([
                QPoint(x - 4, y - 2 * direction), QPoint(x + 4, y - 2 * direction),
                QPoint(x, y + 3 * direction),
            ]))


class NumberBox(_SpinArrows, QSpinBox):
    pass


class DecimalBox(_SpinArrows, QDoubleSpinBox):
    pass


def prepare_desktop_theme(app):
    if app.property("autosubTheme"):
        return
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#f4f6fa", QPalette.ColorRole.WindowText: "#253247",
        QPalette.ColorRole.Base: "#ffffff", QPalette.ColorRole.AlternateBase: "#f8fafc",
        QPalette.ColorRole.Text: "#253247", QPalette.ColorRole.Button: "#ffffff",
        QPalette.ColorRole.ButtonText: "#253247", QPalette.ColorRole.Highlight: "#2563eb",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#eef3fc",
        QPalette.ColorRole.ToolTipText: "#24364b",
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setProperty("autosubTheme", True)

DESKTOP_STYLE = """
QMainWindow, QDialog, QWidget#workspace, QWidget#sidebar { background: #f4f6fa; }
QWidget { color: #253247; font-size: 13px; }
QLabel { background: transparent; }
QLabel#appTitle { font-size: 24px; font-weight: 700; color: #172b46; }
QLabel#muted, QLabel#sourceLabel { color: #61738a; }
QLabel#sourceLabel { padding: 10px 12px; background: #e9eef5; border-radius: 6px; }
QGroupBox { background: #ffffff; border: 1px solid #d7e0eb; border-radius: 8px;
    margin-top: 16px; padding: 16px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #52677e; }
QPushButton, QToolButton { background: #ffffff; border: 1px solid #cad4e0;
    border-radius: 6px; padding: 7px 12px; color: #253247; }
QPushButton:hover, QToolButton:hover { background: #edf3ff; border-color: #7d9fd0; }
QPushButton:pressed, QToolButton:pressed { background: #e1eafa; }
QPushButton:disabled, QToolButton:disabled { color: #909dab; background: #f1f4f8; border-color: #dfe5ed; }
QPushButton#primary { background: #2563eb; border-color: #2563eb; color: #ffffff; font-weight: 700; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #d4e1f8; border-color: #d4e1f8; color: #788dab; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #ffffff; border: 1px solid #cad4e0; border-radius: 5px; padding: 5px 7px;
    selection-background-color: #2563eb; selection-color: white; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #3b82f6; }
QSpinBox, QDoubleSpinBox { padding-right: 30px; min-height: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 24px;
    background: #edf2f8; border-left: 1px solid #cad4e0; border-top-right-radius: 5px; }
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 24px;
    background: #edf2f8; border-left: 1px solid #cad4e0; border-top: 1px solid #d7e0eb;
    border-bottom-right-radius: 5px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #dfeaff; }
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background: #cbdcf7; }
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { image: none; }
QComboBox { min-height: 22px; padding-right: 22px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView, QMenu { background: #ffffff; color: #253247; border: 1px solid #cad4e0; selection-background-color: #2563eb; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #8d9db0; border-radius: 3px; background: #ffffff; }
QCheckBox::indicator:checked { background: #2563eb; border: 2px solid #2563eb; }
QTableWidget { background: #ffffff; alternate-background-color: #f8fafc; border: 1px solid #d7e0eb; border-radius: 6px; gridline-color: #e2e8f0; }
QTableWidget::item { padding: 7px; border-bottom: 1px solid #e8edf4; }
QHeaderView::section { background: #eef2f7; color: #52657b; border: none; border-right: 1px solid #dde4ee; padding: 9px 7px; }
QTableCornerButton::section { background: #eef2f7; border: none; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #edf1f6; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #b7c5d6; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSplitter::handle { background: #f4f6fa; width: 10px; }
QProgressBar { background: #e2e9f3; border: none; border-radius: 3px; color: #52657b; text-align: center; min-height: 7px; }
QProgressBar::chunk { background: #2563eb; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #d3dfed; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; background: #2563eb; border-radius: 6px; }
QToolTip { background: #eef3fc; color: #24364b; border: 1px solid #b8c9df; padding: 5px; }
"""
