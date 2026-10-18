"""Shared desktop colors and control hierarchy."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPolygon
from PySide6.QtWidgets import QComboBox


class ChoiceBox(QComboBox):
    """Keep the popup affordance legible with native and offscreen Qt styles."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#a9bed7" if self.isEnabled() else "#526076"))
        x, y = self.width() - 12, self.height() // 2
        painter.drawPolygon(QPolygon([QPoint(x - 4, y - 2), QPoint(x + 4, y - 2), QPoint(x, y + 3)]))


def prepare_desktop_theme(app):
    if app.property("autosubTheme"):
        return
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#111722", QPalette.ColorRole.WindowText: "#dbe3ef",
        QPalette.ColorRole.Base: "#101925", QPalette.ColorRole.AlternateBase: "#192434",
        QPalette.ColorRole.Text: "#dbe3ef", QPalette.ColorRole.Button: "#223044",
        QPalette.ColorRole.ButtonText: "#dbe3ef", QPalette.ColorRole.Highlight: "#276b9d",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#26354b",
        QPalette.ColorRole.ToolTipText: "#eff5fc",
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setProperty("autosubTheme", True)

DESKTOP_STYLE = """
QMainWindow, QDialog, QWidget#workspace, QWidget#sidebar { background: #111722; }
QWidget { color: #dbe3ef; font-size: 13px; }
QLabel { background: transparent; }
QLabel#appTitle { font-size: 24px; font-weight: 700; color: #f1f5fb; }
QLabel#muted, QLabel#sourceLabel { color: #92a2b8; }
QLabel#sourceLabel { padding: 10px 12px; background: #192230; border-radius: 6px; }
QGroupBox { background: #17202e; border: 1px solid #2a374a; border-radius: 8px;
    margin-top: 16px; padding: 16px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #b7c7dd; }
QPushButton, QToolButton { background: #223044; border: 1px solid #34455e;
    border-radius: 6px; padding: 7px 12px; color: #dbe3ef; }
QPushButton:hover, QToolButton:hover { background: #2d405a; border-color: #6481a5; }
QPushButton:pressed, QToolButton:pressed { background: #172338; }
QPushButton:disabled, QToolButton:disabled { color: #67768b; background: #1a2331; border-color: #283446; }
QPushButton#primary { background: #359cc4; border-color: #359cc4; color: #071c2b; font-weight: 700; }
QPushButton#primary:hover { background: #61bbdb; }
QPushButton#primary:disabled { background: #253b4b; border-color: #253b4b; color: #7b929e; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #101925; border: 1px solid #34455b; border-radius: 5px; padding: 5px 7px;
    selection-background-color: #276b9d; selection-color: white; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #57bbdf; }
QComboBox { min-height: 22px; padding-right: 22px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView, QMenu { background: #1b293b; color: #e3ecf8; border: 1px solid #425571; selection-background-color: #276b9d; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #72839a; border-radius: 3px; background: #101925; }
QCheckBox::indicator:checked { background: #359cc4; border: 2px solid #7ed5ef; }
QTableWidget { background: #151e2b; alternate-background-color: #192434; border: 1px solid #2b394e; border-radius: 6px; gridline-color: #28354a; }
QTableWidget::item { padding: 7px; border-bottom: 1px solid #253246; }
QHeaderView::section { background: #202d40; color: #adbed5; border: none; border-right: 1px solid #2b3b51; padding: 9px 7px; }
QTableCornerButton::section { background: #202d40; border: none; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #131d2a; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #40526b; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSplitter::handle { background: #111722; width: 10px; }
QProgressBar { background: #1c293b; border: none; border-radius: 3px; color: #a9bfd5; text-align: center; min-height: 7px; }
QProgressBar::chunk { background: #359cc4; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #33465f; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; background: #65c5e5; border-radius: 6px; }
QToolTip { background: #26354b; color: #eff5fc; border: 1px solid #526b8d; padding: 5px; }
"""
