"""Render the finished-reference reviewer with a real cached project."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from autosub.app import MainWindow, Waveform
from autosub.comparison import ComparisonSettings, compare_file
from autosub.domain import Project
from autosub.review_ui import ComparisonDialog

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--offset-ms", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = QApplication([])
parent = MainWindow()
dialog = ComparisonDialog(Project.load(args.project), args.output / "preview.autosub.json", Waveform, parent)
dialog.reference.setText(str(args.reference))
dialog.offset.setValue(args.offset_ms)
session = compare_file(dialog.project, args.reference, ComparisonSettings(reference_offset_ms=args.offset_ms))
dialog.receive(session)
dialog.show()
app.processEvents()
assert dialog.grab().save(str(args.output / "comparison-ui.png"))
dialog.close()
parent.close()
print("Comparison UI rendered")
