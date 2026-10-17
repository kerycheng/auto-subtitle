"""Offscreen screenshot and desktop construction check, no interactive window."""
import os
import argparse
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from autosub.app import MainWindow
from autosub.domain import Cue, Project

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
target = root / "test-artifacts"
target.mkdir(exist_ok=True)
app = QApplication([])
window = MainWindow()
window.project = Project(duration_ms=20000, media_path="VTuber_clip.mp4", cues=[
    Cue("1", 980, 3500, "ちょっと待って、今のドライブインパクト見た？", 1000, 3300),
    Cue("2", 3500, 5900, "今のはパリィできなかった。", 3520, 5700, ["辨識信心偏低"]),
    Cue("3", 6200, 9000, "もう一回やってみよう！", 6220, 8800),
])
if args.project:
    window.project = Project.load(args.project)
    window.load_controls()
    window.refresh()
    window.load_playback()
else:
    window.source_label.setText("VTuber_clip.mp4　｜　UI 展示資料，非辨識結果")
    window.refresh()
    window.wave.duration_ms = 20000
    window.wave.values = [0.15 + (i % 17) / 24 for i in range(400)]
window.table.selectRow(0)
window.show()
app.processEvents()
destination = args.output or target / "ui-preview.png"
destination.parent.mkdir(parents=True, exist_ok=True)
assert window.grab().save(str(destination))
window.close()
print("UI construction and screenshot passed")
