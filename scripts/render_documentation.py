"""Render public UI illustrations with synthetic data and isolated settings."""
import argparse
import math
import os
from pathlib import Path
import tempfile


def render(widget, target, preview_dir=None):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgGenerator, QSvgRenderer

    generator = QSvgGenerator()
    generator.setFileName(str(target))
    generator.setSize(widget.size())
    generator.setViewBox(widget.rect())
    generator.setTitle("Auto Subtitle · synthetic UI example")
    generator.setDescription("Illustrative text, timings and waveform; no recognition or private media.")
    painter = QPainter(generator)
    widget.render(painter, QPoint())
    painter.end()
    # Qt emits trailing spaces in attribute lists; keep generated assets diff-clean.
    target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                      + "\n", encoding="utf-8")
    renderer = QSvgRenderer(str(target))
    if not renderer.isValid():
        raise RuntimeError("Invalid SVG illustration")
    if preview_dir:
        image = QImage(widget.size(), QImage.Format.Format_ARGB32)
        image.fill(0xffffffff)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        if not image.save(str(preview_dir / (target.stem + ".png"))):
            raise RuntimeError("Could not save illustration preview")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.pop("AUTOSUB_ENABLE_REPAIR_UI", None)
    os.environ.pop("AUTOSUB_ENABLE_PROJECT_TOOLS", None)
    with tempfile.TemporaryDirectory(prefix="autosub-docs-") as isolated:
        os.environ["AUTOSUB_SETTINGS_PATH"] = str(Path(isolated) / "preferences.json")
        from PySide6.QtWidgets import QApplication
        from autosub.app import MainWindow
        from autosub.domain import Cue, Project, Word
        from autosub.vocabulary import save_library
        from autosub.vocabulary_ui import VocabularySelectionDialog

        library = {"schema_version": 1, "lists": [
            {"id": "people", "name": "人物名稱（示例）", "text": "空野アオ\n星川ハル"},
            {"id": "callouts", "name": "遊戲報點（示例）", "text": "合流\n回復\n高台"},
            {"id": "items", "name": "道具名稱（示例）", "text": "シールド\n回復"},
        ]}
        save_library(library)
        app = QApplication([])
        window = MainWindow()
        window.resize(1320, 880)
        sentences = [
            (1000, 4800, "今日はこのゲームで遊んでいきます。"),
            (6100, 10300, "右の建物を確認してから進もう。"),
            (12100, 15900, "回復したら、みんなで合流しよう。"),
            (18700, 23000, "次は少し高い場所に移動します。"),
        ]
        window.project = Project(duration_ms=32000, language="ja", model="large-v3", status="complete",
            words=[Word(start, end, text) for start, end, text in sentences],
            cues=[Cue(f"example-{i}", start - 20, end + 200, text, start, end)
                  for i, (start, end, text) in enumerate(sentences)])
        window.load_controls()
        window.apply_vocabulary_choice({"ids": ["people", "callouts"], "base_text": ""})
        window.refresh()
        window.source_label.setText("示例影片.mp4 · 合成操作展示")
        window.status.setText("操作示例：畫面文字、時間與波形均為合成資料。")
        window.wave.duration_ms = 32000
        window.wave.values = [abs(math.sin(i * .13)) * (.35 + .35 * math.sin(i * .017) ** 2)
                              if i % 270 < 190 else .02 for i in range(3200)]
        window.wave.show_all()
        window.table.selectRow(1)
        window.show()
        app.processEvents()
        render(window, args.output / "main-window.svg", args.preview_dir)
        dialog = VocabularySelectionDialog(window, window.vocabulary_library, window.vocabulary_choice)
        dialog.show()
        app.processEvents()
        render(dialog, args.output / "vocabulary-selection.svg", args.preview_dir)
        dialog.reject()
        window.advanced.setChecked(True)
        app.processEvents()
        render(window.timing_group, args.output / "timing-settings.svg", args.preview_dir)
        window.autosave.stop()
        window.dirty = False
        window.project_path = None
        window.close()


if __name__ == "__main__":
    main()
