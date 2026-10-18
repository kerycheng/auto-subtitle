import wave

import numpy as np
import pytest
from PySide6.QtGui import QPalette
from PySide6.QtMultimedia import QAudioFormat

from autosub.app import MainWindow
from autosub.domain import Cue, Project
from autosub.playback import pcm_for_output


def test_light_public_view_keeps_generation_hints_hidden(qtbot, monkeypatch):
    monkeypatch.delenv("AUTOSUB_ENABLE_REPAIR_UI", raising=False)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.palette().color(QPalette.ColorRole.Window).lightness() > 200
    for advanced in (False, True, False):
        window.advanced.setChecked(advanced)
        assert window.only_flags.isHidden()
        assert window.table.isColumnHidden(3)


@pytest.mark.parametrize("start,end", [(0, 499), (1000, 1800), (2500, 3000)])
def test_selected_cue_passes_exact_display_bounds(qtbot, monkeypatch, start, end):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(duration_ms=3000,
                             cues=[Cue("synthetic", start, end, "Sample text", start, end)])
    window.refresh()
    window.table.selectRow(0)
    calls = []
    monkeypatch.setattr(window.audio, "audition", lambda a, b, loop: calls.append((a, b, loop)))
    window.play_cue()
    assert (window.play_start, window.play_end) == (start, end)
    assert calls[0][:2] == (start, end)
    window.loop.setChecked(True)
    assert calls[0][2]()  # Repetition retains the same selected interval.
    window.play_cue()
    assert calls[1][:2] == (start, end)


@pytest.mark.parametrize("rate,channels", [(16000, 1), (48000, 2)])
def test_pcm_excludes_distinct_neighboring_sound(tmp_path, rate, channels):
    path = tmp_path / "synthetic.wav"
    # Different neighboring levels make any leaked pre/post samples observable.
    raw = np.concatenate([np.full(8000, 24576), np.full(16000, 4096),
                          np.full(8000, 16384)]).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(raw.tobytes())
    data, peak = pcm_for_output(path, 500, 1500, rate, channels, QAudioFormat.SampleFormat.Int16)
    actual = np.frombuffer(data, dtype="<i2")
    assert len(actual) == rate * channels
    assert np.all(actual == 4096)
    assert peak == .125
