from enum import Enum
import wave

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtMultimedia import QAudioFormat
from PySide6.QtWidgets import QPushButton

from autosub.app import MainWindow
from autosub.domain import Cue, Project
from autosub.export import export_subtitles
from autosub.playback import AudioPreview, pcm_for_output


def signal_wav(path):
    samples = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())


@pytest.mark.parametrize("fmt,dtype", [(QAudioFormat.SampleFormat.Float, "<f4"),
    (QAudioFormat.SampleFormat.Int16, "<i2"), (QAudioFormat.SampleFormat.Int32, "<i4"),
    (QAudioFormat.SampleFormat.UInt8, "u1")])
def test_pcm_range_resamples_to_device_format_without_silencing(tmp_path, fmt, dtype):
    path = tmp_path / "聲音.wav"
    signal_wav(path)
    data, peak = pcm_for_output(path, 250, 750, 48000, 2, fmt)
    samples = np.frombuffer(data, dtype=dtype).reshape(-1, 2)
    assert len(samples) == 24000  # half a second on a 48kHz device
    assert .2 < peak < .3
    assert samples[:, 0].max() > samples[:, 0].min()
    assert np.array_equal(samples[:, 0], samples[:, 1])


def test_pcm_sink_uses_selected_range_volume_and_handles_distinct_qt_enums(qtbot, tmp_path, monkeypatch):
    import autosub.playback as playback
    class Error(Enum):
        NoError = 0
    class State(Enum):
        Stopped = 2
        Idle = 3
    class Sink(QObject):
        stateChanged = Signal(object)
        instances = []
        def __init__(self, device, fmt, parent):
            super().__init__(parent)
            self.instances.append(self)
        def setVolume(self, value): self.volume = value
        def start(self, buffer): self.data = bytes(buffer.data())
        def error(self): return Error.NoError
        def stop(self): pass
        def suspend(self): pass
        def resume(self): self.resumed = True
        def processedUSecs(self): return 200000
    fmt = QAudioFormat()
    fmt.setSampleRate(48000)
    fmt.setChannelCount(2)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
    class Device:
        def isNull(self): return False
        def description(self): return "test device"
        def preferredFormat(self): return fmt
    monkeypatch.setattr(playback, "QAudioSink", Sink)
    audio = AudioPreview()
    monkeypatch.setattr(audio, "_device", lambda: Device())
    monkeypatch.setattr(audio.output, "setDevice", lambda _: None)
    path = tmp_path / "clip.wav"
    signal_wav(path)
    audio.set_source(path, path)
    audio.output.setVolume(.6)
    assert audio.audition(100, 600)
    assert len(audio.sink.data) == 24000 * 2 * 4
    assert audio.sink.volume == pytest.approx(.6)
    audio._tick()
    assert audio.position() == 300
    original_sink = audio.sink
    audio.pause()
    assert not audio.is_playing() and audio.position() == 300
    assert audio.resume() and audio.sink is original_sink and audio.sink.resumed
    audio.sink.stateChanged.emit(State.Idle)
    assert audio.position() == 600 and not audio.is_playing()
    assert not audio.resume()  # a completed range must restart, never resume its empty buffer
    old = audio.sink
    audio.audition(100, 600)
    old.stateChanged.emit(State.Idle)
    assert audio.is_playing()  # late events from an earlier audition cannot stop this one
    audio.audition(0, 500, lambda: True)
    audio.sink.stateChanged.emit(State.Idle)
    count = len(Sink.instances)
    audio.stop()  # cancel queued loop restart
    qtbot.wait(10)
    assert len(Sink.instances) == count
    audio.audition(100, 600)
    audio.pause()
    audio.seek(450)
    assert not audio.resume() and audio.sink is None
    audio.audition(100, 600)
    audio.pause()
    audio.set_device(None)
    assert not audio.resume()  # suspended sink belongs to the old device


def test_pending_media_timeout_stop_and_error_allow_retry(qtbot, tmp_path, monkeypatch):
    from PySide6.QtMultimedia import QMediaPlayer
    audio = AudioPreview()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    audio.source_path = source
    class Device:
        def isNull(self): return False
    monkeypatch.setattr(audio, "_device", lambda: Device())
    monkeypatch.setattr(audio.output, "setDevice", lambda _: None)
    monkeypatch.setattr(audio.player, "mediaStatus", lambda: QMediaPlayer.MediaStatus.LoadingMedia)
    messages = []
    audio.message.connect(messages.append)
    assert audio.audition(0, 1000)
    assert audio._pending and audio.load_timer.isActive()
    audio._load_timeout()
    assert not audio._pending and not audio.load_timer.isActive()
    assert "逾時" in messages[-1]
    assert audio.audition(0, 1000)
    audio._media_error(None, "decode failed")
    assert not audio._pending and "decode failed" in messages[-1]
    audio._media_status(QMediaPlayer.MediaStatus.LoadedMedia)
    assert not audio.is_playing()  # late loaded event after failure cannot start old audition


def test_keyboard_navigation_skips_filtered_rows_and_invalidates_old_range(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(duration_ms=6000, cues=[
        Cue("a", 0, 1000, "A", 0, 900, ["check"]),
        Cue("b", 2000, 3000, "B", 2000, 2900),
        Cue("c", 4000, 5000, "C", 4000, 4900, ["check"])])
    window.refresh()
    window.only_flags.setChecked(True)
    window.audio._resumable = True
    window.move_selection(1)
    assert window.table.currentRow() == 2
    assert not window.audio._resumable and window.audio.position() == 4000
    window.move_selection(-1)
    assert window.table.currentRow() == 0


def test_ass_no_automatic_breaks_by_default_but_can_opt_in(tmp_path):
    long_text = "これは長い字幕のテストです。" * 4
    p = Project(duration_ms=6000, cues=[Cue("a", 0, 5000, long_text, 0, 4800)])
    path = tmp_path / "default.ass"
    export_subtitles(p, path)
    assert r"\N" not in path.read_text(encoding="utf-8-sig")
    export_subtitles(p, tmp_path / "wrapped.ass", wrap_lines=True)
    assert r"\N" in (tmp_path / "wrapped.ass").read_text(encoding="utf-8-sig")
    p.cues[0].text = "手動第一行\n手動第二行"
    export_subtitles(p, path)
    assert r"手動第一行\N手動第二行" in path.read_text(encoding="utf-8-sig")


def test_selection_is_visible_after_play_focus_and_moves_cursor(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(duration_ms=6000, cues=[
        Cue("a", 1000, 2000, "第一句", 1000, 1800), Cue("b", 4000, 5000, "第二句", 4000, 4800)])
    window.refresh()
    window.show()
    qtbot.mouseClick(window.table.viewport(), Qt.MouseButton.LeftButton,
                     pos=window.table.visualItemRect(window.table.item(1, 2)).center())
    assert window.table.currentRow() == 1
    assert "第 2／2 句" in window.selection_label.text()
    assert window.audio.position() == 4000
    assert window.wave.region == (4000, 5000)
    window.play_button.setFocus()
    assert window.table.selectionModel().selectedRows()[0].row() == 1
    assert ":!active" in window.table.styleSheet()
    window.only_flags.setChecked(True)
    assert not window.play_button.isEnabled()


def test_basic_mode_hides_advanced_tools_and_export_wrap_is_off(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert not window.advanced.isChecked() and not window.wrap_export.isChecked()
    assert window.candidate_panel.isHidden() and window.timing_group.isHidden()
    assert window.table.isColumnHidden(4)
    basic_count = sum(b.isVisible() for b in window.findChildren(QPushButton))
    window.advanced.setChecked(True)
    assert window.candidate_panel.isVisible() and window.timing_group.isVisible()
    assert not window.table.isColumnHidden(4)
    assert sum(b.isVisible() for b in window.findChildren(QPushButton)) > basic_count
