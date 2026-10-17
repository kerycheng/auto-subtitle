"""Audition normalized PCM directly, with an explicit output device and cursor.

QMediaPlayer is retained for media preview when analysis WAV is unavailable.
The normal subtitle audition path does not depend on multimedia seek readiness.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import (QAudio, QAudioFormat, QAudioOutput, QAudioSink,
                                 QMediaDevices, QMediaPlayer)


def pcm_for_output(path, start_ms, end_ms, rate, channels, sample_format):
    """Read only the selected range; resample to the output device's preferred format."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError("試聽分析音訊必須是 16kHz 單聲道 PCM16")
        first = max(0, min(stream.getnframes(), start_ms * 16))
        last = max(first, min(stream.getnframes(), end_ms * 16))
        stream.setpos(first)
        audio = np.frombuffer(stream.readframes(last - first), dtype="<i2").astype(np.float32) / 32768
    if not len(audio) or rate <= 0 or channels <= 0:
        raise ValueError("選取範圍沒有可播放的音訊")
    peak = float(np.max(np.abs(audio)))
    if rate != 16000:
        count = max(1, round(len(audio) * rate / 16000))
        audio = np.interp(np.arange(count) * 16000 / rate, np.arange(len(audio)), audio).astype(np.float32)
    if channels > 1:
        audio = np.repeat(audio[:, None], channels, axis=1).reshape(-1)
    if sample_format == QAudioFormat.SampleFormat.Float:
        data = audio.astype("<f4")
    elif sample_format == QAudioFormat.SampleFormat.Int16:
        data = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
    elif sample_format == QAudioFormat.SampleFormat.Int32:
        data = np.clip(np.rint(audio.astype(np.float64) * 2147483648), -2147483648, 2147483647).astype("<i4")
    elif sample_format == QAudioFormat.SampleFormat.UInt8:
        data = np.clip(np.rint((audio + 1) * 128), 0, 255).astype("u1")
    else:
        raise ValueError("輸出裝置的音訊格式不支援，請換一個輸出裝置")
    return data.tobytes(), peak


class AudioPreview(QObject):
    positionChanged = Signal(int)
    message = Signal(str)
    playingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.devices = QMediaDevices(self)
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(QMediaDevices.defaultAudioOutput(), self)
        self.output.setVolume(.8)
        self.player.setAudioOutput(self.output)
        self.output.volumeChanged.connect(self._volume)
        self.output.mutedChanged.connect(lambda _: self._volume(self.output.volume()))
        self.devices.audioOutputsChanged.connect(self._devices_changed)
        self.player.errorOccurred.connect(self._media_error)
        self.player.positionChanged.connect(self._media_position)
        self.player.mediaStatusChanged.connect(self._media_status)
        self.wav_path = None
        self.source_path = None
        self.sink = self.buffer = None
        self.cursor = 0
        self.start_ms = self.end_ms = 0
        self.loop = lambda: False
        self._playing = self._seeking = self._pending = False
        self._resumable = False
        self._generation = 0
        self._device_id = None
        self.track = 0
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(15000)
        self.load_timer.timeout.connect(self._load_timeout)

    def set_source(self, source, wav=None, track=0):
        self.stop()
        self.source_path = Path(source).resolve() if source else None
        self.wav_path = Path(wav).resolve() if wav and Path(wav).is_file() else None
        self.track = track
        self.cursor = 0
        self.player.setSource(QUrl() if self.wav_path else
            QUrl.fromLocalFile(str(self.source_path)) if self.source_path and self.source_path.is_file() else QUrl())

    def _device(self):
        if self._device_id is not None:
            for device in QMediaDevices.audioOutputs():
                if bytes(device.id()) == self._device_id:
                    return device
        return QMediaDevices.defaultAudioOutput()

    def set_device(self, device_id=None):
        self.stop()
        self._device_id = device_id
        self.output.setDevice(self._device())
        self.message.emit("試聽輸出：" + self.output.device().description())

    def _devices_changed(self):
        self.set_device(self._device_id)

    def _volume(self, value):
        if self.sink is not None:
            self.sink.setVolume(0 if self.output.isMuted() else value)

    def audition(self, start_ms, end_ms, loop=lambda: False):
        self.stop()
        self.start_ms, self.end_ms = max(0, int(start_ms)), int(end_ms)
        self.loop = loop
        if self.end_ms <= self.start_ms:
            self.message.emit("試聽範圍無效")
            return False
        device = self._device()
        if device.isNull():
            self.message.emit("找不到音訊輸出裝置；請在進階工具的試聽輸出中選擇喇叭或耳機")
            return False
        self.output.setDevice(device)
        if self.output.isMuted() or self.output.volume() <= 0:
            self.message.emit("目前試聽音量為 0，請調高音量")
            return False
        self.cursor = self.start_ms
        self.positionChanged.emit(self.cursor)
        try:
            if self.wav_path:
                fmt = device.preferredFormat()
                data, peak = pcm_for_output(self.wav_path, self.start_ms, self.end_ms,
                                            fmt.sampleRate(), fmt.channelCount(), fmt.sampleFormat())
                self.buffer = QBuffer(self)
                self.buffer.setData(QByteArray(data))
                self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                self.sink = QAudioSink(device, fmt, self)
                self._volume(self.output.volume())
                active_sink = self.sink
                self.sink.stateChanged.connect(lambda state: self._sink_state(state)
                                                if self.sink is active_sink else None)
                self._playing = True
                self.sink.start(self.buffer)
                # Qt 6.10 exposes separate legacy QAudio and QtAudio enum types.
                if self.sink.error().value != QAudio.Error.NoError.value:
                    raise RuntimeError(f"無法開啟 {device.description()}：{self.sink.error()}")
                self.timer.start()
                self.playingChanged.emit(True)
                self.message.emit(("此範圍的分析音訊為靜音；" if peak == 0 else "播放中；") +
                    f"輸出：{device.description()} · 音量 {self.output.volume():.0%}")
            else:
                if not self.source_path or not self.source_path.is_file():
                    raise FileNotFoundError("找不到分析 WAV 或原媒體；請重新連結媒體")
                self._pending = True
                self.load_timer.start()
                if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                    self._begin_media()
            return True
        except Exception as exc:
            self.stop()
            self.message.emit("無法試聽：" + str(exc))
            return False

    def _begin_media(self):
        self._pending = False
        self.load_timer.stop()
        if 0 <= self.track < len(self.player.audioTracks()):
            self.player.setActiveAudioTrack(self.track)
        self._seeking = True
        self.player.setPosition(self.start_ms)
        self.player.play()
        self._playing = True
        self.playingChanged.emit(True)
        self.message.emit("播放原媒體；輸出：" + self.output.device().description())

    def _media_status(self, status):
        if self._pending and status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self._begin_media()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia and self._playing:
            self._finish()

    def _load_timeout(self):
        if self._pending:
            self.stop()
            self.message.emit("原媒體載入逾時；請重新試聽或重新連結分析 WAV")

    def _media_error(self, _, message):
        self.stop()
        self.message.emit("播放錯誤：" + message)

    def _media_position(self, position):
        if self.wav_path:
            return
        self.cursor = position
        self.positionChanged.emit(position)
        if self._seeking:
            if abs(position - self.start_ms) <= 250:
                self._seeking = False
            return
        if self._playing and position >= self.end_ms:
            self._finish()

    def _sink_state(self, state):
        if not self._playing or self.sink is None:
            return
        if state.value == QAudio.State.IdleState.value:
            self._finish()
        elif state.value == QAudio.State.StoppedState.value and self.sink.error().value != QAudio.Error.NoError.value:
            error = str(self.sink.error())
            self.stop()
            self.message.emit("音訊輸出中斷：" + error)

    def _tick(self):
        if self.sink is not None and self._playing:
            self.cursor = min(self.end_ms, self.start_ms + self.sink.processedUSecs() // 1000)
            self.positionChanged.emit(self.cursor)

    def _finish(self):
        repeat = self.loop()
        start, end, loop = self.start_ms, self.end_ms, self.loop
        self.pause()
        self._resumable = False
        self.cursor = self.end_ms
        self.positionChanged.emit(self.cursor)
        if repeat:
            generation = self._generation
            QTimer.singleShot(0, lambda: self.audition(start, end, loop) if generation == self._generation else None)
        else:
            self.message.emit("試聽完成")

    def position(self):
        return self.cursor if self.wav_path or self.player.source().isEmpty() else self.player.position()

    def is_playing(self):
        return self._playing

    def pause(self):
        was_playing = self._playing
        if was_playing:
            if self.sink is not None:
                self._tick()
            else:
                self.cursor = self.player.position()
            self._resumable = self.start_ms <= self.cursor < self.end_ms
        self._generation += 1
        self._playing = self._pending = False
        self.timer.stop()
        self.load_timer.stop()
        if self.sink is not None:
            self.sink.suspend()
        self.player.pause()
        self.playingChanged.emit(False)

    def resume(self):
        """Continue the same paused range; selection/device/source changes invalidate it."""
        if not self._resumable or self._playing:
            return False
        if self.output.isMuted() or self.output.volume() <= 0:
            self.message.emit("目前試聽音量為 0，請調高音量")
            return False
        self._resumable = False
        self._playing = True
        if self.sink is not None:
            self.sink.resume()
            if self.sink is None or not self._playing:
                return False
            if self.sink.error().value != QAudio.Error.NoError.value:
                self.stop()
                self.message.emit("無法繼續試聽，請重新播放選取句")
                return False
            self.timer.start()
        else:
            self.player.play()
        self.playingChanged.emit(True)
        self.message.emit("繼續試聽")
        return True

    def stop(self):
        self._generation += 1
        self._playing = self._pending = False
        self._resumable = False
        self.timer.stop()
        self.load_timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        if self.buffer is not None:
            self.buffer.close()
            self.buffer.deleteLater()
            self.buffer = None
        self.player.stop()
        self.playingChanged.emit(False)

    def seek(self, position):
        self.stop()
        self.cursor = max(0, int(position))
        if not self.wav_path:
            self.player.setPosition(self.cursor)
        self.positionChanged.emit(self.cursor)
