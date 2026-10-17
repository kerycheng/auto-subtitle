from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from threading import Event

import numpy as np


class Cancelled(Exception):
    pass


def check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise Cancelled("已取消；已完成的階段與辨識區段已保存")


def ffmpeg_path() -> str:
    custom = os.environ.get("AUTOSUB_FFMPEG")
    if custom:
        if not Path(custom).is_file():
            raise FileNotFoundError("AUTOSUB_FFMPEG 指定的檔案不存在")
        return custom
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if bundled.is_file():
        return str(bundled)
    # imageio's bundled executable provides a reproducible decoder without PATH setup.
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def audio_tracks(path: Path) -> list[tuple[int, str]]:
    # PyAV handles probing without requiring a separately installed ffprobe.
    import av
    with av.open(str(path)) as container:
        return [(i, f"音軌 {i + 1} · {stream.metadata.get('language', '未知')} · "
                    f"{stream.codec_context.name or 'audio'}")
                for i, stream in enumerate(container.streams.audio)]


def decode_audio(source: Path, destination: Path, track: int, cancel: Event) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.wav")
    log = destination.with_suffix(".ffmpeg.log")
    command = [ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", str(source),
               "-map", f"0:a:{track}", "-vn", "-af", "aresample=async=1:first_pts=0",
               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temporary)]
    try:
        with log.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                while process.poll() is None:
                    check_cancel(cancel)
                    time.sleep(0.1)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        check_cancel(cancel)
        if process.returncode:
            raise RuntimeError("音訊轉換失敗：" + log.read_text(encoding="utf-8", errors="replace")[-3000:])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("分析音訊必須是 16kHz 單聲道 PCM16")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def waveform(path: Path, points: int | None = None) -> list[float]:
    audio = load_audio(path)
    if not len(audio):
        return []
    # 10ms bins retain enough detail for local timing review on 20-minute clips.
    step = max(1, len(audio) // points) if points else 160
    padded = np.pad(audio, (0, (-len(audio)) % step))
    return np.max(np.abs(padded.reshape(-1, step)), axis=1).tolist()
