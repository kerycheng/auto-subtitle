"""Safe project maintenance: verify audio before relinking; clear alignment only."""
from __future__ import annotations

import copy
import json
import tempfile
import wave
from pathlib import Path
from threading import Event

from .domain import Project
from .media import check_cancel, decode_audio, fingerprint


def _same_pcm(left, right, cancel):
    with wave.open(str(left), "rb") as a, wave.open(str(right), "rb") as b:
        if ((a.getnchannels(), a.getsampwidth(), a.getframerate(), a.getnframes()) !=
                (b.getnchannels(), b.getsampwidth(), b.getframerate(), b.getnframes())):
            return False
        while True:
            check_cancel(cancel)
            x, y = a.readframes(16000), b.readframes(16000)
            if x != y:
                return False
            if not x:
                return True


def relink_media(project: Project, project_path: Path, media: Path, cancel: Event, progress=lambda *_: None):
    audio = Path(project.audio_path)
    if not project.audio_path or not audio.is_file():
        moved = Path(project_path).parent / (Path(project_path).stem + ".cache") / audio.name
        if project.audio_path and moved.is_file():
            audio = moved
        else:
            raise ValueError("缺少原分析音訊，無法驗證重新連結。請先找回專案的分析 WAV／快取。")
    media = Path(media).resolve()
    with wave.open(str(audio), "rb") as cached:
        if abs(cached.getnframes() / 16 - project.duration_ms) > 1:
            raise ValueError("原分析音訊長度與專案不同，拒絕重新連結")
    before = fingerprint(media)
    working = copy.deepcopy(project)
    cache = Path(project_path).parent / (Path(project_path).stem + ".cache")
    cache.mkdir(parents=True, exist_ok=True)
    progress(10, "重新連結：解碼新位置的選定音軌，與原分析音訊逐段核對…")
    with tempfile.TemporaryDirectory(prefix="relink-", dir=cache) as temporary:
        decoded = Path(temporary) / "verified.wav"
        decode_audio(media, decoded, project.audio_stream, cancel)
        check_cancel(cancel)
        if not _same_pcm(audio, decoded, cancel):
            raise ValueError("選定音軌與原分析音訊不同，拒絕沿用舊字幕；請使用相同剪輯版本或建立新專案。")
    check_cancel(cancel)
    if before != fingerprint(media):
        raise ValueError("驗證期間來源檔案已變更，請重試")
    working.media_path = str(media)
    working.audio_path = str(audio.resolve())
    working.source_fingerprint = before
    for cue in working.cues:
        cue.candidate = None  # source identity changed; retain adopted timings and rollback history
    progress(100, "音訊核對一致，已更新媒體位置；字幕與人工修正保留")
    return working


def alignment_cache_info(project_path: Path):
    root = Path(project_path).parent / (Path(project_path).stem + ".cache") / "alignment"
    expected = Path(project_path).parent.resolve() / (Path(project_path).stem + ".cache") / "alignment"
    # Only direct, regular alignment JSON files; never follow directory symlinks.
    files = [] if root.resolve() != expected or root.is_symlink() or root.parent.is_symlink() or not root.exists() else [
        p for p in root.glob("*.json") if p.is_file() and not p.is_symlink()]
    return root, files, sum(p.stat().st_size for p in files)


def clear_alignment_cache(project_path: Path):
    root, files, size = alignment_cache_info(project_path)
    for path in files:
        if path.resolve().parent != root.resolve():
            raise ValueError("快取路徑超出專案對齊目錄")
        path.unlink()
    return len(files), size


def migrate_review_identity(old, new, project_path):
    """Rebase sidecars only after exact PCM verification. Preserve every old file."""
    from .comparison import project_signature, save_session, session_key, validate_session
    old_key, new_key = project_signature(old), project_signature(new)
    store = Path(project_path).parent / (Path(project_path).stem + ".review")
    if not store.is_dir() or store.is_symlink():
        return 0
    migrated = 0
    for path in store.glob("comparison-*.json"):
        if path.is_symlink():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw["project_signature"] != old_key:
                continue
            validate_session(raw)
            raw["project_signature"], raw["source_fingerprint"] = new_key, new.source_fingerprint
            output = store / f"comparison-{session_key(raw)}.json"
            if not output.exists():
                save_session(raw, output)
                migrated += 1
        except (ValueError, KeyError, TypeError):
            continue  # an unrecognized sidecar is preserved, never silently replaced
    annotation = store / f"acoustic-{old_key[:16]}.json"
    output = store / f"acoustic-{new_key[:16]}.json"
    if annotation.is_file() and not annotation.is_symlink() and not output.exists():
        try:
            from .paired_evaluation import paired_report
            raw = json.loads(annotation.read_text(encoding="utf-8"))
            paired_report(old, raw)
            raw["source_fingerprint"] = new.source_fingerprint
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output)
            migrated += 1
        except (ValueError, KeyError, TypeError):
            pass
    return migrated
