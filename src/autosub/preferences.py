"""Small, versioned user defaults; project state always takes precedence."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .domain import TimingSettings


def preferences_path():
    override = os.environ.get("AUTOSUB_SETTINGS_PATH")
    if override:
        return Path(override)
    from PySide6.QtCore import QStandardPaths
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)) / "preferences.json"


def validate(raw):
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("不支援的使用者設定格式")
    result = {"schema_version": 1}
    for key, low, high in [("volume", 0, 100), ("vad_threshold", .05, .95)]:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError("使用者設定範圍無效：" + key)
        result[key] = value
    if type(result["volume"]) is not int:
        raise ValueError("音量必須為整數")
    for key, allowed in [("language", {"ja", "zh", "en", "ko", "auto", "mixed"}),
                         ("device", {"cpu", "cuda"}), ("boundary_policy", {"legacy", "conservative"})]:
        if raw[key] not in allowed:
            raise ValueError("使用者設定選項無效：" + key)
        result[key] = raw[key]
    for key in ("model", "glossary"):
        if not isinstance(raw[key], str) or len(raw[key]) > 100000:
            raise ValueError("使用者文字設定無效：" + key)
        result[key] = raw[key]
    if not result["model"].strip():
        raise ValueError("模型不得為空白")
    output = raw.get("output_device")
    if output is not None:
        if not isinstance(output, str) or len(output) > 4096:
            raise ValueError("音訊裝置設定無效")
        bytes.fromhex(output)
    result["output_device"] = output
    if type(raw.get("wrap_export")) is not bool:
        raise ValueError("換行設定無效")
    result["wrap_export"] = raw["wrap_export"]
    settings = TimingSettings(**raw["timing"])
    settings.validate()
    result["timing"] = vars(settings)
    return result


def load_preferences(path=None):
    path = Path(path) if path else preferences_path()
    if not path.exists():
        return None, ""
    try:
        return validate(json.loads(path.read_text(encoding="utf-8"))), ""
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return None, f"使用者設定無法載入，暫用預設值；原檔保留：{exc}"


def save_preferences(raw, path=None):
    raw = validate(raw)
    path = Path(path) if path else preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _, warning = load_preferences(path)
        if warning:
            # Preserve damaged/unknown settings before replacing with known valid defaults.
            backup = path.with_suffix(path.suffix + ".invalid.bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(raw, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
