"""Local reusable vocabulary lists, separate from immutable project prompts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .file_io import replace_checkpoint
from .preferences import preferences_path


def library_path():
    return preferences_path().with_name("vocabulary.json")


def clean_words(text):
    # One entry per line; preserve spelling, case and punctuation in names.
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def validate_library(raw):
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("不支援的詞彙清單格式")
    lists = raw.get("lists")
    if not isinstance(lists, list) or len(lists) > 500:
        raise ValueError("詞彙清單數量無效")
    result, ids, names = [], set(), set()
    for item in lists:
        if not isinstance(item, dict):
            raise ValueError("詞彙清單內容無效")
        ident, name, text = (item.get(k) for k in ("id", "name", "text"))
        if not isinstance(ident, str) or not ident or len(ident) > 80 or ident in ids:
            raise ValueError("詞彙清單識別碼無效")
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or name.strip().casefold() in names:
            raise ValueError("清單名稱不得空白或重複，最多 80 字")
        if not isinstance(text, str) or not text.strip() or len(text) > 100000:
            raise ValueError("請至少新增一個單詞；清單最多 100,000 字")
        result.append({"id": ident, "name": name.strip(), "text": text})
        ids.add(ident)
        names.add(name.strip().casefold())
    return {"schema_version": 1, "lists": result}


def load_library(path=None):
    path = Path(path) if path else library_path()
    if not path.exists():
        return {"schema_version": 1, "lists": []}
    return validate_library(json.loads(path.read_text(encoding="utf-8")))


def save_library(raw, path=None):
    raw = validate_library(raw)
    path = Path(path) if path else library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an unreadable or unknown-version library.
    if path.exists():
        load_library(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(raw, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    # On failure retain the old library and valid temporary file for recovery.
    replace_checkpoint(temporary, path)
