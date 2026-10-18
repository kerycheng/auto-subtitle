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


def compose_vocabulary(library, choice):
    """Combine checked lists in library order; keep a legacy prompt verbatim."""
    ids, base = choice["ids"], choice["base_text"]
    available = {item["id"] for item in library["lists"]}
    if len(set(ids)) != len(ids) or not set(ids) <= available:
        raise ValueError("詞彙清單已變更，請重新選擇")
    items = [item for item in library["lists"] if item["id"] in ids]
    if not items:
        result = base
    elif not base and len(items) == 1:
        result = items[0]["text"]
    else:
        seen = set(clean_words(base))
        added = []
        for item in items:
            for word in clean_words(item["text"]):
                if word not in seen:
                    seen.add(word)
                    added.append(word)
        result = base
        if added:
            result += ("\n" if base and not base.endswith("\n") else "") + "\n".join(added)
    if len(result) > 100000:
        raise ValueError("合併提示超過 100,000 字，請減少勾選清單")
    return result


def resolve_vocabulary_choice(library, text, hint=None):
    """A saved selection is only valid while it reproduces the exact prompt."""
    if hint is not None:
        try:
            if compose_vocabulary(library, hint) == text:
                return {"ids": list(hint["ids"]), "base_text": hint["base_text"]}
        except (ValueError, KeyError, TypeError):
            pass
    if text:
        for item in library["lists"]:
            if item["text"] == text:
                return {"ids": [item["id"]], "base_text": ""}
    return {"ids": [], "base_text": text}


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
