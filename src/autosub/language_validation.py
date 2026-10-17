"""Offline four-language pipeline smoke test; dataset text never enters ASR."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from threading import Event

from .comparison import read_reference
from .domain import Project
from .export import export_subtitles
from .paired_evaluation import _distance
from .pipeline import run_pipeline


def normalized_text(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z"})


def run_language_validation(manifest_path, model_path, output, device="cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("驗證裝置必須是 cpu 或 cuda")
    manifest_path, model_path, output = Path(manifest_path), Path(model_path), Path(output)
    if not (model_path / "model.bin").is_file():
        raise ValueError("驗證必須指定已下載的本機模型")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = Counter(sample["language"] for sample in manifest["samples"])
    if set(counts) != {"en", "zh", "ja", "ko"} or min(counts.values()) < 2:
        raise ValueError("驗證集須包含中英日韓各至少兩句")
    seen = set()
    for sample in manifest["samples"]:
        name = sample["id"]
        if not re.fullmatch(r"(?:en|zh|ja|ko)-[0-9]+", name) or not name.startswith(sample["language"] + "-") or name in seen:
            raise ValueError("樣本 ID 無效或重複")
        seen.add(name)
        if not normalized_text(sample["reference_text"]):
            raise ValueError("樣本缺少資料集原語文字")
        if hashlib.sha256(Path(sample["audio_path"]).read_bytes()).hexdigest() != sample["audio_sha256"]:
            raise ValueError("下載音訊與紀錄的 SHA256 不符")
    if output.exists() and any(output.iterdir()):
        raise ValueError("請選空白輸出目錄，避免覆蓋驗證結果")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "small read-speech functional check, not livestream or timing accuracy",
        "dataset": manifest.get("dataset"), "license": manifest.get("license"), "split": manifest.get("split"),
        "model": str(model_path.resolve()), "device": device, "samples": [],
        "normalization": "NFKC, casefold, remove whitespace and Unicode punctuation; no Traditional/Simplified or numeral equivalence",
        "acoustic_timing_accuracy": None}
    for sample in manifest["samples"]:
        start = time.perf_counter()
        messages = []
        project = run_pipeline(Project(media_path=str(Path(sample["audio_path"]).resolve()),
            model=str(model_path.resolve()), language=sample["language"], device=device),
            output / (sample["id"] + ".autosub.json"), Event(), lambda _, message: messages.append(message))
        if not project.words or not project.cues:
            raise RuntimeError("樣本沒有產生辨識詞或字幕：" + sample["id"])
        for suffix in (".ass", ".srt"):
            path = output / (sample["id"] + suffix)
            export_subtitles(project, path)
            parsed, _ = read_reference(path)
            if len(parsed) != len(project.cues):
                raise RuntimeError("字幕匯出再讀取的事件數不同")
        separator = " " if sample["language"] in {"en", "ko"} else ""
        hypothesis = separator.join(cue.text for cue in project.cues)
        # Reference text is consulted only after the whole ASR pipeline returns.
        reference = sample["reference_text"]
        truth, predicted = normalized_text(reference), normalized_text(hypothesis)
        edits = _distance(truth, predicted)
        report["samples"].append({"id": sample["id"], "language": sample["language"],
            "dataset_id": sample["dataset_id"], "audio_sha256": sample["audio_sha256"],
            "duration_ms": project.duration_ms, "elapsed_seconds": time.perf_counter() - start,
            "words": len(project.words), "cues": len(project.cues), "reference_text": reference,
            "recognized_text": hypothesis, "normalized_characters": len(truth), "character_edits": edits,
            "mechanical_cer": edits / len(truth) if truth else None,
            "reused_model": any("沿用已載入" in message for message in messages), "ass_srt_round_trip": "passed"})
        (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["status"] = "passed"
    report["by_language"] = {}
    for language in counts:
        rows = [row for row in report["samples"] if row["language"] == language]
        report["by_language"][language] = {"samples": len(rows), "duration_ms": sum(row["duration_ms"] for row in rows),
            "mechanical_cer": sum(row["character_edits"] for row in rows) / sum(row["normalized_characters"] for row in rows)}
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    body = ['<!doctype html><meta charset="utf-8"><title>四語言流程驗證</title>',
        '<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}</style>',
        '<h1>四語言小樣本流程驗證</h1><p>僅少量朗讀語音，不代表直播辨識品質或打軸精度。CER 未轉換繁簡、數字及同義字形；通過只表示辨識與匯出流程成功。</p>',
        '<p>資料：<a href="https://huggingface.co/datasets/google/fleurs">Google FLEURS</a> · CC-BY-4.0 · validation</p>',
        '<table><tr><th>樣本</th><th>資料集文字</th><th>辨識文字</th><th>機械 CER</th></tr>']
    for row in report["samples"]:
        body.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in
            [row["id"], row["reference_text"], row["recognized_text"], f'{row["mechanical_cer"]:.1%}']) + '</tr>')
    (output / "index.html").write_text(''.join(body) + '</table>', encoding="utf-8")
    return report
