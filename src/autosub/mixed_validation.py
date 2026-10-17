"""Constructed, sequential ja/ko regression; not overlapping speech or timing gold."""
import hashlib
import html
import json
import re
import wave
from pathlib import Path
from threading import Event

from .domain import Project
from .export import export_subtitles
from .language_validation import normalized_text
from .media import decode_audio
from .paired_evaluation import _distance
from .pipeline import run_pipeline


def run_mixed_validation(manifest_path, model_path, output, device="cpu", strategy="mixed-vad-v1"):
    if device not in {"cpu", "cuda"} or strategy not in {"fixed30-v1", "mixed-vad-v1"}:
        raise ValueError("混語驗證裝置或策略無效")
    model_path, output = Path(model_path), Path(output)
    if not (model_path / "model.bin").is_file():
        raise ValueError("請指定已下載的本機模型")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = {sample["id"]: sample for sample in manifest["samples"]}
    names = ["ja-0", "ko-0", "ja-1", "ko-1"]
    for name in names:
        sample = samples[name]
        if sample["language"] != name[:2] or hashlib.sha256(Path(sample["audio_path"]).read_bytes()).hexdigest() != sample["audio_sha256"]:
            raise ValueError("混語樣本的語言或 SHA256 不符")
    if output.exists() and any(output.iterdir()):
        raise ValueError("請選空白輸出目錄")
    output.mkdir(parents=True, exist_ok=True)
    chunks, regions, clock = [], [], 0
    for name in names:
        sample = samples[name]
        path = output / (name + ".wav")
        decode_audio(Path(sample["audio_path"]), path, 0, Event())
        with wave.open(str(path), "rb") as stream:
            frames = stream.readframes(stream.getnframes())
        duration = len(frames) // 32
        regions.append({"id": name, "language": sample["language"], "start_ms": clock, "end_ms": clock + duration})
        chunks.extend([frames, bytes(19200)])  # 600ms silence, including final tail
        clock += duration + 600
    audio = output / "alternating-ja-ko.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"".join(chunks))
    project = run_pipeline(Project(media_path=str(audio.resolve()), model=str(model_path.resolve()),
        language="mixed", device=device, asr_strategy=strategy), output / "mixed.autosub.json", Event(), lambda *_: None)
    for region in regions:
        words = [word for word in project.words if region["start_ms"] <= (word.start_ms + word.end_ms)/2 < region["end_ms"]]
        text = "".join(word.text for word in words)
        reference = samples[region["id"]]["reference_text"]
        truth = normalized_text(reference)
        region.update(recognized_text=text, reference_text=reference,
            expected_script_present=bool(re.search("[\u3040-\u30ff]" if region["language"] == "ja" else "[\uac00-\ud7af]", text)),
            mechanical_cer=_distance(truth, normalized_text(text)) / len(truth) if truth else None)
    for suffix in (".ass", ".srt"):
        export_subtitles(project, output / ("mixed" + suffix))
    report = {"language": "mixed", "device": device, "asr_strategy": project.asr_strategy,
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "duration_ms": project.duration_ms,
        "regions": regions, "cues": len(project.cues), "words": len(project.words),
        "script_presence_passed": all(region["expected_script_present"] for region in regions),
        "note": "Constructed sequential read speech with 600ms silence; word-midpoint assignment is a diagnostic, not acoustic gold or full transcript acceptance."}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    body = ['<!doctype html><meta charset="utf-8"><title>日韓交替回歸</title><h1>日韓交替語音回歸</h1>',
        '<p>僅驗證已知語段中預期文字系統是否出現，不能代表全文正確或同時發話能力。資料：Google FLEURS validation · CC-BY-4.0，人工文字不輸入辨識。</p>',
        '<table border="1" cellpadding="8"><tr><th>語段</th><th>資料集文字</th><th>辨識文字</th><th>預期文字系統</th><th>機械 CER</th></tr>']
    for row in regions:
        values = [row["id"], row["reference_text"], row["recognized_text"], str(row["expected_script_present"]), f'{row["mechanical_cer"]:.1%}']
        body.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in values) + '</tr>')
    (output / "index.html").write_text(''.join(body) + '</table>', encoding="utf-8")
    return report
