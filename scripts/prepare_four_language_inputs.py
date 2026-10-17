"""Prepare reference-free four-language audio inputs from cached FLEURS files."""
import argparse
import json
import wave
from pathlib import Path
from threading import Event

from autosub.alignment_lab import file_hash
from autosub.media import decode_audio

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Select a new output directory")
dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
args.output.mkdir(parents=True)
audio_path = args.output / "input.wav"
windows, cursor = [], 0
with wave.open(str(audio_path), "wb") as destination:
    destination.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    for sample in dataset["samples"]:
        if file_hash(sample["audio_path"]) != sample["audio_sha256"]:
            raise ValueError("Dataset audio hash mismatch")
        decoded = args.output / (sample["id"] + ".wav")
        decode_audio(Path(sample["audio_path"]), decoded, 0, Event())
        with wave.open(str(decoded), "rb") as source:
            count = source.getnframes()
            data = source.readframes(count)
        padding = (-count) % 16
        destination.writeframes(data + bytes(padding * 2))
        stop = cursor + (count + padding) // 16
        windows.append({"id": sample["id"], "language": sample["language"],
                        "window_ms": [cursor, stop], "owner_ms": [cursor, stop],
                        "source_audio_sha256": sample["audio_sha256"], "padding_samples": padding})
        cursor = stop
manifest = {"schema": "asr-lab-v1", "language": None, "duration_ms": cursor,
            "audio_path": str(audio_path.resolve()), "audio_sha256": file_hash(audio_path),
            "source_manifest_sha256": file_hash(args.dataset), "text_provided": False,
            "scope": "FLEURS validation 2 samples per language; not livestream accuracy", "windows": windows}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f'{len(windows)} reference-free samples')
