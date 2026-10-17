"""Download a tiny attributed FLEURS sample set; never uploads local media."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path

DATASET = "https://huggingface.co/datasets/google/fleurs"
CONFIGS = {"en": "en_us", "zh": "cmn_hans_cn", "ja": "ja_jp", "ko": "ko_kr"}


def fetch(url, maximum):
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("only HTTPS dataset downloads are allowed")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read(maximum + 1)
            break
        except urllib.error.HTTPError as exc:
            if (exc.code < 500 and exc.code != 429) or attempt == 2:
                raise
            delay = min(60, max(1, int(exc.headers.get("Retry-After", "30")))) if exc.code == 429 else 1
            time.sleep(delay)
    if len(data) > maximum:
        raise ValueError("sample response exceeds size limit")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, choices=range(1, 4), default=2)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("請選空白目錄，保留已下載的測試樣本")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET, "license": "CC-BY-4.0", "split": "validation",
                "purpose": "small functional smoke test; not representative of livestream quality", "samples": []}
    for language, config in CONFIGS.items():
        for position in range(args.per_language):
            query = urllib.parse.urlencode({"dataset": "google/fleurs", "config": config,
                "split": "validation", "offset": position, "length": 1})
            endpoint = "https://datasets-server.huggingface.co/rows?" + query
            metadata = fetch(endpoint, 4 * 1024 * 1024)
            (args.output / f"{language}-{position}-metadata.json").write_bytes(metadata)
            item = json.loads(metadata)["rows"][0]
            row = item["row"]
            audio = row["audio"][0]
            # Dataset viewer serves normalized audio containers; decode using FFmpeg later.
            path = args.output / f"{language}-{position}.audio"
            data = fetch(audio["src"], 20 * 1024 * 1024)
            path.write_bytes(data)
            manifest["samples"].append({"id": f"{language}-{position}", "language": language,
                "config": config, "dataset_id": row["id"], "audio_path": str(path.resolve()),
                "audio_sha256": hashlib.sha256(data).hexdigest(), "reference_text": row["raw_transcription"],
                "metadata_endpoint": endpoint, "mime_type": audio["type"]})
        print(language + ": downloaded " + str(args.per_language) + " samples", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "ATTRIBUTION.txt").write_text(
        "FLEURS — Google, CC-BY-4.0\n" + DATASET + "\n"
        "Selected first validation rows for en_us, cmn_hans_cn, ja_jp, ko_kr.\n"
        "Downloaded audio is unmodified; analysis WAVs are converted to mono 16kHz PCM16.\n"
        "Dataset text is only read by the evaluator after ASR, never supplied to Whisper.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
