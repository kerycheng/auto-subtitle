"""Download pinned official experiment models, never upload media."""
import argparse
import hashlib
import json
import os
from pathlib import Path

MODELS = {"qwen-aligner": ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
          "whisperx-ja": ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "cf031e020336460d15a417eba710bbc5bb43be9a"),
          "qwen-asr": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=list(MODELS))
    parser.add_argument("--output", type=Path, default=Path("models/alignment"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download
    for name in args.models:
        folder = args.output.resolve() / name
        manifest = folder / "experiment-model.json"
        if manifest.exists():
            print(f"{name}: reusing pinned snapshot", flush=True)
            continue
        repo, revision = MODELS[name]
        info = HfApi().model_info(repo, revision=revision)
        files = [f.rfilename for f in info.siblings]
        safe = [f for f in files if f.endswith(".safetensors")]
        weights = safe or [f for f in files if f.endswith("pytorch_model.bin")]
        if not weights:
            raise ValueError(f"No supported weights: {repo}")
        selected = weights + [f for f in files if f.endswith((".json", ".txt", ".model", ".dict"))
                              or Path(f).name.lower() in {"license", "readme.md"}]
        print(f"{name}: {repo}@{info.sha}", flush=True)
        snapshot_download(repo, revision=info.sha, local_dir=str(folder), allow_patterns=selected, max_workers=2)
        hashes = {}
        for filename in selected:
            path = folder / filename
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            hashes[filename] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        manifest.write_text(json.dumps({"repo": repo, "revision": info.sha, "files": hashes}, indent=2), encoding="utf-8")
        print(f"{name}: ready", flush=True)


if __name__ == "__main__":
    main()
