"""Build and package Windows portable release archives (CPU and optional GPU) into dist/."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from autosub import __version__


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_variant(root: Path, *, with_cuda: bool, version: str) -> dict:
    variant = f"AutoSubtitle-GPU-{version}" if with_cuda else f"AutoSubtitle-{version}"
    cmd = [sys.executable, str(root / "scripts" / "build.py"), "--output-name", variant]
    if with_cuda:
        cmd.append("--with-cuda")
    subprocess.run(cmd, cwd=str(root), check=True)

    dist_dir = root / "dist" / variant
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"找不到建置輸出目錄：{dist_dir}")

    archive_base = root / "dist" / variant
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(root / "dist"), base_dir=variant))
    release_meta = json.loads((dist_dir / "release.json").read_text(encoding="utf-8"))
    return {
        "variant": variant,
        "directory": str(dist_dir.relative_to(root)),
        "zip_file": archive_path.name,
        "zip_sha256": sha256_file(archive_path),
        "executable_sha256": release_meta.get("executable_sha256", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Only build and zip the CPU portable package")
    parser.add_argument("--gpu-only", action="store_true", help="Only build and zip the GPU portable package")
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        raise ValueError("不可同時指定 --cpu-only 與 --gpu-only")

    root = Path(__file__).resolve().parents[1]
    results = []
    if not args.gpu_only:
        results.append(build_variant(root, with_cuda=False, version=__version__))
    if not args.cpu_only:
        results.append(build_variant(root, with_cuda=True, version=__version__))

    manifest_path = root / "dist" / f"release-checksums-{__version__}.json"
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
