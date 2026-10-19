"""Collect runtime dependency notices; reviewed supplemental files stay external."""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

from packaging.requirements import Requirement

from autosub.release_notices import collect_notices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplements", type=Path, help="Reviewed notice manifest with package/file/sha256/url rows")
    args = parser.parse_args()
    supplements = {}
    if args.supplements:
        base = args.supplements.resolve().parent
        for row in json.loads(args.supplements.read_text(encoding="utf-8")):
            candidate = base / row["file"]
            path = candidate.resolve()
            if not path.is_relative_to(base) or candidate.is_symlink():
                raise ValueError("Supplement escapes its directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Supplement hash mismatch")
            supplements.setdefault(row["package"], []).append(path)
    pending = ["PySide6", "shiboken6", "faster-whisper", "imageio-ffmpeg",
               "SudachiPy", "SudachiDict-core", "budoux", "PyInstaller",
               "pyinstaller-hooks-contrib"]
    packages = {}
    while pending:
        item = distribution(pending.pop())
        name = item.metadata["Name"]
        key = name.lower().replace("_", "-")
        if key in packages:
            continue
        packages[key] = name
        for raw in item.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate():
                pending.append(requirement.name)
    manifest = collect_notices(args.output, packages.values(), supplements=supplements)
    print(f"Collected notices for {len(manifest)} runtime/packager distributions")


if __name__ == "__main__":
    main()
