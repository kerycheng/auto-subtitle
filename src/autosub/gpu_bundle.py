"""Build an optional CUDA sidecar from installed NVIDIA wheels, with notices."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from .cuda_runtime import PACKAGES


def bundle_inputs():
    packages = []
    for name in PACKAGES:
        distribution = importlib.metadata.distribution(name)
        files = [Path(distribution.locate_file(file)).resolve() for file in distribution.files or []]
        libraries = [path for path in files if path.suffix.lower() == ".dll" and path.is_file()]
        notices = [path for path in files if "license" in path.name.lower() and path.is_file()]
        if not libraries or not notices:
            raise ValueError(f"{name} 缺少 Windows DLL 或授權聲明，無法建立 GPU 可攜版")
        packages.append({"name": name, "version": distribution.version, "libraries": libraries, "notices": notices})
    return packages


def copy_runtime(destination):
    inputs = bundle_inputs()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"packages": [], "files": []}
    seen = {}
    for package in inputs:
        manifest["packages"].append({"name": package["name"], "version": package["version"],
                                     "source": "https://pypi.org/project/" + package["name"] + "/" + package["version"] + "/"})
        for source in package["libraries"]:
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if source.name in seen and seen[source.name] != digest:
                raise ValueError("GPU DLL 同名但內容不同：" + source.name)
            seen[source.name] = digest
            shutil.copy2(source, destination / source.name)
            manifest["files"].append({"name": source.name, "size": source.stat().st_size, "sha256": digest})
        for index, source in enumerate(package["notices"]):
            shutil.copy2(source, destination / f"{package['name']}-{index}-{source.name}")
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
