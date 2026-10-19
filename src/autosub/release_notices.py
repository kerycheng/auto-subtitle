"""Collect installed-package notices without copying local metadata or settings."""
from __future__ import annotations

import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import re
import shutil


def collect_notices(destination, packages, *, supplements=None, lookup=distribution):
    """Preflight every notice before creating a new output directory.

    Supplements are caller-reviewed license files for wheels without notices.
    This records package notices, not a legal assessment or a dependency audit.
    No source filesystem paths, wheel metadata or credentials enter the manifest.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Notice output already exists")
    supplements = supplements or {}
    prepared, seen = [], set()
    for requested in packages:
        package = lookup(requested)
        name, version = package.metadata["Name"], package.version
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("Unsafe package name")
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        files = [Path(package.locate_file(item)) for item in package.files or []
                 if any(word in item.name.lower() for word in ("license", "licence", "copying", "notice"))]
        files += [Path(item) for item in supplements.get(key, [])]
        files = list(dict.fromkeys(files))
        if not files or any(not p.is_file() or p.is_symlink() or not p.stat().st_size for p in files):
            raise ValueError("Missing or unreadable package notice: " + name)
        prepared.append((key, name, version, files))
    if not prepared:
        raise ValueError("No packages selected")
    destination.mkdir(parents=True)
    manifest = []
    for key, name, version, files in sorted(prepared):
        folder = destination / key
        folder.mkdir()
        notices = []
        for index, source in enumerate(files):
            target = folder / f"{index:03d}-{source.name}"
            shutil.copyfile(source, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            notices.append({"file": target.relative_to(destination).as_posix(), "sha256": digest})
        manifest.append({"name": name, "version": version, "notices": notices})
    (destination / "packages.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def copy_verified_notices(source, destination):
    """Copy only manifest-listed notices, rejecting changed or escaped files."""
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError("Notice output already exists")
    manifest = json.loads((source / "packages.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Notice manifest must list packages")
    files = []
    clean = []
    for package in manifest:
        name, version = package["name"], package["version"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not re.fullmatch(r"[A-Za-z0-9.+!_-]+", version):
            raise ValueError("Invalid package metadata")
        notices = []
        for item in package["notices"]:
            relative = item["file"]
            path = source / relative
            if (Path(relative).is_absolute() or "\\" in relative or ".." in Path(relative).parts
                    or path.is_symlink() or not path.resolve().is_relative_to(source)):
                raise ValueError("Notice path escapes its source")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError("Notice hash mismatch")
            files.append((relative, path))
            notices.append({"file": relative, "sha256": digest})
        if not notices:
            raise ValueError("Package has no notices")
        clean.append({"name": name, "version": version, "notices": notices})
    destination.mkdir(parents=True)
    for relative, path in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination / "packages.json").write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean
