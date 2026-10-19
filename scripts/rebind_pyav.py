"""Rebind a copied PyAV package to separately built compatible Windows DLLs.

Only PE import-name strings are changed; source binaries stay untouched.
This does not establish ABI compatibility. Test imports and decoding afterward.
"""
from pathlib import Path
import argparse
import json
import re
import shutil

import pefile


def copy_rebound_package(source, libraries, destination):
    source, libraries, destination = map(Path, (source, libraries, destination))
    if destination.exists():
        raise FileExistsError("Destination already exists")
    required = {}
    prepared = []
    for binary in sorted(source.rglob("*.pyd")):
        data = bytearray(binary.read_bytes())
        pe = pefile.PE(data=bytes(data))
        changes = []
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            old = entry.dll.decode("ascii")
            match = re.fullmatch(r"(av(?:codec|device|filter|format|util)|sw(?:resample|scale))-(\d+)-[a-f0-9]+\.dll", old)
            if not match:
                continue
            new = f"{match[1]}-{match[2]}.dll"
            library = libraries / new
            if not library.is_file() or library.is_symlink():
                raise ValueError("Missing matching library: " + new)
            offset = pe.get_offset_from_rva(entry.struct.Name)
            encoded = new.encode("ascii")
            assert len(encoded) <= len(entry.dll)
            data[offset:offset + len(entry.dll) + 1] = encoded + b"\0" * (len(entry.dll) + 1 - len(encoded))
            required[new] = library
            changes.append({"from": old, "to": new})
        pe.close()
        prepared.append((binary.relative_to(source), data, changes))
    if not prepared or not required:
        raise ValueError("No hashed PyAV library imports found")
    # Preflight all requested libraries before making a distributable copy.
    shutil.copytree(source, destination / "av", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "av.libs").mkdir()
    for name, library in required.items():
        shutil.copyfile(library, destination / "av.libs" / name)
    report = []
    for relative, data, changes in prepared:
        (destination / "av" / relative).write_bytes(data)
        report.append({"binary": relative.as_posix(), "imports": changes})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("libraries", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(copy_rebound_package(args.source, args.libraries, args.destination), indent=2))
