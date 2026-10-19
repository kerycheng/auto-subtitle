"""Synthetic import tables exercise build-time copying without vendor binaries."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("rebind_pyav", Path(__file__).parents[1] / "scripts" / "rebind_pyav.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def package(tmp_path, monkeypatch):
    old = b"avutil-61-0123456789abcdef.dll"
    source = tmp_path / "source"
    source.mkdir()
    original = b"PREFIX" + old + b"\0SUFFIX"
    (source / "core.pyd").write_bytes(original)
    (source / "__init__.py").write_text("# synthetic package", encoding="utf-8")
    class PE:
        def __init__(self, data):
            self.DIRECTORY_ENTRY_IMPORT = [SimpleNamespace(dll=old, struct=SimpleNamespace(Name=6))]
        def get_offset_from_rva(self, rva):
            return rva
        def close(self):
            pass
    monkeypatch.setattr(module.pefile, "PE", PE)
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    return source, libraries, tmp_path / "output", original


def test_rebinding_preserves_source_and_other_bytes(package):
    source, libraries, output, original = package
    (libraries / "avutil-61.dll").write_bytes(b"synthetic library")
    report = module.copy_rebound_package(source, libraries, output)
    assert (source / "core.pyd").read_bytes() == original
    patched = (output / "av" / "core.pyd").read_bytes()
    assert len(patched) == len(original)
    assert patched.startswith(b"PREFIXavutil-61.dll\0") and patched.endswith(b"SUFFIX")
    assert (output / "av.libs" / "avutil-61.dll").read_bytes() == b"synthetic library"
    assert report[0]["imports"][0]["to"] == "avutil-61.dll"


def test_missing_library_does_not_create_output(package):
    source, libraries, output, _ = package
    with pytest.raises(ValueError, match="Missing matching library"):
        module.copy_rebound_package(source, libraries, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(package):
    source, libraries, output, _ = package
    output.mkdir()
    (output / "keep").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.copy_rebound_package(source, libraries, output)
    assert (output / "keep").read_text(encoding="utf-8") == "keep"
