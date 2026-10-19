import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from autosub.release_notices import collect_notices, copy_verified_notices


def package(root, name="example", files=("LICENSE",)):
    return SimpleNamespace(metadata={"Name": name}, version="1.0", files=[PurePosixPath(p) for p in files],
                           locate_file=lambda p: root / p)


def test_copies_only_notices_and_manifest_has_no_source_paths(tmp_path):
    src = tmp_path / "input"
    src.mkdir()
    (src / "LICENSE").write_text("Synthetic permission text", encoding="utf-8")
    (src / "METADATA").write_text("Local metadata must stay behind", encoding="utf-8")
    out = tmp_path / "out"
    result = collect_notices(out, ["example"], lookup=lambda _: package(src, files=("LICENSE", "METADATA")))
    text = (out / "packages.json").read_text(encoding="utf-8")
    assert str(src) not in text and "METADATA" not in text
    assert (out / result[0]["notices"][0]["file"]).read_text(encoding="utf-8") == "Synthetic permission text"
    assert json.loads(text) == result


def test_preflights_all_packages_before_writing(tmp_path):
    (tmp_path / "LICENSE").write_text("Synthetic permission", encoding="utf-8")
    out = tmp_path / "out"
    def lookup(name):
        return package(tmp_path, name, ("LICENSE",) if name == "first" else ())
    with pytest.raises(ValueError, match="Missing"):
        collect_notices(out, ["first", "second"], lookup=lookup)
    assert not out.exists()


def test_reviewed_supplement_covers_wheel_without_license(tmp_path):
    notice = tmp_path / "reviewed.txt"
    notice.write_text("Independent synthetic notice", encoding="utf-8")
    out = tmp_path / "out"
    result = collect_notices(out, ["example"], supplements={"example": [notice]},
                             lookup=lambda _: package(tmp_path, files=()))
    assert len(result[0]["notices"]) == 1


def test_refuses_to_overwrite_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        collect_notices(tmp_path, [], lookup=lambda _: pytest.fail("Must not inspect packages"))


def test_rejects_unsafe_metadata_name_before_output(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe"):
        collect_notices(out, ["example"], lookup=lambda _: package(tmp_path, "../outside"))
    assert not out.exists()


def test_duplicate_distribution_aliases_do_not_duplicate_notices(tmp_path):
    (tmp_path / "LICENSE").write_text("Synthetic permission", encoding="utf-8")
    result = collect_notices(tmp_path / "out", ["example", "Example"],
                             lookup=lambda _: package(tmp_path))
    assert len(result) == 1


def test_copy_ignores_unlisted_files(tmp_path):
    (tmp_path / "LICENSE").write_text("Synthetic notice", encoding="utf-8")
    source = tmp_path / "notices"
    collect_notices(source, ["example"], lookup=lambda _: package(tmp_path))
    (source / "unlisted.txt").write_text("Must stay local", encoding="utf-8")
    copy_verified_notices(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle/unlisted.txt").exists()


def test_changed_notice_rejected_before_creating_bundle(tmp_path):
    (tmp_path / "LICENSE").write_text("Synthetic notice", encoding="utf-8")
    source = tmp_path / "notices"
    result = collect_notices(source, ["example"], lookup=lambda _: package(tmp_path))
    (source / result[0]["notices"][0]["file"]).write_text("Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        copy_verified_notices(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_escaped_notice_rejected_before_creating_bundle(tmp_path):
    source = tmp_path / "notices"
    source.mkdir()
    (source / "packages.json").write_text(json.dumps([
        {"name": "example", "version": "1.0", "notices": [{"file": "../outside", "sha256": "0" * 64}]}
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        copy_verified_notices(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
