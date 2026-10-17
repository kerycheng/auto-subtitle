import sys
import pytest

from autosub import cuda_runtime, gpu_bundle


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader")
def test_cuda_loader_keeps_handles_and_only_preloads_once(tmp_path, monkeypatch):
    for name in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        (tmp_path / name).write_bytes(b"placeholder")
    registered, loaded = [], []
    monkeypatch.setattr(cuda_runtime, "runtime_directories", lambda: [tmp_path])
    monkeypatch.setattr(cuda_runtime, "_directories", {})
    monkeypatch.setattr(cuda_runtime, "_libraries", {})
    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", lambda path: registered.append(path) or object())
    monkeypatch.setattr(cuda_runtime.ctypes, "WinDLL", lambda path: loaded.append(path) or object())
    monkeypatch.setenv("PATH", "original-path")
    assert cuda_runtime.prepare_cuda_runtime() == [str(tmp_path)]
    cuda_runtime.prepare_cuda_runtime()
    assert registered == [str(tmp_path)] and len(loaded) == 3
    assert cuda_runtime.os.environ["PATH"].split(cuda_runtime.os.pathsep).count(str(tmp_path)) == 1
    assert len(cuda_runtime._directories) == 1 and len(cuda_runtime._libraries) == 3


def test_cpu_check_never_loads_optional_cuda_dlls(monkeypatch):
    import ctranslate2
    from autosub.runtime import check_runtime
    monkeypatch.setattr(cuda_runtime, "prepare_cuda_runtime", lambda: (_ for _ in ()).throw(AssertionError("CPU must not load CUDA")))
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda _: {"int8"})
    assert check_runtime("cpu")["cuda_runtime_directories"] == []


def test_optional_bundle_copies_dlls_notices_and_hash_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    library = source / "example.dll"
    license_file = source / "LICENSE.txt"
    library.write_bytes(b"test library")
    license_file.write_text("test notice", encoding="utf-8")
    monkeypatch.setattr(gpu_bundle, "bundle_inputs", lambda: [{"name": "example-package", "version": "1.0",
        "libraries": [library], "notices": [license_file]}])
    destination = tmp_path / "cuda"
    report = gpu_bundle.copy_runtime(destination)
    assert (destination / "example.dll").read_bytes() == b"test library"
    assert (destination / "example-package-0-LICENSE.txt").read_text() == "test notice"
    assert len(report["files"][0]["sha256"]) == 64 and (destination / "runtime-manifest.json").is_file()
