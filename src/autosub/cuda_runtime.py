"""Optional Windows CUDA runtime discovery, scoped to this process."""
from __future__ import annotations

import ctypes
import importlib.metadata
import os
import sys
from pathlib import Path
from threading import RLock

PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12")
_directories = {}
_libraries = {}
_lock = RLock()


def runtime_directories():
    paths = []
    custom = os.environ.get("AUTOSUB_CUDA_DIR")
    if custom:
        for name in custom.split(os.pathsep):
            path = Path(name).resolve()
            if not path.is_dir():
                raise FileNotFoundError("AUTOSUB_CUDA_DIR 不是有效資料夾：" + str(path))
            paths.append(path)
    if getattr(sys, "frozen", False):
        bundled = Path(sys.executable).resolve().parent / "cuda"
        if bundled.is_dir():
            paths.append(bundled)
    else:
        for package in PACKAGES:
            try:
                distribution = importlib.metadata.distribution(package)
            except importlib.metadata.PackageNotFoundError:
                continue
            paths.extend(Path(distribution.locate_file(file)).resolve().parent
                         for file in distribution.files or [] if str(file).lower().endswith(".dll"))
    return list(dict.fromkeys(paths))


def prepare_cuda_runtime():
    if sys.platform != "win32":
        return []
    with _lock:
        paths = runtime_directories()
        # Keep directory handles alive. AddDllDirectory alone is insufficient for
        # native code using plain LoadLibrary, so also update this process's PATH.
        for path in paths:
            key = str(path)
            if key not in _directories:
                _directories[key] = os.add_dll_directory(key)
        old_path = os.environ.get("PATH", "").split(os.pathsep)
        additions = [str(path) for path in paths if str(path) not in old_path]
        if additions:
            os.environ["PATH"] = os.pathsep.join(additions + old_path)
        for name in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
            location = next((path / name for path in paths if (path / name).is_file()), None)
            if location is not None and str(location) not in _libraries:
                _libraries[str(location)] = ctypes.WinDLL(str(location))
        return [str(path) for path in paths]
