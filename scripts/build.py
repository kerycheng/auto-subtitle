"""Build on the target OS: python scripts/build.py. Models remain external."""
import os
import sys
import argparse
from pathlib import Path

import imageio_ffmpeg
import PyInstaller.__main__

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--with-cuda", action="store_true", help="Build AutoSubtitle-GPU with optional NVIDIA runtime sidecar")
parser.add_argument("--output-name", help="Optional separate release directory/executable name")
args = parser.parse_args()
build_name = "AutoSubtitle-GPU" if args.with_cuda else "AutoSubtitle"
if args.output_name:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",args.output_name):
        raise ValueError("Invalid output name")
    build_name = args.output_name
if args.with_cuda:
    from autosub.gpu_bundle import bundle_inputs
    bundle_inputs()  # fail before a long build if optional wheels/notices are missing
os.chdir(root)
os.environ.setdefault("HF_XET_CACHE", str(root / ".cache" / "xet"))
os.environ["PYINSTALLER_CONFIG_DIR"] = str(root / ".cache" / "pyinstaller")
if os.name == "nt":
    # Keep unrelated applications' DLLs off the build search path. In particular,
    # Poppler ships an ICU DLL with the same filename as Windows ICU but an
    # incompatible export ABI; collecting it makes QtWidgets fail at startup.
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    os.environ["PATH"] = os.pathsep.join(map(str, [
        Path(sys.executable).parent, Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs",
        windows / "System32", windows,
    ]))
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
# media.ffmpeg_path expects a stable filename inside the bundle.
target = root / "build" / "vendor" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
target.parent.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy2(ffmpeg, target)

PyInstaller.__main__.run([
    str(root / "scripts" / "launch.py"), "--name", build_name, "--windowed", "--onedir", "--noconfirm", "--clean",
    "--specpath", str(root / "build" / "specs"),
    "--workpath", str(root / "build" / "pyinstaller"),
    "--paths", str(root / "src"),
    "--collect-all", "faster_whisper", "--collect-all", "ctranslate2", "--collect-all", "tokenizers",
    "--collect-all", "onnxruntime", "--collect-all", "av",
    "--collect-all", "budoux", "--collect-all", "sudachipy", "--collect-all", "sudachidict_core",
    "--copy-metadata", "budoux", "--copy-metadata", "SudachiPy", "--copy-metadata", "SudachiDict-core",
    "--copy-metadata", "faster-whisper", "--copy-metadata", "huggingface-hub",
    "--copy-metadata", "ctranslate2",
    "--add-binary", f"{target}{os.pathsep}bin",
    *(["--osx-bundle-identifier", "app.autosubtitle.desktop"] if sys.platform == "darwin" else []),
])
quickstart = root / "docs" / "windows-quickstart.txt"
if quickstart.is_file():
    shutil.copy2(quickstart, root / "dist" / build_name / "README-zh-TW.txt")
license_file = root / "LICENSE"
if license_file.is_file():
    shutil.copy2(license_file, root / "dist" / build_name / "LICENSE.txt")
from importlib.metadata import distribution
import json
notices = root / "dist" / build_name / "licenses" / "japanese-segmentation"
notices.mkdir(parents=True, exist_ok=True)
manifest = []
for name in ("budoux", "SudachiPy", "SudachiDict-core"):
    package = distribution(name)
    licenses = [file for file in package.files or [] if "license" in str(file).lower()]
    if not licenses:
        raise ValueError(f"{name} 缺少授權聲明")
    for file in licenses:
        source = Path(package.locate_file(file))
        shutil.copy2(source, notices / (name + "-" + source.name))
    manifest.append({"package": name, "version": package.version,
                     "project_urls": package.metadata.get_all("Project-URL", [])})
(notices / "packages.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
if args.with_cuda:
    from autosub.gpu_bundle import copy_runtime
    copy_runtime(root / "dist" / build_name / "cuda")
from autosub import __version__
from autosub.alignment_lab import file_hash
executable = root / "dist" / build_name / (build_name + ".exe" if os.name == "nt" else build_name)
(root / "dist" / build_name / "release.json").write_text(json.dumps({
    "version": __version__, "variant": build_name,
    "executable_sha256": file_hash(executable),
    "japanese_segmentation": manifest,
    "models_bundled": False,
}, indent=2), encoding="utf-8")
