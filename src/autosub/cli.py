"""Headless entry point, useful for testing and reproducible jobs."""
import argparse
from pathlib import Path
from threading import Event

from .domain import Project
from .export import export_subtitles
from .media import Cancelled
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="自動打軸機 CLI")
    parser.add_argument("media", nargs="?", type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", choices=["ja", "zh", "en", "ko", "auto", "mixed"], default="ja")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--track", type=int, default=0, help="零起算音軌")
    parser.add_argument("--glossary", default="")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.resume:
        project = Project.load(args.project)
    else:
        if args.media is None:
            parser.error("新專案需要媒體路徑")
        if args.project.exists():
            parser.error("專案已存在，使用 --resume 或指定新檔名")
        project = Project(media_path=str(args.media.resolve()), model=args.model,
                          language=args.language, device=args.device, glossary=args.glossary,
                          audio_stream=args.track)
    cancel = Event()
    import signal
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        result = run_pipeline(project, args.project.resolve(), cancel, lambda percent, text: print(f"{percent:3}% {text}", flush=True))
        if args.export:
            export_subtitles(result, args.export)
    except Cancelled as exc:
        print(exc)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
