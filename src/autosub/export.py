from __future__ import annotations

import math
import re
from pathlib import Path

from .domain import Project


def stamp(units: int, ass: bool = False) -> str:
    scale = 100 if ass else 1000
    hours, rest = divmod(units, 3600 * scale)
    minutes, rest = divmod(rest, 60 * scale)
    seconds, fraction = divmod(rest, scale)
    return (f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}" if ass else
            f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}")


def ass_text(text: str) -> str:
    # Never interpret recognized text as ASS commands or override blocks.
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", r"\N")


def output_bounds(project: Project, cue, *, ass: bool) -> tuple[int, int]:
    """Output units, shared with review so rounding and EOF clipping agree."""
    scale = 10 if ass else 1
    return (math.floor(cue.start_ms / scale + 0.5),
            min(project.duration_ms // scale, math.floor(cue.end_ms / scale + 0.5)))


def wrap_text(text: str) -> str:
    """At most one balanced break. Use spaces for English/Korean; CJK char boundary otherwise."""
    text = " ".join(text.split())
    if len(text) < 26:
        return text
    midpoint = len(text) // 2
    boundaries = [m.start() for m in re.finditer(r"\s", text)]
    if not boundaries:
        boundaries = [i for i in range(1, len(text))
                      if "\u3040" <= text[i] <= "\u9fff" and text[i] not in "、。，！？）」』"]
    if not boundaries:
        return text
    cut = min(boundaries, key=lambda i: abs(i - midpoint))
    return text[:cut].rstrip() + "\n" + text[cut:].lstrip()


def export_subtitles(project: Project, destination: Path, *, wrap_lines: bool = False) -> None:
    project.validate()
    destination = Path(destination)
    ass = destination.suffix.lower() == ".ass"
    if destination.suffix.lower() not in {".ass", ".srt"}:
        raise ValueError("請選擇 .ass 或 .srt")
    if not project.cues:
        raise ValueError("沒有可匯出的字幕")
    lines = []
    if ass:
        lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080",
                 "WrapStyle: 2", "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
                 "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                 "Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1",
                 "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    cues = sorted(project.cues, key=lambda c: c.start_ms)
    for i, cue in enumerate(cues, 1):
        # Shared rounding keeps touching boundaries touching. Never extend past EOF.
        start, end = output_bounds(project, cue, ass=ass)
        if end <= start:
            raise ValueError(f"第 {i} 句在輸出精度下時長為零，請調整")
        # Single-line export by default; preserve intentional manual line breaks.
        text = wrap_text(cue.text) if wrap_lines else "\n".join(" ".join(line.split()) for line in cue.text.splitlines())
        if ass:
            lines.append(f"Dialogue: 0,{stamp(start, True)},{stamp(end, True)},Default,,0,0,0,,{ass_text(text)}")
        else:
            # SRT players can parse HTML-like tags; export literal text safely.
            text = text.replace("<", "＜").replace(">", "＞")
            lines.extend([str(i), f"{stamp(start)} --> {stamp(end)}", text, ""])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
