"""Non-destructive review of the clocks actually written by subtitle export."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from .domain import Project
from .export import export_subtitles, output_bounds, stamp
from .file_io import replace_checkpoint


def review_export(project: Project, *, ass: bool) -> dict:
    project.validate()
    scale = 10 if ass else 1
    rows = []
    findings = []
    for number, cue in enumerate(sorted(project.cues, key=lambda c: c.start_ms), 1):
        start, end = output_bounds(project, cue, ass=ass)
        start, end = start * scale, end * scale
        rows.append((number, start, end))
        codes = []
        if end <= start:
            codes.append("nonpositive")
        elif end - start < project.settings.min_duration_ms:
            codes.append("short")
        if end - start > project.settings.max_duration_ms:
            codes.append("long")
        if not cue.text.strip():
            codes.append("empty")
        if any(marker in cue.text for marker in ("聽不清", "聞き取り不明", "待校")):
            codes.append("uncertain")
        for code in codes:
            findings.append({"code": code, "rows": [number], "start_ms": start, "end_ms": end})
    # Connected overlap groups, rather than quadratically many pair warnings.
    group = []
    group_start = group_end = 0
    for row, start, end in rows + [(0, float("inf"), float("inf"))]:
        if group and start >= group_end:
            if len(group) > 1:
                findings.append({"code": "overlap", "rows": group,
                                 "start_ms": group_start, "end_ms": group_end})
            group = []
        if not group:
            group_start = start
        group.append(row)
        group_end = max(group_end, end)
    return {"schema_version": 1, "format": "ass" if ass else "srt", "cue_count": len(rows),
            "recognition_complete": project.status == "complete",
            "min_duration_ms": project.settings.min_duration_ms,
            "max_duration_ms": project.settings.max_duration_ms, "findings": findings}


def render_review(report: dict, digest: str) -> str:
    labels = {"short": "短於最短顯示時長", "long": "超過分段時長上限", "empty": "空白文字",
              "uncertain": "含待確認文字", "overlap": "時間重疊群組（可能是同時發話）",
              "nonpositive": "輸出時長不大於零"}
    lines = ["字幕匯出檢查", f"字幕 SHA256：{digest}",
             f"格式：{report['format'].upper()}；字幕：{report['cue_count']} 句",
             f"檢查門檻：最短 {report['min_duration_ms']} ms；最長 {report['max_duration_ms']} ms",
             "序號依匯出順序，時間依實際輸出精度；未修改文字或時間。",
             "此報告不判定辨識、翻譯或說話者正確性，也不代表已聽校。", ""]
    if not report["recognition_complete"]:
        lines.append("注意：辨識尚未完成，這是部分字幕匯出。")
    for finding in report["findings"]:
        rows = ", ".join(map(str, finding["rows"]))
        lines.append(f"第 {rows} 句 | {stamp(finding['start_ms'])} → {stamp(finding['end_ms'])} | {labels[finding['code']]}")
    if not report["findings"]:
        lines.append("未發現上述時長、空白、待確認標記與重疊問題；仍需配合原音校對。")
    lines.append("短軸可能受鄰句或媒體結尾限制，請在 Aegisub 檢查；不要直接延長而覆蓋其他發話。")
    return "\n".join(lines) + "\n"


def export_with_review(project: Project, destination: Path, *, wrap_lines: bool = False) -> tuple[dict, Path, str | None]:
    """Export first; report failure must not misrepresent a successful subtitle write.

    A uniquely named temporary preserves an old report if replacement fails.
    The digest identifies stale reports if a later export cannot replace them.
    """
    destination = Path(destination)
    export_subtitles(project, destination, wrap_lines=wrap_lines)
    report = review_export(project, ass=destination.suffix.lower() == ".ass")
    companion = destination.with_name(destination.name + ".review.txt")
    try:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        content = render_review(report, digest)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", suffix=".tmp",
                                         prefix=companion.name + ".", dir=companion.parent,
                                         delete=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        replace_checkpoint(temporary, companion)
    except OSError as exc:
        return report, companion, str(exc)
    return report, companion, None
