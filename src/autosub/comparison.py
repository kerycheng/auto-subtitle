"""Post-ASR comparison with human finished subtitles, without using their text in ASR.

Temporal pairing is a review suggestion. It does not establish semantic equivalence
between translated subtitles and recognized speech, or acoustic ground truth.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from threading import Event

import numpy as np

from .domain import Project
from .evaluation import ReferenceCue, read_ass
from .media import check_cancel, fingerprint

ALGORITHM = "temporal-groups-v1"
DECISIONS = {"unreviewed", "usable", "adjust", "split_merge", "redo", "wrong_pair", "exclude"}


@dataclass
class ComparisonSettings:
    reference_offset_ms: int = 0  # reference clock minus this offset = project clock
    tolerance_ms: int = 100
    max_group: int = 3
    max_group_gap_ms: int = 600
    timeline_confirmed: bool = False

    def validate(self):
        for name in ("reference_offset_ms", "tolerance_ms", "max_group", "max_group_gap_ms"):
            if type(getattr(self, name)) is not int:
                raise ValueError("比較時間參數必須為整數")
        if not 0 <= self.tolerance_ms <= 1000 or not 1 <= self.max_group <= 3 or not 0 <= self.max_group_gap_ms <= 2000:
            raise ValueError("比較參數超出範圍")


def project_signature(project: Project):
    data = {"source": project.source_fingerprint, "media": project.media_path,
            "duration_ms": project.duration_ms, "track": project.audio_stream,
            "cues": [[c.id, c.start_ms, c.end_ms, c.text] for c in project.cues]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_reference(path: Path, exclude_styles=("註解", "注释", "注釋")):
    path = Path(path)
    if path.suffix.lower() == ".ass":
        return read_ass(path, exclude_styles)
    if path.suffix.lower() != ".srt":
        raise ValueError("人工成品目前支援 ASS 或 SRT")
    result = []
    pattern = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.S)
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    for match in pattern.finditer(text):
        nums = [int(x) for x in match.groups()[:8]]
        a = ((nums[0] * 60 + nums[1]) * 60 + nums[2]) * 1000 + nums[3]
        b = ((nums[4] * 60 + nums[5]) * 60 + nums[6]) * 1000 + nums[7]
        content = re.sub(r"<[^>]*>", "", match.group(9)).strip().replace("\n", " ")
        if a < b and content:
            result.append(ReferenceCue(a, b, content, "Default"))
    if not result:
        raise ValueError("人工字幕沒有可讀取的有效事件")
    return sorted(result, key=lambda c: c.start_ms), ""


def _union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def _coverage(left, right):
    left, right = _union(left), _union(right)
    overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in left for c, d in right)
    lsize, rsize = sum(b - a for a, b in left), sum(b - a for a, b in right)
    return overlap / lsize if lsize else 0, overlap / rsize if rsize else 0


def _block(events, end, count, settings):
    if count > end:
        return None
    group = events[end - count:end]
    if count > 1 and (max(x["end_ms"] for x in group) - group[0]["start_ms"] > 12000 or any(
            b["start_ms"] - a["end_ms"] > settings.max_group_gap_ms
            for a, b in zip(group, group[1:]))):
        return None
    return group


def _cost(ref, gen):
    a, b = ref[0]["start_ms"], max(x["end_ms"] for x in ref)
    c, d = gen[0]["start_ms"], max(x["end_ms"] for x in gen)
    if min(b, d) <= max(a, c):
        return None  # timing alone cannot pair two non-overlapping utterances
    rc, gc = _coverage([(x["start_ms"], x["end_ms"]) for x in ref],
                       [(x["start_ms"], x["end_ms"]) for x in gen])
    if min(rc, gc) < .25:
        return None
    return (abs(a - c) + abs(b - d)) / 1000 + (2 - rc - gc) + .35 * (len(ref) + len(gen) - 2)


def make_row(ref, gen, settings, manual=False):
    start_delta = gen[0]["start_ms"] - ref[0]["start_ms"] if ref and gen else None
    end_delta = max(g["end_ms"] for g in gen) - max(r["end_ms"] for r in ref) if ref and gen else None
    overlaps = any(x.get("overlap", False) for x in ref + gen)
    rc, gc = _coverage([(x["start_ms"], x["end_ms"]) for x in ref],
                       [(x["start_ms"], x["end_ms"]) for x in gen])
    individual_coverage = [
        _coverage([(r["start_ms"], r["end_ms"])], [(g["start_ms"], g["end_ms"]) for g in gen])[0]
        for r in ref] + [
        _coverage([(g["start_ms"], g["end_ms"])], [(r["start_ms"], r["end_ms"]) for r in ref])[0]
        for g in gen]
    ambiguous = overlaps or (bool(ref and gen) and min(individual_coverage, default=0) < .5)
    kind = "unmatched_reference" if not gen else "unmatched_generated" if not ref else f"{len(ref)}:{len(gen)}"
    within = (start_delta is not None and abs(start_delta) <= settings.tolerance_ms
              and abs(end_delta) <= settings.tolerance_ms)
    ids = [[x["id"] for x in ref], [x["id"] for x in gen]]
    row_id = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:20]
    return {"id": row_id, "reference_ids": ids[0], "generated_ids": ids[1], "kind": kind,
            "reference_text": " ｜ ".join(x["text"] for x in ref),
            "generated_text": " ｜ ".join(x["text"] for x in gen),
            "reference_span": [ref[0]["start_ms"], max(x["end_ms"] for x in ref)] if ref else None,
            "generated_span": [gen[0]["start_ms"], max(x["end_ms"] for x in gen)] if gen else None,
            "start_delta_ms": start_delta, "end_delta_ms": end_delta,
            "reference_coverage": rc, "generated_coverage": gc, "ambiguous": ambiguous,
            "time_within_tolerance": within,
            "strict_time_estimate": kind == "1:1" and within and not ambiguous,
            "pairing": "manual" if manual else "temporal_suggestion", "decision": "unreviewed",
            "reviewer": "", "notes": ""}


def _mark_overlaps(events):
    active = []
    for event in events:
        active = [x for x in active if x["end_ms"] > event["start_ms"]]
        event["overlap"] = bool(active)
        for other in active:
            other["overlap"] = True
        active.append(event)


def compare(project: Project, references: list[ReferenceCue], settings=None,
            cancel: Event | None = None, progress=lambda *_: None) -> dict:
    settings = settings or ComparisonSettings()
    settings.validate()
    project.validate()
    cancel = cancel or Event()
    refs, clipped = [], 0
    for i, ref in enumerate(references):
        a, b = ref.start_ms - settings.reference_offset_ms, ref.end_ms - settings.reference_offset_ms
        if b <= 0 or a >= project.duration_ms:
            continue
        if a < 0 or b > project.duration_ms:
            clipped += 1
            continue  # do not manufacture a good match at the evaluation window edge
        refs.append({"id": f"ref-{i}", "start_ms": a, "end_ms": b, "text": ref.text, "style": ref.style})
    refs.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    gen = [{"id": c.id, "start_ms": c.start_ms, "end_ms": c.end_ms, "text": c.text}
           for c in sorted(project.cues, key=lambda c: (c.start_ms, c.end_ms))]
    _mark_overlaps(refs)
    _mark_overlaps(gen)
    n, m = len(refs), len(gen)
    if (n + 1) * (m + 1) > 4_000_000:
        raise ValueError("字幕事件過多，請先用較短素材或排除非對話樣式")
    scores = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.uint8)
    scores[:, 0] = np.arange(n + 1) * 1.1
    scores[0, :] = np.arange(m + 1) * 1.1
    steps[1:, 0], steps[0, 1:] = 16, 17
    ref_blocks = {(i, k): _block(refs, i, k, settings) for i in range(1, n + 1) for k in range(1, settings.max_group + 1)}
    gen_blocks = {(j, k): _block(gen, j, k, settings) for j in range(1, m + 1) for k in range(1, settings.max_group + 1)}
    shapes = [(1, 1)] + [(1, k) for k in range(2, settings.max_group + 1)] + [(k, 1) for k in range(2, settings.max_group + 1)]
    for i in range(1, n + 1):
        check_cancel(cancel)
        progress(round(i * 100 / max(1, n)), f"建立時間對應 {i}/{n}")
        for j in range(1, m + 1):
            skip_ref, skip_gen = scores[i - 1, j] + 1.1, scores[i, j - 1] + 1.1
            best, step = (skip_ref, 16) if skip_ref <= skip_gen else (skip_gen, 17)
            for a, b in shapes:
                if a > i or b > j:
                    continue
                left, right = ref_blocks[i, a], gen_blocks[j, b]
                if left is None or right is None:
                    continue
                cost = _cost(left, right)
                if cost is not None and scores[i - a, j - b] + cost < best:
                    best, step = scores[i - a, j - b] + cost, a * 4 + b
            scores[i, j], steps[i, j] = best, step
    check_cancel(cancel)
    rows, i, j = [], n, m
    while i or j:
        step = steps[i, j]
        if step == 16:
            rows.append(make_row(refs[i - 1:i], [], settings))
            i -= 1
        elif step == 17:
            rows.append(make_row([], gen[j - 1:j], settings))
            j -= 1
        else:
            a, b = divmod(int(step), 4)
            rows.append(make_row(refs[i - a:i], gen[j - b:j], settings))
            i, j = i - a, j - b
    rows.reverse()
    return {"schema_version": 1, "algorithm": ALGORITHM, "project_signature": project_signature(project),
            "settings": asdict(settings), "duration_ms": project.duration_ms,
            "source_fingerprint": project.source_fingerprint, "audio_stream": project.audio_stream,
            "reference_events": refs, "generated_events": gen, "clipped_reference_events_excluded": clipped,
            "reference_used_by_asr": False, "rows": rows, "review_active_seconds": 0.0}


def compare_file(project, path, settings=None, cancel=None, progress=lambda *_: None, exclude_styles=("註解", "注释", "注釋")):
    path = Path(path)
    before = fingerprint(path)
    reference_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if project.source_fingerprint and Path(project.media_path).is_file() and fingerprint(Path(project.media_path)) != project.source_fingerprint:
        raise ValueError("來源媒體已改變，請先確認專案來源")
    references, video = read_reference(path, exclude_styles)
    result = compare(project, references, settings, cancel, progress)
    if before != fingerprint(path) or hashlib.sha256(path.read_bytes()).hexdigest() != reference_hash:
        raise ValueError("比較期間人工字幕已變更，請重新比較")
    result.update(reference_fingerprint=before, reference_sha256=reference_hash,
                  reference_video_hint=video, excluded_styles=list(exclude_styles))
    return result


def summary(session):
    rows = session["rows"]
    total = len(session["reference_events"])
    counts = Counter(row["kind"] for row in rows)
    decisions = Counter(row["decision"] for row in rows)
    paired = [r for r in rows if r["reference_ids"] and r["generated_ids"]]
    time_hits = sum(len(r["reference_ids"]) for r in rows if r["strict_time_estimate"])
    # Excluded and unreviewed events stay in the full denominator. Report counts,
    # not an inflated percentage computed only on the convenient matched rows.
    confirmed_usable = sum(len(r["reference_ids"]) for r in rows if r["decision"] == "usable")
    reviewed = sum(len(r["reference_ids"]) for r in rows if r["decision"] != "unreviewed")
    boundary_rates = {f"{t}ms": sum(len(r["reference_ids"]) for r in paired
        if not r["ambiguous"] and abs(r["start_delta_ms"]) <= t and abs(r["end_delta_ms"]) <= t) / total
        if total else None for t in (50, 100, 250, 500)}
    start_errors = sorted(abs(r["start_delta_ms"]) for r in paired)
    end_errors = sorted(abs(r["end_delta_ms"]) for r in paired)
    return {"reference_events": total, "generated_events": len(session["generated_events"]),
            "mapping_counts": dict(counts), "decision_counts": dict(decisions),
            "clipped_reference_events_excluded": session["clipped_reference_events_excluded"],
            "timeline_confirmed": session["settings"]["timeline_confirmed"],
            "strict_time_estimate_rate": time_hits / total if total else None,
            "group_outer_boundary_estimate_rates": boundary_rates,
            "human_reviewed_reference_events": reviewed,
            "human_confirmed_usable_rate": confirmed_usable / total if total and reviewed and session["settings"]["timeline_confirmed"] else None,
            "review_active_seconds": session.get("review_active_seconds", 0),
            "start_absolute_error_median_ms": median(abs(r["start_delta_ms"]) for r in paired) if paired else None,
            "end_absolute_error_median_ms": median(abs(r["end_delta_ms"]) for r in paired) if paired else None,
            "start_absolute_error_p90_ms": start_errors[math.ceil(len(start_errors) * .9) - 1] if start_errors else None,
            "end_absolute_error_p90_ms": end_errors[math.ceil(len(end_errors) * .9) - 1] if end_errors else None,
            "note": "時間配對估計不是語意配對或聲學精度；人工可用率僅累計已確認項目，未複核項目保留於分母"}


def set_decision(session, index, decision, reviewer, notes=""):
    if decision not in DECISIONS or decision == "unreviewed" or not reviewer.strip():
        raise ValueError("請填複核者並選擇有效的判定")
    if not session["settings"]["timeline_confirmed"]:
        raise ValueError("請先確認影片版本、音軌與時間偏移，再記錄人工判定")
    if decision == "exclude" and not notes.strip():
        raise ValueError("排除項目請記錄原因")
    row = session["rows"][index]
    if decision == "usable" and not (row["reference_ids"] and row["generated_ids"]):
        raise ValueError("未配對區間不能標記為可直接使用；請先修正對應")
    row.update(decision=decision, reviewer=reviewer.strip(), notes=notes,
               reviewed_at_utc=datetime.now(timezone.utc).isoformat())


def repair_pair(session, reference_ids, generated_ids):
    """Explicit human regrouping; displaced events remain as unmatched rows."""
    refs = {x["id"]: x for x in session["reference_events"]}
    gens = {x["id"]: x for x in session["generated_events"]}
    if not reference_ids or not generated_ids or not set(reference_ids) <= refs.keys() or not set(generated_ids) <= gens.keys():
        raise ValueError("請各選至少一個有效的人工與自動事件")
    reference_ids, generated_ids = set(reference_ids), set(generated_ids)
    affected = [r for r in session["rows"] if set(r["reference_ids"]) & reference_ids or set(r["generated_ids"]) & generated_ids]
    rows = [r for r in session["rows"] if r not in affected]
    settings = ComparisonSettings(**session["settings"])
    for rid in {x for r in affected for x in r["reference_ids"]} - reference_ids:
        rows.append(make_row([refs[rid]], [], settings))
    for gid in {x for r in affected for x in r["generated_ids"]} - generated_ids:
        rows.append(make_row([], [gens[gid]], settings))
    chosen_ref = sorted((refs[x] for x in reference_ids), key=lambda x: x["start_ms"])
    chosen_gen = sorted((gens[x] for x in generated_ids), key=lambda x: x["start_ms"])
    rows.append(make_row(chosen_ref, chosen_gen, settings, manual=True))
    session["rows"] = sorted(rows, key=lambda r: (r["reference_span"] or r["generated_span"])[0])


def save_session(session, path):
    path = Path(path)
    protected = [session.get("reference_fingerprint", {}).get("path"), session.get("source_fingerprint", {}).get("path")]
    if any(p and path.resolve() == Path(p).resolve() for p in protected):
        raise ValueError("比較結果不能覆寫來源媒體或人工字幕")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {**session, "summary": summary(session)}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_session(path, project, reference_path=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("algorithm") != ALGORITHM:
        raise ValueError("不支援的比較報告版本")
    if raw["project_signature"] != project_signature(project):
        raise ValueError("專案字幕或來源已變更，請另建比較，避免沿用舊判定")
    source = project.source_fingerprint
    if source and Path(project.media_path).is_file() and fingerprint(Path(project.media_path)) != source:
        raise ValueError("專案媒體檔已改變，拒絕沿用舊比較")
    ref = str(reference_path) if reference_path is not None else raw.get("reference_fingerprint", {}).get("path")
    if ref and (not Path(ref).is_file() or hashlib.sha256(Path(ref).read_bytes()).hexdigest() != raw["reference_sha256"]):
        raise ValueError("人工參考檔缺失或已改變，拒絕沿用舊判定")
    if reference_path is not None:
        raw["reference_fingerprint"] = fingerprint(Path(reference_path))
    validate_session(raw)
    return raw


def validate_session(session):
    for field, event_field in [("reference_ids", "reference_events"), ("generated_ids", "generated_events")]:
        expected = [x["id"] for x in session[event_field]]
        actual = [x for row in session["rows"] for x in row[field]]
        if Counter(actual) != Counter(expected) or len(set(expected)) != len(expected):
            raise ValueError("比較配對遺失或重複事件")
    for row in session["rows"]:
        if row["decision"] not in DECISIONS or (row["decision"] != "unreviewed" and not row["reviewer"].strip()):
            raise ValueError("人工判定資料無效")


def export_report(session, path):
    """Self-contained escaped HTML, or spreadsheet-safe UTF-8 CSV."""
    path = Path(path)
    if any(p and path.resolve() == Path(p).resolve() for p in [session.get("reference_fingerprint", {}).get("path"), session.get("source_fingerprint", {}).get("path")]):
        raise ValueError("報告不能覆寫來源檔案")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "reference_text", "generated_text", "start_delta_ms", "end_delta_ms", "ambiguous", "decision", "reviewer", "notes"]
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for row in session["rows"]:
                writer.writerow([("'" + str(row[k]) if isinstance(row[k], str) and row[k].lstrip().startswith(("=", "+", "-", "@")) else row[k]) for k in fields])
    elif path.suffix.lower() == ".html":
        esc = lambda x: html.escape(str(x))
        stats = summary(session)
        percent = lambda x: "尚無已確認結果" if x is None else f"{x:.1%}"
        decisions = {"unreviewed": "未複核", "usable": "可直接使用", "adjust": "需微調時間",
                     "split_merge": "需拆分／合併", "redo": "需重做／補句", "wrong_pair": "對應錯誤", "exclude": "排除"}
        headers = ["人工：自動", "人工時段 ms", "自動時段 ms", "人工字幕", "自動字幕", "起點差 ms", "終點差 ms", "提示", "人工判定", "複核者", "備註"]
        rendered = []
        for row in session["rows"]:
            kind = {"unmatched_reference": "人工有／自動未配對", "unmatched_generated": "自動有／人工未配對"}.get(row["kind"], row["kind"])
            values = [kind, row["reference_span"], row["generated_span"], row["reference_text"], row["generated_text"],
                      row["start_delta_ms"], row["end_delta_ms"], "重疊／配對不確定" if row["ambiguous"] else "待確認",
                      decisions[row["decision"]], row["reviewer"], row["notes"]]
            rendered.append("<tr>" + "".join(f"<td>{esc('—' if v is None else v)}</td>" for v in values) + "</tr>")
        rates = "、".join(f"{t}：{percent(rate)}" for t, rate in stats["group_outer_boundary_estimate_rates"].items())
        decision_counts = "、".join(f"{decisions[k]} {v} 組" for k, v in stats["decision_counts"].items())
        intro = (f"<p>時間基準：{'已由使用者確認' if stats['timeline_confirmed'] else '尚未確認，僅作初步比較'}；"
                 f"參考時間減去 {session['settings']['reference_offset_ms']}ms 後對應專案時間。</p>"
                 f"<p>人工 {stats['reference_events']} 句／自動 {stats['generated_events']} 句；"
                 f"另排除跨出比較視窗的人工事件 {stats['clipped_reference_events_excluded']} 句。</p>"
                 f"<p>一對一且頭尾均在 {session['settings']['tolerance_ms']}ms 容差內的估計："
                 f"<strong>{percent(stats['strict_time_estimate_rate'])}</strong>。"
                 f"已經人工確認可直接使用、占全體人工句數：<strong>{percent(stats['human_confirmed_usable_rate'])}</strong>。</p>"
                 f"<p>配對群組外側頭尾均在容差內（可含拆合句，仍需確認）：{esc(rates)}。</p>"
                 f"<p>已複核 {stats['human_reviewed_reference_events']} 句；{esc(decision_counts)}；"
                 f"主動複核計時 {stats['review_active_seconds']:.1f} 秒。</p>")
        path.write_text("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>人工成品比較</title>"
            "<style>body{font:16px sans-serif;margin:24px;color:#203044}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #bbc;padding:8px;vertical-align:top}th{background:#eaf0f8}td{min-width:65px}</style>"
            "<h1>人工成品比較</h1><p>正值表示自動軸較晚。時間配對為估計，不能當成語意或聲學精度。"
            "人工可用率只累計人工確認項目；未複核項目保留於分母。</p>"
            + intro + "<table><thead><tr>" + "".join(f"<th>{esc(k)}</th>" for k in headers) + "</tr></thead><tbody>" + "".join(rendered) + "</tbody></table></html>", encoding="utf-8")
    else:
        raise ValueError("報告支援 HTML 或 CSV")


def session_key(session):
    return hashlib.sha256(json.dumps([session["project_signature"], session["reference_sha256"],
        session["settings"]["reference_offset_ms"], session["settings"]["tolerance_ms"],
        session["excluded_styles"]], sort_keys=True).encode()).hexdigest()[:16]
