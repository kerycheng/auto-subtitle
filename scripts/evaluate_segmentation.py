"""Read-only Japanese segmentation diagnosis with reviewable candidate exports."""
import argparse
import copy
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from autosub.comparison import ComparisonSettings, compare_file, summary
from autosub.domain import Project, word_key
from autosub.evaluation import read_ass
from autosub.export import export_subtitles
from autosub.lexical_segmentation import STRATEGY, boundary_map, lexical_groups, segment_lexically
from autosub.timing import apply_timing


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(project):
    words = sorted((w for w in project.words if w.text.strip()), key=lambda w: (w.start_ms, w.end_ms))
    mapping = boundary_map(words)
    keys = [word_key(w) for w in words]
    locations = {key: i for i, key in enumerate(keys)}
    boundaries = []
    for cue in project.cues:
        if not cue.source_tokens or cue.source_tokens[-1] not in locations:
            continue
        i = locations[cue.source_tokens[-1]] + 1
        if i == len(words):
            continue
        offset = mapping.offsets[i]
        lexeme = next((m for m in mapping.lexemes if m["start"] < offset < m["end"]), None)
        if lexeme or i not in mapping.allowed:
            boundaries.append({"token": i, "cue_id": cue.id, "cue_text": cue.text,
                "next_text": "".join(w.text for w in words[i:i + 8]),
                "inside_lexeme": lexeme["text"] if lexeme else None,
                "reason": "inside-dictionary-unit" if lexeme else "separated-attached-grammar"})
    owned = Counter(token for cue in project.cues for token in (cue.source_tokens or []))
    return {"words": len(words), "characters": sum(len(w.text) for w in words),
        "cues": len(project.cues), "all_tokens_owned_once": owned == Counter(keys),
        "ambiguous_duplicate_word_keys": len(keys) != len(set(keys)),
        "dictionary_internal_cuts": sum(b["inside_lexeme"] is not None for b in boundaries),
        "protected_boundary_violations": len(boundaries), "violations": boundaries,
        "short_cues": sum(c.end_ms - c.start_ms < 500 for c in project.cues),
        "long_cues": sum(c.end_ms - c.start_ms > project.settings.max_duration_ms for c in project.cues),
        "overlaps": sum(a.end_ms > b.start_ms for a, b in zip(project.cues, project.cues[1:])),
        "first_word_ms": words[0].start_ms if words else None,
        "note": "Dictionary constraint checks, not independently measured semantic or acoustic accuracy."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--generated-ass", type=Path, required=True)
    parser.add_argument("--alternate-project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Project.load(args.project)
    paths = {str(p.resolve()): p for p in (args.project, args.reference, args.generated_ass,
                                         Path(project.media_path), Path(project.audio_path))}
    if args.alternate_project:
        paths[str(args.alternate_project.resolve())] = args.alternate_project
    before = {key: sha(p) for key, p in paths.items()}
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("請選空白輸出目錄")
    args.output.mkdir(parents=True, exist_ok=True)
    reference, _ = read_ass(args.reference)
    exported, _ = read_ass(args.generated_ass)
    variants = [("A-original", project, False), ("B-lexical-only", project, True)]
    if args.alternate_project:
        alternate = Project.load(args.alternate_project)
        if alternate.source_fingerprint != project.source_fingerprint or alternate.speech != project.speech:
            raise ValueError("對照辨識必須使用同一來源與外部 Silero VAD")
        variants.extend([("C-no-vad-baseline", alternate, False), ("D-no-vad-lexical", alternate, True)])
    report = {"scope": "one user-provided Japanese development clip; not held-out",
        "strategy": STRATEGY, "app_default_changed": False, "asr_reference_input": False,
        "human_completion_rate": None, "source_hashes_before": before,
        "reference_events": len(reference), "provided_ass_events": len(exported),
        "provided_ass_text_matches_project": [c.text for c in exported] == [c.text for c in project.cues],
        "variants": {}}
    copies = {}
    for label, source, lexical in variants:
        candidate = copy.deepcopy(source)
        if lexical:
            candidate.cues = apply_timing(candidate, segmenter=segment_lexically)
        assert candidate.words == source.words
        locked_values = lambda p: [{k: v for k, v in vars(c).items() if k != "flags"}
                                  for c in p.cues if c.locked]
        assert locked_values(candidate) == locked_values(source)
        candidate.validate()
        measured = audit(candidate)
        if not measured["all_tokens_owned_once"]:
            raise ValueError(f"{label}: 辨識詞遺失／重複，停止匯出")
        measured["time_pairing_only"] = summary(compare_file(candidate, args.reference, ComparisonSettings()))
        report["variants"][label] = measured
        if lexical:
            _, decisions, _ = lexical_groups(source)
            measured["decisions"] = decisions
        candidate.save(args.output / (label + ".autosub.json"))
        for ext in (".ass", ".srt"):
            export_subtitles(candidate, args.output / (label + ext))
        copies[label] = candidate
    report["source_hashes_after"] = {key: sha(p) for key, p in paths.items()}
    report["sources_unchanged"] = report["source_hashes_after"] == before
    if not report["sources_unchanged"]:
        raise RuntimeError("原始檔案發生變化，請停止測試")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    esc = lambda s: html.escape(str(s))
    body = ['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>日文分段診斷</title>',
        '<style>body{font:16px system-ui;margin:28px;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}th{background:#edf1f7}pre{white-space:pre-wrap}.bad{color:#b21}</style>',
        '<h1>日文分段診斷與候選</h1><p>私人開發素材；人工成品是中文翻譯，只作時間參考，不作日文正確率標準。',
        'B 只改分段；C/D 改變 Whisper 內部 VAD 過濾，辨識文字也不同。所有候選皆未通過人工聲學／語意驗收，正式程式預設未變。</p>',
        '<p>A：提供的原專案　B：原 ASR＋完整詞語　C：停用內部 VAD＋舊分段　D：停用內部 VAD＋完整詞語。</p>',
        '<table><tr><th>版本</th><th>辨識詞／字幕句數</th><th>詞內切開</th><th>詞內／助詞等切點違規</th><th>短句／重疊</th><th>首詞時間</th></tr>']
    for label, row in report["variants"].items():
        vals = [label, f'{row["words"]} / {row["cues"]}', row["dictionary_internal_cuts"], row["protected_boundary_violations"],
                f'{row["short_cues"]} / {row["overlaps"]}', row["first_word_ms"]]
        body.append('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in vals) + '</tr>')
    body.append('</table><p>詞內切開為同一斷詞器的約束檢查；候選零違規是程式保護條件，不代表語意切句全對。停用 VAD 可能增加無語音處幻覺，不可直接推廣為預設。</p>')
    body.append('<h2>原版可疑切點</h2><table><tr><th>前句</th><th>後續原文</th><th>被切開詞／原因</th></tr>')
    for row in report["variants"]["A-original"]["violations"]:
        body.append('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in
            (row["cue_text"], row["next_text"], row["inside_lexeme"] or row["reason"])) + '</tr>')
    body.append('</table><h2>按人工字幕時間查看各版</h2><p>僅列時間相交事件，不表示文字等義；時間以秒表示。</p><table><tr><th>人工時間／中文</th>' +
                ''.join('<th>' + esc(label) + '</th>' for label in copies) + '</tr>')
    for cue in reference:
        body.append(f'<tr><td>{cue.start_ms/1000:.2f}–{cue.end_ms/1000:.2f}<br>{esc(cue.text)}</td>')
        for p in copies.values():
            overlaps = [c for c in p.cues if c.start_ms < cue.end_ms and c.end_ms > cue.start_ms]
            body.append('<td>' + '<br>'.join(f'{c.start_ms/1000:.2f}–{c.end_ms/1000:.2f} {esc(c.text)}' for c in overlaps) + '</td>')
        body.append('</tr>')
    body.append('</table><h2>各版完整原文</h2>')
    for label, p in copies.items():
        body.append('<h3>' + esc(label) + '</h3><pre>' + esc('\n'.join(
            f'{c.start_ms/1000:.2f}–{c.end_ms/1000:.2f} {c.text}' for c in p.cues)) + '</pre>')
    body.append('</html>')
    (args.output / "index.html").write_text(''.join(body), encoding="utf-8")
    print(json.dumps({"sources_unchanged": report["sources_unchanged"], "variants": {
        k: {name: v[name] for name in ("words", "cues", "dictionary_internal_cuts", "protected_boundary_violations", "short_cues", "overlaps")}
        for k, v in report["variants"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
