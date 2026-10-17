"""Read-only original-language reference report, including explicit SRT display artifacts."""
import argparse
import html
import json
from dataclasses import asdict
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.reference_text import read_srt_events, reconstruct_overlap_bridges, compare_text_reference
from autosub.timing import apply_timing


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retime', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    p = Project.load(args.project)
    if not p.native_complete:
        raise ValueError('Expected a completed native project')
    if args.retime:
        p.cues = apply_timing(p)
    events = read_srt_events(args.reference)
    reconstructed, trace = reconstruct_overlap_bridges(events)
    report = {'project_sha256': file_hash(args.project), 'reference_sha256': file_hash(args.reference),
        'retimed_with_current_code': args.retime, 'raw': compare_text_reference(p, events),
        'inferred_overlap_view': compare_text_reference(p, reconstructed), 'overlap_trace': trace,
        'raw_events': events, 'inferred_events': reconstructed, 'cues': [asdict(c) for c in p.cues],
        'reference_role': 'user supplied corrected Japanese; display times not independently verified acoustic boundaries',
        'normalization': 'NFKC/casefold/remove whitespace and punctuation; script, numeral and name spellings remain distinct'}
    args.output.mkdir(parents=True)
    for suffix in ('.ass', '.srt'):
        export_subtitles(p, args.output / ('generated'+suffix))
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    body = ['<!doctype html><meta charset="utf-8"><title>原語參考比較</title>',
        '<style>body{font:16px system-ui;margin:28px;line-height:1.6}table{border-collapse:collapse;width:100%}td{border:1px solid #aaa;padding:8px;max-width:550px;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>',
        '<h1>原語參考與生成字幕</h1><p>原文只供辨識完成後評估，未供應辨識器。括注、字形、片假名／英文人名與遊戲名稱仍影響機械 CER；',
        '參考未涵蓋的聲音也可能被辨識，字元差異不能直接算成漏詞／幻覺。人工顯示時間不是字詞發聲真值。</p>',
        '<p>原始參考 CER：' + f'{report["raw"]["mechanical_cer"]:.2%}' + '；推導重疊顯示重建 CER：' +
        f'{report["inferred_overlap_view"]["mechanical_cer"]:.2%}' + '。後者只是明確標示的次要評估視圖，沒有改寫 SRT。</p>',
        '<h2>結構重建紀錄（未人工確認）</h2><pre>'+html.escape(json.dumps(trace, ensure_ascii=False, indent=2))+'</pre>',
        '<h2>句首跨軸候選（需試聽核對）</h2><pre>'+html.escape(json.dumps(report['inferred_overlap_view']['prefix_carry_candidates'],ensure_ascii=False,indent=2))+'</pre>',
        '<h2>前句句尾跑到後軸的候選（需核對）</h2><pre>'+html.escape(json.dumps(report['inferred_overlap_view']['previous_suffix_carry_candidates'],ensure_ascii=False,indent=2))+'</pre>',
        '<h2>生成字幕與時間相交的參考</h2><table>']
    for cue in p.cues:
        intersect = [e for e in reconstructed if e['start_ms'] < cue.end_ms and e['end_ms'] > cue.start_ms]
        body.append('<tr><td>'+html.escape(str([cue.start_ms,cue.end_ms])+' '+cue.text)+'</td><td>'+html.escape(
            '\n'.join(str(e['source_rows'])+' '+e['text'] for e in intersect))+'</td></tr>')
    (args.output / 'index.html').write_text(''.join(body)+'</table>', encoding='utf-8')
    print(json.dumps({'raw_cer':report['raw']['mechanical_cer'], 'inferred_cer':report['inferred_overlap_view']['mechanical_cer'],
                     'overlap_bridges':len(trace), 'prefix_candidates':len(report['inferred_overlap_view']['prefix_carry_candidates'])}))


if __name__ == '__main__':
    main()
