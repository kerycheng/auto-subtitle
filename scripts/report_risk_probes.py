"""Review-only competing ASR hypotheses; disagreement never changes subtitles."""
import argparse
import html
import json
from pathlib import Path

from autosub.alignment_lab import file_hash, normalized


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cases',type=Path)
    parser.add_argument('whisper',type=Path)
    parser.add_argument('qwen',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Please select a fresh output directory')
    cases=json.loads(args.cases.read_text(encoding='utf-8'))
    manifest=args.cases.with_name('manifest.json')
    payload=json.loads(manifest.read_text(encoding='utf-8'))
    assert not payload['text_provided'] and file_hash(payload['audio_path'])==payload['audio_sha256']
    reports={name:json.loads(path.read_text(encoding='utf-8')) for name,path in [('whisper',args.whisper),('qwen',args.qwen)]}
    ids={case['id'] for case in cases}
    for name,report in reports.items():
        assert report['backend']==name and report['status']=='complete'
        assert report['manifest_sha256']==file_hash(manifest) and not report['text_provided']
        assert len(report['results'])==len(ids) and {r['id'] for r in report['results']}==ids
    for case in cases:
        assert file_hash(case['audio_path'])==case['audio_sha256']
    args.output.mkdir(parents=True)
    rows=[]
    windows={w['id']:w for w in payload['windows']}
    for case in cases:
        entries={name:next(r for r in report['results'] if r['id']==case['id']) for name,report in reports.items()}
        assert all(r['window_ms']==windows[case['id']]['window_ms'] for r in entries.values())
        rows.append({**case,'whisper':entries['whisper']['text'],'qwen':entries['qwen']['text'],
                     'different_normalized_text':normalized(entries['whisper']['text'])!=normalized(entries['qwen']['text']),
                     'decision':'unreviewed','adopted':False})
    result={'scope':'targeted development risks; no reference text or independent correctness judgement',
            'rows':rows,'normalized_disagreements':sum(r['different_normalized_text'] for r in rows),
            'source_audio_unchanged':True,'subtitles_modified':False,'accuracy':None}
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    esc=lambda value:html.escape(str(value))
    audio_url=Path(payload['audio_path']).relative_to(args.output.parent.resolve()).as_posix()
    body=['<!doctype html><meta charset="utf-8"><title>漏辨風險交叉檢查</title>',
          '<style>body{font:16px system-ui;margin:24px;line-height:1.7}td,th{border:1px solid #bbb;padding:9px;vertical-align:top}table{border-collapse:collapse;width:100%}</style>',
          '<h1>局部風險：相同音訊兩個辨識器</h1>',
          '<p>這些是機械條件選出的試聽位置，不是已確認漏詞。兩模型在完全相同的視窗獨立辨識，未供應既有字幕或人工文字。不同結果僅列為待複核，不自動補字或替換字幕。</p>',
          '<p>原流程使用較長上下文；其相交句子可能包含視窗外文字，僅供定位。短視窗可能引入新的幻覺；模型共識也不等於正確。所有判定仍未複核。</p>',
          '<audio id="audio" controls preload="none" src="../'+esc(audio_url)+'"></audio>',
          '<table><tr><th>片段／來源目標</th><th>原流程相交句子</th><th>Whisper 相同窗</th><th>Qwen 相同窗</th><th>判定</th></tr>']
    for row in rows:
        a,b=windows[row['id']]['window_ms']
        label=row['sample']+' '+str(row['target_ms'])+' ms'
        body.append('<tr><td>'+esc(label)+f'<br><button onclick="const a=document.getElementById(\'audio\');a.currentTime={a/1000};a.play()">試聽上下文</button></td>'+
                    ''.join('<td>'+esc(row[k])+'</td>' for k in ['primary_context_text','whisper','qwen'])+'<td>未複核／未採用</td></tr>')
    (args.output/'index.html').write_text(''.join(body)+'</table>',encoding='utf-8')
    print(json.dumps({'cases':len(rows),'different':result['normalized_disagreements'],'modified':False}))


if __name__=='__main__':
    main()
