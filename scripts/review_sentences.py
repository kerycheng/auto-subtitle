"""Review-only sentence proposals; explicit apply/restore always writes new files."""
import argparse
import html
import json
import shutil
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.review_proposals import ALGORITHM, apply_proposal, proposals, restore_proposal, signature


def write_json(value, path):
    with Path(path).open('x',encoding='utf-8') as output:
        json.dump(value,output,ensure_ascii=False,indent=2,allow_nan=False)


def prepare_review(project, source, output):
    items=proposals(project)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    session={'schema_version':1,'algorithm':ALGORITHM,'source_project':str(Path(source).resolve()),
             'source_sha256':file_hash(source),'signature':signature(project),'items':items,
             'decisions':'unreviewed','human_completion_rate':None}
    write_json(session,output/'proposals.json')
    esc=lambda value:html.escape(str(value))
    audio=Path(project.audio_path)
    has_audio=bool(project.audio_path) and audio.is_file()
    if has_audio:
        shutil.copyfile(audio,output/'audition.wav')
    by_id={c.id:c for c in project.cues}
    body=['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>短句複核候選</title>',
          '<style>body{font:16px system-ui;margin:24px;line-height:1.7}td,th{border:1px solid #bbb;padding:10px;vertical-align:top}table{border-collapse:collapse;width:100%}code{word-break:break-all}</style>',
          '<h1>短句複核候選</h1><p>尚未採用任何候選。合併預覽只串接完整相鄰字幕，不能證明同一說話者或同一句話。短促回應、笑聲及專名可能應保持獨立。</p>',
          '<p>弱語音與未覆蓋 VAD 區間僅供試聽，不提供刪字／排除操作。先聽上下文，再使用既有修軸功能或明確採用指定合併候選。</p>',
          '<p>此頁只試聽與顯示預覽；候選與原專案綁定，採用後任何後續修改均會使直接撤回失效，以免覆寫人工修正。操作紀錄與專案需一起保留。</p>',
          '<audio id="audio" controls preload="none" src="audition.wav"></audio>' if has_audio else '<p>分析音訊不存在，無法在本頁試聽。</p>',
          '<table><tr><th>候選／來源時段</th><th>目前字幕</th><th>預覽／證據</th><th>狀態</th></tr>']
    labels={'merge':'合併預覽','listen':'短句／弱語音試聽','listen-gap':'語音時間空檔'}
    for number,item in enumerate(items,1):
        start,end=item['range_ms']
        before='<br>'.join(esc(by_id[i].text) for i in item['cue_ids']) or '語音時間空檔'
        preview=esc(item['preview']['text'])+'<br>' if item['kind']=='merge' else ''
        if item['kind']=='merge':
            preview+='估計發聲間隔：'+str(item['evidence']['gap_ms'])+'ms；說話者與語意仍需試聽確認。'
        elif item['kind']=='listen':
            preview+='語音偵測覆蓋 '+f"{item['evidence']['vad_coverage']:.0%}"+'；低覆蓋也可能是笑聲或時間對齊誤差。'
        else:
            preview+=esc(item['evidence']['reason'])
        preview+='<details><summary>詳細證據</summary><pre>'+esc(json.dumps(item['evidence'],ensure_ascii=False,indent=2))+'</pre></details>'
        audition=(f'<button onclick="const p=document.getElementById(\'audio\');p.currentTime={max(0,start-1000)/1000};p.play()">試聽上下文</button>' if has_audio else '')
        body.append('<tr><td>'+str(number)+'. '+labels[item['kind']]+
                    '<details><summary>操作識別碼</summary><code>'+esc(item['id'])+'</code></details>'+
                    f'<br>{start/1000:.2f}–{end/1000:.2f} 秒<br>'+audition+'</td><td>'+before+'</td><td>'+preview+
                    '</td><td>'+('未複核／未採用' if item['adoptable'] else '僅試聽提醒')+'</td></tr>')
    (output/'index.html').write_text(''.join(body)+'</table></html>',encoding='utf-8')
    return session


def write_project(project, output, receipt=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if receipt is not None:
        write_json(receipt,output/'restore-receipt.json')
    project.save(output/'reviewed.autosub.json')
    for suffix in ('.ass','.srt'):
        export_subtitles(project,output/('reviewed'+suffix))


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['propose','apply','restore']:
        command=commands.add_parser(name)
        command.add_argument('project',type=Path)
        command.add_argument('--output',type=Path,required=True,help='Must be a fresh directory')
        if name=='apply':
            command.add_argument('--session',type=Path,required=True)
            command.add_argument('--id',required=True)
            command.add_argument('--reviewer',required=True,help='Recorded as supplied; not independent human validation')
        if name=='restore':
            command.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    before=file_hash(args.project)
    project=Project.load(args.project)
    if args.command=='propose':
        session=prepare_review(project,args.project,args.output)
        result={'items':len(session['items']),'applied':False}
    elif args.command=='apply':
        session=json.loads(args.session.read_text(encoding='utf-8'))
        if (session.get('schema_version')!=1 or session.get('algorithm')!=ALGORITHM or
                session.get('signature')!=signature(project)):
            raise ValueError('候選清單已過期或格式不支援，請重新產生')
        matches=[p for p in session['items'] if p.get('id')==args.id]
        if len(matches)!=1:
            raise ValueError('找不到唯一的指定候選')
        updated,receipt=apply_proposal(project,matches[0],args.reviewer)
        write_project(updated,args.output,receipt)
        result={'applied':args.id,'receipt':'restore-receipt.json'}
    else:
        restored=restore_proposal(project,json.loads(args.receipt.read_text(encoding='utf-8')))
        write_project(restored,args.output)
        result={'restored':True}
    assert file_hash(args.project)==before
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
