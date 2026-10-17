"""Freeze an unlabelled natural clip, run production ASR, export without text repair."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from threading import Event
import time

import av

from autosub import __version__
from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.comparison import read_reference
from autosub.domain import Project, word_key
from autosub.export import export_subtitles
from autosub.language_validation import normalized_text
from autosub.pipeline import run_pipeline
from autosub.segmentation_audit import audit


def save(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def describe(source):
    with av.open(str(source)) as container:
        return {'duration_seconds':container.duration/1000000 if container.duration else None,
                'audio_tracks':[{'stream_index':s.index,'codec':s.codec_context.name,
                                 'sample_rate':s.codec_context.sample_rate,'channels':s.codec_context.channels,
                                 'metadata':dict(s.metadata)} for s in container.streams.audio]}


def report(project, target, plan):
    before=file_hash(target/'generated.autosub.json')
    diagnostics=audit(project)
    keys=list(map(word_key,project.words))
    owned=[k for c in project.cues for k in c.source_tokens or []]
    if keys!=owned or not diagnostics['source_tokens_preserved']:
        raise ValueError('Original recognized word ownership/order changed')
    if normalized_text(''.join(w.text for w in project.words))!=normalized_text(''.join(c.text for c in project.cues)):
        raise ValueError('Displayed recognition text changed')
    exports={}
    for suffix in ('.ass','.srt'):
        path=target/('recognition-baseline'+suffix)
        export_subtitles(project,path)
        parsed,_=read_reference(path)
        if len(parsed)!=len(project.cues):
            raise ValueError('Export event count changed')
        exports[suffix]={'path':str(path.resolve()),'sha256':file_hash(path),'events':len(parsed)}
    if file_hash(plan['source']['path'])!=plan['source']['sha256']:
        raise ValueError('Original media changed')
    contexts=[e for e in project.review_events if e.get('action')=='asr_trace_context']
    checkpoints=[e for e in project.review_events if e.get('action')=='asr_trace_checkpoint']
    result={'status':'baseline-generated-not-human-validated','source_unchanged':True,
            'project_sha256':before,'version':__version__,'duration_ms':project.duration_ms,
            'words':len(project.words),'native_groups':len(project.native_groups),'cues':len(project.cues),
            'source_word_order_preserved':True,'exports':exports,'diagnostics':diagnostics,
            'observed_temperatures':dict(Counter(str(s['temperature']) for e in checkpoints for s in e['segments'])),
            'trace_contexts':len(contexts),'trace_checkpoints':len(checkpoints),
            'human_completion_rate':None,'word_omission_rate':None,'name_accuracy':None,
            'note':'No original-language or acoustic gold. Mechanical warnings are listening targets, not confirmed errors.'}
    save(target/'report.json',result)
    # Literal ASR forms only. The list is deliberately not a guessed member glossary.
    forms=Counter(re.findall(r'[ァ-ヴー]{2,}|[A-Za-z][A-Za-z0-9_-]+',' '.join(w.text for w in project.words)))
    terms=[{'recognized_form':text,'occurrences':count,'verified_spelling':None,'decision':'unreviewed'}
           for text,count in forms.most_common()]
    save(target/'term-review.json',{'scope':'literal katakana/Latin ASR forms, incomplete term inventory; no replacements',
                                  'terms':terms})
    esc=lambda value:html.escape(str(value),quote=True)
    body=['<!doctype html><meta charset="utf-8"><title>辨識基準</title>',
          '<style>body{font:16px system-ui;margin:28px;max-width:1300px}table{border-collapse:collapse;width:100%}'
          'td,th{border:1px solid #bbb;padding:8px;text-align:left}video{width:min(100%,960px);position:sticky;top:0;background:#111}'
          'button{padding:6px}small{color:#555}</style>',
          '<h1>未經人工校對的辨識基準</h1>',
          '<p>large-v3 + CUDA · 日文 · 正式預設參數 · 無詞彙提示。尚無人工原語／聲學標準，完整度、漏詞率與成員名稱正確率均未評定。</p>',
          '<p>保留所有原始辨識詞；不按遊戲情境猜補名字。低信心、字典切點與 VAD 空檔只是試聽目標，不代表錯字或漏詞。</p>',
          f'<p>{len(project.words)} 詞／{len(project.native_groups)} 原生分組／{len(project.cues)} 軸。'
          '<a href="recognition-baseline.ass">ASS</a> · <a href="recognition-baseline.srt">SRT</a> · '
          '<a href="report.json">診斷紀錄</a> · <a href="term-review.json">原始術語候選</a></p>',
          f'<video id="player" controls preload="metadata" src="{esc(Path(project.media_path).as_uri())}"></video>',
          '<p><small>本機播放器若無法開啟來源，可直接把 ASS／SRT 匯入 Aegisub。此頁供測試聽辨，不修改字幕。</small></p>',
          '<table><tr><th>序號</th><th>時間（秒）</th><th>未校對辨識文字</th><th>生成提示</th></tr>']
    for i,cue in enumerate(project.cues,1):
        body.append(f'<tr><td>{i}</td><td><button onclick="seek({max(0,cue.start_ms-300)/1000},{min(project.duration_ms,cue.end_ms+300)/1000})">'
                    f'{cue.start_ms/1000:.2f}–{cue.end_ms/1000:.2f}</button></td><td>{esc(cue.text)}</td><td>{esc("；".join(cue.flags))}</td></tr>')
    body.append('</table><script>const p=document.getElementById("player");let stopAt=null;'
                'function seek(a,b){p.pause();p.currentTime=a;stopAt=b;p.play().catch(()=>{});}'
                'p.addEventListener("timeupdate",()=>{if(stopAt!==null&&p.currentTime>=stopAt){p.pause();stopAt=null;}});</script>')
    (target/'index.html').write_text(''.join(body),encoding='utf-8')
    if file_hash(target/'generated.autosub.json')!=before:
        raise ValueError('Reporting mutated project')
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--description',default='')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    source=args.source.resolve();target=args.output.resolve()
    if target==source.parent or target.is_relative_to(source.parent):
        raise ValueError('Keep all derivatives outside the original media directory')
    if target.exists() and not args.resume:
        raise ValueError('Use a fresh milestone directory or explicit resume')
    target.mkdir(parents=True,exist_ok=True)
    settings=Project().settings
    plan={'milestone':'unlabelled recognition baseline','created_for_version':__version__,
          'source':{'path':str(source),'size':source.stat().st_size,'sha256':file_hash(source),'media':describe(source)},
          'model_path':str(args.model.resolve()),'model_identity':model_identity(args.model),
          'language':'ja','device':'cuda','glossary':'','boundary_policy':'legacy','settings':asdict(settings),
          'description_not_sent_to_asr':args.description,'human_reference':None,
          'acceptance':['entire source processed and source hash unchanged','original words and ownership order preserved',
                        'ASS/SRT import round-trip','literal terms and boundary concerns retained for listening'],
          'unassessed':['semantic sentence boundaries','later-sentence prefix carry by meaning','omitted speech',
                        'game terminology and member names','acoustic onset/offset accuracy']}
    plan_path=target/'milestone.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Milestone source/model/settings changed; use a separate baseline')
    if not plan_path.exists():save(plan_path,plan)
    path=target/'generated.autosub.json'
    project=Project.load(path) if path.exists() else Project(media_path=str(source),model=str(args.model.resolve()),device='cuda')
    start=time.perf_counter()
    def progress(percent,message):
        save(target/'progress.json',{'status':'running','percent':percent,'message':message,
                                   'at_utc':datetime.now(timezone.utc).isoformat()})
        print(percent,message,flush=True)
    try:
        run_pipeline(project,path,Event(),progress)
        result=report(project,target,plan)
        save(target/'progress.json',{'status':result['status'],'percent':100,'elapsed_seconds_this_invocation':time.perf_counter()-start})
        print('BASELINE COMPLETE',len(project.cues),'cues; human quality unassessed',flush=True)
    except BaseException as exc:
        save(target/'progress.json',{'status':'interrupted-or-failed','error_type':type(exc).__name__,
                                   'error':str(exc),'elapsed_seconds_this_invocation':time.perf_counter()-start})
        raise


if __name__=='__main__':main()
