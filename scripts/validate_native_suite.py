"""Compare segmentation on identical ASR words across prepared development clips."""
import argparse
import copy
import html
import json
import time
from pathlib import Path
from threading import Event

from autosub.alignment_lab import file_hash
from autosub.comparison import ComparisonSettings, compare_file, export_report, read_reference
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.pipeline import run_pipeline
from autosub.segmentation_audit import audit
from autosub.timing import apply_timing, segment_words


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.output.exists() or not (args.model/'model.bin').is_file():
        raise ValueError('Select a fresh output directory and a local model')
    inputs=[]
    for i in range(1,6):
        folder=args.prepared/f'ja-{i:02}'
        source=folder/'source.json'
        info=json.loads(source.read_text(encoding='utf-8'))
        if info['split']!='development' or info['id']!=folder.name:
            raise ValueError('Only frozen development excerpts may be evaluated')
        audio=folder/'excerpt.wav'
        inputs.append((info,audio,source))
    before={str(p.resolve()):file_hash(p) for info,audio,source in inputs
            for p in (audio,source,Path(info['reference'])) if p.is_file()}
    args.output.mkdir(parents=True)
    report={'scope':'five previously seen development excerpts; not held-out or human gold',
            'model':str(args.model.resolve()),'device':args.device,'samples':[],
            'human_completion_rate':None,'acoustic_timing_accuracy':None,
            'source_hashes':before,'status':'running'}
    esc=lambda value:html.escape(str(value))
    for info,audio,source in inputs:
        folder=args.output/info['id']
        folder.mkdir()
        p=Project(media_path=str(audio.resolve()),language='ja',model=str(args.model.resolve()),device=args.device)
        start=time.perf_counter()
        run_pipeline(p,folder/'native.autosub.json',Event(),
                     lambda percent,message:print(info['id'],percent,message,flush=True))
        elapsed=time.perf_counter()-start
        p=Project.load(folder/'native.autosub.json')
        assert apply_timing(p)==p.cues
        # Inject the legacy segmenter on the identical source stream; no second ASR,
        # VAD, punctuation editing or human reference can explain this comparison.
        legacy=copy.deepcopy(p)
        legacy.cues=apply_timing(legacy,segmenter=segment_words)
        old,new=audit(legacy),audit(p)
        assert p.words==legacy.words and p.speech==legacy.speech
        assert old['source_tokens_preserved'] and new['source_tokens_preserved']
        for label,project in [('native',p),('same-asr-legacy',legacy)]:
            for suffix in ('.ass','.srt'):
                destination=folder/(label+suffix)
                export_subtitles(project,destination)
                parsed,_=read_reference(destination)
                assert len(parsed)==len(project.cues)
        reference=Path(info['reference'])
        if reference.is_file():
            session=compare_file(p,reference,
                                 ComparisonSettings(reference_offset_ms=round(info['offset']*1000)))
            export_report(session,folder/'manual-time-comparison.html')
        result={'id':info['id'],'duration_ms':p.duration_ms,'elapsed_seconds':elapsed,
                'legacy_same_asr':old,'native':new,'export_round_trip':'passed',
                'reference_status':'time-comparison-only' if reference.is_file() else 'missing-no-guessed-replacement'}
        report['samples'].append(result)
        (folder/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        audio_url=Path(p.audio_path).relative_to(folder.resolve()).as_posix()
        body=['<!doctype html><meta charset="utf-8"><title>'+esc(info['id'])+' 句界檢查</title>',
            '<style>body{font:16px system-ui;margin:24px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}</style>',
            '<h1>'+esc(info['id'])+' 同辨識文字分段比較</h1>',
            '<p><a href="native.ass">新版 ASS</a> · <a href="same-asr-legacy.ass">同文字舊分段 ASS</a> · '+
            ('<a href="manual-time-comparison.html">人工成品時間參考</a>' if reference.is_file() else '原參考路徑不存在，未猜選其他字幕')+'</p>',
            '<p>詞界檢查由同一斷詞器計算；VAD 未覆蓋區僅為試聽目標，也可能是笑聲、噪音或時間對齊誤差。不能作為漏詞率或免修率。</p>',
            '<audio id="player" controls preload="none" src="'+esc(audio_url)+'"></audio>',
            '<h2>語音未被時間片覆蓋的試聽位置（至少 700ms）</h2>']
        for a,b in new['uncovered_vad_ms']:
            body.append(f'<p><button onclick="const p=document.getElementById(\'player\');p.currentTime={max(0,a-500)/1000};p.play()">試聽 {a/1000:.2f}–{b/1000:.2f} 秒</button></p>')
        body.append('<h2>新字幕與時間相交的舊分段</h2><table><tr><th>新時間／文字</th><th>同文字舊分段</th><th>新複核提示</th></tr>')
        for cue in p.cues:
            prior=[c for c in legacy.cues if c.start_ms<cue.end_ms and c.end_ms>cue.start_ms]
            body.append('<tr><td>'+f'{cue.start_ms/1000:.2f}–{cue.end_ms/1000:.2f}<br>'+esc(cue.text)+'</td><td>'+
                        '<br>'.join(f'{c.start_ms/1000:.2f}–{c.end_ms/1000:.2f} '+esc(c.text) for c in prior)+
                        '</td><td>'+esc(cue.flags)+'</td></tr>')
        body.append('</table><h2>低信心時間片（模型內部分數，非正確率）</h2><pre>'+esc(json.dumps(new['low_confidence_words'],ensure_ascii=False,indent=2))+'</pre>')
        (folder/'index.html').write_text(''.join(body),encoding='utf-8')
        (args.output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['sources_unchanged']={p:file_hash(p)==sha for p,sha in before.items()}
    assert all(report['sources_unchanged'].values())
    report['status']='passed'
    (args.output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    body=['<!doctype html><meta charset="utf-8"><title>跨素材原生句界回歸</title>',
          '<style>body{font:16px system-ui;margin:24px;line-height:1.7}td,th{padding:10px;border:1px solid #bbb}table{border-collapse:collapse}</style>',
          '<h1>五個開發片段：同辨識文字的新舊分段</h1><p>未使用三個保留來源；尚無人工原語／聲學 gold，不計算完成度。原始來源與參考字幕 SHA256 不變。</p>',
          '<table><tr><th>片段／試聽</th><th>時間片</th><th>字幕數 舊→新</th><th>已知詞內切點 舊→新</th><th>顯示重疊對 舊→新</th><th>待聽位置</th></tr>']
    for row in report['samples']:
        old,new=row['legacy_same_asr'],row['native']
        body.append('<tr><td><a href="'+row['id']+'/index.html">'+row['id']+'</a></td>'+''.join('<td>'+esc(v)+'</td>' for v in [
            new['words'],f"{old['cues']} → {new['cues']}",f"{old['known_word_internal_cuts']} → {new['known_word_internal_cuts']}",
            f"{len(old['overlap_pairs'])} → {len(new['overlap_pairs'])}",len(new['uncovered_vad_ms'])])+'</tr>')
    (args.output/'index.html').write_text(''.join(body)+'</table>',encoding='utf-8')
    print(json.dumps({'status':report['status'],'samples':len(report['samples'])}))


if __name__=='__main__':
    main()
