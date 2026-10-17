"""Post-ASR reference comparison. Dataset text is never a recognition prompt."""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

from autosub.alignment_lab import file_hash, validate_result
from autosub.domain import Project, Cue
from autosub.export import export_subtitles
from autosub.comparison import read_reference
from autosub.language_validation import normalized_text
from autosub.paired_evaluation import _distance

parser=argparse.ArgumentParser()
parser.add_argument("dataset",type=Path)
parser.add_argument("asr_manifest",type=Path)
parser.add_argument("asr_result",type=Path)
parser.add_argument("align_manifest",type=Path)
parser.add_argument("align_result",type=Path)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise ValueError("Output exists")
dataset=json.loads(args.dataset.read_text(encoding='utf-8'))
m=json.loads(args.asr_manifest.read_text(encoding='utf-8'))
r=json.loads(args.asr_result.read_text(encoding='utf-8'))
am=json.loads(args.align_manifest.read_text(encoding='utf-8'))
ar=json.loads(args.align_result.read_text(encoding='utf-8'))
if m['source_manifest_sha256']!=file_hash(args.dataset) or r['manifest_sha256']!=file_hash(args.asr_manifest):
    raise ValueError('ASR/dataset identity mismatch')
if am['source_document_sha256']!=file_hash(args.asr_result): raise ValueError('Alignment text changed')
am['manifest_sha256']=file_hash(args.align_manifest)
checks=validate_result(am,ar)
by_id={row['id']:row for row in r['results']}
if set(by_id)!={row['id'] for row in dataset['samples']}: raise ValueError('Missing samples')
args.output.mkdir(parents=True)
report={'scope':'8 FLEURS validation read-speech samples; not livestream quality or acoustic precision',
        'dataset':dataset['dataset'],'license':dataset['license'],'reference_used_by_asr':False,
        'asr_inference_seconds':sum(row['seconds'] for row in r['results']),
        'asr_load_seconds':r['load_seconds'],'peak_asr_tensor_bytes':max(row['peak_tensor_bytes'] for row in r['results']),
        'samples':[],'by_language':{},'alignment_statuses':dict(Counter(c['status'] for c in checks.values()))}
p=Project(duration_ms=m['duration_ms'])
for sample in dataset['samples']:
    row=by_id[sample['id']]
    truth,hyp=normalized_text(sample['reference_text']),normalized_text(row['text'])
    edits=_distance(truth,hyp)
    check=checks[sample['id']]
    report['samples'].append({'id':sample['id'],'language':sample['language'],
        'reference_text':sample['reference_text'],'recognized_text':row['text'],
        'character_edits':edits,'reference_characters':len(truth),'mechanical_cer':edits/len(truth),
        'alignment':check})
    start,end=check['span_ms'] or row['window_ms']
    p.cues.append(Cue(sample['id'],start,end,row['text'],start,end,
        ['朗讀小樣本候選；若未可靠對齊則沿用樣本顯示區間，非人工聲學時間']))
for lang in {s['language'] for s in dataset['samples']}:
    rows=[s for s in report['samples'] if s['language']==lang]
    report['by_language'][lang]={'samples':len(rows),'mechanical_cer':sum(x['character_edits'] for x in rows)/sum(x['reference_characters'] for x in rows)}
for suffix in ('.ass','.srt'):
    path=args.output/('four-languages'+suffix)
    export_subtitles(p,path)
    parsed,_=read_reference(path)
    if len(parsed)!=len(p.cues): raise ValueError('Export roundtrip failed')
report['ass_srt_roundtrip']='passed'
(args.output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
body=['<!doctype html><meta charset="utf-8"><title>Qwen 四語小樣本</title><style>body{font:16px system-ui;margin:28px}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>',
      '<h1>Qwen 四語辨識／對齊小樣本</h1><p>Google FLEURS validation，CC-BY-4.0，每語兩句。CER 只做 NFKC、大小寫、空白與標點正規化，未統一繁簡／數字。非直播品質或 50ms 驗收。</p><table><tr><th>ID</th><th>資料集原文</th><th>辨識</th><th>CER</th><th>對齊狀態</th></tr>']
for row in report['samples']:
    body.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [row['id'],row['reference_text'],row['recognized_text'],f"{row['mechanical_cer']:.1%}",row['alignment']])+'</tr>')
body.append('</table>')
(args.output/'index.html').write_text(''.join(body),encoding='utf-8')
print(json.dumps({'by_language':report['by_language'],'alignment':report['alignment_statuses'],'ass_srt_roundtrip':report['ass_srt_roundtrip']},indent=2))
