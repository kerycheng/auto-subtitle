"""Compare complete level-pair runs, without adopting a decoding strategy."""
import argparse
import html
import json
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import character_alignment
from autosub.domain import Project
from autosub.sentence_benchmark import assess_pair


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists')
    cases=json.loads((args.inputs/'cases.json').read_text(encoding='utf-8'))
    labels=json.loads((args.inputs/'labels.json').read_text(encoding='utf-8'))
    if cases['protocol']!=labels['protocol']: raise ValueError('Protocol pairing changed')
    if file_hash(labels['source_dataset_path'])!=labels['protocol']['dataset_sha256']:
        raise ValueError('Original reference dataset changed')
    dataset=json.loads(Path(labels['source_dataset_path']).read_text(encoding='utf-8'))
    references={r['id']:r for r in dataset['samples']}
    for label in labels['cases']:
        for region in label['regions']:
            if (region['reference_text']!=references[region['sample_id']]['reference_text']
                    or file_hash(region['source_path'])!=region['source_sha256']):
                raise ValueError('Original reference/source pairing changed')
    by_id={c['id']:c for c in labels['cases']}
    if len(by_id)!=len(labels['cases']) or set(by_id)!={c['id'] for c in cases['cases']}:
        raise ValueError('Reference IDs differ')
    result={'protocol':labels['protocol'],'labels_sha256':file_hash(args.inputs/'labels.json'),'runs':[],
        'scope':'controlled amplitude pairs, one run per strategy, default sampling fallback not separately controlled; no acoustic gold'}
    model=None
    body=['<!doctype html><meta charset="utf-8"><title>音量交錯語音對照</title><style>body{font:16px system-ui;margin:30px}td{border:1px solid #aaa;padding:8px;max-width:620px;overflow-wrap:anywhere}table{border-collapse:collapse}</style><h1>正常音量＋低音量完整源句</h1><p>固定公開開發來源，每對少於30秒。保留原聲、降低後句18／30dB、間隔0／800ms。原語僅供辨識後評估，參考不是字詞發聲真值；預設溫度回退未另行固定，單次差異不能全部歸因VAD。</p>']
    for folder in args.runs:
        run=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if (run['status']!='complete' or run['cases_sha256']!=file_hash(args.inputs/'cases.json')
                or len(run['cases'])!=len(by_id) or {c['id'] for c in run['cases']}!=set(by_id)):
            raise ValueError('Incomplete or mismatched run')
        if model is not None and model!=run['model_identity']: raise ValueError('Models differ')
        model=run['model_identity'];measured={'mode':run['mode'],'report_sha256':file_hash(folder/'report.json'),'cases':[],'languages':{}}
        body.append('<h2>'+html.escape(run['mode'])+'</h2><table>')
        for case in run['cases']:
            label=by_id[case['id']];path=folder/case['id']/'generated.autosub.json'
            if file_hash(path)!=case['project_sha256'] or file_hash(label['audio_path'])!=label['audio_sha256']:
                raise ValueError('Project/source changed')
            p=Project.load(path)
            if p.media_path!=label['audio_path'] or p.status!='complete': raise ValueError('Wrong saved project')
            hypothesis=''.join(w.text for w in p.words)
            reference=''.join(r['reference_text'] for r in label['regions'])
            score=character_alignment(reference,hypothesis)
            ownership=assess_pair(p,label)
            if (not ownership['source_tokens_preserved'] or not ownership['source_token_order_preserved']
                    or not ownership['normalized_characters_preserved']):
                raise ValueError('Word ownership changed')
            measured['cases'].append({'id':case['id'],'language':p.language,'reference':reference,'hypothesis':hypothesis,
                'score':score,'ownership':ownership,'second_clean_vad_proxy_coverage':label['second_clean_vad_proxy_coverage']})
            counts=measured['languages'].setdefault(p.language,{'character_edits':0,'reference_characters':0})
            for key in counts: counts[key]+=score[key]
            body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                [case['id'],reference,hypothesis,f'{score["mechanical_cer"]:.2%}',ownership['status']])+'</tr>')
        body.append('</table>');result['runs'].append(measured)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'index.html').write_text(''.join(body),encoding='utf-8')
    print(json.dumps([(r['mode'],r['languages']) for r in result['runs']]))


if __name__=='__main__':
    main()
