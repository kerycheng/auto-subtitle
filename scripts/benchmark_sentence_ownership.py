"""Real native-v1 projects on fixed complete-source sentence concatenations."""
import argparse
import html
import json
import os
from pathlib import Path
from threading import Event

from autosub.alignment_lab import file_hash
from autosub.comparison import read_reference
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.pipeline import run_pipeline
from autosub.sentence_benchmark import prepare_pairs, assess_pair
from autosub.asr_comparison import model_identity


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rescore-existing', type=Path, help='Read existing projects only; never repeat ASR')
    parser.add_argument('--prepared-labels', type=Path, help='Use a separately prepared controlled stress input')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    if args.rescore_existing:
        labels_source = args.prepared_labels or args.rescore_existing / 'inputs/labels.json'
        inputs = json.loads(labels_source.read_text(encoding='utf-8'))
    elif args.prepared_labels:
        inputs = json.loads(args.prepared_labels.read_text(encoding='utf-8'))
    else:
        inputs = prepare_pairs(args.dataset, args.output / 'inputs', [('ja-2', 'ja-3'), ('ja-10', 'ja-11')])
    if inputs['dataset_sha256'] != file_hash(args.dataset):
        raise ValueError('Dataset changed')
    args.output.mkdir(parents=True, exist_ok=True)
    labels_folder = args.output / 'inputs'
    labels_folder.mkdir(exist_ok=True)
    labels_path = labels_folder / 'labels.json'
    if not labels_path.exists():
        labels_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'status': 'running', 'cases': [], 'scope': inputs['scope'], 'model': str(args.model.resolve()),
        'human_sentence_accuracy': None, 'acoustic_timing_accuracy': None, 'model_identity': model_identity(args.model),
        'labels_sha256': file_hash(labels_path)}
    body = ['<!doctype html><meta charset="utf-8"><title>源句所有權控制測試</title>',
        '<style>body{font:16px system-ui;margin:28px;line-height:1.6}td{padding:8px;border:1px solid #bbb;max-width:650px;overflow-wrap:anywhere}</style>',
        '<h1>完整源句串接：正式 native-v1</h1><p>兩對預先指定的完整公開日文音檔，以 0／300／800ms 停頓串接。',
        '來源區間已知，內部實際發聲字詞邊緣未知。文字配對只用最小編輯路徑的完全相同字元；',
        '混合兩個來源不自動等於錯誤，1–3 字句首跨軸僅列候選，非人工語意／50ms 精度驗收。</p>']
    for case in inputs['cases']:
        if file_hash(case['audio_path']) != case['audio_sha256']:
            raise ValueError('Source changed')
        for region in case['regions']:
            if file_hash(region['source_path']) != region['source_sha256']:
                raise ValueError('Original source utterance changed')
        folder = args.output / case['id']
        folder.mkdir()
        if args.rescore_existing:
            p = Project.load(args.rescore_existing / case['id'] / 'native.autosub.json')
            if not p.native_complete or p.media_path != case['audio_path'] or p.model != str(args.model.resolve()):
                raise ValueError('Saved project does not match completed source/model')
        else:
            p = Project(media_path=case['audio_path'], model=str(args.model.resolve()), language='ja', device='cuda')
            run_pipeline(p, folder / 'native.autosub.json', Event(),
                lambda percent, message: print(case['id'], percent, message, flush=True))
        result = assess_pair(p, case)
        if not result['source_tokens_preserved'] or not result['source_token_order_preserved'] or not result['normalized_characters_preserved']:
            raise ValueError('Source content lost')
        for suffix in ('.ass', '.srt'):
            output = folder / ('native' + suffix)
            export_subtitles(p, output)
            parsed, _ = read_reference(output)
            if len(parsed) != len(p.cues):
                raise ValueError('Export event count changed')
        source_project = (args.rescore_existing or args.output) / case['id'] / 'native.autosub.json'
        report['cases'].append({'id': case['id'], 'source_regions': case['regions'], 'assessment': result,
                               'project_sha256': file_hash(source_project)})
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        audio_url = Path(os.path.relpath(case['audio_path'], args.output)).as_posix()
        body.append('<h2>' + html.escape(case['id']) + '</h2><p><a href="' + case['id'] + '/native.ass">ASS</a> · <a href="' +
                    case['id'] + '/native.srt">SRT</a></p><audio controls preload="none" src="' + html.escape(audio_url) + '"></audio>')
        for region in case['regions']:
            body.append('<p>來源音檔區間 ms：' + str([frame / 16 for frame in region['sample_frames']]) +
                        '；原語參考：' + html.escape(region['reference_text']) + '</p>')
        body.append('<pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre><table>')
        for cue in p.cues:
            body.append('<tr><td>' + str([cue.start_ms, cue.end_ms]) + '</td><td>' + html.escape(cue.text) + '</td></tr>')
        body.append('</table>')
    report['status'] = 'complete'
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'index.html').write_text(''.join(body), encoding='utf-8')


if __name__ == '__main__':
    main()
