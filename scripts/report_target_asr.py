"""Post-recognition target comparison; retain all unscored context hypotheses."""
import argparse
import html
import json
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import validate_input, validate_resume
from autosub.target_text import target_distance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    manifest_path = args.inputs/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_input(manifest)
    labels = json.loads((args.inputs/'evidence.json').read_text(encoding='utf-8'))
    proof = json.loads((args.inputs/'provenance.json').read_text(encoding='utf-8'))
    if (file_hash(manifest['audio_path']) != manifest['audio_sha256']
            or proof['manifest_sha256'] != file_hash(manifest_path)
            or proof['evidence_sha256'] != file_hash(args.inputs/'evidence.json')
            or labels['reference_sha256'] != file_hash(proof['reference_path'])
            or labels['project_sha256'] != file_hash(proof['project_path'])):
        raise ValueError('Audio/reference/project/evidence pairing changed')
    expected = {e['id']:e for e in labels['windows']}
    if len(expected) != len(labels['windows']) or set(expected) != {e['id'] for e in manifest['windows']}:
        raise ValueError('Reference window IDs differ')
    report = {'scope': labels['scope'], 'evidence_sha256': file_hash(args.inputs/'evidence.json'),
        'unscored_reference_ids': labels['unscored_reference_ids'],
        'overlap_reconstruction': labels['overlap_reconstruction'], 'runs': []}
    body = ['<!doctype html><meta charset="utf-8"><title>同窗目標原語比較</title><style>body{font:16px system-ui;margin:28px}td{padding:10px;border:1px solid #aaa;max-width:650px;overflow-wrap:anywhere}</style>',
        '<h1>相同音訊窗：目標原語比較</h1><p>參考只用於辨識後評估；每個固定30秒視窗只評完全位於窗內且保留500ms邊緣的參考事件。',
        '配對允許略過前後上下文，不能當成全檔 CER、幻覺率、完成度或獨立測試。字形、人名及數字未統一。</p>',
        '<p>未評參考事件：'+html.escape(str(labels['unscored_reference_ids']))+'</p>']
    for path in args.results:
        run = json.loads(path.read_text(encoding='utf-8'))
        validate_resume(run,file_hash(manifest_path),run['backend'],run['model_identity'],run['configuration'],manifest)
        if run['status'] != 'complete' or {e['id'] for e in run['results']} != set(expected):
            raise ValueError('Incomplete results')
        result = {'backend':run['backend'], 'model':run['model'], 'result_sha256':file_hash(path), 'samples':[]}
        body.append('<h2>'+html.escape(run['backend'])+'</h2><table>')
        for row in run['results']:
            truth = expected[row['id']]['reference_text']
            measured = target_distance(truth,row['text'])
            result['samples'].append({'id':row['id'], 'window_ms':row['window_ms'],
                'source_rows':expected[row['id']]['source_rows'],
                'reference_span_ms':expected[row['id']]['reference_span_ms'],
                'reference':truth, 'hypothesis':row['text'], **measured})
            body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                [row['id'],truth,row['text'],f'{measured["target_only_cer"]:.2%}'])+'</tr>')
        edits = sum(r['character_edits'] for r in result['samples'])
        chars = sum(r['reference_characters'] for r in result['samples'])
        result['target_only_cer'] = edits/chars
        result['character_edits'], result['reference_characters'] = edits, chars
        report['runs'].append(result)
        body.append('</table>')
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'index.html').write_text(''.join(body),encoding='utf-8')
    print(json.dumps([(r['backend'],r['target_only_cer']) for r in report['runs']]))


if __name__ == '__main__':
    main()
