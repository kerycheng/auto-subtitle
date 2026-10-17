"""Audition evidence for context/vocabulary effects; no automatic hypothesis adoption."""
import argparse
import html
import json
import os
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import validate_input, validate_resume
from autosub.asr_experiments import target_word_evidence, repetition_evidence


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--run', nargs=3, action='append', required=True, metavar=('LABEL', 'MANIFEST', 'RESULT'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    for case in cases:
        if file_hash(case['audio_path']) != case['audio_sha256']:
            raise ValueError('Original case audio changed')
    ids = {c['id'] for c in cases}
    runs = []
    for label, manifest_path, result_path in args.run:
        manifest_path, result_path = Path(manifest_path), Path(result_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_input(manifest)
        if (manifest.get('source_cases_sha256') != file_hash(args.cases)
                or file_hash(manifest['audio_path']) != manifest['audio_sha256']
                or file_hash(manifest_path.with_name('evidence.json')) != manifest['evidence_sha256']):
            raise ValueError('Context source changed')
        run = json.loads(result_path.read_text(encoding='utf-8'))
        validate_resume(run, file_hash(manifest_path), run['backend'], run['model_identity'], run['configuration'], manifest)
        if run['status'] != 'complete' or {r['id'] for r in run['results']} != ids:
            raise ValueError('Incomplete or mismatched cases')
        runs.append({'label': label, 'model': run['model'], 'model_identity': run['model_identity']['sha256'],
            'configuration': run['configuration'], 'context_padding_ms': manifest['context_padding_ms'],
            'manifest_sha256': file_hash(manifest_path), 'result_sha256': file_hash(result_path),
            'audio_path': manifest['audio_path'], 'results': [{**row, 'target_evidence': target_word_evidence(row),
                                                            'repetition_evidence': repetition_evidence(row['text'])}
                                                           for row in run['results']]})
    # Evidence stays distinct from any exportable project and all decisions are unreviewed.
    report = {'scope': 'seen development risks; context/vocabulary ablation, not gold evaluation',
        'accuracy': None, 'adopted': False, 'cases_sha256': file_hash(args.cases), 'cases': cases, 'runs': runs}
    args.output.mkdir(parents=True)
    esc = lambda v: html.escape(str(v))
    body = ['<!doctype html><meta charset="utf-8"><title>上下文與詞彙提示實驗</title>',
        '<style>body{font:16px system-ui;margin:24px;line-height:1.6}table{border-collapse:collapse;width:100%;table-layout:fixed}',
        'td,th{border:1px solid #aaa;padding:10px;vertical-align:top}td{overflow-wrap:anywhere}audio{max-width:100%}pre{white-space:pre-wrap;font-size:12px}</style>',
        '<h1>上下文／詞彙提示：待試聽證據</h1><p>沒有原語人工真值，不評正確率，不自動採用。',
        '目標文字保留所有與目標時間相交的完整原始詞；模型時間會漂移，這只是定位線索。',
        '長窗本來就會辨識更多窗外內容，不以總字數增加認定補回漏詞。詞彙提示來自使用者既有遊戲類別，未抄入人工字幕。</p>']
    for case in cases:
        body.append('<h2>' + esc(case['id']) + ' ' + esc(case['target_ms']) + '</h2><table>'
                    '<tr><th>設定／試聽</th><th>與目標時間相交的完整詞（非替換文）</th><th>完整上下文辨識</th></tr>')
        for run in runs:
            row = next(r for r in run['results'] if r['id'] == case['id'])
            a, b = row['window_ms']
            relative = Path(os.path.relpath(run['audio_path'], args.output)).as_posix()
            url = esc(relative + f'#t={a/1000},{b/1000}')
            for word in row['target_evidence']['words']:
                word['source_ms'] = [round(t - row['owner_ms'][0] + case['target_ms'][0], 3)
                                     for t in word['audio_ms']]
            details = esc(json.dumps(row['target_evidence']['words'], ensure_ascii=False, indent=2))
            phrase = row['repetition_evidence']['consecutive_phrase_repetition']
            body.append('<tr><td>' + esc(run['label']) + '<br>' + esc(run['configuration']['prompt']) +
                f'<br><audio controls preload="none" src="{url}"></audio></td><td>' +
                esc(row['target_evidence']['text']) + '<details><summary>原始詞時間／邊界</summary><pre>' +
                details + '</pre></details></td><td>' + esc(row['text']) + '<br><small>字元數：' +
                str(row['repetition_evidence']['characters']) + '；最長連續相同字元：' +
                str(row['repetition_evidence']['max_same_character_run']) +
                ('；相同字串連續出現：' + str(phrase['copies']) + ' 次' if phrase else '') +
                '（不代表幻覺判定）</small></td></tr>')
        body.append('</table>')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'index.html').write_text(''.join(body), encoding='utf-8')
    print(f'{len(cases)} cases, {len(runs)} variants; unreviewed, no text adopted')


if __name__ == '__main__':
    main()
