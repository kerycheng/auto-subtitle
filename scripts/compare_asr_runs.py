"""Score saved, reference-free ASR runs; never invoke ASR using reference text."""
import argparse
import html
import json
from pathlib import Path

from autosub.asr_comparison import compare_runs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; preserve prior evidence")
    report = compare_runs(args.dataset, args.manifest, args.results)
    body = ['<!doctype html><meta charset="utf-8"><title>同輸入 ASR 比較</title>',
        '<style>body{font:16px system-ui;margin:28px;line-height:1.6}table{border-collapse:collapse}',
        'td,th{border:1px solid #bbb;padding:8px;vertical-align:top}td{max-width:650px;overflow-wrap:anywhere}</style>',
        '<h1>同輸入 ASR 比較</h1><p>FLEURS validation，CC-BY-4.0，各語樣本數見表格。',
        '這是朗讀小樣本，不能代表直播、遊戲專名或混語。CER 越低表示此正規化規則下字元編輯較少。',
        '刪除／插入只表示最小編輯路徑，不能直接認定漏詞／幻覺。繁簡與數字未統一。',
        '參考文字可能含外文括注，是否實際朗讀尚未逐段確認，統計保留原文而不自行刪除括注。',
        '各後端使用相同 PCM16 音訊、指定語言、無提示、無 VAD；後端解碼方法仍有差異。',
        '單次耗時且 GPU 與其他應用共用，不能作效能排名。未評分句界或聲學精度。</p>',
        '<table><tr><th>模型</th><th>語言</th><th>音檔數／不同正規化文本</th><th>CER</th><th>替換／刪除／插入</th></tr>']
    for run in report['runs']:
        name = Path(run['model']).name
        # Snapshot names are opaque; the containing repository identifies Whisper variants.
        if name != 'qwen-asr':
            name = Path(run['model']).parent.parent.name
        run['display_name'] = name
        for lang, row in run['by_language'].items():
            values = [name, lang, f'{row["samples"]}/{row["distinct_normalized_references"]}', f'{row["mechanical_cer"]:.2%}',
                f'{row["substitution"]}/{row["deletion"]}/{row["insertion"]}']
            body.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>')
    body.append('</table><h2>逐句原文與辨識</h2>')
    for run in report['runs']:
        body.append('<h3>' + html.escape(run['display_name']) + '</h3><table>')
        for row in run['samples']:
            values = [row['id'], row['reference_text'], row['recognized_text'], f'{row["mechanical_cer"]:.2%}',
                      ('參考含括注，需核對發聲：' + ', '.join(row['reference_parentheticals']) + '\n'
                       if row['reference_parentheticals'] else '') + json.dumps(row['differences'], ensure_ascii=False)]
            body.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>')
        body.append('</table>')
    args.output.mkdir(parents=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'index.html').write_text(''.join(body), encoding='utf-8')
    print(json.dumps({r['display_name']: r['by_language'] for r in report['runs']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
