"""Publish a validated, read-only local compensation comparison."""
import argparse
import html
import json
from pathlib import Path

from autosub.compensation_evaluation import evaluate_study


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    result=evaluate_study(args.inputs,args.runs,args.plan)
    body=['<!doctype html><meta charset="utf-8"><title>局部增益研究</title><style>body{font:16px system-ui;margin:28px;line-height:1.6}td,th{padding:8px;border:1px solid #aaa;max-width:600px;overflow-wrap:anywhere}table{border-collapse:collapse}</style><h1>局部增益：VAD 與 ASR 分開驗證</h1><p>36例×三策略；八份公開開發原音檔衍生，不是108份獨立自然錄音。原語不供ASR，模型預設不變。</p><p>正式維持0.9.3。加總分數改善不能掩蓋日文退化；已知生成非語音中若有詞時間只是待核候選，需區分多辨字與時間漂移。重用16例缺原始解碼溫度，其他新試驗保留實際溫度。非人工聲學gold，不能宣稱直播完成度或50ms。</p>']
    for run in result['runs']:
        body.append('<h2>'+html.escape(run['mode'])+'</h2><table><tr><th>組別</th><th>編輯／參考字元</th><th>機械CER</th></tr>')
        for category,row in run['summary'].items():
            body.append('<tr><td>'+html.escape(category)+'</td><td>'+str(row['character_edits'])+'/'+str(row['reference_characters'])+'</td><td>'+f"{row['character_edits']/row['reference_characters']:.2%}"+'</td></tr>')
        body.append('</table><p>比原流程編輯數增加的案例：'+html.escape(str(run['regressions_vs_raw']))+'</p><details><summary>保留所有參考與假設</summary><table>')
        for c in run['cases']:
            body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [c['id'],c['reference'],c['hypothesis'],c['decoder_temperatures'],c['generated_noise_timestamp_candidates']])+'</tr>')
        body.append('</table></details>')
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'index.html').write_text(''.join(body),encoding='utf-8')
    print(result['production_decision'])


if __name__=='__main__':main()
