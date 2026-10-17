import argparse
from pathlib import Path
from autosub.language_gain_study import run_study

parser=argparse.ArgumentParser('Run one frozen language-gated strategy without reading reference text')
parser.add_argument('inputs',type=Path)
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--mode',required=True,choices=['production-0.9.4','vad-local-gain-union-original-asr'])
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
run_study(args.inputs,args.plan,args.mode,args.output,args.resume)
