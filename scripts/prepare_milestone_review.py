import argparse
from pathlib import Path
from autosub.milestone_review import build_review

parser=argparse.ArgumentParser('Freeze source-clock listening excerpts, without modifying an unlabelled baseline')
parser.add_argument('project',type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
result=build_review(args.project,args.output)
print(len(result['targets']),'unreviewed listening targets; no error or accuracy verdict')
