"""Generate an exact-text native-sentence preview from a whole-clip ASR result."""
import argparse
import copy
import html
import json
from dataclasses import asdict
from pathlib import Path

from autosub.alignment_lab import file_hash, normalized
from autosub.domain import Project, Word
from autosub.export import export_subtitles
from autosub.native_segmentation import segment_native
from autosub.timing import apply_timing

parser=argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("manifest", type=Path)
parser.add_argument("result", type=Path)
parser.add_argument("--output", type=Path, required=True)
args=parser.parse_args()
p=Project.load(args.project)
m=json.loads(args.manifest.read_text(encoding="utf-8"))
r=json.loads(args.result.read_text(encoding="utf-8"))
if r["manifest_sha256"] != file_hash(args.manifest) or len(r["results"]) != 1 or r["status"] != "complete":
    raise ValueError("Expected a complete whole-clip result")
if file_hash(p.audio_path) != m["audio_sha256"]:
    raise ValueError("Audio mismatch")
if args.output.exists() and any(args.output.iterdir()):
    raise ValueError("Select an empty output")
args.output.mkdir(parents=True,exist_ok=True)
sources={path:file_hash(path) for path in [args.project,Path(p.media_path),Path(p.audio_path)]}
row=r["results"][0]
offset=row["window_ms"][0]
groups=[[Word(offset+round(w["start"]*1000), offset+round(w["end"]*1000), w["word"], w["probability"])
         for w in (segment["words"] or [])] for segment in row["segments"]]
candidate=copy.deepcopy(p)
candidate.words=[w for group in groups for w in group]
candidate.cues=[]
candidate.cues=apply_timing(candidate,segmenter=lambda remaining:segment_native(remaining,groups))
candidate.validate()
source_text="".join(w.text for w in candidate.words)
assert normalized(source_text)==normalized("".join(c.text for c in candidate.cues))
for suffix in (".ass",".srt"):
    export_subtitles(candidate,args.output/("whisper-native-sentences"+suffix))
report={"schema":"native-sentence-candidate-v1","human_verified":False,"default_changed":False,
    "not_a_resumable_app_project":True,"source_project_sha256":sources[args.project],
    "sources_unchanged":all(file_hash(path)==digest for path,digest in sources.items()),
    "words":[asdict(w) for w in candidate.words],"native_groups":[[asdict(w) for w in g] for g in groups],
    "cues":[asdict(c) for c in candidate.cues],
    "summary":{"native_segments":len(groups),"cues":len(candidate.cues),"words":len(candidate.words),
               "characters":len(normalized(source_text)),"normalized_text_preserved":True,
               "overlap_pairs":sum(a.end_ms>b.start_ms for a,b in zip(candidate.cues,candidate.cues[1:]))}}
if not report["sources_unchanged"]: raise RuntimeError("Source changed")
(args.output/"candidate.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
body=['<!doctype html><meta charset="utf-8"><title>保留原生句界</title><style>body{font:16px system-ui;margin:28px}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}</style>',
      '<h1>保留 Whisper 原生句界</h1><p>只在超過時長／字數時從詞語或短語邊界拆分。辨識文字未補寫，仍可能有漏字與誤辨；時間與語意未經人工驗收。正式程式預設未更換。</p>',
      '<p>'+html.escape(str(report['summary']))+'</p><table><tr><th>時間</th><th>字幕</th><th>複核項目</th></tr>']
for c in candidate.cues:
    body.append(f'<tr><td>{c.start_ms/1000:.2f}–{c.end_ms/1000:.2f}</td><td>{html.escape(c.text)}</td><td>{html.escape(str(c.flags))}</td></tr>')
body.append('</table>')
(args.output/'index.html').write_text(''.join(body),encoding='utf-8')
print(json.dumps(report['summary'],indent=2))
