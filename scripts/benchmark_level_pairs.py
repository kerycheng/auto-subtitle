"""Reference-free production versus full-window decoding on prepared level pairs."""
import argparse
import json
from pathlib import Path
from threading import Event

import numpy as np

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.media import load_audio
from autosub.pipeline import run_pipeline


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('cases',type=Path)
    parser.add_argument('--mode',choices=['production','full-window'],required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists')
    data=json.loads(args.cases.read_text(encoding='utf-8'))
    if data['text_provided'] is not False or not data['cases']:
        raise ValueError('Expected reference-free cases')
    allowed={'id','language','audio_path','audio_sha256','duration_ms'}
    if len({c['id'] for c in data['cases']})!=len(data['cases']) or any(set(c)!=allowed for c in data['cases']):
        raise ValueError('Invalid/text-bearing cases')
    import faster_whisper
    original=faster_whisper.WhisperModel
    calls=[]
    class FullWindowModel:
        def __init__(self,*a,**kw):
            self.model=original(*a,**kw);self.frames_per_second=self.model.frames_per_second
        def transcribe(self,audio,**kw):
            kw.pop('vad_parameters',None)
            kw.update(vad_filter=False,clip_timestamps=[0,len(audio)/16000])
            calls.append({'samples':len(audio),'options':kw})
            return self.model.transcribe(audio,**kw)
    report={'status':'running','cases_sha256':file_hash(args.cases),'model_identity':model_identity(args.model),
        'mode':args.mode,'cases':[],'decoding_calls':calls,'reference_text_provided':False,
        'scope':'one trial per mode; sub-30s complete-source amplitude pairs, not natural dialogue accuracy'}
    args.output.mkdir(parents=True)
    if args.mode=='full-window': faster_whisper.WhisperModel=FullWindowModel
    try:
        for case in data['cases']:
            if file_hash(case['audio_path'])!=case['audio_sha256']: raise ValueError('Source changed')
            folder=args.output/case['id'];folder.mkdir()
            p=Project(media_path=case['audio_path'],language=case['language'],device='cuda',model=str(args.model.resolve()))
            path=folder/'generated.autosub.json'
            run_pipeline(p,path,Event(),lambda *a:None)
            if not np.array_equal(load_audio(Path(p.audio_path)),load_audio(Path(case['audio_path']))):
                raise ValueError('ASR waveform changed')
            for suffix in (('.ass','.srt') if p.cues else ()):
                export_subtitles(p,folder/('generated'+suffix))
            report['cases'].append({'id':case['id'],'project_sha256':file_hash(path),'words':len(p.words),'cues':len(p.cues)})
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(args.mode,len(report['cases']),len(data['cases']),case['id'],len(p.words),'words',flush=True)
        report['status']='complete'
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        faster_whisper.WhisperModel=original


if __name__=='__main__':
    main()
